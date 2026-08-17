#!/usr/bin/env bash
# Retry provisioning until a zone has capacity, then arm the deadman immediately.
#
# Capacity is transient and unannounced: a v5litepod-4 that is refused in ten
# zones can appear minutes later. This retries on a fixed period rather than
# holding a session open, and the deadman is armed in the same breath as the
# create so a slice that appears while nobody is watching still dies on schedule.
#
#   DEADLINE_MIN=45 SPOT=true bash infra/provision_retry.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEADLINE_MIN="${DEADLINE_MIN:-45}"
PERIOD_SEC="${PERIOD_SEC:-180}"
DEADMAN_SEC="${DEADMAN_SEC:-5400}"
ZONES="${ZONES:-us-west4-a us-west1-c us-east1-d us-central1-a us-east5-a us-south1-a}"
# v5litepod-8 is single-host like the -4, so it is a drop-in. -16 is multi-host
# and needs every worker driven, so it is deliberately not in the default list.
ACCEL_LIST="${ACCEL_LIST:-v5litepod-4 v5litepod-8}"

deadline=$(( $(date +%s) + DEADLINE_MIN * 60 ))
attempt=0

while [[ $(date +%s) -lt $deadline ]]; do
  attempt=$(( attempt + 1 ))
  # Alternate pools: spot and on-demand are scheduled from different capacity.
  if (( attempt % 2 == 1 )); then pool=spot; else pool=ondemand; fi

  # Rotate topology too. Three hunts asked only for v5litepod-4 on the
  # assumption that a bigger slice is harder to schedule -- an assumption
  # never tested, while the project holds quota for 16 chips. A larger slice
  # is a different scheduling unit, not merely a scarcer one, so it is worth
  # asking for when the small one is refused everywhere.
  for accel in $ACCEL_LIST; do
    echo "[retry] attempt $attempt ($pool, $accel) $(date -u +%H:%M:%S)"
    SPOT=$([[ $pool == spot ]] && echo true || echo false) ZONES="$ZONES" \
      ACCELERATOR_TYPE="$accel" bash "$HERE/provision_first_available.sh" 2>&1 | tail -1
  done

  # Ask the API, never the log prose -- parsing output for success is what
  # billed two slices at once earlier in this project.
  for z in $ZONES; do
    state=$(timeout 45 gcloud compute tpus tpu-vm describe bucketladder-tpu \
              --zone="$z" --format='value(state)' 2>/dev/null)
    if [[ -n "$state" ]]; then
      echo "[retry] GOT ONE: $z state=$state after $attempt attempt(s)"
      ZONE="$z" nohup bash "$HERE/deadman.sh" "$DEADMAN_SEC" >/tmp/dm_retry.log 2>&1 &
      sleep 2
      echo "[retry] deadman armed ${DEADMAN_SEC}s in $z"
      echo "$z" > /tmp/bucketladder_zone
      exit 0
    fi
  done
  sleep "$PERIOD_SEC"
done

echo "[retry] no capacity in any zone after ${DEADLINE_MIN} min; nothing created, nothing billing"
exit 1
