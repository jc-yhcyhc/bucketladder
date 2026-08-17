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

deadline=$(( $(date +%s) + DEADLINE_MIN * 60 ))
attempt=0

while [[ $(date +%s) -lt $deadline ]]; do
  attempt=$(( attempt + 1 ))
  # Alternate pools: spot and on-demand are scheduled from different capacity.
  if (( attempt % 2 == 1 )); then pool=spot; else pool=ondemand; fi
  echo "[retry] attempt $attempt ($pool) $(date -u +%H:%M:%S)"

  if SPOT=$([[ $pool == spot ]] && echo true || echo false) ZONES="$ZONES" \
       bash "$HERE/provision_first_available.sh" 2>&1 | tail -2; then
    :
  fi

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
