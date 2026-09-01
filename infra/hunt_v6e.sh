#!/usr/bin/env bash
# =============================================================================
# hunt_v6e.sh — probe for v6e spot capacity via the GCE path, separately from
#                the v5e hunt (hunt_and_run.sh). Diagnostic, not an experiment
#                runner: no deploy/arms script exists yet for TP=1/v6e, so a
#                landed slice is held just long enough to arm a deadman and
#                confirm capacity, then reported and left for a deliberate
#                next decision rather than auto-torn-down or auto-used.
# =============================================================================
# On-demand v6e quota is not provisioned in this project (checked via
# `gcloud alpha services quota list`); preemptible/spot v6e quota shows
# unlimited. So this only ever asks for spot -- an on-demand attempt would
# fail on quota, not capacity, and that is a different question than the one
# this script exists to answer.
#
# Uses a name and teardown path distinct from the v5e hunt (bucketladder-tpu)
# so the two can run concurrently without colliding, and so
# teardown_tpu_gce.sh's sweep -- not teardown_tpu.sh's, which cannot see a
# `compute instances` resource at all -- is the one that must report clean.
#
#   CYCLES=12 bash infra/hunt_v6e.sh
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.env
source "$HERE/config.env"
# BUG FOUND LIVE, 2026-09-01: this line was missing on first run. Without it
# $PROJECT is empty in THIS script's own scope (create_tpu.sh sources
# config.env in its own subshell, which does not propagate back), so
# exists_in()'s `--project="$PROJECT"` expanded to `--project=` -- an
# invalid empty value gcloud rejects, silently swallowed by `2>/dev/null`.
# The result: exists_in() returned false even after a real, billing create
# succeeded (confirmed via BILLING HAS STARTED in the log and a direct
# out-of-band `gcloud ... describe` a moment later), and the loop moved on
# to try more zones instead of stopping -- exactly the failure mode
# provision_first_available.sh's own comments warn about. Caught by manually
# cross-checking the log against a direct API query, not by this script.

CYCLES="${CYCLES:-12}"
PERIOD_MIN="${PERIOD_MIN:-15}"
DEADMAN_SEC="${DEADMAN_SEC:-1200}"     # 20 min: long enough to notice, short by design
LOG="${LOG:-/tmp/hunt_v6e.log}"
NAME="${TPU_GCE_NAME:-bucketladder-tpu-v6e}"
ZONES="${ZONES:-europe-west4-a europe-west4-b europe-west4-c asia-northeast1-a asia-northeast1-b asia-northeast1-c us-west4-a us-central1-a us-central1-b us-east5-a us-east5-b us-south1-a us-west1-c}"

log() { echo "[$(date -u '+%m-%d %H:%M:%S')] [hunt_v6e] $*" | tee -a "$LOG"; }

teardown_everything() {
  local z; z="$(cat /tmp/bucketladder_v6e_zone 2>/dev/null)"
  if [[ -n "$z" ]]; then
    log "tearing down in $z"
    TPU_GCE_ZONE="$z" TPU_GCE_NAME="$NAME" bash "$HERE/teardown_tpu_gce.sh" --yes >>"$LOG" 2>&1
  fi
  TPU_GCE_NAME="$NAME" bash "$HERE/teardown_tpu_gce.sh" --status >>"$LOG" 2>&1
  tail -2 "$LOG"
  rm -f /tmp/bucketladder_v6e_zone
}
trap teardown_everything EXIT

exists_in() {
  gcloud compute instances describe "$NAME" --zone="$1" --project="$PROJECT" \
    --format='value(name)' 2>/dev/null | grep -q .
}

for cycle in $(seq 1 "$CYCLES"); do
  log "cycle $cycle/$CYCLES"
  landed=""
  for z in $ZONES; do
    log "  trying v6e (spot) in $z"
    PROVISION_PATH=gce SPOT=true ZONE="$z" TPU_NAME="$NAME" \
      bash "$HERE/create_tpu.sh" >>"$LOG" 2>&1
    if exists_in "$z"; then
      landed="$z"
      break
    fi
  done

  if [[ -z "$landed" ]]; then
    log "cycle $cycle: no v6e spot capacity in any zone; sleeping to the next slot"
    sleep $(( PERIOD_MIN * 60 ))
    continue
  fi

  log "GOT ONE: v6e spot landed in $landed"
  echo "$landed" > /tmp/bucketladder_v6e_zone
  ZONE="$landed" GPU_NAME="$NAME" GPU_ZONE="$landed" \
    bash "$HERE/deadman_gpu.sh" "$DEADMAN_SEC" >>"$LOG" 2>&1
  log "deadman armed ${DEADMAN_SEC}s in $landed -- holding for a deliberate next decision, not auto-using it"
  log "no deploy/arms script exists yet for TP=1/v6e; report this and wait rather than run anything on it"
  exit 0
done

log "no v6e spot capacity across $CYCLES cycles; nothing created"
