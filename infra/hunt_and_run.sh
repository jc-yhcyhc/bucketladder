#!/usr/bin/env bash
# =============================================================================
# hunt_and_run.sh — retry for capacity every 30 min; on success run the M2 arms
#                   and tear down, without waiting for anyone to notice.
# =============================================================================
# A slice that appears at 03:00 and idles until someone looks at it costs the
# same as one that is used. So the hunt does not merely provision: it runs the
# experiment and deletes the slice in the same unattended sequence.
#
# Cost is bounded three ways, deliberately overlapping, because each one has
# failed at least once in this project:
#   1. provision_retry.sh arms a deadman at create time.
#   2. This script tears down in an EXIT trap, so a crash still deletes.
#   3. The teardown sweep runs afterwards and its output is kept, so the next
#      session reads a verified all-clear rather than an assumption.
#
#   CYCLES=12 bash infra/hunt_and_run.sh    # ~6 hours of half-hourly attempts
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

CYCLES="${CYCLES:-12}"
PERIOD_MIN="${PERIOD_MIN:-30}"
HUNT_MIN="${HUNT_MIN:-20}"        # hunt for 20 of every 30 minutes
DEADMAN_SEC="${DEADMAN_SEC:-10800}"
LOG="${LOG:-/tmp/hunt_and_run.log}"
ZONES="${ZONES:-europe-west4-a europe-west4-b europe-west4-c asia-northeast1-a asia-northeast1-b asia-northeast1-c us-west4-a us-central1-a us-central1-b us-east5-a us-east5-b us-south1-a us-west1-c}"

log() { echo "[$(date -u '+%m-%d %H:%M:%S')] [hunt] $*" | tee -a "$LOG"; }

teardown_everything() {
  local z; z="$(cat /tmp/bucketladder_zone 2>/dev/null)"
  if [[ -n "$z" ]]; then
    log "tearing down in $z"
    ZONE="$z" bash "$HERE/teardown_tpu.sh" --yes >>"$LOG" 2>&1
  fi
  # Sweep regardless: the zone file is a claim, the sweep is the check.
  bash "$HERE/teardown_tpu.sh" --status >>"$LOG" 2>&1
  tail -2 "$LOG"
  rm -f /tmp/bucketladder_zone
}
trap teardown_everything EXIT

for cycle in $(seq 1 "$CYCLES"); do
  log "cycle $cycle/$CYCLES"
  rm -f /tmp/bucketladder_zone

  ACCEL_LIST="v5litepod-4" ZONES="$ZONES" DEADLINE_MIN="$HUNT_MIN" \
    PERIOD_SEC=120 DEADMAN_SEC="$DEADMAN_SEC" \
    bash "$HERE/provision_retry.sh" >>"$LOG" 2>&1

  zone="$(cat /tmp/bucketladder_zone 2>/dev/null)"
  if [[ -z "$zone" ]]; then
    log "cycle $cycle: no capacity; sleeping to the next slot"
    sleep $(( (PERIOD_MIN - HUNT_MIN) * 60 ))
    continue
  fi

  log "SLICE in $zone on cycle $cycle -- running the M2 arms"
  ZONE="$zone" bash "$HERE/deploy.sh" >>"$LOG" 2>&1 \
    || log "deploy reported a problem; continuing, run_m2_arms will surface it"
  ZONE="$zone" bash "$HERE/run_m2_arms.sh" 2>&1 | tee -a "$LOG"
  log "arms finished; tearing down via trap"
  exit 0   # EXIT trap performs teardown + sweep
done

log "no capacity across $CYCLES cycles (~$(( CYCLES * PERIOD_MIN / 60 ))h); nothing created"
