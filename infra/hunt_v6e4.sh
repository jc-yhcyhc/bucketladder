#!/usr/bin/env bash
# =============================================================================
# hunt_v6e4.sh — probe for v6e-4 spot capacity via the SAME tpu-api surface
#                 already used for v5litepod-4 (not the GCE/ct6e-1chip path
#                 hunt_v6e.sh covers). TP=4, matching the paper's existing
#                 setup exactly -- the only difference from the v5e hunt is
#                 the chip generation itself.
# =============================================================================
# On-demand v6e quota is not provisioned in this project; preemptible/spot
# shows a real 16-chip limit. So this only ever asks for spot, same
# reasoning as hunt_v6e.sh.
#
# Because this creates a `gcloud compute tpus tpu-vm` resource -- the exact
# same resource TYPE as v5litepod-4, just a different accelerator string --
# the EXISTING teardown_tpu.sh already covers it correctly (same command
# family, same 13-zone sweep). No new teardown script needed here, unlike
# the GCE/ct6e path. A distinct TPU_NAME keeps it from colliding with the
# v5e hunt's own bucketladder-tpu.
#
#   CYCLES=12 bash infra/hunt_v6e4.sh
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Every one of config.env's own `: "${VAR:=default}"` lines pre-empts any
# fallback this script tries to apply AFTER sourcing it, because by then
# "unset" and "set by config.env's own default" are indistinguishable. Bit
# twice live already (MODEL, then TPU_NAME -- the second one wrongly named
# a real, billing v6e-4 resource "bucketladder-tpu", colliding with the v5e
# hunt's own name, until caught by re-reading the log's own "name=" line
# and torn down by hand). Every caller-overridable var this script cares
# about is now captured BEFORE sourcing config.env, not after.
_bl_caller_model="${MODEL:-}"
_bl_caller_name="${TPU_NAME:-}"
# shellcheck source=./config.env
source "$HERE/config.env"
if [[ -z "$_bl_caller_model" ]]; then
  export MODEL=Qwen/Qwen3-4B   # a capacity probe has no business needing gated-repo credentials
fi
NAME="${_bl_caller_name:-bucketladder-tpu-v6e4}"
unset _bl_caller_model _bl_caller_name

CYCLES="${CYCLES:-12}"
PERIOD_MIN="${PERIOD_MIN:-15}"
LOG="${LOG:-/tmp/hunt_v6e4.log}"
ZONES="${ZONES:-europe-west4-a europe-west4-b europe-west4-c asia-northeast1-b asia-northeast1-c us-west4-a us-central1-a us-central1-b us-east5-a us-east5-b us-south1-a us-west1-c}"

log() { echo "[$(date -u '+%m-%d %H:%M:%S')] [hunt_v6e4] $*" | tee -a "$LOG"; }

teardown_everything() {
  local z; z="$(cat /tmp/bucketladder_v6e4_zone 2>/dev/null)"
  if [[ -n "$z" ]]; then
    log "tearing down in $z"
    ZONE="$z" TPU_NAME="$NAME" bash "$HERE/teardown_tpu.sh" --yes >>"$LOG" 2>&1
  fi
  TPU_NAME="$NAME" bash "$HERE/teardown_tpu.sh" --status >>"$LOG" 2>&1
  tail -2 "$LOG"
  rm -f /tmp/bucketladder_v6e4_zone
}
trap teardown_everything EXIT

exists_in() {
  gcloud compute tpus tpu-vm describe "$NAME" --zone="$1" --project="$PROJECT" \
    --format='value(state)' 2>/dev/null | grep -q .
}

for cycle in $(seq 1 "$CYCLES"); do
  log "cycle $cycle/$CYCLES"
  landed=""
  for z in $ZONES; do
    log "  trying v6e-4 (spot) in $z"
    PROVISION_PATH=tpu-api ACCELERATOR_TYPE=v6e-4 RUNTIME_VERSION=v2-alpha-tpuv6e \
      SPOT=true ZONE="$z" TPU_NAME="$NAME" \
      bash "$HERE/create_tpu.sh" >>"$LOG" 2>&1
    if exists_in "$z"; then
      landed="$z"
      break
    fi
  done

  if [[ -z "$landed" ]]; then
    log "cycle $cycle: no v6e-4 spot capacity in any zone; sleeping to the next slot"
    sleep $(( PERIOD_MIN * 60 ))
    continue
  fi

  log "GOT ONE: v6e-4 spot landed in $landed"
  echo "$landed" > /tmp/bucketladder_v6e4_zone
  ZONE="$landed" TPU_NAME="$NAME" bash "$HERE/deadman.sh" 1200 >>"$LOG" 2>&1
  log "deadman armed 1200s in $landed -- TP=4 v6e slice up, report and wait for a deliberate next decision"
  exit 0
done

log "no v6e-4 spot capacity across $CYCLES cycles; nothing created"
