#!/usr/bin/env bash
# =============================================================================
# teardown_gpu.sh — delete the GPU VM. Run at the END OF EVERY GPU SESSION.
# =============================================================================
# Mirrors teardown_tpu.sh: a GCE instance with an attached GPU bills while it
# EXISTS, not while it computes, exactly like a TPU VM. Same cost-control
# argument, same discipline, different resource type.
#
# Usage:
#   ./infra/teardown_gpu.sh --dry-run
#   ./infra/teardown_gpu.sh            # asks for confirmation
#   ./infra/teardown_gpu.sh --yes      # no prompt, for scripted session ends
#   ./infra/teardown_gpu.sh --status   # is anything billing right now?
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.env
source "$HERE/config.env"

: "${GPU_NAME:=bucketladder-gpu}"
: "${GPU_ZONE:=us-central1-a}"
# Every zone this project has ever checked for L4 availability, not just the
# one it happened to provision in -- same reasoning as TEARDOWN_SWEEP_ZONES.
: "${GPU_TEARDOWN_SWEEP_ZONES:=us-central1-a us-central1-b us-central1-c}"

DRY_RUN=false
ASSUME_YES=false
STATUS_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --yes|-y)  ASSUME_YES=true ;;
    --status)  STATUS_ONLY=true ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] [teardown_gpu] $*"; }
die() { echo "[$(date '+%H:%M:%S')] [teardown_gpu] ERROR: $*" >&2; exit 1; }

command -v gcloud >/dev/null 2>&1 || {
  if [[ "$DRY_RUN" == "true" ]]; then
    log "gcloud not on PATH (fine for --dry-run)"
  else
    die "gcloud not on PATH"
  fi
}

sweep_other_zones() {
  local z found stray=0
  for z in ${GPU_TEARDOWN_SWEEP_ZONES:-}; do
    [[ "$z" == "$GPU_ZONE" ]] && continue
    found=$(gcloud compute instances list --zone="$z" --project="$PROJECT" \
              --filter="name=$GPU_NAME" --format='value(name,status)' 2>/dev/null || true)
    if [[ -n "$found" ]]; then
      echo "STRAY: '$GPU_NAME' EXISTS in $z: $found — NOT all-clear." >&2
      stray=1
    fi
  done
  return $stray
}

status() {
  local here found=0
  here=$(gcloud compute instances describe "$GPU_NAME" --zone="$GPU_ZONE" --project="$PROJECT" \
           --format='value(status)' 2>/dev/null || true)
  if [[ -n "$here" ]]; then
    log "'$GPU_NAME' exists in $GPU_ZONE, status=$here — BILLING."
    found=1
  else
    log "'$GPU_NAME' does not exist in $GPU_ZONE."
  fi
  if sweep_other_zones; then
    log "swept zones [${GPU_TEARDOWN_SWEEP_ZONES}] — nothing billing in any of them."
  else
    found=1
  fi
  if [[ $found -eq 0 ]]; then
    log "Nothing billing."
    return 0
  fi
  return 1
}

if [[ "$STATUS_ONLY" == "true" ]]; then
  status; exit $?
fi

exists=$(gcloud compute instances describe "$GPU_NAME" --zone="$GPU_ZONE" --project="$PROJECT" \
           --format='value(name)' 2>/dev/null || true)
if [[ -z "$exists" ]]; then
  log "'$GPU_NAME' does not exist in $GPU_ZONE — nothing to delete here."
  sweep_other_zones || log "but a stray instance was found elsewhere (above) — delete it too."
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  log "DRY RUN — would run: gcloud compute instances delete $GPU_NAME --zone=$GPU_ZONE --project=$PROJECT --quiet"
  exit 0
fi

if [[ "$ASSUME_YES" != "true" ]]; then
  read -r -p "Delete GPU VM '$GPU_NAME' in $GPU_ZONE? Results must already be off it. [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { log "aborted, nothing deleted."; exit 1; }
fi

log "deleting '$GPU_NAME' in $GPU_ZONE…"
gcloud compute instances delete "$GPU_NAME" --zone="$GPU_ZONE" --project="$PROJECT" --quiet
log "deleted. Billing stopped."
sweep_other_zones && log "swept zones clean too." || log "WARNING: a stray instance remains elsewhere — see above."
