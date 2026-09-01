#!/usr/bin/env bash
# =============================================================================
# teardown_tpu_gce.sh — delete a TPU-shaped GCE instance (the v6e/GCE
#                         provisioning path). Run at the end of every v6e probe.
# =============================================================================
# teardown_tpu.sh only sweeps `gcloud compute tpus tpu-vm list` -- the
# tpu-api surface. A v6e slice created via PROVISION_PATH=gce is a
# `gcloud compute instances` resource instead (see infra/_paths.sh), which
# that sweep cannot see at all. That is a real blind spot: "nothing billing"
# from teardown_tpu.sh --status would not catch a stray v6e instance.
# This mirrors teardown_gpu.sh's pattern exactly -- same command family,
# same by-name sweep -- so the v6e path has the same safety net the L4 GPU
# path already has.
#
# Usage:
#   ./infra/teardown_tpu_gce.sh --dry-run
#   ./infra/teardown_tpu_gce.sh            # asks for confirmation
#   ./infra/teardown_tpu_gce.sh --yes      # no prompt, for scripted session ends
#   ./infra/teardown_tpu_gce.sh --status   # is anything billing right now?
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.env
source "$HERE/config.env"

: "${TPU_GCE_NAME:=bucketladder-tpu-v6e}"
: "${TPU_GCE_ZONE:=us-central1-a}"
: "${TPU_GCE_SWEEP_ZONES:=europe-west4-a europe-west4-b europe-west4-c asia-northeast1-a asia-northeast1-b asia-northeast1-c us-west4-a us-central1-a us-central1-b us-east5-a us-east5-b us-south1-a us-west1-c}"

DRY_RUN=false
ASSUME_YES=false
STATUS_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --yes|-y)  ASSUME_YES=true ;;
    --status)  STATUS_ONLY=true ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] [teardown_tpu_gce] $*"; }
die() { echo "[$(date '+%H:%M:%S')] [teardown_tpu_gce] ERROR: $*" >&2; exit 1; }

command -v gcloud >/dev/null 2>&1 || {
  if [[ "$DRY_RUN" == "true" ]]; then
    log "gcloud not on PATH (fine for --dry-run)"
  else
    die "gcloud not on PATH"
  fi
}

sweep_other_zones() {
  local z found stray=0
  for z in ${TPU_GCE_SWEEP_ZONES:-}; do
    [[ "$z" == "$TPU_GCE_ZONE" ]] && continue
    found=$(gcloud compute instances list --zone="$z" --project="$PROJECT" \
              --filter="name=$TPU_GCE_NAME" --format='value(name,status)' 2>/dev/null || true)
    if [[ -n "$found" ]]; then
      echo "STRAY: '$TPU_GCE_NAME' EXISTS in $z: $found — NOT all-clear." >&2
      stray=1
    fi
  done
  return $stray
}

status() {
  local here found=0
  here=$(gcloud compute instances describe "$TPU_GCE_NAME" --zone="$TPU_GCE_ZONE" --project="$PROJECT" \
           --format='value(status)' 2>/dev/null || true)
  if [[ -n "$here" ]]; then
    log "'$TPU_GCE_NAME' exists in $TPU_GCE_ZONE, status=$here — BILLING."
    found=1
  else
    log "'$TPU_GCE_NAME' does not exist in $TPU_GCE_ZONE."
  fi
  if sweep_other_zones; then
    log "swept zones [${TPU_GCE_SWEEP_ZONES}] — nothing billing in any of them."
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

exists=$(gcloud compute instances describe "$TPU_GCE_NAME" --zone="$TPU_GCE_ZONE" --project="$PROJECT" \
           --format='value(name)' 2>/dev/null || true)
if [[ -z "$exists" ]]; then
  log "'$TPU_GCE_NAME' does not exist in $TPU_GCE_ZONE — nothing to delete here."
  sweep_other_zones || log "but a stray instance was found elsewhere (above) — delete it too."
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  log "DRY RUN — would run: gcloud compute instances delete $TPU_GCE_NAME --zone=$TPU_GCE_ZONE --project=$PROJECT --quiet"
  exit 0
fi

if [[ "$ASSUME_YES" != "true" ]]; then
  read -r -p "Delete v6e instance '$TPU_GCE_NAME' in $TPU_GCE_ZONE? Results must already be off it. [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { log "aborted, nothing deleted."; exit 1; }
fi

log "deleting '$TPU_GCE_NAME' in $TPU_GCE_ZONE…"
gcloud compute instances delete "$TPU_GCE_NAME" --zone="$TPU_GCE_ZONE" --project="$PROJECT" --quiet
log "deleted. Billing stopped."
sweep_other_zones && log "swept zones clean too." || log "WARNING: a stray instance remains elsewhere — see above."
