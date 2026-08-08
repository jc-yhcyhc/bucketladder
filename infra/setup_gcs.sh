#!/usr/bin/env bash
# =============================================================================
# setup_gcs.sh — create the results bucket. Run ONCE, before the first session.
# =============================================================================
# Results must never live only on the TPU VM: the VM is deleted at the end of
# every session (that is the cost discipline), and its disk goes with it.
#
# The bucket costs approximately nothing — a few hundred MB of Parquet and logs
# — and is independent of the TPU lifecycle. Create it before session 1 or the
# session-end rsync has nowhere to write.
#
# Usage:
#   ./infra/setup_gcs.sh --dry-run
#   ./infra/setup_gcs.sh
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.env
source "$HERE/config.env"

DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] [setup_gcs] $*"; }
die() { echo "[$(date '+%H:%M:%S')] [setup_gcs] ERROR: $*" >&2; exit 1; }

[[ -n "${PROJECT:-}" ]] || die "PROJECT is empty; run: gcloud config set project <id>"

# Same region as the TPU: colocating avoids cross-region egress on every sync.
REGION="${ZONE%-*}"

if command -v gcloud >/dev/null 2>&1 && [[ "$DRY_RUN" != "true" ]]; then
  if gcloud storage ls "$GCS_BUCKET" >/dev/null 2>&1; then
    log "$GCS_BUCKET already exists — nothing to do."
    exit 0
  fi
fi

CMD=(gcloud storage buckets create "$GCS_BUCKET"
     --project="$PROJECT"
     --location="$REGION"
     --uniform-bucket-level-access)

if [[ "$DRY_RUN" == "true" ]]; then
  log "DRY RUN — the command that would run:"
  printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

log "creating $GCS_BUCKET in $REGION…"
"${CMD[@]}"
log "created. Results sync here at the end of every session."
