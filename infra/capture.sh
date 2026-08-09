#!/usr/bin/env bash
# =============================================================================
# capture.sh — pull artifacts OFF the VM. This is session 1's entire deliverable.
# =============================================================================
# Session 1 exists to get a real warmup log onto a laptop so the parser can be
# fixed offline at $0 rather than at $4.80/hr. Until this runs, nothing has been
# retrieved and the session achieved nothing.
#
# Also used at the end of every later session to retrieve results before
# teardown. Run this BEFORE teardown_tpu.sh, always — the VM disk is deleted
# with the VM.
#
# Usage:
#   ./infra/capture.sh --dry-run
#   ./infra/capture.sh                       # warmup log + results
#   ./infra/capture.sh --tag gap512          # label this capture
#   ./infra/capture.sh --push                # also sync results to GCS
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# shellcheck source=./config.env
source "$HERE/config.env"
# shellcheck source=./_paths.sh
source "$HERE/_paths.sh"

: "${WARMUP_LOG:=/tmp/vllm_warmup.log}"

DRY_RUN=false
PUSH=false
TAG="$(date -u '+%Y%m%dT%H%M%SZ')"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --push)    PUSH=true; shift ;;
    --tag)     TAG="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] [capture] $*"; }
die() { echo "[$(date '+%H:%M:%S')] [capture] ERROR: $*" >&2; exit 1; }

DEST="$REPO/captured/$TAG"

if [[ "$DRY_RUN" == "true" ]]; then
  log "DRY RUN — would create $DEST and run:"
  log "  gcloud compute scp ${TPU_NAME}:$WARMUP_LOG $DEST/vllm_warmup.log --zone=$ZONE"
  log "  gcloud compute scp --recurse ${TPU_NAME}:~/bucketladder/results $DEST/ --zone=$ZONE"
  [[ "$PUSH" == "true" ]] && log "  gcloud storage rsync -r $DEST $GCS_BUCKET/captured/$TAG"
  log "DRY RUN — nothing copied."
  exit 0
fi

command -v gcloud >/dev/null 2>&1 || die "gcloud not on PATH"
mkdir -p "$DEST"

log "pulling warmup log…"
if tpu_scp "${TPU_NAME}:$WARMUP_LOG" "$DEST/vllm_warmup.log" 2>/dev/null; then
  log "  got $(wc -l < "$DEST/vllm_warmup.log") lines -> $DEST/vllm_warmup.log"
else
  log "  WARNING: no warmup log at $WARMUP_LOG on the VM"
fi

log "pulling results…"
tpu_scp --recurse "${TPU_NAME}:~/bucketladder/results" "$DEST/" 2>/dev/null \
  || log "  (no results directory yet — expected in session 1)"

if [[ "$PUSH" == "true" ]]; then
  log "syncing to $GCS_BUCKET/captured/$TAG…"
  gcloud storage rsync -r "$DEST" "$GCS_BUCKET/captured/$TAG"
fi

log "captured to $DEST"
log ""
log "Now, with the VM torn down, iterate on the parser at \$0:"
log "  python scripts/e00_smoke_test.py --config configs/e00_default_ladder.json \\"
log "         --warmup-log $DEST/vllm_warmup.log"
