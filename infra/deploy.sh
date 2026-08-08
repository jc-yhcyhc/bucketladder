#!/usr/bin/env bash
# =============================================================================
# deploy.sh — copy this repo to the TPU VM.
# =============================================================================
# create_tpu.sh's old instructions said to scp vm_setup.sh alone, and then
# vm_setup.sh's closing message told you to run scripts/e00_smoke_test.py —
# which was never copied. This script closes that gap: it puts everything the
# VM needs there in one step.
#
# Excludes .venv, results, and git objects: the VM builds its own environment
# and results are written there, not shipped there.
#
# Usage:
#   ./infra/deploy.sh --dry-run
#   ./infra/deploy.sh
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
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

log() { echo "[$(date '+%H:%M:%S')] [deploy] $*"; }
die() { echo "[$(date '+%H:%M:%S')] [deploy] ERROR: $*" >&2; exit 1; }

# What the VM actually needs: the harness, the configs, the infra scripts.
PAYLOAD=(scripts configs infra requirements.txt)

TARBALL="/tmp/bucketladder-deploy.tar.gz"

if [[ "$DRY_RUN" == "true" ]]; then
  log "DRY RUN — would package: ${PAYLOAD[*]}"
  log "  tar -czf $TARBALL -C $REPO --exclude=__pycache__ ${PAYLOAD[*]}"
  log "  gcloud compute scp $TARBALL ${TPU_NAME}:~/ --zone=$ZONE --project=$PROJECT"
  log "  gcloud compute ssh $TPU_NAME --zone=$ZONE --command='mkdir -p ~/bucketladder && tar -xzf ~/$(basename $TARBALL) -C ~/bucketladder'"
  log "DRY RUN — nothing copied."
  exit 0
fi

command -v gcloud >/dev/null 2>&1 || die "gcloud not on PATH"

log "packaging ${PAYLOAD[*]}…"
tar -czf "$TARBALL" -C "$REPO" --exclude=__pycache__ --exclude='*.pyc' "${PAYLOAD[@]}"
log "  $(du -h "$TARBALL" | cut -f1)"

log "copying to $TPU_NAME…"
gcloud compute scp "$TARBALL" "${TPU_NAME}:~/" --zone="$ZONE" --project="$PROJECT"

log "unpacking on the VM…"
gcloud compute ssh "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" \
  --command="mkdir -p ~/bucketladder && tar -xzf ~/$(basename "$TARBALL") -C ~/bucketladder && ls ~/bucketladder"

log "deployed to ~/bucketladder on $TPU_NAME"
