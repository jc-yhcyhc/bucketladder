#!/usr/bin/env bash
# =============================================================================
# deadman_gpu.sh — a backstop teardown for the GPU VM, same identity-scoped
#                   design as deadman.sh (a later VM with the same name is
#                   never touched -- creation timestamp must match).
# =============================================================================
# Usage:  infra/deadman_gpu.sh <seconds>          # arms in the background
# =============================================================================
set -uo pipefail
ZONE="${GPU_ZONE:-us-central1-a}"
NAME="${GPU_NAME:-bucketladder-gpu}"
DELAY="${1:?usage: deadman_gpu.sh <seconds>}"

STAMP=$(gcloud compute instances describe "$NAME" --zone="$ZONE" \
          --format='value(creationTimestamp)' 2>/dev/null)
[[ -z "$STAMP" ]] && { echo "[deadman_gpu] no VM '$NAME' in $ZONE; not arming"; exit 1; }

(
  sleep "$DELAY"
  now=$(gcloud compute instances describe "$NAME" --zone="$ZONE" \
          --format='value(creationTimestamp)' 2>/dev/null)
  if [[ -z "$now" ]]; then
    exit 0                                  # already gone; nothing to do
  fi
  if [[ "$now" != "$STAMP" ]]; then
    echo "[deadman_gpu] $(date -u) DECLINED: '$NAME' was created $now, not $STAMP" \
      >> "$(dirname "$0")/../deadman.log"
    exit 0                                  # a LATER VM — not ours to delete
  fi
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet \
    && echo "[deadman_gpu] $(date -u) tore down '$NAME' created $STAMP" \
       >> "$(dirname "$0")/../deadman.log"
) >/dev/null 2>&1 &

echo "[deadman_gpu] armed for ${DELAY}s, scoped to '$NAME' created $STAMP"
