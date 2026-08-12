#!/usr/bin/env bash
# =============================================================================
# deadman.sh — a backstop teardown that cannot kill someone else's VM.
# =============================================================================
# Session 19's TPU was deleted nine minutes into its run by a switch armed in
# session 17. The old switch was `sleep N; if <name exists>; then delete <name>`,
# which targets a NAME and has no notion of which VM it was armed for. Every
# stale switch is therefore a live grenade for every future session.
#
# This version captures the VM's creation timestamp at arm time and deletes only
# if the VM still carries it. A later VM with the same name has a different
# timestamp and is left alone.
#
# Usage:  infra/deadman.sh <seconds>          # arms in the background
# =============================================================================
set -uo pipefail
ZONE="${ZONE:-us-west4-a}"
NAME="${TPU_NAME:-bucketladder-tpu}"
DELAY="${1:?usage: deadman.sh <seconds>}"

STAMP=$(gcloud compute tpus tpu-vm describe "$NAME" --zone="$ZONE" \
          --format='value(createTime)' 2>/dev/null)
[[ -z "$STAMP" ]] && { echo "[deadman] no VM '$NAME' in $ZONE; not arming"; exit 1; }

(
  sleep "$DELAY"
  now=$(gcloud compute tpus tpu-vm describe "$NAME" --zone="$ZONE" \
          --format='value(createTime)' 2>/dev/null)
  if [[ -z "$now" ]]; then
    exit 0                                  # already gone; nothing to do
  fi
  if [[ "$now" != "$STAMP" ]]; then
    echo "[deadman] $(date -u) DECLINED: '$NAME' was created $now, not $STAMP" \
      >> "$(dirname "$0")/../deadman.log"
    exit 0                                  # a LATER VM — not ours to delete
  fi
  gcloud compute tpus tpu-vm delete "$NAME" --zone="$ZONE" --quiet \
    && echo "[deadman] $(date -u) tore down '$NAME' created $STAMP" \
       >> "$(dirname "$0")/../deadman.log"
) >/dev/null 2>&1 &

echo "[deadman] armed for ${DELAY}s, scoped to '$NAME' created $STAMP"
