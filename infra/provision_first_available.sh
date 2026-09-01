#!/usr/bin/env bash
# Provision in the first zone that has capacity, and STOP.
#
# Written after a loop that detected success by grepping the creation log's prose
# missed a successful create twice, kept iterating, and left two billing slices in
# two zones. Success is now determined by asking the API whether the TPU exists,
# which cannot be defeated by a wording change, and the loop exits on the first
# slice that is READY.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${TPU_NAME:-bucketladder-tpu}"
ACCEL="${ACCELERATOR_TYPE:-v5litepod-4}"
ZONES="${ZONES:-us-west4-a us-central1-a us-east5-a us-south1-a}"

# Found live, 2026-09-01: a second, concurrent hunt for a different
# accelerator (v6e-4) reused the default name "bucketladder-tpu" by mistake
# (a bug in that script, since fixed), and THIS check -- which only asked
# "does something named $NAME exist", never "is it the $ACCEL we're
# actually trying to create" -- read that unrelated v6e-4 resource as its
# own successful create. The hunt then ran deploy.sh and an experiment
# script against the wrong hardware generation (it failed at the SSH/patch
# stage before anything booted, so nothing was corrupted, but that was luck,
# not this check working). exists() now verifies acceleratorType too, so a
# same-named resource of the WRONG type is treated as a collision to warn
# about, not a success to act on.
exists() {
  local state accel
  state=$(gcloud compute tpus tpu-vm describe "$NAME" --zone="$1" \
            --format='value(state)' 2>/dev/null)
  [[ -z "$state" ]] && return 1
  accel=$(gcloud compute tpus tpu-vm describe "$NAME" --zone="$1" \
            --format='value(acceleratorType)' 2>/dev/null)
  if [[ "$accel" != "$ACCEL" ]]; then
    echo "[provision] WARNING: '$NAME' exists in $1 but as '$accel', not the" \
         "'$ACCEL' this run wants -- not treating it as this run's success," \
         "and NOT deleting it either. Resolve the name collision by hand." >&2
    return 1
  fi
  return 0
}

for z in $ZONES; do
  # Never start a create while one already exists anywhere in the list.
  for c in $ZONES; do
    if exists "$c"; then
      echo "[provision] '$NAME' already exists in $c — stopping."
      echo "$c"; exit 0
    fi
  done
  echo "[provision] trying $ACCEL in $z"
  ZONE="$z" ACCELERATOR_TYPE="$ACCEL" bash "$HERE/create_tpu.sh" >/dev/null 2>&1
  if exists "$z"; then
    echo "[provision] created in $z"
    echo "$z"; exit 0
  fi
  echo "[provision]   no capacity in $z"
done
echo "[provision] no zone had capacity" >&2
exit 1
