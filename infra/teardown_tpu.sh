#!/usr/bin/env bash
# =============================================================================
# teardown_tpu.sh — delete the TPU VM. Run at the END OF EVERY SESSION.
# =============================================================================
#
# This is the single most important cost control in the project. A TPU VM bills
# while it EXISTS, not while it computes. plan_v4.md budgets 135 billed VM-hours
# (~$450) against a $1000 ceiling; the ~$550 of headroom is almost exactly one
# forgotten VM-week at $4.80/hr. Forgetting this script once can cost more than
# every measurement in the paper.
#
# Results must already be in GCS. This deletes the VM disk with no warning
# beyond the reminder below.
#
# Usage:
#   ./infra/teardown_tpu.sh --dry-run
#   ./infra/teardown_tpu.sh            # asks for confirmation
#   ./infra/teardown_tpu.sh --yes      # no prompt, for scripted session ends
#   ./infra/teardown_tpu.sh --status   # is anything billing right now?
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.env
source "$HERE/config.env"
# shellcheck source=./_paths.sh
source "$HERE/_paths.sh"

DRY_RUN=false
ASSUME_YES=false
STATUS_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --yes|-y)  ASSUME_YES=true ;;
    --status)  STATUS_ONLY=true ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log() { echo "[$(date '+%H:%M:%S')] [teardown] $*"; }
die() { echo "[$(date '+%H:%M:%S')] [teardown] ERROR: $*" >&2; exit 1; }

command -v gcloud >/dev/null 2>&1 || {
  if [[ "$DRY_RUN" == "true" ]]; then
    log "gcloud not on PATH (fine for --dry-run)"
  else
    die "gcloud not on PATH"
  fi
}

# ── Status ──────────────────────────────────────────────────────────────────
# Returns 0 if no stray TPU is billing in any swept zone, 1 if one is found.
sweep_other_zones() {
  local z found stray=0
  for z in ${TEARDOWN_SWEEP_ZONES:-}; do
    [[ "$z" == "$ZONE" ]] && continue
    found=$(gcloud compute tpus tpu-vm list --zone="$z" --project="$PROJECT" \
              --format='value(name)' 2>/dev/null | grep -Fx "$TPU_NAME" || true)
    if [[ -n "$found" ]]; then
      log "STRAY: '$TPU_NAME' EXISTS in $z and IS BILLING."
      log "       Re-run as:  ZONE=$z $0 --yes"
      stray=1
    fi
  done
  (( stray )) && return 1
  log "swept zones [${TEARDOWN_SWEEP_ZONES:-none}] — nothing billing in any of them."
  return 0
}

exists=unknown
if command -v gcloud >/dev/null 2>&1 && [[ -n "${PROJECT:-}" ]]; then
  if tpu_exists; then
    exists=yes
  else
    exists=no
  fi
fi

if [[ "$STATUS_ONLY" == "true" ]]; then
  case "$exists" in
    yes)     log "'$TPU_NAME' EXISTS in $ZONE and is BILLING. Run this script without --status." ; exit 1 ;;
    no)      log "'$TPU_NAME' does not exist in $ZONE."
             sweep_other_zones || { log "NOT all-clear."; exit 1; }
             log "Nothing billing." ; exit 0 ;;
    unknown) log "could not determine state (no gcloud or no project set)." ; exit 0 ;;
  esac
fi

mapfile -t CMD < <(tpu_delete_argv)

if [[ "$DRY_RUN" == "true" ]]; then
  log "DRY RUN — the command that would run:"
  printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
  log "DRY RUN — nothing deleted."
  exit 0
fi

if [[ "$exists" == "no" ]]; then
  log "'$TPU_NAME' does not exist in $ZONE — nothing to delete."
  # "Nothing to delete" is the most dangerous sentence this script can print,
  # because it is what it says both when nothing is billing and when something
  # is billing SOMEWHERE ELSE. Session 25 provisioned in us-west4-a (the zone
  # with v5e capacity that day), ran teardown with ZONE unset, and got exactly
  # this message and exit 0 while a v5litepod-4 sat READY at $4.80/hr. Only a
  # manual `tpu-vm list` caught it. So never report all-clear on the strength of
  # one zone: sweep the others before claiming nothing is billing.
  if ! sweep_other_zones; then
    log "NOT all-clear — a TPU is billing in another zone (see above)."
    exit 1
  fi
  exit 0
fi

# Refuse to silently destroy uncaptured work.
if [[ ! -d "$HERE/../captured" ]]; then
  log "WARNING: no ./captured/ directory — has ./infra/capture.sh been run?"
  log "  The VM disk is deleted with the VM. Anything not captured is gone."
fi

if [[ "$ASSUME_YES" != "true" ]]; then
  echo "About to DELETE TPU VM '$TPU_NAME' in $ZONE (project $PROJECT)."
  echo "Anything on its local disk is lost. Results should already be in $GCS_BUCKET."
  read -r -p "Type 'yes' to confirm: " reply
  [[ "$reply" == "yes" ]] || die "aborted; VM still exists and is still billing."
fi

log "deleting…"
"${CMD[@]}"
log "deleted. Billing stopped."
log ""
log "Now record the billed VM-hours for this session in DECISIONS.md."

# Disarm any backstop this session armed. Belt and braces alongside the
# timestamp scoping in infra/deadman.sh: once the VM is gone deliberately, no
# switch should still be counting down toward a VM name that a later session
# may reuse.
pkill -f "tpu-vm delete ${TPU_NAME:-bucketladder-tpu}" 2>/dev/null || true
