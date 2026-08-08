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
exists=unknown
if command -v gcloud >/dev/null 2>&1 && [[ -n "${PROJECT:-}" ]]; then
  if gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" \
       >/dev/null 2>&1; then
    exists=yes
  else
    exists=no
  fi
fi

if [[ "$STATUS_ONLY" == "true" ]]; then
  case "$exists" in
    yes)     log "'$TPU_NAME' EXISTS in $ZONE and is BILLING. Run this script without --status." ; exit 1 ;;
    no)      log "'$TPU_NAME' does not exist in $ZONE. Nothing billing." ; exit 0 ;;
    unknown) log "could not determine state (no gcloud or no project set)." ; exit 0 ;;
  esac
fi

CMD=(gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" --quiet)

if [[ "$DRY_RUN" == "true" ]]; then
  log "DRY RUN — the command that would run:"
  printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
  log "DRY RUN — nothing deleted."
  exit 0
fi

if [[ "$exists" == "no" ]]; then
  log "'$TPU_NAME' does not exist in $ZONE — nothing to delete."
  exit 0
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
