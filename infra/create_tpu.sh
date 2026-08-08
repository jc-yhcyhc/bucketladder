#!/usr/bin/env bash
# =============================================================================
# create_tpu.sh — provision the v5e-4 TPU VM.
# =============================================================================
#
# PREREQUISITES — satisfy these before running, or this will fail late and
# confusingly. They are checked by --check where checkable.
#
#   1. TPU quota approved for **v5e specifically, in $ZONE**. Quota approved
#      for one generation or one zone does not grant another. The v1 review
#      recorded "quota approved but unprovisioned" without naming the
#      generation — confirm which you actually have.
#        gcloud compute regions describe "${ZONE%-*}" --format='value(quotas)'
#
#   2. Gated `meta-llama` repo access requested and granted on HuggingFace.
#      Commonly a multi-hour stall. Start it first.
#        https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#
#   3. HF_TOKEN exported, or ~/.cache/huggingface/token present.
#
# THIS SCRIPT IS RE-RUNNABLE BY DESIGN. The cost discipline in plan_v4.md is
# "delete the VM at the end of every working session and re-create it", which
# only works if re-creating is one command. It is idempotent: if the VM already
# exists it says so and exits 0.
#
# Usage:
#   ./infra/create_tpu.sh --dry-run     # print the gcloud command, change nothing
#   ./infra/create_tpu.sh --check       # verify prerequisites only
#   ./infra/create_tpu.sh               # actually create (SPENDS MONEY)
#   SPOT=true ./infra/create_tpu.sh     # ~$1.40/hr instead of ~$4.80/hr
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.env
source "$HERE/config.env"

DRY_RUN=false
CHECK_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --check)   CHECK_ONLY=true ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log()  { echo "[$(date '+%H:%M:%S')] [create_tpu] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] [create_tpu] WARNING: $*" >&2; }
die()  { echo "[$(date '+%H:%M:%S')] [create_tpu] ERROR: $*" >&2; exit 1; }

# ── Preflight ───────────────────────────────────────────────────────────────
preflight() {
  local problems=0

  command -v gcloud >/dev/null 2>&1 || { warn "gcloud not on PATH"; problems=$((problems+1)); }

  [[ -n "${PROJECT:-}" ]] || { warn "PROJECT is empty; run: gcloud config set project <id>"; problems=$((problems+1)); }

  if [[ -z "${HF_TOKEN:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
    warn "no HF_TOKEN and no ~/.cache/huggingface/token — the gated meta-llama"
    warn "  download will fail on the VM. Request access first (prerequisite 2)."
    problems=$((problems+1))
  fi

  if [[ -z "${TPU_INFERENCE_VERSION:-}" ]]; then
    warn "TPU_INFERENCE_VERSION is unpinned (latest). Fine for bring-up;"
    warn "  pin it before any run whose numbers reach the paper."
  fi

  log "config:"
  log "  project=$PROJECT  zone=$ZONE"
  log "  name=$TPU_NAME  type=$TPU_TYPE  runtime=$RUNTIME_VERSION"
  log "  spot=$SPOT  model=$MODEL"

  local rate chips_rate
  if [[ "$SPOT" == "true" ]]; then rate="$PRICE_PER_CHIP_HOUR_SPOT"; else rate="$PRICE_PER_CHIP_HOUR_ONDEMAND"; fi
  chips_rate=$(awk -v c="$CHIPS" -v r="$rate" 'BEGIN{printf "%.2f", c*r}')
  log "  BILLED RATE: \$${chips_rate}/hr while this VM EXISTS (not while it computes)"
  log "  24h of forgetting to tear down = \$$(awk -v x="$chips_rate" 'BEGIN{printf "%.0f", x*24}')"

  return $problems
}

if ! preflight; then
  if [[ "$CHECK_ONLY" == "true" || "$DRY_RUN" == "true" ]]; then
    log "preflight reported problems (above). Not fatal in --check/--dry-run."
  else
    die "preflight failed; fix the warnings above or re-run with --dry-run to inspect."
  fi
fi

[[ "$CHECK_ONLY" == "true" ]] && { log "--check only; nothing created."; exit 0; }

# ── Idempotency ─────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" != "true" ]] && command -v gcloud >/dev/null 2>&1; then
  if gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" \
       >/dev/null 2>&1; then
    log "TPU '$TPU_NAME' already exists in $ZONE — nothing to do."
    log "It is BILLING right now. Tear down with: ./infra/teardown_tpu.sh"
    exit 0
  fi
fi

# ── Build the command ───────────────────────────────────────────────────────
CMD=(gcloud compute tpus tpu-vm create "$TPU_NAME"
     --zone="$ZONE"
     --project="$PROJECT"
     --accelerator-type="$TPU_TYPE"
     --version="$RUNTIME_VERSION")
[[ "$SPOT" == "true" ]] && CMD+=(--spot)

if [[ "$DRY_RUN" == "true" ]]; then
  log "DRY RUN — the command that would run:"
  printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
  log "then: ./infra/vm_setup.sh would be copied to the VM and executed"
  log "DRY RUN — nothing created, nothing billed."
  exit 0
fi

log "creating TPU VM (this takes a few minutes)…"
"${CMD[@]}"
log "created. BILLING HAS STARTED."
log ""
log "Next:"
log "  1. record the start time in DECISIONS.md (billed VM-hours, not benchmark-hours)"
log "  2. gcloud compute tpus tpu-vm scp infra/vm_setup.sh ${TPU_NAME}: --zone=$ZONE"
log "  3. gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --command='bash vm_setup.sh'"
log "  4. WHEN DONE FOR THE SESSION: ./infra/teardown_tpu.sh"
