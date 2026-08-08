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
# These are REAL read-only API calls, not prose. Every one of them catches a
# failure that would otherwise happen after `create` has already started
# billing, or several minutes into it.

have_gcloud() { command -v gcloud >/dev/null 2>&1; }

check_accelerator_type() {
  have_gcloud || return 0
  local types
  types=$(timeout 60 gcloud compute tpus accelerator-types list \
            --zone="$ZONE" --project="$PROJECT" --format='value(type)' 2>/dev/null) || return 0
  [[ -z "$types" ]] && return 0
  if grep -qx "$TPU_TYPE" <<<"$types"; then
    log "  ok  accelerator-type '$TPU_TYPE' available in $ZONE"
    return 0
  fi
  warn "accelerator-type '$TPU_TYPE' is NOT offered in $ZONE."
  warn "  Run ./infra/find_zone.sh for the current list — v5e is offered in ~25"
  warn "  zones across four continents, so a bad zone is a one-line fix."
  return 1
}

check_runtime_version() {
  have_gcloud || return 0
  local versions
  # `value(name)` returns full resource paths; the version is the last segment.
  versions=$(timeout 60 gcloud compute tpus versions list \
               --zone="$ZONE" --project="$PROJECT" --format='value(name)' 2>/dev/null \
             | awk -F/ '{print $NF}') || return 0
  [[ -z "$versions" ]] && return 0
  if grep -qx "$RUNTIME_VERSION" <<<"$versions"; then
    log "  ok  runtime version '$RUNTIME_VERSION' valid in $ZONE"
    return 0
  fi
  warn "runtime version '$RUNTIME_VERSION' is NOT valid in $ZONE."
  warn "  v5e candidates seen: v2-alpha-tpuv5-lite, v2-tpuv5-litepod, v2-tpuv5-lite-cgroup1"
  return 1
}

check_quota() {
  have_gcloud || return 0
  local region metric quotas limit
  region="${ZONE%-*}"
  # v5litepod-N is a PODSLICE, not a DEVICE. Device quota being 0 is normal and
  # irrelevant here — checking the wrong metric is exactly how "quota approved"
  # turns into a failed provision.
  if [[ "$SPOT" == "true" ]]; then
    metric="PREEMPTIBLE_TPU_LITE_PODSLICE_V5"
  else
    metric="TPU_LITE_PODSLICE_V5"
  fi
  quotas=$(timeout 60 gcloud compute regions describe "$region" --project="$PROJECT" \
             --format="value(quotas)" 2>/dev/null) || return 0
  [[ -z "$quotas" ]] && return 0

  limit=$(tr ';' '\n' <<<"$quotas" | grep -F "'$metric'" | sed -E "s/.*'limit': ([0-9.]+).*/\1/" | head -1)
  if [[ -z "$limit" ]]; then
    warn "could not read quota metric $metric in $region — check manually"
    return 1
  fi
  if awk -v l="$limit" -v c="$CHIPS" 'BEGIN{exit !(l>=c)}'; then
    log "  ok  quota $metric = $limit chips in $region (need $CHIPS)"
    return 0
  fi
  warn "INSUFFICIENT QUOTA: $metric = $limit in $region, need $CHIPS chips."
  warn "  Request an increase before provisioning; create will fail otherwise."
  return 1
}

preflight() {
  local problems=0

  have_gcloud || { warn "gcloud not on PATH"; problems=$((problems+1)); }

  [[ -n "${PROJECT:-}" ]] || { warn "PROJECT is empty; run: gcloud config set project <id>"; problems=$((problems+1)); }

  if [[ -z "${HF_TOKEN:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
    warn "no HF_TOKEN and no ~/.cache/huggingface/token — the gated meta-llama"
    warn "  download will fail on the VM. Request access first (prerequisite 2)."
    problems=$((problems+1))
  fi

  if have_gcloud && [[ -n "${PROJECT:-}" ]]; then
    log "verifying against the live API (read-only):"
    check_accelerator_type || problems=$((problems+1))
    check_runtime_version  || problems=$((problems+1))
    check_quota            || problems=$((problems+1))
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
log "  2. ./infra/deploy.sh                       # repo -> VM"
log "  3. gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE \\"
log "       --command='bash ~/bucketladder/infra/vm_setup.sh'"
log "  4. ./infra/capture.sh --tag <label>        # GET THE LOG OFF"
log "  5. ./infra/teardown_tpu.sh                 # ALWAYS"
