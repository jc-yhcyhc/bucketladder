#!/usr/bin/env bash
# =============================================================================
# create_tpu.sh — provision the TPU.
# =============================================================================
#
# PREREQUISITES — satisfy these before running, or this will fail late and
# confusingly. They are checked by --check where checkable.
#
#   1. TPU quota for v5e in $ZONE's region. VERIFIED: TPU_LITE_PODSLICE_V5 = 16
#      chips (and 16 preemptible) as a global default. Note `regions describe`
#      hides some TPU metrics — use `gcloud alpha services quota list
#      --service=compute.googleapis.com` for the full picture.
#
#      TWO SURFACES EXIST and they have separate quota — see infra/_paths.sh.
#      PROVISION_PATH defaults to `tpu-api` (Cloud TPU API), because that is
#      where our quota is unambiguous and it offers v5litepod-4 even though the
#      console's instance-creation flow does not list CT5LP at all.
#      PROVISION_PATH=gce falls back to the console's GCE-native path (v6e).
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
#   SPOT=true ./infra/create_tpu.sh     # spot instead of on-demand
#   PROVISION_PATH=gce ./infra/create_tpu.sh   # GCE fallback (v6e, TP=1)
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.env
source "$HERE/config.env"
# shellcheck source=./_paths.sh
source "$HERE/_paths.sh"

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

check_machine_type() {
  have_gcloud || return 0
  if [[ "$PROVISION_PATH" == "tpu-api" ]]; then
    if timeout 60 gcloud compute tpus accelerator-types list --zone="$ZONE" \
         --project="$PROJECT" --format='value(type)' 2>/dev/null | grep -qx "$ACCELERATOR_TYPE"; then
      log "  ok  accelerator-type '$ACCELERATOR_TYPE' available in $ZONE (Cloud TPU API)"
      return 0
    fi
    warn "accelerator-type '$ACCELERATOR_TYPE' is NOT offered in $ZONE."
    warn "  List them: gcloud compute tpus accelerator-types list --zone=$ZONE"
    return 1
  fi
  if timeout 60 gcloud compute machine-types describe "$MACHINE_TYPE" \
       --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
    log "  ok  machine-type '$MACHINE_TYPE' available in $ZONE (GCE)"
    return 0
  fi
  warn "machine-type '$MACHINE_TYPE' is NOT offered in $ZONE."
  warn "  Run ./infra/find_zone.sh for the current list — v5e is offered in ~25"
  warn "  zones across four continents, so a bad zone is a one-line fix."
  return 1
}

check_image() {
  have_gcloud || return 0
  if [[ "$PROVISION_PATH" == "tpu-api" ]]; then
    local versions
    versions=$(timeout 60 gcloud compute tpus versions list --zone="$ZONE" \
                 --project="$PROJECT" --format='value(name)' 2>/dev/null | awk -F/ '{print $NF}')
    if grep -qx "$RUNTIME_VERSION" <<<"$versions"; then
      log "  ok  runtime version '$RUNTIME_VERSION' valid in $ZONE"
      return 0
    fi
    warn "runtime version '$RUNTIME_VERSION' is NOT valid in $ZONE."
    return 1
  fi
  local img
  img=$(timeout 60 gcloud compute images describe-from-family "$IMAGE_FAMILY" \
          --project="$IMAGE_PROJECT" --format='value(name)' 2>/dev/null) || true
  if [[ -n "$img" ]]; then
    log "  ok  image family '$IMAGE_FAMILY' resolves to $img"
    return 0
  fi
  warn "image family '$IMAGE_FAMILY' not found in project '$IMAGE_PROJECT'."
  warn "  v5e images live in ubuntu-os-accelerator-images; list them with:"
  warn "  gcloud compute images list --filter='name~tpu'"
  return 1
}

check_quota() {
  have_gcloud || return 0
  local region metric quotas limit
  region="${ZONE%-*}"
  # v5litepod-N is a PODSLICE, not a DEVICE. Device quota being 0 is normal and
  # irrelevant here — checking the wrong metric is exactly how "quota approved"
  # turns into a failed provision.
  # Metric depends on the TPU family. v5e is "TPU LITE ... V5"; v6e has its own.
  # NOTE `gcloud compute regions describe` does NOT surface every TPU metric —
  # v6e in particular is only visible via `gcloud alpha services quota list
  # --service=compute.googleapis.com`. A blank result here is inconclusive, not
  # a failure, which is why check_quota returns 0 when it cannot read a limit.
  local target
  if [[ "$PROVISION_PATH" == "tpu-api" ]]; then target="$ACCELERATOR_TYPE"; else target="$MACHINE_TYPE"; fi
  case "$target" in
    v5litepod-*|ct5lp-*) metric="TPU_LITE_PODSLICE_V5" ;;
    v6e-*|ct6e-*)        metric="TPU_V6E" ;;
    v5p-*|ct5p-*)        metric="TPU_V5P" ;;
    tpu7x-*)             metric="TPU7X" ;;
    *)                   metric="TPU_LITE_PODSLICE_V5" ;;
  esac
  [[ "$SPOT" == "true" ]] && metric="PREEMPTIBLE_${metric}"
  quotas=$(timeout 60 gcloud compute regions describe "$region" --project="$PROJECT" \
             --format="value(quotas)" 2>/dev/null) || return 0
  [[ -z "$quotas" ]] && return 0

  limit=$(tr ';' '\n' <<<"$quotas" | grep -F "'$metric'" | sed -E "s/.*'limit': ([0-9.]+).*/\1/" | head -1)
  if [[ -z "$limit" ]]; then
    log "  --  quota metric $metric not exposed by 'regions describe' in $region"
    log "      (expected for v6e; check with: gcloud alpha services quota list \\"
    log "       --service=compute.googleapis.com --consumer=projects/$PROJECT)"
    return 0
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
    check_machine_type || problems=$((problems+1))
    check_image        || problems=$((problems+1))
    check_quota            || problems=$((problems+1))
  fi

  if [[ -z "${TPU_INFERENCE_VERSION:-}" ]]; then
    warn "TPU_INFERENCE_VERSION is unpinned (latest). Fine for bring-up;"
    warn "  pin it before any run whose numbers reach the paper."
  fi

  log "config:"
  log "  project=$PROJECT  zone=$ZONE"
  log "  name=$TPU_NAME  path=$PROVISION_PATH"
  log "  target=$(tpu_target_desc)  TP=$TP_SIZE  chips=$CHIPS"
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
  if tpu_exists; then
    log "TPU '$TPU_NAME' already exists in $ZONE — nothing to do."
    log "It is BILLING right now. Tear down with: ./infra/teardown_tpu.sh"
    exit 0
  fi
fi

# ── Build the command ───────────────────────────────────────────────────────
mapfile -t CMD < <(tpu_create_argv)

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
log "  3. gcloud compute ssh $TPU_NAME --zone=$ZONE \\"
log "       --command='bash ~/bucketladder/infra/vm_setup.sh'"
log "  4. ./infra/capture.sh --tag <label>        # GET THE LOG OFF"
log "  5. ./infra/teardown_tpu.sh                 # ALWAYS"
