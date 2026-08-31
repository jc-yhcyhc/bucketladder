#!/usr/bin/env bash
# =============================================================================
# create_gpu.sh — provision the L4 GCE instance for the GPU control's repeats.
# =============================================================================
# A standalone script, not a third PROVISION_PATH branch in _paths.sh: the L4
# is a different resource type (a GPU attached to a generic GCE VM, not a TPU
# accelerator-type or a TPU-shaped machine type), and threading it through TPU
# -specific argv builders would make both harder to read for no shared benefit.
# Same safety shape as create_tpu.sh on purpose: preflight, --dry-run,
# idempotency, an explicit billing-rate warning, a stable success marker.
#
# PREREQUISITES:
#   1. NVIDIA_L4_GPUS quota in $ZONE's region. VERIFIED 2026-08-31: 1.0 in
#      us-central1 (on-demand, spot, and committed all read 1.0).
#   2. gcloud authenticated, PROJECT set (config.env, shared with the TPU
#      scripts -- same project, different resource).
#
# Usage:
#   ./infra/create_gpu.sh --dry-run
#   ./infra/create_gpu.sh --check
#   ./infra/create_gpu.sh              # actually create (SPENDS MONEY)
#   SPOT=true ./infra/create_gpu.sh    # spot instead of on-demand
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./config.env
source "$HERE/config.env"

: "${GPU_NAME:=bucketladder-gpu}"
: "${GPU_ZONE:=us-central1-a}"
: "${GPU_TYPE:=nvidia-l4}"
: "${GPU_MACHINE_TYPE:=g2-standard-4}"   # the paired CPU/RAM shape for one L4
: "${GPU_IMAGE_FAMILY:=common-cu129-ubuntu-2204-nvidia-580}"
: "${GPU_IMAGE_PROJECT:=deeplearning-platform-release}"  # CUDA + drivers preinstalled
: "${GPU_BOOT_DISK_SIZE:=100GB}"
: "${GPU_PRICE_PER_HOUR_ONDEMAND:=0.86}"   # g2-standard-4 + 1x L4, us-central1, list price
: "${GPU_PRICE_PER_HOUR_SPOT:=0.30}"
: "${SPOT:=false}"

DRY_RUN=false
CHECK_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --check)   CHECK_ONLY=true ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

log()  { echo "[$(date '+%H:%M:%S')] [create_gpu] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] [create_gpu] WARNING: $*" >&2; }
die()  { echo "[$(date '+%H:%M:%S')] [create_gpu] ERROR: $*" >&2; exit 1; }

have_gcloud() { command -v gcloud >/dev/null 2>&1; }

check_accel() {
  have_gcloud || return 0
  # `gcloud compute accelerator-types list` (unlike the TPU-API equivalent)
  # takes no --zone flag; zone must be a filter term, not a flag -- found by
  # hitting "unrecognized arguments: --zone" when this was written by analogy
  # with create_tpu.sh's TPU-API check.
  if timeout 60 gcloud compute accelerator-types list \
       --filter="name=$GPU_TYPE AND zone:$GPU_ZONE" \
       --project="$PROJECT" --format='value(name)' 2>/dev/null | grep -qx "$GPU_TYPE"; then
    log "  ok  '$GPU_TYPE' offered in $GPU_ZONE"
    return 0
  fi
  warn "'$GPU_TYPE' is NOT offered in $GPU_ZONE."
  return 1
}

check_image() {
  have_gcloud || return 0
  # Written once by guessing a plausible Deep Learning VM family name
  # (common-cu124-ubuntu-2204-py310); it does not exist -- the actual
  # families use a driver-version suffix instead (...-nvidia-580). Checked
  # for real here so a wrong guess fails before billing starts, not after.
  local img
  img=$(timeout 60 gcloud compute images describe-from-family "$GPU_IMAGE_FAMILY" \
          --project="$GPU_IMAGE_PROJECT" --format='value(name)' 2>/dev/null) || true
  if [[ -n "$img" ]]; then
    log "  ok  image family '$GPU_IMAGE_FAMILY' resolves to $img"
    return 0
  fi
  warn "image family '$GPU_IMAGE_FAMILY' not found in project '$GPU_IMAGE_PROJECT'."
  warn "  List real ones: gcloud compute images list --project=$GPU_IMAGE_PROJECT --filter='family~cu1'"
  return 1
}

check_quota() {
  have_gcloud || return 0
  local region metric quotas limit
  region="${GPU_ZONE%-*}"
  metric="NVIDIA_L4_GPUS"
  [[ "$SPOT" == "true" ]] && metric="PREEMPTIBLE_${metric}"
  quotas=$(timeout 60 gcloud compute regions describe "$region" --project="$PROJECT" \
             --format="value(quotas)" 2>/dev/null) || return 0
  [[ -z "$quotas" ]] && return 0
  limit=$(tr ';' '\n' <<<"$quotas" | grep -F "'$metric'" | sed -E "s/.*'limit': ([0-9.]+).*/\1/" | head -1)
  if [[ -z "$limit" ]]; then
    log "  --  quota metric $metric not exposed by 'regions describe' in $region"
    return 0
  fi
  if awk -v l="$limit" 'BEGIN{exit !(l>=1)}'; then
    log "  ok  quota $metric = $limit in $region (need 1)"
    return 0
  fi
  warn "INSUFFICIENT QUOTA: $metric = $limit in $region, need 1."
  return 1
}

gpu_exists() {
  gcloud compute instances describe "$GPU_NAME" --zone="$GPU_ZONE" --project="$PROJECT" >/dev/null 2>&1
}

preflight() {
  local problems=0
  have_gcloud || { warn "gcloud not on PATH"; problems=$((problems+1)); }
  [[ -n "${PROJECT:-}" ]] || { warn "PROJECT is empty"; problems=$((problems+1)); }
  if have_gcloud && [[ -n "${PROJECT:-}" ]]; then
    log "verifying against the live API (read-only):"
    check_accel || problems=$((problems+1))
    check_image || problems=$((problems+1))
    check_quota || problems=$((problems+1))
  fi
  log "config: project=$PROJECT zone=$GPU_ZONE name=$GPU_NAME"
  log "  type=$GPU_TYPE machine=$GPU_MACHINE_TYPE spot=$SPOT"
  local rate; if [[ "$SPOT" == "true" ]]; then rate="$GPU_PRICE_PER_HOUR_SPOT"; else rate="$GPU_PRICE_PER_HOUR_ONDEMAND"; fi
  log "  BILLED RATE: \$${rate}/hr while this VM EXISTS (not while it computes)"
  log "  24h of forgetting to tear down = \$$(awk -v r="$rate" 'BEGIN{printf "%.0f", r*24}')"
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

if [[ "$DRY_RUN" != "true" ]] && have_gcloud && gpu_exists; then
  log "GPU VM '$GPU_NAME' already exists in $GPU_ZONE — nothing to do."
  log "It is BILLING right now. Tear down with: ./infra/teardown_gpu.sh"
  exit 0
fi

CMD=(gcloud compute instances create "$GPU_NAME"
     --zone="$GPU_ZONE" --project="$PROJECT"
     --machine-type="$GPU_MACHINE_TYPE"
     --accelerator="type=$GPU_TYPE,count=1"
     --maintenance-policy=TERMINATE   # required whenever a GPU is attached
     --image-family="$GPU_IMAGE_FAMILY" --image-project="$GPU_IMAGE_PROJECT"
     --boot-disk-size="$GPU_BOOT_DISK_SIZE"
     --scopes=https://www.googleapis.com/auth/cloud-platform)
if [[ "$SPOT" == "true" ]]; then
  CMD+=(--provisioning-model=SPOT --instance-termination-action=DELETE)
fi

if [[ "$DRY_RUN" == "true" ]]; then
  log "DRY RUN — the command that would run:"
  printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
  log "DRY RUN — nothing created, nothing billed."
  exit 0
fi

log "creating GPU VM (this takes a few minutes, plus driver install on first boot)…"
"${CMD[@]}"
log "created. BILLING HAS STARTED."
echo "BUCKETLADDER_GPU_CREATED_MARKER ${GPU_ZONE}"
log ""
log "Next:"
log "  1. ./infra/deploy.sh (repo -> VM) may need GPU_NAME/GPU_ZONE if it assumes TPU_NAME"
log "  2. gcloud compute ssh $GPU_NAME --zone=$GPU_ZONE --command='nvidia-smi'  # confirm the driver"
log "  3. install vLLM + deps (no vm_setup.sh equivalent exists yet for this image)"
log "  4. ./infra/capture.sh                      # GET THE LOG OFF"
log "  5. ./infra/teardown_gpu.sh                 # ALWAYS"
