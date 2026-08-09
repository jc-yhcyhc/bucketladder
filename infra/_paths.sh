#!/usr/bin/env bash
# =============================================================================
# _paths.sh — abstracts the TWO ways to provision a TPU. Source, don't execute.
# =============================================================================
# GCP exposes TPUs through two independent surfaces, with separate quota and
# separate CLI verbs. Which one you can use is not a preference — it is decided
# by which quota this project actually holds, and they disagree:
#
#   tpu-api  Cloud TPU API. `gcloud compute tpus tpu-vm ...`, accelerator types
#            like v5litepod-4 / v6e-1, runtime "versions" like
#            v2-alpha-tpuv5-lite. Quota metric: TPU_LITE_PODSLICE_V5 = 16 here,
#            on-demand AND preemptible. **This is the metric we verifiably
#            hold.** Not visible in the console's instance-creation flow, which
#            is why v5e appeared unavailable.
#
#   gce      GCE-native. `gcloud compute instances ...`, machine types like
#            ct5lp-hightpu-4t / ct6e-standard-1t, an accelerator boot image.
#            This is what the console's "Lightweight Exploration" creates. Its
#            v5e option (CT5LP) is absent from the console series list, and the
#            only v6e quota metric found carries preemptible-only limits.
#
# Default is tpu-api because that is where our quota is unambiguous. Override:
#   PROVISION_PATH=gce ./infra/create_tpu.sh
# =============================================================================

: "${PROVISION_PATH:=tpu-api}"

case "$PROVISION_PATH" in
  tpu-api|gce) ;;
  *) echo "PROVISION_PATH must be 'tpu-api' or 'gce', got: $PROVISION_PATH" >&2; return 1 2>/dev/null || exit 2 ;;
esac

# --- existence --------------------------------------------------------------
tpu_exists() {
  if [[ "$PROVISION_PATH" == "tpu-api" ]]; then
    gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
  else
    gcloud compute instances describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1
  fi
}

# --- create -----------------------------------------------------------------
# Emits the command as an array on stdout, one argument per line, so callers can
# print it for --dry-run or execute it via mapfile.
tpu_create_argv() {
  if [[ "$PROVISION_PATH" == "tpu-api" ]]; then
    printf '%s\n' gcloud compute tpus tpu-vm create "$TPU_NAME" \
      --zone="$ZONE" --project="$PROJECT" \
      --accelerator-type="$ACCELERATOR_TYPE" --version="$RUNTIME_VERSION"
    [[ "$SPOT" == "true" ]] && printf '%s\n' --spot
  else
    printf '%s\n' gcloud compute instances create "$TPU_NAME" \
      --zone="$ZONE" --project="$PROJECT" \
      --machine-type="$MACHINE_TYPE" \
      --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
      --boot-disk-size="$BOOT_DISK_SIZE" \
      --scopes=https://www.googleapis.com/auth/cloud-platform
    if [[ "$SPOT" == "true" ]]; then
      # DELETE, not the STOP default: a preempted VM must stop billing outright
      # rather than linger as a stopped instance with a paid disk.
      printf '%s\n' --provisioning-model=SPOT --instance-termination-action=DELETE
    fi
  fi
}

tpu_delete_argv() {
  if [[ "$PROVISION_PATH" == "tpu-api" ]]; then
    printf '%s\n' gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" --quiet
  else
    printf '%s\n' gcloud compute instances delete "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" --quiet
  fi
}

# --- ssh / scp --------------------------------------------------------------
tpu_ssh() {
  if [[ "$PROVISION_PATH" == "tpu-api" ]]; then
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" "$@"
  else
    gcloud compute ssh "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" "$@"
  fi
}

tpu_scp() {
  if [[ "$PROVISION_PATH" == "tpu-api" ]]; then
    gcloud compute tpus tpu-vm scp --zone="$ZONE" --project="$PROJECT" "$@"
  else
    gcloud compute scp --zone="$ZONE" --project="$PROJECT" "$@"
  fi
}

# What this path is actually going to create, for logging.
tpu_target_desc() {
  if [[ "$PROVISION_PATH" == "tpu-api" ]]; then
    echo "$ACCELERATOR_TYPE (runtime $RUNTIME_VERSION) via Cloud TPU API"
  else
    echo "$MACHINE_TYPE (image $IMAGE_FAMILY) via GCE instance"
  fi
}
