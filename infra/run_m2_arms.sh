#!/usr/bin/env bash
# =============================================================================
# run_m2_arms.sh — the three M2 MoE arms, end to end, on a slice that already exists.
# =============================================================================
# Arms, all at TP=4 and all on the vLLM implementation. TP=4 is forced by the
# hardware, not chosen: the mesh is (model_dp_size, tp_size) over every visible
# chip, so DP x TP must equal the slice's 4 -- and any DP>1 disables the very
# flag under test, because the interface gates it on `not is_dp`.
#
#   A  m2_moe_qwen_default     Qwen3-30B-A3B-FP8, MOE_ROUTE_PADDING_TO_EXPERT0=0 (stock)
#   B  m2_moe_qwen_expert0     Qwen3-30B-A3B-FP8, MOE_ROUTE_PADDING_TO_EXPERT0=1 (mitigated)
#   C  m2_dense_control        Qwen3-4B,          MODEL_IMPL_TYPE=vllm           (dense reference)
#
# A vs B is the strong comparison: same model, same slice, same ladder, one env
# var. C gives the dense per-padded-token reference at the paper's own TP=4, so
# this arm is directly comparable to the existing results.
#
# Two things are verified from the warmup log rather than assumed, because both
# fail silently:
#   * MODEL_IMPL_TYPE must resolve to 'vllm'. On flax_nnx the flag does not
#     exist and arm B would be a byte-identical rerun of arm A.
#   * MOE_ROUTE_PADDING_TO_EXPERT0 fails OPEN -- if query_start_loc cannot be
#     read the interface warns once and serves the unmitigated path.
#
# Usage:  ZONE=us-west4-a bash infra/run_m2_arms.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

ZONE="${ZONE:-$(cat /tmp/bucketladder_zone 2>/dev/null)}"
NAME="${TPU_NAME:-bucketladder-tpu}"
BOOT_BUDGET="${BOOT_BUDGET:-2700}"   # 45 min per arm, then move on
: "${ZONE:?ZONE not set and /tmp/bucketladder_zone absent}"

log() { echo "[$(date -u '+%H:%M:%S')] [m2] $*"; }
ssh_try() {
  local cmd="$1" out
  for _ in 1 2 3; do
    if out=$(timeout 150 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" \
               --command="$cmd" 2>/dev/null); then printf '%s' "$out"; return 0; fi
    sleep 20
  done
  return 1
}

run_arm() {
  local arm="$1" cfg="$2" model="$3" impl="$4" expert0="$5"
  log "=== arm $arm : $cfg ==="
  log "    model=$model impl=$impl expert0=$expert0"

  if ! ZONE="$ZONE" MODEL_IMPL_TYPE="$impl" MOE_ROUTE_PADDING_TO_EXPERT0="$expert0" \
       bash "$HERE/boot_and_poll.sh" "$model" 4 "$BOOT_BUDGET"; then
    log "    arm $arm FAILED TO BOOT -- recording and moving on"
    ssh_try 'tail -40 /tmp/vllm_warmup.log' > "$ROOT/results/m2_${arm}_bootfail.log" 2>/dev/null
    return 1
  fi

  # Did the run we just booted actually have the property the arm claims?
  local resolved warned
  resolved=$(ssh_try "grep -ao \"Resolved MODEL_IMPL_TYPE '[a-z]*' to '[a-z_]*'\" /tmp/vllm_warmup.log | tail -1")
  warned=$(ssh_try 'grep -c "MOE_ROUTE_PADDING_TO_EXPERT0: failed to read" /tmp/vllm_warmup.log 2>/dev/null || echo 0')
  log "    $resolved"
  log "    fail-open warnings: ${warned:-?}"
  if [[ "$resolved" != *"to 'vllm'"* ]]; then
    log "    REFUSING arm $arm: implementation is not vllm, so the flag is inert here"
    return 1
  fi
  if [[ "$expert0" == "1" && "${warned:-1}" != "0" ]]; then
    log "    REFUSING arm $arm: flag failed open, this is arm A again, not arm B"
    return 1
  fi

  ssh_try "cd ~/bucketladder && \$HOME/venv/bin/python scripts/m1_boundary.py \
             --config configs/${cfg}.json 2>&1 | tail -25"
  log "    arm $arm measurement complete"
  return 0
}

mkdir -p "$ROOT/results"
rc=0
run_arm A m2_moe_qwen_default Qwen/Qwen3-30B-A3B-FP8 auto 0 || rc=1
run_arm B m2_moe_qwen_expert0 Qwen/Qwen3-30B-A3B-FP8 auto 1 || rc=1
run_arm C m2_dense_control    Qwen/Qwen3-4B          vllm 0 || rc=1

log "pulling artifacts back before anything is torn down"
bash "$HERE/capture.sh" || log "capture reported a problem -- check before teardown"
log "arms finished (rc=$rc). Teardown is deliberately NOT automatic here;"
log "run: ZONE=$ZONE bash infra/teardown_tpu.sh --yes && bash infra/teardown_tpu.sh --status"
exit "$rc"
