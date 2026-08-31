#!/usr/bin/env bash
# =============================================================================
# run_review_arms.sh — the MLSys review's hardware items, on a slice that
#                       already exists.
# =============================================================================
# One boot, Qwen3-4B, TP=4 -- the paper's own default config, not a special
# arm -- with infra/patch_step_logger.py applied so every step of everything
# below is logged for e15's reconciliation, and infra/patch_ladder.py NOT
# applied (nothing here needs a non-default token ladder).
#
#   1. e14_n1_all_boundaries  — the n<=2 paid share, all four boundaries,
#      with a real interval (review priority #2). m1_boundary.py.
#   2. m1_boundary.json (the paper's own default: n=4,8 at two boundaries)
#      — re-run here only to widen e15's concurrency coverage; not a new
#      claim, the existing n=4/8 numbers already stand.
#   3. e16_lens_n8_n16 — LENS at n=8 and n=16 under the synchronized
#      launcher (review priority #8). m5_lens_form.py.
#
# Together, 1-3 sweep n in {1, 4, 8, 16} while the step logger is live, which
# is what e15_step_reconcile.py needs to answer review priority #3: does
# per-step padded-token overhead stay flat or fall as concurrency rises.
#
# Usage:  ZONE=us-west4-a bash infra/run_review_arms.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

ZONE="${ZONE:-$(cat /tmp/bucketladder_zone 2>/dev/null)}"
NAME="${TPU_NAME:-bucketladder-tpu}"
BOOT_BUDGET="${BOOT_BUDGET:-4200}"   # 70 min: cold boot, no cache on a fresh slice
: "${ZONE:?ZONE not set and /tmp/bucketladder_zone absent}"

log() { echo "[$(date -u '+%H:%M:%S')] [review] $*"; }
ssh_try() {
  local cmd="$1" out
  for _ in 1 2 3; do
    if out=$(timeout 150 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" \
               --command="$cmd" 2>/dev/null); then printf '%s' "$out"; return 0; fi
    sleep 20
  done
  return 1
}

mkdir -p "$ROOT/results"
rc=0

log "applying infra/patch_step_logger.py on the VM"
if ! ssh_try 'cd ~/bucketladder && $HOME/venv/bin/python infra/patch_step_logger.py --apply 2>&1'; then
  log "FAILED to apply the step-logger patch -- aborting before anything boots"
  exit 2
fi

log "booting Qwen/Qwen3-4B TP=4 with BUCKETLADDER_LOG_STEP_SHAPES=1"
if ! ZONE="$ZONE" BUCKETLADDER_LOG_STEP_SHAPES=1 \
     bash "$HERE/boot_and_poll.sh" "Qwen/Qwen3-4B" 4 "$BOOT_BUDGET"; then
  log "BOOT FAILED -- recording the log tail and stopping"
  ssh_try 'tail -60 /tmp/vllm_warmup.log' > "$ROOT/results/review_bootfail.log" 2>/dev/null
  ssh_try '$HOME/venv/bin/python infra/patch_step_logger.py --revert' >/dev/null 2>&1
  exit 1
fi

run_config() {
  local script="$1" cfg="$2"
  log "=== $cfg ==="
  if ssh_try "cd ~/bucketladder && \$HOME/venv/bin/python scripts/$script \
                --config configs/${cfg}.json 2>&1 | tail -30"; then
    log "    $cfg complete"
  else
    log "    $cfg FAILED -- continuing to the next arm"
    rc=1
  fi
}

run_config m1_boundary.py e14_n1_all_boundaries
run_config m1_boundary.py m1_boundary
run_config m5_lens_form.py e16_lens_n8_n16

log "pulling the step log for e15's reconciliation"
ssh_try 'grep BUCKETLADDER_STEP /tmp/vllm_warmup.log' > "$ROOT/results/review_step_log.txt" 2>/dev/null
wc -l "$ROOT/results/review_step_log.txt" 2>/dev/null

log "reverting infra/patch_step_logger.py"
ssh_try '$HOME/venv/bin/python infra/patch_step_logger.py --revert' >/dev/null 2>&1

log "pulling artifacts back before anything is torn down"
ZONE="$ZONE" bash "$HERE/capture.sh" || log "capture reported a problem -- check before teardown"

log "arms finished (rc=$rc). Teardown is deliberately NOT automatic here;"
log "run: ZONE=$ZONE bash infra/teardown_tpu.sh --yes && bash infra/teardown_tpu.sh --status"
exit "$rc"
