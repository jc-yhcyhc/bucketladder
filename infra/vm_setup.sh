#!/usr/bin/env bash
# =============================================================================
# vm_setup.sh — run once on the TPU VM after create_tpu.sh.
# =============================================================================
# Installs tpu-inference + vLLM and starts a server, capturing the warmup log
# that e00_smoke_test.py parses. Deploy the repo first with infra/deploy.sh if
# you intend to run experiments on the VM rather than only capture the log.
#
# The tpu-ubuntu2204-base image quirks handled below were worked out the
# expensive way in a previous project (infersim/calibration/vm_setup.sh) and
# are the genuinely reusable part of it. THAT script installs MaxText and
# JetStream, which this project does not use — only these workarounds carry
# over. See notes/plan_v4.md, "Reuse".
#
# Known quirks of tpu-ubuntu2204-base:
#   - /etc/apt/sources.list.d/gcsfuse.list is malformed  -> apt-get update exits 1
#   - cnf-update-db post-invoke script exits non-zero    -> apt-get update exits 1
#   - system pip imports distutils (gone in 3.12)        -> pip crashes
#   - /usr/local/{lib,bin,share} are root-owned          -> user installs fail
#   - snap-installed gsutil throws a traceback           -> use the Python client
#
# Re-runnable: every step checks whether it is already done.
#
# Usage (on the VM):
#   bash vm_setup.sh                 # install + start server
#   bash vm_setup.sh --dry-run       # print what it would do, change nothing
#   bash vm_setup.sh --install-only  # no server
# =============================================================================

set -euo pipefail

DRY_RUN=false
INSTALL_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --dry-run)     DRY_RUN=true ;;
    --install-only) INSTALL_ONLY=true ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

: "${MODEL:=meta-llama/Llama-3.1-8B-Instruct}"
: "${TPU_INFERENCE_VERSION:=}"
: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"
: "${TP_SIZE:=1}"
: "${SERVER_PORT:=8000}"
: "${WARMUP_LOG:=/tmp/vllm_warmup.log}"
# Empty = vLLM default exponential (power-of-two) padding. An integer switches
# to linear buckets 16 -> max_model_len. THE independent variable.
: "${VLLM_TPU_BUCKET_PADDING_GAP:=}"

log() { echo "[$(date '+%H:%M:%S')] [vm_setup] $*"; }
run() {
  if [[ "$DRY_RUN" == "true" ]]; then printf '  WOULD RUN: '; printf '%q ' "$@"; printf '\n';
  else "$@"; fi
}

# ── Step 1: system packages ────────────────────────────────────────────────
log "Step 1: system packages"
if [[ "$DRY_RUN" != "true" ]]; then
  sudo rm -f /etc/apt/sources.list.d/gcsfuse.list 2>/dev/null || true
  sudo apt-get update -qq 2>/dev/null || true    # exits 1 on this image; non-fatal
  sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git curl jq 2>/dev/null || true
else
  log "  WOULD: rm malformed gcsfuse.list; apt-get update (tolerating exit 1); install python3.11 git curl jq"
fi

# ── Step 2: writable /usr/local + a clean venv ─────────────────────────────
log "Step 2: python environment"
if [[ "$DRY_RUN" != "true" ]]; then
  sudo chown -R "$(whoami)" /usr/local/lib /usr/local/bin /usr/local/share 2>/dev/null || true
  if [[ ! -d "$HOME/venv" ]]; then
    # --without-pip + get-pip.py because the system pip imports distutils and
    # python3-venv's ensurepip is unreliable on this image.
    python3 -m venv --without-pip "$HOME/venv"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$HOME/venv/bin/python" /tmp/get-pip.py -q
  fi
  "$HOME/venv/bin/python" -m pip install -q --upgrade pip setuptools wheel
else
  log "  WOULD: chown /usr/local/{lib,bin,share}; venv --without-pip + get-pip.py at ~/venv"
fi

# ── Step 3: tpu-inference + vLLM ───────────────────────────────────────────
log "Step 3: tpu-inference + vLLM"
PKG="tpu-inference"
[[ -n "$TPU_INFERENCE_VERSION" ]] && PKG="tpu-inference==${TPU_INFERENCE_VERSION}"
run "$HOME/venv/bin/python" -m pip install -q "$PKG"
run "$HOME/venv/bin/python" -m pip install -q pandas pyarrow

if [[ "$DRY_RUN" != "true" ]]; then
  log "  installed versions:"
  "$HOME/venv/bin/python" -m pip list 2>/dev/null | grep -Ei 'vllm|tpu-inference|jax|libtpu' || true
fi

# ── Step 4: HuggingFace auth (gated repo) ──────────────────────────────────
log "Step 4: HuggingFace auth"
if [[ "$DRY_RUN" != "true" ]]; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    mkdir -p "$HOME/.cache/huggingface"
    printf '%s' "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
    log "  token written"
  elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    log "  token already present"
  else
    log "  WARNING: no HF token. A gated model download will fail."
  fi
else
  log "  WOULD: write \$HF_TOKEN to ~/.cache/huggingface/token"
fi

[[ "$INSTALL_ONLY" == "true" ]] && { log "--install-only; done."; exit 0; }

# ── Step 5: serve, capturing the warmup log ────────────────────────────────
# The warmup log is the artifact e00_smoke_test.py parses to enumerate the
# ladder. Capturing it is the whole point of this step.
log "Step 5: start server (warmup log -> $WARMUP_LOG)"

SERVE=("$HOME/venv/bin/vllm" serve "$MODEL"
       --tensor-parallel-size "$TP_SIZE"
       --max-model-len "$MAX_MODEL_LEN"
       --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
       --no-enable-prefix-caching            # controlled variable: MUST be off
       --port "$SERVER_PORT")

if [[ "$DRY_RUN" == "true" ]]; then
  log "  WOULD export VLLM_TPU_BUCKET_PADDING_GAP='${VLLM_TPU_BUCKET_PADDING_GAP}'"
  printf '  WOULD RUN: '; printf '%q ' "${SERVE[@]}"; printf '\n'
  log "DRY RUN complete — nothing installed, nothing started."
  exit 0
fi

export VLLM_TPU_BUCKET_PADDING_GAP
log "  VLLM_TPU_BUCKET_PADDING_GAP='${VLLM_TPU_BUCKET_PADDING_GAP}' (empty = exponential default)"
log "  XLA warmup is slow: 5-30 min for the first bucket, 30-120 s per additional one."

"${SERVE[@]}" 2>&1 | tee "$WARMUP_LOG" &
SERVER_PID=$!

log "  waiting for /health on port $SERVER_PORT…"
for _ in $(seq 1 240); do
  if curl -sf "http://localhost:${SERVER_PORT}/health" >/dev/null 2>&1; then
    log "  server up (pid $SERVER_PID)"
    log ""
    log "The warmup log is at $WARMUP_LOG. Session 1's job is to GET IT OFF"
    log "the VM and tear down — the parser is then fixed offline at \$0:"
    log "  ./infra/capture.sh --tag <label>"
    log "  ./infra/teardown_tpu.sh"
    log ""
    log "Later sessions, with the repo deployed via ./infra/deploy.sh:"
    log "  cd ~/bucketladder && ~/venv/bin/python scripts/e00_smoke_test.py \\"
    log "      --config configs/e00_default_ladder.json --warmup-log $WARMUP_LOG"
    exit 0
  fi
  sleep 15
done

log "ERROR: server did not become healthy within 60 minutes. Check $WARMUP_LOG."
exit 1
