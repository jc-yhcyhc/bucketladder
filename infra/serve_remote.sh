#!/usr/bin/env bash
# =============================================================================
# serve_remote.sh — start/stop the vLLM server ON THE VM, restartably.
# =============================================================================
# vm_setup.sh starts a server as the last step of installation, which is right
# for session 1 (get a warmup log, tear down) and wrong for any session that
# needs to serve a SECOND model. Two problems it does not solve:
#
#   1. The server is a background child of the ssh command's shell. Whether it
#      survives the ssh disconnect is luck, not design.
#   2. **Stopping it is genuinely hard.** vLLM rewrites its worker process title
#      to `VLLM::EngineCore`, so it does not match `vllm`, `python`, or the
#      serve command line — session 1 lost a restart to this, and there is no
#      /dev/accel* to lsof on this runtime, so the usual "who holds the device"
#      trick does not work either.
#
# Both are fixed by owning the process group: `setsid` puts the server in a new
# session whose PGID we record, so it outlives ssh AND can be killed as a unit
# regardless of what any child renames itself to.
#
# The TPU is held exclusively by one process group. A second server cannot start
# until the first is gone, so `start` refuses rather than producing a confusing
# device-in-use failure deep in the log.
#
# Usage (on the VM):
#   bash serve_remote.sh start Qwen/Qwen3-4B
#   bash serve_remote.sh stop
#   bash serve_remote.sh status
# =============================================================================

set -uo pipefail

: "${MAX_MODEL_LEN:=8192}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"
: "${TP_SIZE:=4}"
: "${SERVER_PORT:=8000}"
: "${WARMUP_LOG:=/tmp/vllm_warmup.log}"
: "${PGID_FILE:=/tmp/vllm.pgid}"
: "${VENV:=$HOME/venv}"
# Empty = vLLM's default exponential (power-of-two) ladder. MUST be unset rather
# than empty when defaulting: vLLM does int(os.environ[...]) unconditionally
# when the variable is present, so "" kills the engine (verified 2026-08-09).
: "${VLLM_TPU_BUCKET_PADDING_GAP:=}"

log() { echo "[$(date '+%H:%M:%S')] [serve] $*"; }

healthy() { curl -sf "http://localhost:${SERVER_PORT}/health" >/dev/null 2>&1; }

cmd_status() {
  if healthy; then
    log "server HEALTHY on port $SERVER_PORT"
    [[ -f "$PGID_FILE" ]] && log "  pgid $(cat "$PGID_FILE")"
    "$VENV/bin/python" - <<'PY' 2>/dev/null || true
import json, urllib.request
with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=10) as r:
    for m in json.load(r)["data"]:
        print(f"  serving: {m['id']}  max_model_len={m.get('max_model_len')}")
PY
    return 0
  fi
  log "server NOT healthy on port $SERVER_PORT"
  return 1
}

cmd_stop() {
  if [[ ! -f "$PGID_FILE" ]]; then
    log "no $PGID_FILE — nothing this script started"
  else
    local pgid; pgid="$(cat "$PGID_FILE")"
    log "stopping process group $pgid…"
    # Negative PID = the whole group. This is the part that works when pkill
    # does not: children that renamed themselves are still in the group.
    kill -TERM "-$pgid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "-$pgid" 2>/dev/null || break
      sleep 2
    done
    if kill -0 "-$pgid" 2>/dev/null; then
      log "  still alive after 60s; SIGKILL"
      kill -KILL "-$pgid" 2>/dev/null || true
      sleep 3
    fi
    rm -f "$PGID_FILE"
  fi
  # The TPU is not released the instant the process dies.
  for _ in $(seq 1 30); do
    healthy || break
    sleep 2
  done
  if healthy; then log "ERROR: something is STILL serving on $SERVER_PORT"; return 1; fi
  log "stopped; port $SERVER_PORT free"
}

cmd_start() {
  local model="${1:?usage: serve_remote.sh start <model>}"
  if healthy; then
    log "ERROR: a server is already healthy on port $SERVER_PORT. Run 'stop' first —"
    log "       the TPU is held exclusively and a second server cannot attach."
    return 1
  fi

  # Keep the previous model's warmup log; it is the artifact e00 parses and it
  # is not reproducible without paying for another boot.
  if [[ -f "$WARMUP_LOG" ]]; then
    local keep="/tmp/vllm_warmup_$(date -u '+%Y%m%dT%H%M%SZ').log"
    mv "$WARMUP_LOG" "$keep"
    log "previous warmup log kept at $keep"
  fi

  local -a serve=("$VENV/bin/vllm" serve "$model"
                  --tensor-parallel-size "$TP_SIZE"
                  --max-model-len "$MAX_MODEL_LEN"
                  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
                  --no-enable-prefix-caching     # controlled variable: MUST be off
                  --port "$SERVER_PORT")

  # EXTRA_SERVE_ARGS exists for ONE purpose: flipping a controlled variable on
  # purpose, as a control. e07 found that a ragged batch pays only for its
  # packed tokens, not its padding -- but every run had chunked prefill ON, and
  # chunked prefill is exactly what packs requests into a step. Whether the
  # finding is "TPU serving does not pay length padding" or merely "chunked
  # prefill removes it" is the difference between a general claim and a narrow
  # one, and it is what connects this work to LAPS and BucketServe.
  # Anything set here MUST be recorded in the run config, or the controlled-
  # variable audit is quietly lying.
  if [[ -n "${EXTRA_SERVE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local -a extra=(${EXTRA_SERVE_ARGS})
    serve+=("${extra[@]}")
    log "EXTRA_SERVE_ARGS: ${EXTRA_SERVE_ARGS}  <- a controlled variable is being changed"
  fi

  if [[ -n "$VLLM_TPU_BUCKET_PADDING_GAP" ]]; then
    export VLLM_TPU_BUCKET_PADDING_GAP
    log "VLLM_TPU_BUCKET_PADDING_GAP=$VLLM_TPU_BUCKET_PADDING_GAP (linear ladder)"
  else
    unset VLLM_TPU_BUCKET_PADDING_GAP
    log "VLLM_TPU_BUCKET_PADDING_GAP unset (default exponential ladder)"
  fi

  log "starting: $model  (TP=$TP_SIZE, max_model_len=$MAX_MODEL_LEN)"
  setsid nohup "${serve[@]}" > "$WARMUP_LOG" 2>&1 < /dev/null &
  local child=$!
  sleep 2
  # setsid makes the child a session leader, so its PGID is its own PID.
  local pgid; pgid="$(ps -o pgid= -p "$child" 2>/dev/null | tr -d ' ')"
  echo "${pgid:-$child}" > "$PGID_FILE"
  log "  pgid $(cat "$PGID_FILE") -> $PGID_FILE ; log -> $WARMUP_LOG"
  log "  XLA warmup is slow: 5-30 min for the first bucket, 30-120 s per additional one."

  for i in $(seq 1 240); do
    if healthy; then
      log "server up after ~$((i * 15))s"
      cmd_status
      return 0
    fi
    # Fail fast instead of burning an hour: if the process group is gone the
    # server has crashed and no amount of waiting will help.
    if ! kill -0 "-$(cat "$PGID_FILE")" 2>/dev/null; then
      log "ERROR: server process group died during startup. Tail of $WARMUP_LOG:"
      tail -40 "$WARMUP_LOG"
      return 1
    fi
    sleep 15
  done
  log "ERROR: not healthy within 60 minutes. Tail of $WARMUP_LOG:"
  tail -40 "$WARMUP_LOG"
  return 1
}

case "${1:-}" in
  start)  shift; cmd_start "$@" ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  *) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
