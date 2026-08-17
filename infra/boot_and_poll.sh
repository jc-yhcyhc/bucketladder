#!/usr/bin/env bash
# Start a vLLM server on the TPU VM and wait for it, without holding an SSH session.
#
# Why this exists: a previous session lost three boot attempts to `ssh exited with
# return code [255]`. The server was being started inside a foreground SSH command
# that stayed open for the whole 30-75 minute warmup, so any transient network
# fault killed the boot along with the session. Here the start returns in seconds
# (setsid + nohup on the remote side) and progress is read by separate short
# calls, each of which is individually retryable and none of which owns the run.
#
#   ZONE=us-west4-a bash infra/boot_and_poll.sh <model> <tp> [max_wait_sec]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${1:?usage: boot_and_poll.sh <model> <tp> [max_wait_sec]}"
TP="${2:?}"
MAX_WAIT="${3:-4500}"
ZONE="${ZONE:-$(cat /tmp/bucketladder_zone 2>/dev/null)}"
NAME="${TPU_NAME:-bucketladder-tpu}"
: "${ZONE:?ZONE not set and /tmp/bucketladder_zone absent}"

ssh_try() {  # one short command, up to 3 attempts; never holds the run
  local cmd="$1" out
  for _ in 1 2 3; do
    if out=$(timeout 120 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" \
               --command="$cmd" 2>/dev/null); then
      printf '%s' "$out"; return 0
    fi
    sleep 20
  done
  return 1
}

echo "[boot] $MODEL TP=$TP in $ZONE (max ${MAX_WAIT}s)"

# Clear anything from a previous arm. A half-dead process holds the TPU lockfile
# and the next boot fails for a reason that has nothing to do with the model.
ssh_try 'cd ~/bucketladder 2>/dev/null; bash infra/serve_remote.sh stop >/dev/null 2>&1
pkill -9 -f "vllm serve" 2>/dev/null; sleep 10
rm -f /tmp/vllm_warmup.log /tmp/libtpu_lockfile 2>/dev/null; echo cleared' >/dev/null \
  || { echo "[boot] FAIL: could not reach the VM to clean up"; exit 2; }

ssh_try "cd ~/bucketladder
setsid env TP_SIZE=$TP MAX_MODEL_LEN=4096 nohup bash infra/serve_remote.sh start '$MODEL' \
  > /tmp/boot_${TP}.log 2>&1 < /dev/null &
sleep 5; echo started" >/dev/null \
  || { echo "[boot] FAIL: could not launch"; exit 2; }
echo "[boot] launched detached; polling"

start=$(date +%s)
while true; do
  elapsed=$(( $(date +%s) - start ))
  (( elapsed > MAX_WAIT )) && { echo "[boot] TIMEOUT after ${elapsed}s"; exit 1; }

  status=$(ssh_try 'code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://localhost:8000/health 2>/dev/null)
echo "health=$code"
if ! pgrep -f "vllm serve" >/dev/null 2>&1; then echo "PROC_GONE"; fi
grep -aoE "(RuntimeError|IndivisibleError|ValueError|NotImplementedError|OutOfMemory)[^\"]{0,110}" \
  /tmp/vllm_warmup.log 2>/dev/null | tail -2') || { sleep 60; continue; }

  if grep -q "health=200" <<<"$status"; then
    echo "[boot] READY after ${elapsed}s"; exit 0
  fi
  if grep -q "PROC_GONE" <<<"$status"; then
    echo "[boot] DIED after ${elapsed}s:"
    grep -vE "health=|PROC_GONE" <<<"$status" | sed 's/^/    /'
    exit 3
  fi
  echo "  ${elapsed}s $(head -1 <<<"$status")"
  sleep 60
done
