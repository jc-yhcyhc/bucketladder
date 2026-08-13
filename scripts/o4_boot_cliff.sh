#!/usr/bin/env bash
# =============================================================================
# o4_boot_cliff.sh — how much memory headroom does a longer ladder actually need?
# =============================================================================
#
# §4.9 reported that the twenty-one-shape token ladder costs 15.0% of KV cache
# capacity. That was wrong, and wrong in a specific way worth naming: it
# differenced capacity measured at gpu_memory_utilization 0.92 against capacity
# measured at 0.80 and attributed the gap to shape count. At equal fraction both
# ladders report the same 312,320 tokens. The fraction was the variable.
#
# What is real is a boot cliff: at the 0.92 default the long ladder dies in
# warmup with RESOURCE_EXHAUSTED, asking for 32.50 MB against 12.40 MB free. That
# is a 20 MB miss, which is nothing like 15% of a 16 GB chip — so the cliff should
# be narrow, and the remedy should be a small backoff rather than the coarse one
# we happened to pick first.
#
# This measures where the cliff is. Descending fractions until the long ladder
# boots, then the short ladder at that same fraction, so the comparison is finally
# at one value of the confounding variable instead of two.
#
# What each outcome means:
#   boots at 0.90-0.91  -> the remedy costs ~1-2% of capacity, not 15%, and the
#                          honest headline is "a boot cliff at the default,
#                          recoverable for about a percent"
#   needs <= 0.85       -> the headroom requirement is real and substantial, and
#                          the cost is worth reporting as capacity after all
#   short ladder differs at the same fraction -> shape count DOES move capacity
#                          and the identical-312,320 reading was a coincidence of
#                          block rounding
#
# Usage (on the TPU VM):
#   bash scripts/o4_boot_cliff.sh
# =============================================================================

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

MODEL="${MODEL:-Qwen/Qwen3-4B}"
FRACTIONS="${FRACTIONS:-0.92 0.91 0.90 0.88 0.85}"
OUT="${OUT:-/tmp/o4_boot_cliff.tsv}"
: > "$OUT"

boot_once() {           # $1 = gap ("" for default ladder), $2 = fraction
  local gap="$1" frac="$2" code="" i kv shapes
  bash infra/serve_remote.sh stop >/dev/null 2>&1; sleep 8
  if [[ -n "$gap" ]]; then export VLLM_TPU_BUCKET_PADDING_GAP="$gap"
  else unset VLLM_TPU_BUCKET_PADDING_GAP; fi
  export EXTRA_SERVE_ARGS="--gpu-memory-utilization $frac"
  (TP_SIZE=4 nohup bash infra/serve_remote.sh start "$MODEL" \
      > "/tmp/o4_${gap:-def}_$frac.log" 2>&1 &)
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
    [[ "$code" == "200" ]] && break
    grep -q "RESOURCE_EXHAUSTED" /tmp/vllm_warmup.log 2>/dev/null && { code="OOM"; break; }
    sleep 15
  done
  kv=$(grep -oE "GPU KV cache size: [0-9,]+" /tmp/vllm_warmup.log 2>/dev/null | tail -1 | grep -oE "[0-9,]+$")
  shapes=$(grep -oE "Prepared token paddings: \[[^]]*\]" /tmp/vllm_warmup.log 2>/dev/null | tail -1 | tr -cd ',' | wc -c)
  [[ -n "$shapes" ]] && shapes=$((shapes + 1))
  printf '%s\t%s\t%s\t%s\t%s\n' "${gap:-default}" "$frac" "${code:-none}" "${kv:-NA}" "${shapes:-NA}" \
      | tee -a "$OUT"
}

echo -e "ladder\tfraction\tboot\tkv_tokens\tn_shapes" | tee -a "$OUT"

# 1. Descend until the LONG ladder boots.
CLIFF=""
for f in $FRACTIONS; do
  line=$(boot_once 512 "$f")
  if [[ "$(echo "$line" | cut -f3)" == "200" ]]; then CLIFF="$f"; break; fi
done

# 2. Same-fraction control: the SHORT ladder at the fraction the long one needed,
#    and at the default. Without this the capacity comparison repeats the very
#    provenance error this experiment exists to correct.
if [[ -n "$CLIFF" ]]; then
  boot_once "" "$CLIFF" >/dev/null
fi
boot_once "" 0.92 >/dev/null

echo
echo "=== o4 boot cliff ==="
column -t "$OUT" 2>/dev/null || cat "$OUT"
if [[ -n "$CLIFF" ]]; then
  echo "long ladder (21 shapes) boots at fraction $CLIFF"
else
  echo "long ladder did not boot at any fraction in: $FRACTIONS"
fi
