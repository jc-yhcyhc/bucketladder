# What MoE changes about padding cost — read from tpu-inference 0.25.0 source

Established by reading the shipped wheel (`tpu_inference-0.25.0-py3-none-any.whl`,
pure Python), not by measurement. No hardware, no cost. Line numbers are that version's.

## 1. The ladder is architecture-independent — provably

`tpu_inference/runner/utils.py:180`

```python
def get_token_paddings(min_token_size: int, max_token_size: int,
                       padding_gap: int) -> list[int]:
```

Three integers. No model, no config, no architecture. The D2 token ladder a MoE
model gets is byte-identical to the one a dense model gets at the same
`max_num_batched_tokens` and `VLLM_TPU_BUCKET_PADDING_GAP`.

This also confirms from source the ladder rule we derived empirically — double
while `num <= padding_gap`, then step linearly by `padding_gap` — including the
`num //= 2` backstep that produces the off-by-one bucket count we measured.

**Consequence:** "does MoE change the ladder" is answered, and the answer is no.
Any MoE result reuses the dense ladder unchanged.

## 2. What MoE does change: a padded token activates k experts

`tpu_inference/layers/common/fused_moe_gmm.py:638`

```python
# Route padding tokens to expert 0 instead of picking a selected expert. This
# is especially useful when we have a low number of tokens (e.g. low
# concurrency), where padding tokens may activate unnecessary expert weights
# and slow down the gmm kernel.
if num_valid_tokens is not None:
    token_valid = (jnp.arange(num_tokens) < num_valid_tokens)[:, None]
    topk_indices = jnp.where(token_valid, topk_indices, 0)
    topk_weights = jnp.where(token_valid, topk_weights, 0.0)
```

Padding tokens carry whatever `top_k` returned for their garbage rows. Unless
suppressed, each padded token is dispatched to `top_k` real experts — 4 for
gpt-oss-20b, 8 for Qwen3-30B-A3B — and can *widen the set of active expert
groups*, which is what drives grouped-matmul cost.

Dense: a padded token costs one FFN row.
MoE, default: a padded token costs k expert dispatches and may activate weights
no real token in the batch needed.

**So the direction is that MoE padding is structurally worse than dense, not better.**

## 3. The mitigation exists and is OFF by default

`tpu_inference/envs.py:71` and `:427`

```python
MOE_ROUTE_PADDING_TO_EXPERT0: bool = False
env_bool("MOE_ROUTE_PADDING_TO_EXPERT0", default=False)
```

Also gated on `not is_dp` (`layers/vllm/interface/moe.py:138`), and it fails
open — if `query_start_loc` can't be read it logs a warning and silently serves
with padding tokens routed normally.

Two things follow. First, the stock configuration is the expensive one, so a
default-configuration measurement is the honest baseline. Second, the comment is
independent corroboration from the implementers of this paper's central claim:
padding costs real time, and it costs most **at low concurrency** — exactly the
regime where padding fraction is highest.

## 4. No fourth quantized dimension

Searched for a per-batch expert-capacity bucket — the thing that would add a
quantized dimension beyond D2/D3. Not found. The `padded_hidden_size` padding in
`fused_moe_gmm.py:734` is a static weight-layout property, fixed at load, not a
per-step shape the scheduler picks. Group sizes into the gmm are dynamic.

**D1/D2/D3 remains the complete list for MoE too.**

## What still needs hardware

The mechanism above is settled; the magnitude is not.

1. Per-padded-token cost for MoE vs the dense ~35 µs, at matched TP.
2. Whether `MOE_ROUTE_PADDING_TO_EXPERT0=1` recovers it, and how much — this is a
   measurable optimization with a real knob, in the same family as the O-series.
3. Whether the k-way dispatch makes the cost superlinear in padding fraction
   (widening the active-expert set) rather than linear as in dense.

Ready to run when capacity returns: `configs/m2_moe_gptoss.json` and
`configs/m2_dense_control.json`, matched at TP=2.
