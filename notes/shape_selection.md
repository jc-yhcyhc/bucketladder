# What shape does a step actually execute at?

Read from `tpu_inference` 0.25.0's own source (`pip download`, no TPU needed),
2026-08-10, after six hardware sessions of inferring it from timings.

## The three padded dimensions, and only one of them varies

`runner/tpu_runner.py:2133` is the decision site. Per step:

| dimension | ladder | source |
|---|---|---|
| scheduled **tokens** | `[16, 32, …, 8192]` | `get_token_paddings` |
| **requests** (non-attention) | `[8, 16, 32, 64, 128, 256]` | `get_req_paddings` |
| **requests (attention)** | **`[256]` — a single bucket** | `get_attn_req_paddings` |

The third row is the one that matters and it was invisible from timings.
`envs.ATTN_BUCKETIZED_NUM_REQS` defaults to **`False`**, and when it is off:

```python
def get_attn_req_paddings(min_req_size, max_req_size):
    if not envs.ATTN_BUCKETIZED_NUM_REQS:
        reqs = [max_req_size]        # ONE bucket, at the maximum
```

Our own warmup logs said so all along and we read past it:

```
Prepared request paddings: [8, 16, 32, 64, 128, 256]
Prepared attn request paddings: [256]          <- one entry
```

**Attention always executes at 256 requests, whatever the batch size.** It is
the dominant cost, and its shape never changes, so there is no batch-ladder step
to find — in prefill or decode. Session 4's search for a promotion cost at the
8→16 edge, and session 6's finding that decode does not pay batch padding, were
both looking for something the configuration had already ruled out.

The request ladder `[8, 16, …]` still applies to non-attention work (sampling,
logits), which is small. That is consistent with the measured decode step:
n=9 costs 51.4 ms against n=16's 91.8 ms — scaling with *actual* sequences, not
with the padded 16.

## Why the token staircase "disappeared" in batches — it didn't

The dispatch-level cost curve (`sim/measured_cost_curve.json`) shows no
staircase: within token bucket 4096, costs run 41.2 / 72.0 / 73.9 / 69.1 ms. I
took that as evidence against step-level token padding. **That inference was
invalid.** A dispatch is not a step: e04 and e05 measured 2–4 engine iterations
per dispatch, so the dispatch cost is a *sum over steps* whose individual token
counts land wherever the scheduler put them. Summing over a staircase smears it.

This is the same error as inferring a step property from a request metric, one
level up. The dispatch curve is **uninformative** about per-step padding, not
contradictory to it.

## The step-level measurement we already have

**e01 is one.** A single request, one prefill step, sweeping length within one
token bucket: flatness **0.97** at buckets ≤1024. With n=1 the step's token
count *is* the request's length, so nothing is smeared.

So the token ladder **is** paid, per step. That was never in doubt; what was in
doubt was whether it survives batching, and the dispatch curve could not answer.

## What is actually established, and what is not

**Established:**
- The step's scheduled-token count is padded to `[16 … 8192]` and paid (e01, n=1).
- Per-request *length* padding does not exist: a mixed-length batch costs its
  packed tokens (e07), and not because of chunked prefill (e08).
- The attention request dimension is a single compiled bucket at 256, so batch
  size does not change the attention shape at all (source, above).

**Not established:**
- Whether a step with n > 1 pays the same token padding as a step with n = 1.
  Every ladder observation we have at n > 1 is dispatch-scoped and therefore
  smeared.

## The measurement that would close it

Force a **single-step** dispatch and sweep the step's token count within one
bucket:

- n requests, total tokens small enough that the scheduler cannot split them
  (well under `max_num_batched_tokens`);
- verify one step per dispatch from `iteration_tokens_total`'s count delta —
  the instrument already exists and is already used by e04/e05;
- compare, say, 4×96 = 384 tokens against 4×128 = 512 tokens. Both pad to the
  512 bucket. Equal cost means the padding is paid at n > 1; proportional cost
  means it is not.

That is a step-scoped test of a step-scoped property, which is what every
previous attempt at this question was missing.

## What it does not do

It sharpens the mechanism; it does not revive the admission-control paper. The
lever is the *step's* token count, and the scheduler sets that — it packs
whatever is runnable up to `max_num_batched_tokens`. A client can influence it
by release timing but cannot choose it. And the measured 26% saving is still
principally batching amortisation (25.7 µs/token at n=1 against 16.9 at n=8),
which is standard dynamic batching.
