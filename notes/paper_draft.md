# What Compiled-Shape Padding Actually Costs in Production TPU Serving

**Draft 2 — rewritten 2026-08-11 from `notes/research_summary.md`.** Supersedes
draft 1 entirely; that version carried claims since withdrawn.

Target: **MLSys 2027 Industrial Track** — *"No requirement for novelty or new
methods,"* and the track invites *"submissions that challenge or reinforce
existing solutions, provide deeper insights into known problems, or rigorously
validate published techniques in a real-world setting."* 10 pages excluding
references. MLSys 2026's industrial deadline was 30 Oct 2025, so 2027's is
expected **late Oct 2026**; that call is not yet posted.

Stack: vLLM 0.25.0 + `tpu-inference` 0.25.0, JAX 0.10.2, libtpu 0.0.42.1, on
`v5litepod-4` (4 chips, TP=4). Twelve hardware sessions, **[redacted]**.

---

## Abstract

TPU executables are compiled for fixed tensor shapes, so a serving stack rounds
every workload up to one of a precompiled ladder. The inference that rounding up
means paying for what you rounded up to motivates a family of proposed
optimisations — length bucketing, shape-aware admission control, ladder design.
We measure what a production stack actually pays.

Three results. **The request ladder the system reports is not the one attention
executes at**: a default-off environment flag pins the attention kernel to a
single 256-request shape, and enabling the six-entry ladder it advertises changes
decode latency by 0.0%. **A published latency predictor does not transfer**: LENS
reports 2.15% mean error on NPUs; reproduced on TPU with a withheld point it gives
5.23%, worst 22.4%, near-perfect at batch sizes 1–2 and failing at 4. And **the
cost of a compiled step is not a property of the step** — roughly 85% of nominal
padding is paid at batch size 1–2 against 10–25% at 4–8, and above 8 the quantity
cannot be isolated at all, because the scheduler splits every dispatch.

36% of executed tokens are padding and most of it is free: the recoverable share
is ~4–9% of execution. Per-request length padding does not exist, and not because
of chunked prefill — disabling it changes nothing. Decode, which dominates
production serving, is well-behaved: per-step cost rises 2.4× while batch size
rises 32×, with no discontinuity.

We report four optimisations we designed, measured and rejected, and six invalid
inferences we made and caught — all sharing one cause, and now blocked by a
mechanical check rather than by intent.

---

## 1. Introduction

A GPU serving stack resolves kernel shapes at runtime. A TPU stack cannot: XLA
compiles for fixed shapes and recompiling per request is impossible at serving
latencies, so vLLM's TPU backend precompiles a ladder and rounds every step up.

Quantizations usually cost something, and the literature assumes this one does.
**BucketServe** derives an optimal length-bucket boundary then declines to compute
it as *"computationally expensive to calculate in practice."* **LAPS** captures a
CUDA Graph per `(length, batch)` cell and notes *"the number of graphs must be
limited."* Both are GPU work; both take for granted that the padding they manage
is paid.

We set out to build the TPU equivalent — an admission policy deciding whether to
promote a request into a larger warm bucket or queue it. Six sessions in, a
control experiment that should have run first rejected the premise. This paper
reports what is true instead.

**Contributions.**

1. **The attention request ladder is not the printed one** (§4.1). Readable from
   source, confirmed on hardware, absent from RPA, LENS and vendor documentation.
2. **LENS does not transfer to TPU at serving batch sizes** (§4.2). Validation of
   a published technique on hardware its authors did not run.
3. **Shape-quantization cost is batch-size-dependent** (§4.3), which explains six
   distinct-looking failures with one mechanism.
4. **Four optimisations measured and rejected** (§5), with the measurement that
   killed each.
5. **A methodological rule with a mechanical guardrail** (§6).

We do **not** claim an admission-control policy, a ladder redesign, or any
throughput improvement.

---

## 2. Method

**Controlled variables.** Prefix caching off and asserted; chunked prefill,
`max_model_len`, `max_num_batched_tokens`, TP size, `XLA_FLAGS` and
`ATTN_BUCKETIZED_NUM_REQS` all recorded. Every run parses the server's own
engine-config line and **aborts** if any controlled variable disagrees with the
config it claims to be running. This has fired twice on real mistakes.

**Units.** Server-side, from Prometheus histogram *deltas* taken around each
measurement block — never client wall-clock, which includes RTT, HTTP, tokenizer
and queueing.

**Scope of instruments.** A step-scoped property requires a step-scoped
instrument. Single-step execution is verified per dispatch from
`iteration_tokens_total`'s count delta, and dispatches that split are excluded
rather than averaged. §6 reports two inferences we made before adopting this rule.

**Models.** Qwen3-4B primary; SmolLM2-1.7B (head_dim 64, MHA) and TinyLlama-1.1B
(head_dim 64, GQA 8:1) for the architecture contrast.

**Traceability.** Every run writes `meta.json` before doing work, records config
hash, git SHA and dirty flag, appends to a manifest, and is never overwritten. All
42 numerical claims in this paper are tied to `run_id`s and recomputed from
captured data by `scripts/paper_numbers.py`.

---

## 3. Three quantized dimensions

From `tpu_inference/runner/tpu_runner.py:2133` and `runner/utils.py`, per step:

| | quantizes | ladder |
|---|---|---|
| **D1** | prompt length → prefill shape | *does not exist* |
| **D2** | scheduled tokens / step | `[16, 32, …, 8192]` |
| **D3** | requests / step | `[8, …, 256]` non-attention; **`[256]` attention** |

---

## 4. Results

### 4.1 The attention ladder is not the one the system reports

`envs.ATTN_BUCKETIZED_NUM_REQS` defaults to `False`, and when off,
`get_attn_req_paddings` returns `[max_req_size]` — one bucket. Every boot prints
both ladders and they disagree:

```
Prepared request paddings:      [8, 16, 32, 64, 128, 256]
Prepared attn request paddings: [256]
```

**Attention executes at 256 requests whatever the batch size.** Hardware confirms:
if decode padded 9 sequences up to 16, decode at n=9 would cost what n=16 costs.

| n | 8 | **9** (padded to 16) | 16 |
|---|---|---|---|
| decode phase | 53.3 ms | **51.4 ms** | 91.8 ms |

Enabling the flag compiles the full ladder — verified in the warmup log — and
changes decode by **0.0%** (identical to 0.1 ms at n=8 and n=9), because RPA's
padded request slots hold no KV blocks. **The default is correct**, stated with a
number rather than inferred from a code comment.

Two sessions were spent searching for a promotion cost at the 8→16 edge that the
default configuration had already excluded.

### 4.2 LENS does not transfer at serving batch sizes

LENS predicts NPU inference latency to 2.15% using a per-bucket
`intercept + slope × length` fitted from two end-to-end measurements per bucket.
We reproduced its protocol on TPU across bucket and batch size, withholding a
mid-bucket point from each fit:

**MAPE 5.23%, worst 22.4%.** Near-perfect at n=1–2 (0.0–0.6%); failing at n=4
(17–24%) — which is where serving operates.

Its single-regime form does not survive the batch sizes production uses. We report
this as validation rather than criticism: the terms of §4.3 are already in LENS's
model; what it does not do is hold across batch size, and no prior work had tested
it on this hardware.

### 4.3 The cost of a step is not a property of the step

Measured as the share of nominal padding actually paid, straddling a compiled
boundary at fixed batch size and near-fixed sequence length:

| batch size | share of padding paid |
|---|---|
| 1–2 | **~85%** |
| 4 | 24% |
| 8 | 16% |
| ≥8 | *not isolable* |

At n=1 a single request pays its full sequence bucket (flatness 0.97 at buckets
≤1024). At n=4 it pays a fraction. **Above n=8 the quantity cannot be measured at
all**: zero of fourteen cells produced a single-step dispatch, because the
scheduler splits every one — and that is the regime production runs in.

**We do not claim a shape for this dependence.** A within-bucket slope sweep gave
1.61 / 0.75 / 17.18 µs/token at n=1/2/4, but the third value rests on a single
measurement (points 9.78 / 13.13 / 13.15 at bucket 512), and all three sequence
lengths there pad to the same sequence *and* token bucket, so no padding model
predicts a difference. Three independent observations converge on n=4 — this
slope, the paid-padding fraction, and LENS's failure — and **we could not identify
what changes there.**

### 4.4 Padding is abundant and mostly free

**35.9% of executed tokens are padding** (p95 per-dispatch ratio 99.6%). The share
paid rises with the boundary: 10.0% at 512→1024, 22.1% at 1024→2048, 24.0% at
2048→4096, 24.8% at 4096→8192. **Recoverable: ~4–9% of execution**, and only under
a constant-step-count counterfactual no mechanism we tested achieves.

Per-request *length* padding does not exist. Holding batch size and total tokens
fixed and varying only the spread of request lengths, the batch-padding model is
rejected by **44–618%**; cost tracks packed tokens. Uniform controls, where all
candidate models agree, match to 1.9%. **This is not chunked prefill**: with
`--no-enable-chunked-prefill` the result is unchanged (packed wins 8/10 ragged
cells, batch padding rejected by 75–579%).

### 4.5 Decode is well-behaved

| n | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ms/step | 3.80 | 4.25 | 4.30 | 4.98 | 6.52 | 9.13 |
| µs/step/sequence | 3802 | 2127 | 1075 | 622 | 407 | **285** |

Per-step cost rises **2.4×** while batch size rises **32×**; per-sequence cost
falls **13×** monotonically with no discontinuity — across exactly the range
prefill could not reach. **The pathology is real and lives in the phase that
matters least.**

### 4.6 Per-dispatch variance is a prefill phenomenon

Spread over 9 repeats, same server: decode **1.00–1.04×** at every batch size;
prefill **1.00–1.03×** at n≤4 and **1.18–1.26×** at n≥8. Variance appears exactly
where the scheduler begins splitting dispatches, and decode — which has no
chunking decision — never shows it. Localisation, not mechanism: step *count* does
not correlate with cost within a cell, so the variance is in *how* a dispatch
splits.

---

## 5. Four optimisations, measured and rejected

| | outcome |
|---|---|
| bucket-aware admission control | premise false (§4.4) |
| ladder redesign | D1 does not exist; D3 inert by default |
| last-chunk decomposition | **20.6% worse** measured (51.06 vs 42.33 ms) |
| bucket-aligned step packing | implemented twice: inert, then output-corrupting |

The one positive measurement — release timing saving 26% of TPU time against
stock at 25 req/s (p=0.001, six paired seeds) — is **dynamic batching**, not a
shape effect. We report it as a re-measurement rather than a contribution.

Bucket-aligned packing deserves its own note. The second implementation measured
**−29% TPU time and −49% p99**; the correctness gate then showed 4 of 48 greedy
completions differed, every one at a prompt length just above a bucket boundary.
The patch was silently dropping prompt tokens. Trimming `num_scheduled_tokens`
after the scheduling loop leaves the request's bookkeeping untouched, so deferred
tokens are skipped rather than rescheduled — the step is cheaper because it does
less work.

---

## 6. Six failures, one cause

| looked like | was |
|---|---|
| cost model failing its holdout at 105.7% | fitted at `output_len=8`, run at 1 |
| "the premise is wrong, padding is free" | two experiments both right, different n |
| curve extrapolating small steps 15× low | reading the n=1 staircase as a floor |
| decomposition winning in model, losing measured | same |
| a scheduler patch sitting inert | 512-token prompts in the losing regime |
| a "fixed cost" that was not constant | all of the above, named |

One cause: **a quantity measured under one configuration, used under another.**

The guardrail took three versions. *"No derivation may combine quantities measured
at different batch sizes"* would not have caught the `output_len` failure. A
whitelist of config keys missed the largest error, because batch size is not a
top-level field — it lives inside experiment-specific structures. The working
form diffs **every** config key across a claim's source runs, exempts only free
text, and requires each difference to be named explicitly. It flags the retired
derivation, and it flagged five claims already believed correct.

Two further inferences, both self-caught, both measuring a step-scoped property
with a coarser instrument: a request-scoped metric contaminated by prefill/decode
interleaving, and a dispatch-scoped curve summing 2–4 engine iterations and
smearing the staircase it was meant to detect.

---

## 7. Limitations

**Prefill step cost is unmeasurable above n=8**, which is the regime production
runs in. Resolving it requires profiler traces, not `/metrics` deltas.

**The n=4 convergence is unexplained.** Three independent observations break
there; we searched the stack and did not find what changes.

**The 36% figure is not recoverable headroom.** Only 10–25% of nominal padding is
paid, so the real figure is ~4–9%, under a counterfactual nothing achieves.

**One accelerator, one primary model, synthetic uniform workloads**, `output_len=1`
for most cost work. No production trace.

---

## 8. Related work

**RPA** is the technique this work validates: our finding that per-request padding
costs nothing is what its ragged-tiling design predicts. It does not discuss the
request-count dimension, does not report cost against batch size, and does not
quantify how much padding survives it — which is the gap §4.4 fills.

**LENS** (§4.2) supplies the model form; we supply the hardware it was not tested
on and the batch sizes it does not survive.

**BucketServe** and **LAPS** manage length-bucketing overhead on GPU. We do not
refute them; we bound their transferability. On this stack the padding they target
is not paid, and the batch dimension they would bucket over is pinned to a single
shape by default.

**Vidur** established simulator-fidelity validation as the standard for this kind
of work; our holdout discipline follows it.

---

## 9. Conclusion

A production TPU serving stack quantizes shapes in three dimensions. One does not
exist, one is disabled by a default flag, and the third is paid in proportion that
depends on batch size — heavily at 1–2, lightly at 4–8, unmeasurably above that.
36% of executed tokens are padding and most of it is free.

The practical advice is negative and worth stating: do not build length bucketing,
shape-aware admission control, or ladder design for this stack. The phase that
dominates production serving is smooth and close to linear.

We arrived here by trying to build the opposite paper. The control experiment that
refuted it cost $3 and should have run first.
