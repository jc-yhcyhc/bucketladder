# What Compiled-Shape Padding Actually Costs in Production TPU Serving

Stack: vLLM 0.25.0 + `tpu-inference` 0.25.0, JAX 0.10.2, libtpu 0.0.42.1, on
`v5litepod-4` (4 chips, TP=4). Fourteen hardware sessions, **[redacted]**.

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
decode latency by 0.0%. **A published latency predictor's model form does not
earn its place**: LENS (a per-bucket linear latency model) reports 2.15% mean
absolute percentage error (MAPE) on NPUs; reproduced on TPU with a withheld
point it gives 5.23%, and a constant-only predictor with no length term matches
it at batch sizes 1–2 and *beats* it at 4. And **the cost of a compiled step is
not a property of the step** — roughly 85% of nominal padding is paid at batch
size 1–2, 10–25% at 4–8, and approximately none at 16.

Padding is abundant, and how abundant is a property of the workload rather than
of the stack: across four prompt-length distributions at one arrival rate the
padded share of executed tokens spans **27.3% to 51.0%**. Most of it is not paid.
Per-request *length* padding does not exist, and not because of chunked prefill —
disabling it changes nothing. Decode, which dominates production serving, is
well-behaved: per-step cost rises 2.4× while batch size rises 32×, with no
discontinuity, and a roofline analysis shows why — the step reads the entire
weight set regardless of batch size, so padding on the request dimension falls
inside a floor that batch size does not move.

We report four optimisations we designed, measured and rejected, and nine
invalid inferences we made and caught — most sharing one cause, and now blocked
by a mechanical check rather than by intent.

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
   source, confirmed on hardware, absent from Ragged Paged Attention (RPA), LENS
   and vendor documentation.
2. **LENS's model form does not transfer to TPU** (§4.2), and the length term it
   turns on never earns its place at any batch size we measured.
3. **Shape-quantization cost is batch-size-dependent** (§4.3), with a
   memory-bandwidth mechanism (§4.5) that explains why the request dimension is
   nearly free, and a sharding ablation (§4.7) showing the mechanism is not an
   artifact of the layout we measured it in.
4. **Four optimisations measured and rejected** (§5), with the measurement that
   killed each.
5. **A methodological rule with a mechanical guardrail** (§6), including two
   headline numbers of our own that it killed.

We do **not** claim an admission-control policy, a ladder redesign, or any
throughput improvement.

---

## 2. Method

**Controlled variables.** Prefix caching off and asserted; chunked prefill,
`max_model_len`, `max_num_batched_tokens`, tensor-parallel (TP) size, `XLA_FLAGS`
and `ATTN_BUCKETIZED_NUM_REQS` all recorded. Every run parses the server's own
engine-config line and **aborts** if any controlled variable disagrees with the
config it claims to be running. This has fired three times on real mistakes, once
on an experiment of ours that deliberately varied a control (§7).

**Units.** Server-side, from Prometheus histogram *deltas* taken around each
measurement block — never client wall-clock, which includes round-trip time,
HTTP, tokenizer and queueing.

**Scope of instruments.** A step-scoped property requires a step-scoped
instrument. Single-step execution is verified per dispatch from
`iteration_tokens_total`'s count delta, and dispatches that split are excluded
rather than averaged. §6 reports three inferences we made before adopting this
rule, and one we made after.

**Request arrival.** Where a measurement requires *n* requests to reach the
scheduler in one step, they are released from a thread barrier after every
connection is established, so arrival spread is microseconds rather than
milliseconds. §4.3 shows this changes what is measurable.

**Definitions.** Three quantities are used throughout and are defined here rather
than at first use.

- **Flatness** — the ratio of measured cost growth to growth proportional to real
  tokens. 1.0 means cost rises exactly with real work; below 1.0 is sublinear.
- **Share of nominal padding paid** — `(measured − real) / (padded − real)`,
  where `real` is the cost ratio predicted by real tokens alone and `padded` the
  ratio predicted if the full compiled shape were paid. 0 means padding is free;
  1 means it is fully paid, which is what the compiled-shape premise predicts.
- **Model rejection, quoted as a percentage** — the amount by which a candidate
  model's *prediction* exceeds the measurement, as a fraction of the prediction.

**Statistics.** Medians over repeats, with 95% confidence intervals from 10,000
bootstrap resamples. Intervals are reported wherever a claim turns on the size of
a difference, because §4.6 establishes that some cells are far noisier than
others.

**Models.** Qwen3-4B primary; SmolLM2-1.7B (head_dim 64, MHA) and TinyLlama-1.1B
(head_dim 64, GQA 8:1) for the architecture contrast.

**Traceability.** Every run writes `meta.json` before doing work, records config
hash, git SHA and dirty flag, appends to a manifest, and is never overwritten.
All 61 numerical claims in this paper are tied to `run_id`s and recomputed from
captured data by `scripts/paper_numbers.py`; `./reproduce_all.sh` regenerates
every number and figure from `captured/` and exits non-zero if any disagrees.

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

**Attention executes at 256 requests whatever the batch size.**

The load-bearing evidence is a paired experiment: enabling the flag compiles the
full ladder — verified in the warmup log — and changes decode by **0.0%**
(identical to 0.1 ms at n=8 and n=9), because RPA's padded request slots hold no
KV blocks. Because both arms run on one server under one workload, this
comparison is insensitive to the noise that affects the table below.

A second, weaker line of evidence points the same way. If decode padded 9
sequences up to 16, decode at n=9 would cost what n=16 costs
(`prompt_len=512, output_len=8`, 21 repeats):

| n | 8 | **9** (padded to 16) | 16 |
|---|---|---|---|
| decode phase | 53.3 ms | **51.4 ms** | 91.8 ms |

Expressed as where n=9 falls between n=8 and n=16 — 0% meaning unpadded, 100%
meaning padded to 16 — three independent runs give −5%, −3% and −3%, but with
95% intervals of [−43%, 19%], [−66%, 49%] and [−67%, 59%]. **They exclude the
100% the premise predicts and cannot pin the value further.** We report this as
corroboration of the paired experiment and the source reading, not as
independent confirmation.

Two sessions were spent searching for a promotion cost at the 8→16 edge that the
default configuration had already excluded.

### 4.2 LENS's model form does not earn its place

LENS predicts NPU inference latency to 2.15% MAPE using a per-bucket
`intercept + slope × length` fitted from two end-to-end measurements per bucket.
We reproduced its protocol on TPU across 5 buckets × 3 batch sizes, 7 repeats per
point, withholding a mid-bucket point from each fit: **MAPE 5.23%, worst 22.4%**,
near-perfect at n=1–2 and failing at n=4.

A cross-hardware MAPE shift is weak evidence on its own, so we ran two ablations.

**The length term never helps.** Replacing the model with a constant — the mean
of the two calibration points, no length term at all:

| batch size | LENS | constant-only |
|---|---|---|
| 1 | 0.38% | 0.96% |
| 2 | 0.39% | 0.86% |
| 4 | 19.77% | **14.80%** |

At n=1–2 the within-bucket curve is nearly flat (flatness 0.97), so any two-point
fit is near-perfect and a constant is within 0.6 percentage points. The
near-perfect accuracy there is therefore **not evidence that the model form
transfers**. At n=4 the length term is *actively harmful*. It earns its place at
no batch size we measured.

**The failure is not an artifact of which points were fitted.** LENS specifies
two measurements per bucket but not which two. Over all three choices per cell,
the n=4 error swings by up to 44.8 percentage points — but its *minimum* over
every choice is still 16.97%, far above LENS's 2.15%. The magnitude must be
reported as a range; the localisation survives.

**[Figure 1 — `figures/fig1_lens.png`]** *Held-out prediction error against batch
size, with LENS's reported 2.15% as a reference line and the failure region
shaded. The claim is not that the predictor is inaccurate; it is that the error
is localised.*

We report this as validation rather than criticism: no prior work had tested LENS
on this hardware, and the regime where it breaks is the one serving uses.

### 4.3 The cost of a step is not a property of the step

Measured as the share of nominal padding paid, straddling a compiled boundary at
fixed batch size and near-fixed sequence length:

| batch size | share of padding paid |
|---|---|
| 1–2 | **~85%** |
| 4 | 24% |
| 8 | 16% |
| 16 | **≈0%** (−15.4%, −2.7%, +0.5% at three boundaries) |

At n=1 a single request pays its full sequence bucket (flatness 0.97 at buckets
≤1024). At n=4 it pays a fraction.

**How far up this can be measured was, in part, our own limitation.** Earlier
drafts reported the quantity as unmeasurable above n=8 because the scheduler
split every dispatch. Splitting turned out to track request count, not token
count — a dispatch of 8192 tokens at `max_num_batched_tokens=8192` never splits,
while one of ~1024 tokens at n=8 splits half the time — which is the signature of
an arrival race, not a capacity limit. Releasing all requests from a thread
barrier after connection setup cut arrival spread 7.6× at n=32 (15.4 ms → 1.7 ms)
and changed what is reachable:

| n | split under the old launcher | split under a synchronised launch |
|---|---|---|
| 4 | 0% | 0% |
| 8 | 20% | **0%** |
| 16 | 100% | **60%** |
| 32 | 100% | 100% |

**The real barrier sits between 16 and 32, not at 8.** Under a synchronised
launch, n=16 becomes measurable, and the paid share there is indistinguishable
from zero across three boundaries that each double the padded token count. The
trend is monotone: padding stops being paid as batch size rises, and by n=16 it
is free. The clean sample is small — 7 to 11 dispatches per arm after excluding
splits — so we report the sign and the trend rather than a precise value.

**We do not claim a shape for this dependence.** A within-bucket slope sweep gave
1.61 / 0.75 / 17.18 µs/token at n=1/2/4, but the third value rests on a single
measurement (points 9.78 / 13.13 / 13.15 at bucket 512), and all three sequence
lengths there pad to the same sequence *and* token bucket, so no padding model
predicts a difference. Three independent observations converge on n=4 — this
slope, the paid-padding fraction, and LENS's failure — and **we could not identify
what changes there.**

### 4.4 Padding is abundant, workload-dependent, and mostly free

How much padding a stack executes is a property of the workload, not of the
stack. Across four prompt-length distributions at one Poisson arrival rate
(8 req/s, `output_len=64`, 120 requests each):

| length distribution | CV | padded share of executed tokens | TTFT p50 / p95 | ITL p50 / p95 |
|---|---|---|---|---|
| fixed-256 | 0.00 | **51.0%** | 19 / 27 ms | 4.4 / 5.1 ms |
| lognormal | 1.20 | 38.4% | 19 / 100 ms | 4.4 / 7.5 ms |
| bimodal | 1.30 | 32.7% | 17 / 108 ms | 4.8 / 7.5 ms |
| uniform | 0.60 | **27.3%** | 89 / 261 ms | 6.4 / 14.9 ms |

The spread is **27.3% to 51.0%**, and the ordering is not intuitive: the *most*
uniform workload pads most, because a fixed length that sits just above a
boundary pads every single step by the same large amount, while a spread
distribution lands across buckets and averages out. Any single figure here
characterises a workload. We previously reported 35.9% as a stack property; it
was not one.

**[Figure 2 — `figures/fig2_padding.png`]** *Share of nominal padding actually
paid at each compiled boundary, against the 100% the compiled-shape premise
predicts.*

The share paid rises with the boundary: 10.0% at 512→1024, 22.1% at 1024→2048,
24.0% at 2048→4096, 24.8% at 4096→8192, all at n=4.

**We do not report a single recoverable-headroom figure.** An earlier draft
multiplied a padded share measured under one workload by a paid share measured at
one batch size to obtain "~4–9% of execution." Our own guardrail (§6) rejects
that derivation, because §4.3's finding is precisely that the paid share moves
with batch size. The two quantities are reported separately and not multiplied.

Per-request *length* padding does not exist. Holding batch size and total tokens
fixed and varying only the spread of request lengths, the batch-padding model is
rejected by **44–618%**; cost tracks packed tokens. Uniform controls, where all
candidate models agree, match to 1.9%. **This is not chunked prefill**: with
`--no-enable-chunked-prefill` the result is unchanged (packed wins 8/10 ragged
cells, batch padding rejected by 75–579%).

### 4.5 Decode is well-behaved, and the reason is bandwidth

`prompt_len=256`, `output_len=64`:

| n | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ms/step | 3.80 | 4.25 | 4.30 | 4.98 | 6.52 | 9.13 |
| µs/step/sequence | 3802 | 2127 | 1075 | 622 | 407 | **285** |

**[Figure 3 — `figures/fig3_decode.png`]** *Decode cost per sequence against
batch size, log–log. Smooth and monotone across the full range prefill could not
reach.*

Per-step cost rises **2.4×** while batch size rises **32×**; per-sequence cost
falls **13×** monotonically with no discontinuity.

A roofline built from these step times and the model's published dimensions gives
the mechanism. Per chip, per decode step:

| n | achieved HBM BW | BW utilisation | MFU | bound by |
|---|---|---|---|---|
| 1 | 532 GB/s | 64.9% | 0.27% | memory |
| 8 | 421 GB/s | 51.4% | 1.68% | memory |
| 32 | 258 GB/s | 31.4% | 3.65% | memory |

**2.01 GB of weights crosses HBM every decode step regardless of batch size** —
99% of all bytes moved at n≤2 and 89% at n≥16. Model-FLOPs utilisation never
exceeds 3.65%; every cell is memory-bound. The step pays a weight-load floor that
batch size does not move, so additional sequences — real or padded — are nearly
free until that floor is left. This is why padding on the request dimension costs
nothing, and it is the same memory-bound regime Pope et al. characterise
analytically for TPU inference; our contribution here is the measurement landing
in it and the consequence for compiled-shape ladders, not the regime itself.

**The pathology is real and lives in the phase that matters least.**

### 4.6 Per-dispatch variance is a prefill phenomenon

Spread over 9 repeats, same server: decode **1.00–1.04×** at most batch sizes;
prefill **1.00–1.03×** at n≤4 and **1.18–1.26×** at n≥8. Variance appears exactly
where the scheduler begins splitting dispatches, and decode — which has no
chunking decision — never shows it. Localisation, not mechanism: step *count*
does not correlate with cost within a cell, so the variance is in *how* a
dispatch splits.

The 1.00–1.04× figure does not describe every cell. Bootstrapping the decode
cells §4.1 depends on gives 95% interval widths of 38.7% at n=8 and 28.2% at
n=9 over 21 repeats — far wider than the aggregate spread suggests, and wider
than several differences a reader might otherwise treat as signal.

### 4.7 The cheapness of padding is not an artifact of the sharding

Holding model, chips and workload fixed and varying only tensor-parallel degree.
The prediction was registered in the configs before the measurement: per-chip
weight bytes scale as 1/TP, so the level should scale with 1/TP and the shape
should be preserved.

| TP | per-step level vs TP=4 | predicted | cost rise, n=1→32 |
|---|---|---|---|
| 4 | 1.00× | 1.00× | 2.33× |
| 2 | 1.63× | 2.00× | 2.41× |
| 1 | 2.86× | **4.00×** | **1.83×** |

**Both halves of the prediction missed.** The level scales *sub*-proportionally,
which the roofline cannot explain because it does not model the inter-chip
collectives the higher-TP arms pay. And the shape is not preserved: the curve
gets **flatter** with less sharding, which is what a larger per-chip weight floor
implies — more of the step is floor, so batch size moves it less.

Both misses point the same way, and together they answer the objection. If
request-dimension padding were cheap only because that dimension is not the
bottleneck at TP=4, reducing TP would expose it. Instead padding is cheapest at
TP=1, where the floor is largest. The result is about the weight floor, not about
this layout — for this model on this chip.

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
shape effect. We report it as a re-measurement rather than a contribution. It is
a single load point with no saturation curve.

Bucket-aligned packing deserves its own note. The second implementation measured
**−29% TPU time and −49% p99**; the correctness gate then showed 4 of 48 greedy
completions differed, every one at a prompt length just above a bucket boundary.
The patch was silently dropping prompt tokens. Trimming `num_scheduled_tokens`
after the scheduling loop leaves the request's bookkeeping untouched, so deferred
tokens are skipped rather than rescheduled — the step is cheaper because it does
less work.

---

## 6. Nine failures, one dominant cause

| looked like | was |
|---|---|
| cost model failing its holdout at 105.7% | fitted at `output_len=8`, run at 1 |
| "the premise is wrong, padding is free" | two experiments both right, different n |
| curve extrapolating small steps 15× low | reading the n=1 staircase as a floor |
| decomposition winning in model, losing measured | same |
| a scheduler patch sitting inert | 512-token prompts in the losing regime |
| a "fixed cost" that was not constant | all of the above, named |
| a headline "~4–9% recoverable" | padded share and paid share from different runs |
| "no single-step dispatch above n=8" | a step-count test that could never pass |
| paid padding at n=16 | split dispatches pooled into the median, not excluded |

The dominant cause: **a quantity measured under one configuration, used under
another.**

The guardrail took three versions. *"No derivation may combine quantities measured
at different batch sizes"* would not have caught the `output_len` failure. A
whitelist of config keys missed the largest error, because batch size is not a
top-level field — it lives inside experiment-specific structures. The working
form diffs **every** config key across a claim's source runs, exempts only free
text, and requires each difference to be named explicitly. It flagged five claims
already believed correct, and it has since killed two of our own headline
numbers: a crossover point, and the recoverable-headroom figure in §4.4.

**Its coverage is the set of *registered* claims, not the set of claims made.**
The recoverable-headroom figure evaded it for three drafts by living in prose. It
was caught only when we registered it as a claim in order to check it.

The last two are a different class and are not covered at all. A step-count
criterion tested whether a whole dispatch ran in one scheduler step — never true,
since every request needs a decode step — instead of whether its *prefill* was
split; it was caught because it reported 0% single-step at a cell another
experiment had independently measured as never splitting. And the boundary
experiment *counted* split dispatches but pooled their cost into the median
anyway, contradicting the rule §2 states. That was harmless while splits were
zero at n≤8 and wrong the moment n=16 became reachable, where more than half of
the dispatches split. Recomputing with splits excluded moved the n=16 result by
under one percentage point, so the conclusion stands — but the bias runs upward,
which is the direction that would have manufactured a positive result.

**The guardrail checks claim provenance, not analysis definitions**, and we do
not have a mechanical check for the latter. Both of these were caught by a
measurement disagreeing with an independent one, which is not a method.

---

## 7. Limitations

**One accelerator, one primary model.** v5e with a 4B model places per-chip
weights in the low hundreds of megabytes — an operating point where decode is
bandwidth-bound. The sharding objection, at least, we can answer: a TP=1/2/4
ablation (§4.7) finds the weight floor dominates at every sharding and dominates
*more* with less of it. The remaining exposure is model scale and multi-host
topology, where weight streaming and inter-chip collectives change what padding
hides under, and we measured neither.

**No production trace.** §4.4's four length distributions are parametric families
chosen to span a plausible range, not a trace. The result is a sensitivity range,
not a corrected point estimate.

**Prefill step cost above n=16 is still not isolable.** The barrier moved from
n=8 to somewhere between 16 and 32 (§4.3); it did not disappear. At n=16 itself
the clean sample is 7–11 dispatches per arm.

**The n=4 convergence is unexplained.** Three independent observations break
there; we searched the stack and did not find what changes. Resolving it needs an
operator-level profile, which we did not run.

**Co-located prefill and decode only.** On a disaggregated deployment the padding
question splits into two independent questions, and §4.6's finding that variance
is a prefill phenomenon is precisely the asymmetry disaggregation exploits.

---

## 8. Related work

**Pope et al.** derive the memory-bound/compute-bound transition for transformer
inference on TPU analytically. §4.5's roofline is a measured instance of that
regime, not a discovery; what we add is the consequence for compiled-shape
ladders — that the weight-load floor is what makes request-dimension padding
free.

**RPA** is the technique this work validates: our finding that per-request
padding costs nothing is what its ragged-tiling design predicts. It does not
discuss the request-count dimension, does not report cost against batch size, and
does not quantify how much padding survives it — the gap §4.4 fills.

**LENS** (§4.2) supplies the model form; we supply the hardware it was not tested
on, the batch sizes it does not survive, and the ablation showing its length term
is not what carries its reported accuracy.

**PagedAttention/vLLM** is the stack measured throughout. **Orca**'s
iteration-level scheduling is the mechanism producing the per-step batches this
paper measures. **Sarathi-Serve** introduced chunked prefill, which §4.4
explicitly controls for and disables.

**DistServe** and **Splitwise** disaggregate prefill from decode, which is the
architectural response to §4.6's finding — and which bounds our advice to
co-located deployments.

**BucketServe** and **LAPS** manage length-bucketing overhead on GPU. We do not
refute them; we bound their transferability. On this stack the padding they
target is largely not paid, and the batch dimension they would bucket over is
pinned to a single shape by default.

**Vidur** established simulator-fidelity validation as the standard for this kind
of work; our holdout discipline follows it.

---

## 9. Conclusion

A production TPU serving stack quantizes shapes in three dimensions. One does not
exist, one is disabled by a default flag, and the third is paid in a proportion
that depends on batch size — heavily at 1–2, lightly at 4–8, and not at all by
16. How much padding is executed is a property of the workload, spanning 27.3%
to 51.0% of executed tokens across plausible length distributions.

The practical advice is negative, and it is bounded by what we measured: on this
stack, at batch sizes up to 16, with a 4B model at any sharding of four v5e
chips, do not build
length bucketing, shape-aware admission control, or ladder design. The phase that
dominates production serving is smooth, close to linear, and memory-bound for a
reason that has nothing to do with compiled shapes.

We arrived here by trying to build the opposite paper. The control experiment
that refuted it cost $3 and should have run first.
