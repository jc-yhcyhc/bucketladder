# What Compiled-Shape Padding Actually Costs in Production TPU Serving

Stack: vLLM 0.25.0 + `tpu-inference` 0.25.0, JAX 0.10.2, libtpu 0.0.42.1, on
`v5litepod-4` (4 chips, TP=4). Twenty hardware sessions, **[redacted]**.

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
size 1–2, a median of 23.1% at n=4 and 14.3% at n=8 with ranges that overlap,
and approximately none at 16.

Padding is abundant, and **how abundant is a property of the workload rather than
of the stack** — a single-figure characterisation of the stack is not available.
Most of it is not paid.
Per-request *length* padding does not exist, and not because of chunked prefill —
disabling it changes nothing. Decode, which dominates production serving, is
well-behaved: per-step cost rises 2.4× while batch size rises 32×, with no
discontinuity to n=32. Request-dimension padding is free for a data-structure
reason rather than a bandwidth one: with per-sequence KV held constant, absolute
attention time rises 9.72× as real requests rise 16×, and cutting the compiled
slot count from 256 to 8 changes it by **−0.9% at n=1**, where a per-slot padding
cost would have been ~42%. §4.5 also reports the memory-bandwidth account we held
earlier and the measurement that withdrew it.

We report four optimisations we designed, measured and rejected, and eleven
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
3. **Shape-quantization cost is batch-size-dependent** (§4.3), and our own
   memory-bandwidth explanation for it is withdrawn on the measurement that
   contradicts it (§4.5). We name the surviving hypothesis and state what would
   discriminate it, rather than claiming it is established.
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

- **Flatness** — how far a short request's cost sits from what proportional
  scaling would predict, as a fraction of the distance to the full-bucket cost:
  `(cost(L) − p) / (cost(B) − p)` where `p = cost(B) · L/B`. **1.0 is a pure
  staircase** — cost independent of true length, padding fully paid — and **0.0
  is pure linear**, cost proportional to real tokens, padding free. An earlier
  draft defined this with the polarity reversed, which contradicted both places
  it is used; the usages were correct and the definition was not.
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
All 73 numerical claims in this paper are tied to `run_id`s and recomputed from
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

An operator profile taken later closes this independently: the decode attention
kernel is emitted as `RPAd-p_256-bq_1_1-bkv_8192_8192`. **The compiler writes the
256-request padding into the kernel's own name**, whatever the batch size.

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

**These are PREFILL steps.** The distinction is load-bearing rather than
expository: decode stays memory-bound to a batch of roughly 240 (§4.5), so if
these were decode steps, 85% of padding being paid at n=1–2 would contradict the
mechanism. A prefill step carries hundreds to thousands of tokens, is past the
ridge, and both FLOPs and attention scale with padded tokens — which is why
padding is paid there and not in decode.

Measured as the share of nominal padding paid, straddling a compiled boundary at
fixed batch size and near-fixed sequence length:

| batch size | median | mean | 95% CI over boundaries | boundaries | clean dispatches/arm |
|---|---|---|---|---|---|
| 1–2 | **~85%** | — | *not computed* | 1 | see below |
| 4 | 23.1% | 20.2% | [13.5%, 24.4%] | 4 | 9–15 |
| 8 | 14.3% | 11.8% | [0.2%, 21.0%] | 3 | 3–5 |
| 16 | **−2.7%** | −5.9% | [−15.4%, +0.5%] | 3 | 7–11 |

The intervals are bootstrapped over *boundaries*, not repeats, because with three
or four boundaries per row that is the dominant source of spread — and they are
correspondingly weak. Tested pairwise, **only n=4 and n=16 separate.** n=8
separates from neither neighbour. The n≤2 row rests on a single boundary and we
do not quote an interval for it at all; it is the least well supported number in
the paper and also the largest, which is the wrong way round.

**We no longer describe this as monotone, and the intervals are stricter than
the earlier prose.** The defensible statement is ordinal: **substantially paid at
n≤2, intermediate at n=4–8, indistinguishable from zero at n=16, with only the
n=4/n=16 contrast surviving at interval level.** Anyone wanting a per-level
number needs more boundaries per row than we ran.
The n=8 row is also a correction: it was previously 16%, computed with split
dispatches pooled into the median. Recomputed with them excluded it is 14.3% —
the conclusion survives, the number moved.

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
ends of the range are separable even though the middle is not: padding is
substantially paid at n≤2 and indistinguishable from zero at n=16. The clean
sample is small — 7 to 11 dispatches per arm after excluding
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

The ordering is not intuitive: the **fixed-length** workload pads most, because a
length sitting just above a boundary pads every single step by the same large
amount, while a spread distribution lands across buckets and averages out.

**The 27.3–51.0% spread is confounded and we withdraw it as a range.** The four
arms were matched on request rate, not on offered *tokens*: mean prompt lengths
are roughly 256, 384, 704 and 2056, so the uniform arm carries about 8× the token
load of fixed-256. Its TTFT p50 of 89 ms against 17–19 ms for the others is the
tell — that is a different point on the load curve, not only a different shape.
The spread therefore mixes distribution shape with utilisation, which is the
error class §6 names as this project's dominant one, found in our own table. What
survives is the qualitative claim, which does not depend on the magnitudes:
**padded share is a property of the workload, not of the stack**, and the 35.9%
we previously reported as a stack property was not one. A matched re-run is
needed before any range is quoted.

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

### 4.5 Decode is well-behaved, and the reason is not bandwidth

`prompt_len=256`, `output_len=64`, measured across the full compiled request
ladder:

| n | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| ms/step | 4.02 | 4.91 | 9.19 | 15.32 | 27.90 | 51.83 |
| µs/step/sequence | 4020 | 614 | 287 | 239 | 218 | **203** |
| queue (ms) | 0.0 | 0.0 | 0.0 | 0.1 | 62.2 | 298.3 |

**[Figure 3 — `figures/fig3_decode.png`]** *Decode cost per sequence against
batch size, log–log.*

Two regimes, and the boundary is sharp. Below n≈8 the step barely moves: batch
rises 8× for 1.22× the cost, and per-sequence cost falls almost as fast as batch
rises. **Above n≈32 the step is nearly linear in batch** — 1.67×, 1.82×, 1.86×
for successive doublings — and per-sequence cost flattens at roughly 200 µs
rather than continuing to fall.

Queue time at n=128 and n=256 means those two columns are not clean wide batches;
requests are not all resident. Nothing below rests on them. The n=64 column has a
0.1 ms queue and already shows the second regime.

**What the mechanism is not.** Earlier drafts explained free request padding with
a memory-bandwidth argument: the step reads the whole weight set regardless of
batch, so padded slots ride inside a floor batch size cannot move. That story
predicts a specific thing — utilisation climbing toward the compute roof as batch
grows past the ridge near 240 — and we published a frontier table predicting
**MFU ≈49% at n=256**. Measured:

| n | 1 | 8 | 32 | 64 | 256 |
|---|---|---|---|---|---|
| MFU | 0.3% | 1.7% | 3.6% | 4.4% | **5.1%** |
| HBM BW utilisation | 61.4% | 52.1% | 31.2% | 21.4% | **11.1%** |

**The refutation rests on n≤64, where the queue is 0.1 ms and the columns are
clean.** A memory-bound step is one whose achieved bandwidth sits near the roof.
Ours falls monotonically — 61.4% → 52.1% → 31.2% → **21.4% by n=64** — which the
account cannot accommodate at any batch size, let alone at the ridge. The claim
that every cell is memory-bound is **withdrawn on that evidence alone**.

The n=256 column corroborates and is not required: MFU 5.1% against a predicted
49%, bandwidth 11.1%. Its caveat has since been half-resolved. Sampling
`vllm:num_requests_running` throughout the window shows the ladder **is** fully
reachable — under a synchronised launch, n=128 and n=256 both reach and hold
their full requested batch, so the 298 ms of queue time in the sweep above was
our arrival pattern again, the same artifact as the prefill splits in §4.3, not a
capacity limit.

That establishes reachability, not cleanliness: the decode numbers in the table
were themselves taken under the old launcher, with the queueing present. They
should be re-measured under a synchronised launch before anyone leans on them,
and we have not done that. **The frontier table stays withdrawn** — it predicted
49% MFU in a regime where the cleanest reading we have says 5.1%, and no
launcher change rescues an order of magnitude. The
roofline retains one honest use, byte accounting: 2.01 GB of weights crosses HBM
every decode step regardless of batch, which is 99% of bytes moved at n≤2 and
falls below half by n=64 as KV traffic overtakes it. That is arithmetic about
bytes, not an explanation of time — achieved bandwidth is `bytes / measured
time`, so it restates the step time it is computed from.

**One mechanism per dimension, stated once.** The paper measures two different
padding dimensions and they do not share an explanation, which earlier drafts
conflated:

| dimension | is padding paid? | mechanism |
|---|---|---|
| **D3, requests/step** | no, at every batch size measured | RPA does no work for padded request slots — a data-structure property |
| **D2, tokens/step** | yes in prefill, falling with batch | arithmetic intensity: padded tokens are real FLOPs once the step is past the ridge |

§4.3's paid-share table is the **token** dimension in prefill. §4.8's dtype
prediction applies to the **token** dimension only, because quantization moves the
ridge and the ridge is not in the request-dimension story at all. Where §4.7
concludes "the result is about the weight floor," it is describing the level of
decode step cost, not why request padding is free.

**What the request-dimension mechanism is, measured directly.**
`ATTN_BUCKETIZED_NUM_REQS` is off, so attention executes at 256 request slots
whatever the batch — and the operator profile says what that costs. With prompt
and output length fixed, so per-sequence KV is constant, absolute attention
device time is:

| n | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| attention (µs) | 16 830 | 26 011 | 45 919 | 85 103 | **163 535** |
| per real request | 16 830 | 13 005 | 11 480 | 10 638 | 10 221 |

**This does not discriminate the hypothesis, and an earlier version of this
section claimed it did.** The argument was: a kernel doing work for 256 padded
slots would be flat in n; this is not flat; therefore padded slots are skipped.
But *flat* was never the alternative. Padded slots hold no KV blocks (§4.1), so
no kernel could do KV-proportional work for them. The only cost padding can
plausibly carry is a **fixed per-slot overhead** — walking 256 block-table
entries, loading 256 metadata records, launching tiles across 256 slots
regardless of occupancy — and that is constant in n by construction.

Fitting the table to `T = a·n + b` through its endpoints:

| n | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| measured | 16 830 | 26 011 | 45 919 | 85 103 | 163 535 |
| 9 780·n + 7 050 | 16 830 | 26 610 | 46 171 | 85 292 | 163 535 |

Residuals are under 2.3%, and **the fixed term is 7 050 µs — 42% of attention
time at n=1**, falling to 4% at n=16. If that term were a 256-slot padding cost
it would be 27.5 µs per padded slot. The data is equally consistent with "RPA
skips padded slots" and with "RPA pays a fixed cost for all 256, large at n=1 and
negligible by n=16." **We cannot claim the first from this table**, and the
request-dimension mechanism is therefore *supported but not established*.

**We ran the discriminating experiment: `b` survives, so it is not padding.**
Enabling `ATTN_BUCKETIZED_NUM_REQS` compiles attention at 8 slots instead of 256
— a 32× reduction in padded slots — and the attention operator was profiled in
both arms at the same real batch:

| n | flag off (256 slots) | flag on (8-slot ladder) | change |
|---|---|---|---|
| 1 | 16 898 µs | 16 749 µs | **−0.9%** |
| 2 | 26 130 µs | 25 999 µs | −0.5% |
| 8 | 85 081 µs | 85 055 µs | −0.0% |
| fitted fixed term `b` | 7 158 µs | 6 991 µs | **−2%** |

A per-padded-slot cost would fall by roughly 97% when the slot count drops 32×.
It falls by 2%. **The fixed term is block-table and dispatch overhead, not
padding**, and the request-dimension mechanism is established: Ragged Paged
Attention does no work for padded request slots.

This also runs the high-power version of §4.1's test. That paired experiment used
n=8 and n=9, where a fixed 256-slot cost would have been ~8% of attention time;
at n=1 it would be ~42%, and at n=1 the measured difference is **−0.9%**. The
0.0% in §4.1 was not a low-power result hiding an effect — the effect is absent
where it would have been largest.

**An operator profile**, which unlike the roofline is a direct observation of
where time went. Share of TPU device time:

| n | attention | collective | matmul/fusion |
|---|---|---|---|
| 1 | 6.8% | 13.5% | **78.5%** |
| 4 | 15.4% | 13.9% | 69.6% |
| 16 | 34.2% | 13.4% | **51.4%** |

Matmuls dominate at low batch and give way to attention as KV grows; collectives
hold a flat ~13.4%, and their latency component is not modelled by any roofline.
Nothing moves discontinuously at n=4 — every category's share changes less into
n=4 than across some other adjacent pair — so **§4.3's convergence is not visible
at operator granularity.**

**A microbenchmark that measured the wrong thing — retracted.** An isolated
matmul at the model's real sharded shapes returned 142.9 µs at M=1, flat to
143.6 µs at M=256, which an earlier version called the weight-load floor with
confounds removed. The qkv projection holds 7.86 MB per chip; at peak bandwidth
that is 9.6 µs, so the measurement sat 15× above the floor at an implied 55 GB/s
— **7% of peak**. What was timed is per-dispatch overhead, and the per-row column
was `constant / M`. The design also could not discriminate: for that shape the
memory floor and compute at M=256 cross near M≈240, so bandwidth alone predicts a
flat curve across the whole sweep while tile padding predicts flat only to M≈8 —
tiling's prediction is a subset of bandwidth's everywhere the test looked.

Two further attempts did not settle it. An amortised version reported 1250% of
peak, because XLA hoisted the loop-invariant matmul; a third, chaining each
iteration onto the previous, is physically valid at 79% of peak bandwidth and
shows the curve flat from M=1 to M=16 with no knee at 4 or 8 — but it streams
weights from HBM every iteration, so it is bandwidth-bound and still does not
isolate tile structure. Settling it needs weights genuinely resident in VMEM,
which here means a Pallas kernel. **The tiling hypothesis is untested, not
rejected.**

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

**The two misses are one omitted term.** Fitting `T(TP) = W/TP + F` — a weight
load that shards, plus a fixed cost that does not — to the three levels gives

    T(TP) = 2.48/TP + 0.38     (normalised to the TP=4 step)

    TP        4      2      1
    measured  1.00   1.63   2.86
    fitted    1.00   1.62   2.86

with a total absolute error of 0.01 across three points and one free parameter.
The prediction failed because it omitted `F`, not because the weight-floor idea
was wrong: **fixed, non-sharding cost is 38% of the TP=4 step** — roughly 1.5 ms
at n=1 — and that single term explains both the sub-proportional level and the
flattening of the curve.

**`F` is not the collectives, and an earlier version of this section said it
was.** A constant term cannot represent inter-chip communication: collectives are
zero at TP=1 and largest at TP=4, so they scale *with* TP while `F` by
construction does not. That a TP-invariant constant fits to 0.01 is evidence the
term is dispatch and host overhead rather than communication — which connects
directly to the per-dispatch floor §4.5 retracts, and is the more interesting
reading. We have not partitioned `F` into device and host time; an operator
profile at TP=1 and TP=2 would do it, and if the ~13.4% collective share does not
fall toward zero at TP=1 then one of the two measurements is wrong. Note also
that with `T(4)=1` by normalisation this is one free parameter against two
informative points, so the fit quality is weaker evidence than "0.01 across three
points" suggests. What was reported as a failed prediction is better read
as a calibrated two-term cost model, and it independently bounds the per-step
fixed overhead that §4.5's retracted microbenchmark was groping for.

Both misses point the same way, and together they answer the objection. If
request-dimension padding were cheap only because that dimension is not the
bottleneck at TP=4, reducing TP would expose it. Instead padding is cheapest at
TP=1, where the floor is largest. Whatever makes request-dimension padding free
is not a property of this sharding — which is what the section set out to test.
It does not follow that the weight floor is the cause; §4.5 withdraws that, and
this ablation is equally consistent with the ragged-tiling account, since ragged
tiling does not care how the weights are split.


### 4.8 Model scale does not move the regime map; dtype does

A registered prediction failed here, and the failure is the result. TinyLlama-1.1B
has a 3.6× smaller per-chip weight floor than Qwen3-4B, so if the mechanism is
that floor, its paid share should be higher. It is lower — −1.2% and 13.4% at
n=4, −1.1% and 5.9% at n=8.

The prediction was wrong because model size is the wrong lever. For dense,
weight-stationary decode, bytes ≈ 2·params and FLOPs per token ≈ 2·params, so

| | weights/chip | FLOPs/token/chip | intensity |
|---|---|---|---|
| Qwen3-4B | 2.01 GB | 2.01 GF | 1.00 FLOP/byte/token |
| TinyLlama-1.1B | 0.55 GB | 0.55 GF | 1.00 FLOP/byte/token |

**Arithmetic intensity is the batch size, independent of parameter count.** The
ridge is a property of the chip (v5e: 197 TFLOP/s ÷ 819 GB/s ≈ 240 FLOP/byte),
and shrinking the model shrinks the floor and the work in the same proportion.

This partly answers the "one primary model" limitation analytically: the regime
map is a function of batch size and dtype, not of model scale. **Scope:** it holds
for dense weight-stationary decode. It breaks for mixture-of-experts, where bytes
read scale with the distinct experts touched so intensity falls below the batch
size, and at long context, where KV bytes rather than weights set the floor. It
does not extend to prefill.

**A scope error we made in this very section, and the eleventh instance of the
class §6 catalogs.** The paid-share numbers above are §4.3's quantity, and §4.3
is emphatic that those are *prefill* steps. The intensity identity used to
explain them is derived for dense weight-stationary *decode* and this section
closes by saying it does not extend to prefill. Applying it here is exactly the
lever/target mismatch that registration was added to catch — and registration did
not catch it, because the check we added tests whether the lever moves the
target, not whether the argument's domain matches the data's. The comparison
needs redoing on decode steps, or replacing with a prefill argument in which
FLOPs scale with padded tokens and the identity does not hold. Until then the
n=4/n=8 numbers below stand as measurements and the explanation attached to them
does not.

**Why the paid share moved *down* rather than staying flat** is a separate
question the intensity argument does not answer. The likely reason is that
non-weight fixed costs — collectives, which our operator profile puts at a flat
~13.4%, plus dispatch and attention — are a larger fraction of a 0.55 GB model's
step, leaving *more* slack for padding to hide in, not less.

**Quantization is the lever that does move it**, and we can now predict
quantitatively rather than directionally. W8 weights halve bytes and leave FLOPs
alone, doubling intensity per token, so the crossing moves from batch ≈ 240 to
batch ≈ 120. Both sit above our measured ceiling of 32 and near the ladder top of
256. **Registered prediction: for this model on this chip, request-dimension
padding is free across the entire compiled ladder in bf16, and int8 halves the
batch size at which that stops.** We have not run it.

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

## 6. Eleven failures, one dominant cause

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
| a smaller model standing in for a quantized one | a lever that cannot move the target quantity |
| a decode identity explaining prefill data (§4.8) | an argument applied outside its stated domain |

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

The tenth is a third variety again, and the most useful. Testing whether a
smaller per-chip weight floor raises the paid share, we substituted a smaller
*model* for a quantized one. But bytes ≈ 2·params and FLOPs/token ≈ 2·params, so
arithmetic intensity is 1 FLOP/byte/token for any dense model and the derivative
with respect to parameter count is **zero**. The lever could not move the target.
Two lines of algebra, available before the session, and unwritten because nothing
required them.

**The guardrail checks claim provenance, not analysis definitions or lever
validity.** Errors eight and nine were caught only by a measurement disagreeing
with an independent one, which is not a method. The tenth was different: it was
caught by its own **registered prediction** failing in the wrong direction, which
is a mechanism, and the first one this project has had for the non-provenance
class. Registration now requires a `prediction_mechanism` field stating the
target as a formula in the lever and why the derivative is nonzero — the check
that would have caught it before any hardware was provisioned. That is one
instance and we do not oversell it, but the class is no longer entirely open.

### The pattern the failure list does not show

Eleven entries above are inferences from numbers. Counting them alone hides
something the project's history makes obvious: **the measurements have survived
three rounds of external review largely intact, and the explanations have not.**

Every headline measurement in §4 still stands as measured. What has been
withdrawn, in order, is a crossover rule, a recoverable-headroom figure, a
microbenchmark and the mechanism it claimed to isolate, a memory-bandwidth
account of why request padding is free, and the frontier bound derived from that
account. Four of the last five retractions were mechanism claims. Not one was a
number that failed to reproduce.

The asymmetry has a cause, and it is structural rather than careless. Every
measurement in this repo runs through a contract that aborts on an unstated
variable, is tied to a `run_id`, and is recomputed from captured data by a script
that exits non-zero on disagreement. **No comparable machinery exists for
explanations.** A mechanism is prose; it can be written, believed, cited by three
later sections, and carried across drafts without ever being executed. The
bandwidth account survived four sessions and two reviews not because evidence
supported it but because nothing in the pipeline was capable of rejecting it.

The registered-prediction discipline of §4.7 and §4.8 is the closest thing we
have to a fix, and its record is instructive: both predictions **failed**, and
both failures were more informative than the successes would have been — one
became a calibrated two-term cost model, the other established that the regime
map is independent of parameter count. A mechanism that never generates a
falsifiable number is not doing work, and this project shipped three of them.

We state this as the paper's least comfortable finding rather than as a
methodological flourish. A reader should trust §4's numbers considerably more
than §4's explanations, and we would rather say so than have it discovered.

---

## 7. Limitations

**One accelerator, one primary model.** v5e with a 4B model places per-chip
weights in the low hundreds of megabytes. Decode is bandwidth-bound only at
n≤8; by n=64 it is bound by neither roof (§4.5). The sharding objection, at least, we can answer: a TP=1/2/4
ablation (§4.7) finds request-dimension padding cheap at every sharding, so
whatever causes it is not a property of this layout. The remaining exposure is model scale and multi-host
topology, where weight streaming and inter-chip collectives change what padding
hides under, and we measured neither.

**No production trace.** §4.4's four length distributions are parametric families
chosen to span a plausible range, not a trace. The result is a sensitivity range,
not a corrected point estimate.

**Prefill step cost above n=16 is still not isolable.** The barrier moved from
n=8 to somewhere between 16 and 32 (§4.3); it did not disappear. At n=16 itself
the clean sample is 7–11 dispatches per arm.

**Decode above n=64 is measured but not clean.** Queue time at n=128 and n=256
means those batches are not fully resident, so the reachable width of the request
ladder under this configuration is itself unmeasured.

**The n=4 convergence is unexplained, and is not an operator effect.** Three
independent observations break there. An operator-level profile (§4.5) shows
every category of device time moving smoothly through n=4, so whatever changes is
not a shift between kernels. We do not know what it is, and we no longer expect a
profiler to show it.

**Co-located prefill and decode only.** On a disaggregated deployment the padding
question splits into two independent questions, and §4.6's finding that variance
is a prefill phenomenon is precisely the asymmetry disaggregation exploits.

---

## 8. Related work

**Pope et al.** derive the memory-bound/compute-bound transition for transformer
inference on TPU analytically. §4.5's roofline is a measured instance of that
regime, not a discovery; what we add is the consequence for compiled-shape
ladders. We do **not** claim the weight-load floor is what makes
request-dimension padding free; §4.5 withdraws that and replaces it with a
measured data-structure account.

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
refute them, and we can now say precisely why not. The regime argument is not
TPU-specific: intensity ≈ batch size on any weight-stationary accelerator, and
the ridge differs only modestly (v5e ≈240 FLOP/byte, H100 ≈295). What differs is
architectural — **CUDA-graph capture makes the batch dimension a paid quantity on
GPU in a way it is not here**, because a graph is captured per shape and replayed,
so an unseen batch size costs a capture rather than riding inside a compiled
step. That is the real distinction between this stack and theirs, and stating it
as architecture is more useful than leaving it as an unmeasured caveat. We have
not run a GPU control; this is an analytic comparison, not a measurement.

We bound their transferability. On this stack the padding they
target is largely not paid, and the batch dimension they would bucket over is
pinned to a single shape by default.

**Vidur** established simulator-fidelity validation as the standard for this kind
of work; our holdout discipline follows it.

---

## 9. Conclusion

A production TPU serving stack quantizes shapes in three dimensions. One does not
exist, one is disabled by a default flag, and the third is paid in a proportion
that depends on batch size — heavily at 1–2, lightly at 4–8, and not at all by
16. How much padding is executed is a property of the workload rather than of
the stack; we withdraw our earlier range as confounded by offered token load and
do not replace it with another.

The practical advice is negative **for batched, throughput-oriented serving**:
on this stack, at batch sizes from roughly 4 up to 16, with a 4B model at any
sharding of four v5e chips, do not build length bucketing, shape-aware admission
control, or ladder design.

**It does not extend to the low-batch regime, and we say so rather than let the
scope be assumed.** At n≤2 we measure ~85% of nominal padding paid on the token
dimension, and a fixed-length workload executes roughly half its tokens as
padding. That is interactive single-stream serving, tight-TTFT deployments, and
the prefill half of any disaggregated system — regimes where the headroom our
advice dismisses may well be real. We did not measure whether a finer token
ladder recovers it, and the staircase at buckets ≤1024 is a reason to think the
question is live. Turning this prohibition into a regime map is the obvious next
piece of work and we have not done it. The phase that
dominates production serving is smooth to n=32 and close to linear above it, and
request-dimension padding is free for a reason that has nothing to do with
compiled shapes: the ragged attention kernel does no work for padded slots.

We arrived here by trying to build the opposite paper. The control experiment
that refuted it cost $3 and should have run first.
