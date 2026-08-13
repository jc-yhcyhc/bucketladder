# Shape Coverage Is a Warmup Cost: Compiled-Shape Padding in Production TPU and GPU Serving

Stack: vLLM 0.25.0 on `v5litepod-4` (`tpu-inference` 0.25.0, JAX 0.10.2,
tensor-parallel degree TP=4) and on an NVIDIA L4. Twenty-six hardware sessions, **well under budget**.

---

## Abstract

Accelerator serving stacks execute a fixed set of compiled or captured shapes and
round every step up to one of them. A family of proposed optimisations — length
bucketing, shape-aware admission control, ladder design — assumes that rounding
up means paying for the shape you rounded up to. We measure what is actually paid,
on a production TPU stack and, with the same serving framework and the same
instrument, on a GPU.

**The premise is largely false.** A batch placed just above a compiled entry
costs about what the entry *below* costs: on TPU it sits 3–5% *under* the lower
entry, and on GPU padding a batch from 8 up to a captured 16 costs **67 µs —
0.6% of the step**, isolated by differencing against an eager arm whose constant
launch overhead cancels to 9 µs. The GPU arm varies batch size, so what it
establishes is that **request**-dimension padding is close to free on both
architectures; the token dimension is measured on TPU only. What is
paid is **shape coverage itself, once, at warmup**: enabling CUDA-graph capture
costs **+108 s** of startup, and XLA compiles a TPU ladder in 5–30 minutes for the
first bucket; neither is a per-step or high-bandwidth-memory (HBM) cost. Work that reduces the number of compiled or captured shapes buys
startup time and memory; work that routes requests to avoid run-time padding is
optimising something close to free.

Three findings support this. **The request ladder a TPU stack reports is not the
one it executes**: a default-off environment flag pins the attention kernel to a
single 256-request shape, enabling the advertised six-entry ladder changes decode
by 0.0%, and the compiler writes the padding into the kernel's own name. **Padded
request slots cost under 0.7 µs each, against 27.5 µs if they were paid, because
the Ragged Paged Attention (RPA) kernel skips them** — a data-structure property,
established by cutting the compiled slot count 32× for a −0.9% change. We test the memory-bandwidth explanation the same data invites and
reject it. **Token padding is different**: it is real arithmetic, paid at 23.1% of
nominal at batch 4 falling to indistinguishable from zero at 16, and it is the
one dimension where bucketing could pay.

**On that dimension an intervention does pay, and we quantify the trade.**
Doubling the token ladder from ten compiled shapes to twenty-one cuts end-to-end
latency by **8.7% and 12.5%** at two concurrent requests for prompts that straddle
ladder entries, while prompts that pad identically on both ladders move by 0.5 ms
— a placebo built into the design. The two treated cells agree on cost per padded
token to 1% (34.9 and 35.3 µs). The price is **+53% cold startup**, twice the
resident executables, and a **boot cliff**: the finer ladder will not start above
`gpu_memory_utilization` 0.85 against the stack's default of 0.92, because
compilation asks for scratch after the KV cache has already been sized to fill the
budget. At equal fraction the two ladders hold identical KV capacity to the token,
so shape count does not shrink the cache — it caps the fraction you may request,
and that cap costs **8.8% of KV capacity** (367,360 tokens at 0.92 against 335,104
at 0.85).

We predicted this benefit would decay with concurrency and vanish by batch 16,
following our own paid-share curve, and **registered that prediction before
measuring it. It is wrong.** Across concurrency 1→16 in both arm orders the
benefit persists at 3.5–12% of end-to-end latency with no crossing, because under
chunked prefill the scheduler packs requests into steps sized to a token budget
rather than to a compiled shape — so padding migrates from per-request to
per-packed-step instead of dissolving. That leaves a tension with the paid-share
curve which we report rather than resolve.

We also reproduce a published latency predictor and find its length term
contributes negligibly where errors are already below 1% and turns actively
harmful at batch 4; report five optimisations, four rejected and one that works; and catalogue fourteen invalid inferences of our own in four
classes, three of which now have mechanical checks.

---

## 1. Introduction

A GPU serving stack resolves kernel shapes at runtime; a TPU stack cannot. XLA
compiles for fixed shapes and recompiling per request is impossible at serving
latencies, so vLLM's TPU backend precompiles a ladder of shapes and rounds every
step up to one. vLLM's CUDA path does something structurally similar for a
different reason: it captures a CUDA graph per batch size in a fixed set, and
pads a batch up to the next captured size.

Both designs create the same apparent inefficiency, and a literature has grown
around removing it. **BucketServe** derives an optimal length-bucket boundary and
then declines to compute it as *"computationally expensive to calculate in
practice."* **LAPS** captures a graph per `(length, batch)` cell and notes that
*"the number of graphs must be limited."* Both take for granted that the padding
they manage is paid at run time.

This paper measures the premise directly. We instrument the three quantized
dimensions of a TPU serving stack, isolate the mechanism behind each, and repeat
the central measurement on a GPU with the same framework and instrument. The
result is that run-time padding is close to free on both, and that the cost of
shape coverage is a warmup charge that neither literature measures.

**Contributions.**

1. **The premise, measured** (§4.3, §4.5, §8). Padding a batch up to a compiled
   or captured entry is close to free on a TPU ladder and on CUDA-graph capture
   alike; what shape coverage costs is warmup. The GPU comparison covers the
   request dimension, which is what graph capture quantizes.
2. **The request ladder a TPU stack reports is not the one it executes** (§4.1).
   Readable from source, confirmed by a paired hardware experiment, and visible
   in the compiler-emitted kernel name — and absent from the Ragged Paged
   Attention (RPA) paper, from LENS, and from vendor documentation.
3. **A mechanism for why request padding is free** (§4.5): the ragged kernel does
   under 0.7 µs of work per padded slot against 27.5 µs if it were paid,
   established by cutting the compiled slot count 32× for a −0.9% change. We also test and reject the memory-bandwidth explanation
   the same section's data invites.
4. **The dimension where bucketing does pay** (§4.3, §4.9, §4.10): token padding
   is real arithmetic, its paid share is high at batch ≤2, and a finer compiled
   token ladder converts that into 8.7–12.5% lower end-to-end latency.
5. **A published latency predictor validated and scoped** (§4.2): LENS's length
   term beats a constant by 0.6 percentage points where errors are already under
   1%, and is beaten by one at batch 4, so its reported accuracy is a property of
   within-bucket flatness rather than of the model form.
6. **Five optimisations, four rejected and one that works** (§5), and
   **fourteen invalid inferences of our own** in four classes (§6), three now
   blocked mechanically.

**Scope.** This is primarily a measurement study: it characterises what shape
quantization costs and where an optimisation can pay. The one intervention it
lands (§4.9) is a documented environment variable rather than a patch, which is
why it is measurable without a custom scheduler to be suspicious of. §5 reports
the four we designed and rejected on measurement, including one that appeared to
win by 29% until a correctness gate showed it was dropping tokens.

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
rather than averaged. §6 classes three of our own errors as instrument-definition
failures, which is the class no guardrail here covers.

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
  is pure linear**, cost proportional to real tokens, padding free.
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
(identical to 0.1 ms at n=8 and n=9).

**That end-to-end comparison is far less powerful than its point estimate looks.**
Bootstrapped, the paired difference is +0.00 ms with a 95% interval of
[−10.7, +10.8] ms at n=8 — **±20% of the decode phase** — so this design could
not have resolved anything smaller. It excludes a large effect and nothing more.
The claim rests instead on the operator-level measurement in §4.5, which compares
attention time directly at 256 versus 8 compiled slots and resolves **−0.9% at
n=1**, where a per-slot padding cost would have been ~42%.

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

### 4.2 Reproducing a published latency predictor on TPU

LENS predicts NPU inference latency to 2.15% mean absolute percentage
error (MAPE) using a per-bucket
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
fit is near-perfect. LENS does beat a constant there — 0.38% against 0.96% — but
the absolute gap is 0.6 percentage points on errors already below 1%, which is
immaterial rather than absent, and it is not evidence that the model **form**
transfers. At n=4 the length term is *actively harmful*. The defensible statement
is that its contribution is negligible where errors are already tiny and negative
where they are large.

**The failure is not an artifact of which points were fitted.** LENS specifies
two measurements per bucket but not which two. Over all three choices per cell,
the n=4 error swings by up to 44.8 percentage points — but its *minimum* over
every choice is still 16.97%, far above LENS's 2.15%. The magnitude must be
reported as a range; the growth in error by n=4 survives. Whether accuracy
returns above n=4 is not sampled and not claimed.

**[Figure 1 — `figures/fig1_lens.png`]** *Held-out prediction error against batch
size, with LENS's reported 2.15% as a reference line and the failure region
shaded. The claim is not that the predictor is inaccurate; it is that the error
grows sharply by n=4. Batch sizes 1, 2 and 4 are sampled, with the failure at the
endpoint, so these data cannot say whether accuracy returns above 4.*

We report this as validation rather than criticism: no prior work had tested LENS
on this hardware, and the regime where it breaks is the one serving uses.

### 4.3 The cost of a step is not a property of the step

**These are PREFILL steps, and they measure the TOKEN dimension (D2).** The
distinction is load-bearing rather than expository, and its justification is not
the memory-bound account §4.5 withdraws: a prefill step carries hundreds to
thousands of tokens, and **padded tokens are real FLOPs** — the kernel computes
them — so token padding is paid wherever the step's cost is dominated by
arithmetic on those tokens. Request-dimension padding (D3) is a different
quantity with a different mechanism (§4.5) and is free at every batch size
measured; the two must not be read off the same table.

Measured as the share of nominal padding paid, straddling a compiled boundary at
fixed batch size and near-fixed sequence length:

| batch size | median | mean | 95% CI over boundaries | boundaries | clean dispatches/arm |
|---|---|---|---|---|---|
| 1–2 | **~85%** | — | *not computed* | 1 | see below |
| 4 | 23.1% | 20.2% | [13.5%, 24.4%] | 4 | 9–15 |
| 8 | 14.3% | 11.8% | [0.2%, 21.0%] | 3 | 3–5 |
| 16 | **−2.7%** | −5.9% | [−15.4%, +0.5%] | 3 | 7–11 |

**The rows above are not matched on boundary, and that confounds them.** At fixed
n=4 the paid share rises with boundary size — 10.0% at 512→1024 to 24.8% at
4096→8192 — so which boundaries a row contains shifts it independently of batch
size. The sets are not the same: n=8 lacks 512→1024, the *lowest*-paying
boundary, and n=16 lacks 4096→8192, the *highest*. Both omissions push in the
same direction and **exaggerate the decline the table is used to show.** This is
the error class §6 names as the project's dominant one, appearing in the headline
table, and we found it only when a reviewer asked which boundaries each row used.

Restricted to the two boundaries present in every row:

| n | 1024→2048 | 2048→4096 | mean |
|---|---|---|---|
| 4 | 22.1% | 24.0% | **23.1%** |
| 8 | 0.2% | 21.0% | **10.6%** |
| 16 | −2.7% | +0.5% | **−1.1%** |

**The decline survives matching**, and the matched n=8 mean is 10.6% rather than
14.3%. Bootstrapping over two boundaries is not worth reporting as an interval;
what the matched table supports is the ordering, not the levels.

The n≤2 row rests on a single boundary, which we do not identify with either of
the matched pair, and we quote no interval for it. It is the least well supported
number in the paper and also the largest, which is the wrong way round.

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

**How far up this can be measured was, in part, our own limitation.** A naive launcher makes this quantity look
unmeasurable above n=8, because the scheduler splits every dispatch. Splitting turned out to track request count, not token
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
predicts a difference. Three observations were previously described as converging on n=4 — this slope,
the paid-padding fraction, and LENS's failure. **They are not three.** LENS's
per-bucket linear model works at n=1–2 because within-bucket flatness is 0.97 and
fails at n=4 because flatness has dropped; "flatness dropped" and "the paid share
dropped" are two descriptions of one measured quantity. The slope rests on a
single measurement by our own account. What remains is one observation, seen
twice, plus a fragile third — and **we could not identify what changes there.**

### 4.4 Padding is abundant, workload-dependent, and mostly free

How much padding a stack executes is a property of the workload, not of the
stack. Across four prompt-length distributions at one Poisson arrival rate
(8 req/s, `output_len=64`, 120 requests each), reporting time to first token
(TTFT) and inter-token latency (ITL):

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
**padded share is a property of the workload, not of the stack**, and a single
stack-level padded-share figure of the kind this experiment was built to produce
is not a well-defined quantity. A matched re-run is
needed before any range is quoted.

**[Figure 2 — `figures/fig2_padding.png`]** *Share of nominal padding actually
paid at each compiled boundary, against the 100% the compiled-shape premise
predicts.*

The share paid rises with the boundary: 10.0% at 512→1024, 22.1% at 1024→2048,
24.0% at 2048→4096, 24.8% at 4096→8192, all at n=4.

**We do not report a single recoverable-headroom figure.** Multiplying a padded
share measured under one workload by a paid share measured at one batch size —
which yields "~4–9% of execution" — is the derivation §6 classes as this work's
dominant error, because §4.3's finding is precisely that the paid share moves with
batch size. The two quantities are reported separately and not multiplied.

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

**The memory-bandwidth account does not survive.** A natural explanation for free
request padding is that the step reads the whole weight set regardless of batch,
so padded slots ride inside a floor batch size cannot move. That account predicts
utilisation climbing toward the compute roof past the arithmetic-intensity ridge
near 240 FLOP/byte, reaching **MFU ≈49% at n=256**. Measured:

| n | 1 | 8 | 32 | 64 | 256 |
|---|---|---|---|---|---|
| MFU | 0.3% | 1.7% | 3.6% | 4.4% | **5.1%** |
| HBM BW utilisation | 61.4% | 52.1% | 31.2% | 21.4% | **11.1%** |

A memory-bound step is one whose achieved bandwidth sits near the roof. Ours
falls monotonically to **21.4% by n=64**, where the queue is 0.1 ms and the
column is clean, so the account fails without needing the high-batch cells at
all. **Neither holds.** The roofline keeps one honest use — byte accounting, 2.01 GB of
weights per decode step regardless of batch — but achieved bandwidth is
`bytes / measured time` and therefore restates the step time it is computed from.

(Sampling `vllm:num_requests_running` later confirmed the ladder is fully
reachable: n=128 and n=256 both hold their requested batch under a synchronised
launch, so the queueing in the sweep above was our arrival pattern, as in §4.3.
The decode numbers there were still taken under the old launcher and want
re-measuring; no conclusion rests on them.)

**The two padding dimensions do not share an explanation.** **D3, requests/step:** free at
every batch size measured, because RPA does substantially no work for padded
request slots — under 0.7 µs each, a data-structure property. **D2, tokens/step:** paid in prefill and falling with
batch, because padded tokens are real FLOPs. §4.3's table is D2; §4.8's dtype
prediction applies to D2 only, since quantization moves the arithmetic-intensity
ridge and the ridge is not in the D3 story.

**What the request-dimension mechanism is, measured directly.**
`ATTN_BUCKETIZED_NUM_REQS` is off, so attention executes at 256 request slots
whatever the batch — and the operator profile says what that costs. With prompt
and output length fixed, so per-sequence KV is constant, attention device time
**aggregated over the full 64-step generation** (not per step) is:

| n | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| attention (µs) | 16 830 | 26 011 | 45 919 | 85 103 | **163 535** |
| per real request | 16 830 | 13 005 | 11 480 | 10 638 | 10 221 |

**This does not discriminate the hypothesis.** The tempting argument — a kernel
doing work for 256 padded slots would be flat in n, this is not flat, therefore
padded slots are skipped — fails because *flat* was never the alternative. Padded slots hold no KV blocks (§4.1), so
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
padding.** Stated as the bound the experiment actually supports: removing 248 of
256 slots moves the fitted fixed term by 167 µs, so a padded request slot costs
**under 0.7 µs**, against the 27.5 µs per slot that fully-paid padding would imply
if the 7 050 µs fixed term were padding. That is a factor of 41, and it is a bound
rather than a zero — at 256 slots the residue is up to ~170 µs, about 1% of
attention time at n=1.

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

**A microbenchmark that does not measure what it appears to.** An isolated matmul at
the model's real sharded shapes returns 142.9 µs at M=1, flat to 143.6 µs at
M=256, which invites reading as the weight-load floor with confounds removed. The
qkv projection holds 7.86 MB per chip, so the bandwidth floor is 9.6 µs: the
measurement sits 15× above it at **7% of peak**, timing per-dispatch overhead.
Amortising over a loop reports 1250% of peak because XLA hoists the
loop-invariant matmul; chaining iterations is physically valid at 79% of peak but
streams weights from HBM every iteration and stays bandwidth-bound. **The MXU
tiling hypothesis for §4.3's n=4 convergence is untested**; settling it needs
weights resident in VMEM, which here means a Pallas kernel.

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
weight bytes scale as 1/TP, where TP is the tensor-parallel degree — the number
of chips a single model's weights are sharded across — so the level should scale
with 1/TP and the shape should be preserved.

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

**`F` is not the collectives.** A constant term cannot represent inter-chip
communication: collectives are
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

A second registered prediction failed here. TinyLlama-1.1B has a 3.6× smaller
per-chip weight floor than Qwen3-4B, so if the mechanism were that floor its paid
share should be higher. It is lower — −1.2% and 13.4% at n=4, −1.1% and 5.9% at
n=8.

Model size is the wrong lever. For dense weight-stationary decode, bytes ≈
2·params and FLOPs per token ≈ 2·params, so both models sit at **1.00
FLOP/byte/token** and **arithmetic intensity is the batch size, independent of
parameter count**. The ridge is a property of the chip (v5e: 197 TFLOP/s ÷
819 GB/s ≈ 240 FLOP/byte); shrinking the model shrinks the floor and the work in
the same proportion. This answers the "one primary model" limitation
analytically for that regime — it breaks for mixture-of-experts, where bytes
scale with distinct experts touched, and at long context, where KV rather than
weights sets the floor.

Two cautions. The identity is derived for *decode*, while the paid-share numbers
above are *prefill* (§4.3) — applying it to them is the domain error §6 classes
as our twelfth, and we report the numbers as measurements with the explanation
withdrawn. And the intensity argument explains why model size should not move the
paid share, not why it moved *down*; the likely reason is that non-weight fixed
costs are a larger fraction of a 0.55 GB model's step, leaving more slack for
padding to hide in.

**Dtype is the lever that does move it, on the token dimension.** W8 weights halve
bytes and leave FLOPs alone, doubling intensity per token, so the crossing moves
from batch ≈ 240 to ≈ 120. **Registered prediction: under W8, the token-dimension
paid share at a fixed boundary rises.** We have not run it.

### 4.9 A finer ladder does pay — and the price is startup, not capacity

Every result above says token padding is paid at small batch and vanishes as
batch grows (§4.3, §4.4). That is a claim about where bucketing's premise could
still hold, and it is testable directly: make the ladder finer and see whether
anything is bought. `VLLM_TPU_BUCKET_PADDING_GAP` is the lever — unset, the TPU
backend compiles ten exponentially spaced token shapes; set to 512, twenty-one
linear ones — so no patch is required, only an environment variable.

The design has a placebo built into it. Of four prompt lengths, two pad to the
*same* compiled shape on both ladders and two pad smaller on the fine ladder
only. The placebo cells therefore measure everything that differs between two
server boots except padding. Qwen3-4B, `v5litepod-4`, TP=4, two concurrent
requests, `output_len=32`, 18 repeats per cell pooled over both arm orders:

| prompt | pads to (10 shapes) | pads to (21 shapes) | tokens saved | 10-shape | 21-shape | difference [95% CI] |
|---|---|---|---|---|---|---|
| 300 | 512 | 512 | 0 | 149.8 ms | 150.6 ms | +0.8 ms [+0.1, +1.6] |
| 600 | 1024 | 1024 | 0 | 168.5 ms | 168.6 ms | +0.1 ms [−0.5, +0.6] |
| 1200 | 2048 | **1536** | 512 | 206.0 ms | 188.2 ms | **−17.9 ms** [−18.2, −17.5] |
| 3000 | 4096 | **3072** | 1024 | 289.9 ms | 253.7 ms | **−36.2 ms** [−36.5, −36.0] |

The two treated cells agree on cost per padded token to 1% — **34.9 and 35.3 µs**
— while the placebo cells sit at +0.5 ms combined. That agreement is the load-
bearing evidence: an arm-level offset, such as one boot simply running slower,
produces a *constant* difference and therefore a per-token figure that scales as
1/(tokens saved). A padding effect scales with tokens saved, which is what is
observed. For scale, a *real* prefill token costs 46.6 µs on the same arm
(1200→3000 tokens, 83.9 ms), so at two concurrent requests a padded token costs
about three quarters of a real one. Token padding here is arithmetic, as §4.3
found, and at this batch size it is arithmetic that is actually executed.

This is the paper's one optimisation that works, and it is not free. Measured
against the same stack with the compiled-shape cache cleared before each boot:

| | 10 shapes | 21 shapes |
|---|---|---|
| cold warmup | 285 s | 436 s (+53%) |
| compiled-shape cache on disk | 43 MB | 92 MB |
| highest memory fraction that boots | 0.92 (the default) | **0.85** |
| KV cache at 0.92 | 367,360 tokens | *does not boot* |
| KV cache at 0.85 | 335,104 tokens | 335,104 tokens |

**Shape count does not move KV capacity; it moves the memory fraction you are
allowed to ask for.** At 0.85 the two ladders report the same 335,104 tokens — the
same figure to the token, which also bounds any residual effect, since capacity is
block-quantized and the 49 MB executable difference would be roughly 1,300 tokens
or 81 blocks and would have shown. What the finer ladder actually costs is the
backoff it forces: the ten-shape ladder runs at the 0.92 default for 367,360
tokens, the twenty-one-shape ladder does not start above 0.85, and **the
difference between those two operating points is 32,256 tokens, or 8.8% of KV
capacity.**

Reaching that number required measuring the cliff rather than assuming it. The
long ladder fails at 0.92, 0.90 and 0.88, and boots at 0.85; each failure is a
`RESOURCE_EXHAUSTED` in warmup, at 0.92 requesting 32.50 MB against 12.40 MB free,
byte-identical on two independently provisioned v5e-4 hosts. A 20 MB miss invites
the inference that a hair of headroom would fix it. It does not: the requirement
survives three successive backoffs, and only the fourth clears it.

**The mechanism is not steady-state competition between executables and cache**,
which the identical capacities rule out. Every failed boot *logs a KV cache size
before it dies* — 358,144 tokens at 0.90, 348,928 at 0.88 — so the cache is sized
to fill the fraction first and compilation then asks for scratch against what is
left. Shape coverage is charged to transient compilation headroom that the sizing
step does not reserve, which is why the price appears as a boot cliff rather than
as a smaller cache.

One earlier version of this table reported the cost as 15.0%, differencing the
ten-shape ladder's 367,360 tokens at 0.92 against 312,320 at 0.80 and attributing
the gap to shape count. Those two numbers come from different memory fractions,
and 0.80 was a coarse guess rather than the measured requirement, so the figure
was both misattributed and too large. It is a provenance error of the class §6
names as this project's most common, recorded there as the fourteenth.

### 4.10 The benefit does not decay with concurrency, and the prediction that it would was wrong

§4.9's benefit looks like it should be conditional on load, because padding is
paid at small batch and free at large (§4.3). That reasoning predicts a crossing,
and we registered it before measuring:
the difference should shrink monotonically and reach zero between n=4 and n=16,
tracking §4.3's paid-share curve. Sweeping concurrency 1→16 on both ladders, in
both arm orders, refutes it. Placebo-corrected difference, in ms, negative where
the finer ladder is faster:

| prompt | n=1 | n=2 | n=4 | n=8 | n=16 |
|---|---|---|---|---|---|
| 1200 | −9.9 | −18.0 | −8.9 | −16.2 | −23.0 |
| 3000 | −18.7 | −37.3 | −21.5 | −50.4 | −37.0 |

There is no crossing. The benefit is 3.5–12% of end-to-end latency at every
concurrency sampled, it is non-monotone rather than decaying, and at n=16 it is
larger in absolute terms than at n=1. The treated cells reproduce across arm
orders to within a few ms at n=4 and n=16.

**Why the prediction failed is visible in what the stack executes.** The
prediction reasoned about a *request's* padding shrinking as more requests share
a step. Under chunked prefill the scheduler does not work that way: it admits
requests in waves and packs them, and the size of the packed step — not any
request's length — selects the compiled shape. Snapshotting vLLM's
`iteration_tokens_total` histogram around one n=16 cell shows a single repeat
costing about three prefill steps, landing in (256,512], (512,1024] and
(2048,4096], alongside 93 decode steps of at most 16 tokens; the two arms produce
near-identical step distributions, differing only in which compiled shape those
steps round up to. Padding does not dissolve as concurrency rises. It migrates
from per-request to per-packed-step, and packed steps go on straddling ladder
entries, because the scheduler is packing to a token budget rather than to a
compiled shape.

**This sits in tension with §4.3**, which finds the paid share of token padding
falling to indistinguishable from zero by batch 16. We report the tension rather
than resolve it. The two measurements differ in what is held fixed: §4.3 varies
batch size at a fixed boundary and attributes padding per request, while this
sweep varies offered concurrency and lets the scheduler choose step composition.
If both are right, the reconciliation is that per-request padding does vanish
while per-step padding does not, and the quantity a ladder acts on is the second.
Testing that needs padded tokens per packed step, and the obvious instrument does
not supply them: `iteration_tokens_total` bins on powers of two, which is coarser
than the ladder spacing being compared — a step recorded in (2048,4096] pads to
2048, 2560, 3072, 3584 or 4096 depending where in that bin it fell, and the two
ladders differ precisely inside the bin. The histogram establishes that packed
steps land where the ladders differ; it cannot say by how much. Resolving the
tension needs per-step token counts, not a bucketed distribution.

**A design defect, stated because it bounds the result.** Prompt 300 was intended
as a placebo at every concurrency, on the reasoning that it pads to 512 on both
ladders. The step histogram shows that is true only while each request prefills
in its own dispatch — at n≥4 the packed step lands where the ladders differ, so
the cell is treated, not inert. Its measured difference is correspondingly
unstable across arm orders at n=8 and n=16 (−6.0 against −27.7 ms, −1.5 against
−17.2 ms) where the treated cells are stable. The correction is therefore applied
with a negative offset at those levels, which shrinks the reported benefit: the
n≥4 figures above are **floors on the effect, not unbiased estimates of it**, and
arm order is the only control they carry. The n=1 and n=2 rows are unaffected.

**The sign survives this; the magnitude does not.** The placebo's own spread
across arm orders at n=8 and n=16 is about 20 ms, comparable to several treated
differences in the same table, so no n≥4 magnitude here is resolvable at interval
level. What is resolvable is direction: the raw differences at n=16 are −32.2 and
−46.3 ms, and subtracting even the most extreme placebo estimate observed (−27.7
ms) leaves both negative in both arm orders. So "the benefit does not reverse by
n=16" is supported, while "it is 23.0 ms at n=16" is not. The absence of a
crossing is established only at n=1 and n=2, where the placebo is valid; above
that it is an observation with a floor, not a measurement with an interval.

Scope. One model, one topology, and prompt lengths chosen to straddle ladder
entries — the most favourable case for a finer ladder, by construction. The
warmup figures in §4.9 are cold; a persistent compilation cache amortises them
across restarts, and the warm boots we measured (165–315 s) reflect cache reuse
rather than ladder size.

The recommendation that survives is therefore simpler than the load-conditional
one the paid-share curve implies, and weaker than the sweep alone suggests:
**where prompt lengths straddle a coarse ladder's entries, a finer token ladder
earns its startup at low concurrency, and does not reverse anywhere up to n=16.**
The low-concurrency claim carries an interval; the rest carries a direction only.
Where the benefit stops we cannot say, because within 1→16 it does not.

---

## 5. Five optimisations, four rejected and one that works

| | outcome |
|---|---|
| **finer token ladder** | **works: −8.7% and −12.5% end-to-end at n=2, persisting 1→16, at +53% startup and a boot cliff (§4.9, §4.10)** |
| bucket-aware admission control | premise false (§4.4) |
| ladder redesign on the request dimension | D1 does not exist; D3 inert by default |
| last-chunk decomposition | **20.6% worse** measured (51.06 vs 42.33 ms) |
| bucket-aligned step packing | implemented twice: inert, then output-corrupting |

The one that works is the one aimed at the single dimension the measurements left
open. Three of the four rejected optimisations target the request dimension or
per-request length padding, and §4.1 and §4.4 show neither carries cost; the
fourth restructures work the stack already packs. The token dimension is the only
place the premise survived contact with measurement, and it is the only place an
intervention paid — though not for the reason we expected, since we predicted the
payoff would be confined to low concurrency and it was not (§4.10).

A second positive measurement — release timing saving 26% of TPU time against
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

## 6. Fourteen failures, one taxonomy

Fourteen invalid inferences were made and caught during this work. The full
catalogue is in the artifact; what matters here is that they fall into four
classes, and that the guardrails cover only the first.

| class | count | what it is | covered? |
|---|---|---|---|
| **provenance** | 8 | a quantity measured under one configuration, used under another | **yes** — config-diff over registered claims |
| **instrument definition** | 4 | an analysis that measures something other than the target | no |
| **lever validity** | 1 | a lever that cannot move the quantity claimed | **yes** — `prediction_mechanism` |
| **dimension** | 1 | a result in one quantized dimension licensing a claim in another | **yes** — required `dimension` field |

**The provenance guardrail took three versions.** *"No derivation may combine
quantities measured at different batch sizes"* would not have caught the
`output_len` failure. A whitelist of config keys missed the largest error,
because batch size is not a top-level field — it lives inside experiment-specific
structures. The working form diffs **every** config key across a claim's source
runs, exempts only free text, and requires each difference to be named. It
flagged five claims already believed correct and has since killed two of our own
headline numbers: a crossover point and a recoverable-headroom figure.

**Its coverage is the set of *registered* claims, not the set of claims made.**
The headroom figure evaded it for three drafts by living in prose, and was caught
only when we registered it in order to check it. §4.9's KV-capacity price — the
eighth provenance failure and the most recent — evaded it the same way: capacity
at memory fraction 0.92 was differenced against capacity at 0.80 and the gap
attributed to ladder length, in a table assembled by hand from two servers' boot
logs rather than by a script over registered runs. Boot-time facts are exactly
the quantities that do not flow through `save_table`, and nothing checks them.

**The four instrument-definition errors are not covered by anything.** A
step-count criterion that could never pass; a boundary experiment that pooled
split dispatches it claimed to exclude; a microbenchmark that timed dispatch
overhead at 7% of peak bandwidth and called it a weight-load floor; and §4.10's
placebo, a control cell chosen to be inert that stops being inert above n=2 once
the scheduler packs requests into shared steps. Each was caught by a measurement
disagreeing with an independent one, which is luck rather than method — and one
of them, the split pooling, biased upward, the direction that manufactures a
positive result.

The placebo failure is the clearest instance of the class, because the reasoning
behind it was checkable and never checked. "Prompt 300 pads to 512 on both
ladders" is a statement about what the stack executes, and the stack exports what
it executes: one histogram scrape settled it in under a minute, after the arms had
already been run. Nothing in our machinery asks whether a control is a control.
The check that would have fired is cheap and we did not have it — **assert the
executed shape distribution matches the one the design assumes, before treating a
cell as inert** — and it is the one guardrail this work still owes.

The last two classes each produced a mechanical check, and both checks are cheap:
state the target as a formula in the lever and show the derivative is nonzero;
name the quantized dimension a claim belongs to and reject derivations that cross
one. Both would have fired before hardware was provisioned.

### The pattern the failure list does not show

Fourteen entries above are inferences from numbers. Counting them alone hides
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

**One TPU slice, one GPU, one primary model.** v5litepod-4 with a 4B model, and a
single L4 for the control. The sharding objection we can answer — a TP=1/2/4
ablation (§4.7) finds request padding cheap at every sharding — but model scale
and multi-host topology are unmeasured, and both change what padding hides under.
§4.8 argues analytically that the regime map is a function of batch size and
dtype rather than parameter count, for dense weight-stationary decode only.

**The token-ladder benefit has no measured upper bound in concurrency.** §4.10
sweeps 1→16 and finds no crossing, so we cannot say where a finer ladder stops
paying — only that it has not stopped by 16. The ~85% paid share at n≤2 still
rests on one boundary with no interval (§4.3), and §4.10's finding sits in tension
with that curve's decay to zero by batch 16, unresolved. Above n=4 the experiment
has no valid placebo, because the scheduler packs requests into shared steps that
straddle ladder entries, so those rows are floors on the effect rather than
unbiased estimates.

**No production trace.** §4.4's four length distributions are parametric families
and were not matched on offered tokens, which is why we withdraw the range they
produced rather than report it.

**Prefill step cost above n=16 is still not isolable**, and at n=16 the clean
sample is 7–11 dispatches per arm.

**The n=4 convergence is unexplained and is not an operator effect.** An
operator-level profile shows every category of device time moving smoothly
through n=4. We also no longer claim three independent observations converge
there: LENS's failure and the paid-share drop are the same quantity described
twice (§4.3).

**The GPU control is one point, not a curve.** We measured startup at vLLM's
default capture set and did not vary the number of captured shapes — which is
precisely the axis BucketServe and LAPS trade along. The +108 s is one sample of
the quantity we identify as the real cost, not its shape.

**Co-located prefill and decode only.** On a disaggregated deployment the padding
question splits in two, and §4.6's finding that variance is a prefill phenomenon
is the asymmetry that motivates disaggregation.

---

## 8. Related work

**Pope et al.** derive the memory-bound/compute-bound transition for transformer
inference on TPU analytically; §4.5's byte accounting is a measured instance of
that regime rather than a discovery. We do not claim the weight-load floor
explains free request padding — §4.5 tests that and rejects it.

**RPA** is the technique this work validates: per-request padding costing nothing
is what its ragged-tiling design predicts, though it does not discuss the
request-count dimension, report cost against batch size, or quantify how much
padding survives it. **LENS** (§4.2) supplies the model form; we supply the
hardware it was not tested on and the ablation showing its length term is not
what carries its reported accuracy.

**PagedAttention/vLLM** is the stack measured throughout; **Orca**'s
iteration-level scheduling produces the per-step batches; **Sarathi-Serve**
introduced the chunked prefill §4.4 controls for. **DistServe** and **Splitwise**
disaggregate prefill from decode, which is the architectural response to §4.6's
finding that variance is a prefill phenomenon and which bounds our advice to
co-located deployments. **Vidur** established simulator-fidelity validation as
the standard for this kind of work; our holdout discipline follows it.

**BucketServe** and **LAPS** manage length-bucketing overhead on GPU, and we
measured the comparison rather than asserting it. Same vLLM 0.25.0, same
instrument, an L4 in place of the v5e:

| arm | n=8 | n=9 | n=16 | 8→9 | 8→16 | startup |
|---|---|---|---|---|---|---|
| CUDA graphs on | 10.605 | 10.887 | 12.298 | **0.283** | **1.693** | 118.7 s |
| `--enforce-eager` | 19.934 | 20.150 | 21.618 | **0.215** | **1.684** | 10.7 s |

**The increments are the measurement; the levels are not.** Eager execution pays
a constant per-op launch overhead — the levels differ by roughly 9.3 ms
throughout — and that constant cancels in any increment. It does so to within
**9 µs on the 8→16 step** (1.693 vs 1.684 ms), which is the control this
comparison needs and which we did not design: it says the two arms measure the
same underlying work plus an offset, so their difference isolates what capture
adds.

On that basis, **padding a batch from 8 up to the captured entry at 16 costs
about 67 µs** — the 0.283 ms increment with graphs against 0.215 ms without.
That is **0.6% of a 10.9 ms step, or 4.0% of the nominal padding** implied by
rounding 9 up to 16. Run-time batch padding is close to free on GPU as well as
on TPU.

**This tests one of the three dimensions, and it is the one already least in
doubt.** vLLM's CUDA path captures a graph per batch size, so what the arms above
vary is the request dimension — D3, the dimension the TPU results explain away
with a data-structure argument (§4.5) and where no proposed optimisation was
going to pay. The token dimension, D2, is where this paper locates the only
surviving effect, and these measurements do not reach it: no arm here varies
tokens per step at fixed batch. The cross-architecture statement this table
supports is therefore **"request-dimension padding is close to free on both
architectures"**, not "the premise is false on both". Whether a GPU stack pays
for token padding the way a TPU stack does is untested here, and it is the
experiment that would make the cross-architecture claim general.

**We state the cross-architecture comparison as a bound, not an equality.** §4.1's
TPU statistic carries intervals of roughly ±50 percentage points, so this table
cannot resolve a difference between the architectures; what both support is that
the paid share is small on each. The millisecond
increments above are the load-bearing form; the normalised percentages are a
convenience and turn on a 4-point difference that this design cannot resolve.
These remain single measurements: **three batch points, one GPU, no repeats.**

**What is paid is the capture, and it is paid at startup.** Enabling graphs costs
**+108 s** of initialisation — 118.7 s against 10.7 s — for a capture set fixed
in advance. That is precisely the quantity BucketServe and LAPS manage when they
write that the number of graphs must be limited, and it is a warmup cost. The TPU
analogue is XLA compilation: 5–30 minutes for the first bucket and 30–120 s per
additional one.

So the cross-architecture statement is: **both stacks pay for shape coverage once,
up front, and neither pays materially for it per step.**

**Where, then, do the reported gains come from?** If the padding premise is false
on both architectures, systems reporting end-to-end improvements from bucketing
are improving something else, and §5 supplies a candidate from our own data: the
one positive measurement in this work — release timing saving 26% of TPU time at
25 req/s (p=0.001, six paired seeds) — is **dynamic batching**, an
arrival-and-composition effect, not a shape effect. Bucketing schemes change
which requests occupy a step together, and that is worth something independent of
padding. We cannot separate the two in others' reported settings, but we can say
that our own shape-motivated intervention paid off through a mechanism that had
nothing to do with shape, and that a bucketing result which does not control for
batch composition cannot distinguish the two.

**Scope.** One GPU (L4, 23 GB), one model, three batch points, no repeats. The
startup figure is one sample at vLLM's default capture set; we did not vary the
number of captured shapes, which is the axis BucketServe and LAPS trade along.

---

## 9. Conclusion

Two accelerator families reach the same design by different routes — XLA compiles
a ladder of shapes, CUDA captures a graph per batch size — and both round every
step up to the nearest entry. The optimisation literature treats that rounding as
a cost to be recovered. **It is not, on either.** A batch just above an entry
costs what the entry below costs; the padding rides inside work the step was
doing anyway, because a ragged attention kernel does almost no work for slots
holding no KV blocks — under 0.7 µs per slot, against 27.5 µs if it were paid and a captured graph does not care that some of its batch is unused.

What shape coverage costs is **warmup**. Enabling CUDA-graph capture costs 108
seconds of startup; XLA compiles the first TPU bucket in 5–30 minutes. That is
the quantity BucketServe and LAPS are managing when they write that the number of
graphs must be limited — and it is a startup and memory-footprint budget, not a
throughput one. **Reducing the number of shapes is worth doing for time-to-serve
and resident executables — everywhere except the one regime named below.
Routing requests to dodge run-time padding is not worth doing anywhere.**

The exception is the token dimension. Padded *tokens* are real arithmetic: their
paid share is 23.1% of nominal at batch 4, falls to indistinguishable from zero by
16, and is around 85% at batch ≤2 — a figure that rests on a single boundary and
carries no interval, the least well supported number here and also the largest.
That low-batch regime is interactive serving,
tight-TTFT deployments, and the prefill half of any disaggregated system — and it
is the one place where making the ladder finer buys something. Twenty-one compiled
token shapes instead of ten cut end-to-end latency 8.7% and 12.5% at two
concurrent requests, at a cost of 53% more cold startup, twice the resident
executables, and a boot cliff: the finer ladder does not start at all at the
stack's default memory fraction, missing by 20 MB, though at equal fraction the
two ladders hold identical KV capacity (§4.9).

We expected that benefit to be confined to low concurrency and predicted, in
advance, that it would vanish by batch 16. It does not: it holds at 3.5–12% across
1→16 with no crossing (§4.10), because the scheduler packs to a token budget
rather than to a compiled shape, so padding moves from per-request to
per-packed-step instead of disappearing. So the rule is not that shape coverage
should always be minimised, and not the load-conditional rule the paid-share curve
implies either: **where prompt lengths straddle a coarse ladder's entries, a finer token
ladder earns its startup and its capacity across the whole load range we tested.**
Where it stops earning them, we do not know.

Three things we got wrong are worth carrying forward as much as the results. We
explained free request padding with a memory-bandwidth argument, published it,
and withdrew it when the step turned out to sit at a fifth of the bandwidth roof.
We predicted that graph capture would make the batch dimension a paid quantity on
GPU in a way it is not on TPU, and measured it not to be. And our own guardrails
caught seven provenance errors while three instrument-definition errors slipped
past every check we had, each surfacing only when a measurement disagreed with an
independent one. The measurements in this paper have survived four rounds of
review intact; the explanations have not, and §6 argues that the asymmetry is
structural — a pipeline can execute a number and cannot execute prose.

We arrived here by trying to build the opposite paper. The control experiment
that refuted it cost $3 and should have run first.
