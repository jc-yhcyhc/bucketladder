# Shape Coverage Is a Warmup Cost: Compiled-Shape Padding in Production TPU and GPU Serving

Stack: vLLM 0.25.0 and `tpu-inference` 0.25.0 on a `v5litepod-4` TPU slice
(JAX 0.10.2, tensor-parallel degree TP=4), with an NVIDIA L4 for the GPU control.

---

## Abstract

Accelerator serving stacks execute a fixed set of compiled or captured shapes and
round every step up to one of them. A family of proposed optimisations — length
bucketing, shape-aware admission control, ladder design — assumes that rounding up
means paying for the shape that was rounded up to. This paper measures what is
actually paid, on a production TPU stack and, using the same serving framework and
the same instrument, on a GPU.

**This assumption does not hold in the stacks we measured.** A batch placed just
above a compiled entry costs
approximately what the entry below it costs: on TPU it falls 3–5% under the lower
entry, and on GPU, padding a batch from 8 up to a captured 16 costs 67 µs, or 0.6%
of the step. **The two architectures are compared on the request dimension only.**
CUDA-graph capture quantizes batch size, so the GPU arms vary batch size and
establish request-padding parity; the token dimension, on which this paper's
recommendation rests, is measured on TPU alone. Compilation overhead is instead
incurred as a static
startup cost: enabling CUDA-graph capture costs 108 s of initialisation, and XLA
compiles a TPU ladder in 5 to 30 minutes for its first bucket. Neither is a
per-step cost.

The three quantized dimensions behave differently, and separating them is the
paper's organising claim. Per-request prompt-length padding **does not exist**: the
stack has no such ladder. Request-slot padding is **free**, at under 0.7 µs per
padded slot against 27.5 µs if it were paid, because the Ragged Paged Attention
(RPA) kernel does no work for slots holding no key–value blocks. Token padding is
**real arithmetic**, paid at 23.1% of nominal at batch 4 and indistinguishable from
zero at batch 16, and it is the one dimension on which an intervention can pay.

**On that dimension, ladder design pays — and the effective variable is placement
rather than shape count.** A fourteen-shape ladder that adds one entry the default
lacks reduces end-to-end latency by 12.1% at the prompt length that straddles it,
by 0.2% at a length it does not, and boots at the stack's default memory fraction
with unchanged key–value cache capacity. A twenty-one-shape ladder achieves the
same reduction and additionally fails to start above a memory fraction of 0.85
against a 0.92 default, which costs 8.8% of cache capacity and 53% more startup
time. Every cost measured scales with the number of compiled shapes; the benefit
depends only on whether a boundary falls between the prompt length and the next
entry. A ladder can be chosen offline: expected padding computed from a length
distribution predicted the measured reduction to within 5%.

Two measurements bound what that reduction is worth in deployment. Swept against
offered load, it reduces median latency by 46% just below saturation but increases
sustained throughput by only 2.6% at saturation, making it a latency optimisation
for under-saturated serving rather than a capacity one. With prefix caching
enabled, as production vLLM ships it, the same placement yields 1.7% instead of
12.3%, because caching shortens the prefill onto a different compiled entry.

A prediction registered before measurement held that this benefit would decay with
concurrency and vanish by batch 16. It does not: latency remains 3.5–12% lower
across concurrencies 1 to 16. Under chunked prefill the scheduler assembles steps
against a token budget rather than a compiled shape, so padding moves from
individual requests to the packed step rather than being eliminated.

A final result concerns methodology rather than TPU serving. This work catalogues
fourteen invalid inferences of its own, and their distribution is asymmetric:
across four rounds of external review every headline measurement was retained,
while four of the five most recent retractions were claims about mechanism. Every
number reported here passes a configuration contract that aborts on an unstated
variable and is recomputed by a script that fails on disagreement, whereas no
equivalent machinery constrains an explanation. Readers should accordingly place
greater confidence in the measurements than in the mechanisms offered for them.

---

## 1. Introduction

A GPU serving stack resolves kernel shapes at runtime; a TPU stack cannot. XLA
compiles for fixed shapes, and recompiling per request is impossible at serving
latencies, so vLLM's TPU backend precompiles a ladder of shapes and rounds every
step up to one of them. vLLM's CUDA path does something structurally similar for a
different reason: it captures one CUDA graph per batch size from a fixed set and
pads a batch up to the next captured size.

Both designs create the same apparent inefficiency, and a literature has grown
around removing it. **BucketServe** derives an optimal length-bucket boundary and
then declines to compute it, describing it as *"computationally expensive to
calculate in practice."* **LAPS** captures a graph per `(length, batch)` cell and
notes that *"the number of graphs must be limited."* Both take for granted that the
padding they manage is paid at run time.

This paper presents an empirical evaluation of run-time shape-padding overhead. We instrument the three quantized dimensions of a
TPU serving stack, isolate the mechanism responsible for each, and repeat the
central measurement on a GPU using the same framework and instrument. Run-time
padding proves close to free on both, and the cost of shape coverage proves to be a
warmup charge that neither system measures.

We additionally document how the experimental methodology evolved to enforce
falsifiability, because that evolution shaped what was measured. Our measurements
have proved durable and our explanations have not: over four rounds of
external review, no headline number was withdrawn, while four of the last five
retractions were claims about mechanism. This asymmetry is structural rather than
accidental. Numbers here pass through a contract that aborts on an unstated
variable and a script that recomputes them, whereas a mechanism is prose and
nothing in the pipeline can reject it. The response adopted here is to require that
a proposed mechanism emit a falsifiable number before hardware is provisioned.
Three such registered predictions failed (§4.4, §4.8), and each failure was
more informative than a confirmation would have been.

**Contributions.**

1. **The premise, measured** (§4.1, §4.2, §8). Padding a batch up to a compiled or
   captured entry is close to free on a TPU ladder and on CUDA-graph capture
   alike; what shape coverage costs is warmup. The GPU comparison covers the
   request dimension, which is the dimension graph capture quantizes.
2. **The request ladder reported by the TPU stack differs from the one its
   attention kernel executes** (§4.1). This is readable from source, confirmed by
   a paired hardware experiment, and visible in the compiler-emitted kernel name.
   It is absent from the RPA paper, from LENS, and from vendor documentation.
3. **A mechanism for free request padding** (§4.1): the ragged kernel performs
   under 0.7 µs of work per padded slot, against 27.5 µs if it were paid,
   established by reducing the compiled slot count by a factor of 32 for a −0.9%
   change. The memory-bandwidth explanation that the same data invites is tested
   and rejected.
4. **Ladder design as placement rather than cardinality** (§4.3, §4.6): a
   fourteen-shape ladder placed against the workload reduces end-to-end latency by
   12.1% at no memory cost, while a uniformly finer ladder achieves the same
   reduction and additionally costs 8.8% of cache capacity and 53% more startup.
   The choice can be made offline from a length distribution, predicting the
   measured result to within 5%.
5. **The conditions under which that gain holds** (§4.4, §4.5): it is a latency
   optimisation below saturation rather than a capacity one, and prefix caching
   reduces it from 12.3% to 1.7% by moving the prefill onto a different entry.
6. **Five optimisations, four rejected and one that works** (§5), and **fourteen
   invalid inferences of our own** in four classes (§6), three of which now have
   mechanical checks.
7. **An asymmetry between measurements and explanations** (§6), reported as a
   finding rather than a disclaimer.

**Scope.** This is primarily a measurement study. The single intervention it
reports (§4.3) is set through a documented environment variable rather than a
patch, so the measurement does not depend on the correctness of a modified
scheduler.

---

## 2. Method

**Controlled variables.** Prefix caching is disabled and asserted, except in §4.5
where it is the treatment. Chunked prefill, `max_model_len`,
`max_num_batched_tokens`, tensor-parallel size, `gpu_memory_utilization`,
`XLA_FLAGS` and `ATTN_BUCKETIZED_NUM_REQS` are recorded. Every run parses the
server's own engine-configuration line and aborts if any controlled variable
disagrees with the configuration it reports. The check has rejected three runs in
which a controlled variable had drifted, and one in which a control was varied
deliberately and declared.

**Units.** All timings are measured server-side, as differences between Prometheus
histogram snapshots taken before and after each measurement block. Client-side
wall-clock timings are not used, because they include round-trip time, HTTP
overhead, tokenisation and queueing delay.

**Scope of instruments.** A step-scoped property requires a step-scoped instrument.
Single-step execution is verified per dispatch from the count delta of
`iteration_tokens_total`, and dispatches that split are excluded rather than
averaged. §6 classifies three of our own errors as instrument-definition failures,
the one class for which no guardrail described here provides coverage.

**Request arrival.** Where a measurement requires *n* requests to reach the
scheduler within one step, they are released from a thread barrier after every
connection is established, so arrival spread is measured in microseconds rather
than milliseconds. §4.2 shows that this changes what is measurable.

**Definitions.** Three quantities are used throughout.

- **Flatness** — how far a short request's cost sits from what proportional
  scaling would predict, as a fraction of the distance to the full-bucket cost:
  `(cost(L) − p) / (cost(B) − p)`, where `p = cost(B) · L/B`. A value of 1.0 is a
  pure staircase, meaning cost is independent of true length and padding is fully
  paid; 0.0 is pure linearity, meaning cost is proportional to real tokens and
  padding is free.
- **Share of nominal padding paid** — `(measured − real) / (padded − real)`, where
  `real` is the cost ratio predicted by real tokens alone and `padded` the ratio
  predicted if the full compiled shape were paid. Zero means padding is free; one
  means it is fully paid, which is what the compiled-shape premise predicts.
- **Model rejection, quoted as a percentage** — the amount by which a candidate
  model's prediction exceeds the measurement, as a fraction of the prediction.

**Statistics.** Medians over repeats, with 95% confidence intervals from 10,000
bootstrap resamples. Intervals are reported wherever a claim depends on the size of
a difference, because §4.7 establishes that some cells are far noisier than others.

**Models.** Qwen3-4B is the primary model; SmolLM2-1.7B (head dimension 64,
multi-head attention) and TinyLlama-1.1B (head dimension 64, grouped-query
attention at 8:1) are used for the architecture contrast.

**Traceability.** Every run writes `meta.json` before doing work, records the
configuration hash, git SHA and dirty flag, appends to a manifest, and is never
overwritten. Numerical claims are tied to run identifiers and recomputed from
captured data by `scripts/paper_numbers.py`; `./reproduce_all.sh` regenerates every
number and figure and exits non-zero if any disagrees.

---

## 3. Three quantized dimensions

From `tpu_inference/runner/tpu_runner.py:2133` and `runner/utils.py`, per step:

Table: The three quantized dimensions of the TPU serving stack, read from source. {#tab:dims}
| | quantizes | ladder |
|---|---|---|
| **D1** | prompt length → prefill shape | *does not exist* |
| **D2** | scheduled tokens / step | `[16, 32, …, 8192]` |
| **D3** | requests / step | `[8, …, 256]` non-attention; **`[256]` attention** |

The three are measured separately throughout, and results from one are not used to
support claims about another. §6 records one occasion on which that rule was
broken.

---

## 4. Results

### 4.1 Request-dimension padding is free, and the mechanism is the data structure

`envs.ATTN_BUCKETIZED_NUM_REQS` defaults to `False`, and when it is off,
`get_attn_req_paddings` returns `[max_req_size]`, a single bucket. Every boot
prints both ladders, and they disagree:

```
Prepared request paddings:      [8, 16, 32, 64, 128, 256]
Prepared attn request paddings: [256]
```

##### Executed request-slot count

The attention kernel therefore executes at a fixed size of 256 request slots,
independently of the batch size. Three measurements establish that this padding
carries negligible cost.

First, a paired experiment: enabling the flag compiles the full ladder, verified in
the warmup log, and changes decode by 0.0%, identical to 0.1 ms at n=8 and n=9.
This comparison excludes a large effect and supports nothing finer, since the
paired difference is +0.00 ms with a 95% bootstrap interval of [−10.7, +10.8] ms at
n=8, or ±20% of the decode phase.

Second, the compiler names the padding. The decode attention kernel is emitted as
`RPAd-p_256-bq_1_1-bkv_8192_8192`, writing the 256-request padding into the name of
the emitted kernel independently of the batch size.

Third, and decisively, an operator-level measurement. With prompt and output length
fixed, so that per-sequence key–value state is constant, attention device time
aggregated over a full 64-step generation is:

Table: Attention device time aggregated over a 64-step generation, at fixed per-sequence key-value state. {#tab:attn}
| n | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| attention (µs) | 16 830 | 26 011 | 45 919 | 85 103 | **163 535** |
| per real request | 16 830 | 13 005 | 11 480 | 10 638 | 10 221 |

These figures alone do not discriminate between the hypotheses. The apparent
argument — that a kernel doing work for 256 padded slots would be flat in n, that
this is not flat, and therefore that padded slots are skipped — fails because
flatness was never the alternative. Padded slots hold no key–value blocks, so no
kernel could do work proportional to their state. The only cost padding can
plausibly carry is a fixed per-slot overhead: walking 256 block-table entries,
loading 256 metadata records, launching tiles across 256 slots regardless of
occupancy. Such a cost is constant in n by construction. Fitting `T = a·n + b` through the endpoints ([tab:attnfit]):

Table: Fitting the attention table to a linear model with a fixed term. {#tab:attnfit}
| n | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| measured | 16 830 | 26 011 | 45 919 | 85 103 | 163 535 |
| 9 780·n + 7 050 | 16 830 | 26 610 | 46 171 | 85 292 | 163 535 |

Residuals are under 2.3%, and the fixed term is 7 050 µs, or 42% of attention time
at n=1, falling to 4% at n=16. If that term were a 256-slot padding cost it would
amount to 27.5 µs per padded slot. The table is equally consistent with padded
slots being skipped and with a fixed cost being paid for all 256, and so cannot
establish either.

##### Discriminating experiment: reducing the slot count

The discriminating experiment reduces the compiled slot count and observes the
fixed term. Enabling `ATTN_BUCKETIZED_NUM_REQS` compiles attention at 8 slots
rather than 256, a 32-fold reduction, profiled in both arms at the same real batch
size:

Table: Attention device time with the compiled slot count reduced 32-fold. A per-slot padding cost would fall by about 97%; it falls by 2%. {#tab:slotcut}
| n | flag off (256 slots) | flag on (8-slot ladder) | change |
|---|---|---|---|
| 1 | 16 898 µs | 16 749 µs | **−0.9%** |
| 2 | 26 130 µs | 25 999 µs | −0.5% |
| 8 | 85 081 µs | 85 055 µs | −0.0% |
| fitted fixed term `b` | 7 158 µs | 6 991 µs | **−2%** |

A per-padded-slot cost would be expected to fall by approximately 97% when the slot
count is reduced by a factor of 32. The measured reduction is 2%. The fixed term
therefore represents block-table and dispatch overhead rather than padding. Stated
as the bound the experiment supports: removing 248 of 256 slots moves the fitted
fixed term by 167 µs, so a padded request slot costs **under 0.7 µs**, against the
27.5 µs per slot that fully-paid padding would imply. This is a factor of 41, and
it is a bound rather than zero: at 256 slots the residue is up to approximately
170 µs, about 1% of attention time at n=1.

This measurement also runs the high-power version of the paired experiment above.
That test used n=8 and n=9, where a fixed 256-slot cost would have been about 8% of
attention time; at n=1 it would be 42%, and at n=1 the measured difference is
−0.9%. The 0.0% result was not a low-power test concealing an effect, because the
effect is absent where it would have been largest.

##### Testing the memory-bandwidth account

The memory-bandwidth explanation is tested and rejected. A natural account of
free request padding is that the step reads the whole weight set regardless of
batch, so padded slots are absorbed into a fixed cost that batch size does not
change. That account predicts utilisation climbing toward the compute roof past the
arithmetic-intensity ridge near 240 FLOP/byte, reaching a model FLOPs utilisation
(MFU) of approximately 49% at n=256. The measurement, with `prompt_len=256` and
`output_len=64`:

Table: Utilisation against batch size, testing the memory-bandwidth account of free request padding. The account predicts about 49% MFU at n=256. {#tab:roofline}
| n | 1 | 8 | 32 | 64 | 256 |
|---|---|---|---|---|---|
| MFU | 0.3% | 1.7% | 3.6% | 4.4% | **5.1%** |
| high-bandwidth memory (HBM) utilisation | 61.4% | 52.1% | 31.2% | 21.4% | **11.1%** |

At n=256 the requests are not all resident, so that column is queue-contaminated.
It is retained because it is the batch size the prediction names, and the
refutation below is stated at n=64, where the queue is 0.1 ms.

MFU here is `2 × parameters × tokens` against the chip's 197 TFLOP/s bf16 peak,
with attention FLOPs excluded, which is the same dense weight-stationary accounting
used for the intensity argument. Two caveats apply. MFU during memory-bound decode
is close to a tautology, since it restates step time against a fixed numerator; and
it is used here only to falsify a prediction expressed in the same units, not as a
figure of merit.

A memory-bound step is one whose achieved bandwidth sits near the roof. This one
falls monotonically to 21.4% by n=64, where the queue is 0.1 ms and the column is
clean, so the account fails without recourse to the contaminated cells. Neither the
compute-bound nor the memory-bound account is supported. One use of the roofline
remains valid, namely byte accounting: the step reads 2.01 GB of weights regardless
of batch size. Achieved bandwidth, however, is computed as bytes divided by
measured time, and therefore restates the step time from which it is derived.

##### Promotion cost at the 8-to-16 request edge

No promotion cost exists at the 8→16 request edge under the default
configuration, because the attention kernel is already compiled at 256 slots and
does not distinguish the two batch sizes.

### 4.2 Token-dimension padding is paid, and its share falls with batch size

Token padding is a different quantity with a different mechanism. A prefill step
carries hundreds to thousands of tokens, and padded tokens are real floating-point
operations, because the kernel computes them. Token padding is therefore paid
wherever the step's cost is dominated by arithmetic on those tokens.

[tab:paidraw] reports the share of nominal padding paid, straddling a compiled
boundary at fixed batch size and near-fixed sequence length:

Table: Share of nominal token padding paid, by batch size. Rows are confounded: each was measured over a different set of ladder boundaries. {#tab:paidraw}
| batch size | median | mean | 95% CI over boundaries | boundaries | clean dispatches/arm |
|---|---|---|---|---|---|
| 1–2 | **~85%** | — | *not computed* | 1 | see below |
| 4 | 23.1% | 20.2% | [13.5%, 24.4%] | 4 | 9–15 |
| 8 | 14.3% | 11.8% | [0.2%, 21.0%] | 3 | 3–5 |
| 16 | **−2.7%** | −5.9% | [−15.4%, +0.5%] | 3 | 7–11 |

##### Confounding across ladder boundaries

These rows are confounded, because each was measured over a different set of
ladder boundaries. At fixed n=4 the paid share rises with boundary size, from
10.0% at 512→1024 to 24.8% at 4096→8192, so the particular boundaries a row
contains shift its value independently of batch size. The sets differ: n=8 lacks
512→1024, the lowest-paying boundary, and n=16 lacks 4096→8192, the highest. Both
omissions push in the same direction and exaggerate the decline. This is the error
class §6 identifies as this work's most frequent, occurring here in a headline
table.

Restricted to the two boundaries present in every row ([tab:paidmatched]):

Table: Paid share restricted to the two boundaries present in every row. {#tab:paidmatched}
| n | 1024→2048 | 2048→4096 | mean |
|---|---|---|---|
| 4 | 22.1% | 24.0% | **23.1%** |
| 8 | 0.2% | 21.0% | **10.6%** |
| 16 | −2.7% | +0.5% | **−1.1%** |

The decline remains after matching, though the n=8 mean falls from 14.3% to 10.6%.
An interval bootstrapped over two boundaries would not be meaningful, so the
matched table establishes the ordering of the rows rather than their absolute
values. The defensible statement is ordinal rather than monotone: the paid share is
substantial at n≤2, intermediate at n=4–8, and indistinguishable from zero at n=16,
with only the n=4 against n=16 contrast separated at interval level.

The n≤2 row rests on a single boundary, which is not one of the matched pair, and
carries no interval. It is simultaneously the largest figure in the paper and the
least well supported. The n=8 value of 14.3% excludes split dispatches; including
them pools partial steps into the median and yields 16%, and the exclusion is the
correct treatment.

##### Measurable range and the request launcher

The upper limit of what can be measured was partly a property of the measurement
apparatus. A naive launcher makes this quantity appear unmeasurable above n=8,
because the scheduler splits every dispatch. Splitting tracks request count rather
than token count — a dispatch of 8192 tokens at `max_num_batched_tokens=8192` never
splits, while one of approximately 1024 tokens at n=8 splits half the time — which
is the signature of an arrival race rather than a capacity limit. Releasing all
requests from a thread barrier after connection setup reduced arrival spread by a
factor of 7.6 at n=32, from 15.4 ms to 1.7 ms:

Table: Fraction of dispatches the scheduler splits, before and after synchronising request release. {#tab:splits}
| n | split under the old launcher | split under a synchronised launch |
|---|---|---|
| 4 | 0% | 0% |
| 8 | 20% | **0%** |
| 16 | 100% | **60%** |
| 32 | 100% | 100% |

The real barrier lies between 16 and 32 rather than at 8. Under a synchronised
launch n=16 becomes measurable, and the paid share there is indistinguishable from
zero across three boundaries that each double the padded token count.

##### Paid share against boundary size

The share paid rises with the size of the boundary. At fixed n=4 it is 10.0% at
512→1024, 22.1% at 1024→2048, 24.0% at 2048→4096 and 24.8% at 4096→8192. Batch size
is held constant across these figures, so the comparison is between boundaries and
not between load points.

**[Figure 2 — `figures/fig2_padding.png`]** *Share of nominal padding actually paid
at each compiled boundary, against the 100% the compiled-shape premise predicts.*

##### Per-request length padding

Per-request length padding does not exist. Holding batch size and total tokens
fixed and varying only the spread of request lengths, the batch-padding model is
rejected in every ragged cell by at least 44%, and cost tracks packed tokens
instead. The rejection range extends to 618%, but a span of more than an order of
magnitude is a directional refutation rather than a measured effect size, so the
minimum is the figure that carries the claim. Uniform controls, in which all
candidate models agree, match to 1.9%. This is not an artefact of chunked prefill:
with `--no-enable-chunked-prefill` the result is unchanged, the packed model winning
8 of 10 ragged cells and batch padding being rejected by 75–579%.

##### Padded share under four length distributions

Padded share is governed by where a distribution's mass falls relative to the
compiled boundaries, and not by its dispersion. Four prompt-length distributions
were served at a single Poisson arrival rate (8 req/s, `output_len=64`, 120
requests each), reporting time to first token (TTFT) and inter-token latency (ITL).
Their parameters were solved so that every family has the same mean length of 1000
tokens, so equal request rate means equal offered tokens and distribution shape is
the only quantity that differs:

Table: Padded share under four prompt-length distributions matched on mean length, so that offered tokens are equal across arms. {#tab:dists}
| length distribution | CV | padded share | TTFT p50 / p95 | ITL p50 / p95 |
|---|---|---|---|---|
| fixed-1000 | 0.00 | **8.5%** | 30.0 / 147.5 ms | 4.9 / 9.0 ms |
| uniform | 0.58 | 27.4% | 45.6 / 74.7 ms | 5.3 / 7.0 ms |
| lognormal | 0.91 | **33.3%** | 43.6 / 253.3 ms | 5.1 / 10.2 ms |
| bimodal | 1.51 | 28.0% | 17.1 / 114.2 ms | 4.6 / 7.2 ms |

The padded share spans 8.5% to 33.3%, a factor of four, at equal offered tokens.

##### Dispersion as a predictor of padding

Coefficient of variation does not order it. The three dispersed families run CV
0.58, 0.91 and 1.51 against padded shares of 27.4%, 33.3% and 28.0%, which is not
monotone. Raggedness is therefore not the variable that governs padding, which is
the assumption the bucketing literature rests on. What governs it is where the
distribution's mass sits relative to the compiled boundaries: the fixed-length arm
pads least because 1000 tokens sits just below the 1024 entry, so little is rounded
away, and the same family would pad far more at 1100.

##### A retracted ordering

A prediction registered before this measurement was wrong, and its failure
retracts the explanation that motivated it. The same four families measured
without matching offered tokens — mean lengths of 256, 384, 704 and 2056, so the
uniform arm carried about eight times the tokens of the fixed arm — placed the
fixed-length family highest, at 51.0% against 27.3% for uniform. The explanation
offered for that ordering was that a fixed length just above a boundary pads every
step by the same large amount while a spread distribution averages across buckets,
and the prediction registered here was that the ordering would survive matching. It
does not: at equal offered tokens the fixed family pads least of the four. The
ordering followed from unequal token load, and the explanation built on it does not
stand.

This is the conclusion of §4.3 reached from the other direction. There, ladder
*placement* rather than shape count determines what a ladder buys; here,
distribution *position* rather than dispersion determines what a workload pays.
Both identify the same governing quantity: where lengths fall relative to
boundaries.

##### Recoverable headroom, reported as two factors

In place of a recoverable-headroom figure we report its two factors separately.
Those are the padded share of executed tokens, which the table above establishes
for four synthetic families and which remains workload-specific, and the paid share
at a given batch size,
reported with intervals earlier in this section. Multiplying them yields a headroom
figure of roughly 4–9% of execution, and that product is invalid: the paid share
moves with batch size, so a product formed from one workload's padded share and one
batch size's paid share describes no configuration that was run.

### 4.3 Ladder design is placement, not cardinality

Token padding is the one dimension on which the premise survives, so it is the
dimension an intervention should target. `VLLM_TPU_BUCKET_PADDING_GAP` provides the
lever: unset, the TPU backend compiles ten exponentially spaced token shapes; set
to 512, it compiles twenty-one linearly spaced ones. No patch is required.

To separate the effect of the ladder from differences between server instances,
the design includes control prompt lengths. Of four prompt lengths, two pad to the same
compiled shape on both ladders and two pad to a smaller shape on the fine ladder
only, so the placebo cells measure everything that differs between two server boots
except padding. Qwen3-4B, `v5litepod-4`, TP=4, two concurrent requests,
`output_len=32`, 18 repeats per cell pooled over both arm orders:

Table: Ten-shape against twenty-one-shape ladder at two concurrent requests. Prompts 300 and 600 pad identically on both ladders and act as controls. {#tab:ladder21}
| prompt | pads to (10 shapes) | pads to (21 shapes) | tokens saved | 10-shape | 21-shape | difference [95% CI] |
|---|---|---|---|---|---|---|
| 300 | 512 | 512 | 0 | 149.8 ms | 150.6 ms | +0.8 ms [+0.1, +1.6] |
| 600 | 1024 | 1024 | 0 | 168.5 ms | 168.6 ms | +0.1 ms [−0.5, +0.6] |
| 1200 | 2048 | **1536** | 512 | 206.0 ms | 188.2 ms | **−17.9 ms** [−18.2, −17.5] |
| 3000 | 4096 | **3072** | 1024 | 289.9 ms | 253.7 ms | **−36.2 ms** [−36.5, −36.0] |

The two treated cells agree on cost per padded token to within 1%, at 34.9 and
35.3 µs, while the two placebo cells differ by +0.5 ms in total. This agreement is
the principal evidence for the effect. An arm-level offset, such as one server
instance running uniformly slower, would produce a constant difference between the
arms and therefore a per-token figure inversely proportional to the number of
tokens saved; a padding effect scales with tokens saved, which is what the
measurements show. For scale, a real prefill token costs 46.6 µs on the same arm (1200→3000 tokens
for 83.9 ms),
so at two concurrent requests a padded token costs about three quarters of a real
one. Expressed as percentages, the twenty-one-shape ladder reduces end-to-end
latency by 8.7% at prompt 1200 and 12.5% at prompt 3000.

##### Cost of the finer ladder

The cost of the finer ladder is startup and memory headroom, not cache
capacity. Measured with the compiled-shape cache cleared before each boot:

Table: What the finer ladder costs: startup, resident executables, and the memory fraction at which the server will start. {#tab:laddercost}
| | 10 shapes | 21 shapes |
|---|---|---|
| cold warmup | 285 s | 436 s (+53%) |
| compiled-shape cache on disk | 43 MB | 92 MB |
| highest memory fraction that boots | 0.92 (the default) | **0.85** |
| KV cache at 0.92 | 367,360 tokens | *does not boot* |
| KV cache at 0.85 | 335,104 tokens | 335,104 tokens |

The warmup figures are cold. A persistent compilation cache amortises them across
restarts, and warm boots of the same two ladders measured 165–315 s, reflecting
cache reuse rather than ladder size.

At a memory fraction of 0.85 the two ladders report the same 335,104 tokens, to the
token. That equality also bounds any residual effect, since capacity is
block-quantized and the 49 MB difference in executables would amount to roughly
1,300 tokens, or 81 blocks, and would have been visible. What the finer ladder
costs is the backoff it forces: the ten-shape ladder runs at the 0.92 default for
367,360 tokens, the twenty-one-shape ladder does not start above 0.85, and the
difference between those operating points is 32,256 tokens, or 8.8% of capacity.

That figure required measuring the cliff rather than assuming it. Configuring
twenty-one shapes triggers an out-of-memory error (`RESOURCE_EXHAUSTED`) during
warmup at memory fractions of 0.92, 0.90 and 0.88, and requires a reduction to 0.85
before the server starts. At 0.92 the allocation requests 32.50 MB against 12.40 MB
free, and the failure is byte-identical across two independently provisioned hosts.
A shortfall of 20 MB might suggest that a small reduction would suffice; it does
not, since the requirement persists through three successive reductions.

The mechanism is not steady-state competition between executables and cache, which
the identical capacities rule out. Every failed boot records a key–value cache size
before terminating — 358,144 tokens at 0.90, 348,928 at 0.88 — so the cache is sized
to fill the fraction first, and compilation then requests scratch memory against
what remains. Shape coverage is charged to transient compilation headroom that the
sizing step does not reserve, which is why the cost appears as a boot cliff rather
than as a smaller cache.

##### Separating placement from cardinality

The benefit, however, is bought by a different variable than the cost. The
padding-gap lever changes the spacing law and the shape count together, so the
measurements above attribute a benefit and a price to the same parameter without
separating them. Every cost scales with cardinality: warmup, resident executables,
and the headroom that sets the boot cliff. Two intermediate gaps separate them ([tab:placement]),
all measured at 0.92 in one session:

Table: Separating placement from cardinality. Shape count ranges from 10 to 21 while the benefit follows only whether an entry falls between the prompt and the next default entry. {#tab:placement}
| ladder | shapes | 1200 pads to | 3000 pads to | prompt 1200 | prompt 3000 |
|---|---|---|---|---|---|
| default | 10 | 2048 | 4096 | 215.4 ms | 297.2 ms |
| gap 1024 | 14 | 2048 | **3072** | 210.9 ms | **256.5 ms** |
| gap 512 | 21 | **1536** | **3072** | 188.2 ms | 253.7 ms |
| gap 2048 | 11 | 2048 | 4096 | 214.1 ms | 296.1 ms |

Against the ten-shape ladder, and correcting by the prompt-300 cell, the
fourteen-shape ladder is +0.2 ms at prompt 1200 and −36.0 ms, or −12.1%, at prompt
3000. It gains exactly where it places an entry the default lacks, at 3072, and
nowhere else: it has no entry at 1536, so prompt 1200 is unaffected. The
eleven-shape ladder places nothing new near either prompt and tracks the default at
both. Shape count ranges from 10 to 21 across these arms while the benefit depends
only on whether a boundary falls between the prompt and the next default entry.

##### Feasibility at the default memory fraction

The fourteen-shape ladder boots at the stock 0.92 with 367,360 tokens, the same
capacity as the default. The 8.8% is therefore the price of the twenty-one-shape
ladder rather than of the optimisation: a 12.1% reduction at prompt 3000 is
available at full memory, for the warmup of four additional shapes. The
twenty-one-shape ladder buys a further 8.7% at prompt 1200, and that increment is
what costs 8.8% of capacity and 53% more startup.

The recommendation is therefore not to compile more shapes but to **compile the
shapes the workload's prompt distribution straddles**. Cardinality is what the
stack charges for; placement is what latency responds to. A badly placed boundary
is an executable that the compiler pays to produce and the scheduler never uses.

### 4.4 The benefit does not decay with concurrency, contrary to a registered prediction

Because padding is paid at small batch and free at large (§4.2), the benefit of
§4.3 appears as though it should be conditional on load. That reasoning predicts a
crossing, and it was registered before measurement: the difference should shrink
monotonically and reach zero between n=4 and n=16. Sweeping concurrency from 1 to
16 on both ladders, in both arm orders, does not support it. Placebo-corrected
differences in milliseconds, negative where the finer ladder is faster:

Table: Placebo-corrected latency difference against concurrency, in milliseconds. Negative values favour the finer ladder. {#tab:conc}
| prompt | n=1 | n=2 | n=4 | n=8 | n=16 |
|---|---|---|---|---|---|
| 1200 | −9.9 | −18.0 | −8.9 | −16.2 | −23.0 |
| 3000 | −18.7 | −37.3 | −21.5 | −50.4 | −37.0 |

There is no crossing. End-to-end latency is 3.5% to 12% lower at every concurrency
sampled, the effect is non-monotone rather than decaying, and at n=16 it is larger
in absolute terms than at n=1.

##### Why the prediction failed

The reason the prediction failed is visible in what the stack executes. The
prediction reasoned about an individual request's padding shrinking as more
requests share a step. Under chunked prefill the scheduler admits requests in waves
and packs them, and the size of the packed step, rather than any request's length,
selects the compiled shape. Snapshotting the `iteration_tokens_total` histogram
around one n=16 cell shows a single repeat costing about three prefill steps,
landing in (256,512], (512,1024] and (2048,4096], alongside 93 decode steps of at
most 16 tokens. The two arms produce near-identical step distributions, differing
only in which compiled shape those steps round up to. Padding is therefore
transferred from individual requests to the packed step rather than eliminated.

##### Tension with the paid-share curve

This is in tension with §4.2, which finds the paid share falling to
indistinguishable from zero by batch 16. The two measurements differ in what is
held fixed: §4.2 varies batch size at a fixed boundary and attributes padding per
request, while this sweep varies offered concurrency and allows the scheduler to
choose step composition. If both are correct, the reconciliation is that
per-request padding vanishes while per-step padding does not, and that a ladder
acts on the second. Testing this requires padded tokens per packed step, which the
available instrument cannot supply: `iteration_tokens_total` bins on powers of two,
which is coarser than the ladder spacing under comparison. A step recorded in
(2048,4096] pads to 2048, 2560, 3072, 3584 or 4096 depending on where in that bin
it falls, and the two ladders differ precisely inside the bin.

##### A defect in the control cell

One design defect bounds these results. Prompt 300 was intended as a placebo at
every concurrency, on the reasoning that it pads to 512 on both ladders. The step
histogram shows this holds only while each request prefills in its own dispatch: at
n≥4 the packed step lands where the ladders differ, so the cell is treated rather
than inert. Its measured difference is correspondingly unstable across arm orders
at n=8 and n=16 (−6.0 against −27.7 ms, and −1.5 against −17.2 ms) where the
treated cells are stable. The n≥4 figures above are therefore floors on the effect
rather than unbiased estimates, and arm order is their only control.

The sign survives this; the magnitude does not. The placebo's spread across arm
orders at n=8 and n=16 is approximately 20 ms, comparable to several treated
differences in the same table, so no n≥4 magnitude is resolvable at interval level.
What is resolvable is direction: the raw differences at n=16 are −32.2 and −46.3 ms,
and subtracting even the most extreme placebo estimate observed leaves both
negative in both arm orders. The absence of a crossing is established at n=1 and
n=2, where the placebo is valid; above that it is an observation with a floor.

### 4.5 The gain is a latency reduction below saturation, not additional capacity

The results above are latencies at fixed, small concurrency, while the cost of a
longer ladder is denominated in cache tokens. A deployment choosing whether to
spend memory on shapes requires the conversion. [tab:load] sweeps offered load open-loop against both ladders, at the stock 0.92
fraction where both boot, with 60 requests per rate at prompt 3000:

Table: Sustained goodput and latency against offered load, both ladders at the stock memory fraction. {#tab:load}
| offered req/s | goodput (default → gap1024) | p50 ms | p95 ms |
|---|---|---|---|
| 2 | 2.30 → 2.30 | 286 → 246 | 504 → 318 |
| 4 | 4.59 → 4.59 | 326 → 256 | 758 → 496 |
| 8 | 9.03 → 9.08 | **974 → 529** | **1622 → 996** |
| 12 | 12.45 → 12.68 | 2088 → 1895 | 3471 → 3093 |
| 16 | 13.45 → 13.84 | 2571 → 2275 | 3570 → 3418 |
| 24 | **14.03 → 14.40** | 3026 → 2921 | 3881 → 3727 |

Both ladders knee between 8 and 12 req/s and saturate near 14. Sustained goodput
rises from 14.03 to 14.40 req/s, or 2.6%, so the padding removed was not entirely
slack. But 2.6% is far short of what "padded tokens are real arithmetic" implies
for a 25% reduction in the prefill shape, and it is not where the effect is
concentrated. The effect is concentrated just below the knee: at 8 req/s the
placement ladder is 46% faster at p50 and 39% faster at p95.

That shape is consistent with §4.2 and repairs part of the tension in §4.4. A
saturated server packs prefills to the `max_num_batched_tokens` budget, so padding
is amortised across a full step and the ladder has little influence, which is what
the paid-share curve predicts. Below saturation the steps are smaller and closer to
per-request, and the ladder entry accounts for most of the step's cost. §4.4's
persistent benefit was measured under burst arrival at fixed concurrency, which
loads the server differently from a steady arrival process at the same nominal
rate.

For the fourteen-shape ladder this trade is favourable, since it costs no capacity
at all: four additional compiled shapes reduce tail latency by up to 46% in the
regime most interactive deployments run in. For the twenty-one-shape ladder, which
costs 8.8% of capacity, the same arithmetic argues against it, since surrendering
8.8% of the cache to gain 2.6% in sustained throughput is an unfavourable exchange
at saturation.

### 4.6 Prefix caching moves the target the ladder is placed against

Prefix caching is disabled elsewhere in this work, and production vLLM enables it
by default. It removes already-computed prefix tokens from the prefill, so the step
lands on a different compiled shape than the request's length implies, which bears
directly on a recommendation about where to place entries.

Testing this requires a workload with a cacheable prefix. Requests built from
independent token sequences share nothing, so a cache cannot hit them, and an
experiment run against such a workload would report caching having no effect while
saying nothing about production. Here every request shares a fixed 2048-token
prefix and varies only its 952-token tail, which is the structure of a system
prompt or a few-shot preamble. Prompt 3000, two concurrent requests ([tab:apc]):

Table: Ladder placement with and without prefix caching, over a workload sharing a 2048-token prefix. {#tab:apc}
| ladder | caching | e2e | prompt tokens cached |
|---|---|---|---|
| default (10) | off | 292.5 ms | 0 |
| gap 1024 (14) | off | 256.6 ms | 0 |
| default (10) | **on** | 181.5 ms | +43,008 |
| gap 1024 (14) | **on** | 178.5 ms | +43,008 |

##### Effect of prefix caching on the placement benefit

The placement benefit falls from 35.9 ms, or 12.3%, to 3.0 ms, or 1.7%. A
prediction registered before measurement anticipated this, and for the stated
reason: with 2048 tokens cached the server prefills approximately 952, which pads
to 1024 on both ladders, while the two ladders differ only at 3072 against 4096.
The entry chosen in §4.3 is no longer the entry the step uses. The cached-token
counter is the arm's own evidence that the treatment applied, since 43,008 is
exactly 21 × 2048, so twenty-one of twenty-two requests hit the prefix and the
first populated it.

The mechanism is unchanged and the operating point has moved. Caching does not make
padded tokens cheap; it removes tokens from the prefill, and a ladder placed
against prompt length is then placed against the wrong distribution. **Entries
should therefore be placed against the distribution of uncached prefill lengths.**
This also bounds §4.3 and §4.5, which are measured with caching disabled and so
describe workloads with little prefix reuse. On this workload caching is worth
considerably more than the ladder: 292.5 ms to 181.5 ms, a 38% reduction, against
the 12.3% the best placement achieves without it.

### 4.7 A ladder can be chosen offline from a length distribution

§4.3's placement was chosen by knowing two prompt lengths in advance, which is an
existence proof rather than a method. A method begins from a length distribution,
selects a ladder without reference to latency, and is then correct.

Sampling 120 prompts from a lognormal distribution with median 1200 and σ=0.9,
replaying the same sampled lengths against every ladder so that arms are paired on
workload, and predicting each arm's latency from expected padded tokens multiplied
by the 35 µs per padded token measured in §4.3:

Table: Ladders selected offline from a lognormal length distribution, with predicted and measured latency. {#tab:fit}
| ladder | shapes | mean padded tok/req | predicted Δ | measured e2e | boots at 0.92? |
|---|---|---|---|---|---|
| default | 10 | 602 | — | 219.8 ms | yes |
| gap 1024 | 14 | 389 | −7.5 ms | **212.7 ms** (−7.1) | yes |
| gap 512 | 21 | — | −16.2 ms | *does not boot* | **no** |
| gap 256 | 35 | — | — | *does not boot* | **no** |

##### Predictive accuracy of the offline model

The offline model predicted the measured result to within 5%: 7.5 ms predicted
against 7.1 ms measured. Inverting it gives 33.3 µs per padded token against the
34.9–35.3 µs of §4.3, and the two workloads share nothing, since §4.3 used two
fixed lengths straddling known entries and this a heavy-tailed mixture over the
whole ladder. A constant fitted on one workload and confirmed on another supports
using the model for design.

Ladder design therefore does not require a hardware sweep: sample the length
distribution, compute expected padding per candidate ladder, and multiply by the
per-token cost.

##### Constraining the objective

The objective must be constrained, and the unconstrained objective is the premise
this paper refutes. Expected padding falls monotonically with shape count, so
minimising it alone drives the ladder toward the finest the stack can compile,
which §4.3 shows the stack cannot afford. Both finer arms failed to boot at the
stock memory fraction, the 35-shape ladder more decisively (23.75 MB requested
against 4.94 MB free) than the 21-shape one (32.50 MB against 12.40 MB). The
feasible set here was {10, 14} and the answer was 14. The configuration rule is to select the
greatest shape density that satisfies the memory constraint, with entries placed
against the length distribution.

One caveat bounds the model's reach. Predicting padding requires knowing the ladder
a gap will produce, and the rule is not the obvious one: the stack continues
doubling while the doubling step is no larger than the gap, then switches to linear
spacing. At gap 256 this inserts an entry at 768, which a reading of "powers of two,
then linear" does not predict. The predictions above were computed from the
corrected rule, and each arm re-reads the ladder its server printed.

### 4.8 Supporting ablations and controls

##### A published latency predictor

A published latency predictor, reproduced and scoped. LENS predicts NPU
inference latency to 2.15% mean absolute percentage error (MAPE) using a per-bucket
`intercept + slope × length` fitted from two end-to-end measurements per bucket.
Reproducing its protocol on TPU across 5 buckets × 3 batch sizes, with 7 repeats
per point and a mid-bucket point withheld from each fit, gives MAPE 5.23% and a
worst cell of 22.4%, near-perfect at n=1–2 and failing at n=4. Replacing the model
with a constant, the mean of the two calibration points:

Table: LENS against a constant-only model, held-out mean absolute percentage error. {#tab:lens}
| batch size | LENS | constant-only |
|---|---|---|
| 1 | 0.38% | 0.96% |
| 2 | 0.39% | 0.86% |
| 4 | 19.77% | **14.80%** |

At n=1–2 the within-bucket curve is nearly flat, at a flatness of 0.97, so any
two-point fit is near-perfect. LENS does beat a constant there, at 0.38% against
0.96%, but the gap is 0.6 percentage points on errors already below 1%, and it is
not evidence that the model form transfers. At n=4 the length term is actively
harmful. The failure is not an artefact of which points were fitted: over all three
choices per cell the n=4 error varies by up to 44.8 percentage points, but its
minimum is still 16.97%, far above LENS's reported 2.15%. Batch sizes above 4 were
not measured, because this experiment reproduces LENS's own protocol, which is
specified over bucket structure rather than batch size.

**[Figure 1 — `figures/fig1_lens.png`]** *Held-out prediction error against batch
size, with LENS's reported 2.15% as a reference line and the failure region shaded.
The claim is not that the predictor is inaccurate; it is that the error grows
sharply by n=4. Batch sizes 1, 2 and 4 are sampled, with the failure at the
endpoint, so these data cannot say whether accuracy returns above 4.*

##### Decode cost against batch size

Decode cost against batch size. With `prompt_len=256` and `output_len=64`,
measured across the full compiled request ladder:

Table: Decode cost against batch size across the compiled request ladder. {#tab:decode}
| n | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| ms/step | 4.02 | 4.91 | 9.19 | 15.32 | 27.90 | 51.83 |
| µs/step/sequence | 4020 | 614 | 287 | 239 | 218 | **203** |
| queue (ms) | 0.0 | 0.0 | 0.0 | 0.1 | 62.2 | 298.3 |

**[Figure 3 — `figures/fig3_decode.png`]** *Decode cost per sequence against batch
size, log–log.*

There are two regimes and the boundary between them is sharp. Below n≈8 the step
barely moves, with batch rising eightfold for 1.22 times the cost. Above n≈32 the
step is nearly linear in batch, at 1.67, 1.82 and 1.86 times for successive
doublings, and per-sequence cost flattens at roughly 200 µs. Queue time at n=128
and n=256 means those columns are not clean wide batches and no claim depends on
them.

##### Distribution of device time

Where device time is spent. An operator profile is a direct observation, which
the roofline is not. Share of TPU device time:

Table: Share of TPU device time by operator category. {#tab:ops}
| n | attention | collective | matmul/fusion |
|---|---|---|---|
| 1 | 6.8% | 13.5% | **78.5%** |
| 4 | 15.4% | 13.9% | 69.6% |
| 16 | 34.2% | 13.4% | **51.4%** |

Matmuls dominate at low batch and give way to attention as key–value state grows,
while collectives hold a flat 13.4% whose latency component no roofline models.
Nothing moves discontinuously at n=4.

##### An invalid microbenchmark

One microbenchmark does not measure the quantity it appears to. An isolated
matmul at the model's real sharded shapes returns 142.9 µs at M=1 and 143.6 µs at
M=256, a flatness that can be read as the weight-load floor with confounds removed.
The qkv projection holds 7.86 MB per chip, so the bandwidth floor is 9.6 µs: the
measurement sits fifteen times above it, at 7% of peak, and is timing per-dispatch
overhead. Amortising over a loop reports 1250% of peak because XLA hoists the
loop-invariant matmul; chaining iterations is physically valid at 79% of peak but
streams weights from HBM every iteration and remains bandwidth-bound.

##### Per-dispatch variance

Per-dispatch variance is a prefill phenomenon. Over 9 repeats on the same
server, decode spread is 1.00–1.04× at most batch sizes, while prefill is 1.00–1.03×
at n≤4 and 1.18–1.26× at n≥8. Variance appears exactly where the scheduler begins
splitting dispatches, and decode, which makes no chunking decision, never exhibits
it. This localises the effect without explaining it. The aggregate figure does not
describe every cell: bootstrapping the decode cells §4.1 depends on gives 95%
interval widths of 38.7% at n=8 and 28.2% at n=9 over 21 repeats, wider than
several differences a reader might otherwise treat as signal.

##### Sharding ablation

Free request padding is not an artefact of the sharding. Model, chips and
workload were held fixed while tensor-parallel degree alone was varied. The
prediction was registered before measurement: per-chip weight bytes scale as 1/TP,
so the level should scale with 1/TP and the shape should be preserved.

Table: Tensor-parallel ablation. Both halves of the registered prediction fail, and a single omitted fixed term explains both. {#tab:tp}
| TP | per-step level vs TP=4 | predicted | cost rise, n=1→32 |
|---|---|---|---|
| 4 | 1.00× | 1.00× | 2.33× |
| 2 | 1.63× | 2.00× | 2.41× |
| 1 | 2.86× | **4.00×** | **1.83×** |

Both halves of the prediction were wrong. The level scales sub-proportionally,
which the roofline cannot explain because it does not model the inter-chip
collectives the higher-TP arms pay, and the curve becomes flatter with less
sharding rather than preserving its shape. Both errors follow from a single omitted
term. Fitting `T(TP) = W/TP + F`, a weight load that shards plus a fixed cost that
does not, gives `T(TP) = 2.48/TP + 0.38` normalised to the TP=4 step, with fitted
values 1.00, 1.62 and 2.86 against measured 1.00, 1.63 and 2.86. Fixed,
non-sharding cost is 38% of the TP=4 step, roughly 1.5 ms at n=1, and that single
term explains both the sub-proportional level and the flattening.

`F` is not the collectives: a constant term cannot represent inter-chip
communication, which is zero at TP=1 and largest at TP=4 and therefore scales with
TP, whereas `F` by construction does not. With `T(4)=1` by normalisation this is
one free parameter against two informative points, so the fit quality is weaker
evidence than the residual suggests. The ablation nonetheless answers its question:
if request-dimension padding were cheap only because that dimension is not the
bottleneck at TP=4, reducing TP would expose it, and instead padding is cheapest at
TP=1 where the fixed cost is largest.

##### Model scale, and dtype as the lever

Model scale does not move the regime map. A second registered prediction failed
here. TinyLlama-1.1B has a 3.6-fold smaller per-chip weight floor than Qwen3-4B, so
if the mechanism were that floor its paid share should be higher. It is lower, at
−1.2% and 13.4% at n=4 and −1.1% and 5.9% at n=8. Model size is the wrong lever:
for dense weight-stationary decode, bytes ≈ 2·params and FLOPs per token ≈ 2·params,
so both models sit at 1.00 FLOP/byte/token and arithmetic intensity is the batch
size, independent of parameter count. The ridge is a property of the chip (v5e:
197 TFLOP/s ÷ 819 GB/s ≈ 240 FLOP/byte), and shrinking the model shrinks the floor
and the work in the same proportion. This answers the single-model limitation
analytically for that regime, and breaks for mixture-of-experts models, where bytes
scale with distinct experts touched, and at long context, where key–value state
rather than weights sets the floor.

Two cautions apply. The identity is derived for decode, while the paid-share
numbers are prefill, and applying it to them is the domain error §6 records as our
twelfth. The intensity argument also explains why model size should not move the
paid share, not why it moved downward; the likely reason is that non-weight fixed
costs form a larger fraction of a 0.55 GB model's step.

Dtype is the lever that should move the token dimension. W8 weights halve bytes and
leave FLOPs unchanged, doubling intensity per token and moving the crossing from
batch ≈ 240 to ≈ 120. That yields a registered prediction — under W8 the
token-dimension paid share at a fixed boundary rises — which **has not been run**.
It is future work rather than a result, and nothing above depends on it.

---

## 5. Five optimisations, four rejected and one that works

Table: Five optimisations designed against these measurements, and their outcomes. {#tab:opts}
| | outcome |
|---|---|
| **ladder placed against the workload** | **works: −12.1% at n=2 and −46% p50 below the knee, at full memory; −1.7% once prefix caching is on (§4.3–§4.7)** |
| bucket-aware admission control | premise false (§4.2) |
| ladder redesign on the request dimension | D1 does not exist; D3 inert by default |
| last-chunk decomposition | **20.6% worse** measured (51.06 vs 42.33 ms) |
| bucket-aligned step packing | implemented twice: inert, then output-corrupting |

The one that works targets the single dimension the measurements left open. Three
of the four rejected optimisations address the request dimension or per-request
length padding, and §4.1 and §4.2 show that neither carries cost; the fourth
restructures work the stack already packs. The token dimension is the only place
where the premise was not refuted by measurement, and the only place where an
intervention produced a gain — though not for the reason expected, since the payoff
was predicted to be confined to low concurrency and was not (§4.4).

A second positive measurement — release timing saving 26% of TPU time against stock
at 25 req/s (p=0.001, six paired seeds) — is dynamic batching rather than a shape
effect, and is reported as a re-measurement rather than as a contribution.

**It is a low-load effect that reverses.** Swept across arrival rate against stock,
positive being a saving:

Table: Release-timing policies against arrival rate, positive being a saving in TPU time against stock. {#tab:ratecurve}
| req/s | wait | hybrid | hybrid p95 |
|---|---|---|---|
| 10 | +36.2% | +29.0% | −32.6% |
| 25 | +22.2% | +22.8% | +28.9% |
| 40 | +7.1% | +17.6% | +21.5% |
| 55 | +7.2% | +11.6% | −6.6% |
| 70 | +2.3% | **−11.7%** | +18.7% |

The saving decays monotonically with load and becomes an 11.7% penalty by 70 req/s.
A single measurement at 25 req/s cannot distinguish that from a robust effect, and
it samples the most favourable region of the curve.

**It is also not free.** The harness scrapes `/metrics` around every batch, adding a
median 22.6–24.9 ms of inter-dispatch overhead to a policy that never waits. That
inflates stock's p95 from roughly 24 ms to 86 ms and conceals the delay the waiting
policy introduces deliberately. Measured under that harness the cost appears as
+14.8% p95 (p=0.570); simulated on an efficiently driven server the same policy
costs approximately +188% p95 at 25 req/s. The defensible statement is narrower:
the hybrid policy reaches nearly all of wait-to-fill's saving, at 30.2% against
31.9%, for a small fraction of its latency cost, at +188% against +1461%. The
result is a point on a cost–latency trade-off curve.

**The cost model does not survive a wider sweep.** Holding out rates it was never
fitted on gives 4.9% MAPE overall but a worst cell of 19.7% (hybrid at 55 req/s),
against this project's own 15%-per-cell rule, so it fails. The earlier holdout
varied neither rate nor prompt length and so could not detect an error constant
across them. The simulated policy numbers are internally consistent predictions
that do not transfer to unseen load.

Bucket-aligned packing requires a separate note. The second implementation measured
−29% TPU time and −49% p99; the correctness gate then showed that 4 of 48 greedy
completions differed, every one at a prompt length just above a bucket boundary.
The patch was silently dropping prompt tokens. Trimming `num_scheduled_tokens`
after the scheduling loop leaves the request's bookkeeping untouched, so deferred
tokens are skipped rather than rescheduled, and the step is cheaper because it does
less work.

---

## 6. Fourteen invalid inferences, and one asymmetry

Fourteen invalid inferences were made and caught during this work. They fall into
four classes, and the guardrails cover only some of them.

Table: Fourteen invalid inferences by class, and whether a mechanical check now covers each. {#tab:failures}
| class | count | what it is | covered? |
|---|---|---|---|
| **provenance** | 8 | a quantity measured under one configuration, used under another | **yes** — config-diff over registered claims |
| **instrument definition** | 4 | an analysis that measures something other than the target | no |
| **lever validity** | 1 | a lever that cannot move the quantity claimed | **yes** — `prediction_mechanism` |
| **dimension** | 1 | a result in one quantized dimension licensing a claim in another | **yes** — required `dimension` field |

**The provenance guardrail took three versions.** A rule forbidding derivations
that combine quantities measured at different batch sizes would not have caught the
`output_len` failure. A whitelist of configuration keys missed the largest error,
because batch size is not a top-level field and lives inside experiment-specific
structures. The working form diffs every configuration key across a claim's source
runs, exempts only free text, and requires each difference to be named. It flagged
five claims already believed correct, and has since invalidated two of our own
headline numbers: a crossover point and a recoverable-headroom figure.

**Its coverage is the set of registered claims, not the set of claims made.** The
headroom figure evaded it for three drafts by living in prose, and was caught only
when it was registered in order to be checked. §4.3's cache-capacity price, the
eighth provenance failure and the most recent, evaded it the same way: capacity at
memory fraction 0.92 was differenced against capacity at 0.80 and the gap
attributed to ladder length, in a table assembled by hand from two servers' boot
logs rather than by a script over registered runs. Boot-time facts do not flow
through the run-recording path, and nothing checks them.

**The four instrument-definition errors are not covered by anything.** A step-count
criterion that could never pass; a boundary experiment that pooled split dispatches
it claimed to exclude; a microbenchmark that timed dispatch overhead at 7% of peak
bandwidth and called it a weight-load floor; and §4.4's placebo, a control cell
chosen to be inert that ceases to be inert above n=2 once the scheduler packs
requests into shared steps. Each was caught by a measurement disagreeing with an
independent one, which is fortunate rather than systematic, and one of them, the
split pooling, biased upward, in the direction that manufactures a positive result.

The placebo failure is the clearest instance of the class, because the reasoning
behind it was checkable and was never checked. "Prompt 300 pads to 512 on both
ladders" is a statement about what the stack executes, and the stack exports what it
executes: one histogram scrape settled it in under a minute, after both arms had
already run. Nothing in the machinery asks whether a control is a control. The
check that would have caught it is cheap and does not exist: assert that the
executed shape distribution matches the one the design assumes, before treating a
cell as inert.

The last two classes each produced a mechanical check, and both are cheap: state
the target as a formula in the lever and show the derivative is nonzero; name the
quantized dimension a claim belongs to and reject derivations that cross one. Both
would have fired before hardware was provisioned.

### 6.1 Measurements outlast explanations

Counting the fourteen entries alone conceals a pattern: **the measurements have
survived four rounds of external review largely intact, and the explanations have
not.** Every headline measurement in §4 still stands as measured. What has been
withdrawn, in order, is a crossover rule, a recoverable-headroom figure, a
microbenchmark and the mechanism it claimed to isolate, a memory-bandwidth account
of free request padding, and the frontier bound derived from that account. Four of
the last five retractions were mechanism claims, and not one was a number that
failed to reproduce.

The asymmetry is structural rather than careless. Every measurement runs through a
contract that aborts on an unstated variable, is tied to a run identifier, and is
recomputed from captured data by a script that exits non-zero on disagreement. **No
comparable machinery exists for explanations.** A mechanism is prose: it can be
written, accepted, cited by later sections, and carried across drafts without ever
being evaluated. The bandwidth account survived four sessions and two reviews not
because evidence supported it but because nothing in the pipeline could reject it.

The registered-prediction discipline is the closest available remedy, and its
record is instructive. Three predictions failed — the sharding ablation, the model-
scale ablation and the concurrency sweep — and each failure was more informative
than a confirmation would have been. One produced a calibrated two-term cost model,
one established that the regime map is independent of parameter count, and one
identified that padding migrates to the packed step under chunked prefill. A
mechanism that never generates a falsifiable number does no work, and this work
published three such mechanisms before withdrawing them.

We report this as a finding rather than as a methodological remark, and §1 states it
before any result for that reason. A reader who accepts it will assign greater
confidence to the measurements of §4 than to its explanations, which is the response
the evidence warrants.

---

## 7. Limitations

**One TPU slice, one GPU, one primary model.** A v5litepod-4 with a 4B model, and a
single L4 for the control. The sharding objection is answered by the TP=1/2/4
ablation (§4.8), which finds request padding cheap at every sharding, but model
scale and multi-host topology are unmeasured, and both change the fixed costs
within which padding can be absorbed. §4.8 argues analytically that the regime map
is a function of batch size and dtype rather than parameter count, for dense
weight-stationary decode only.

**The strongest single number is the least well supported.** The ~85% paid share at
n≤2 rests on one boundary and carries no interval (§4.2).

**Prefix caching is measured but not swept.** §4.6 tests it at one prefix length,
2048 of 3000 tokens, on one workload shape. This is enough to show that the
placement target moves and not enough to say where it lands for a given amount of
prefix reuse. Everything outside §4.6 is measured with caching disabled and so
describes workloads with little prefix sharing.

**No production trace.** §4.2's four length distributions are parametric families
matched on mean length, not a trace of real traffic. They establish that padded
share varies by a factor of four with distribution shape at equal offered tokens,
and that dispersion does not order it, but the figures belong to those families
rather than to any deployment. §4.6 further shows that any such figure depends on
how much prefix reuse a workload has, since caching changes the length actually
prefilled.

**The ladder benefit has no measured upper bound in concurrency.** §4.4 sweeps 1 to
16 and finds no crossing, so we cannot say where a finer ladder ceases to pay, only
that it has not ceased by 16. Above n=4 that experiment has no valid placebo, so
those rows are floors on the effect rather than unbiased estimates.

**Prefill step cost above n=16 is not isolable**, and at n=16 the clean sample is
7–11 dispatches per arm.

**The n=4 convergence is unexplained and is not an operator effect.** An
operator-level profile shows every category of device time moving smoothly through
n=4. The three observations that appear to converge there are not independent:
LENS's failure and the paid-share drop are the same quantity described twice.

**The GPU control is one point rather than a curve.** Startup was measured at
vLLM's default capture set, and the number of captured shapes was not varied, which
is precisely the axis BucketServe and LAPS trade along.

**Co-located prefill and decode only.** On a disaggregated deployment the padding
question divides in two, and §4.8's finding that variance is a prefill phenomenon
is the asymmetry that motivates disaggregation.

---

## 8. Related work

**Pope et al.** derive the memory-bound to compute-bound transition for transformer
inference on TPU analytically; §4.1's byte accounting is a measured instance of that
regime rather than a discovery. We do not claim that the weight-load floor explains
free request padding, since §4.1 tests that account and rejects it.

**RPA** is the technique this work validates: per-request padding costing nothing is
what its ragged-tiling design predicts, though it does not discuss the request-count
dimension, report cost against batch size, or quantify how much padding survives it.
**LENS** (§4.8) supplies the model form; we supply the hardware it was not tested on
and the ablation showing that its length term is not what carries its reported
accuracy.

**PagedAttention/vLLM** is the stack measured throughout; **Orca**'s iteration-level
scheduling produces the per-step batches; **Sarathi-Serve** introduced the chunked
prefill that §4.2 controls for. **DistServe** and **Splitwise** disaggregate prefill
from decode, which is the architectural response to the finding that variance is a
prefill phenomenon, and which bounds our advice to co-located deployments.
**Vidur** established simulator-fidelity validation as the standard for
simulation-based serving studies; our holdout discipline follows it.

**BucketServe** and **LAPS** manage length-bucketing overhead on GPU, and we
measured the comparison rather than asserting it. Same vLLM 0.25.0, same instrument,
an L4 (23 GB, TP=1) in place of the v5e:

Table: GPU control on an L4: CUDA-graph capture against eager execution. The increments are the measurement; the levels differ by a constant launch overhead. {#tab:gpu}
| arm | n=8 | n=9 | n=16 | 8→9 | 8→16 | startup |
|---|---|---|---|---|---|---|
| CUDA graphs on | 10.605 | 10.887 | 12.298 | **0.283** | **1.693** | 118.7 s |
| `--enforce-eager` | 19.934 | 20.150 | 21.618 | **0.215** | **1.684** | 10.7 s |

**The increments are the measurement; the levels are not.** Eager execution pays a
constant per-operation launch overhead, so the levels differ by roughly 9.3 ms
throughout, and that constant cancels in any increment. It cancels to within 9 µs on
the 8→16 step, at 1.693 against 1.684 ms, which is the control this comparison
requires: the two arms measure the same underlying work plus an offset, so their
difference isolates what capture adds.

On that basis, padding a batch from 8 up to the captured entry at 16 costs
approximately **67 µs**, being the 0.283 ms increment with graphs against 0.215 ms
without. That is 0.6% of a 10.9 ms step, or 4.0% of the nominal padding implied by
rounding 9 up to 16.

**This tests one of the three dimensions, and it is the one least in doubt.** vLLM's
CUDA path captures a graph per batch size, so what these arms vary is the request
dimension, which the TPU results explain with a data-structure argument (§4.1) and
where no proposed optimisation was going to pay. The token dimension, where this
paper locates the only surviving effect, is not reached: no arm here varies tokens
per step at fixed batch. The cross-architecture statement this table supports is
therefore that **request-dimension padding is close to free on both architectures**,
not that the premise is false on both. Whether a GPU stack pays for token padding as
a TPU stack does is untested, and is the experiment that would make the
cross-architecture claim general.

The comparison is also a bound rather than an equality. §4.1's TPU statistic carries
intervals of roughly ±50 percentage points, so this table cannot resolve a
difference between the architectures; what both support is that the paid share is
small on each. These remain single measurements: three batch points, one GPU, no
repeats.

**What is paid is the capture, and it is paid at startup.** Enabling graphs costs
108 s of initialisation, 118.7 s against 10.7 s, for a capture set fixed in advance.
That is precisely the quantity BucketServe and LAPS manage when they write that the
number of graphs must be limited, and it is a warmup cost. The TPU analogue is XLA
compilation: 5–30 minutes for the first bucket and 30–120 s per additional one.

**How, then, do prior bucketing techniques achieve their reported speedups?** If
the padding premise is false on
both architectures, systems reporting end-to-end improvements from bucketing are
improving something else, and §5 supplies a candidate from our own data. A positive
measurement in this work — release timing saving 26% of TPU time at 25 req/s — is
dynamic batching, an arrival-and-composition effect rather than a shape effect.
Bucketing schemes change which requests occupy a step together, and that is worth
something independent of padding. We place weight on the mechanism rather than the
magnitude, because the magnitude is the weaker half: the saving costs latency once
the harness overhead inflating the baseline is removed, and it reverses at high load
(§5). The defensible claim is the negative one — **a bucketing result that does not
control for batch composition cannot distinguish a shape effect from a scheduling
one** — and it rests on the premise measurements of §4.1 and §4.2.

---

## 9. Conclusion

Two accelerator families reach the same design by different routes: XLA compiles a
ladder of shapes, and CUDA captures a graph per batch size. Both round every step up
to the nearest entry, and the optimisation literature treats that rounding as a cost
to be recovered. It is not, on either. A batch just above an entry costs what the
entry below costs, because a ragged attention kernel does almost no work for slots
holding no key–value blocks — under 0.7 µs per slot, against 27.5 µs if it were paid
— and a captured graph does not care that part of its batch is unused.

What shape coverage costs is warmup. Enabling CUDA-graph capture costs 108 seconds
of startup, and XLA compiles the first TPU bucket in 5–30 minutes. That is the
quantity BucketServe and LAPS manage when they write that the number of graphs must
be limited, and it is a startup and memory-footprint budget rather than a throughput
one. Reducing the number of shapes is worth doing for time-to-serve and resident
executables. Routing requests to avoid run-time padding is not worth doing.

The exception is the token dimension, where padded tokens are real arithmetic: the
paid share is 23.1% of nominal at batch 4, falls to indistinguishable from zero by
16, and is around 85% at batch ≤2 — a figure resting on a single boundary with no
interval, and the least well supported number here. That low-batch regime is
interactive serving, tight-latency deployments, and the prefill half of any
disaggregated system, and it is where the ladder buys something.

**The variable that pays is placement, not shape count.** A fourteen-shape ladder
adding a single entry the default lacks reduces end-to-end latency by 12.1% at the
prompt length that straddles it, gains nothing at a length it does not, and boots at
the stock memory fraction with unchanged cache capacity. A twenty-one-shape ladder
achieves the same reduction and will not start above a memory fraction of 0.85,
costing 8.8% of cache capacity and 53% more startup. Every cost measured scales with
cardinality. The ladder can be chosen offline: expected padding computed from a
length distribution predicted the measured reduction to within 5%, so the design
rule is the finest ladder that still boots, with entries placed against the
distribution the server actually prefills.

Two conditions bound that advice. The gain is a latency reduction below saturation
rather than additional capacity: median latency falls 46% just below the knee,
while sustained throughput rises 2.6% at it. And prefix caching, which production
vLLM enables by default, reduces the gain from 12.3% to 1.7% by shortening the
prefill onto a different compiled entry, so entries must be placed against uncached
prefill lengths rather than prompt lengths.

Three predictions registered before measurement failed, and each failure was worth
more than a confirmation. That record is the paper's methodological result: across
four rounds of review every headline measurement survived and four of the last five
retractions were claims about mechanism, because a measurement passes through
machinery that can reject it and an explanation does not. The remedy adopted here is
to require that a mechanism emit a falsifiable number before hardware is
provisioned. It is cheap, and it is the practice we would most recommend carrying
into other measurement work.
