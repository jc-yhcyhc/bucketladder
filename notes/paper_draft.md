# What Compiled-Shape Padding Actually Costs in Production TPU Serving

### A measurement study: 36% of executed tokens are padding, and ~92% of that is free

**Draft — 2026-08-10.** Target: **MLSys 2027 Industrial Track**. Confirmed from the
2026 call: *"No requirement for novelty or new methods,"* and the track explicitly
invites *"submissions that challenge or reinforce existing solutions, provide deeper
insights into known problems, or rigorously validate published techniques in a
real-world setting."* 10 pages excluding references. MLSys 2026's industrial
deadline was 30 Oct 2025, so the 2027 equivalent is expected **late Oct 2026**; the
2027 call is not yet posted. Backup venue still unchosen.

Stack under test: vLLM 0.25.0 + `tpu-inference` 0.25.0, JAX 0.10.2, libtpu 0.0.42.1,
on `v5litepod-4` (4 chips, TP=4). Total measurement cost: **[redacted]** across ten
hardware sessions.

---

## Abstract

TPU executables are compiled for fixed tensor shapes, so a serving stack must round
every workload up to one of a precompiled ladder. The natural inference — that
rounding up means paying for what you rounded up to — motivates a family of
proposed optimisations: length bucketing, shape-aware admission control, ladder
design. We measure what is actually paid in a production TPU serving stack and find
that **most of it is not**.

Shape quantization here is three-dimensional. Per-request sequence padding **does
not exist**: a batch of mixed-length requests costs its packed tokens, and rejecting
the per-request-padding model by 44–618% does not depend on chunked prefill. The
attention kernel's request dimension is **pinned to a single compiled bucket by
default configuration**, so batch size never changes the attention shape at all.
The step's token count *is* padded and *is* paid — but only at batch size one,
which is not a serving regime.

Over 150 instrumented dispatches, **35.9% of executed tokens are padding** (p95
per-dispatch ratio 99.6%). Randomised straddles that double the padded token count
while moving real work by <2% show the share actually paid rising with the
boundary: **10% at 512→1024, ~24% at 1024→2048 and above**, against the 100% the
compiled-shape premise predicts. Enabling the attention request ladder that the
stack disables by default changes decode latency by 0.0%.

The practical consequence follows from one further measurement rather than from a
patch. A scheduler step carries a **6.11 ms fixed cost**, and paid padding scales
with the bucket while that cost does not, so the two cross at ~2048 tokens:
deferring work to align a step **loses by 4.7× at the 512→1024 boundary** and wins
only above 2048. For the small steps that dominate decode-heavy serving the advice
inverts — reduce step count and tolerate padding.

We report the mechanism for each dimension read from the serving stack's own
source and confirmed on hardware, a cost model validated on two independent
holdouts (2.9% and 3.6% MAPE), a provable optimality bound for the scheduling
policies the model admits, and two invalid inferences we made and caught — both
instances of measuring a step-scoped property with a request-scoped instrument.

---

## 1. Introduction

A GPU serving stack launches kernels whose shapes are resolved at runtime. A TPU
serving stack cannot: XLA compiles for fixed shapes, and recompiling per request is
impossible at serving latencies. vLLM's TPU backend therefore precompiles a ladder
of shapes and rounds every step up to the nearest.

This is a quantization, and quantizations usually cost something. The literature
assumes it does. **BucketServe** derives an optimal length-bucket boundary and then
declines to compute it, calling it "computationally expensive to calculate in
practice". **LAPS** captures a CUDA Graph per `(length, batch)` cell, pads to the
nearest, and notes that "the number of graphs must be limited". Both are GPU work,
and both take for granted that the padding they are managing is paid.

We set out to design an admission-control policy for exactly that cost on TPU:
promote a request into a larger already-warm shape and pay the padding, or queue it
and pay the delay. Six sessions in, a control experiment we should have run first
showed the premise was wrong. This paper reports what is true instead.

**Contributions, in order of strength.**

1. **The attention request ladder is not the printed request ladder** (§4.3). The
   ladder the log advertises is not the one attention executes at; attention is
   pinned to a single bucket by a default-off environment flag. Verifiable from
   source by anyone, and it explains an otherwise anomalous decode measurement.
2. **A measured negative result on the cost lever** (§5), framed as validation:
   Ragged Paged Attention's fine-grained tiling works as designed, and what remains
   after it had not been measured. 36% of executed tokens are padding, and doubling
   that padding costs 7–11% rather than 100%.
3. **A 6.11 ms fixed cost per scheduler step**, measured directly, and the
   **crossover rule it implies** (§5): step-for-alignment trades lose below ~2048
   tokens and win above. It rejects two natural-looking optimisations — decomposing
   a padded residual is **20.6% worse** measured, and bucket-aligned packing is
   net-negative by 4.7× at the smallest boundary — and it explains why our own
   implementation attempt was inert at 512-token prompts: it was optimising the
   term that loses.
4. **A validated cost model and a provable bound** (§6) for what scheduling *can*
   buy once padding is excluded.
5. **A methodological rule** (§7), reported with both of the errors that produced
   it: step-scoped properties require step-scoped instruments.

We explicitly do **not** claim a ladder redesign, an admission-control policy, or a
throughput improvement.

---

## 2. Method

**Controlled variables.** Prefix caching off (asserted at run start, aborting rather
than warning), chunked prefill recorded, `max_model_len` 8192,
`max_num_batched_tokens` 8192, TP=4, `VLLM_TPU_BUCKET_PADDING_GAP` unset (default
exponential ladder). Every run parses the server's own engine-config line and
aborts if any controlled variable disagrees with the config it claims to be running.

**Units.** All costs are server-side, from vLLM's Prometheus histograms scraped as
*deltas* around each measurement block, never client wall-clock. A client stopwatch
measures network RTT, HTTP, tokenizer and queueing along with compute; the claims
here are about compute.

**Traceability.** Every run writes `meta.json` before doing any work, records a
config hash, git SHA and dirty flag, and appends to a manifest. Runs are never
overwritten. Interrupted runs are marked and excluded from analysis automatically.

**Models.** Qwen3-4B primary (head_dim 128, GQA 4:1); SmolLM2-1.7B-Instruct
(head_dim 64, MHA) and TinyLlama-1.1B-Chat (head_dim 64, GQA 8:1) for the
architecture contrast in §4.2.

---

## 3. The three quantized dimensions

Read from `tpu_inference/runner/tpu_runner.py:2133` and `runner/utils.py`, then
confirmed on hardware. Per scheduler step:

| | quantizes | ladder | source |
|---|---|---|---|
| **D1** | prompt length → prefill shape | — | *does not exist* |
| **D2** | scheduled tokens / step | `[16, 32, …, 8192]` | `get_token_paddings` |
| **D3** | requests / step | `[8, 16, …, 256]` non-attention; **`[256]` attention** | `get_req_paddings`, `get_attn_req_paddings` |

---

## 4. What each dimension does

### 4.1 D1 — per-request length padding does not exist

We hold batch size and total token count fixed and vary only the *spread* of
request lengths. Three models coincide on a uniform batch and diverge on a ragged
one: **packed** (cost tracks summed true lengths), **per-request padded**, and
**batch padded** (one compiled sequence dimension, everyone padded to the longest).

At `n=4`, `total=2048`, `max_len=1536` the three predict 39.2 / 56.6 / 144.8 ms —
a 3.7× spread, far outside measurement noise.

Pooling the two independent spread runs (§4.1's chunked-prefill control showed
the setting is irrelevant, so they are replicates), at n=8, total=4096, where
packed predicts 69.08 ms:

| max_len | 512 | 768 | 1024 | 1536 | 2048 | 3072 | 3900 |
|---|---|---|---|---|---|---|---|
| median measured (ms) | 68.47 | 72.86 | 73.38 | 76.55 | 73.50 | 89.40 | 88.28 |
| penalty vs packed | −0.9% | +5.5% | +6.2% | +10.8% | +6.4% | **+29.4%** | **+27.8%** |
| run-to-run spread | 2.0% | 8.1% | 7.3% | 0.8% | 0.3% | **14.4%** | 0.2% |

Uniform controls, where all three models agree, match to 1.9% — so the instrument
is sound. **Batch padding is rejected by 44–618%** across every ragged cell.

The third row is why the second is reported as a trend rather than a curve: the
same cell measured twice on the same server instance differs by up to 14.4%,
which is the §8 bimodality showing up again. The monotone growth of the penalty
with spread survives pooling; the individual values should not be read to better
than ~10%.

The obvious objection is that chunked prefill does the packing, which would make
this a narrow claim about a scheduler option. It does not. Re-run with
`--no-enable-chunked-prefill` (vLLM warns this is unofficial for this model but
accepts it), packed still wins 8 of 10 ragged cells and batch padding is still
rejected by 75–579%, cell by cell nearly unchanged.

Raggedness is not entirely free: the penalty over pure packed grows with spread to
**+28%** when one request holds 95% of a batch's tokens (pooled median; 27.8%). Neither pure model fits
well (packed mean error 11–12%), and we report the residual rather than choosing a
winner.

### 4.2 D2 — the token ladder is paid, at batch size one

With a single request in flight, the step's token count *is* the request's length,
so nothing is smeared. Sweeping length within one compiled bucket and measuring
server-side prefill time gives a *flatness* statistic — 1.0 means cost is
independent of true length within the bucket (a perfect staircase), 0.0 means cost
is proportional to true length.

| bucket | Qwen3-4B (dim 128, GQA 4:1) | TinyLlama (dim 64, GQA 8:1) | SmolLM2 (dim 64, MHA) |
|---|---|---|---|
| 512 | 1.00 | 0.90 | 0.84 |
| 1024 | 0.96 | 0.85 | 0.78 |
| 2048 | 0.91 | 0.82 | 0.73 |
| 4096 | 0.81 | — | 0.54 |

The staircase is real and **architecture-dependent**. Holding head_dim at 64 and
moving only the GQA ratio (SmolLM2 → TinyLlama) raises flatness by +0.07 to +0.09
at every bucket — roughly half the Qwen3–SmolLM2 gap. Neither head_dim nor the GQA
ratio alone explains it. Mechanistically the direction is coherent: fewer KV heads
means less real attention work per token, so the fixed padded cost is a larger
fraction of the total.

A mechanism check: `request_prefill_kv_computed_tokens` tracks the **true** length,
never the padded bucket, on every model. RPA does skip padding inside attention;
whatever is paid is paid outside it.

### 4.3 D3 — the printed request ladder is not the attention ladder

Every boot logs two request ladders and we read past the second one six times:

```
Prepared request paddings:      [8, 16, 32, 64, 128, 256]
Prepared attn request paddings: [256]                    <- one entry
```

The source explains it:

```python
def get_attn_req_paddings(min_req_size, max_req_size):
    if not envs.ATTN_BUCKETIZED_NUM_REQS:   # defaults to False
        reqs = [max_req_size]               # ONE bucket, at the maximum
```

**Attention always executes at 256 requests, whatever the batch size.** It is the
dominant cost and its shape never varies, so there is no batch-ladder step to find
— in prefill or in decode. The `[8, 16, …]` ladder still applies to non-attention
work (sampling, logits), which is small.

**And enabling the ladder buys nothing.** Setting `ATTN_BUCKETIZED_NUM_REQS=1`
compiles the full `[8, 16, …, 256]` attention ladder — verified in the warmup log,
which prints it instead of `[256]` — and we measured the same concurrency sweep
on the same server instance in both modes:

| n | prefill (off → on) | decode (off → on) | e2e (off → on) |
|---|---|---|---|
| 8 | 75.8 → 75.8 | 53.2 → 53.2 | 139.3 → 139.1 |
| 9 | 86.7 → 104.0 | 61.8 → 61.8 | 159.1 → 193.3 |
| 16 | 158.8 → 153.0 | 86.2 → 88.4 | 254.2 → 254.6 |

Decode is identical to 0.1 ms at n=8 and n=9: compiling attention at the actual
batch size rather than at 256 changes nothing. The padded request slots hold no
KV blocks, so carrying them is free — RPA behaving exactly as designed. The n=9
prefill gap is the bimodal cell of §8, where decode is unchanged, not a flag
effect. **The default is correct**, and we can now say so with a number instead of
inferring it from the comment above the function.

Hardware agrees on the batch ladder generally. If decode padded 9 sequences up to
16, decode at n=9 would cost what n=16 costs:

| n | 8 | **9** (padded to 16) | 16 |
|---|---|---|---|
| decode phase | 53.3 ms | **51.4 ms** | 91.8 ms |

It costs what n=8 costs. Two sessions were spent hunting a promotion cost at the
8→16 edge that the default configuration had already excluded.

---

## 5. How much padding is executed, and whether any of it is paid

**The ceiling.** The Prometheus histogram bucket edges for `iteration_tokens_total`
are powers of two, and so is the compiled token ladder, so a step's reporting bucket
edge *is* the size it executed at. Over 150 instrumented dispatches:

- padded / real = **1.56×** → **35.9% of executed tokens are padding**
- mean per-dispatch padding ratio **56.8%**, p95 **99.6%**

**Whether it is paid.** At fixed batch size the real token count is constant, but
the scheduler chunks differently between repeats — so identical real work executes
at padded totals differing by up to 2×. This is a natural experiment with a
property no designed one can match: real work is held *exactly* constant, not to
within a few percent.

| n | real tokens | padding rises | cost |
|---|---|---|---|
| 8 | 4104 | ×1.40 | **×0.80** |
| 9 | 4617 | ×1.50 | **×1.00** |
| 10 | 5130 | ×1.43 | **×0.98** |
| 12 | 6156 | ×1.22 | ×1.61 ← sole exception |
| 14 | 7182 | ×1.22 | ×0.99 |
| 16 | 8208 | ×1.18 | ×0.99 |

Doubling padded tokens moves cost by ≤2%, frequently downward. The n=12 cell is
the sole exception and is also the cell with an unexplained bimodality (§8); its
ordering is non-monotone (9216→82.6 ms, 10240→136.5, 11264→132.7), so it is not a
padding effect either.

**Quantified under randomised assignment.** The above is a natural experiment —
the scheduler chose the splits — so we also ran designed straddles: batch size and
per-sequence length held fixed, the step's token count moved just across a compiled
boundary, at four boundaries. Only dispatches verified to be a single step count.

| boundary (n=4) | real work | padded work | measured cost | share of padding paid |
|---|---|---|---|---|
| 512 → 1024 | ×1.016 | ×2.00 | ×1.114 | **10.0%** |
| 1024 → 2048 | ×1.008 | ×2.00 | ×1.227 | **22.1%** |
| 2048 → 4096 | ×1.004 | ×2.00 | ×1.243 | **24.0%** |
| 4096 → 8192 | ×1.002 | ×2.00 | ×1.249 | **24.8%** |

**The share paid grows with the boundary and plateaus near 25%.** An earlier
version of this paper reported 6–10%, from two edges of which one had split
dispatches and the other sampled the *smallest* boundary. At the boundaries where
most tokens live it is roughly a quarter.

The same sweep at n=8 produced split dispatches in 4–7 of 9 runs per arm and is
excluded; a split reintroduces exactly the smearing these experiments exist to
avoid, and the split arms give visibly inconsistent shares (6.0%, 24.3%, 7.3%).

**Consequence.** All three dimensions are inert under batching, in the sense
that none of them turns a doubling of nominal padding into a doubling of cost. A scheduler-side
optimisation we designed — deferring a marginal chunk rather than spilling into the
next bucket — is **analysed and rejected on a computed ceiling** rather than left
untested, and the answer follows from two measured quantities rather than from a
patch. Deferring a marginal chunk to align a step avoids the *paid* part of that
step's padding, and — when the deferred work does not already have a step waiting
— creates one, at the measured fixed cost of **6.11 ms**. Both sides are
measurable, so the rule is derived:

| boundary | paid padding | vs the 6.11 ms step | verdict |
|---|---|---|---|
| 512 → 1024 | 1.30 ms | 0.2× | **deferring loses by 4.7×** |
| 1024 → 2048 | 4.62 ms | 0.8× | loses |
| 2048 → 4096 | 9.36 ms | 1.5× | wins |
| 4096 → 8192 | 18.82 ms | 3.1× | wins |

Paid padding scales with the bucket; the step cost does not. **They cross between
2048 and 4096 tokens.** Below that, an alignment optimisation that trades a step
for alignment is not marginal but decisively wrong — by nearly 5× at the smallest
boundary. Above it, alignment pays.

**This is the paper's actionable result, and it inverts the usual intuition.** In
this stack the fixed per-step cost exceeds the paid padding cost at every
boundary below 2048 tokens. For the small steps that dominate a
decode-heavy workload, the advice is the opposite of shape alignment: **reduce
step count, tolerate padding.**

Two conditions attach to it, and both matter.

*Saturation removes the penalty.* The 6.11 ms is charged only when deferral
creates a step that would not otherwise have run. Under sustained load the next
step exists regardless, deferral merely moves tokens into it, and alignment wins
at every boundary. The rule above is the unsaturated case, which is also the
case where the padding matters least in absolute terms.

*The 12% ceiling assumes a constant step count.* §5's figure is already
discounted by the paid fraction — it is 36% of tokens × ~24% paid, not 36% of
execution — but it presumes alignment is free, i.e. achieved by repacking rather
than by deferring. Under the deferral counterfactual the recoverable amount is
much smaller and boundary-dependent, and at boundaries below 2048 it is negative.
Quoting 12% without that assumption attached would be quoting a different
quantity.

**What does not work.** Decomposing a padded residual into exact bucket sizes —
1808 tokens as 1024+512+256+16 rather than one step padded to 2048 — is **20.6%
worse** measured (51.06 ms vs 42.33). The cost model predicted it would win by
1.85 ms, and was wrong because it priced a 16-token step at 0.41 ms by
extrapolating linearly from the origin. Measured, a 16-token step costs **6.10 ms**
and a 32-token step costs 6.08: there is a **6.11 ms fixed cost per scheduler
step**, and four steps pay it four times.

*Weakness, stated:* this is a natural experiment. The scheduler chose the splits, so
a lurking variable correlated with both split and cost is not excluded the way
randomisation would exclude it. A designed straddle at fixed n and per-sequence
length remains worth running as confirmation.

---

## 6. What scheduling can buy once padding is excluded

With padding largely unrecoverable, the remaining lever is batching efficiency,
and that one is real:

| batch | tokens | cost | per token |
|---|---|---|---|
| 1 | 512 | 13.15 ms | **25.7 µs** |
| 4 | 2048 | 39.22 ms | 19.2 µs |
| 8 | 4096 | 69.08 ms | **16.9 µs** |
| 16 | 8192 | 144.75 ms | 17.7 µs |

The mechanism is a **fixed per-step cost of 6.11 ms**, measured directly by
extending the curve below its lowest knot: a 16-token step costs 6.10 ms and a
32-token step 6.08. Sublinearity is that constant being amortised over more
tokens, and it is nearly half the cost of a 512-token step. The model previously
extrapolated linearly from the origin below 512 tokens, understating a 16-token
step by **15×**.

**Cost model.** A piecewise-linear interpolation of the measured curve — not a
parametric form. Our first model assumed a ladder step and failed its hardware
holdout at **105.7% MAPE**; the refit passes two independent holdouts: seeds the
fit never saw (**3.6%**, worst cell 7.9%) and *rates* the fit never saw
(**2.9%**, worst cell 8.9%).

**Its stated limits.** Calibrated at `prompt_len=512` and `output_len=1`. We tested
the generalisation and it fails: at `prompt_len=256` and matched total tokens, costs
differ by −35% to +25%. Above 2048 tokens the transfer is a consistent ~9%; below,
it is not. Cost is **not** a function of total tokens alone.

**Policy measurement.** Because stock vLLM never holds a request back (measured
queue time 0.0 ms at every concurrency), admission policy is implementable entirely
client-side by choosing release timing — no scheduler patch, deployable as a proxy.
Six paired seeds at 25 req/s:

| | TPU cost vs stock | p95 latency vs stock |
|---|---|---|
| hybrid | **−26.1%** CI [2.53, 4.17] ms, p=0.001 | +12.7 ms CI [−22, +35], p=0.570 |
| wait-to-fill | −26.6%, p=0.001 | **+338 ms** CI [+273, +400], p=0.001 |

Hybrid reaches nearly all of wait-to-fill's saving for a small fraction of its
latency. **We do not claim this is free.** Our own harness spends ~24 ms per
dispatch scraping metrics, which inflates *stock's* p95 from ~24 ms to ~86 ms and
masks the delay hybrid introduces deliberately; simulated without it, hybrid costs
roughly +188% p95 at 25 req/s. The p95 result above is a property of our driver as
much as of the policy, and we report it as such.

**A bound, not an oracle.** We compute the offline optimum by dynamic programming
over contiguous batchings, minimising `cost + λ·latency`, verified against
brute-force enumeration of every partition at n=9. Comparison uses the Lagrangian
dual `max_λ [g(λ) − λL]`, which provably lower-bounds every schedule no slower than
the policy. Cost above that bound, at each policy's own latency:

| rate | promote (stock) | hybrid | wait-to-fill |
|---|---|---|---|
| 10 | **2.2%** | 2.7% | 15.7% |
| 25 | **4.8%** | 6.6% | 8.2% |
| 55 | 12.8% | 11.1% | **6.7%** |
| 90 | 17.9% | 14.2% | **7.6%** |
| worst case | 17.9% | **14.2%** | 15.7% |

**Stock is not badly suboptimal** — 2–5% above the bound at low load. It occupies
the low-latency end of the frontier. Every fixed policy is near-optimal somewhere
and poor somewhere else; hybrid's only distinction is the lowest *worst-case*
regret. That is a minimax claim and a modest one.

This section is dynamic batching, measured. We flag that plainly: the mechanism is
per-step overhead amortisation, not shape.

---

## 7. Methodological findings

Two of our inferences were invalid. Both are the same error and both were caught by
our own controls, so we report them rather than quietly fixing them.

**Request-scoped metric → step property.** At `output_len=8`, a batch of 9 behaved
like a batch of 16 (+62 ms) — exactly the promotion cost the project predicted. At
`output_len=1` it did not. The likely explanation is interleaving: with multiple
output tokens some requests decode while others prefill, so a request's measured
"prefill time" spans other requests' steps. We built a cost model on that 62 ms
number and it failed its holdout at 105.7%.

**Dispatch-scoped curve → step property.** The dispatch cost curve shows no
staircase, which we read as evidence against per-step token padding. Invalid: a
dispatch is 2–4 engine iterations, so its cost is a *sum over steps* whose token
counts land wherever the scheduler put them, and summing over a staircase smears
it. The flat curve is *uninformative* about per-step padding, not contrary to it.

**The rule.** Step-scoped properties require step-scoped instruments. Verify
single-step execution from the `iteration_tokens_total` count delta before
attributing anything to a step.

Three further process findings, each of which cost real money or nearly produced a
wrong published number:

- **A three-repeat median of a bimodal cell is not an estimate.** Three repeats put
  a promotion cost at +16 ms; twenty-one put it at +62.1 ms, matching an earlier
  session's +62.3 ms to 0.3%.
- **A cost curve that is optimised over needs more support than one that is only
  read.** Our DP found the cheapest point on the curve to be a knot resting on
  three observations of a bimodal cell, and built "optimal" schedules on it.
- **Model choice on this stack is constrained by packaging, not architecture.**
  Three server boots were lost to a registered-but-broken architecture, a checkpoint
  with no safetensors, and a checkpoint carrying `rotary_emb.inv_freq` buffers the
  loader rejects — none visible in `config.json`. A preflight that reads
  architecture, weight format, tensor names, TP divisibility and context length now
  retrodicts all of them in seconds.

---

## 8. Limitations and open questions

- **`output_len=1`** for every cost measurement. Decode is where production serving
  spends most of its time. We measure decode's *batch* padding (§4.3) but not a
  decode-dominated cost model.
- **One accelerator** (v5e-4, TP=4), one primary model, one prompt length for the
  cost model — whose generalisation across prompt length we tested and rejected.
- **Synthetic workloads.** Arrivals are Poisson and prompts uniform. No production
  trace.
- **The cost curve below 512 tokens is unmeasured.** The model scales linearly
  from the origin there, which is an assumption, not a measurement. It decides
  whether decomposing a padded residual into exact bucket sizes wins or loses:
  on the extrapolation, `C(1024+512+256+16) = 37.37 ms` beats `C(2048) = 39.22`,
  but a fixed per-step launch cost would reverse it. Five cells would close it.
- **The end-to-end value of bucket-aligned packing is unmeasured.** M1 bounds the
  per-step saving (6.6–9.9% on a maximally spilled step); what fraction of real
  steps spill, and what deferral costs in latency, is not known.
- **An unexplained bimodality.** Per-dispatch cost has two modes ~1.6× apart at
  n=9–14. Excluded so far: differing scheduled tokens (identical between modes),
  differing step count (identical at n=12 and n=14), drift/warmup/thermal (a runs
  test matches randomness — 6 runs observed against 6.1 expected at n=10), and
  top-bucket occupancy (ratios 1.00 and 0.98 at n=9 and n=10). It is a per-dispatch
  coin flip whose cause is not on `/metrics`; resolving it needs XLA profiler
  traces.

---

## 9. Related work

**Ragged Paged Attention** is the technique our results validate. Its design goal is
fine-grained tiling for ragged execution; we find per-request padding costs nothing
measurable, which is what that design predicts. We are not aware of a published
measurement of what remains after it in a production stack, which is the gap this
paper fills.

**BucketServe** and **LAPS** both manage length-bucketing overhead, on GPU. Our
result does not refute them; it bounds their transferability. On this TPU stack the
padding they target is not paid, so length bucketing has no purchase — and D3 shows
the batch dimension they would bucket over is pinned to a single shape by default.

**LENS** characterises shape-induced latency steps on NPUs and predicts them to
2.15%. It is a predictor, not a scheduler, and it is our methodological precedent
for the flatness statistic rather than a competitor.

**Vidur** established simulator-fidelity validation (<9% error) as the standard for
this kind of work; our holdout discipline follows it.

---

## 10. Conclusion

A production TPU serving stack quantizes shapes in three dimensions. One does not
exist, one is disabled by a default configuration flag, and one is paid only at a
batch size that never occurs in serving. **36% of executed tokens are padding, and doubling
it costs 7–11% rather than the 100% the premise predicts.**

The practical advice is negative and worth stating: do not build length bucketing,
shape-aware admission control, or ladder design for this stack. What remains for a
scheduler is ordinary batching amortisation, worth ~26% of TPU time at moderate load
against a stock baseline that is already within 2–5% of a provable optimum at its
own latency.

We arrived at this by trying to build the opposite paper, and the control experiment
that refuted it cost $3 and should have run first.

---

## Appendix A — reproduction

Every number above regenerates from committed configs and captured runs. Hardware
sessions total [redacted]. `scripts/check_model.py` preflights a model in seconds;
`scripts/refit_cost_model.py --write` regenerates the cost curve and re-runs both
holdouts; `scripts/h1_headroom.py` recomputes §5 offline from captured dispatches.

**Number provenance.** `scripts/paper_numbers.py` recomputes every figure in this
paper from the captured runs and diffs it against the text, emitting
`results/paper_numbers.parquet` with a `claim_id` per number and the `run_id` it
derives from. **35 claims, 35 verified.** It found two real defects on first run:
an 8192-token cost transcribed from an exploratory dump over all seeds rather than
the fitted curve (145.58 vs 144.75 ms), and a §4.1 table taken from one of two
replicate runs that differ by up to 14.4% on the same cell — now pooled, with the
run-to-run spread reported alongside.

**A falsified external prediction, recorded.** Before D1 was measured, the
explanation offered for it was that chunked prefill makes the compiled shape the
*chunk* rather than the prompt, so per-request padding cannot exist. That
prediction is wrong: with `--no-enable-chunked-prefill` the result is unchanged
(§4.1). The packing is structural to the TPU serving path, not a consequence of
that scheduler feature. We report it because a plausible mechanistic argument
that survives scrutiny and dies on measurement is the paper's own thesis applied
to itself.

**Still to build:** `reproduce_all.sh` regenerating every figure from the
manifest, and the figures themselves.
