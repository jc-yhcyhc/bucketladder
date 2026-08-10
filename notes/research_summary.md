# Research summary — what was done, and what it showed

**Closed 2026-08-10.** 12 hardware sessions, **[redacted]** of a $1,000 ceiling.
42 verified claims, 184 tests, 69 commits. Stack: vLLM 0.25.0 + `tpu-inference`
0.25.0, JAX 0.10.2, libtpu 0.0.42.1, on `v5litepod-4` (4 chips, TP=4).

---

## 1. What we set out to do, and what happened instead

**The plan:** on compiled-shape accelerators every request is padded up to one of
N precompiled bucket shapes. When a request's bucket is saturated the scheduler
must either promote it into a larger warm bucket and pay the padding, or queue it
and pay the delay. Build the admission policy that decides, and design the ladder
it runs against.

**What happened:** the premise is false. Per-request length padding does not
exist on this stack, and the padding that does exist is mostly not paid at the
batch sizes serving actually uses. Six sessions in, a control experiment that
should have run in week 0 rejected the founding assumption by 44–618%.

The project became a characterisation of what compiled-shape quantization
actually costs — which is the paper that exists now.

---

## 2. What was measured

| | question | answer |
|---|---|---|
| **e00** | is the ladder enumerable and are controls asserting? | yes; gate has passed every session |
| **e01** | does a single request pay its full sequence bucket? | **yes** — flatness 0.97 at ≤1024 |
| **e02** | does stock vLLM queue, or promote? | **never queues** — 0.0 ms at every level |
| **e04** | does batch splitting explain the cost bimodality? | **no** — controls vary too |
| **e05** | does the per-step token histogram explain it? | **no** — top-bucket occupancy uncorrelated at n=9–10 |
| **e07** | does a mixed-length batch pay per-request padding? | **no** — rejected by 44–618% |
| **e08** | is that because of chunked prefill? | **no** — same result with it disabled |
| **e21** | how far from optimal are the policies? | DP + Lagrangian bound, brute-force verified |
| **e30/e40/e41** | does the cost model predict held-out hardware? | 3.6% (seeds), 2.9% (rates) |
| **H1/M1** | how much padding is executed, and paid? | **36% executed**; 10% paid at 512→1024 |
| **M2** | what does `ATTN_BUCKETIZED_NUM_REQS=1` buy? | **0.0%** on decode |
| **M3** | what does the curve do below 512 tokens? | it was extrapolating 15× low |
| **M4** | does the paid share depend on the boundary? | **yes** — 10% → 25% |
| **M5** | is the step cost a constant? | **no** — varies ×7.7 with n |
| **M6** | map step cost across n and phase | prefill unmeasurable at n≥8; decode smooth |

---

## 3. Results that stand

### 3.1 The attention ladder is not what the system reports

`envs.ATTN_BUCKETIZED_NUM_REQS` defaults to `False`, and when it is off
`get_attn_req_paddings` returns `[max_req_size]` — **one bucket**. Attention
always executes at 256 requests regardless of batch size, while every boot prints
a six-entry request ladder:

```
Prepared request paddings:      [8, 16, 32, 64, 128, 256]
Prepared attn request paddings: [256]
```

Hardware agrees: if decode padded 9 sequences to 16, decode at n=9 would cost
what n=16 costs. It costs what n=8 costs — **51.4 ms against 91.8**. And enabling
the flag changes decode latency by **0.0%** (identical to 0.1 ms at n=8 and n=9),
because RPA's padded request slots hold no KV blocks. **The default is correct**,
now with a number rather than an inference from a code comment.

Absent from RPA, absent from LENS, absent from vLLM's TPU documentation.

### 3.2 Shape-quantization cost is not a property of the step

It depends on batch size, and the dependence is large:

| | prefill slope | reading |
|---|---|---|
| n=1 | 1.61 µs/token | flat within a bucket — a staircase |
| n=2 | 0.75 µs/token | flat |
| n=4 | 17.18 µs/token | linear in tokens |

Equivalently, as the share of nominal padding actually paid: **~85% at n=1–2,
10–25% at n=4–8**. Above n=8 the quantity **cannot be isolated** — zero of
fourteen cells produced a single-step dispatch, because the scheduler splits every
one. Whether the transition is a step or a smooth decay is **unknown**, and not
obtainable with a `/metrics`-delta instrument.

### 3.3 Padding is abundant and mostly free

**35.9% of executed tokens are padding** (p95 per-dispatch ratio 99.6%), and the
share actually paid rises with the boundary — 10.0% at 512→1024, 22.1% at
1024→2048, 24.0% at 2048→4096, 24.8% at 4096→8192.

Per-request *length* padding does not exist at all: a mixed-length batch costs its
packed tokens, rejecting the batch-padding model by 44–618%, and **not because of
chunked prefill** — disabling it leaves the result unchanged.

### 3.4 Decode is smooth, and decode is what production runs

| n | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ms/step | 3.80 | 4.25 | 4.30 | 4.98 | 6.52 | 9.13 |
| µs/step/seq | 3802 | 2127 | 1075 | 622 | 407 | **285** |

Per-step cost rises **2.4×** while n rises **32×**; per-sequence cost falls **13×**
monotonically, no discontinuity anywhere — across exactly the range prefill could
not reach. **The pathology is real and lives in the phase that matters least.**

### 3.5 LENS's protocol does not transfer

LENS predicts NPU latency to 2.15% with a per-bucket `intercept + slope × length`
fitted from two measurements per bucket. Run on TPU with a withheld mid-bucket
point: **MAPE 5.23%, worst 22.41%** — near-perfect at n=1–2 (0.0–0.6%) and failing
at n=4 (17–24%). Its single-regime form does not survive the batch sizes
production uses. Validation of a published technique on hardware its authors never
ran.

### 3.6 The bimodality, localised

Per-cell spread over 9 repeats: **decode 1.00–1.04× at every n; prefill 1.00–1.03×
at n≤4 and 1.18–1.26× at n≥8.** Variance switches on exactly where the scheduler
begins splitting dispatches, and decode — which has no chunking decision — never
shows it. Localisation, not mechanism: e04 showed step *count* does not correlate
with cost, so the variance is in *how* a dispatch splits, not into how many pieces.

### 3.7 Supporting

A cost model passing two independent holdouts (3.6% on unseen seeds, 2.9% on
unseen rates); a provable optimality bound by DP with Lagrangian dual, verified
against brute force; and R3 decomposed — the staircase is architecture-dependent,
with the GQA ratio accounting for about half the gradient (Qwen3-4B 0.91,
TinyLlama 0.82, SmolLM2 0.73 at bucket 2048).

---

## 4. What was withdrawn

| claim | why |
|---|---|
| bucket-aware admission control | premise false (§1) |
| ladder redesign as a cost lever | D1 does not exist, D3 inert |
| the crossover rule (~2048 tokens) | combined an n=1 constant with an n=4 fraction |
| a 6.11 ms fixed per-step cost | `C(16) ≈ C(32)` read out of the n=1 staircase |
| "two regimes" | two points; consistent with a smooth decay |
| any throughput improvement | I1 failed twice — inert, then corrupting |
| bucket-aligned step packing | ceiling computed, then the ceiling's basis withdrawn |

The 26% policy saving measured on hardware **stands as a measurement** but is
reframed: the mechanism is per-step overhead amortisation — dynamic batching —
not shape.

---

## 5. Methodological findings

**Six failures, one cause.** A quantity measured under one configuration, used
under another:

| looked like | was |
|---|---|
| cost model fails holdout at 105.7% | fitted at `output_len=8`, run at 1 |
| "the premise is wrong, padding is free" | e01 and e07 both right, different n |
| curve extrapolates small steps 15× low | reading the n=1 staircase as a floor |
| decomposition wins in model, loses measured | same |
| I1 patch inert | 512-token prompts sit in the losing regime |
| the fixed cost is not constant | all of the above, named |

**The guardrail took three versions.** "Different batch sizes" would not have
caught the `output_len` failure. A whitelist of config keys missed the crossover,
because batch size is not a top-level field — it lives inside `edges` and is
implicit in `token_sizes`. Final form: **diff every config key**, exempt only free
text, require each differing field to be named in the claim's `invariant_over`. It
flags the retired crossover, and it flagged five live claims until an invariance
we had assumed was stated explicitly.

**Two invalid inferences, both self-caught**, both measuring a step-scoped
property with a coarser instrument: a request-scoped metric (the `output_len=8`
anomaly, contaminated by prefill/decode interleaving) and a dispatch-scoped curve
(2–4 engine iterations summed, smearing any staircase).

**The correctness gate paid for the whole project.** I1's second attempt showed
−29% TPU and −49% p99 — and 4 of 48 greedy completions differed, every one at a
prompt length just above a bucket boundary. Without that gate, a patch that
silently drops prompt tokens ships as a 29% throughput win.

---

## 6. Scope

Measured: one accelerator (v5e-4, TP=4), one primary model (Qwen3-4B) with two
others for the architecture contrast, uniform 512-token prompts with Poisson
arrivals, `output_len=1` for most cost work and 16–64 for the decode and policy
work, batch sizes 1–32.

Not measured, and stated as such: prefill step cost above n=4; production traces;
the mechanism behind non-deterministic splitting; any hardware other than v5e.

---

## 7. Artifacts

`scripts/check_model.py` preflights a model in seconds and retrodicts all four
load failures; `scripts/paper_numbers.py` ties 42 claims to `run_id`s, recomputes
them from captured data, and enforces the invariance guardrail; the traceability
contract aborts rather than warns and has done so twice on real mistakes;
`captured/` holds every run from twelve sessions.

**Still owed:** one deliberate rewrite of the paper draft, a README that stands
alone, `reproduce_all.sh`, and figures.
