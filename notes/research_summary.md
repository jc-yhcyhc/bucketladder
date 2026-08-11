# Research summary — what was done, and what it showed

**Fifteen hardware sessions on `v5litepod-4`, [redacted] of a $1,000 ceiling, 66
verified numerical claims, 190 tests.** Every number regenerates from
`captured/` via `./reproduce_all.sh`, which exits non-zero if any disagrees.

Stack: vLLM 0.25.0 + `tpu-inference` 0.25.0, JAX 0.10.2, libtpu 0.0.42.1, TP=4.

---

## 0. The claim, in one sentence

**The cost of a compiled step in a production TPU serving stack is not a property
of the step. It depends on batch size, and treating it as a constant produces a
series of plausible-looking wrong conclusions — including several we made and
caught.**

---

## 1. What we set out to do, and what happened instead

The plan was bucket-aware admission control: when a request arrives and its
compiled shape is saturated, promote it into a larger warm bucket and pay the
padding, or queue it and pay the wait. The premise — that rounding up to a
compiled shape means paying for the shape you rounded up to — is what
**BucketServe** and **LAPS** assume on GPU, and it is what motivates the whole
family.

Six sessions in, a $3 control experiment rejected it. Per-request length padding
does not exist on this stack: a mixed-length batch costs its packed tokens, and
the batch-padding model is rejected by 44–618%. The paper became a measurement
study of what is actually paid.

A second phase began after peer review of draft 2 raised seven major findings and
six questions. Answering them consumed three more sessions, **withdrew two
headline numbers**, overturned one stated limitation, and added the mechanism the
work had been missing for three drafts.

---

## 2. What was measured

Three quantized dimensions, read from `tpu_inference/runner/tpu_runner.py:2133`:

| | quantizes | ladder |
|---|---|---|
| **D1** | prompt length → prefill shape | *does not exist* |
| **D2** | scheduled tokens / step | `[16, 32, …, 8192]` |
| **D3** | requests / step | `[8, …, 256]` non-attention; **`[256]` attention** |

Everything is server-side, from Prometheus histogram *deltas* — never client
wall-clock. Single-step execution is verified per dispatch from
`iteration_tokens_total`; split dispatches are excluded rather than averaged.
Where a measurement needs *n* requests inside one scheduler step, they are
released from a thread barrier after every connection is established.

---

## 3. Results that stand

### 3.1 The attention ladder is not the one the system reports

`ATTN_BUCKETIZED_NUM_REQS` defaults to `False`, and when off
`get_attn_req_paddings` returns `[max_req_size]` — one bucket. Every boot prints
two ladders that disagree:

```
Prepared request paddings:      [8, 16, 32, 64, 128, 256]
Prepared attn request paddings: [256]
```

Three independent confirmations:

1. **Source.** The flag's default and the one-bucket return path.
2. **A paired hardware experiment.** Enabling the flag compiles the full ladder
   and changes decode by **0.0%** — because RPA's padded request slots hold no KV
   blocks.
3. **The compiler's own naming.** The decode attention kernel is emitted as
   `RPAd-p_256-bq_1_1-bkv_8192_8192`. **The 256-request padding is in the kernel
   name.** This came free from the operator profile in session 15.

A weaker fourth line — that decode at n=9 costs what n=8 costs (51.4 vs 53.3 ms)
rather than what n=16 costs (91.8 ms) — is reported as corroboration only. Its
bootstrap intervals exclude the 100% the premise predicts but cannot pin the
value: [−43%, 19%], [−66%, 49%], [−67%, 59%] across three runs.

This finding is absent from the RPA paper, from LENS, and from vendor
documentation. **Two sessions were spent hunting a promotion cost at the 8→16
edge that the default configuration had already excluded.**

### 3.2 LENS's model form does not earn its place

LENS predicts NPU latency to 2.15% MAPE with a per-bucket
`intercept + slope × length` fitted from two measurements per bucket. Reproduced
on TPU across 5 buckets × 3 batch sizes, 7 repeats per point, withholding a
mid-bucket point: **5.23% MAPE, worst 22.4%.**

A MAPE shift across hardware is weak evidence by itself, so two ablations:

| batch size | LENS | constant-only (no length term) |
|---|---|---|
| 1 | 0.38% | 0.96% |
| 2 | 0.39% | 0.86% |
| 4 | 19.77% | **14.80%** |

At n=1–2 the within-bucket curve is nearly flat (flatness 0.97), so *any*
two-point fit is near-perfect — a constant lands within 0.6 pp. **The reported
accuracy at low batch size is evidence about the fit protocol, not the model
form.** At n=4 the length term is actively harmful. It earns its place at no
batch size measured. The n=4 comparison is paired across the three buckets and
excludes zero: LENS minus constant-only is **+4.97 pp, 95% CI [+4.28, +5.61]**.
Three buckets is a small sample, but the direction is consistent in every one.

The failure is not a sampling artifact: over all three choices of which two
points calibrate, the n=4 error swings up to 44.8 pp but its **minimum is still
16.97%**, far above 2.15%.

### 3.3 Shape-quantization cost depends on batch size

Share of nominal padding actually paid, straddling a compiled boundary at fixed
batch size and near-fixed sequence length:

| batch size | median share paid | range across boundaries | clean cells |
|---|---|---|---|
| 1–2 | **~85%** | — | — |
| 4 | 23.1% | [10.0%, 24.8%] | 4 |
| 8 | 14.3% | [0.2%, 21.0%] | 3 |
| 16 | **−2.7%** | [−15.4%, +0.5%] | 3 |

**We no longer describe this as monotone.** The n=4 and n=8 ranges overlap
substantially, so those two levels are not separable with the data we have; only
the n≤2 and n=16 ends are. The defensible statement is **high at n≤2,
intermediate and not separable at n=4–8, indistinguishable from zero at n=16**.
The n=8 row is also a correction: it was previously 16%, computed with split
dispatches pooled into the median. Recomputed with them excluded it is 14.3% —
the conclusion survives, the number moved.

**How far up this could be measured was partly our own limitation.** Three drafts
said the quantity was unmeasurable above n=8 because the scheduler split every
dispatch. Splitting turned out to track *request count*, not token count — 8192
tokens at `max_num_batched_tokens=8192` never splits, ~1024 tokens at n=8 splits
half the time — which is an arrival race, not a capacity limit:

| n | split, old launcher | split, synchronised launch |
|---|---|---|
| 4 | 0% | 0% |
| 8 | 20% | **0%** |
| 16 | 100% | **60%** |
| 32 | 100% | 100% |

A thread barrier cut arrival spread 7.6× at n=32 (15.4 ms → 1.7 ms). **The real
barrier is between 16 and 32.** The peer review proposed sweeping
`max_num_batched_tokens` to fix this; that experiment would have measured nothing.

**No shape is claimed for the dependence.** A within-bucket slope sweep gave
1.61 / 0.75 / 17.18 µs/token at n=1/2/4, but the third rests on a single
measurement, and all three sequence lengths there pad to the same sequence *and*
token bucket, so no padding model predicts a difference.

### 3.4 Padding is abundant, workload-dependent, and mostly free

Across four prompt-length distributions at one Poisson arrival rate (8 req/s,
`output_len=64`, 120 requests each):

| distribution | CV | padded share of executed tokens | TTFT p50/p95 | ITL p50/p95 |
|---|---|---|---|---|
| fixed-256 | 0.00 | **51.0%** | 19 / 27 ms | 4.4 / 5.1 ms |
| lognormal | 1.20 | 38.4% | 19 / 100 ms | 4.4 / 7.5 ms |
| bimodal | 1.30 | 32.7% | 17 / 108 ms | 4.8 / 7.5 ms |
| uniform | 0.60 | **27.3%** | 89 / 261 ms | 6.4 / 14.9 ms |

**27.3% to 51.0%** — and the ordering is counter-intuitive: the *most uniform*
workload pads *most*, because a fixed length just above a boundary pads every
step by the same large amount, while a spread distribution lands across buckets
and averages out. Any single figure characterises a workload, not the stack.

Per-request *length* padding does not exist: cost tracks packed tokens, the
batch-padding model is rejected by 44–618%, and uniform controls where all
candidate models agree match to 1.9%. **Not chunked prefill** — with
`--no-enable-chunked-prefill`, packed still wins 8/10 ragged cells and batch
padding is rejected by 75–579%.

### 3.5 Decode is smooth, and bandwidth is why

| n | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ms/step | 3.80 | 4.25 | 4.30 | 4.98 | 6.52 | 9.13 |
| µs/step/sequence | 3802 | 2127 | 1075 | 622 | 407 | **285** |

Per-step cost rises **2.4×** while batch size rises **32×**; per-sequence cost
falls **13×** monotonically, no discontinuity.

**A roofline from measured step times and published dimensions gives the
mechanism**, computed offline for $0:

| n | achieved HBM BW | BW utilisation | MFU | bound by |
|---|---|---|---|---|
| 1 | 532 GB/s | 64.9% | 0.27% | memory |
| 8 | 421 GB/s | 51.4% | 1.68% | memory |
| 32 | 258 GB/s | 31.4% | 3.65% | memory |

**2.01 GB of weights crosses HBM every decode step regardless of batch size** —
99% of all bytes at n≤2, 89% at n≥16. Every cell is memory-bound; MFU never
exceeds 3.65%. Additional sequences, real or padded, are nearly free until that
floor is left. This is the memory-bound regime **Pope et al.** characterise
analytically; the measurement lands in it, and the contribution is the
consequence for compiled-shape ladders, not the regime.

**The roofline is not independent evidence about step time, and we should not
have implied it was.** Achieved bandwidth is `bytes / measured time`, so it is an
algebraic restatement of the step time it is computed from; what it independently
establishes is the *byte accounting* — that the weight term dominates, and that
the compute roof is far away (MFU ≤3.65%). The operator profile below is genuine
independent evidence, and settles what kind of answer the n=4 convergence can
have:

| n | attention | collective | matmul/fusion |
|---|---|---|---|
| 1 | 6.8% | 13.5% | **78.5%** |
| 4 | 15.4% | 13.9% | 69.6% |
| 16 | 34.2% | 13.4% | **51.4%** |

Matmuls — where weights are read — dominate at low batch and give way to
attention as KV grows; collectives are flat. **Nothing moves discontinuously at
n=4**: every category's share changes less into n=4 than across some other
adjacent pair. The convergence is not visible at operator granularity.

**The mechanism, isolated.** A matmul microbenchmark with the model's real
sharded weight shapes, no server and no scheduler, times the same matmul at
M = 1 … 256 rows:

| M | 1 | 8 | 64 | 256 |
|---|---|---|---|---|
| qkv_proj | 142.9 µs | 142.5 | 140.7 | 143.6 |
| per row | 142.9 | 17.8 | 2.20 | **0.56** |

**Total time is flat across the whole range** — 1.04× from M=1 to M=256 — so
per-row cost falls as 1/M with no knee anywhere. This is the weight-load floor
with every confound removed, and it is the cleanest statement of the mechanism in
this paper. It also rejects the tiling hypothesis for §4.3's n=4 convergence: an
MXU tile boundary would produce a knee, and there is none at 4 or anywhere else
below 256.

**Where does free padding end?** Decode is measured to n=32 and the compiled
request ladder runs to 256, so the claim needs a bound rather than an
extrapolation. From the same byte accounting: as batch grows the weight term is
amortised away and arithmetic intensity rises toward a limit set by the
per-sequence terms, against a ridge point of 241 FLOP/byte.

| context | limit (FLOP/byte) | margin to ridge | MFU at n=256 |
|---|---|---|---|
| 256 | 217 | **10%** | **49%** |
| 1024 | 57 | 76% | 20% |
| 4096 | 17 | 93% | 7% |
| 8192 | 11 | 96% | 4% |

There is no formal crossover within n≤4096 — KV bytes grow with batch alongside
the flops — so the whole ladder stays nominally memory-bound. **But the margin is
not uniform.** At 256-token context it is 10%, and MFU at the top of the ladder
reaches 49%. Free padding is comfortable at long context and marginal at short
context and high batch. This is a bound, not a measurement, and it is
falsifiable: measuring a nonzero paid share at n=64 or n=128 well inside the
frontier would show the mechanism is incomplete.

### 3.6 The cheapness of padding is not an artifact of the sharding

Holding model, chips and workload fixed and varying only TP. The prediction was
registered in the configs *before* measurement: level scales 1/TP, shape
preserved.

| TP | level vs TP=4 | predicted | cost rise n=1→32 |
|---|---|---|---|
| 4 | 1.00× | 1.00× | 2.33× |
| 2 | 1.63× | 2.00× | 2.41× |
| 1 | 2.86× | **4.00×** | **1.83×** |

**Both halves missed.** Level scales sub-proportionally — the roofline does not
model the inter-chip collectives the higher-TP arms pay. And shape is not
preserved: the curve gets *flatter* with less sharding, which a larger per-chip
weight floor implies.

Both misses point the same way and answer the objection. If request-dimension
padding were cheap only because that dimension is not the bottleneck at TP=4,
reducing TP would expose it. Instead padding is cheapest at **TP=1**, where the
floor is largest.

### 3.7 Per-dispatch variance is a prefill phenomenon

Decode **1.00–1.04×** over 9 repeats at most batch sizes; prefill **1.00–1.03×**
at n≤4 and **1.18–1.26×** at n≥8. Variance appears exactly where the scheduler
begins splitting, and decode — which has no chunking decision — never shows it.
Localisation, not mechanism: step *count* does not correlate with cost within a
cell.

That aggregate hides real spread. Bootstrapping the decode cells §3.1 depends on
gives 95% interval widths of **38.7% at n=8** and 28.2% at n=9 over 21 repeats.

---

## 4. What was withdrawn

Five claims that were written down and are no longer made.

| withdrawn | why |
|---|---|
| **"Step-for-alignment crosses at ~2048 tokens"** | Combined a step cost measured at n=1 with a padding fraction measured at n=4. Killed by the invariance guardrail. |
| **"~4–9% of execution is recoverable"** | Multiplied a padded share from one workload by a paid share at one batch size. Appeared in the abstract, §4.4, §7 and §9 across three drafts. Killed by the same guardrail once registered as a claim. |
| **"36% of executed tokens are padding"** (as a stack property) | It is a workload property; the range is 27.3–51.0%. |
| **"Unmeasurable above n=8, the regime production runs in"** | Partly our own launcher. n=16 is measurable and the paid share there is ≈0%. |
| **"LENS does not transfer"** (on the MAPE shift) | Narrowed to the *localisation* of the error, plus the stronger claim that the length term never earns its place. |

---

## 5. Methodological findings

Nine invalid inferences, in two classes.

**Class one — provenance (six).** A quantity measured under one configuration,
used under another: a cost model fitted at `output_len=8` and run at 1; "padding
is free" from two experiments both right at different n; a curve extrapolating
small steps 15× low; a decomposition winning in model and losing measured; a
scheduler patch inert because its prompts were in the losing regime; and a "fixed
cost" that was not constant.

The guardrail took **three versions**. *"No derivation may combine quantities
measured at different batch sizes"* would not have caught the `output_len`
failure. A whitelist of config keys missed the largest error, because batch size
is not a top-level field — it lives inside experiment-specific structures. The
working form diffs **every** config key across a claim's source runs, exempts
only free text, and requires each difference to be named. It flagged five claims
already believed correct and has since killed two headline numbers.

**Its coverage is the set of *registered* claims, not the set of claims made.**
The recoverable-headroom figure evaded it for three drafts by living in prose.

**Class two — instrument definition (three), which the guardrail does not see.**
A request-scoped metric contaminated by prefill/decode interleaving; a step-count
test that could never pass (asking whether a *whole dispatch* ran in one step,
which is never true, instead of whether its *prefill* split); and a boundary
experiment that counted split dispatches but pooled their cost into the median
anyway, contradicting the rule the method section states.

The last one mattered: harmless while splits were zero at n≤8, wrong the moment
n=16 became reachable. Excluding them moved the result under a percentage point —
**luck, not method**, since pooling biases the paid share *upward*, the direction
that manufactures a positive result.

Two parser defects in session 15 belong here too: pooling the "XLA Modules" lane
with "XLA Ops" double-counted every op inside its module, and the attention
pattern could not match `RPAd`. Both surfaced as an impossible 0.0% attention
share.

**Every class-two error was caught by a measurement disagreeing with an
independent one. That is not a mechanism, and it is the open methodological
problem.**

### Other machinery that paid for itself

- **The traceability contract aborts, it does not warn.** `meta.json` first,
  config hash, git SHA, dirty flag, never overwritten. It has fired three times
  on real mistakes — and once on a legitimate experiment, blocking the TP
  ablation for a session because it could not distinguish declared variation from
  drift. Fixed by letting an experiment *declare* a control as its independent
  variable, with a mandatory reason that lands in `meta.json`.
- **`check_model.py`** preflights four failure modes that each cost a server boot
  to discover. It retrodicts all four.
- **A correctness gate.** Bucket-aligned packing measured **−29% TPU time and
  −49% p99**; the gate then showed 4 of 48 greedy completions differed, every one
  at a prompt length just above a bucket boundary. The patch was silently
  dropping prompt tokens. Without the gate it ships as a 29% throughput win.
- **`--parse-only`.** Trace parsing was wrong twice; both fixes cost $0 against
  bytes already pulled.

---

## 6. Four optimisations, measured and rejected

| | outcome |
|---|---|
| bucket-aware admission control | premise false |
| ladder redesign | D1 does not exist; D3 inert by default |
| last-chunk decomposition | **20.6% worse** measured (51.06 vs 42.33 ms) |
| bucket-aligned step packing | implemented twice: inert, then output-corrupting |

The one positive measurement — release timing saving 26% of TPU time at 25 req/s
(p=0.001, six paired seeds) — is **dynamic batching**, not a shape effect, and is
a single load point with no saturation curve.

---

## 7. Limitations

- **One accelerator, one primary model.** The sharding objection is answered
  (§3.6); model scale and multi-host topology are not, and both change what
  padding hides under.
- **No production trace.** The four length distributions span a plausible range;
  the result is a sensitivity, not a corrected point estimate.
- **Prefill step cost above n=16 is still not isolable.** At n=16 the clean
  sample is 7–11 dispatches per arm.
- **The n=4 convergence is unexplained, and is not an operator effect.** We no
  longer expect a profiler to show it.
- **Co-located prefill and decode only.** Disaggregation splits the question in
  two.

---

## 8. Cost

| phase | sessions | spend |
|---|---|---|
| Bring-up through the premise falsification | 1–6 | ~$21 |
| Mechanism, LENS, regime map | 7–12 | ~$20 |
| Review response: workload, launcher | 13 | [redacted] |
| Review response: TP ablation, n=16 | 14 | [redacted] |
| Review response: operator profile | 15 | [redacted] |
| **Total** | **15** | **[redacted]** of $1,000 |

A separate GCE instance unrelated to this project, running since 2026-07-31,
has cost roughly **four times** the entire research programme. The per-session
teardown discipline here was built for $4.80/hr TPUs and would never have caught
it — a resource that never had a session has no session boundary to be torn down
at.

---

## 9. Artifacts

- `notes/paper_draft.md` — the paper, draft 3
- `scripts/paper_numbers.py` — 66 claims tied to `run_id`s, recomputed
- `reproduce_all.sh` — regenerates every number and figure; non-zero on any
  disagreement
- `captured/` — raw output of all fifteen sessions
- `DECISIONS.md` — per-session log: cost, findings, and what each invalidated
