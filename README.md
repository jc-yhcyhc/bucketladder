# bucketladder

**What compiled-shape padding actually costs in production TPU serving.**

A measurement study of vLLM 0.25.0 + `tpu-inference` 0.25.0 on `v5litepod-4`.
Fifteen hardware sessions, **[redacted]**, 66 verified claims, 184 tests.

TPU executables are compiled for fixed tensor shapes, so a serving stack rounds
every workload up to a precompiled ladder. The obvious inference — that rounding
up means paying for what you rounded up to — motivates length bucketing,
shape-aware admission control and ladder design. This repo set out to build the
admission policy, found the premise false, and measured what is actually true
instead.

---

## Findings

**The attention ladder is not the one the system reports.** Every boot prints a
six-entry request ladder, and attention ignores it:

```
Prepared request paddings:      [8, 16, 32, 64, 128, 256]
Prepared attn request paddings: [256]        <- one bucket
```

`ATTN_BUCKETIZED_NUM_REQS` defaults off, so attention always executes at 256
requests whatever the batch size — the profiler shows the compiler emitting the
kernel as `RPAd-p_256-…`, with the padding in its name. Hardware agrees — if decode padded 9 sequences
to 16, n=9 would cost what n=16 costs (91.8 ms); it costs what n=8 costs (51.4).
Enabling the flag changes decode by **0.0%**. Absent from the RPA paper, from
LENS, and from vLLM's TPU documentation.

**A published latency predictor's length term never earns its place.** LENS
reports 2.15% on NPUs using a per-bucket linear model. Reproduced on TPU with a
withheld point: **5.23% MAPE, worst 22.4%**. A constant-only predictor with no
length term matches it at batch sizes 1–2 (0.96% vs 0.38%) and **beats** it at 4
(14.80% vs 19.77%) — so the near-perfect accuracy at low batch size is evidence
about the fit protocol, not the model form.

**Step cost is not a property of the step.** It depends on batch size: ~85% of
nominal padding is paid at n=1–2, 10–25% at n=4–8, and **approximately none at
n=16**. How far up this is measurable was partly our own limitation — splitting tracks request count, not tokens, and
releasing requests from a thread barrier cut arrival spread 7.6× and made n=16
measurable for the first time — where the paid share is indistinguishable from
zero across three boundaries. A TP=1/2/4 ablation shows this is not an artifact
of the sharding: padding is cheapest at TP=1, where the per-chip weight floor is
largest.

**Padding is abundant, and how abundant is a property of the workload.** Across
four prompt-length distributions at one arrival rate the padded share of executed
tokens spans **27.3% to 51.0%** — and the *most uniform* workload pads most,
because a fixed length just above a boundary pads every step identically. Only
10–25% of it is paid. Per-request length padding doesn't exist at all — a
mixed-length batch costs its packed tokens (batch-padding model rejected by
44–618%), and not because of chunked prefill: disabling it changes nothing.

**Decode is well-behaved, and bandwidth is why.** Per-step cost rises 2.4× while
batch size rises 32×; per-sequence cost falls 13× monotonically. A roofline gives
the mechanism: **2.01 GB of weights crosses HBM every decode step regardless of
batch size** — 99% of all bytes moved at n≤2 — and MFU never exceeds 3.65%. Every
cell is memory-bound, so extra sequences, real or padded, are nearly free until
that floor is left. The pathology is real and lives in the phase that matters
least.

Full detail: [`notes/research_summary.md`](notes/research_summary.md).

---

## What was tried and rejected

Bucket-aware admission control (premise false) · ladder redesign (nothing to
optimise) · last-chunk decomposition (**20.6% worse** measured) · bucket-aligned
step packing (implemented twice: inert, then output-corrupting).

Every negative result is recorded with the measurement that killed it. The one
positive number — a 26% TPU-time saving from release timing — is reframed as
ordinary dynamic batching rather than a shape effect, because that is what it is.

---

## Why the numbers are trustworthy

**The traceability contract aborts, it doesn't warn.** Every run writes
`meta.json` before doing any work, records config hash / git SHA / dirty flag,
and appends to a manifest. Runs are never overwritten. A config that can't prove
prefix caching is off refuses to run — this has fired twice on real mistakes.

**`scripts/paper_numbers.py`** ties all 66 claims to `run_id`s and recomputes
them from captured data. On first run it found two real defects: a cost
transcribed from an exploratory dump rather than the fitted curve, and a table
presented as precise that came from one of two replicates differing by 14%.

**An invariance guardrail.** Most of this project's errors share one cause — a
quantity measured under one configuration used under another. The check diffs
every config key across a claim's source runs and requires each difference to be
explicitly asserted. *It needed three versions before it caught the error it was
built for*, it flagged five claims already believed correct, and it has since
killed two of our own headline numbers: a crossover point and a
recoverable-headroom figure. Its coverage is the set of **registered** claims, so
the headroom figure evaded it for three drafts by living in prose.

**`scripts/check_model.py`** preflights a model in seconds against four failure
modes that each cost a server boot to discover — a registered-but-broken
architecture, missing safetensors, `inv_freq` buffers the loader rejects, and TP
indivisibility. It retrodicts all four.

---

## Errors caught, and one that mattered

An implementation of bucket-aligned packing measured **−29% TPU time and −49%
p99**. The correctness gate then showed 4 of 48 greedy completions differed —
every one at a prompt length just above a bucket boundary. The patch was silently
dropping prompt tokens. Without that gate it ships as a 29% throughput win.

Others, all self-caught and all documented rather than quietly fixed: a cost
model built on a 3-repeat median of a bimodal cell; a "provable bound" a policy
beat; an `Oracle` documented as an upper bound it wasn't; "padding is free"
written three days before measuring that a quarter of it isn't; and a headline
optimisation rejected by judgement overriding a pre-committed decision rule.

---

## Reproducing

```bash
python -m pytest tests/ -q                 # 184 tests, no hardware needed
python scripts/paper_numbers.py            # recompute all 66 claims from captured/
python scripts/check_model.py <hf-model>   # preflight before provisioning
python scripts/refit_cost_model.py         # refit + both holdouts, offline
```

Hardware sessions: `infra/create_tpu.sh` → `infra/deploy.sh` →
`infra/serve_remote.sh start <model>` → experiment → `infra/capture.sh` →
`infra/teardown_tpu.sh`. Every session's raw output is under `captured/`.

`DECISIONS.md` is the running log: every session, what it cost, what it found,
and what it invalidated.
