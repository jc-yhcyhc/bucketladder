# Bucket Ladder / Admission Policy — Assessment and Revised Plan (v2)

## Context

You asked what I think of the bucket-ladder cost plan. This is the second pivot in a week — `gapcache` was killed by its own prior-art gate on Aug 2, the determinism version got a v1 review this afternoon (`please-see-this-and-modular-wave.md`), and this doc demotes determinism to a paragraph and reframes around cost.

The reframe is a good instinct and the doc is a real improvement on v1 in two places: it has a "Why this is not already done" section (the exact discipline `gapcache` lacked), and it correctly promotes the runtime edge cases from v1's review into a full phase.

Your three answers set the shape of everything below:

- **Budget under $1k.** Survivable, but only on the right chip — the `v6e-8` in the doc's `meta.json` example puts the grid 2.5× over the ceiling, while spot v5e-4 puts it comfortably under. The binding constraint turns out to be wall-clock and statistical power, not money. See the arithmetic below.
- **Determinism fully dead.** Cost only. The v1 sufficiency-boundary material is retired; don't carry the `DeterministicInfer` name into the paper.
- **Admission policy (E3.3) is the spine.** Ladder characterization becomes the setup, not the claim.

Intended outcome: an MLSys 2027 industrial-track submission (~2026-10-30, verify) on **bucket-aware admission control for ragged workloads on compiled-shape accelerators**, calibrated on real TPU, swept in simulation, validated on held-out hardware runs.

---

## The budget arithmetic, because it decides the plan

The doc never states a hardware config except a stray `"tpu_type": "v6e-8"` in the `meta.json` example. That one field is the difference between the plan fitting and not fitting.

Steady-state measurement in the doc as written:

| Phase | Grid | Runs | Hours @ 360s/run |
|---|---|---|---|
| E1.1 | 3 traces × 6 rates × 3 repeats | 54 | 5.4 |
| E1.3 | 12 ladders × 4 traces × 2 rates × 3 repeats | 288 | 28.8 |
| E4.1 | 4 ladders × 2 admission × 2 models × 3 traces × 6 rates × 3 repeats | 864 | 86.4 |
| **Total (excl. Phases 2, 3, E1.2)** | | **1,206** | **~121** |

At the v6e-8 named in the doc's own `meta.json` (~$21/hr) that is **~$2,500 of steady state alone** — 2.5× over the ceiling before Phase 2, Phase 3, bring-up, or a single failed run. At v5e-4 it is ~$580 on-demand and ~$170 on spot. So the chip choice, not the grid size, is what decides whether this fits; see the hardware section below.

What the grid does *not* survive is the calendar. Two costs the doc omits entirely, both of which it should have caught since compile cost is its own central premise:

1. **Every ladder variant requires a recompile.** Your own calibration doc puts XLA warmup at 5–30 min first-bucket and 30–120 s per additional bucket. A 32-bucket ladder is a meaningful warmup. E1.3 sweeps ≥12 ladders × 2 models — that's 24 server bring-ups minimum, and the doc budgets zero time for them while simultaneously arguing that compile budget is the thing that makes this problem interesting.
2. **Discarded warmup.** §0.4 throws away the first 60 s of every run. On 1,206 runs that's 20 hours of paid-for, discarded TPU time.

On 1,206 runs those two together are days of wall-clock in a 12-week schedule with no slack.

**Resolution: v5e-4 on spot, and move the sweep into a calibrated simulator.** This is not a compromise — it is the methodology that already produced a paper for you, and the reusable pieces are on disk.

---

## What I'd change, in priority order

### 1. There is no bring-up phase, again

`/home/yhcyhc1991/DeterministicInfer` is empty. No JAX, no vLLM, no TPU on this host — I checked. The doc's "Start here" says E1.2 "is one day, requires no ladder design." It requires a working `tpu-inference` stack, which does not exist. This is the same omission v1 flagged and it has come back.

### 2. Chunked prefill may delete L1 entirely — verify before anything else

The doc's L1 level is "total prompt tokens → prefill bucket." If `tpu-inference` runs chunked prefill (vLLM V1 enables it by default on other backends), long prefills are chopped into fixed-size chunks and **L1 padding is bounded by one chunk, not by the ladder**. The prefill ladder stops being an interesting cost parameter and the three-level story collapses to two.

This is a single-afternoon check and it is load-bearing for a third of the paper. The doc does not mention chunked prefill anywhere. Neither does it mention prefix caching, which changes measured prompt-token counts and would silently corrupt every L1 padding number. v1's controlled-variables contract was dropped in the pivot; it matters *more* under a cost framing, not less.

### 3. State what padding survives RPA — a reviewer will lead with this

Ragged Paged Attention exists specifically to remove padding waste from variable-length batches. The doc cites RPA as motivation but never says what padding it leaves behind. If the answer is "the outer compiled shape and the block tiling granularity," say exactly that, early, with a measurement. Otherwise the first review reads: *RPA already solved this.*

### 4. Run the prior-art gate before writing code, and search wider than two papers

`gapcache` died because forward/keyword search surfaced five systems the reading list didn't anticipate. This doc names two works (RPA, BucketServe). The exposure here is not BucketServe — it's that **shape bucketing under a cardinality budget is old and adjacent to textbook 1-D quantization**. Search list, ~half a day:

- Forward citations of RPA and BucketServe.
- Compiled-shape bucketing outside LLMs: TensorRT optimization-profile selection, ONNX Runtime dynamic shapes, TF Serving batch buckets, `tf.data.bucket_by_sequence_length`.
- **JAX/XLA shape polymorphism** (`jax.export` polymorphic dims). If the prefill dimension can be compiled polymorphically, the ladder is optional and the premise weakens badly. Cheap to check, high impact.
- Zero-padding / varlen transformer serving (ByteTransformer, Effective Transformer) — the "eliminate padding instead of bucketing it" answer.
- **The spine's actual neighbourhood:** admission control and batch composition in LLM serving — Sarathi-Serve, QLM, Andes, Llumnix. Specifically: has anyone published "promote to a larger batch bucket and eat the padding" vs "wait for the right bucket"?
- **Vidur** (LLM inference simulator with fidelity validation). Directly relevant methodologically; a reviewer will ask why you didn't use it. Either use it or state why not.

### 5. Bound the headroom on day one, before designing anything

Missing from the doc: **what is the cost gap between the default ladder and an infinite ladder?** Compile a handful of exact shapes for lengths drawn from the agentic trace, measure against the default ladder. That is the ceiling on everything Phase 2 could ever win. It costs under a day and under $20. If the oracle gap is 4%, you have learned that in week 0 rather than week 6.

Pair this with E1.2 (narrowed) and you bound the whole paper's headroom in two days.

### 6. Phase 2's method should be dynamic programming, not Lloyd–Max

Two problems with the doc's formulation. First, the objective: minimizing padded *tokens* is not minimizing cost — padding from length `L` to ceiling `B` costs `C(B) − C(L)`, and prefill cost is superlinear in length. Optimize on the measured cost curve, not token counts. Second, the method: under a hard cardinality budget with a separable additive cost, a 1-D DP over a discretized length axis gives the **global** optimum in `O(K·N²)`, which is milliseconds for a few thousand candidate edges. Lloyd–Max is an iterative local-optimum scheme and is strictly worse here for no benefit. Keep Lloyd–Max only as a baseline to beat.

With admission as the spine this phase shrinks to a section, which is the right size for it.

### 7. Smaller fixes

- **E1.2's sweep is 20× larger than it needs to be.** `1:2048:1` × 5 repeats = 10,240 isolated requests. You only need density near edges: sample every 64 tokens globally, then ±8 tokens around each edge from the ladder enumeration. Also, `tpot_ms_p50` from a prompt-length sweep needs a specified output length — TPOT is not a function of input length in the same staircase way.
- **Don't invent a raggedness index.** "Define and defend" is a reviewer magnet for a quantity that isn't your contribution. Use coefficient of variation, or better, use expected padding fraction under the default ladder — which is the objective you already care about.
- **L3 is not a ladder parameter.** L1/L2 are compiled-shape edges; L3 is kernel tiling, conventionally autotuned. Grouping them as "three quantization levels nobody has separated" inflates the claim and invites the response that kernel autotuning is well-studied. Report L3 as a measured third source of shape-induced waste, not as a third ladder level. It is also the doc's own highest-instrumentation-risk item.
- **Define the cost model explicitly.** `eval_cost.parquet` asserts TPU-seconds per 1M output tokens, but padding costs money only when you're throughput-bound; at low rate it costs latency. The padding→dollars relationship is rate-dependent. You did this properly in `infersim`; do it again.
- **Gate thresholds need units.** "Padding waste > 10%" — of tokens, or of TPU-seconds? Only the second one is the paper's metric.
- **Drop the negative-result escape hatch as a landing zone.** "The default ladder is near-optimal, and *that* is a finding" is a workshop paper, not an industrial-track paper. Your actual safe landing is the admission policy, which is now the spine — that's the right structure and it makes the hatch unnecessary.

---

## Revised plan

### Hardware: v5e is already the cheap option — older is not

Verified pricing (Aug 2026), against `tpu-inference`'s compatibility list:

| TPU | On-demand $/chip-hr | Spot | `tpu-inference` status |
|---|---|---|---|
| v4 | [redacted] | ~[redacted] | experimental |
| **v5e** | **[redacted]** | **~[redacted]** | **recommended** |
| v5p | [redacted] | ~[redacted] | experimental |
| v6e | [redacted] | — | recommended |
| v7x | — | — | recommended |

Going older costs *more*, not less: v4 is 2.7× v5e on-demand and is experimental in `tpu-inference`, as are v3 and v5p. An experimental backend is the worst place to run this particular paper, because every pathology you find becomes ambiguous between "real finding" and "porting artifact" — the same argument your doc already makes for preferring Gemma's well-tested path.

**Decision: v5e-4 (`v5litepod-4`), single-host, TP=4 fixed.** It is the cheapest chip *and* on the recommended list. Your $4.80/hr on-demand figure in `infersim/calibration/fitted_params.yaml` is measured rather than guessed, `vm_setup.sh` already encodes the `tpu-ubuntu2204-base` image workarounds (gcsfuse list, broken `cnf-update-db`, distutils pip, `/usr/local` ownership) that otherwise cost a day, and a fixed single-host topology removes TP size as a confound for free.

Gemma-3-4b at bf16 is ~8 GB of weights and fits comfortably; Llama-3.1-8B at ~16 GB fits with reduced `max_model_len`. Verify both against the recommended-models list at a pinned version in W0, and verify Gemma's sliding-window path uses the RPA kernel rather than falling back — the doc is right that a fallback demotes Gemma to secondary.

### Budget: spot for sweeps, on-demand for anything you intend to keep

Spot v5e-4 is ~$1.40/hr against [redacted] on-demand. Split by whether a preemption is recoverable:

| Allocation | Hours | Pricing | Cost |
|---|---|---|---|
| W0 bring-up, smoke tests, debugging | 40 | on-demand (stability while debugging) | $192 |
| Primitive measurement — staircase, promotion cost, oracle gap | 40 | spot (re-runnable) | $56 |
| Holdout validation | 30 | on-demand (a preempted holdout run is a corrupted holdout run) | $144 |
| Reserve — failed runs, reruns, second model | 46 | mixed | ~$150 |
| **Total** | **156** | | **~$540** |

That is comfortable under $1k with real slack. Detect preemption and discard the affected run automatically — record `preempted: true` in `meta.json` and exclude from paper figures, same rule as `dirty: true`.

Worth one email regardless: **TPU Research Cloud** grants free TPU access for academic work and would take the budget question off the table entirely. Check which generations they currently offer — if it is v2/v3 only, it does not help here, for the compatibility reason above.

### Method: calibrate on hardware, sweep in simulation, validate on holdout

**Correcting my own framing from earlier in this document:** at spot v5e pricing, your original all-hardware grid comes to roughly $170–400, so money was never the real objection once the chip is right. The case for simulation is different and, I think, stronger:

1. **Wall-clock.** 121 hours of steady state plus ~24 ladder recompiles plus preemption retries does not fit a 12-week schedule that also contains bring-up and write-up. Spot capacity for v5e-4 can be unavailable for hours at a stretch; wall-clock becomes unpredictable in a way money does not.
2. **Statistical power.** 3 repeats on hardware supports medians with min/max. 30 seeds in simulation supports matched-trace paired bootstrap tests — which is what `infersim` reported and what let you write "p = 0.19" and "p < 0.001" instead of "appears better." For a policy comparison paper, that difference is the result.
3. The combinatorics live in the policy × ladder × trace × rate sweep, and that is exactly the part a calibrated simulator handles for free.

Name the weakness before a reviewer does: a simulator result is weaker than a hardware result. The mitigation is the one `gcp_calibration_plan.md` already specifies — MAPE < 15% on every calibrated parameter, validated on a held-out workload the fit never saw, reported per parameter in the paper.

The saved budget buys back scope: with meaningful headroom left against the ceiling, the second model (Llama-8B) and a wider holdout are affordable rather than aspirational.

### W0 (Aug 7–14) — bring-up and kill checks, blocking

Nothing downstream is meaningful until these pass. Run them in this order; each can end the project cheaply.

1. Provision v5e-4; install `tpu-inference` + vLLM at a pinned version. Start the gated `meta-llama` repo access request **first** — it is a common multi-hour stall.
2. **Chunked prefill and prefix caching audit.** Determine the defaults, force both to a known state, record in `DECISIONS.md`. If chunked prefill is on by default and cannot be disabled, L1 is largely dead — reframe to L2 + admission and say so.
3. **Ladder enumeration** (the doc's E0.2). Parse warmup logs and the compilation cache into `ladder_default.json`. Without this there is no independent variable.
4. **Oracle headroom probe.** Exact-shape compiles vs default ladder on agentic-trace lengths. Bounds the entire contribution.
5. **Prior-art gate**, per the search list above. Written verdict in `notes/kill_condition.md`, same format as `gapcache`'s.

**Gate:** ladder enumerated, controlled variables pinned, oracle gap measured, prior-art verdict written. A negative on 4 or 5 stops the project in week 0 for ~$200, which is the entire point of running them first.

### W1–3 — primitives on hardware

- **Traces** (CPU, parallel with bring-up): ShareGPT, agentic tool-use, long-context, synthetic-controlled. Keep the doc's `trace_stats.parquet` schema. Drop the invented raggedness index.
- **E1.2 staircase, narrowed** as described in §7. Produces the paper's motivating figure and the L1 cost curve `C(B)`.
- **Promotion cost primitive** — the measurement the spine depends on and the doc does not isolate: for a request of length `L` served in bucket `B`, the marginal latency and TPU-seconds cost of `B` vs the tightest fitting bucket. This is the simulator's core parameter.
- **Queue-wait primitive** — occupancy and admission behaviour at a saturated bucket.
- **Padding decomposition** at a small number of rates only (not the doc's 6-point sweep) — enough to fit and validate, not to sweep.

### W4–6 — simulator and policy sweep (free)

Extend the `infersim` shape: `sim/`, `policies/` as ABC-with-hooks, `eval/` with matched-trace evaluation and paired bootstrap. The five-hook `CapacityPolicy` interface in `infersim/policies/capacity/base.py` is the right template; here the hooks are bucket-scoped rather than machine-scoped.

Policies to compare: `queue`, `promote`, `hybrid` (promote below a padding-cost threshold, queue above), plus an oracle that knows future arrivals as an upper bound.

Sweep across ladders (default, quantile-k, DP-optimal, agentic-fitted), traces, and request rates, 30 seeds each. This is the paper's core result table and it costs nothing.

### W7–8 — holdout validation on hardware

Reserve the TPU window in advance. Take ~12 configurations spanning the policy × rate space, run them for real, report predicted vs measured with MAPE. Any parameter over 15% gets refit and the sweep re-run — the rule you already used.

Also here: **distribution shift** (the doc's E2.3). Fit on the agentic trace, evaluate on the others, report regret vs oracle and vs default. "How far can the workload drift before the fitted policy loses" is the question reviewers will ask, and having a number beats hedging.

### W9–12 — write-up, with slack

Draft intro and related work during W4–6; they depend on no results and the prior-art gate already wrote most of related work. Keep §0's traceability contract **unchanged** — directory layout, `RUN_ID` composition, `meta.json`-written-first, Parquet, never-overwrite, `MANIFEST.jsonl`, `start_run`/`save_table`/`finish_run` in `try/finally`. It is the strongest part of the doc and it matches what `infersim` already does. Keep `paper_numbers.parquet` with `claim_id` indirection; it is `extract_paper_numbers.py` done properly.

Protect the ~2 weeks of slack. Both prior projects consumed theirs.

---

## Reuse (don't rebuild)

| Need | Already exists |
|---|---|
| TPU VM image workarounds | `infersim/calibration/vm_setup.sh` |
| vLLM HTTP measurement harness | `infersim/calibration/measure_throughput_grid_vllm.py`, `measure_w1_warmup_vllm.py`, `measure_holdout_vllm.py` — server-agnostic over the OpenAI API, so they point at vLLM TPU unchanged |
| Parameter fitting + MAPE validation | `infersim/calibration/fit_parameters.py`, `process_calibration.py` |
| DES core, policy ABC, matched-trace eval | `infersim/sim/`, `infersim/policies/capacity/base.py`, `infersim/eval/{runner,schema,trace,aggregate}.py` |
| Paired bootstrap CI / permutation test | `infersim/extract_paper_numbers.py` (`bootstrap_ci`, `bootstrap_p`) |
| Reproduction entry point | `infersim/reproduce_all.sh` |

Per your own `gapcache` README convention: borrow engineering convention, not research content, and say so in the README.

## Where this lives

New repo at **`~/bucketlad`**, matching the layout root the doc already uses in §0.1. Leave `~/DeterministicInfer` alone — it is empty, the name no longer describes the work, and a fresh `git init` keeps the traceability contract's "git clean or `dirty: true`" rule honest from the first commit.

Follow the `gapcache` README convention: state up front that `infersim` is a separate artifact for a different paper, and that what is borrowed is engineering convention (DES shape, policy-as-ABC-with-hooks, result-schema dataclass, matched-trace evaluation, one script per finding) — not research content. That paragraph pre-empts a self-plagiarism question at review time, and you already have the wording.

## Files to create

- `scripts/_common.py` — `start_run`, `finish_run`, `save_table`, plus `assert_controlled_vars` (aborts if prefix caching or chunked prefill differ from the recorded config — abort, not warn)
- `scripts/e00_smoke_test.py` — W0 gate: ladder enumeration + controlled-variable audit
- `scripts/e00_oracle_gap.py` — headroom bound; the cheapest kill check
- `scripts/e12_latency_steps.py` — narrowed staircase sweep
- `scripts/e13_promotion_cost.py` — the spine's core primitive
- `sim/`, `policies/admission/`, `eval/` — ported shape from `infersim`
- `notes/prior_art.md`, `notes/kill_condition.md` — W0, before code
- `DECISIONS.md` — starts W0
- `configs/` — one YAML per variant, all committed, each carrying the controlled-variables block

## Verification

- **W0:** `e00_smoke_test.py` exits 0; `ladder_default.json` non-empty and matches warmup-log bucket count; a config with prefix caching in an unrecorded state aborts at `start_run` (test this deliberately).
- **Primitives:** re-running an identical config yields a byte-identical Parquet; `config_hash` stable across process restarts and dict ordering.
- **Simulator:** MAPE < 15% against holdout hardware runs on every fitted parameter, reported per parameter, not in aggregate.
- **End to end:** `reproduce_all.sh` regenerates every figure from `MANIFEST.jsonl` with no manual steps — the bar `infersim` already meets.
- **Spend:** a running total in `DECISIONS.md`, checked at each phase boundary against the table above.

## Open items

- **Confirm the MLSys 2027 industrial-track deadline and that the track exists this cycle.** The whole schedule keys off ~2026-10-30, which came from search, not an official CFP. MLSys has run an industry track inconsistently. Verify in W0 and pick a backup venue now.
- **Chunked prefill's default in `tpu-inference`** — the single fact most likely to reshape the paper.
- **Spot v5e-4 capacity in your zone.** If spot is routinely unavailable, the budget still works on-demand at ~$750; the schedule is what suffers. Check early — it changes how much stays on hardware.
- **TPU Research Cloud generations.** Free access would remove the budget constraint, but only if they offer v5e or v6e; v2/v3 are experimental or unsupported in `tpu-inference`.
