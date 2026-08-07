# Bucket-Aware Admission Control for Ragged Workloads on Compiled-Shape Accelerators
## Execution Plan v3 — 2026-08-07

## Context

This is the third topic in five days. `gapcache` died on its own prior-art gate on Aug 2. The determinism framing got a v1 review on Aug 7 and was retired the same day. The v2 assessment (`what-do-you-think-partitioned-kahan.md`) reframed the work around cost and named admission policy as the spine — the right call — but I then reviewed v2 against disk and found ten problems, four of them consequential: a budget line off by roughly 4×, three reuse claims softer than stated, an unnoticed tension between the plan's own optimizer and its own headline policy, and a missing baseline that decides whether there is a paper at all.

This document is v3: the full plan, standalone, with those corrections folded in. It supersedes v2 — you should be able to execute from this file without reading the previous two. It ends with the commit that puts it under version control.

**Intended outcome:** an MLSys 2027 industrial-track submission (~2026-10-30, *unverified — see Open Items*) on bucket-aware admission control for ragged workloads on compiled-shape accelerators. Calibrated on real TPU, swept in simulation, validated on held-out hardware runs.

**Decisions already locked:**
- Budget ceiling **under $1,000**.
- Determinism is **fully dead** — cost framing only; the `DeterministicInfer` name does not carry forward.
- **Admission policy is the spine.** Ladder characterization is the setup, not the claim.
- **You provision, I script.** Every hardware step is a re-runnable script with stated prerequisites. Nothing I run spends money without you creating the VM first.
- Repo lives at **`~/bucketladder`** (exists, empty). `~/DeterministicInfer` is left alone.

---

## The claim

Serving stacks on compiled-shape accelerators pad every request up to one of N precompiled bucket shapes. When a request arrives and its bucket is saturated, the scheduler faces a choice nobody has studied: **promote it into a larger bucket and pay the padding, or queue it and pay the wait.** The claim is that this decision is workload-dependent, that the right policy is measurably better than either fixed strategy, and that the gain is large enough to matter in dollars.

**Why this isn't already done** (the discipline `gapcache` lacked — the W0 gate tests it properly, but this is the going-in position):
- Ragged Paged Attention removes padding waste *inside* the attention kernel. It does not remove the outer compiled shape. The paper must state precisely what padding survives RPA — the outer executable shape and block tiling granularity — and measure it early. Otherwise the first review reads *"RPA already solved this."*
- BucketServe designs ladders. It does not study admission against a fixed ladder.
- Sarathi-Serve, QLM, Andes, Llumnix do admission and batch composition, but none on compiled-shape bucket ladders.

The gate in W0 is what decides whether that position survives.

---

## Hardware

**Decision: v5e-4 (`v5litepod-4`), single-host, TP=4 fixed.** Cheapest chip that is also on `tpu-inference`'s *recommended* list. Fixed single-host topology removes TP size as a confound for free.

**List prices** (to re-verify at provision time — these are published rates, not measurements):

| TPU | $/chip-hr | `tpu-inference` status |
|---|---|---|
| v4 | 3.22 | experimental |
| **v5e** | **1.20** | **recommended** |
| v5p | 4.20 | experimental |
| v6e | 2.70 | recommended |

Older is *not* cheaper: v4 is 2.7× v5e and experimental. An experimental backend is the worst place for this paper, because every pathology found becomes ambiguous between "real finding" and "porting artifact."

v5e-4 = **$4.80/hr on-demand**, **~$1.40/hr spot**.

> Correction carried from the v2 review: `infersim/calibration/fitted_params.yaml` contains `price_per_chip_hour: 1.2` — a list-price constant, not a measurement. What *is* measured in that file is `warmup_time_sec: 57.0`, `prefill_tokens_per_sec: 13310`, `decode_tokens_per_sec: 1874`, and it was calibrated against **Gemma-2 7B on JetStream/MaxText**, not vLLM. Do not cite it as a measured price.

Models: Gemma-3-4b (bf16, ~8 GB) primary; Llama-3.1-8B (~16 GB, reduced `max_model_len`) secondary. Verify both against the recommended-models list at a pinned version in W0, and verify Gemma's sliding-window path uses the RPA kernel rather than falling back — a fallback demotes Gemma to secondary.

---

## Budget — billed VM-hours, not benchmark-hours

**This is the correction that matters most.** A TPU VM bills while it *exists*, not while it runs a benchmark. v2 allocated "40 hours for W0 bring-up = $192," but bring-up is a week of calendar — gated-repo stalls, install, audits. A VM left up across that week is 168 h × [redacted] = **$806**, which alone nearly exhausts the ceiling.

**Teardown discipline, non-negotiable:**
- Delete the VM at the end of every working session. Re-create from `infra/create_tpu.sh` — which is why that script must be re-runnable rather than one-shot.
- Persist everything to GCS. Never leave state on the VM disk.
- Track **billed VM-hours** in `DECISIONS.md`, checked at every phase boundary. Benchmark-hours and billed hours are different numbers and only one is charged.

| Phase | Billed VM-h | Pricing | Cost |
|---|---|---|---|
| W0 — bring-up, gate checks, debugging | 30 | on-demand (stability while debugging) | $144 |
| W1–3 — primitives (re-runnable) | 45 | spot | $63 |
| W7–8 — holdout validation | 25 | on-demand (a preempted holdout is a corrupted holdout) | $120 |
| Reserve — reruns, second model, overrun | 40 | mixed | ~$128 |
| **Total** | **140** | | **~$455** |

Headroom to the $1,000 ceiling is ~$545 — which is **exactly one forgotten VM-week**. That is the risk the discipline above exists to prevent.

Spot preemption: detect it, record `preempted: true` in `meta.json`, exclude from paper figures automatically. Same rule as `dirty: true`.

Worth one email regardless: **TPU Research Cloud** grants free academic access and would remove the budget question entirely — but only if they currently offer v5e or v6e. v2/v3 are experimental or unsupported in `tpu-inference` and would not help.

---

## Method: calibrate on hardware, sweep in simulation, validate on holdout

> v2 argued that money forces simulation, then retracted it two-thirds of the way down ("money was never the real objection"). That argument is deleted. The real case is these two:

1. **Statistical power.** Three hardware repeats support medians with min/max. Thirty simulation seeds support matched-trace paired bootstrap tests — which is what let `infersim` write `p < 0.001` instead of "appears better." For a policy-comparison paper, that difference *is* the result.
2. **Wall-clock.** The full hardware grid is ~121 h of steady state plus ~24 ladder recompiles (XLA warmup is 5–30 min first-bucket, 30–120 s per additional bucket) plus preemption retries plus 20 h of discarded warmup. That does not fit 12 weeks alongside bring-up and write-up. Spot capacity can vanish for hours; wall-clock becomes unpredictable in a way money does not.

The combinatorics live in the policy × ladder × trace × rate sweep, which is exactly what a calibrated simulator handles for free.

**Name the weakness before a reviewer does:** a simulator result is weaker than a hardware result. Mitigation is the rule already used in `infersim` — **MAPE < 15% on every calibrated parameter**, validated on a held-out workload the fit never saw, reported **per parameter, not in aggregate**.

---

## Controlled-variables contract

v2 said this was "dropped in the pivot" but never restated it. Inlining it here so v3 is standalone. Every one of these is a confound orthogonal to shape; each must be pinned, recorded in `meta.json`, and **asserted at runtime**. Under a cost framing this matters *more*, not less — prefix caching silently corrupts measured prompt-token counts, which corrupts every padding number in the paper.

| Variable | Required setting | Why |
|---|---|---|
| **Prefix caching (APC)** | **off, asserted** | Changes measured prompt-token counts. Alone, invalidates all L1 padding data. |
| **Chunked prefill** | off for primitives; a recorded variant later | See kill check below — this one can delete a third of the paper. |
| **Tensor parallelism** | fixed at 4, recorded | Held constant; state as a limitation. |
| **Speculative decoding** | off | Changes accepted-token paths entirely. |
| **KV cache dtype** | explicit, recorded | |
| **Server restart** | recorded as its own axis | Is the ladder stable across restarts and recompiles? |
| **`XLA_FLAGS` / libtpu flags** | verbatim in `meta.json` | Load-bearing. |

`assert_controlled_vars(config)` runs inside `start_run` before any work. A run that cannot prove APC is off **aborts — it does not warn**.

---

## Traceability contract

Carried forward from the source doc unchanged; it is the strongest thing any of these drafts produced. Directory layout, `RUN_ID` composition, `meta.json` written *first*, Parquet tables, never overwrite, `MANIFEST.jsonl`, `start_run`/`save_table`/`finish_run` in `try/finally`. `paper_numbers.parquet` with `claim_id` indirection.

> Correction: v2 claimed this "matches what `infersim` already does." It does not. Grepping `infersim` for `start_run`, `save_table`, `MANIFEST` returns **zero hits**. infersim writes CSV via a dataclass schema (`eval/schema.py`), and `reproduce_all.sh` *opens by `rm -f`-ing six result files* — the opposite of never-overwrite. `pyarrow` is not installed here. **`scripts/_common.py` is a from-scratch build.** Budget it as such.

---

## W0 (Aug 7–14) — gate, blocking

Nothing downstream is meaningful until these pass. Ordered so the cheapest kill checks come first.

### W0a — CPU-only, no VM, runs before anything is provisioned

1. **Prior-art gate.** Written verdict in `notes/kill_condition.md`, `gapcache` format. Search list: forward citations of RPA and BucketServe; compiled-shape bucketing outside LLMs (TensorRT optimization profiles, ONNX Runtime dynamic shapes, TF Serving batch buckets, `tf.data.bucket_by_sequence_length`); **JAX/XLA shape polymorphism** (`jax.export` polymorphic dims — if prefill can be compiled polymorphically the ladder is optional and the premise weakens badly); varlen/zero-padding serving (ByteTransformer, Effective Transformer); admission and batch composition (Sarathi-Serve, QLM, Andes, Llumnix) — specifically, has anyone published *promote-and-pad* vs *queue-and-wait*?; and **Vidur**, the LLM inference simulator with fidelity validation — either use it or state why not, because a reviewer will ask.
2. **Traces** — ShareGPT, agentic tool-use, long-context, synthetic-controlled. Keep the `trace_stats.parquet` schema. **Do not invent a raggedness index** — use coefficient of variation, or better, expected padding fraction under the default ladder, which is the objective you already care about.
3. **`scripts/_common.py`** + `DECISIONS.md` + `configs/`. Testable without a TPU via mock mode.

### W0b — hardware. You provision, I script.

**Prerequisites you hit before I can do anything** (`infra/create_tpu.sh` states these in its header):
- TPU quota approved **for v5e specifically, in a named zone**. The v1 review recorded "quota approved but unprovisioned" — approval for one generation does not grant another. Confirm the generation and zone.
- Gated `meta-llama` repo access request **started first** — a common multi-hour stall.

Then, in order:

4. **Chunked prefill + prefix caching audit.** Determine defaults, force both to a known state, record in `DECISIONS.md`. **This is the single fact most likely to reshape the paper:** if `tpu-inference` runs chunked prefill by default (vLLM V1 does on other backends), long prefills are chopped into fixed-size chunks and L1 padding is bounded by one chunk, not by the ladder. L1 stops being an interesting cost parameter and the three-level story collapses to two. If it's on and cannot be disabled, reframe to L2 + admission and say so.
5. **Ladder enumeration.** Parse warmup logs and the compilation cache into `ladder_default.json`. Without this there is no independent variable.
6. **Oracle headroom probe.** Compile a handful of exact shapes for lengths drawn from the agentic trace; measure against the default ladder. **This bounds the ceiling on everything the ladder work could ever win.** Under a day, under $20. If the gap is 4%, you learn it in week 0 instead of week 6.
7. **Stock-vLLM baseline characterization.** *New in v3, and it decides whether there is a paper.* Measure what `tpu-inference` + vLLM actually does today when a bucket saturates. If the deployed default already behaves like the hybrid policy, `hybrid`-beats-`queue`-and-`promote` is not a contribution. Nearly free once the VM is up — same server, same logs as step 5.

**Gate:** prior-art verdict written; ladder enumerated; controlled variables pinned and asserting; oracle gap measured; stock behaviour characterized. A negative on 1, 6, or 7 ends the project in week 0 for ~$200.

### Pre-committed response if the gate fires

*This is new in v3 and it is the point of writing it down now rather than in week 3.* `gapcache` died on prior art; determinism died on framing; this is topic three in five days. If this gate fires there is no calendar left for a fourth cold start before the deadline. **The pre-committed answer is: narrow the venue — a workshop submission, or skip this cycle and target the next — not pivot to a fourth topic.** Agreeing to that now is much cheaper than negotiating it under deadline pressure.

---

## W1–3 — primitives on hardware (spot)

- **`e10_latency_steps.py` — staircase sweep, narrowed.** v2's `1:2048:1` × 5 repeats = 10,240 isolated requests is ~20× larger than needed. Sample every 64 tokens globally, then ±8 tokens around each edge from the ladder enumeration. Note that `tpot_ms_p50` from a *prompt-length* sweep needs a specified output length — TPOT is not a function of input length in the same staircase way. Produces the paper's motivating figure and the L1 cost curve `C(B)`.
- **`e11_promotion_cost.py` — the spine's core primitive**, and the measurement v2's source doc never isolated: for a request of length `L` served in bucket `B`, the marginal latency and TPU-seconds of `B` versus the tightest fitting bucket. This is the simulator's central parameter.
- **`e12_queue_wait.py`** — occupancy and admission behaviour at a saturated bucket.
- **`e13_padding_decomposition.py`** — at a small number of rates only, enough to fit and validate, not to sweep.

**Cost model, defined explicitly.** Padding costs *money* only when throughput-bound; at low rate it costs *latency*. The padding→dollars relationship is rate-dependent. This was done properly in `infersim`; do it again. Every gate threshold carries units — "padding waste > 10%" must say **of TPU-seconds**, not of tokens, because only the former is the paper's metric.

---

## W4–6 — simulator and policy sweep (free)

Extend the `infersim` shape: `sim/`, `policies/admission/` as ABC-with-hooks, `eval/` with matched-trace evaluation and paired bootstrap. `infersim/policies/capacity/base.py` is the right template — same five-hook structure, but bucket-scoped rather than machine-scoped.

**Policies:** `stock` (measured in W0 step 7 — the baseline that matters), `queue`, `promote`, `hybrid` (promote below a padding-cost threshold, queue above), `oracle` (knows future arrivals; upper bound).

**Ladder design — `e20_ladder_dp.py`.** Optimize on the measured cost curve, not token counts: padding from `L` to ceiling `B` costs `C(B) − C(L)`, and prefill cost is superlinear. Under a hard cardinality budget with separable additive cost, a 1-D DP over a discretized length axis gives the **global** optimum in `O(K·N²)` — milliseconds for a few thousand candidate edges. Lloyd–Max is an iterative local-optimum scheme, strictly worse here; keep it only as a baseline to beat.

**`e31_ladder_policy_joint.py` — the separability check.** *New in v3.* The DP above is optimal only because cost is assumed separable — and admission control is exactly what breaks that assumption. Once you promote-or-queue, a request's cost depends on queue occupancy and what else is co-batched, not on its own bucket alone. So the DP optimizes the ladder **open-loop** while the headline policy is **closed-loop**. Measure how much the open-loop-optimal ladder loses under the closed-loop policy. If it loses little, that's a clean result. If it loses a lot, that's a *better* result and probably the actual paper. A reviewer will find this either way, so find it first.

Sweep across ladders (default, quantile-k, DP-optimal, agentic-fitted) × traces × rates, 30 seeds each. Core result table, costs nothing.

---

## W7–8 — holdout validation on hardware (on-demand)

Reserve the window in advance. Take ~12 configurations spanning the policy × rate space, run for real, report predicted vs measured with MAPE. Any parameter over 15% gets refit and the sweep re-run.

Also here — **`e41_distribution_shift.py`**: fit on the agentic trace, evaluate on the others, report regret versus oracle and versus default. "How far can the workload drift before the fitted policy loses" is a question reviewers ask, and a number beats hedging.

---

## W9–12 — write-up, with slack

Draft intro and related work during W4–6 — they depend on no results, and the prior-art gate already wrote most of related work.

**Protect the slack.** Both prior projects consumed theirs.

**Drop the negative-result escape hatch as a landing zone.** "The default ladder is near-optimal, and that is a finding" is a workshop paper, not an industrial-track paper. The actual safe landing is the admission policy, which is the spine — that structure makes the hatch unnecessary.

---

## Reuse — three-way split

> v2 presented one flat table that implied more transfer than exists. Corrected:

**Works unchanged:**
| Need | File |
|---|---|
| vLLM HTTP measurement harness | `infersim/calibration/{measure_throughput_grid_vllm,measure_w1_warmup_vllm,measure_holdout_vllm}.py` — verified: POSTs to `/v1/completions` with streaming over plain `urllib`, has a mock mode, genuinely server-agnostic |
| Paired bootstrap CI / permutation test | `infersim/extract_paper_numbers.py` — `bootstrap_ci` (L15), `bootstrap_p` (L31) |
| DES core, policy ABC, matched-trace eval | `infersim/sim/`, `policies/capacity/base.py`, `eval/{runner,schema,trace,aggregate}.py` |

**Works after porting:**
| Need | Reality |
|---|---|
| TPU VM image workarounds | Only the ~40 lines in `vm_setup.sh`'s early steps — malformed `gcsfuse.list`, `cnf-update-db` post-invoke failure, distutils-broken system pip, root-owned `/usr/local`, snap gsutil traceback. Genuinely a day saved. **The rest of that script installs MaxText + JetStream and downloads Gemma-2 to orbax — a stack this project does not use.** |
| MAPE validation | `fit_parameters.py` fits exactly three parameters against a `SIM_DEFAULTS` dict. What transfers is the **accept/reject harness shape** (L13, L213), not the fitter. |

**Build from scratch:**
- `scripts/_common.py` — `start_run`, `finish_run`, `save_table`, `assert_controlled_vars`, `MANIFEST.jsonl`
- Parquet output (needs `pyarrow`, not installed) — infersim is CSV throughout
- Per-bucket cost model `C(B)`, promotion cost, queue-wait parameters

Per the `gapcache` README convention: borrow engineering convention, not research content, and say so in the README. That paragraph pre-empts a self-plagiarism question at review time.

---

## Repo layout

```
~/bucketladder/
├── README.md              status, claim, infersim-relationship paragraph
├── DECISIONS.md           starts W0; carries the billed-VM-hour running total
├── notes/                 prior_art.md, kill_condition.md, design reviews
├── infra/                 create_tpu.sh, teardown_tpu.sh, vm_setup.sh
├── scripts/               _common.py + one script per finding
├── sim/  policies/admission/  eval/
├── configs/               one YAML per variant, all committed, each with the controlled-variables block
└── traces/
```

**Script IDs.** v2 reused `e13` for promotion cost while the source doc had assigned it to the ladder sweep — a collision with the `claim_id` indirection in `paper_numbers.parquet`. Clean namespace: `e00_smoke_test`, `e01_oracle_gap`, `e02_stock_baseline`, `e10_latency_steps`, `e11_promotion_cost`, `e12_queue_wait`, `e13_padding_decomposition`, `e20_ladder_dp`, `e30_policy_sweep`, `e31_ladder_policy_joint`, `e40_holdout`, `e41_distribution_shift`.

**L3 is not a ladder parameter.** L1/L2 are compiled-shape edges; L3 is kernel tiling, conventionally autotuned. Grouping them as "three quantization levels nobody has separated" inflates the claim and invites the response that kernel autotuning is well-studied. Report L3 as a measured third source of shape-induced waste. It is also the highest-instrumentation-risk item in the whole plan.

---

## Verification

- **W0:** `e00_smoke_test.py` exits 0; `ladder_default.json` non-empty and matching the warmup-log bucket count; a config with prefix caching in an unrecorded state **aborts** at `start_run` — test this deliberately.
- **Primitives:** re-running an identical config yields a byte-identical Parquet; `config_hash` stable across process restarts and dict ordering.
- **Simulator:** MAPE < 15% against holdout hardware runs on **every fitted parameter, reported per parameter**.
- **End to end:** `reproduce_all.sh` regenerates every figure from `MANIFEST.jsonl` with no manual steps. (Note this is a *higher* bar than infersim's, which regenerates from configs and `rm -f`s stale CSVs.)
- **Spend:** billed VM-hours in `DECISIONS.md`, checked at each phase boundary against the table above.

---

## Open items

- **Confirm the MLSys 2027 industrial track exists this cycle and its deadline.** The whole schedule keys off ~2026-10-30, which came from search, not a CFP. MLSys has run an industry track inconsistently. **Pick the backup venue now and put it in the schedule**, not in a list of open questions — a 12-week plan that exactly fills an unverified interval needs its fallback inline.
- **Chunked prefill's default in `tpu-inference`** — most likely single fact to reshape the paper.
- **Spot v5e-4 capacity in your zone.** If spot is routinely unavailable the budget still works on-demand; the schedule is what suffers.
- **TPU Research Cloud generations** — free access helps only if they offer v5e or v6e.

---
