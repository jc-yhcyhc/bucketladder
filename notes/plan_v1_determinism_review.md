# Deterministic Inference on Compiled-Shape Accelerators — Revised Execution Plan

## Context

You asked for an assessment of the draft plan. The traceability contract (§0) is the strongest part and survives unchanged — it is better than most published artifacts and matches the habits already visible in `infersim` (`extract_paper_numbers.py`, `reproduce_all.sh`). The experimental *sequencing* instinct is also right: E0.2 is the load-bearing experiment and belongs first.

Three things in the draft would have cost weeks, and they are what this revision fixes:

1. **Nothing can run today.** `/home/yhcyhc1991/DeterministicInfer` is empty; the host is an `n2-standard-16` with no TPU, no JAX, no vLLM. "E0.2 is one day" presumes a working `tpu-inference` stack that does not exist. There is no Phase −1 in the draft.
2. **The schedule does not fit the deadline.** MLSys 2027 industrial track appears to close **2026-10-30** — 12 weeks from today (2026-08-07). The draft is a 12-week plan with write-up in weeks 11–12 and zero slack, and its week 1 is already consumed by bring-up.
3. **Confounds that would silently invalidate Phase 0.** The draft pins temperature and seed but never mentions prefix caching, chunked prefill, or tensor-parallel all-reduce order. Any one of these makes E0.2's answer meaningless.

The intended outcome is unchanged: a shape-pinning determinism paper with a measured cost curve, submitted 2026-10-30.

**Confirmed decisions:** TPU quota approved but unprovisioned. Route B prioritizes RMSNorm + matmul; attention kept as a stretch goal, off the critical path. Phase 4 stays hardware-only with a narrowed grid.

---

## Competitive position (verified 2026-08-07)

This field moved fast while the draft was being written. Grounding, with what each implies:

| Work | Implication |
|---|---|
| [Ragged Paged Attention](https://arxiv.org/abs/2604.15464) (Jiang, Chen, Hechtman, Zhang, Mu — Apr 2026) | The TPU attention kernel in **both vLLM and SGLang**. This is what Route B's E3.3 would be reimplementing. Also mandatory related work — the draft doesn't cite it. |
| [vLLM batch invariance docs](https://docs.vllm.ai/en/latest/features/batch_invariance/) | **NVIDIA SM80+ only.** No TPU support. This is the strongest novelty signal available: the compiled-shape backend is genuinely unclaimed. |
| [MarginGate](https://arxiv.org/abs/2605.30218) (Chu, Zhou, Zhang — May 2026) | Real; the 0.3–1.3% flip-rate range and the margin protocol are correctly cited. Not your paper, so E0.4 must reimplement its protocol rather than reuse code. |
| [LLM-42](https://arxiv.org/abs/2601.17768), [HEAL](https://arxiv.org/pdf/2606.21023) | Verification-based determinism. A third route you compare against by citation, not implementation. |
| [TBIK](https://arxiv.org/abs/2511.17826) | Determinism across **tensor-parallel sizes**. Proves TP is a recognized independent axis — see the controlled-variables contract below. |

## The tautology risk, and the fix

This is the single most important framing issue in the plan. It needs to be designed for in W1, not discovered in W11.

### What the risk is

On GPU, nondeterminism arises because kernels select a reduction strategy *at runtime* from batch size: flash-decoding picks its KV split count to fill available SMs, cuBLAS/CUTLASS picks tile and split-K configs by shape. Different split → different reduction tree → different bits. That is why the GPU fix is *rewriting reductions*.

XLA does not work this way. It compiles static-shaped HLO in which tiling and accumulation order are frozen **at compile time**; vLLM TPU absorbs dynamism by padding to one of N precompiled buckets, each a distinct executable. So "same bucket + same input bits ⇒ same output bits" comes dangerously close to restating what a compiled executable *is*. A headline of "we pinned the shape and got determinism" invites the response: *you defined your way to the result.*

### Why it is not actually a tautology

Where the batch axis enters the reductions in Llama's forward pass:

| Op | Reduces over | Batch axis participates? |
|---|---|---|
| RMSNorm | hidden dim, per token | No |
| QKV / O proj, MLP gate/up/down | input feature dim K | No — batch is a parallel (M) axis |
| **Attention** | **KV positions** | **Yes — via ragged/paged loop structure** |
| TP all-reduce | across chips | No (but TP *size* changes the order) |

This yields a falsifiable prediction: bucket-internal invariance holds everywhere **except** attention, where the KV-page loop trip count can depend on co-resident sequence lengths. That is a mechanism claim E0.2 can refute — not a definition.

### Fix (a) — E0.2 becomes a mechanism decomposition, not a binary gate

Target the prediction directly: hold the bucket fixed and vary co-resident **sequence lengths and page counts** specifically, since that is the only site where divergence is predicted. Report per-op, against the table above, rather than reporting one pass/fail bit. Both outcomes are results — divergence gives a named cause; no divergence is a citable finding that Ragged Paged Attention is already shape-invariant.

### Fix (b) — E0.6, sufficiency boundary (NEW, ~1 day, W2)

The stronger fix. Show the **converse fails**: shape pinning is *necessary but not sufficient*. Construct cases where shape is fixed and determinism still breaks.

| Probe | Expectation | Why it matters |
|---|---|---|
| APC on, fixed shape | breaks | Already a planned E0.2 variant; promote it here |
| **Cross-restart recompilation**, nominally identical shape | unknown — the interesting one | Does XLA emit a bit-identical executable across process restarts, libtpu versions, and flag changes? If not, "pin the shape" is insufficient, and that is a genuinely surprising result |
| TP size change at fixed per-chip shape | breaks | Connects to TBIK; bounds the claim |

**Save `sufficiency.parquet`:** one row per (probe, config) with `shape_identical` (bool), `output_sha256`, `matches_reference`, `divergence_cause`.

This converts the central claim from a definition into an empirical statement with measured boundaries:

> Shape pinning is necessary but not sufficient for determinism on compiled-shape accelerators. Here are the additional invariants required, and here is what the complete set costs.

### Consequent retitling

> Determinism on compiled-shape accelerators is a scheduling constraint, not a numerics problem — and here is what that constraint costs.

This promotes `edge_cases.parquet` (E2.1) from appendix to **core result table**. Bucket-crossing growth, preemption/resume, and pin-unavailable queuing are the actual hard problems and the actual contribution.

---

## Controlled-variables contract

The single largest technical gap in the draft. Every one of these is a determinism source orthogonal to shape; each must be pinned, recorded in `meta.json`, and asserted at runtime.

| Variable | Required setting | Why it matters |
|---|---|---|
| **Prefix caching (APC)** | **off**, asserted | If on, the probe's KV depends on what ran *before* it. Alone, this invalidates all of E0.2. |
| **Chunked prefill** | off for Phase 0; a recorded variant later | Chunk boundary placement changes reduction structure. |
| **Tensor parallelism** | fixed; recorded | v6e-8 at TP=8 has all-reduce ordering. TBIK exists for exactly this. Hold fixed, state the limitation. |
| **Speculative decoding** | off | Changes accepted-token paths entirely. |
| **KV cache dtype** | explicit, recorded | |
| **Server restart** | recorded as its own axis | Is the bucket ladder stable across restarts and recompiles? Cross-restart determinism is a separate, weaker guarantee than in-process — the paper must say which it claims. |
| **`XLA_FLAGS` / libtpu flags** | verbatim in `meta.json` | Already in the draft's schema; now load-bearing. |

Add `assert_controlled_vars(config)` to `scripts/_common.py`, called by `start_run` before any work. A run that cannot prove APC is off must abort, not warn.

---

## Phase −1 — Bring-up (W0, Aug 7–14) — NEW, blocking

Quota is approved but nothing is provisioned. Two capabilities are the entire experimental apparatus and **both can fail**; prove them before writing any experiment code.

1. Provision the TPU VM; install `tpu-inference` + vLLM; resolve gated `meta-llama` repo access (a common multi-hour stall — start it first).
2. **Smoke test A — bucket ladder extraction.** Serve `Llama-3.2-1B`, parse the warmup log, emit `bucket_ladder.json`. If the ladder can't be enumerated, E0.3 has no independent variable.
3. **Smoke test B — bit-exact logit capture.** This is the higher risk. Full logit vectors require `logprobs`, which on TPU **changes the compiled graph** (extra top-k, possibly different shapes). So the draft's E1.1 check "compare hooks on vs off" may be comparing two different programs and is not always available.

   Decide the fallback here, in W0, because it determines the entire results schema: capture **top-k logprobs only (k=20)**, define `logit_sha256` over that top-k slice with a pinned dtype and byte order, and hold `logprobs=k` constant across *every* arm including references so the graph never varies. Record the decision in `DECISIONS.md`.
4. Run **E0.5** (prompt construction) — pure CPU, no TPU needed, can proceed in parallel with provisioning.

**Gate:** both smoke tests pass and `bucket_ladder.json` is on disk. Nothing downstream is meaningful otherwise.

---

## Phase 0 — Characterization (W1–2, Aug 14–28)

Keep E0.1–E0.4 as drafted, with these corrections.

- **E0.2 runs first** (correct in the draft), but restructured per Fix (a) above: it is a **per-op mechanism decomposition**, not a pass/fail gate. Add a composition variant that holds the bucket fixed while varying co-resident **sequence lengths and KV page counts** — the one site where the mechanism table predicts divergence. Report results against that table.
- **E0.6 — sufficiency boundary** (new, ~1 day, W2). The falsification experiment from Fix (b): APC-on at fixed shape, cross-restart recompilation, TP-size change. Writes `sufficiency.parquet`. This is what keeps the headline from reading as a definition, so it is not optional and not cuttable.
- **E0.1 — state the power.** 50 prompts × 256 positions × 100 repeats ≈ 1.28M comparisons; zero observed divergence gives a 95% upper bound of ≈2.3×10⁻⁶. Put this in the paper; reviewers will ask.
- **E0.2 gate logic is inverted in the draft.** It reads "no divergence → Route A is headline, Route B becomes comparison." But if there is no divergence at fixed shape, Route B is *unnecessary on TPU* — you'd be building it solely to be beaten. Correct rule: **E0.2's result sets Route B's budget.** Clean result → Route B is capped at the two easy ops, permanently.
- **E0.4** — keep the MarginGate protocol (synchronous positions only, batch-1 reference) and the margin-binned flip-probability figure. That figure is the paper's mechanism and is worth protecting from cuts.

### Schema corrections

- `first_divergent_position` → nullable **Int32**, not float-with-NaN. Float will destroy exactness on round-trip.
- Drop `margin_bits` as a stored column — it is a *difference*, and `top5_logit_bits` already determines it. Store `margin_float` as **float64** for log-binning; derive the rest.
- `logit_sha256` needs a written spec (dtype, byte order, pre-softmax, top-k slice). Put the hash function in `_common.py` and make E1.1 test it directly — a hash whose definition drifts silently invalidates every comparison.

---

## Phase 1 — Instrumentation (folded into W2)

E1.1 and E1.2 as drafted, with E1.1's "hooks on vs off" check replaced by the W0 decision: assert `logprobs=k` is constant across all arms, and verify Parquet `uint32` round-trip exactness. E1.2 (instrumentation overhead) is unchanged and remains required — Phase 4's overhead numbers must be attributable to determinism, not to logging.

---

## Phase 2 — Route A: Shape Pinning (W3–4, Aug 28–Sep 11)

E2.1 and E2.2 as drafted. One change in emphasis: `edge_cases.parquet` is now a **core deliverable**, not a supplement. For each of `growth_crosses_bucket`, `preempt_resume`, `pin_unavailable`, record frequency under realistic load *and* the policy chosen, with its latency cost. This table carries the paper.

E2.2's "report overhead as a curve over request rate" is a genuinely good call — keep it.

---

## Phase 3 — Route B: Invariant Kernels (W5–7, Sep 11–Oct 2)

Scoped per your decision: **RMSNorm and matmul are the deliverable; attention is a stretch goal off the critical path.**

- **E3.1 RMSNorm** (W5), **E3.2 matmul** (W6) — full artifact set as drafted: `invariance.parquet`, `microbench.parquet`, `kernel_meta.json`.
- **E3.3 attention** — attempt only in W7 and only if E3.1/E3.2 are complete and Phase 2 is closed. Hard stop at end of W7 regardless of state; record the outcome in `DECISIONS.md`. Do not let this slip into the eval window.
- **Extrapolate attention's cost** rather than measuring it: use XLA profiler op time-share (already planned as `ablation.parquet`) to bound what invariant attention *would* cost. State it as an estimate.

**Strawman risk — address it explicitly in the paper.** Your Route B kernels will be slower than the stock ones partly because they are invariant and partly because you wrote them in three weeks and Google wrote the baseline. A reviewer will say so. Defuse it by reporting Route B's efficiency **against the stock kernel's roofline**, so readers can separate "rewriting reductions is inherently costly" from "this particular rewrite is unoptimized." Without that, the headline comparison is not defensible.

---

## Phase 4 — Evaluation (W8–9, Oct 2–Oct 16)

Hardware-only, narrowed grid per your decision. The draft's grid was 4 modes × 2 models × 2 workloads × 6 rates × 3 repeats × 300 s ≈ **24 h of steady-state measurement alone**, before per-config bucket-ladder precompilation — realistically 3–5 days of continuous v6e-8.

**Narrowed grid:**

| Axis | Value | Rationale |
|---|---|---|
| Modes | **3**: `baseline`, `route_a`, `hybrid` | Pure `route_b` won't exist with attention deprioritized. `hybrid` = invariant RMSNorm+matmul, pinned attention. |
| Models | **both** (1B + 8B) | Keep. "Is this a small-model artifact?" is a guaranteed reviewer question. |
| Workloads | ShareGPT at all 6 rates; synthetic length-controlled at **2 rates only** | Synthetic becomes a targeted padding-sensitivity probe, not a full sweep. |
| Repeats | 3, index recorded, median with min/max | Keep — single-run perf numbers will not survive review. |

≈ 3 × 2 × 8 × 3 × 300 s ≈ **12 h** steady-state. Book the TPU window in advance.

**Pre-register the headline comparison** in `DECISIONS.md` *before* Phase 4 runs: at each request rate, the paired difference in p50 TTFT and throughput, with a stated acceptance threshold. "Materially lower overhead" needs an effect size and a test, not eyeballed curves.

---

## Phase 5 — Write-up (W10–12, Oct 16–30; start related work in W5)

E5.1 and E5.2 as drafted — the provenance sidecars and `paper_numbers.parquet` with `claim_id` indirection are the right mechanism and mirror `infersim`'s `extract_paper_numbers.py`.

Draft the intro and related-work sections during Phase 3 (W5–7). They depend on no results, and the related-work section is already substantially written in the table above.

---

## Revised timeline

| Weeks | Dates | Work |
|---|---|---|
| W0 | Aug 7–14 | **Phase −1** bring-up, smoke tests, E0.5 prompts |
| W1–2 | Aug 14–28 | Phase 0 characterization + Phase 1 harness |
| W3–4 | Aug 28–Sep 11 | Phase 2 Route A |
| W5–7 | Sep 11–Oct 2 | Phase 3 RMSNorm + matmul (attention stretch, W7 only); begin related work |
| W8–9 | Oct 2–16 | Phase 4 eval, narrowed grid |
| W10–12 | Oct 16–30 | Write-up, figures, submission |

Buffer is ~1 week, concentrated in W7. Protect it.

---

## Revised decision gates

| Gate | Check | Source | If it fails |
|---|---|---|---|
| End W0 | TPU serving + logit capture + bucket ladder all work | Phase −1 | Blocking. Everything is downstream. |
| End W0 | Logit capture strategy fixed and recorded | `DECISIONS.md` | Schema cannot be written without it |
| End W2 | batch-1 bitwise stable | E0.1 | Stop; reassess |
| End W2 | divergence exists across buckets | E0.2, E0.3 | If TPU is already deterministic, pivot to *why* — E0.6 carries the paper |
| End W2 | **at least one sufficiency probe breaks at fixed shape** | E0.6 | If all three hold, shape pinning genuinely *is* sufficient — say so as the finding, and lean the paper entirely onto the cost result |
| End W2 | harness self-test passes | E1.1 | Fix before Phase 2 |
| End W4 | Route A overhead < 15% | E2.2 | Publishable; comparison weakens |
| End W7 | RMSNorm + matmul invariant | E3.1/E3.2 | Ship shape-pinning-only paper |
| End W9 | Route A materially beats hybrid, per pre-registered test | E4.1 | Reframe as "determinism costs the same everywhere" — still a result |

Record every outcome in `DECISIONS.md` with date, data reference, and rationale, as drafted.

---

## Unchanged from the draft

§0 in full — directory layout, `RUN_ID` composition, `meta.json`-written-first, Parquet-not-CSV, `uint32` bit patterns, never-overwrite, git-clean enforcement, `MANIFEST.jsonl`, and the `start_run`/`save_table`/`finish_run` skeleton in `try/finally`. Do not weaken any of it.

---

## Critical files to create

- `scripts/_common.py` — `start_run`, `finish_run`, `save_table`, plus **new**: `assert_controlled_vars`, `logit_sha256`
- `scripts/e05_build_prompts.py` — runs on CPU, start immediately
- `scripts/e00_smoke_test.py` — **new**, Phase −1 gate: ladder extraction + logit capture
- `scripts/e02_bucket_internal.py` — first real experiment; per-op decomposition, includes the varying-KV-page-count variant
- `scripts/e06_sufficiency.py` — **new**; the falsification experiment. Cross-restart probe needs the runner to survive a server bounce mid-experiment
- `configs/` — one YAML per variant, all committed, all carrying the controlled-variables block
- `DECISIONS.md` — starts in W0, not W8

## Verification

- **Phase −1:** `e00_smoke_test.py` exits 0, `bucket_ladder.json` non-empty, two identical single-prompt calls produce identical `logit_sha256`.
- **Harness (E1.1):** re-running an identical config yields a byte-identical `results.parquet`; `config_hash` stable across process restarts and dict orderings; Parquet `uint32` round-trip exact; the "same config hash + same git SHA ⇒ identical results" property from §0.2 asserted as a test.
- **Controlled vars:** a run configured with APC on must abort at `start_run`, not warn. Test this deliberately.
- **End-to-end:** `reproduce_all.sh` regenerates every figure from `MANIFEST.jsonl` with no manual steps — the same bar `infersim` already meets.

## Open items

- **Confirm the MLSys 2027 industrial-track deadline directly.** mlsys.org still serves 2026 content; 2026-10-30 comes from search results, not the official CFP. The entire schedule keys off it. Verify in W0 and pick a backup venue if it moves earlier.
