# Bucket-Aware Admission Control on Compiled-Shape Accelerators
## Execution Plan v4 — 2026-08-07

Supersedes `plan_v3.md`. v3 was written before the W0 prior-art gate ran. The gate has now
run and closed, and it changed five things materially. This plan is standalone.

---

## What the gate changed

| v3 said | Gate found |
|---|---|
| Three quantization levels (L1 prefill, L2 batch, L3 tiling) | **L1 is dead.** Chunked prefill is default-on in vLLM V1 and ✅-validated in `tpu-inference`. A full 8,192-token chunk is already a power of two; only each request's final partial chunk pads. ~2.4% waste on a 10k prompt vs ~64% unchunked. |
| Ladder must be enumerated from warmup logs | **The ladder is a documented env var.** `VLLM_TPU_BUCKET_PADDING_GAP` gives linear buckets 16→`max_model_len`; default is exponential (nearest power of 2). Manipulating the ladder is config, not scheduler patching. |
| Related work: RPA, BucketServe | **LAPS (MLSys 2026) is primary related work**, plus LENS, Multi-Bin Batching, Vidur. See `prior_art.md`. |
| `e10_latency_steps` produces the motivating staircase | **LENS already published it** — 2.15% mean error, two measurements per bucket, cross-vendor. Cite and reuse; don't re-derive. |
| Gemma-3-4b primary, Llama-3.1-8B secondary | **Inverted.** Llama-3.1-8B-Instruct is on `tpu-inference`'s tested list; Gemma-3-4b is not. |

Unchanged: v5e-4 / TP=4, budget ceiling and the billed-VM-hour rule, you-provision-I-script,
the traceability contract, the pre-committed response if a later gate fires.

---

## The claim

On compiled-shape accelerators every request is padded up to one of N precompiled bucket
shapes, and **N is bounded by compile cost, not by choice**. When a request's bucket is
saturated, the scheduler faces a decision: **dispatch now into a larger already-warm bucket
and pay the padding, or wait for the right bucket and pay the queueing delay.**

The claim is that this decision is workload-dependent, that the right policy beats either
fixed strategy measurably, and that the ladder those policies run against should be chosen
by optimisation rather than by convention.

### Why this is not already done — stated against the actual prior art

- **LAPS** ([2601.11589](https://arxiv.org/abs/2601.11589), MLSys 2026) captures a CUDA Graph
  per `(length, batch)` cell and pads to the nearest. Its AWD scheduler computes `W_GR`, the
  expected time to fill the target depth, waits, then pads to nearest. **It never asks whether
  dispatching now into a larger already-warm bucket beats waiting.** It is also short-prefill
  only (`L ≤ 256`) by explicit design — *"graph capture [is] expensive and rarely amortized.
  Hence, mainstream serving systems avoid CUDA Graphs in prefill"* — and falls back to an
  uncaptured kernel when no shape fits. **On TPU there is no fallback and no opt-out:** every
  request at every length runs a compiled executable.
- **Both closest papers name the cardinality budget and decline to solve it.** BucketServe
  derives the optimal boundary (its Eq. 4, the Lloyd–Max condition) then rejects computing it
  as *"computationally expensive to calculate in practice."* LAPS §4.2: *"the number of graphs
  must be limited to balance memory usage and performance"* — 228–277 MB per graph, 8–12 s to
  capture — then uses a fixed power-of-two grid.
- **LENS** ([2606.18042](https://arxiv.org/abs/2606.18042)) characterises the phenomenon on
  NPUs and predicts it to 2.15%. It is a predictor: no admission, no promotion, no ladder
  design. It is our measurement methodology, not our competitor.
- **Multi-Bin Batching** ([2412.04504](https://arxiv.org/abs/2412.04504)) gives a
  throughput-optimal control policy over predetermined bins — but bins over predicted
  *execution time* (straggler waste in the batch-max regime), not input length padded to a
  compiled shape.

Contribution, in one sentence: **a measured cost model for bucket promotion, a
cardinality-budgeted optimal ladder, and an admission policy that decides promote-vs-wait —
on hardware where the compile budget binds and there is no uncompiled fallback.**

---

## Hardware and budget

**v5e-4 (`v5litepod-4`), single-host, TP=4 fixed.** Cheapest chip on `tpu-inference`'s
*recommended* list (recommended: v7x, v5e, v6e; experimental: v3, v4, v5p — verified). Fixed
topology removes TP as a confound. List prices to re-verify at provision time: v5e
$1.20/chip-hr → **$4.80/hr on-demand, ~$1.40/hr spot**.

**Models:** Llama-3.1-8B-Instruct primary (production-ready on the tested list; ~16 GB bf16,
may need reduced `max_model_len`). Gemma-3-27b-it secondary if a second model is affordable.
Start the gated `meta-llama` access request first — it is a common multi-hour stall.

**Budget — billed VM-hours, not benchmark-hours.** A TPU VM bills while it *exists*. Delete
it at the end of every session; re-create from `infra/create_tpu.sh`; persist to GCS only.

| Phase | Billed VM-h | Pricing | Cost |
|---|---|---|---|
| W0b — bring-up, gate checks, debugging | 30 | on-demand | $144 |
| W1–3 — primitives (re-runnable) | 40 | spot | $56 |
| W7–8 — holdout validation | 25 | on-demand | $120 |
| Reserve | 40 | mixed | ~$128 |
| **Total** | **135** | | **~$450** |

Headroom to the $1,000 ceiling is ~$550 — **one forgotten VM-week**. Running total lives in
`DECISIONS.md`, checked at every phase boundary. Preemption → `preempted: true` in
`meta.json`, excluded from figures, same rule as `dirty: true`.

---

## Controlled-variables contract

Pinned, recorded in `meta.json`, and **asserted at runtime by `assert_controlled_vars`, which
aborts rather than warns.**

| Variable | Setting | Why |
|---|---|---|
| **Prefix caching (APC)** | **off, asserted** | Corrupts measured prompt-token counts → corrupts every padding number. |
| **Chunked prefill** | **recorded, both states swept** | Now a first-class independent variable, not a nuisance — it determines whether prefill padding exists at all. |
| `max_num_batched_tokens` | explicit | Sets the chunk size and therefore the residual prefill padding. |
| **Tensor parallelism** | fixed at 4, recorded | Held constant; stated as a limitation. |
| Speculative decoding | off | Changes accepted-token paths. |
| KV cache dtype | explicit, recorded | |
| `VLLM_TPU_BUCKET_PADDING_GAP` | explicit, recorded | **The independent variable.** |
| Server restart | recorded as its own axis | Is the ladder stable across restarts? |
| `XLA_FLAGS` / libtpu flags | verbatim | Load-bearing. |

---

## Traceability contract

Carried forward unchanged from v3 — it is the strongest thing these drafts produced.
`RUN_ID` composition, `meta.json` written **first**, Parquet tables, never overwrite,
`MANIFEST.jsonl`, `start_run`/`save_table`/`finish_run` in `try/finally`,
`paper_numbers.parquet` with `claim_id` indirection.

Built from scratch — `infersim` has none of these (zero hits for `start_run`, `save_table`,
`MANIFEST`; it writes CSV and `reproduce_all.sh` opens by `rm -f`-ing results).

---

## W0b — hardware bring-up. You provision, I script.

**Prerequisites you satisfy first** (stated in `infra/create_tpu.sh`'s header):
- TPU quota approved **for v5e specifically, in a named zone** — approval for one generation
  does not grant another.
- Gated `meta-llama` repo access requested.

Then, in order:

1. **`infra/create_tpu.sh`** → VM up, `infra/vm_setup.sh` → `tpu-inference` + vLLM at a
   pinned version.
2. **`e00_smoke_test.py`** — ladder enumeration + controlled-variable audit. Confirms
   `VLLM_TPU_BUCKET_PADDING_GAP` behaves as documented, records the default ladder, and
   proves `assert_controlled_vars` aborts on an unrecorded APC state.
3. **`e01_oracle_gap.py`** — exact-shape compiles vs the default ladder on trace lengths.
   Bounds the ceiling on everything the ladder work can win. Under a day, under $20.
4. **`e02_stock_baseline.py`** — what `tpu-inference` actually does when a bucket saturates.
   If the deployed default already behaves like the hybrid policy, there is no contribution.
5. **`infra/teardown_tpu.sh`** — every session ends here.

**Gate:** ladder enumerated and controllable, controlled variables asserting, oracle gap
measured, stock behaviour characterised.

---

## W1–3 — primitives on hardware (spot)

- **`e11_promotion_cost.py` — the spine's core primitive.** For a request of length `L`
  served in bucket `B`: marginal latency and TPU-seconds of `B` versus the tightest fitting
  bucket. This is the simulator's central parameter and the measurement nobody has published.
- **`e12_queue_wait.py`** — occupancy and admission behaviour at a saturated bucket.
- **`e13_padding_decomposition.py`** — at a small number of rates, enough to fit and validate.
- **`e10_latency_steps.py` — reduced to a validation check.** LENS published the staircase;
  we measure enough to confirm its two-measurements-per-bucket composition holds on TPU, then
  cite it for `C(B)`. **This is the main hardware saving in v4.**

**Cost model, explicit.** Padding costs money only when throughput-bound; at low rate it
costs latency. The relationship is rate-dependent. Every gate threshold carries units —
"padding waste > 10%" means **of TPU-seconds**, never of tokens.

---

## W4–6 — simulator and policy sweep (free)

Extend the `infersim` shape: `sim/`, `policies/admission/` as ABC-with-hooks, `eval/` with
matched-trace evaluation and paired bootstrap. `infersim/policies/capacity/base.py` is the
template — same five-hook structure, bucket-scoped rather than machine-scoped.

**Policies:** `stock` (measured in W0b), `wait` (LAPS-like: queue until the bucket fills,
then pad to nearest), `promote` (dispatch now into the smallest warm bucket that fits),
`hybrid` (promote when promotion cost < expected queueing cost), `oracle` (knows future
arrivals; upper bound).

**`e20_ladder_dp.py`** — optimise on the **measured** cost curve, not token counts: padding
from `L` to `B` costs `C(B) − C(L)` and prefill cost is superlinear. Under a hard cardinality
budget with separable additive cost, a 1-D DP over a discretised length axis is **globally
optimal** in `O(K·N²)` — milliseconds for a few thousand candidate edges. Frame as answering
two stated concerns (BucketServe's "computationally expensive", LAPS's "must be limited"),
not as novelty. Lloyd–Max and fixed powers-of-two are baselines.

**`e31_ladder_policy_joint.py` — the separability check.** The DP is optimal only because
cost is assumed separable, and admission control is what breaks that: under promote-or-wait a
request's cost depends on queue occupancy and co-batching. So the DP optimises **open-loop**
while the headline policy is **closed-loop**. Measure how much the open-loop-optimal ladder
loses. Little → clean result. A lot → better result, and probably the actual paper.

Sweep ladders × traces × rates, 30 seeds, matched-trace paired bootstrap.

---

## W7–8 — holdout validation on hardware (on-demand)

~12 configurations spanning policy × rate. Predicted vs measured, **MAPE < 15% on every
parameter, reported per parameter**. Anything over refits and the sweep re-runs. Cite Vidur
(MLSys'24, <9% error) as the precedent for this methodology rather than presenting it as new.

Also **`e41_distribution_shift.py`** — fit on one trace, evaluate on others, report regret vs
oracle and vs default.

---

## W9–12 — write-up

Draft intro and related work during W4–6; the gate already wrote most of related work.
Protect the slack — both prior projects consumed theirs.

**LAPS is distinguished in paragraph one, not buried.**

---

## Reuse — three-way split

**Works unchanged:** `infersim/calibration/*_vllm.py` (verified: `/v1/completions` streaming
over plain `urllib`, has mock mode, server-agnostic); `extract_paper_numbers.py`
`bootstrap_ci` L15 / `bootstrap_p` L31; `infersim/{sim,policies/capacity,eval}/` shape.

**Works after porting:** the ~40 lines of `tpu-ubuntu2204-base` workarounds in
`vm_setup.sh`'s early steps (malformed `gcsfuse.list`, `cnf-update-db`, distutils-broken pip,
root-owned `/usr/local`, snap gsutil) — **the rest of that script installs MaxText/JetStream,
which this project does not use**; the MAPE accept/reject harness *shape* from
`fit_parameters.py` (which fits three parameters against a `SIM_DEFAULTS` dict — not our
parameter set).

**From scratch:** `scripts/_common.py`; Parquet + `MANIFEST.jsonl`; the per-bucket cost model,
promotion cost, and queue-wait parameters.

---

## Script namespace

`e00_smoke_test`, `e01_oracle_gap`, `e02_stock_baseline`, `e10_latency_steps`,
`e11_promotion_cost`, `e12_queue_wait`, `e13_padding_decomposition`, `e20_ladder_dp`,
`e30_policy_sweep`, `e31_ladder_policy_joint`, `e40_holdout`, `e41_distribution_shift`.

**L3 is not a ladder level.** L1/L2 are compiled-shape edges; L3 is kernel tiling,
conventionally autotuned — and JAXBench now benchmarks exactly that. Report L3 as a measured
third source of shape-induced waste, not a third ladder level.

---

## Verification

- **W0b:** `e00_smoke_test.py` exits 0; ladder JSON non-empty and matching the warmup-log
  bucket count; a config with APC in an unrecorded state **aborts** at `start_run` — tested
  deliberately.
- **Primitives:** identical config → byte-identical Parquet; `config_hash` stable across
  process restarts and dict ordering.
- **Simulator:** MAPE < 15% per parameter against holdout.
- **End to end:** `reproduce_all.sh` regenerates every figure from `MANIFEST.jsonl`.
- **Spend:** billed VM-hours in `DECISIONS.md` at every phase boundary.

---

## Open items

- **Read LENS in full before W1** — most likely to save hardware money, possibly supplies
  `C(B)` outright.
- **Read Multi-Bin Batching before the DP** — its throughput-optimal bin policy is the nearest
  theoretical neighbour.
- **Can Vidur (or Frontier, [2605.21312](https://arxiv.org/pdf/2605.21312)) be extended to
  compiled-shape ladders?** If yes, build on it.
- **Confirm the MLSys 2027 industrial track exists and its deadline**, and **put the backup
  venue in the schedule**, not in this list.
- TPU quota generation + zone; spot v5e-4 capacity; TPU Research Cloud generations.
- Unpinned: the system that "sorts the pending queue by prompt length"; the 60–80%
  padding-overhead figure; *Beyond Prediction: Tail-Aware Scheduling*
  ([2606.18431](https://arxiv.org/abs/2606.18431)).
