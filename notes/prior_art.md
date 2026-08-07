# Prior art — W0 gate

**Date:** 2026-08-07 · **Depth:** abstracts, official docs, and HTML renders. **Not full
paper reads.** Normal diligence for a first pass; explicitly not enough to make a final
call on a multi-month investment. `gapcache`'s pass carried the same caveat and it
mattered — see "Confidence" at the end.

Verdict is in `kill_condition.md`. This file is the evidence.

---

## 1. JAX/XLA shape polymorphism — does the ladder even exist?

**The premise-killer, checked first. It does not fire. The ladder is real.**

`jax.export` supports shape polymorphism via symbolic dimensions, with constraints
(`a >= 16`) and dimension polynomials. It is real and maintained. But it is an *export*
mechanism — dynamism traces back to input shapes and StableHLO carries symbolic dims —
and it is not what the vLLM TPU serving path uses.

What the serving path actually does, from vLLM's own TPU docs:

> "TPUs are specialized accelerators (ASICs) that require a specific compiled graph for
> each tensor shape (e.g., batch size and sequence length). vLLM pre-compiles the model
> for various common input shapes and saves these compiled graphs to a cache on disk"
> (`~/.cache/vllm/xla_cache`).

And the ladder is a **documented, configurable object** — better than the plan assumed:

- **Default: exponential padding** — pad to the nearest power of 2.
- **Alternative: bucket padding** via `VLLM_TPU_BUCKET_PADDING_GAP`. Buckets "start from
  16, end at `max_model_len`, and increment by `VLLM_TPU_BUCKET_PADDING_GAP`."
  Documented example: `max_model_len=512`, gap 64 → `[16, 32, 64, 128, 192, 256, 320,
  384, 448, 512]`.
- Docs state the tradeoff the paper is about, explicitly: too small a gap → "increased
  warmup (precompile) time and higher memory to store the compiled graph, and too many
  compiled graphs may lead to HBM OOM"; too large → no advantage over exponential.

RPA's own paper reinforces it: TPUs are "optimized for more regular and statically shaped
workloads," and RPA's third technique is "a distribution-aware compilation strategy that
**generates specialized kernels** for decode, prefill, and mixed workloads."

**Implication for the plan, in our favour:** ladder enumeration (`e00`) is largely a
config read plus a cache listing, not the log-parsing exercise the plan budgeted for. And
`VLLM_TPU_BUCKET_PADDING_GAP` is a ready-made independent variable — the ladder is
*already* a supported knob, which is a much easier experimental story than patching a
scheduler.

**Also in our favour:** the *default* ladder is powers of two. That is a very coarse
ladder — worst-case ~2× padding on the outer shape — so oracle headroom (`e01`) is
plausibly large. Do not assume it; measure it.

## 2. Chunked prefill — the L1 kill check

**This fires. L1 is largely dead as a cost parameter.**

- vLLM V1: **chunked prefill is always enabled by default** (V0 enabled it conditionally).
- `tpu-inference` is the V1 code path, and its feature matrix lists **Chunked Prefill ✅
  across all frameworks (Flax, Torchax, Default)**, alongside **Prefix Caching ✅**.
- Default chunk size for online serving is `--max-num-batched-tokens = 8192` in recent
  vLLM (up from 512 historically).

Arithmetic on a 10,000-token prompt under power-of-2 padding:

| | Padded shape | Waste |
|---|---|---|
| No chunking | 10,000 → 16,384 | 6,384 tokens (**64%**) |
| Chunked at 8,192 | 8,192 (exact) + 1,808 → 2,048 | 240 tokens (**2.4%**) |

A full chunk of 8,192 is *already a power of two*, so it needs no padding at all. Only the
final partial chunk of each request is padded, and it is padded against a ladder bounded
by the chunk size, not by `max_model_len`. The prefill ladder stops being an interesting
cost parameter, and "total prompt tokens → prefill bucket" is not how the system behaves.

This is exactly the outcome the plan pre-specified a response for: **reframe to L2 (batch
/ decode shape) + admission control**, which is the spine anyway. The paper narrows; it
does not die.

**Two caveats before treating this as settled.** The chunked-prefill default is documented
for vLLM V1 generally — the TPU-specific default and the TPU default for
`max_num_batched_tokens` are *not* separately documented and must be confirmed on hardware
in W0b. And prefix caching being ✅-supported makes the controlled-variables contract
load-bearing rather than theoretical: if APC is on, measured prompt-token counts are not
what the model actually prefilled.

## 3. Ragged Paged Attention — what padding survives it

RPA ([2604.15464](https://arxiv.org/abs/2604.15464), Apr 2026) is the TPU attention kernel
in both vLLM and SGLang. Three techniques: fine-grained tiling for dynamic slicing over
ragged memory; a software pipeline fusing KV-cache updates into attention; and
distribution-aware compilation generating specialized decode/prefill/mixed kernels.

**The stack has already moved past the version the plan cites.** `tpu-inference` now ships
**RPA v3**: v2 "could only support model specs with a head dim of 128" and had sequential
KV cache updates; v3 supports "arbitrary model specs, quantization dtypes, and arbitrary
tensor-parallelism," fuses the KV scatter into the kernel, and is ~10% faster than v2 on
v6e. Any claim about what RPA leaves behind must be made against v3.

**The answer to "what survives RPA," as far as this pass can tell:** RPA removes padding
waste *within* attention over ragged KV. It does not remove the **outer compiled shape** —
the batch/sequence bucket the executable is compiled for — because it *is itself*
compiled into specialized kernel variants. That is the padding this paper is about, and it
is a defensible position. It needs a measurement (`e01`, `e13`), not just an argument.

Evidence that vLLM has been managing RPA-specific shape constraints by hand:
[PR #14597](https://github.com/vllm-project/vllm/pull/14597) padded `block_table.shape[1]`
to satisfy `block_table.shape[1] % NUM_KV_PAGES_PER_BLOCK == 0`, later removed in
[PR #14846](https://github.com/vllm-project/vllm/pull/14846) once the kernel was fixed.
Useful colour: shape-alignment padding in this stack is real, tracked, and iterated on.

## 4. BucketServe — full read, 9 pages. The spine survives; the ladder claim changes.

[BucketServe (2507.17120)](https://arxiv.org/abs/2507.17120) — Zheng, Xu, Song, Ye
(SUSTech + Shenzhen Institutes of Advanced Technology, CAS), 23 Jul 2025. **Read in full.**

Setup: built on vLLM, disaggregated prefill/decode, **4× NVIDIA A100 40 GB**, LLaMA-2-13B
and OPT, datasets Stanford Alpaca (mean 83.7 / median 69 tokens) and LongBench (median
41,417, truncated to 1024). Baselines **UELLM** and **DistServe**. Three-tier architecture:
Request Bucketing Manager → Dynamic Batching Controller → P/D Scheduler, with a Global
Monitor feeding memory/queue telemetry.

### The spine is clear — no promote-vs-queue

Algorithm 1, lines 2–6: each request is assigned to the bucket whose range *contains* its
length (`if b_low ≤ S < b_up then add r to b.requests; break`). **A request is never placed
in a larger bucket than it needs.** The adaptive machinery moves *boundaries* (split when
>50% of a bucket's requests fall below its midpoint and the bucket exceeds `N_max`
requests; merge everything back to one bucket when total load drops below `N_max`), not
requests. Ordering within a bucket is SJF or LJF for offline, earliest-arrival for online;
FCFS governs the P/D handoff.

So the decision this paper is about — *when your bucket is saturated, promote and pay the
padding, or queue and pay the wait* — **is not made by BucketServe.** Its answer is always
"stay in your bucket, and we will move the bucket." Confirmed across the full text.

### But the ladder-design claim now has a named opponent, and it is close

BucketServe formalises padding waste and derives the optimal boundary:

- Eq (2): `Waste_Ratio = (S_max − S_avg) / S_max`
- Eq (3): `E[Waste] = Σ_{b=1..K} ∫_{L_b}^{U_b} (1 − S/U_b) f(S) dS`
- Eq (4): `U_b* = ∫_{L_b}^{U_b} S f(S) dS / ∫_{L_b}^{U_b} f(S) dS`

— "the upper bound of each bucket should be set to the **conditional expectation** of
sequence lengths within that bucket." **That is the Lloyd–Max centroid condition**, already
published for this exact problem.

Then they decline to solve it:

> "Although the optimal bucket boundary that minimizes E[Waste] is theoretically defined as
> the conditional expectation of sequence lengths within a bucket, it is **computationally
> expensive to calculate in practice**. Moreover, since request length distributions can
> change over time, maintaining such boundaries dynamically introduces significant overhead
> and algorithmic complexity. To address this challenge, we adopt a simple but efficient
> approach based on **interval bisection**, which approximates the optimal boundary."

Midpoint splitting, threshold θ = 0.5, overall complexity `O(n·k + 4k)`.

**This is simultaneously the biggest threat and the clearest opening found so far.**

- *Threat:* "we optimise the bucket ladder" is no longer novel framing. The stationarity
  condition is published. v3's plan to "keep Lloyd–Max as a baseline to beat" must now be
  written as *"beat BucketServe's stated fallback,"* with the citation, or a reviewer will
  supply it.
- *Opening:* they rejected exact optimisation as **computationally expensive**. A 1-D DP
  over a discretised length axis is `O(K·N²)` — milliseconds for a few thousand candidate
  edges — and returns the **global** optimum, not a stationary point. That directly refutes
  the stated reason for the heuristic. It is a small, sharp, defensible contribution.
- *Also differentiating:* their objective is **token-based** (`1 − S/U_b`). v3 already
  argues this is the wrong objective — padding from `L` to `B` costs `C(B) − C(L)` on a
  superlinear cost curve, not `B − L` tokens. That critique now has a concrete target.

### The differentiator, and BucketServe's own data makes it crisp

BucketServe reports **bucketing overhead below 1% of execution time**, and Fig. 6b shows
per-bucket processing time flat (0.12 s) as bucket count grows 1→8. On a GPU **a bucket is
nearly free**, which is exactly why they can afford to split and merge boundaries at
runtime.

On a compiled-shape accelerator that is false. Every boundary is an XLA executable: 30–120 s
to compile, plus HBM to hold the graph, and vLLM's own TPU docs warn that too many compiled
graphs "may lead to HBM OOM." **The cardinality budget is not a modelling convenience — it
is a hardware constraint that makes BucketServe's central mechanism (dynamic split/merge)
inapplicable.** That contrast, stated with both papers' numbers, is the cleanest framing of
this project's contribution found so far.

Their future work is "multi-level load balancing on multi-node clusters" — not our
direction.

## 4b. LAPS / PLA-Serve — found by forward citation. The closest work by a wide margin.

**[PLA-Serve / LAPS](https://arxiv.org/abs/2601.11589)** — She, Li, Du, Wu, Zheng, Xing,
Liu, Yao, Xue, Ho (MBZUAI + UNC Chapel Hill). **MLSys 2026, Seattle — oral.** Surfaced by
the forward-citation sweep of BucketServe, not by any keyword search in the plan's reading
list. Read from the MLSys slide deck (12 slides) plus abstract; **full paper not yet read.**

This is the outcome the forward sweep exists to catch, and it is much closer than
BucketServe.

### What LAPS does

**Innovation 2 — "CUDA Graph Bucketization for Batched Short Prefills."** A **2-D bucket
grid over (length L × batch B)**: L ∈ {8, 16, 32, 64, …}, B ∈ {1, 2, 4, 8, 16}, with **one
CUDA Graph captured per cell** at init, stored by key `(L, B)`, "allowed to dynamically
change based on hit frequency." At replay, each request does a **nearest-bucket lookup and
is padded up**: their example is 29→32, 47→64, 13→16, 58→64.

Their stated rationale is our rationale: *"Prefill length varies every request → tensor
shapes change → attention kernel cannot be captured."* CUDA Graphs are captured per fixed
shape. **This is compiled-shape bucketing, on GPU.**

**They present both sides of the promote-vs-queue decision, as named strategies:**

| Strategy | Grouping of {64, 64, 32, 16} | Description |
|---|---|---|
| **Memory first** | `batch(2×64) + batch(1×32) + batch(1×16)` | "minimal padding, length-homogeneous" — *queue in your own bucket* |
| **Latency first** | `batch(4×64)` | "same length" — ***promote 32 and 16 into the 64 bucket and eat the padding*** |

**Innovation 3 — Adaptive Wait-Depth (AWD) Scheduler.** Two adaptive thresholds per
scheduling round:

- `W_SLA` — "latest moment we can dispatch without any request missing its deadline"
- `W_GR` — **"Graph Fill Window: expected time for enough requests to arrive and fill a CUDA
  Graph bucket"**
- `W = clip(min{W_SLA, W_GR})`; wait up to `W` accumulating requests until batch depth `D`;
  dispatch immediately if any request's SLA slack ≤ σ; `W` and `D` update from observed
  arrival rate. Reported optimum ≈ 6 ms.

They also derive an analytic length boundary from a latency model —
`T_compute ≈ αL(L+2H) + βL`, `T_memory ≈ γ_w·L + γ_r·H`, giving
`L_m = (γ_w − β)/α ≈ 256 tokens` on H200 — and an M/G/1 head-of-line-blocking penalty
`ΔW = λp(1−p)(S_L−S_s)² / 2(1−ρ)`.

Baselines: vanilla SGLang under PD disaggregation, SGLang router. >30% prefill latency
reduction, 28% fewer SLO violations multi-instance, 35% throughput on Qwen2.5-32B.

### Assessment — this is very bad for the plan as written

Compare against v3's stated contributions:

| v3 claim | LAPS status |
|---|---|
| Requests are padded to one of N precompiled shapes | **Done** — CUDA Graph grid, nearest-bucket padding |
| Bucket ladder over length **and** batch | **Done** — the (L × B) grid *is* L1 × L2 |
| Promote-and-pad vs queue-and-wait | **Both implemented**, as memory-first / latency-first |
| Wait for the right bucket vs dispatch now | **Done** — AWD, and `W_GR` is literally "time to fill the bucket" |
| Cost model deriving a length boundary | **Done** — `L_m` from an analytic roofline-style model |
| Adaptive to arrival rate | **Done** — `W`, `D` update from observed rate |

**The spine, as v3 phrases it, is substantially published — at the immediately preceding
edition of the target venue.** My earlier verdict that "nothing found makes the
promote-vs-queue decision" was based on BucketServe and keyword search; it does not survive
this paper.

### What LAPS does *not* do — the remaining sliver, stated honestly

1. **It offers both strategies; it does not study which wins when.** Memory-first and
   latency-first are presented as configuration choices, not as a policy question with a
   workload-dependent answer, a cost model for the promotion, or a comparison. "When does
   promoting beat waiting, and by how much in dollars" is not answered.
2. **No cardinality-budgeted optimisation of the grid.** Boundaries are powers of two,
   adjusted by hit frequency. No optimality claim, no DP, no budget constraint.
3. **Short prefills only.** The whole design rests on short prefill behaving like decode
   ("stable compact shapes → perfect for CUDA Graph"); long prefill goes to an uncaptured LP
   instance. On TPU *everything* is compiled, so the problem does not partition this way.
4. **GPU / CUDA Graph, not TPU / XLA** — and the cost asymmetry still favours us. LAPS can
   let the grid "dynamically change based on hit frequency" because capture is cheap. An XLA
   recompile is 30–120 s plus HBM. **The cardinality budget remains a real constraint that
   LAPS does not face** — the same argument that differentiates us from BucketServe.
5. Their `L_m` is analytic (roofline); v3 proposes a **measured** cost curve.

Whether that sliver is an MLSys paper is a judgement call, not a search result. See
`kill_condition.md`.

## 5. Admission control and batch composition

Searched: Sarathi-Serve, QLM, Andes, Llumnix, SLOs-Serve, FairBatching, AlignedServe.

- **SLOs-Serve** ([2504.08784](https://arxiv.org/pdf/2504.08784)) — "soft admission control
  mechanism that guarantees SLO attainment for admitted requests," periodically selecting
  an optimal subset of new requests. Admission control in LLM serving is established; the
  novelty cannot be "we do admission control."
- **QLM** ([2407.00047](https://arxiv.org/pdf/2407.00047)) — queue management for
  SLO-oriented serving.
- **Length-aware admission already exists.** One of the surveyed systems "sorts the pending
  queue by prompt length to reduce padding overhead." **Track this down and cite it** — it
  is the nearest neighbour to our claim and reviewers will know it. ⚠️ *Source not yet
  pinned; see Open threads.*
- A frequently repeated motivation figure — **padding overhead of 60–80% for typical batch
  sizes** — appears in this literature. Pin the primary source; it is either the paper's
  best motivating number or a number we must argue down given finding 2.
- **Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not
  Just Heuristics** ([2605.01280](https://arxiv.org/abs/2605.01280), Zhou, May 2026,
  submitted to SOSP 2026) — argues serving systems' algorithmic cores are unchanged
  classical heuristics (JSQ/round-robin routing, FIFO scheduling, LRU eviction) and the
  field needs provable-guarantee algorithms. **This helps us.** It is a citable framing for
  why a DP with an optimality guarantee over a measured cost curve is a contribution and
  not just engineering.

**Not yet searched properly:** Sarathi-Serve, Andes, and Llumnix were named in the plan
but did not surface with usable detail in this pass. Forward citations of RPA and
BucketServe were not systematically enumerated. Both are Open threads.

## 6. Compiled-shape bucketing outside LLMs — old, and it gives us our vocabulary

As expected: bucketing shapes is **not novel and must be conceded early**. Two findings are
more useful than that concession, though.

**`tf.data.experimental.bucket_by_sequence_length` has a `pad_to_bucket_boundary` flag.**
That single API distinguishes the two regimes this project is about:

| Flag | Padding target | Who lives here |
|---|---|---|
| `False` (default) | max length **in the batch** | GPU runtime batching — **BucketServe's `Waste_Ratio = (S_max − S_avg)/S_max`** |
| `True` | the **bucket boundary** | compiled-shape accelerators — us |

This is a clean, citable way to say what the paper is about in one sentence, and it shows
the distinction is recognised in a decade-old API rather than invented here.

**TensorRT optimisation profiles are the same problem in another stack.** Dynamic dims are
declared with `-1`; you must supply one or more profiles at build time, each a
`(min, opt, max)` range; profiles may be disjoint or overlapping; and the practitioner
guidance is to "ensure that all requests fall within the `opt` range." Docs are explicit
that "dynamic shapes are convenient but not performance-friendly" and that fixed shapes
permit more aggressive optimisation.

So: a **cardinality-budgeted set of compiled shape ranges chosen against a workload
distribution** is a live engineering problem in TensorRT and ONNX Runtime too. Nothing found
formalises profile selection as an optimisation problem. That widens the paper's claimed
scope from "TPU" to "compiled-shape accelerators generally" *and* raises the bar — the
related-work section must state plainly that bucketing is old and that the contribution is
the cost model, the cardinality-budgeted optimum, and the admission policy.

## 7. Vidur — the methodological precedent, and the "why not use it" answer

[Vidur (2405.05465)](https://arxiv.org/abs/2405.05465), **MLSys 2024**, Microsoft Research.
Models operator performance by combining experimental profiling with predictive modelling;
estimates end-to-end latency and throughput. Reported fidelity: **<9% error** overall, TTFT
within 5–10%, throughput within 10–15%. Vidur-Search finds the best deployment config for
LLaMA-2-70B **in one hour on a CPU** versus ~42K GPU-hours of real exploration.

Two consequences, both important:

1. **It validates the method.** Calibrate-then-simulate with reported fidelity is an
   established MLSys contribution shape, at the same venue being targeted. v3's MAPE < 15%
   bar sits right at Vidur's own throughput error band — defensible, and worth citing as
   the precedent rather than presenting as our own invention.
2. **A reviewer will ask why we did not just use Vidur, and we need the answer written
   down.** The honest answer as far as this pass goes: Vidur models GPU operators and
   parallelism configs; nothing found indicates it models compiled-shape bucket ladders,
   XLA executables, or TPU. But this has **not** been verified against Vidur's actual
   extension points, and "we didn't check" is not an answer. Open thread.

Also surfaced: **Frontier ([2605.21312](https://arxiv.org/pdf/2605.21312))**, "Towards
Comprehensive and Accurate LLM Inference Simulation" — newer, unexamined. Check it.

**Varlen serving (ByteTransformer, Effective Transformer)** — still not searched. Open
thread.

---

## Incidental finding: the plan's primary model is not on the tested list

`tpu-inference`'s recommended-models page lists as fully tested and production-ready:
Gemma **gemma-4-26B-A4B-it, gemma-4-31b-it, gemma-3-27b-it**; **Llama-3.1-8B-Instruct**
and Llama-3.3-70B-Instruct; Qwen3 4B / 30B-A3B / Coder-480B / 3.5-397B.

**Gemma-3-4b — the plan's primary model — is not on it. Llama-3.1-8B-Instruct is.**
The plan's primary/secondary assignment should probably invert. Also relevant: TP shows ❌
for Torchax and ✅ for Flax, so the framework choice interacts with the TP=4 decision.
Recorded in `DECISIONS.md` as an open item; not decided here.

---

## Forward-citation sweep — what it surfaced

Run via Semantic Scholar citations of BucketServe. RPA's citation list returned HTTP 429 and
**has not been retrieved — still outstanding.**

Citing BucketServe: *Generative AI at the Edge* (survey); *ASAP* (MoE prefill
disaggregation); *RouteBalance* (model routing / load balancing); ***Requests of a Feather
Must Flock Together*** ([2605.06046](https://arxiv.org/abs/2605.06046), ISCA 2026 — FEATHER,
an RL scheduler trading batch size against *prefix* homogeneity, 2–10× throughput; different
axis from ours, but the same "batch composition is a learned tradeoff" shape); *InversePep*
(irrelevant); and ***LAPS / PLA-Serve*** — §4b, which changes the verdict.

## Open threads — what this pass did not close

1. **RPA's forward citations** — blocked on rate limit, not yet retrieved. Given that
   BucketServe's list alone produced LAPS, this is not a formality.
2. **Read the LAPS paper in full** (12 pages, arXiv 2601.11589). §4b is from the MLSys slide
   deck. Every "what LAPS does not do" claim there needs confirming against the paper before
   anything is built on it.
3. **Can Vidur be extended to compiled-shape ladders?** Also **Frontier**
   ([2605.21312](https://arxiv.org/pdf/2605.21312)).
4. **Pin the "sorts pending queue by prompt length" system** and the 60–80% padding-overhead
   figure to primary sources.
5. **Sarathi-Serve, Andes, Llumnix** — proper reads (all in BucketServe's bibliography).
   **Varlen serving** — ByteTransformer, Effective Transformer.
6. **Confirm on hardware:** chunked prefill's TPU-specific default and TPU
   `max_num_batched_tokens`; prefix caching default.

## Confidence

BucketServe was read in full (9 pages); its findings are firm. **LAPS is from a 12-slide
MLSys deck plus abstract — the single most consequential finding here rests on the
thinnest evidence, and reading the paper is the top priority.** Everything else is
abstracts, official documentation, and HTML renders.

The `gapcache` pattern repeated exactly: keyword search over the plan's own reading list
found nothing doing promote-vs-queue, and the first forward-citation query surfaced a paper
that does most of it, at the target venue's previous edition. **The lesson is that the
reading list is not the search.** RPA's forward citations are still unretrieved, so the
sweep is incomplete and the current verdict could still get worse.
