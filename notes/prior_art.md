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

## 4. BucketServe — the closest work, and the main novelty threat

[BucketServe (2507.17120)](https://arxiv.org/abs/2507.17120), Jul 2025. Groups requests
into size-homogeneous buckets by sequence length to minimise padding; **dynamically
adjusts bucket boundaries (split/merge)**; determines safe batch sizes from real-time GPU
memory state; **priority-aware scheduling to meet latency SLOs**. Baselines UELLM and
DistServe; 3.58× throughput, 1.93× load under 80% SLO attainment.

**v2's characterisation — "BucketServe designs ladders, it does not study admission
against a fixed ladder" — is too comfortable.** BucketServe does bucket design *and*
scheduling, on a GPU, with SLO-aware prioritisation. The distinguishing claims available
to us are narrower than v2 implied:

1. BucketServe's buckets are a **runtime batching abstraction on GPU**, where any batch
   shape is legal. Ours are **compiled executables** under a hard cardinality budget —
   changing the ladder costs an XLA recompile, which is precisely why the cardinality
   constraint exists. That is a genuinely different problem.
2. It is GPU-only (abstract references GPU memory and OOM).
3. Whether it makes the **promote-vs-queue** decision explicitly is not determinable from
   the abstract.

**Point 3 must be settled by a full read before any code is written.** This is the single
highest-value remaining action in the gate.

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

## 6. Compiled-shape bucketing outside LLMs

Not yet searched. TensorRT optimisation-profile selection, ONNX Runtime dynamic shapes, TF
Serving batch buckets, `tf.data.bucket_by_sequence_length`. Expect these to be genuine
prior art for *the idea of bucketing shapes* — the defensible novelty is the cost model and
the admission decision, never the bucketing itself. The related-work section must concede
this early and clearly. Open thread.

## 7. Varlen serving (ByteTransformer, Effective Transformer) and Vidur

Not yet searched. Both were on the plan's list. Vidur matters methodologically — a reviewer
will ask why we built a simulator instead of using it. Open threads.

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

## Open threads — what this pass did not close

1. **Full read of BucketServe.** Does it make the promote-vs-queue decision explicitly?
   Highest-value remaining action.
2. **Pin the "sorts pending queue by prompt length" system** and the 60–80% padding-overhead
   figure to primary sources.
3. **Sarathi-Serve, Andes, Llumnix** — proper searches.
4. **Forward citations of RPA and BucketServe** — systematic enumeration, the search that
   killed `gapcache`.
5. **Compiled-shape bucketing outside LLMs** (§6) and **varlen serving + Vidur** (§7).
6. **Confirm on hardware:** chunked prefill's TPU-specific default and TPU
   `max_num_batched_tokens`; prefix caching default.

## Confidence

Everything above is from abstracts, official documentation, and HTML renders — not full
paper reads. `gapcache` died because forward/keyword search surfaced five systems its
reading list never anticipated, and **item 4 of the open threads is exactly that search,
still outstanding.** Treat this pass as sufficient to justify continuing to spend *time*,
and not yet sufficient to justify spending *money* on hardware.
