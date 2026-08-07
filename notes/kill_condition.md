# Kill condition verdict — W0 gate

**Date:** 2026-08-07. Evidence in `prior_art.md`. **BucketServe and LAPS both read in full**
(9 and 12 pages); everything else is docs and abstracts.

The plan's stated gate: *"A negative on the prior-art check or the oracle-gap probe stops
the project in week 0."* All checks runnable without hardware are now run.

---

## Verdict: the gate does NOT fire. Reframing is mandatory; the spine survives.

**This is the third verdict today and it supersedes both earlier ones.** History, because it
matters more than the conclusion:

1. Keyword search over the plan's reading list → "spine is clear." **Wrong** — the reading
   list is not the search.
2. Forward-citation sweep surfaced LAPS; assessed from its **12-slide deck** → "gate fires,
   spine is published." **Also wrong** — I treated a presentation slide's framing as the
   paper's contribution.
3. **Full 12-page paper read** → the deck's "memory first vs latency first" comparison **does
   not exist in the paper.** Algorithm 1 has one policy: greedily group by length to
   *minimise* padding, then pad to the nearest captured shape.

| Check | Result |
|---|---|
| Shape polymorphism removes the ladder | **No.** Ladder is real, documented, configurable. |
| Chunked prefill deletes L1 | **Yes.** Default-on in vLLM V1, validated in `tpu-inference`. |
| Prior art claims ladder optimisation | **No — and two papers say why they didn't.** BucketServe derives the Lloyd–Max condition then rejects computing it as "computationally expensive." LAPS says "the number of graphs must be limited" and uses a fixed power-of-two grid. |
| **Prior art claims the spine (promote vs queue)** | **No.** LAPS's AWD *waits* for a bucket to fill (`W_GR`) and then pads to nearest. It never asks whether dispatching now into a larger already-warm bucket beats waiting. Untouched. |

### Why LAPS is not the scoop I called it

- **Short prefill only, ≤256 tokens, by explicit design.** Their stated reason: for general
  prefill, "graph capture [is] expensive and rarely amortized. Hence, mainstream serving
  systems **avoid CUDA Graphs in prefill**." On GPU, compiled-shape bucketing is an opt-in
  optimisation for a cheap corner. **On TPU it is mandatory at every length for every
  request.** Difference in kind, and LAPS supplies the reason.
- **There is a fallback.** Algorithm 1: `if G* exists then pad B to G* else use standard
  prefill kernel`. **XLA has no uncaptured path.**
- **Their own ablation shows bucketing alone is a wash on GPU** — "enabling CUDA Graphs alone
  yields limited improvements and can even degrade throughput." Figure 6's Graph-only arm
  tracks baseline; the wins come from disaggregation. On TPU you cannot turn it off.
- **The grid is fixed powers of two**, `L ∈ {8…256}`, `B ∈ {1…64}`, captured at init. No
  optimisation, no adaptation to workload.

### The strongest single finding: LAPS §4.2 names our problem and declines it

> "Each graph is bound to a fixed kernel configuration… **the number of graphs must be
> limited to balance memory usage and performance.**" — with measurements: 228/240/277 MB per
> graph, and **8–12 seconds to capture one**.

One correction against myself: capture is **not** milliseconds, so the XLA cost asymmetry is
~3–10×, not orders of magnitude. But this strengthens the position rather than weakening it —
**both closest papers state the cardinality constraint explicitly and neither solves it.**
That is about as clean an opening as prior art ever provides.

## What survives, and it is the part the plan already called the spine

The plan pre-committed a response to exactly this outcome: *"If chunked prefill is on by
default and cannot be disabled, L1 is largely dead — reframe to L2 + admission and say
so."* That is now the operative branch. Specifically:

- **Dead:** "total prompt tokens → prefill bucket" as a cost parameter. A full 8,192-token
  chunk is already a power of two and pads to nothing; only each request's final partial
  chunk is padded. On a 10k-token prompt that is ~2.4% waste versus ~64% unchunked.
- **Alive:** the **outer compiled shape** — the batch/decode bucket the executable is
  compiled for. RPA removes ragged waste *inside* attention; it does not remove the shape
  the kernel was specialized for, because RPA v3 is itself compiled into specialized
  decode/prefill/mixed variants.
- **Alive and unaffected:** the **promote-vs-queue admission decision**. Chunked prefill
  changes how prefill is shaped; it does not change what happens when the bucket a request
  needs is saturated. The spine is untouched.
- **Newly attractive:** the ladder is exposed as `VLLM_TPU_BUCKET_PADDING_GAP` (linear
  buckets from 16 to `max_model_len`), with power-of-2 exponential padding as the default.
  A ready-made independent variable — no scheduler patching to change the ladder — and a
  coarse default that plausibly leaves real headroom.

**The paper narrows from three levels to one level plus a policy.** That is a smaller
paper than v2 described and a more honest one. It is also, notably, the paper v2 said it
wanted: *"Admission policy is the spine. Ladder characterization becomes the setup, not
the claim."*

## The sharpest available framing, courtesy of BucketServe's own numbers

BucketServe reports bucketing overhead **below 1% of execution time**, flat as bucket count
grows 1→8 (its Fig. 6b). On a GPU a bucket is nearly free — which is precisely why it can
split and merge boundaries at runtime.

On a compiled-shape accelerator that is false. Every boundary is an XLA executable: 30–120 s
to compile, plus HBM to hold the graph, and vLLM's TPU docs warn too many compiled graphs
"may lead to HBM OOM." **The cardinality budget is a hardware constraint, not a modelling
convenience, and it makes BucketServe's central mechanism inapplicable.** Stated with both
papers' numbers, that is the cleanest one-paragraph statement of this project's contribution
found so far — and it came from the closest competing work.

## What could still kill it

Ranked by probability × cost:

1. **The forward-citation sweep is only half done.** BucketServe's citation list was
   enumerated and produced LAPS. **RPA's returned HTTP 429 and was never retrieved.** One
   citation list alone overturned the verdict twice today, so the missing half is the top
   risk. Also unexamined: **Multi-Bin Batching** ([2412.04504](https://arxiv.org/abs/2412.04504)),
   which LAPS names alongside BucketServe as the most related length-bucketing work.
2. **Oracle headroom turns out small.** Untestable without hardware. §1 of `prior_art.md`
   cuts in our favour — the default ladder is powers of two, which is coarse — but that is
   an argument, not a measurement.
3. **Length-aware admission is already published.** At least one surveyed system "sorts the
   pending queue by prompt length to reduce padding overhead." Nearest published neighbour;
   source still not pinned.
4. **Vidur already does this, or extends to it cheaply.** MLSys'24, same venue family,
   <9% error. If it models compiled-shape ladders, the simulator contribution evaporates and
   the right move is to build on it. Not checked against its extension points.

## Recommendation: continue. Reframe, do not narrow the venue.

The pre-committed response is **not** triggered — the gate did not fire. What is required is
a reframing, and the prior art now dictates it fairly precisely:

1. **LAPS is the primary related work** and must be distinguished in the paper's first
   paragraph, not buried. The distinction is clean and factual: they bucket a cheap corner of
   the workload on hardware where bucketing is optional and has a fallback; TPU compiles
   every shape for every request with no fallback.
2. **The spine is narrower and sharper than v3 wrote it.** Not "promote-vs-queue is
   unstudied" but: *LAPS waits for a bucket to fill and then pads to nearest; nobody asks
   whether dispatching now into a larger already-warm bucket beats waiting.* That is the
   contribution, and it is defensible because the closest system implements exactly one half
   of it.
3. **The DP now has two independent invitations**, both quotable: BucketServe's "computa-
   tionally expensive to calculate in practice" and LAPS's "the number of graphs must be
   limited." Frame it as answering a stated open cost concern, not as novelty.
4. **The compile-budget argument is the paper's spine, not a differentiator.** Both papers
   name the constraint; on TPU it binds hardest and cannot be escaped.

Still do not provision hardware — but for a different reason than before. The remaining
prior-art work is cheap, and `e10_latency_steps` needs respecifying after the chunked-prefill
finding regardless. Nothing is lost by closing those first.

## What is genuinely still open — after the full read

1. **Nobody compares promoting into a larger already-warm bucket against waiting for the
   right one.** LAPS's AWD computes `W_GR`, the expected time to fill the target depth,
   waits, then pads to the nearest captured shape. It never evaluates dispatching *now* into
   a bigger bucket instead. This is the spine, and it is untouched by the closest system —
   which implements exactly one half of it.
2. **Nobody optimises the ladder under a cardinality budget**, and both closest papers say
   why not, quotably. BucketServe: the optimal boundary is "computationally expensive to
   calculate in practice." LAPS: "the number of graphs must be limited to balance memory
   usage and performance," then a fixed power-of-two grid. A globally-optimal `O(K·N²)` DP
   answers a stated open cost concern.
3. **Nobody covers the full length range, because on GPU nobody has to.** LAPS caps at 256
   tokens by design; general prefill makes "graph capture expensive and rarely amortized."
   On TPU every request at every length is compiled, with no fallback kernel.
4. **No measured cost curve.** LAPS's `L_m` is analytic/roofline; BucketServe's objective is
   token count. v3's `C(B) − C(L)` on a measured superlinear curve is differentiated from
   both.

## What does not change

- The engineering record stands: L1 is dead, the ladder is `VLLM_TPU_BUCKET_PADDING_GAP`,
  v5e-4 is the right chip, Vidur is the methodological precedent, `tf.data`'s
  `pad_to_bucket_boundary` names the regime.
- **Total spend remains $0**, across three killed or narrowed projects. The gate discipline
  is working — this is the third time it has caught something before hardware, and the first
  time it caught it in under a day.

## What this does not license

The plan's pre-committed response to a firing gate (narrow the venue, do not pivot to a
fourth topic) is **not** triggered. The gate did not fire. L1's death is a scope change
the plan explicitly anticipated and pre-authorised, not a kill.
