# Kill condition verdict — W0 gate

**Date:** 2026-08-07. Evidence in `prior_art.md`. BucketServe read in full; everything else
is docs and abstracts.

The plan's stated gate: *"A negative on the prior-art check or the oracle-gap probe stops
the project in week 0."* All checks runnable without hardware are now run.

---

## Verdict: the gate does not fire. The spine is clear. L1 is dead. The ladder claim shrinks.

| Check | Result |
|---|---|
| Shape polymorphism removes the ladder | **No.** The ladder is real, documented, configurable (`VLLM_TPU_BUCKET_PADDING_GAP`). Premise survives and is easier to manipulate than the plan assumed. |
| Chunked prefill deletes L1 | **Yes.** Default-on in vLLM V1, validated in `tpu-inference`. L1 padding is bounded by one chunk, not by the ladder. |
| **Prior art claims the spine (promote-vs-queue)** | **No — settled by full read.** BucketServe assigns every request to the bucket containing its length and moves *boundaries*, never requests. Nothing found makes this decision. |
| Prior art claims ladder optimisation | **Partly.** BucketServe publishes the Lloyd–Max stationarity condition (its Eq. 4), then rejects exact optimisation as "computationally expensive" and uses midpoint bisection. Our DP is now a refutation of a stated claim, not a novelty. |

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

1. **The forward-citation sweep has not been run.** This is the search that killed
   `gapcache` — five systems its reading list never anticipated. Keyword search found no
   successor doing promote-vs-queue on compiled shapes, but keyword search is exactly what
   missed those five. **Until this runs, the verdict is provisional in the same way
   `gapcache`'s pre-search optimism was.** Now the top risk.
2. **Oracle headroom turns out small.** Untestable without hardware. §1 of `prior_art.md`
   cuts in our favour — the default ladder is powers of two, which is coarse — but that is
   an argument, not a measurement.
3. **Length-aware admission is already published.** At least one surveyed system "sorts the
   pending queue by prompt length to reduce padding overhead." Nearest published neighbour;
   source still not pinned.
4. **Vidur already does this, or extends to it cheaply.** MLSys'24, same venue family,
   <9% error. If it models compiled-shape ladders, the simulator contribution evaporates and
   the right move is to build on it. Not checked against its extension points.

## Recommendation

**Continue. Still do not provision hardware.** Three reasons, one of which has changed:

- The remaining free action — the forward-citation sweep — is now the *top* risk rather than
  the second. Spending $144 on a W0 VM before running the search that killed the last
  project would repeat the exact mistake this gate exists to prevent.
- The L1 finding changes what the hardware experiments *are*. `e10_latency_steps` was
  specified as a prompt-length staircase producing the L1 cost curve `C(B)`. Under chunked
  prefill that staircase largely flattens. The primitive needs respecifying against the
  outer compiled shape and the batch bucket before any TPU time is bought — otherwise the
  first hardware money is spent measuring something the paper no longer claims.
- Nothing about `e01_oracle_gap` or `e02_stock_baseline` changes, so no time is lost.

**This is a recommendation, not a decision.** Whether a one-level-plus-policy paper is worth
twelve weeks is the author's call, as is whether to provision in parallel and accept the
forward-citation risk.

## Consequences for the plan, if this verdict stands

1. **`e20_ladder_dp` must be reframed** from "we optimise the ladder" to "BucketServe
   derived the optimality condition and rejected exact optimisation as computationally
   expensive; a 1-D DP is `O(K·N²)`, milliseconds, and globally optimal — here is the
   refutation, on a cost curve rather than token counts." Smaller claim, better defended,
   with the citation supplied by us rather than by a reviewer.
2. **Related work must concede up front** that shape bucketing is old — `tf.data`'s
   `pad_to_bucket_boundary`, TensorRT optimisation profiles — and locate the contribution in
   the cost model, the cardinality-budgeted optimum, and the admission policy.
3. **Vidur must be cited as the methodological precedent** for calibrate-then-simulate at
   this venue, not treated as a competitor to avoid mentioning.
4. **The scope may widen usefully.** TensorRT profile selection is the same problem in
   another stack, so "compiled-shape accelerators" is honest framing, not overreach.

## What this does not license

The plan's pre-committed response to a firing gate (narrow the venue, do not pivot to a
fourth topic) is **not** triggered. The gate did not fire. L1's death is a scope change
the plan explicitly anticipated and pre-authorised, not a kill.
