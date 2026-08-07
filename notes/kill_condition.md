# Kill condition verdict — W0 gate, first pass

**Date:** 2026-08-07. Evidence in `prior_art.md`. Depth: docs + abstracts, not full reads.

The plan's stated gate: *"A negative on the prior-art check or the oracle-gap probe stops
the project in week 0."* Two of the three checks that can be run without hardware are now
run.

---

## Verdict: the gate does not fire, but one of the paper's three levels does.

Neither the premise nor the spine is dead. **L1 is.** Summary:

| Check | Result |
|---|---|
| Shape polymorphism removes the ladder | **No.** The ladder is real, documented, and configurable. Premise survives — and is easier to manipulate than the plan assumed. |
| Chunked prefill deletes L1 | **Yes.** Chunked prefill is on by default in vLLM V1 and validated in `tpu-inference`. L1 padding is bounded by one chunk, not by the ladder. |
| Prior art claims the spine | **Not established, not refuted.** BucketServe is closer than v2 allowed. Requires a full read before code. |

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

## What could still kill it

Ranked by probability × cost:

1. **BucketServe already makes the promote-vs-queue decision.** Not determinable from the
   abstract. It does bucket split/merge *and* priority-aware SLO scheduling, so it is
   substantially closer than v2's "designs ladders, doesn't study admission." If it makes
   this decision explicitly, the contribution reduces to "same idea, on compiled shapes,
   on TPU" — survivable but much weaker. **Settle this with a full read before writing any
   code.**
2. **The forward-citation search has not been run.** This is the search that killed
   `gapcache` — five systems its reading list never anticipated. Until it is run, this
   verdict is provisional in the same way `gapcache`'s pre-search optimism was.
3. **Oracle headroom turns out small.** Untestable without hardware. But finding 1 in
   `prior_art.md` cuts in our favour: the default ladder is powers of two, which is coarse.
4. **Length-aware admission is already published.** At least one surveyed system "sorts the
   pending queue by prompt length to reduce padding overhead." Nearest neighbour to our
   claim; source not yet pinned.

## Recommendation

**Continue — but do not provision hardware yet.** Three reasons:

- The two remaining prior-art actions (BucketServe full read; forward citations of RPA and
  BucketServe) are free, and item 1 above can still end the project. Spending $144 on a W0
  VM before running the search that killed the last project would repeat the mistake this
  gate exists to prevent.
- The L1 finding changes what the hardware experiments *are*. `e10_latency_steps` was
  specified as a prompt-length staircase producing the L1 cost curve `C(B)`. Under chunked
  prefill that staircase largely flattens. The primitive needs respecifying against the
  outer compiled shape and the batch bucket before any TPU time is bought — otherwise the
  first hardware money is spent measuring something the paper no longer claims.
- Nothing about `e01_oracle_gap` or `e02_stock_baseline` changes, so no time is lost.

**This is a recommendation, not a decision.** The call on whether a one-level-plus-policy
paper is worth twelve weeks is the author's, not mine — as is whether to read BucketServe
first or provision in parallel and accept the risk.

## What this does not license

The plan's pre-committed response to a firing gate (narrow the venue, do not pivot to a
fourth topic) is **not** triggered. The gate did not fire. L1's death is a scope change
the plan explicitly anticipated and pre-authorised, not a kill.
