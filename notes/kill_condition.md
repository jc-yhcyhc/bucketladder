# Kill condition verdict — W0 gate

**Date:** 2026-08-07. Evidence in `prior_art.md`. BucketServe read in full; everything else
is docs and abstracts.

The plan's stated gate: *"A negative on the prior-art check or the oracle-gap probe stops
the project in week 0."* All checks runnable without hardware are now run.

---

## Verdict: the gate fires. The spine is substantially published — at MLSys 2026.

**This supersedes the verdict I wrote earlier today.** That one said "the spine is clear,
nothing found makes the promote-vs-queue decision." It was based on BucketServe plus keyword
search over the plan's reading list. The forward-citation sweep — run afterwards, and the
one thing the plan flagged as the top risk — surfaced **LAPS / PLA-Serve (MLSys 2026)**,
which does most of the spine. See `prior_art.md` §4b.

| Check | Result |
|---|---|
| Shape polymorphism removes the ladder | **No.** Ladder is real, documented, configurable. Premise survives. |
| Chunked prefill deletes L1 | **Yes.** Default-on in vLLM V1, validated in `tpu-inference`. |
| Prior art claims ladder optimisation | **Partly.** BucketServe publishes the Lloyd–Max condition (Eq. 4), rejects exact optimisation as "computationally expensive," uses midpoint bisection. |
| **Prior art claims the spine** | **Yes, largely — LAPS.** A CUDA-Graph bucket grid over (length × batch); nearest-bucket padding; *both* promote and queue implemented as "latency first" / "memory first"; an Adaptive Wait-Depth scheduler whose `W_GR` is literally "expected time to fill a bucket"; an analytic cost model giving a length boundary; adaptation to observed arrival rate. |

LAPS is not a distant neighbour. It is the plan's Phase 2 and Phase 3, on GPU, published
four months ago at the previous edition of the venue this was aimed at.

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

## What is genuinely still open

Stated as narrowly and honestly as I can. LAPS **offers** promote and queue as two
configuration strategies; it does not **study** them.

1. **Nobody has answered "when does promoting beat waiting, and by how much?"** LAPS gives
   memory-first and latency-first as choices, with no cost model for the promotion, no
   comparison, and no workload-dependence result. This is real, and it is the question v3
   was built around — but it is a *characterisation* result, not a new mechanism, and the
   mechanism is what MLSys industrial-track papers are usually made of.
2. **Nobody optimises the grid under a cardinality budget.** LAPS uses powers of two adjusted
   by hit frequency; BucketServe derives the Lloyd–Max condition and then rejects computing
   it. A DP that is globally optimal in `O(K·N²)` refutes BucketServe's stated reason. Small,
   sharp, defensible — and roughly one section, not a paper.
3. **Nobody does this on TPU/XLA, and the cost asymmetry is real.** LAPS lets its grid
   "dynamically change based on hit frequency" because CUDA Graph capture is cheap;
   BucketServe reports bucketing overhead <1% and splits/merges at runtime. On XLA every
   boundary is a 30–120 s compile plus HBM, and vLLM's own docs warn of HBM OOM from too many
   graphs. **The cardinality budget is a hardware constraint neither faces.** This is the
   strongest remaining differentiator — and it is the same "port it to TPU" argument
   `gapcache`'s verdict called "genuine and substantive" but declined to treat as sufficient
   on its own.
4. **LAPS partitions short vs long prefill; TPU does not permit that partition.** Their
   design rests on short prefill having "stable compact shapes." On TPU everything is
   compiled, so the problem does not decompose the same way. Possibly a real structural
   difference; possibly a detail.

## Recommendation

**Do not provision hardware. Do not start building. Read LAPS in full first, then decide.**

§4b rests on a 12-slide deck plus an abstract — the most consequential finding in this gate
is also the thinnest-evidenced, and it would be wrong to kill or continue the project on
that basis. The paper is 12 pages and free. Two outcomes:

- **If the full read confirms the deck**, then items 1–4 above are the entire remaining
  contribution, and the honest description is *"a characterisation study of a tradeoff LAPS
  exposed but did not analyse, plus a DP, ported to TPU."* That is a workshop paper or a
  short paper. It is not, in my judgement, an MLSys industrial-track submission — and this is
  exactly the situation the plan's pre-committed response was written for.
- **If the full read shows LAPS's bucketization is narrower than the deck suggests** — short
  prefill only, no real promotion policy, grid fixed at powers of two — then item 1 grows
  back toward a paper, and the TPU cardinality-budget argument (item 3) carries more weight.

**The pre-committed response now applies, and I am not going to quietly walk it back.**
`DECISIONS.md`, 2026-08-07: *"if the gate fires, the response is narrow the venue — a
workshop submission, or skip this cycle and target the next — not pivot to a fourth topic."*
That was written before the outcome was known, precisely so it would not be renegotiated
under deadline pressure. It should be honoured, and the options are:

- **(a) Narrow the venue.** Take items 1–3 as a workshop paper or short paper. Cheap, honest,
  finishes something. Some hardware spend, well under the ceiling.
- **(b) Skip the cycle.** Target the next MLSys with a properly scoped problem, using the
  ~$0 spent so far and three gates' worth of hard-won judgement about this area.
- **(c) Re-aim at the one thing LAPS cannot do** — the TPU/XLA cardinality budget as the
  *central* claim rather than a differentiator, i.e. "recompilation cost makes bucket
  selection a fundamentally different problem on compiled-shape accelerators, and here is
  the theory and the measurement." This is the most interesting option and the least
  proven; it needs its own prior-art pass before it is anything more than a hunch.

**This is not my decision to make.** The gate fired; what to do about it is the author's
call. What I would push back on is a fourth pivot to an unrelated topic — that pattern is
what produced three gates in six days, and the pre-commitment exists to stop it.

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
