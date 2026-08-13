# Review v6 — what was addressed, and what is still open

External review received 2026-08-13 (recommendation: borderline, leaning accept,
confidence 4/5). Session 27 acted on the findings below; this file records what
remains so the next session does not have to reconstruct it from chat.

## Addressed in session 27 (commits 134bf30, d03bb9d, 30d9254, 9876583)

| finding | outcome |
|---|---|
| xrefs off by one | **Real.** LaTeX numbered the unnumbered Abstract as 1. Fixed; NUM-DRIFT lint check added and verified firing. |
| M1 — 15.0% KV price misattributed | **Real.** Provenance error across memory fractions. Retracted; cliff measured at 0.85; true price 8.8%, and it belongs to the 21-shape ladder, not the optimisation. Failure 14. |
| M2 — lever confounds count with placement | **Real, and it inverted the result.** 14 shapes match 21 at prompt 3000 and boot at stock 0.92 with full capacity. Recommendation is now placement, not cardinality. |
| M3 — n≥4 rows have no valid control | Accepted. Sign vs magnitude separated; only n=1,2 carry intervals. |
| M4 — resolve §4.3/§4.10 tension from data in hand | **Partly disputed.** `iteration_tokens_total` bins on powers of two, coarser than the ladder spacing under comparison, so it cannot yield padded tokens per step. Stated in §4.10. |
| M5 — GPU arm is one point, may test only D3 | Accepted. Cross-architecture claim scoped to the request dimension. |
| minor: LENS 3-way inconsistency, 85% hedge, RPA as bound, Figure 1 "localised", stray ",.", dangling 35.9% | All fixed. RPA now stated as <0.7 µs/slot against 27.5 µs — a bound, and stronger than the "zero" it replaced. |

## Still open

**M8 — no throughput or goodput curve.** The strongest remaining objection. The
benefit is measured in latency at low concurrency and the cost in tokens of KV
cache, and the two are never converted. A reader cannot judge whether trading
capacity for latency is a good deal. Less pressing now that the placement result
costs no capacity — but the 21-shape arm still needs it, and the paper recommends
about memory budgets without a throughput axis anywhere.

**M7 — workload realism.** Prefix caching is off and asserted throughout while
production vLLM defaults it on, and it changes the number and length of prefill
dispatches, which is the quantity that sets padded share. Fixed output lengths,
no preemption, no multi-turn, no production trace. After §4.4's range was
withdrawn as confounded on offered tokens, **no quantitative statement about how
much padding a realistic workload executes survives** — so a reader cannot tell
whether their workload is in scope for the recommendation. A rate-matched *and*
token-matched re-run of §4.4 would close the loop.

**M6 — the release-timing 26% rests on one load point.** §9 uses it to attribute
others' reported bucketing gains to batch composition. No saturation curve, so it
could be a large effect near saturation or an artifact of where 25 req/s sits.
The "cannot distinguish the two" sentence is defensible; the attribution is not.

**Minor leftovers.** MFU FLOP-accounting convention unstated; the MFU/HBM table
still prints the n=128 and n=256 columns the text disclaims as queue-contaminated;
bare citations with no bibliography; the registered-but-unrun W8 dtype prediction
sits in a results section and belongs in future work unless run; flatness and
share-of-nominal-padding are near-duplicate definitions in §3.

**Ladder fitting.** The placement that wins in §4.9 was chosen by knowing the two
prompt lengths in advance. Choosing a ladder from a measured length distribution
is the natural next experiment and is not done.

## Reviewer's closing suggestion, not yet acted on

That the framing device of the submission should be the asymmetry between
measurements and explanations — every headline measurement survived four review
rounds while four of the last five retractions were mechanism claims — with shape
padding as the case study that produced it. Currently that argument sits in §6
behind twelve pages of caveats.
