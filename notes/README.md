# notes/

Research records: prior-art research, kill-condition verdicts, design reviews, and
the planning documents the project is being built against. Prose only — no code, no
results. Following the `gapcache` convention, where the prior-art verdict was written
before the build and ended up being the most valuable file in the repo.

## Here now

| File | What it is |
|---|---|
| `plan_v4.md` | **The execution plan. Start here.** Standalone. Written after the W0 gate closed. |
| `solidity.md` | **The bar every result must clear before it enters the paper.** Five requirements, plus an honest audit of where session 2 falls short of them. |
| `session_plan.md` | **Hardware runbook.** Session sizing, the two independent gates, compile budget, contingency, and the start/end checklist. Read with plan_v4. |
| `plan_v3.md` | Superseded. Written before the gate ran; its three-level story and staircase experiment did not survive it. |
| `design_review_v2.md` | Disk-verified review of v2. Ten findings, four consequential. Explains why v3 differs from v2 rather than just asserting it. |
| `plan_v2_assessment.md` | Superseded. The cost reframe that named admission control as the spine and killed the determinism framing. |
| `plan_v1_determinism_review.md` | Retired direction. Kept because L82–96 holds the controlled-variables contract, which v3 inlines and which matters more under a cost framing, not less. |

## The gate, now closed

| File | What it contains |
|---|---|
| `prior_art.md` | The search, in full: shape polymorphism (does not remove the ladder), chunked prefill (kills L1), RPA v3, BucketServe (read in full), **LAPS/PLA-Serve (read in full — primary related work)**, LENS, Multi-Bin Batching, Vidur, and compiled-shape bucketing outside LLMs. Includes both forward-citation sweeps. |
| `kill_condition.md` | The verdict, in `gapcache`'s format. Carries all three verdicts written on 2026-08-07 in sequence, including the two that were wrong and why.

`gapcache`'s prior-art pass surfaced five systems its reading list never anticipated
and ended that project in a week. This gate repeated the lesson: keyword search over
the plan's own reading list found nothing, and the first forward-citation query
surfaced LAPS. **The reading list is not the search.**
