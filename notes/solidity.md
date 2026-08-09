# What "solid" means here — the bar every result must clear

Written 2026-08-09, after the decision to stop pacing the work against an
unverified MLSys deadline and optimise for correctness instead. Deadline
pressure is what produces results that look fine and do not replicate; removing
it means the standard below is affordable, so it is now mandatory.

This file is the checklist a number must pass **before it is allowed into the
paper**. It is deliberately stricter than what session 2 produced, and the first
section is an honest audit of where session 2 falls short of it.

---

## The five requirements

**R1 — Measured in the units the paper claims.**
The thesis is about **cost**: TPU-seconds and dollars. Client-observed
wall-clock is a *proxy* containing network RTT, tokenizer time, HTTP handling
and scheduler queueing. A proxy is fine for a kill check; it is not fine for a
headline number. Any number that appears in a cost claim must either be measured
server-side or have its proxy overhead measured and subtracted.

**R2 — Reproducible across a server restart.**
Every multi-session comparison silently assumes this. XLA recompiles, the
compile cache repopulates, HBM fragments differently. Within-run CV is the wrong
denominator for a claim built from two different days. **Effects must exceed the
across-restart floor, not the within-run floor.**

**R3 — Not an artifact of one model.**
A result from a single architecture is a result about that architecture.
Anything structural (a staircase, a gradient with shape size) must appear in at
least two models with different attention configurations, or be explicitly
scoped in the paper to the one it was measured on.

**R4 — Effect reported with an interval, not a point.**
Median plus a paired-bootstrap CI, using `bootstrap_ci` / `bootstrap_p` from
`infersim/extract_paper_numbers.py`. "9.0%" alone is not a result.

**R5 — The mechanism is identified, not just the correlation.**
If cost rises with true length inside a fixed bucket, *say why*. Ragged
attention doing real work? Memory traffic? Without a mechanism the finding is
fragile and a reviewer will supply a worse explanation than the true one.

---

## Audit: how session 2's e01 measures against this

The headline (median flatness 0.97, padding is fully paid) is **statistically
very clean within its run** — within-cell CV is 0.05–0.88%, and the per-bucket
effects sit 4.9–127 standard deviations from zero. There is no sampling-noise
problem. The problems are elsewhere.

| | Status | Consequence |
|---|---|---|
| **R1 units** | ❌ TTFT in ms, client-side | The strongest claim in the project is measured in the wrong units |
| **R2 restart** | ❌ one restart, `e03 --restart-block 0` only | The 2048/4096 gradient (4.8%, 9.0%) is **unvalidated against restart drift** |
| **R3 models** | ❌ Qwen3-4B only | The gradient may be architecture-specific |
| **R4 intervals** | ❌ medians only | No CI on any reported number |
| **R5 mechanism** | ⚠️ plausible, untested | "RPA recovers padding at large shapes" is inference, not evidence |

**The main claim (flatness ≈ 1.0 at ≤1024) is robust to all of this** — a −0.8%
to 1.5% effect is indistinguishable from a perfect staircase under any plausible
drift, and that is the load-bearing result. It survives.

**The gradient is the fragile part.** 4.8% at 2048 and 9.0% at 4096 are large
against a 1.34% within-run CV but have never been tested against restart-to-
restart variation, which is exactly the noise that a small effect measured once
tends to be made of. It is currently the *more interesting* finding and the
*less trustworthy* one. It must not be written up until R2 is satisfied.

---

## What this changes about how sessions run

1. **`e03` runs at every restart, not once per session.** Three restarts in a
   session gives the across-restart floor. Cheap: seconds of compute, minutes of
   XLA warmup.
2. **Any headline effect is measured twice, in different restarts**, before it
   is believed.
3. **Server-side timing is required** for cost claims. vLLM's metrics endpoint
   or engine logs, not the client's stopwatch — to be identified before the next
   session.
4. **A second model** joins once the first is fully characterised — different
   attention shape, not just a different size.

## What this does *not* change

The gate structure stands. e01 passed on its load-bearing claim and that verdict
holds; the work above sharpens a result rather than reopening a decision. And
cost discipline is unaffected — this standard costs more *sessions*, not more
dollars per session, and spend remains a small fraction of the $1,000 ceiling.
