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
| **R1 units** | ✅ **fixed** — headline from `vllm:request_prefill_time_seconds` | Client TTFT retained alongside as a proxy |
| **R2 restart** | ✅ **satisfied 2026-08-09** — two restarts agree within 0.02; across-restart CV 0.32% | The gradient is real, not drift |
| **R3 models** | ✅ **satisfied** — SmolLM2 replication (granite failed TP=4 sharding) | **The staircase is architecture-dependent**: flatness 0.54 vs 0.81 at 4096 |
| **R4 intervals** | ✅ **fixed** — `_stats.flatness_ci` bootstraps the statistic itself | e01 now prints `0.97 [0.94, 1.01]` and warns when a CI exceeds 0.5 |
| **R5 mechanism** | ✅ **satisfied** — `prefill_kv_computed_tokens` == true length always | RPA *does* skip padding in attention; the cost is the dense path running on the padded shape |

**The main claim (flatness ≈ 1.0 at ≤1024) is robust to all of this** — a −0.8%
to 1.5% effect is indistinguishable from a perfect staircase under any plausible
drift, and that is the load-bearing result. It survives.

~~**The gradient is the fragile part.**~~ **RESOLVED 2026-08-09.** Reproduced
across two restarts to within 0.02 (across-restart CV 0.32%), and independently
in session 2 on a different day with a different instrument. It is now among the
better-supported numbers in the project — and R3 showed it is
*architecture-dependent*, which turned it from a curiosity into the finding that
scopes the paper's central claim.

---

## What this changes about how sessions run

1. **`e03` runs at every restart, not once per session.** Three restarts in a
   session gives the across-restart floor. Cheap: seconds of compute, minutes of
   XLA warmup.
2. **Any headline effect is measured twice, in different restarts**, before it
   is believed.
3. **Server-side timing is required** for cost claims, and the source is now
   identified and wired in: `scripts/_metrics.py` scrapes vLLM's Prometheus
   endpoint and takes `_sum`/`_count` **deltas** around each cell, giving the
   exact mean over precisely the requests we issued.

   ```
   vllm:request_prefill_time_seconds   prefill duration   -> e01's headline
   vllm:request_queue_time_seconds     arrival->scheduled -> resolves e02
   vllm:request_decode_time_seconds
   vllm:e2e_request_latency_seconds
   ```

   `request_queue_time_seconds` is what makes e02 answerable at all: client TTFT
   under concurrency mixes waiting with computing, while queue time and prefill
   time separate exactly those. e01 now reports flatness from server prefill
   time as the headline and keeps client TTFT alongside as a proxy, because a
   divergence between them is itself informative.
4. **A second model** joins once the first is fully characterised — different
   attention shape, not just a different size. Chosen and justified in
   `notes/model_selection.md`: **`ibm-granite/granite-3.1-2b-instruct`**, a
   single-variable change (head_dim 64 vs 128, GQA held at 4:1). Worth noting
   that **every model on `tpu-inference`'s tested list is head_dim=128 with
   GQA**, so replicating inside that list would vary parameter count and nothing
   structural.

## What this does *not* change

The gate structure stands. e01 passed on its load-bearing claim and that verdict
holds; the work above sharpens a result rather than reopening a decision. And
cost discipline is unaffected — this standard costs more *sessions*, not more
dollars per session, and spend remains a small fraction of the $1,000 ceiling.
