# bucketladder

Research artifact for **Bucket-Aware Admission Control for Ragged Workloads on
Compiled-Shape Accelerators** (target: MLSys 2027 industrial track — deadline
unverified, see `notes/plan_v3.md` open items).

Serving stacks on compiled-shape accelerators pad every request up to one of N
precompiled bucket shapes. When a request arrives and its bucket is saturated, the
scheduler faces a choice nobody has studied: **promote it into a larger bucket and
pay the padding, or queue it and pay the wait.** This project's claim is that the
decision is workload-dependent, that the right policy is measurably better than
either fixed strategy, and that the gain is large enough to matter in dollars.

## Status: pre-gate. No results, no scaffolding, deliberately.

This repo contains records only — the execution plan, the two drafts it supersedes,
and the design review that produced it. There is no `scripts/`, no `sim/`, no
`configs/`, and that is on purpose.

The W0 prior-art gate runs **before** anything gets built. Two prior projects by this
author were killed at exactly this stage (`gapcache` on Aug 2, on its own prior-art
gate; the determinism framing of this work on Aug 7). Writing code before the gate
passes is how you end up with a scaffold for a project that shouldn't exist. See
`notes/plan_v3.md` → "W0 — gate, blocking", and the pre-committed response if it
fires.

## What's here

```
notes/
├── plan_v3.md                     the execution plan — start here
├── design_review_v2.md            disk-verified review of v2; ten findings, why v3 differs
├── plan_v2_assessment.md          superseded — the cost reframe that named admission as the spine
└── plan_v1_determinism_review.md  retired direction; source of the controlled-variables contract
```

Still to be written, in W0, before any code: `notes/prior_art.md` and
`notes/kill_condition.md`.

## Relationship to infersim

This author has a separate, unrelated research artifact,
[infersim](https://github.com/jc-yhcyhc/infersim), for a different paper
(fleet-level machine warmup/cold-start capacity control). The research problems don't
overlap — bucketladder is about admission and shape quantization *within* a running
server, infersim is about when to turn machines on.

What is borrowed from infersim is purely **engineering convention** — discrete-event
simulator shape, policy-as-ABC-with-hooks, a canonical result-schema dataclass,
matched-trace evaluation with paired bootstrap, one script per finding, MAPE < 15%
calibration acceptance. No research content is shared between the two repos.
`notes/plan_v3.md` splits the reuse explicitly into what works unchanged, what needs
porting, and what has to be built from scratch — infersim's calibration targets
JetStream/MaxText, this project targets vLLM + `tpu-inference`, and the difference
matters.
