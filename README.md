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

## Status: gate passed, W0b harness written, no hardware provisioned. $0 spent.

The W0 prior-art gate ran **before** any code was written, and it changed the plan
substantially — see `notes/kill_condition.md`. It did not fire, but L1 (prefill-length
bucketing) is dead because chunked prefill is default-on, and LAPS (MLSys 2026) is now
primary related work. `notes/plan_v4.md` is the current plan.

What exists now is the W0b harness, written so it is **fully testable before a TPU
exists**: mock modes, dry-runs, and a 59-test suite that runs on a laptop.

```bash
./run_tests.sh          # everything verifiable without hardware
```

## What's here

```
notes/          the plan, the prior-art gate, the kill-condition verdicts
infra/          create_tpu.sh, teardown_tpu.sh, vm_setup.sh — all support --dry-run
scripts/        _common.py (traceability contract), ladder.py, e00_smoke_test.py
configs/        one JSON per variant, including a deliberately invalid one
tests/          59 tests, no accelerator required
```

## The two rules that cost money if broken

**Tear the VM down at the end of every session.** A TPU VM bills while it *exists*,
not while it computes. `infra/teardown_tpu.sh` is the single most important cost
control here; `--status` tells you whether anything is billing right now. The budget
is 135 billed VM-hours (~$450) against a $1,000 ceiling, and the headroom is almost
exactly one forgotten VM-week.

**Never run with prefix caching in an unrecorded state.** `assert_controlled_vars`
aborts rather than warns, because a warning in a log is not something anyone reads six
weeks later while writing a paper. `configs/e00_BAD_apc_unrecorded.json` exists to be
rejected, and a test asserts that it is.

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
