# Session 13 — the hardware half of the review response

Written before provisioning, as every session plan has been. The offline half is
already done and committed; this file covers only what needs a TPU, in the order
it should run, with what each step decides.

**Budget:** ~$15 at on-demand $4.80/hr, ~3 hours of billed VM time. Running total
would move from [redacted] to ~$56. Ceiling is $1,000.

**Why on-demand rather than spot:** two of these four are paired A/B comparisons
on one server. A preemption mid-arm does not corrupt a preemptible sweep, but it
does corrupt a paired comparison, and the saving is about $10.

---

## What the offline half already settled, so nobody re-measures it

- **Q3's proposed cause is wrong.** Splitting is not governed by
  `max_num_batched_tokens`: `n4:4096/8192` runs 8192 tokens at `mnbt=8192` and
  never splits; `n8:1024/2048` runs ~1024 tokens and splits half the time.
  Sweeping `mnbt` would have measured nothing. **Do not run that sweep.**
- **M1's roofline needs no profiler.** Achieved bandwidth and MFU are computed
  and committed (`m9_roofline.py`). Only the operator breakdown still needs one.
- **M6/Q4 is answered.** A constant-only predictor matches LENS at n≤2 and beats
  it at n=4. No hardware involved.

---

## Order, and why this order

### 1. `m8_split_barrier` — first, because it can invalidate the other three

~25 min. Paired A/B between the ThreadPoolExecutor launcher every prior
experiment used and a barrier-synchronised one, at n = 4, 8, 16, 32.

```bash
infra/serve_remote.sh start Qwen/Qwen3-4B
python scripts/m8_split_barrier.py --config configs/m8_split_barrier.json \
    --base-url http://localhost:8000
```

This runs first because **if the barrier works, every n>8 number in the paper was
limited by our own launcher**, and the boundary experiments should be re-run
under the new launcher before anything else consumes VM time. If it does not
work, the splitting is the scheduler's and the paper can finally state a
mechanism instead of a symptom.

Decision rule, pre-committed: single-step fraction above 50% at n=16 **and**
n=32 under the barrier, where the threadpool is at or below 50%, counts as fixed.

### 2. `m1_boundary` re-run at n=16 and n=32 — only if step 1 succeeds

~30 min. The paid-padding share in the regime production actually runs in, which
the paper currently declares unmeasurable. Reuses the existing experiment with
the new launcher; no new analysis code.

If step 1 fails, skip this and spend the time on step 4 instead.

### 3. `m10_trace_workload` — the workload objection

~45 min. Four prompt-length distributions at a common Poisson arrival rate, with
TTFT and ITL recorded.

```bash
python scripts/m10_trace_workload.py --config configs/m10_trace_workload.json \
    --base-url http://localhost:8000
```

Decides whether 35.9% is a stack property or a workload property. The abstract
currently states it as the former. Also supplies the TTFT/ITL a serving paper is
expected to report and this one does not.

### 4. `m11` TP ablation — three server boots

~60 min including two extra warmups. TP=4 → TP=2 → TP=1, same workload each time.

```bash
for tp in 4 2 1; do
  infra/serve_remote.sh stop
  TP_SIZE=$tp infra/serve_remote.sh start Qwen/Qwen3-4B
  python scripts/e02_stock_baseline.py --config configs/m11_tp${tp}.json \
      --base-url http://localhost:8000
done
```

Each config carries a **pre-registered prediction** derived from the roofline:
per-chip weight bytes scale as 1/TP, so the curve should keep its shape and scale
its level. Writing the prediction into the config before the measurement is the
point — it is the difference between a test and a post-hoc story.

TP=1 puts ~8.0 GB of weights on one 16 GB chip; `max_model_len` is lowered to
4096 in that arm to keep the server bootable, and that difference is recorded in
its `controlled` block, so the guardrail will require any claim spanning arms to
assert invariance over it. It should not be asserted lightly.

### 5. xprof operator breakdown — last, and explicitly exploratory

Q2 asks which kernel changes at n=4. The roofline cannot see kernel choice, and
this is the one part of M1 that genuinely needs a profiler.

**Stated honestly: I do not know the flag.** `tpu-inference` 0.25.0's profiling
hook has not been read from source, and I am not going to invent an environment
variable and present it as a runbook step. First action is to grep the installed
package for a profiler entry point, then use `jax.profiler.trace` around a
decode loop if vLLM exposes no hook.

Scheduled last because it is the only step that might return nothing, and
because the four before it are the ones that change the paper.

---

## Teardown

`infra/teardown_tpu.sh` at the end, and confirm with `gcloud compute tpus
tpu-vm list`. Every prior session has done this; the one risk to the budget is a
forgotten VM, not the experiments.

---

## What this session cannot fix

- **M4's scaling axis.** A TP ablation on four v5e chips is not a v6e result and
  not a 70B result. The generality limitation stands and belongs in §8 as a
  limitation, not as future work dressed up as coverage.
- **M5's trace.** Four parametric families are not a production trace. The
  outcome is a sensitivity range, and the paper must say "across plausible length
  distributions" rather than implying trace grounding it does not have.
