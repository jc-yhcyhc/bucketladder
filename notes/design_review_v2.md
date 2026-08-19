# Design review of plan v2 — 2026-08-07

Reviews `plan_v2_assessment.md` (the doc that reframed this work around cost and
named admission policy as the spine). Every claim in that doc touching disk was
checked against `~/infersim`, `~/gapcache`, and this host. Ten findings; four are
consequential. `plan_v3.md` is the result — it supersedes v2 and folds these in.

The strategic calls in v2 hold up and carry forward unchanged: v5e over v6e,
admission as the spine, prior-art gate before code, DP over Lloyd–Max, oracle-gap
probe as the cheap kill check.

---

## Verified against disk

Every file v2's reuse table names exists. That part is real:

`infersim/calibration/{vm_setup.sh, fit_parameters.py, process_calibration.py,
measure_throughput_grid_vllm.py, measure_w1_warmup_vllm.py, measure_holdout_vllm.py,
fitted_params.yaml}`, `infersim/{sim,policies/capacity,eval}/`,
`extract_paper_numbers.py` (`bootstrap_ci` L15, `bootstrap_p` L31),
`reproduce_all.sh`, `gapcache/notes/{kill_condition.md,prior_art.md}`.

Host state confirms v2's bring-up argument: Python 3.11.2, git, gcloud (project
id redacted), 16 cores / 62 GB, **no `/dev/accel*`, no jax, no vllm,
no pyarrow**. `~/DeterministicInfer` is empty. Nothing runs today.

---

## Findings, by consequence

### 1. The budget's biggest line item is missing: TPUs bill for existing, not for computing

v2's thesis is that the source plan hid wall-clock costs. It then commits the same
error one level up.

The table allocates "40 hours, on-demand, W0 bring-up, smoke tests, debugging —
$192." But a TPU VM bills while it *exists*, not while it runs a benchmark. Bring-up
is realistically a week of calendar: gated-repo access stalls, `tpu-inference`
install, chunked-prefill audit, ladder enumeration. If the VM stays up across that
week it is **168 hours, not 40 — about $800 on its own**, which alone blows the
ceiling v2 spends its opening section defending.

Nothing in v2 says to tear the VM down between sessions. It needs to, explicitly:

- Delete the VM at the end of every working session; re-create from a scripted path
  (this is what makes "you provision, I script" workable — the script has to be
  re-runnable, not one-shot).
- Persist state to GCS, never to the VM disk.
- Track **billed VM-hours**, not benchmark-hours, in `DECISIONS.md`. They are
  different numbers and only one of them is charged.
- The v5e-4 on-demand rate is $4.80/hr, so idle time costs the same as measurement
  time. Say so.

This is the one finding that changes the budget rather than the prose.

### 2. `fitted_params.yaml` does not contain the number v2 cites it for

v2: *"Your $4.80/hr on-demand figure in `infersim/calibration/fitted_params.yaml`
is measured rather than guessed."*

The file contains `price_per_chip_hour: 1.2` — a **list price constant**, not a
measurement, and not [redacted] (that's 4 × 1.2, derived in the doc, not read from the
file). What *is* measured in that file is `warmup_time_sec: 57.0`,
`prefill_tokens_per_sec: 13310`, `decode_tokens_per_sec: 1874`.

The effect is that a file full of genuine measurements was used to lend authority to
a price. The whole pricing table is labelled "Verified pricing (Aug 2026)" with no
source, and two of its five rows have `—` in the columns being compared while still
carrying a recommendation. v3 labels these as list prices to verify at provision time.

### 3. The calibration stack doesn't match, and v2 doesn't mention it

`fitted_params.yaml` header: *"Calibrated against GCP v5e-4 serving **Gemma-2 7B
(JetStream/MaxText)**."* `vm_setup.sh` installs **MaxText + JetStream**, downloads
Gemma-2 from HF, converts to orbax. This project is **vLLM + `tpu-inference`**.

What survives the stack change is narrower than v2 implies:

- **Survives:** the ~40 lines of `tpu-ubuntu2204-base` image workarounds in
  `vm_setup.sh`'s early steps (malformed `gcsfuse.list`, `cnf-update-db` post-invoke
  failure, distutils-broken system pip, root-owned `/usr/local`, snap gsutil
  traceback). Genuinely a day saved. v2's reuse-table wording ("TPU VM image
  workarounds") is accurate; its body text ("`vm_setup.sh` already encodes...")
  oversells a script whose back half is for a stack you're not using.
- **Survives:** the `*_vllm.py` measurement scripts. `measure_throughput_grid_vllm.py`
  POSTs to `/v1/completions` with streaming over plain `urllib`, and has a mock mode.
  Server-agnostic as claimed. Strongest reuse in the table.
- **Does not survive:** `fit_parameters.py` fits exactly three parameters against a
  `SIM_DEFAULTS` dict. Bucket-ladder needs a different parameter set — per-bucket cost
  curve `C(B)`, promotion cost, queue-wait. What transfers is the **MAPE < 15%
  accept/reject harness shape** (L13, L213), not the fitter.

### 4. The §0 traceability contract is new code, not inherited convention

v2: *"Keep §0's traceability contract unchanged ... it matches what `infersim`
already does."*

It does not. Grepping `infersim` for `start_run`, `save_table`, `MANIFEST` returns
**zero hits**. infersim writes **CSV** through a dataclass schema (`eval/schema.py`,
`RESULT_FIELDS`), and `reproduce_all.sh` opens with `rm -f` on six result files
because they're append-mode CSVs. That is the *opposite* of the never-overwrite rule
§0 mandates. Parquet needs pyarrow, which isn't installed here.

So `scripts/_common.py` is a **from-scratch build**, and `reproduce_all.sh`
regenerates from *configs*, not from a manifest — a different and lower bar than the
one v2 holds it up as meeting. The contract is still the right design. It's just a
cost v2 books as free.

### 5. v2 argues itself wrong in public and leaves both halves in

It opens with a table and a bolded resolution — *"Resolution: v5e-4 on spot, and move
the sweep into a calibrated simulator"* — built on the claim that money forces
simulation. Two-thirds down: *"Correcting my own framing from earlier in this
document: at spot v5e pricing, your original all-hardware grid comes to roughly
$170–400, so money was never the real objection."*

That retracts the opening argument while the opening argument is still standing. The
real case for simulation is the other two reasons given — **wall-clock** and
**statistical power** (3 hardware repeats → medians with min/max; 30 sim seeds →
matched-trace paired bootstrap, which is what let you write `p < 0.001` in infersim).
Both are strong and neither needs the budget claim. v3 deletes the money argument and
leads with power.

Related, smaller: the "2.5× over budget" figure attacks `v6e-8`, which appears in the
source doc **only as a stray example field in a `meta.json` snippet**. The source doc
never chose hardware. "The doc doesn't specify hardware; here is the choice and why"
is the fair framing and loses nothing.

### 6. §6's DP and the paper's spine are in tension — and v2 doesn't notice

The DP claim is right *as stated*: hard cardinality budget, separable additive cost,
1-D DP over a discretized length axis gives the global optimum in `O(K·N²)`.
Lloyd–Max is strictly worse. Keep that.

But separability is exactly what the admission spine breaks. Once you're deciding
"promote into a larger bucket and eat the padding" vs "queue for the right one," a
request's cost depends on **queue occupancy and what else is co-batched** — not on
its own bucket alone. The DP therefore optimizes the ladder under an open-loop
assumption that the paper's own headline policy violates.

Not fatal, and arguably interesting. But v3 says it out loud: the DP is optimal for
the **characterization** setting (the part v2 demotes to "setup"), and the joint
ladder × policy problem is not separable. Then uses the simulator — free — to measure
how much the open-loop-optimal ladder loses under the closed-loop policy. If it loses
little, that's a clean result. If it loses a lot, that's a *better* result and
probably the actual paper. A reviewer will find this, so find it first.
→ `e31_ladder_policy_joint.py` in v3.

### 7. The baseline that decides whether there's a paper isn't in the policy list

Policies to compare: `queue`, `promote`, `hybrid`, `oracle`. Missing: **what vLLM +
`tpu-inference` actually does today**.

If `hybrid` beats `queue` and `promote` but the deployed default already behaves like
`hybrid`, there is no contribution. Characterizing the stock admission/batching
behaviour belongs in W0 next to ladder enumeration — same server, same logs — and
it's nearly free once the VM is up. It also feeds the "Why this is not already done"
section v2 rightly praises. → `e02_stock_baseline.py` in v3.

### 8. Priority order is by "how wrong is the doc," not "what kills the project soonest"

Items 2 (chunked prefill) and 5 (oracle headroom) are both sub-$20, sub-day, and each
can end the project. They're separated by three items about methodology. In v3 the
cheap kill checks come first and the method critique follows.

### 9. A third week-0 kill needs an answer, and "stop" isn't one

v2: *"A negative on 4 or 5 stops the project in week 0 for ~$200, which is the entire
point of running them first."*

Correct in isolation. But `gapcache` died on its prior-art gate on Aug 2, determinism
died Aug 7, and this is topic three in five days. If the prior-art gate fires again
there is no time for a fourth cold start before the target deadline. v2 names the
pattern ("Both prior projects consumed their slack") and then doesn't price it.

v3 pre-commits, in writing, before W0 starts: if the gate fires, the response is
*narrow the venue* (workshop, or skip the cycle and target the next) — not pivot
again. That decision is much cheaper made now than in week 3.

### 10. Smaller, but fixed in v3

- **Quota.** The v1 review recorded "TPU quota approved but unprovisioned." v2 assumes
  `v5litepod-4` availability without checking the approval is for **v5e specifically,
  in a named zone**. Quota for one generation doesn't grant another. Under "you
  provision, I script," the script's header must state the exact quota + zone
  prerequisite.
- **Path.** v2 says `~/bucketlad`; `~/bucketladder` is what exists.
- **Script naming.** v2's `scripts/e13_promotion_cost.py` reuses the `e13` slot the
  source doc assigned to the ladder sweep, colliding with the `claim_id` indirection
  in `paper_numbers.parquet`. v3 defines a clean namespace.
- **Controlled-variables contract.** v2 says v1's contract "was dropped in the pivot"
  and should come back, but doesn't restate it. It's in
  `plan_v1_determinism_review.md` L82–96. v3 inlines the table rather than pointing
  at a retired determinism doc — under a cost framing, prefix caching silently
  corrupting prompt-token counts matters just as much.
- **Deadline.** The entire W0–W12 schedule keys off ~2026-10-30, which v2's own open
  items admit came from search, not a CFP. v3 puts the backup venue in the schedule.

---

## Re-checking these claims

- `grep -n price_per_chip_hour ~/infersim/calibration/fitted_params.yaml` → `1.2`,
  and the `_calibration_note` naming JetStream/MaxText + Gemma-2 7B (findings 2, 3)
- `grep -rn "start_run\|save_table\|MANIFEST" ~/infersim --include=*.py --include=*.sh`
  → no matches (finding 4)
- `head -20 ~/infersim/reproduce_all.sh` → the `rm -f` block (finding 4)
- `head -20 ~/infersim/calibration/vm_setup.sh` → the "Known quirks of
  tpu-ubuntu2204-base" list, then MaxText below it (finding 3)
- `head -70 ~/infersim/calibration/measure_throughput_grid_vllm.py` →
  `/v1/completions` streaming, mock mode (finding 3)
- `sed -n '82,96p' notes/plan_v1_determinism_review.md` → the controlled-variables
  table (finding 10)

Finding 1 is arithmetic, not disk: 168 h × $4.80/hr = $806 for one week of
un-torn-down v5e-4, against a $1,000 ceiling.
