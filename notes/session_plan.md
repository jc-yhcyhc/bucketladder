# Hardware session runbook — v2

Supersedes the four-session sketch. Revised after review; the substantive
changes are recorded at the bottom under "What changed and why".

**The correction that drove this rewrite: money is not the binding constraint.**
Hardware spend for the first block remains well under the $1,000 ceiling. The
scarce resource is *calendar* — twelve weeks — and human attention. Provisioning
and tearing down four separate times costs four setup cycles of context-switching
to save ~$60. That is a bad trade. Sessions are now sized to the work, and dollar
figures below are noise unless a line runs to hundreds.

---

## Session 1 — capture and leave (~3 h on-demand, ~$15)

The only job is to get a **real warmup log off the machine**. Everything likely to
be wrong in the harness is a string-format assumption, and none of it needs a
running TPU to fix.

1. You: `./infra/create_tpu.sh --check`, then `./infra/create_tpu.sh`
2. `infra/vm_setup.sh` → server up, log at `/tmp/vllm_warmup.log`
3. `gcloud compute tpus tpu-vm scp` the log **off the box**
4. Repeat once with `VLLM_TPU_BUCKET_PADDING_GAP=512` — two logs, two ladders
5. `./infra/teardown_tpu.sh`

Then, offline at $0: fix `parse_warmup_log` and `parse_server_config` against the
real thing via `--warmup-log`, until `e00` passes on both captured logs.

**Do not proceed to session 2 until e00 passes offline on a real log.**

## Session 2 — READY TO RUN. Scripts written and mock-tested (2026-08-09).

`e01_oracle_gap.py`, `e02_stock_baseline.py`, `e03_noise_floor.py` all exist,
run in `--mock`, and — the part that matters — each has a mock for **both**
hypotheses, with tests asserting the analysis separates them. A mock that only
ever produces the hoped-for answer proves nothing.

Order on the day: **e03 first** (sets threshold units), then e00, then e01, e02.

Two things session 1 changed here:

- **e01 must run on the DEFAULT (power-of-two) ladder.** Achievable occupancy is
  bounded by the ladder itself: bucket B spans (B/2, B], so occupancy varies over
  at most 2x, and 0.5B/0.25B fall into *smaller* buckets — a different executable.
  Gap ladders are far worse (gap=512 at B=4096 spans only 0.875–1.0).
- **e02 now targets the `request paddings` ladder** `[8,16,32,64,128,256]`
  discovered on hardware, not the token ladder. That is the axis where
  promote-vs-wait actually bites, and `VLLM_TPU_BUCKET_PADDING_GAP` does not
  move it.

## Session 2 — gate, noise floor, and both kill checks (~10 h on-demand, ~$48)

Sessions 2 and 3 from the original sketch, merged. One setup cycle, not two.

**2a. Noise floor first — nothing downstream is interpretable without it.**
Three repeats of one identical config, back to back, VM already up. Produces the
run-to-run coefficient of variation on the primary metric. *Every threshold below
is expressed as a multiple of this number.* A "5% gap" is meaningless until we
know whether repeat noise is 1% or 8%. Cost: minutes.

**2b. e00 gate for real.** Ladder enumeration confirmed against prediction;
controlled-variable audit against vLLM's own engine-config line. Sweep
`VLLM_TPU_BUCKET_PADDING_GAP ∈ {default, 512, 256}` and confirm the ladder
actually changes as `ladder.py` predicts.

Compile budget for that sweep, from `sweep_compile_budget`: **1.0–4.6 h of pure
warmup**, which is a third to a half of this session. Budgeted, not overhead.

**2c. e01 — marginal cost of padding.** Redesigned (see below).

**2d. e02 — stock admission behaviour.** What `tpu-inference` actually does when a
bucket saturates.

## The two gates are independent, and so are the responses

The original sketch ran these together, which was wrong: they kill *different
contributions*, and either can survive the other's failure.

| Gate | Measures | If it fails | What survives |
|---|---|---|---|
| **e01** | marginal TPU-seconds of serving length `L` in bucket `B` vs the tightest fitting bucket | padding is not measurably expensive → **the ladder contribution dies** | the admission policy, if e02 passes. Paper becomes "when to promote", with the ladder fixed at default. |
| **e02** | does stock behave like our `hybrid` policy already? | stock already does it → **the admission contribution dies** | the ladder optimisation, if e01 passes. Paper becomes "cardinality-budgeted ladder design", answering BucketServe's and LAPS's stated open concerns. |

**Both fail → the pre-committed response applies** (`DECISIONS.md`: narrow the
venue or skip the cycle, not a fourth pivot). **Either alone fails → the paper
narrows to the survivor and continues.**

Thresholds are set *after* 2a, as multiples of measured noise — not picked now.
The placeholder "5%" from the earlier sketch was not executable and is withdrawn.

## e01, redesigned as a marginal-cost measurement

The earlier design — compile an exact-shape ladder over trace lengths — needed
hundreds of buckets at 30–120 s each. Hours of compilation to measure a bound.

Better: **hold one bucket, vary occupancy, 3–5 shapes.** For a fixed bucket `B`,
serve requests of length `L ∈ {B, 0.9B, 0.75B, 0.5B, 0.25B}` and measure
TPU-seconds per request. The slope is the marginal cost of padding, directly.

Three advantages:

1. **~5 compiles instead of hundreds.** Fits in a session.
2. It tests **the assumption the entire project rests on** — that padding costs
   wall-clock at all. If serving `0.25B` in bucket `B` costs the same as serving
   `B`, then padding is free on this hardware and *both* contributions die. That
   is the single highest-value measurement available and the old design buried it.
3. It yields `C(B)` directly rather than via an oracle-gap subtraction.

## Sessions 3+ — primitives (spot)

`e11_promotion_cost`, `e12_queue_wait`, `e13_padding_decomposition`.

| | Hours |
|---|---|
| Measurement | 40 |
| **Compile/warmup** (was invisible) | **+6** |
| **Contingency at 40%** (was absent) | **+18** |
| **Total** | **64 spot-hours ≈ $90** |

Contingency is not padding: it is the redo budget for a bad flag, a wrong trace,
or a confound discovered mid-run. Without it the first redo eats the primitives.

**The 12-ladder × 2-model sweep from plan v3 is cancelled.** `sweep_compile_budget`
puts it at **12.5–54.1 h of pure warmup across 24 bring-ups** — up to 135% of the
entire measurement allocation, before a single request is served. Replaced by
4 ladders × 1 model (1.0–4.6 h). The rest of the ladder space is swept in
simulation, which is what the simulator is for.

## Beyond W3 — deliberately unbudgeted

This runbook covers roughly weeks 0–3 of a twelve-week project. **That is a
choice, not an omission:** committing hardware budget past the gates would
pre-commit to a paper the gates might have narrowed or killed.

Rough shape of what remains, to be costed at the gate, not now:

| Weeks | Work | Hardware |
|---|---|---|
| 4–6 | simulator, policy sweep, ladder DP | none — free |
| 7–8 | holdout validation, MAPE < 15% per parameter | ~25 h on-demand |
| 9–12 | analysis, figures, writing | none |

**Analysis and writing get calendar, not just leftovers.** Hardware hours produce
Parquet files; a paper needs sections. Weeks 9–12 are write-up, intro and related
work drafted during 4–6 while the sweep runs, and the ~2 weeks of slack are
protected. Both prior projects consumed theirs.

## Every session, without exception

**Start:**
```bash
python scripts/_common.py --mark-interrupted results/   # flag runs the last VM took with it
./infra/create_tpu.sh --check                            # live quota/zone/version check
```

**End:**
```bash
gsutil -m rsync -r results/ "$GCS_BUCKET/results/"
./infra/teardown_tpu.sh
./infra/teardown_tpu.sh --status                         # must say nothing is billing
# then record billed VM-hours in DECISIONS.md
```

Billed VM-hours, not benchmark-hours. The headroom to the ceiling is about one
forgotten VM-week.

---

## What changed and why

1. **Sessions 2 and 3 merged.** Optimising dollars at the expense of calendar was
   backwards at a 12-week deadline.
2. **Noise floor added, before any threshold.** Thresholds are now multiples of
   measured run-to-run variance. The old "5%" was not executable.
3. **The two gates split**, with separate consequences. They kill different
   contributions and either can survive alone.
4. **e01 redesigned** to a 5-shape marginal-cost measurement, which fits in a
   session and tests the load-bearing assumption directly.
5. **Compile time is a line item**, computed by `sweep_compile_budget`. It
   cancelled the 12-ladder sweep outright.
6. **40% contingency** on the primitives block.
7. **Post-W3 explicitly deferred**, with the reason stated, plus calendar for
   analysis and writing.
8. **Per-run atomicity implemented**, not just planned: `save_table` writes
   through a temp file and renames; `MANIFEST.jsonl` is appended only by
   `finish_run`, so a run with no manifest entry never finished regardless of
   what files it left; `mark_interrupted_runs` flags abandoned runs as
   `preempted` at session start; `usable_runs` is the only function analysis
   code may use to find results.
