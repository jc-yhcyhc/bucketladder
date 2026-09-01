# v6e-4 review-hardware capture, 2026-09-01

Real hardware data from a v6e-4 spot podslice (TP=4, `asia-northeast1-b`),
obtained after the v5e capacity hunt repeatedly failed to find a slice. This
is the same tpu-api surface, same resource type, as v5litepod-4 -- just a
different chip generation, requested via `PROVISION_PATH=tpu-api
ACCELERATOR_TYPE=v6e-4`. Not previously exercised by this project.

## What this answers

- **Does the vLLM/tpu-inference stack work on v6e at all?** Yes. Same D2/D3
  ladder mechanism (`Prepared token paddings`, `Prepared attn request
  paddings: [256]`), same RPA kernel, real inference output confirmed via a
  live completion request before this capture was taken.
- **MLSys review priority #2** (n<=2 paid share needs an interval, not one
  boundary): `m1_boundary/...778bde56d335` — n=1 across three boundaries
  (512/1024, 1024/2048, 2048/4096; 4096/8192 exceeds this boot's
  `max_model_len=4096`, see caveat below). COST ratios 1.554 / 1.604 / 1.745
  -- i.e. 55.4% / 60.4% / 74.5% of nominal padding paid, same direction as
  the paper's TPU findings.
- **Concurrency contrast**: `m1_boundary/...6ca57e58ed55` (n=4, n=8, default
  config) -- paid share falls to 6.9% at n=4 (512/1024), and n=8 is
  correctly flagged unmeasurable (every dispatch split under chunked
  prefill) rather than reporting a misleading number.
- **MLSys review priority #8** (LENS at n=8/n=16): `m5_lens_form/...`. Fixed
  cost varies by both bucket and batch size (x7.92 spread across 9 clean
  cells) -- "the crossover is not a single threshold." Holdout MAPE 0.26%
  mean, 0.96% worst -- tighter than LENS's own reported 2.15% on NPUs.
- **Second review's step-vs-request reconciliation question** (does
  per-step padded-token overhead stay flat/rise while per-request paid
  share falls to ~0?): `vllm_warmup.log`'s `BUCKETLADDER_STEP` lines (1757
  of them, parsed by `scripts/e15_step_reconcile.py`) show per-step overhead
  in the 33-59% range across n=1 to n=15, not falling toward zero the way
  §4.2's per-request measure does -- supports the paper's own Appendix
  §10.1 reconciliation (padding migrates to the packed step).

## Caveat: max_model_len mismatch

`configs/e14_n1_all_boundaries.json`'s own `controlled.max_model_len` reads
8192 (copied from a v5e-oriented config), but this boot was launched via
`infra/boot_and_poll.sh`, which hardcodes `MAX_MODEL_LEN=4096` regardless of
what any config file declares -- confirmed from the server's own reported
config in `vllm_warmup.log` (`'max_model_len': 4096`). This was not caught
by `assert_controlled_vars`, which validates a config's own internal
consistency, not the config against the live server's actual reported
state (a live-server check exists elsewhere in this project's method but
was not exercised in this path). Consequence: the 4096/8192 boundary
genuinely could not be tested (prompts near 4104 tokens exceed
`max_model_len=4096`), which is exactly the "splits 0/0, cost nan" result
recorded -- consistent with reality, not corrupted. The three boundaries
that were tested (512/1024, 1024/2048, 2048/4096) are unaffected by this
mismatch.

## Infrastructure bugs found and fixed getting here

All committed on `main`, all with self-contained explanations in their own
commit messages and inline comments:

1. `infra/hunt_v6e.sh` / `hunt_v6e4.sh`: `config.env`'s own
   `: "${VAR:=default}"` lines pre-empt a caller script's later fallback
   logic for the same variable -- caused a real v6e-4 slice to be created
   under the wrong (shared, default) name, which a concurrently-running v5e
   hunt then mistook for its own successful creation and ran `deploy.sh` +
   an experiment script against (both failed fast; nothing was corrupted).
2. `infra/provision_first_available.sh`: hardened `exists()` to verify
   `acceleratorType`, not just that a same-named resource exists, closing
   the collision class in (1) at the source.
3. Deadmen do not extend or replace each other -- arming a second, longer
   one does not cancel an earlier, shorter one already counting down. Cost
   real progress once (a landed slice was torn down by its own creation-time
   deadman mid-experiment). Lesson: arm once, with the real intended
   duration, not incrementally.
4. `infra/boot_and_poll.sh`'s cleanup step used `pkill -9 -f "vllm serve"`
   inline over `ssh --command=`, whose own invoking shell's argv contains
   that literal text -- pkill matched and killed its own parent process,
   severing the SSH session, on every invocation, independent of whether a
   real target existed. Fixed with the standard bracket-escape
   (`"[v]llm serve"`). Also fixed the same pattern in `scripts/o4_boot_cliff.sh`.
5. `infra/boot_and_poll.sh` never forwarded `BUCKETLADDER_LOG_STEP_SHAPES`
   (set locally by the caller) into the remote launch command -- the
   step-logger patch was applied to the source but its runtime gate was
   never set on the actual server process, so no step logs would ever have
   been produced by any review-arms run, on any hardware, silently, until
   this was the first time the path was exercised end-to-end.
