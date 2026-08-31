#!/usr/bin/env python3
"""
Log the realized (requests, real tokens, padded tokens) triple every step, so
the §4.2-vs-§4.4 reconciliation the review asks for is a measurement instead
of a conjecture.

MLSys review, question 3: "Why can the realized-padded-tokens-per-step counter
you specify in §4.7 not be built to settle §4.4? ... Logging
(selected_shape, num_scheduled_tokens) per step converts your
per-request-vs-per-step reconciliation from a conjecture into a measurement."

§4.2 finds per-request paid share indistinguishable from zero by n=16. §4.4
finds the ladder still worth 3.5-12% at n=16. The paper's own reconciliation --
per-request padding vanishes while per-step padding does not -- was declared
unmeasurable because `iteration_tokens_total` bins on powers of two, coarser
than the ladder spacing. This patch reads the exact pre-bucketing numbers
instead of that histogram.

`tpu_inference.runner.tpu_runner.TPUModelRunner._prepare_input_metadata` computes
both quantities every step already, to size the compiled buffers: real tokens as
`max_num_scheduled_tokens_across_dp`, the compiled shape as
`padded_total_num_scheduled_tokens`. Request count is `self.input_batch.num_reqs`,
read one line earlier in the same function. Nothing here is a new computation --
the patch only logs three numbers the runner had already computed.

Gated behind BUCKETLADDER_LOG_STEP_SHAPES: unset, the function is
byte-identical to shipped. Applied and reverted like patch_ladder.py, so the
arm is recorded rather than assumed.

Usage (on the TPU VM):
  python infra/patch_step_logger.py --apply
  BUCKETLADDER_LOG_STEP_SHAPES=1 vllm serve ... 2> /tmp/vllm_step_log.txt
  grep BUCKETLADDER_STEP /tmp/vllm_step_log.txt | python scripts/e15_step_reconcile.py
  python infra/patch_step_logger.py --revert
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

MARKER = "# BUCKETLADDER_STEP_LOGGER"
ANCHOR = (
    "        padded_total_num_scheduled_tokens = (\n"
    "            padded_num_scheduled_tokens_per_dp_rank * dp_size)\n"
)

# Prefixed BUCKETLADDER_STEP so the analysis script's grep can't confuse this
# with vLLM's own log lines; one field per token so a malformed step (a
# request count of 0, a step that raced the shutdown path) is still parseable
# rather than corrupting the whole log's parse.
PATCH = f'''{ANCHOR}        {MARKER}
        # Logs the three numbers this function already computed, gated so an
        # unset env var leaves the function byte-identical to shipped.
        import os as _bl_os
        if _bl_os.environ.get("BUCKETLADDER_LOG_STEP_SHAPES", "").strip():
            logger.info(
                f"BUCKETLADDER_STEP n_reqs={{num_reqs}} "
                f"real_tokens={{max_num_scheduled_tokens_across_dp}} "
                f"padded_tokens={{padded_total_num_scheduled_tokens}}")
'''


def target() -> pathlib.Path:
    import tpu_inference.runner.tpu_runner as r     # noqa: PLC0415
    return pathlib.Path(r.__file__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    path = target()
    backup = path.with_suffix(".py.bucketladder-steplogger-orig")
    src = path.read_text()

    if args.status:
        print(f"{path}: {'PATCHED' if MARKER in src else 'clean'}")
        return 0

    if args.revert:
        if not backup.exists():
            print("no backup to revert from", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        print(f"reverted {path}")
        return 0

    if MARKER in src:
        print("already patched")
        return 0
    if ANCHOR not in src:
        print("anchor not found; _prepare_input_metadata has changed and this "
              "patch must be re-checked against it", file=sys.stderr)
        return 2
    shutil.copy2(path, backup)
    path.write_text(src.replace(ANCHOR, PATCH, 1))
    print(f"patched {path}\nbackup at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
