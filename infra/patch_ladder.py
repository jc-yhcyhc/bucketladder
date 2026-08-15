#!/usr/bin/env python3
"""
Let the compiled token ladder be set explicitly, so a ladder chosen by an
algorithm can actually be compiled and measured.

`VLLM_TPU_BUCKET_PADDING_GAP` expresses one family of ladders: keep doubling
while the doubling step is no larger than the gap, then step linearly. Neither
BucketServe's boundaries nor any other derived ladder is in that family, so
without this patch a head-to-head between ladder-design algorithms can only be
argued offline.

The patch adds one branch to `tpu_inference.runner.utils.get_token_paddings`:
if `BUCKETLADDER_TOKEN_LADDER` is set to a comma-separated list, that list is the
ladder. Everything else is untouched, and with the variable unset the function is
byte-identical to the original.

Applied and reverted by this script so the arm is recorded rather than assumed:
`--apply` prints the patched function for the log, `--revert` restores from the
backup it wrote.

Usage (on the TPU VM):
  python infra/patch_ladder.py --apply
  BUCKETLADDER_TOKEN_LADDER=16,32,64,128,320,512 vllm serve ...
  python infra/patch_ladder.py --revert
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

MARKER = "# BUCKETLADDER_EXPLICIT_LADDER"

PATCH = f'''    {MARKER}
    # An explicit ladder, so a ladder derived by an algorithm can be compiled.
    # With the variable unset this branch does not execute and the function
    # behaves exactly as shipped.
    import os as _bl_os
    _bl_explicit = _bl_os.environ.get("BUCKETLADDER_TOKEN_LADDER", "").strip()
    if _bl_explicit:
        _bl_paddings = sorted({{int(_x) for _x in _bl_explicit.split(",") if _x.strip()}})
        logger.info(f"Prepared token paddings: {{_bl_paddings}}")
        logger.info("BUCKETLADDER: explicit ladder in use, padding_gap ignored")
        return _bl_paddings
'''


def target() -> pathlib.Path:
    import tpu_inference.runner.utils as u          # noqa: PLC0415
    return pathlib.Path(u.__file__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    path = target()
    backup = path.with_suffix(".py.bucketladder-orig")
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
    anchor = ("def get_token_paddings(min_token_size: int, max_token_size: int,\n"
              "                       padding_gap: int) -> list[int]:\n")
    if anchor not in src:
        print("anchor not found; the function signature has changed and this "
              "patch must be re-checked against it", file=sys.stderr)
        return 2
    # Insert after the docstring, before the power-of-two assertion.
    key = "    # assert min_token_size is power of 2\n"
    if key not in src:
        print("insertion point not found", file=sys.stderr)
        return 2
    shutil.copy2(path, backup)
    path.write_text(src.replace(key, PATCH + key, 1))
    print(f"patched {path}\nbackup at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
