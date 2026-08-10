#!/usr/bin/env python3
"""
I1 — bucket-aligned step packing, patched into vLLM's scheduler behind a flag.

THE MECHANISM. Two components do not know about each other. vLLM's scheduler
packs whatever is runnable into a step, landing on an arbitrary token count.
`tpu-inference`'s runner then rounds that up to the next compiled bucket
(`tpu_runner.py:2133`). A step that assembles 4104 tokens executes the 8192
shape, and M4 measured that roughly a quarter of the difference is genuinely
paid — 24.8% at the 4096→8192 boundary.

THE PATCH. When the step's token total lands just above a bucket boundary, trim
the marginal chunk so the step lands ON the boundary. The trimmed tokens are not
dropped; they stay in the request and are scheduled next step, which under load
would have run anyway. That is why this does not simply pay M3's 6.11 ms
per-step floor an extra time: in steady state it reshapes steps rather than
adding them.

WHY IT IS SAFE TO TRIM. Chunked prefill already splits a request's prefill
across steps, so shortening a prefill chunk is something the scheduler does
constantly. The rule here is deliberately narrower than it could be:

  - only trim requests still in **chunked prefill**, never a decode request.
    Trimming a decode request to zero would remove it from the step entirely,
    which changes which requests produce output and is a far bigger change than
    this experiment needs.
  - never trim below **1 token**, so no request is silently dropped.
  - if no trimmable request exists, **do nothing** and let the step spill.

The flag is read once at import. `BUCKETLADDER_ALIGN=1` enables it, anything
else leaves vLLM byte-for-byte in its default behaviour, so one build serves
both arms of the A/B and the comparison cannot be confounded by a different
install.

CORRECTNESS IS NOT ASSUMED. `--verify` re-runs the same prompts in both modes
and compares generated text. A patch that changes outputs is wrong regardless of
what it does to throughput, and no timing number from this should be believed
until that passes.

Usage (on the VM):
  python3 infra/patch_scheduler.py --apply
  python3 infra/patch_scheduler.py --status
  python3 infra/patch_scheduler.py --revert
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

ANCHOR = "        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())"

BLOCK = '''
        # --- bucketladder I1: bucket-aligned step packing ------------------
        # Enabled only by BUCKETLADDER_ALIGN=1; otherwise this is dead code and
        # the scheduler behaves exactly as shipped, so one build serves both
        # arms of the A/B.
        if _BL_ALIGN and total_num_scheduled_tokens > 0:
            _bl_target = _bl_largest_bucket_le(total_num_scheduled_tokens)
            _bl_excess = total_num_scheduled_tokens - _bl_target
            if _bl_excess > 0:
                # Trim the LAST-scheduled requests first: they are the ones the
                # budget loop admitted most recently, so deferring them is the
                # smallest departure from what the scheduler intended.
                for _bl_rid in reversed(list(num_scheduled_tokens)):
                    if _bl_excess <= 0:
                        break
                    _bl_req = self.requests.get(_bl_rid)
                    if _bl_req is None:
                        continue
                    _bl_alloc = num_scheduled_tokens[_bl_rid]
                    # Still in chunked prefill? Only then is deferring tokens a
                    # no-op for correctness: they are computed next step.
                    _bl_left = _bl_req.num_tokens - (
                        _bl_req.num_computed_tokens + _bl_alloc)
                    if _bl_left <= 0 or _bl_alloc <= 1:
                        continue
                    _bl_cut = min(_bl_excess, _bl_alloc - 1)
                    num_scheduled_tokens[_bl_rid] = _bl_alloc - _bl_cut
                    _bl_excess -= _bl_cut
                total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
'''

HEADER = '''
# --- bucketladder I1 ---------------------------------------------------------
import os as _bl_os

_BL_ALIGN = _bl_os.environ.get("BUCKETLADDER_ALIGN", "") == "1"
# tpu-inference's token ladder: powers of two from 16 up to
# max_num_batched_tokens (runner/utils.py::get_token_paddings with
# VLLM_TPU_BUCKET_PADDING_GAP unset).
_BL_BUCKETS = [16 << i for i in range(0, 20)]


def _bl_largest_bucket_le(n: int) -> int:
    """Largest compiled bucket <= n; n itself if it is below the smallest."""
    best = 0
    for b in _BL_BUCKETS:
        if b <= n:
            best = b
        else:
            break
    return best or n
# -----------------------------------------------------------------------------
'''


def target() -> pathlib.Path:
    import vllm  # noqa: PLC0415
    return pathlib.Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    path = target()
    backup = path.with_suffix(".py.bl-orig")
    src = path.read_text()
    patched = "_BL_ALIGN" in src

    if args.status:
        print(f"[patch] {path}")
        print(f"[patch] patched: {patched}   backup present: {backup.exists()}")
        return 0

    if args.revert:
        if not backup.exists():
            print("[patch] no backup; nothing to revert", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        print("[patch] reverted to the shipped scheduler")
        return 0

    if patched:
        print("[patch] already patched; nothing to do")
        return 0
    if src.count(ANCHOR) != 1:
        print(f"[patch] anchor found {src.count(ANCHOR)} times, expected exactly 1 — "
              "vLLM's scheduler has changed and the patch must be re-derived",
              file=sys.stderr)
        return 1

    shutil.copy2(path, backup)
    # Header goes after the last top-level import block; appending after the
    # module docstring is fragile, so anchor on the first 'class ' instead.
    idx = src.index("\nclass ")
    out = src[:idx] + "\n" + HEADER + src[idx:]
    out = out.replace(ANCHOR, ANCHOR + "\n" + BLOCK, 1)
    path.write_text(out)
    print(f"[patch] applied to {path}")
    print(f"[patch] backup at {backup}")
    print("[patch] inert unless BUCKETLADDER_ALIGN=1 is set for the server")
    return 0


if __name__ == "__main__":
    sys.exit(main())
