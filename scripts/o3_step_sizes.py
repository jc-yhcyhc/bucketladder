#!/usr/bin/env python3
"""
What token counts does the stack actually execute for a given workload?

This exists because an assumption about compiled-shape mapping turned out to be
wrong, and no amount of rereading the config would have shown it. O3's design
treated prompt 300 as inert -- it pads to 512 on both ladders -- which is true
only if each request prefills in a dispatch of its own. It does not at higher
concurrency: the scheduler admits requests in waves and packs them, so a step
carries several requests' tokens and its size, not any request's length, picks
the compiled shape.

vLLM exports `vllm:iteration_tokens_total` as a Prometheus histogram of tokens
per engine step. Snapshotting it around a workload gives the step-size
distribution that workload actually produced, which is the ground truth for
whether two ladders can differ on it at all.

The buckets are powers of two and therefore coarse: this measures which RANGE
each step falls in, which is enough to establish that a step lands where the
ladders differ, and not enough to compute padded tokens exactly. It is a
diagnostic, not a padding measurement.

Usage:
  python scripts/o3_step_sizes.py --concurrency 16 --prompt-len 300 --reps 3
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _client import complete  # noqa: E402

BUCKET_RE = re.compile(r'iteration_tokens_total_bucket\{[^}]*le="([^"]+)"[^}]*\}\s+([0-9.]+)')


def snapshot(base_url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as r:
        body = r.read().decode("utf-8", errors="replace")
    return {m.group(1): float(m.group(2)) for m in BUCKET_RE.finditer(body)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv)

    before = snapshot(args.base_url)
    if not before:
        print("[o3s] no iteration_tokens_total buckets at /metrics", file=sys.stderr)
        return 1

    for rep in range(args.reps):
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(
                lambda i: complete(args.base_url, args.model, args.prompt_len,
                                   args.output_len, seed=rep * 100 + i),
                range(args.concurrency)))
    after = snapshot(args.base_url)

    def as_float(k: str) -> float:
        return float("inf") if k.startswith("+") else float(k)

    keys = sorted(after, key=as_float)
    print(f"steps executed for n={args.concurrency}, prompt_len={args.prompt_len}, "
          f"{args.reps} repeats:")
    prev_cum = 0.0
    prev_key = "0"
    total = 0
    for k in keys:
        cum = after[k] - before.get(k, 0.0)
        n = int(cum - prev_cum)
        if n:
            print(f"  ({prev_key}, {k}]  {n:5d} steps"
                  f"{'   <- decode' if as_float(k) <= 16 else ''}")
            total += n
        prev_cum, prev_key = cum, k
    print(f"  total {total} steps over {args.reps} repeats "
          f"({total / max(args.reps, 1):.0f} per repeat)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
