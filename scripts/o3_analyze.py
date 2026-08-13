#!/usr/bin/env python3
"""
O3 analysis — locate the concurrency at which the finer ladder stops paying.

§4.9 measured the 21-shape ladder beating the 10-shape ladder at n=2. The
recommendation that follows is conditional on concurrency, so the number a
deployment actually needs is the crossing: below it, compile the finer ladder and
pay in KV capacity; above it, compile as coarse a ladder as the tail tolerates.

**The placebo is valid only at n<=2, established by measurement, not assumed.**
The design intended prompt 300 as a placebo at every concurrency, reasoning that
300 pads to 512 on both ladders. That holds only while each request prefills in
its own dispatch. Scraping vLLM's `iteration_tokens_total` histogram around one
n=16, plen=300 cell (scripts/o3_step_sizes.py) shows a single repeat costing
three prefill steps -- one in (256,512], one in (512,1024], one in (2048,4096] --
alongside 92 decode steps of at most 16 tokens. The scheduler admits requests in
waves and packs them, so the largest packed step lands where the two ladders
differ and the cell is treated, not inert.

The consequence is stated rather than buried: at n>=4 there is no clean placebo,
because every prompt length produces packed steps that can straddle ladder
entries. The offset is still subtracted there, and at n=8 and n=16 it is
negative, which makes the correction conservative -- it shrinks the measured
benefit rather than inflating it. A corrected number at n>=4 is therefore a floor
on the effect rather than an unbiased estimate, and arm order is the only control
those rows carry.

Two other things this does that the O1 analysis did not need to:

1. **Subtract the placebo per concurrency, not once**, wherever it means
   anything. The offset is not assumed constant in n -- at higher concurrency the
   two servers are doing more work and any systematic difference between them has
   more room to grow -- so it is estimated level by level.

2. **Report the crossing as a bracket, not a point.** Concurrency is sampled at
   1, 2, 4, 8, 16, so the crossing is located between two sampled levels. Quoting
   an interpolated value would invent precision the design does not have.

Usage:
  python scripts/o3_analyze.py
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _stats import bootstrap_ci_unpaired, bootstrap_p  # noqa: E402

PLACEBO_LEN = 300          # pads to 512 on both ladders at every concurrency


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=pathlib.Path, default=pathlib.Path("results"))
    args = ap.parse_args(argv)

    try:
        import pandas as pd
    except ImportError:
        print("[o3a] pandas required", file=sys.stderr)
        return 1

    root = args.results_root / "o3_concurrency_crossover"
    files = sorted(root.glob("*/latency_reps.parquet"))
    if not files:
        print(f"[o3a] no latency_reps.parquet under {root}", file=sys.stderr)
        return 1
    reps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def cell(n: int, plen: int, arm: str) -> list[float]:
        return reps[(reps.concurrency == n) & (reps.prompt_len == plen)
                    & (reps.arm == arm)]["e2e_ms"].tolist()

    levels = sorted(reps["concurrency"].unique())
    treated = [p for p in sorted(reps["prompt_len"].unique()) if p != PLACEBO_LEN]

    print("=== Raw difference (gap512 - default), negative = fine ladder faster ===")
    print("NOTE: the prompt-300 column is a valid placebo only at n<=2. Above that "
          "the\n      scheduler packs requests into shared steps that straddle "
          "ladder entries,\n      so it is a treated cell and the correction is a "
          "conservative floor.\n")
    print(f"{'n':>4}{'prompt':>8}{'default':>10}{'gap512':>10}{'raw':>9}"
          f"{'placebo':>9}{'corrected':>11}  {'95% CI (raw)':<22}{'p':>8}")

    corrected: dict[int, dict[int, float]] = {}
    for n in levels:
        pl_d, pl_g = cell(n, PLACEBO_LEN, "default"), cell(n, PLACEBO_LEN, "gap512")
        offset = (statistics.fmean(pl_g) - statistics.fmean(pl_d)) if pl_d and pl_g else 0.0
        corrected[n] = {}
        for plen in treated:
            d, g = cell(n, plen, "default"), cell(n, plen, "gap512")
            if not d or not g:
                continue
            raw = statistics.fmean(g) - statistics.fmean(d)
            lo, hi = bootstrap_ci_unpaired(g, d)
            p = bootstrap_p(g, d)
            corrected[n][plen] = raw - offset
            print(f"{n:>4}{plen:>8}{statistics.fmean(d):>10.1f}{statistics.fmean(g):>10.1f}"
                  f"{raw:>9.1f}{offset:>9.1f}{raw - offset:>11.1f}"
                  f"  {f'[{lo:+.1f}, {hi:+.1f}]':<22}{p:>8.4f}")
        print()

    print("=== Placebo-corrected benefit vs concurrency (ms, negative = fine ladder wins) ===")
    print(f"{'prompt':>8}" + "".join(f"{'n=' + str(n):>10}" for n in levels))
    for plen in treated:
        print(f"{plen:>8}" + "".join(
            f"{corrected[n].get(plen, float('nan')):>10.1f}" for n in levels))

    print("\n=== Crossing ===")
    for plen in treated:
        seq = [(n, corrected[n][plen]) for n in levels if plen in corrected[n]]
        neg = [n for n, v in seq if v < 0]
        nonneg = [n for n, v in seq if v >= 0]
        if not neg:
            print(f"prompt {plen}: no benefit at any sampled concurrency")
        elif not nonneg:
            print(f"prompt {plen}: benefit persists through n={max(neg)} "
                  f"(no crossing within the sampled range)")
        else:
            last_neg = max(neg)
            first_nonneg = min(n for n in nonneg if n > last_neg) if any(
                n > last_neg for n in nonneg) else None
            if first_nonneg is None:
                print(f"prompt {plen}: sign is not monotone in n; report the row, "
                      "not a crossing")
            else:
                print(f"prompt {plen}: crosses zero between n={last_neg} and "
                      f"n={first_nonneg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
