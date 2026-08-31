#!/usr/bin/env python3
"""Bootstrap CIs for the GPU control at n=20 repeats/cell.

MLSys review: "An L4 is ~$0.30/hr; run 20 repeats per cell and report the
interval." g1_cuda_graph.py's own --report prints only point estimates
(ms_per_step, and the position-of-n=9 statistic) even when the underlying
--out files carry 20 raw samples per cell. This computes the interval those
samples support, reusing _stats.bootstrap_ci and the same paired-resample
convention already used throughout this project rather than inventing a new
one.

Two intervals per arm:
  * per-cell ms_per_step, bootstrapped over the 20 samples directly
  * the position-of-n=9 statistic (0% = costs what n=8 costs, i.e. free;
    100% = costs what n=16 costs, i.e. fully paid), bootstrapped by
    resampling n=8/9/16 independently and recomputing the ratio each time --
    same shape as paid_share_ci's independent-arm resampling.

Usage:
  python scripts/g1_analyze_repeats.py results/g1_repeats/g1_graphs_r20.json \\
                                       results/g1_repeats/g1_eager_r20.json
  python scripts/g1_analyze_repeats.py --self-test
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stats import bootstrap_ci  # noqa: E402


def cell_ci(step_ms: list[float], n: int = 10000, alpha: float = 0.05,
           seed: int = 42) -> tuple[float, float, float]:
    """(point, ci_lo, ci_hi) for one cell's ms/step, from its raw samples.

    Median, not mean: the same statistic g1_cuda_graph.py's own measure()
    reports per cell (med = statistics.median(ts)), and the convention §2
    states for every other reported quantity in this project.
    """
    if len(step_ms) < 2:
        return (statistics.median(step_ms) if step_ms else float("nan"),
                float("nan"), float("nan"))
    rng = random.Random(seed)
    point = statistics.median(step_ms)
    meds = sorted(statistics.median(rng.choices(step_ms, k=len(step_ms))) for _ in range(n))
    return point, meds[int(alpha / 2 * n)], meds[int((1 - alpha / 2) * n)]


def position_point(v8: list[float], v9: list[float], v16: list[float]) -> float:
    m8, m9, m16 = statistics.median(v8), statistics.median(v9), statistics.median(v16)
    if m16 == m8:
        return float("nan")
    return (m9 - m8) / (m16 - m8) * 100.0


def position_ci(v8: list[float], v9: list[float], v16: list[float],
                n: int = 10000, alpha: float = 0.05, seed: int = 42
                ) -> tuple[float, float, float]:
    """(point, ci_lo, ci_hi) for the position-of-n=9 statistic.

    Independent resampling of the three cells, same pattern as
    _stats.flatness_ci and _stats.paid_share_ci: they are separate
    measurements, not paired observations.
    """
    point = position_point(v8, v9, v16)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        r8 = [v8[rng.randint(0, len(v8) - 1)] for _ in range(len(v8))]
        r9 = [v9[rng.randint(0, len(v9) - 1)] for _ in range(len(v9))]
        r16 = [v16[rng.randint(0, len(v16) - 1)] for _ in range(len(v16))]
        p = position_point(r8, r9, r16)
        if p == p:
            vals.append(p)
    if not vals:
        return point, float("nan"), float("nan")
    vals.sort()
    return point, vals[int(alpha / 2 * len(vals))], vals[int((1 - alpha / 2) * len(vals))]


def load_cells(path: Path) -> tuple[dict[int, list[float]], str, int]:
    d = json.loads(path.read_text())
    return ({c["n"]: c["samples"] for c in d["cells"]}, d.get("arm", path.stem),
            d.get("output_len", 64))


def analyse(path: Path) -> dict:
    cells, arm, output_len = load_cells(path)
    out = {"arm": arm, "cells": {}, "position": None}
    step_cells: dict[int, list[float]] = {}
    for n, samples in sorted(cells.items()):
        # 'samples' are raw per-dispatch wall_ms; ms_per_step divides by
        # output_len (decode steps), matching measure()'s own
        # med / output_len -- NOT by n, which was this script's first bug.
        step_ms = [s / output_len for s in samples]
        step_cells[n] = step_ms
        pt, lo, hi = cell_ci(step_ms)
        out["cells"][n] = {"point": pt, "ci_lo": lo, "ci_hi": hi, "n_samples": len(samples)}
    if {8, 9, 16} <= set(step_cells):
        # A ratio of differences is scale-invariant, so step_cells vs raw
        # wall_ms would give the identical answer -- using step_cells anyway
        # for consistency with the per-cell figures above.
        pt, lo, hi = position_ci(step_cells[8], step_cells[9], step_cells[16])
        out["position"] = {"point": pt, "ci_lo": lo, "ci_hi": hi}
    return out


def render(results: dict[str, dict]) -> str:
    L = ["", "GPU control, 20 repeats/cell, with bootstrap 95% CIs", "-" * 62]
    for path, d in results.items():
        L.append(f"\n  arm={d['arm']}")
        for n, c in d["cells"].items():
            if c["point"] != c["point"]:
                L.append(f"    n={n:<3} no estimate")
            else:
                L.append(f"    n={n:<3} {c['point']:7.3f} ms/step  "
                        f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]  ({c['n_samples']} reps)")
        p = d["position"]
        if p and p["point"] == p["point"]:
            L.append(f"    position of n=9: {p['point']:5.1f}%  "
                    f"[{p['ci_lo']:.1f}%, {p['ci_hi']:.1f}%]")
            excludes_100 = p["ci_hi"] < 100.0
            excludes_0 = p["ci_lo"] > 0.0
            L.append(f"      excludes fully-paid (100%): {excludes_100}   "
                    f"excludes fully-free (0%): {excludes_0}")
    return "\n".join(L) + "\n"


def _self_test() -> int:
    # position=0% means "n=9 costs what n=8 costs" (free padding), not that
    # cost scales with n -- v9 equal to v8 in expectation, v16 merely
    # different from both so the denominator isn't zero.
    v8 = [80.0 + (i % 3) for i in range(20)]
    v9 = [80.0 + (i % 3) for i in range(20)]
    v16 = [160.0 + (i % 3) for i in range(20)]
    pt, lo, hi = position_ci(v8, v9, v16)
    assert abs(pt) < 5, pt
    assert hi < 100.0, (lo, hi)

    # A single sample per cell must not fabricate a confident interval.
    pt2, lo2, hi2 = cell_ci([42.0])
    assert pt2 == 42.0
    assert lo2 != lo2 and hi2 != hi2
    print("self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.paths:
        ap.error("give at least one g1_cuda_graph.py --out file")

    results = {str(p): analyse(p) for p in args.paths}
    print(json.dumps(results, indent=2) if args.json else render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
