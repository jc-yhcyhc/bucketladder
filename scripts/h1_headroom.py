#!/usr/bin/env python3
"""
H1 — how much padding is executed, and does any of it cost anything?

Two questions, both answerable from runs already captured, at zero hardware cost.

CEILING. `vllm:iteration_tokens_total`'s histogram buckets are powers of two and
so is the compiled token ladder, so a step's reporting bucket edge IS the size it
executed at. Summing edges over a dispatch gives padded tokens; the workload
gives real tokens exactly. The ratio bounds anything a packing optimisation
could ever recover.

WHETHER IT IS PAID — a natural experiment nobody designed. At fixed batch size
the real token count is constant, but the scheduler chunks differently from one
dispatch to the next, so the SAME real work executes at padded totals differing
by up to 2x. That is precisely the comparison a designed boundary-straddling
experiment sets out to construct, and it is already in the e05 capture across
150 dispatches — with real work held exactly constant rather than to 1.6%.

    if cost tracks PADDED tokens -> cost rises with the padding ratio
    if cost tracks REAL tokens   -> cost is flat while padding doubles

The second is what the data shows, at five of six batch sizes.

CAVEAT, stated because it is the weakness of any natural experiment: the
scheduler chose the splits, so a lurking variable correlated with both the split
and the cost cannot be excluded the way randomisation would exclude it. The one
cell that does not fit (n=12) is also the cell with the unexplained bimodality.

Usage:
  python scripts/h1_headroom.py
  python scripts/h1_headroom.py --results-root results
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import pathlib
import statistics
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pyarrow.parquet as pq  # noqa: E402

from _common import finish_run, save_table, start_run  # noqa: E402


def load(pattern: str) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    out = []
    for d in sorted(glob.glob(pattern)):
        f = pathlib.Path(d) / "dispatches.parquet"
        m = pathlib.Path(d) / "meta.json"
        if not (f.exists() and m.exists()):
            continue
        rows = pq.read_table(f).to_pylist()
        if not rows or "hist" not in rows[0]:
            continue
        out.append((json.loads(m.read_text())["config"], rows))
    return out


def padded_tokens(hist_str: str) -> int:
    """Sum of executed step sizes. The bucket edge is the executed size."""
    h = ast.literal_eval(hist_str)
    return int(sum(float(k) * v for k, v in h.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-glob", default="captured/*/results/e05_step_shape/*")
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    runs = load(args.capture_glob)
    if not runs:
        print(f"[h1] no captured runs with per-step histograms at {args.capture_glob}",
              file=sys.stderr)
        return 1

    # Inherit the analysed runs' controlled block, then state any variable the
    # contract has gained since the data was taken. The captured e05 runs predate
    # ATTN_BUCKETIZED_NUM_REQS joining CONTROLLED_VARS, so their configs cannot
    # name it -- and the contract correctly refuses to run rather than assume.
    # Every run before session 7 used the shipped default, which is False; that
    # is asserted here rather than defaulted silently, because the whole point of
    # the contract is that provenance is stated.
    controlled = dict(runs[0][0].get("controlled", {}))
    controlled.setdefault("ATTN_BUCKETIZED_NUM_REQS", False)
    cfg = {"experiment": "h1_headroom", "source_glob": args.capture_glob,
           "n_runs": len(runs), "mode": "offline", "controlled": controlled,
           "model": runs[0][0].get("model"),
           "note_controlled": ("ATTN_BUCKETIZED_NUM_REQS is not in the source runs' "
                               "configs because it predates them; False is the shipped "
                               "default and is what those runs used.")}
    run = start_run("h1_headroom", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        steps: list[dict[str, Any]] = []
        paid: list[dict[str, Any]] = []
        tot_real = tot_pad = 0

        for conf, rows in runs:
            plen, olen = conf["prompt_len"], conf["output_len"]
            trace = f"{conf['model']}@prompt{plen}"
            by_n: dict[int, list[dict[str, Any]]] = {}
            for i, r in enumerate(rows):
                if r["prefill_ms"] != r["prefill_ms"]:
                    continue
                real = r["n"] * (plen + olen)
                pad = padded_tokens(r["hist"])
                tot_real += real
                tot_pad += pad
                steps.append({"trace": trace, "step_idx": i, "n": r["n"],
                              "tokens_real": real, "tokens_padded": pad,
                              "padding_tokens": pad - real,
                              "padding_ratio": pad / real - 1.0,
                              "n_steps": r.get("n_steps", float("nan")),
                              "cost_ms": r["prefill_ms"]})
                by_n.setdefault(r["n"], []).append({"pad": pad, "cost": r["prefill_ms"],
                                                    "real": real})

            # The natural experiment: fixed real work, varying padding.
            for n, rs in sorted(by_n.items()):
                g: dict[int, list[float]] = {}
                for x in rs:
                    g.setdefault(x["pad"], []).append(x["cost"])
                usable = {p: c for p, c in g.items() if len(c) >= 3}
                if len(usable) < 2:
                    continue
                lo, hi = min(usable), max(usable)
                paid.append({"trace": trace, "n": n, "tokens_real": rs[0]["real"],
                             "pad_lo": lo, "pad_hi": hi,
                             "pad_ratio_change": hi / lo,
                             "cost_lo_ms": statistics.median(usable[lo]),
                             "cost_hi_ms": statistics.median(usable[hi]),
                             "cost_ratio": statistics.median(usable[hi]) / statistics.median(usable[lo]),
                             "n_lo": len(usable[lo]), "n_hi": len(usable[hi])})

        save_table(run, "steps", steps)
        save_table(run, "padding_paid", paid)

        ceiling = tot_pad / tot_real - 1.0
        ratios = [s["padding_ratio"] for s in steps]
        ratios.sort()
        print(f"[h1] {len(steps)} dispatches, {len(runs)} run(s)")
        print(f"[h1] padding ratio: mean {statistics.fmean(ratios) * 100:.1f}%   "
              f"p50 {ratios[len(ratios) // 2] * 100:.1f}%   "
              f"p95 {ratios[int(0.95 * (len(ratios) - 1))] * 100:.1f}%")
        print(f"[h1] CEILING: padded/real = {tot_pad / tot_real:.2f}x  -> "
              f"{100 * (tot_pad - tot_real) / tot_pad:.1f}% of executed tokens are padding")

        print("[h1] --- is any of it paid? fixed real work, varying padding ---")
        for p in paid:
            print(f"[h1]   n={p['n']:<3} real={p['tokens_real']:<6} "
                  f"padding x{p['pad_ratio_change']:.2f}  ->  cost x{p['cost_ratio']:.2f}"
                  f"   ({p['cost_lo_ms']:.1f} -> {p['cost_hi_ms']:.1f} ms)")
        if paid:
            flat = [p for p in paid if abs(p["cost_ratio"] - 1.0) < 0.10]
            print(f"[h1] {len(flat)}/{len(paid)} batch sizes: cost flat within 10% while "
                  f"padding rose by up to {max(p['pad_ratio_change'] for p in paid):.2f}x")
            med = statistics.median([p["cost_ratio"] for p in paid])
            if med < 1.10:
                print("[h1] VERDICT: padding is NOT paid at n>1. Cost tracks REAL tokens. "
                      "The ceiling above is free padding, so bucket-aligned packing has "
                      "nothing to recover and is analysed-and-rejected.")
            else:
                print("[h1] VERDICT: cost rises with padding at n>1 — packing has headroom.")
            save_table(run, "verdict", [{"ceiling_ratio": tot_pad / tot_real,
                                         "median_cost_ratio": med,
                                         "n_flat": len(flat), "n_cells": len(paid)}])
        print(f"[h1] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
