#!/usr/bin/env python3
"""
M6 — is the LENS reproduction measuring its MODEL, or only its FIT PROTOCOL?

Review question Q4/M6, and it is the sharpest criticism the draft received:

  "At n=1-2 the paper independently establishes (SS4.3, flatness 0.97) that the
   within-bucket curve is nearly flat -- under which ANY two-point fit is
   near-perfect, including a constant. The reported 0.0-0.6% error there is
   therefore weak evidence that the model form transfers."

That is a testable statement about data already captured, so it costs nothing to
answer, and it can only come out one of two ways:

  constant-only is much worse   -> the length term is doing real work; the
                                   n=1-2 result IS evidence the form transfers
  constant-only is comparable   -> it is not, and the paper must say so; the
                                   defensible claim narrows to the LOCALISATION
                                   of the error, which is what SS4.2 actually shows

Two ablations, both over the same captured points, both offline:

  A. CONSTANT-ONLY PREDICTOR. Drop the length term. Predict the held-out
     mid-bucket point as the mean of the two calibration points. This is LENS's
     protocol with its model replaced by the simplest thing that could work.

  B. FIT-POINT SENSITIVITY. Three points per cell, so three ways to choose which
     two calibrate and which is held out. LENS specifies "two measurements per
     bucket" but not WHICH two. If the n=4 error swings with that choice, the
     22.4% is a property of our sampling, not of a regime break -- and the
     reviewer's alternative explanation wins.

Neither ablation can be run on the fits table; both need the raw points, which is
why `m5_lens_form.py` saved them.

Usage:
  python scripts/m6_lens_ablation.py
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import pathlib
import statistics
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pyarrow.parquet as pq  # noqa: E402

from _common import finish_run, save_table, start_run  # noqa: E402


def fit_two(p0: dict[str, Any], p1: dict[str, Any]) -> tuple[float, float]:
    """LENS's model: cost = intercept + slope * tokens, from exactly two points."""
    dt = p1["tokens"] - p0["tokens"]
    if dt == 0:
        raise ValueError("calibration points share a token count")
    slope = (p1["cost_ms"] - p0["cost_ms"]) / dt
    return p0["cost_ms"] - slope * p0["tokens"], slope


def ape(pred: float, meas: float) -> float:
    return abs(pred - meas) / meas * 100.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-glob", default="captured/*/results/m5_lens_form/*")
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    src = None
    for d in sorted(glob.glob(args.capture_glob)):
        if (pathlib.Path(d) / "points.parquet").exists():
            src = pathlib.Path(d)
    if src is None:
        print(f"[m6] no captured m5 runs with raw points at {args.capture_glob}",
              file=sys.stderr)
        return 1

    meta = json.loads((src / "meta.json").read_text())
    conf = meta["config"]
    rows = pq.read_table(src / "points.parquet").to_pylist()

    # A capture from before a control existed has no opinion on it, and
    # start_run must not launder that silence into "controlled and passing."
    # These three were promoted to CONTROLLED_VARS after some early m5
    # sessions ran (gpu_memory_utilization in session 25; the other two with
    # the M2 MoE arms) -- backfilled here with the values that were actually
    # in force throughout, same figures used when the configs/ files
    # themselves were backfilled, not guessed fresh for this script.
    controlled = dict(conf.get("controlled", {}))
    controlled.setdefault("gpu_memory_utilization", 0.92)
    controlled.setdefault("MODEL_IMPL_TYPE", "auto")
    controlled.setdefault("MOE_ROUTE_PADDING_TO_EXPERT0", False)

    cfg = {"experiment": "m6_lens_ablation", "dimension": "D2", "mode": "offline",
           "source_run": src.name, "source_glob": args.capture_glob,
           "model": conf.get("model"), "controlled": controlled,
           "note_source": ("Re-analysis of one captured m5 run; no new measurement, so "
                           "every controlled variable is inherited verbatim from it, "
                           "backfilled where the source predates a control's addition.")}
    run = start_run("m6_lens_ablation", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        cells: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
        for r in rows:
            if r["splits"] != 0:
                continue
            cells.setdefault((r["bucket"], r["n"]), {})[r["point"]] = r

        # --- A. constant-only vs LENS linear, on the same held-out point -----
        comp: list[dict[str, Any]] = []
        for (bucket, n), pts in sorted(cells.items()):
            if not {"lo", "hi", "mid"} <= set(pts):
                continue
            lo, hi, mid = pts["lo"], pts["hi"], pts["mid"]
            b, m = fit_two(lo, hi)
            lens_pred = b + m * mid["tokens"]
            const_pred = statistics.fmean([lo["cost_ms"], hi["cost_ms"]])
            comp.append({
                "bucket": bucket, "n": n, "holdout_tokens": mid["tokens"],
                "measured_ms": mid["cost_ms"],
                "lens_pred_ms": lens_pred, "lens_ape_pct": ape(lens_pred, mid["cost_ms"]),
                "const_pred_ms": const_pred, "const_ape_pct": ape(const_pred, mid["cost_ms"]),
                # How much of the calibration spread the length term must explain.
                "calib_spread_pct": abs(hi["cost_ms"] - lo["cost_ms"]) / lo["cost_ms"] * 100.0,
            })
        save_table(run, "constant_vs_lens", comp)

        # --- B. sensitivity to WHICH two points calibrate -------------------
        sens: list[dict[str, Any]] = []
        for (bucket, n), pts in sorted(cells.items()):
            have = [pts[k] for k in ("lo", "hi", "mid") if k in pts]
            if len(have) < 3:
                continue
            apes = []
            for pair in itertools.combinations(range(3), 2):
                held = [i for i in range(3) if i not in pair][0]
                p0, p1 = have[pair[0]], have[pair[1]]
                if p1["tokens"] == p0["tokens"]:
                    continue
                b, m = fit_two(p0, p1)
                h = have[held]
                apes.append({"held": h["point"], "ape": ape(b + m * h["tokens"], h["cost_ms"]),
                             "slope": m * 1000.0})
            if len(apes) < 2:
                continue
            vals = [a["ape"] for a in apes]
            sens.append({
                "bucket": bucket, "n": n, "n_choices": len(apes),
                "ape_min_pct": min(vals), "ape_max_pct": max(vals),
                "ape_median_pct": statistics.median(vals),
                "ape_swing_pct": max(vals) - min(vals),
                "slope_min_us": min(a["slope"] for a in apes),
                "slope_max_us": max(a["slope"] for a in apes),
                "detail": json.dumps([{k: round(v, 3) if isinstance(v, float) else v
                                       for k, v in a.items()} for a in apes]),
            })
        save_table(run, "fit_point_sensitivity", sens)

        # --- report ----------------------------------------------------------
        buckets = sorted({c["bucket"] for c in comp})
        print(f"[m6] {len(comp)} cells: {len(buckets)} buckets {buckets} x "
              f"{sorted({c['n'] for c in comp})} batch sizes, 3 points/cell, "
              f"{conf.get('repeats', '?')} repeats/point")
        print("[m6] --- A. does the LENGTH TERM earn its place? ---")
        print(f"[m6] {'n':>3} {'bucket':>7} {'LENS%':>7} {'const%':>7} {'calib spread%':>14}")
        for c in sorted(comp, key=lambda c: (c["n"], c["bucket"])):
            print(f"[m6] {c['n']:>3} {c['bucket']:>7} {c['lens_ape_pct']:>7.2f} "
                  f"{c['const_ape_pct']:>7.2f} {c['calib_spread_pct']:>14.2f}")

        summary = []
        for n in sorted({c["n"] for c in comp}):
            g = [c for c in comp if c["n"] == n]
            lens_m = statistics.fmean([c["lens_ape_pct"] for c in g])
            const_m = statistics.fmean([c["const_ape_pct"] for c in g])
            summary.append({"n": n, "cells": len(g), "lens_mape_pct": lens_m,
                            "const_mape_pct": const_m,
                            "length_term_gain_pct_points": const_m - lens_m})
            print(f"[m6] n={n:<3} LENS {lens_m:5.2f}%   constant-only {const_m:5.2f}%   "
                  f"length term buys {const_m - lens_m:+5.2f} pp")
        save_table(run, "summary_by_n", summary)

        flat = [s for s in summary if s["n"] <= 2]
        if flat and all(s["length_term_gain_pct_points"] < 1.0 for s in flat):
            print("[m6] VERDICT A: at n<=2 the length term buys <1 pp. The near-perfect "
                  "n=1-2 accuracy is NOT evidence that LENS's model form transfers -- a "
                  "constant does as well. The reviewer is right; the claim must narrow "
                  "to the localisation of the error.")
        else:
            print("[m6] VERDICT A: the length term earns its place at n<=2; the "
                  "reported accuracy does support the model form.")

        print("[m6] --- B. is the n=4 error an artifact of WHICH points were fit? ---")
        for s in sorted(sens, key=lambda s: (s["n"], s["bucket"])):
            print(f"[m6] n={s['n']:<3} bucket={s['bucket']:<5} APE over the "
                  f"{s['n_choices']} calibration choices: {s['ape_min_pct']:6.2f}% .. "
                  f"{s['ape_max_pct']:6.2f}%  (swing {s['ape_swing_pct']:.2f} pp)")
        s4 = [s for s in sens if s["n"] == 4]
        if s4:
            worst = max(s["ape_swing_pct"] for s4_ in [s4] for s in s4_)
            lo4 = min(s["ape_min_pct"] for s in s4)
            print(f"[m6] at n=4 the error is {lo4:.2f}% at best and swings up to "
                  f"{worst:.2f} pp with the choice of calibration points.")
            if lo4 > 5.0:
                print("[m6] VERDICT B: even the most favourable choice of calibration "
                      "points leaves n=4 far above LENS's reported 2.15%. The "
                      "localisation survives the sensitivity check.")
            else:
                print("[m6] VERDICT B: a favourable choice of calibration points brings "
                      "n=4 near LENS's reported accuracy -- the error is at least "
                      "partly a sampling artifact and must be reported as a range.")
        print(f"[m6] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
