#!/usr/bin/env python3
"""
M5 — fit LENS's latency model on TPU, and test whether our fixed cost is a constant.

LENS (2606.18042) predicts NPU inference latency to 2.15% with a PER-BUCKET linear
model, `latency = intercept + slope * length`, calibrated from two end-to-end
measurements per bucket. The intercept is fixed overhead. We measured ONE global
intercept — 6.11 ms — at ONE batch size, n=1, which is the regime this project has
repeatedly found degenerate: per-request padding is ~97% paid at n=1 and ~10% at
n=4, so "it looked constant at n=1" is not evidence it is constant.

The paper's lead actionable result depends on it being constant. Paid padding is
compared against the cost of an added step to decide whether a step-for-alignment
trade pays, and that comparison yields a single crossover (~2048 tokens) only if
the step cost does not itself move. LENS fits its intercept per bucket, which is
published reason to expect it does.

WHAT THIS MEASURES. For each token bucket and batch size, LENS's two calibration
points — one just above b/2 where in-bucket padding is greatest, one at exactly b
where there is none — then intercept and slope per cell.

    intercept flat in both axes  -> 6.11 ms stands, the crossover is a NUMBER
    varies by bucket             -> LENS's per-bucket form holds on TPU too;
                                    the crossover is a per-boundary comparison
    varies by n                  -> the 6.11 ms is an n=1 artifact and the
                                    crossover must be rederived at serving
                                    batch sizes

AND VALIDATES LENS. A third, mid-bucket point is measured but NOT used in the fit.
Predicting it from the two calibration points tests LENS's protocol on hardware
LENS never ran on, against its stated 2.15%. That is the whole cost of turning a
characterisation into a validation: one extra measurement per cell.

Every dispatch is verified to be a single scheduler step from the
`iteration_tokens_total` count delta. Split dispatches are excluded, not averaged:
a split smears the very quantity being fitted.

Usage:
  python scripts/m5_lens_form.py --config configs/m5_lens_form.json --mock
  python scripts/m5_lens_form.py --config configs/m5_lens_form.json \\
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))

from _client import complete, complete_mock  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import MockMetrics, delta, metrics_available, scrape  # noqa: E402

PREFILL = "vllm:request_prefill_time_seconds"
ITER = "vllm:iteration_tokens_total"


def dispatch(base_url: str, model: str, n: int, seq_len: int, olen: int, seed: int,
             mock=None, mock_intercept: float = 6.11) -> tuple[float, float]:
    """One n-request dispatch of `seq_len` tokens each. Returns (ms, n_steps)."""
    before = mock.snapshot() if mock else scrape(base_url)

    def fn(i: int):
        if mock is not None:
            return complete_mock(seq_len, olen, ladder=None, staircase=False, seed=seed * 100 + i)
        return complete(base_url, model, seq_len, olen, seed=seed * 100 + i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(fn, range(n)))

    if mock is not None:
        per = mock_intercept + 0.0139 * n * seq_len
        for _ in range(n):
            mock.record(per / 1000.0, 0.0)
        mock.record_iteration(n * seq_len)
        after = mock.snapshot()
    else:
        after = scrape(base_url)
    d = delta(before, after)
    pf, it = d.get(PREFILL), d.get(ITER)
    return (pf["mean_ms"] if pf else float("nan"),
            it["count"] if it else float("nan"))


def measure(base_url, model, n, seq_len, olen, reps, discard, mock, mi) -> tuple[float, int]:
    costs, splits = [], 0
    for rep in range(-discard, reps):
        c, st = dispatch(base_url, model, n, seq_len, olen, rep, mock, mi)
        if rep < 0:
            continue
        if c == c:
            costs.append(c)
        if st == st and st > 1 + olen:
            splits += 1
    return (statistics.median(costs) if costs else float("nan")), splits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-intercept", type=float, default=6.11,
                    help="fixed cost the mock charges; used to prove the fit recovers it")
    ap.add_argument("--mock-intercept-scales-with-n", action="store_true",
                    help="mock an intercept that VARIES with n, to prove the analysis "
                         "can detect the outcome that would invalidate the crossover")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["mode"] = "mock" if args.mock else "live"
    olen = cfg.get("output_len", 1)
    reps = cfg.get("repeats", 9)
    discard = cfg.get("warmup_discard", 3)

    run = start_run("m5_lens_form", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        mock = MockMetrics() if args.mock else None
        if not mock and not metrics_available(args.base_url):
            print("[m5] /metrics unavailable — aborting.", file=sys.stderr)
            return 1

        pts: list[dict[str, Any]] = []
        fits: list[dict[str, Any]] = []
        print("[m5] LENS protocol: two calibration points per (bucket, n), plus a "
              "mid-bucket holdout the fit never sees")
        for cell in cfg["cells"]:
            b, n = cell["bucket"], cell["n"]
            mi = args.mock_intercept * (n if args.mock_intercept_scales_with_n else 1)
            got = {}
            for tag in ("lo", "hi", "mid"):
                T = cell[f"tokens_{tag}"]
                seq = T // n
                c, sp = measure(args.base_url, cfg["model"], n, seq, olen, reps, discard, mock, mi)
                got[tag] = (T, c, sp)
                pts.append({"bucket": b, "n": n, "point": tag, "tokens": T,
                            "seq_len": seq, "cost_ms": c, "splits": sp})
            (t0, c0, s0), (t1, c1, s1), (tm, cm, sm) = got["lo"], got["hi"], got["mid"]
            if c0 != c0 or c1 != c1 or t1 == t0:
                continue
            slope = (c1 - c0) / (t1 - t0)
            intercept = c0 - slope * t0
            pred = intercept + slope * tm
            ape = abs(pred - cm) / cm * 100.0 if cm == cm and cm else float("nan")
            fits.append({"bucket": b, "n": n, "intercept_ms": intercept,
                         "slope_us_per_token": slope * 1000.0,
                         "holdout_tokens": tm, "holdout_measured_ms": cm,
                         "holdout_predicted_ms": pred, "holdout_ape_pct": ape,
                         "splits": s0 + s1 + sm})
            flag = "" if s0 + s1 + sm == 0 else f"  [{s0+s1+sm} splits]"
            print(f"[m5]   bucket {b:>5} n={n:<2}  intercept {intercept:6.2f} ms  "
                  f"slope {slope*1000:5.1f} us/tok   holdout APE {ape:5.1f}%{flag}")
        save_table(run, "points", pts)
        save_table(run, "fits", fits)

        clean = [f for f in fits if f["splits"] == 0]
        if not clean:
            print("[m5] every cell split; nothing fittable", file=sys.stderr)
            return 1

        # --- is the intercept a constant? --------------------------------
        ints = [f["intercept_ms"] for f in clean]
        by_n: dict[int, list[float]] = {}
        by_b: dict[int, list[float]] = {}
        for f in clean:
            by_n.setdefault(f["n"], []).append(f["intercept_ms"])
            by_b.setdefault(f["bucket"], []).append(f["intercept_ms"])
        spread = max(ints) / min(ints) if min(ints) > 0 else float("inf")
        print(f"\n[m5] intercept across {len(clean)} clean cells: "
              f"{min(ints):.2f} to {max(ints):.2f} ms  (spread x{spread:.2f})")
        for label, grp in (("n", by_n), ("bucket", by_b)):
            meds = {k: statistics.median(v) for k, v in sorted(grp.items())}
            rng = max(meds.values()) / min(meds.values()) if min(meds.values()) > 0 else float("inf")
            print(f"[m5]   by {label}: " + "  ".join(f"{k}:{v:.2f}" for k, v in meds.items())
                  + f"   (x{rng:.2f})")

        CONST = 1.25   # within 25% counts as constant for the crossover's purposes
        verdict = ("CONSTANT" if spread < CONST else "VARIES")
        print(f"[m5] VERDICT: the fixed cost {verdict} across bucket and batch size.")
        if verdict == "CONSTANT":
            print("[m5]   The global 6.11 ms stands and the crossover is a single "
                  "threshold, as the draft claims.")
        else:
            print("[m5]   The crossover is NOT a single threshold. It must be rederived "
                  "per (bucket, n), and the draft's ~2048-token figure is a special case.")

        # --- does LENS's protocol hold here? ------------------------------
        apes = [f["holdout_ape_pct"] for f in clean if f["holdout_ape_pct"] == f["holdout_ape_pct"]]
        if apes:
            print(f"[m5] LENS validation: mid-bucket holdout MAPE {statistics.fmean(apes):.2f}%  "
                  f"worst {max(apes):.2f}%   (LENS reports 2.15% on NPUs)")
        save_table(run, "verdict", [{"intercept_min": min(ints), "intercept_max": max(ints),
                                     "spread": spread, "verdict": verdict,
                                     "holdout_mape_pct": statistics.fmean(apes) if apes else float("nan"),
                                     "n_clean_cells": len(clean)}])
        print(f"[m5] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ControlledVarError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        sys.exit(2)
