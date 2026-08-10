#!/usr/bin/env python3
"""
e30 — policy sweep. The paper's core result table, and it costs nothing.

Compares admission policies on MATCHED traces (identical arrivals per seed, so
differences are policy, not luck), across arrival rates, with paired-bootstrap
intervals on the cost difference.

Baseline is `promote (stock)` because session 3 measured that stock vLLM
dispatches immediately and pads the batch — queue time was 0.0 ms at every
concurrency tested. The question is where that is the wrong call.

Usage:
  python scripts/e30_policy_sweep.py --config configs/e30_policy_sweep.json
"""
from __future__ import annotations

import argparse, json, statistics, sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "sim"))

from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _stats import bootstrap_ci, bootstrap_p  # noqa: E402
from cost_model import CostModel  # noqa: E402
from policies import Hybrid, ALL_POLICIES  # noqa: E402
from simulator import Simulator  # noqa: E402


def build(name: str, cfg: dict[str, Any]):
    if name == "hybrid":
        return Hybrid(latency_weight=cfg.get("latency_weight", 1.0),
                      max_wait_s=cfg.get("max_wait_s", 0.5))
    cls = ALL_POLICIES[name]
    try:
        return cls(max_wait_s=cfg.get("max_wait_s", 0.5))
    except TypeError:
        return cls()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["mode"] = "simulation"
    rates = cfg.get("rates_hz", [20, 50, 100, 200, 400])
    seeds = cfg.get("seeds", 30)
    n_req = cfg.get("n_requests", 400)
    plen = cfg.get("prompt_len", 512)
    names = cfg.get("policies", ["promote", "wait", "hybrid", "downshift", "oracle"])

    run = start_run("e30_policy_sweep", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        sim = Simulator(CostModel())
        rows: list[dict[str, Any]] = []
        for rate in rates:
            for seed in range(seeds):
                trace = sim.make_trace(n_req, rate, plen, seed)   # MATCHED across policies
                for nm in names:
                    p = build(nm, cfg)
                    p._rate_hz, p._seed = rate, seed
                    rows.append({"rate_hz": rate, "seed": seed, "policy_key": nm,
                                 **sim.run(trace, p).as_row()})
        save_table(run, "runs", rows)

        # --- paired comparison against stock, per rate ---
        summary = []
        print(f"[e30] baseline = promote (stock). cost = TPU-ms per request; lower is better.")
        for rate in rates:
            base = [r["cost_per_request_ms"] for r in rows
                    if r["rate_hz"] == rate and r["policy_key"] == "promote"]
            print(f"[e30] --- {rate} req/s ---")
            for nm in names:
                arm = [r["cost_per_request_ms"] for r in rows
                       if r["rate_hz"] == rate and r["policy_key"] == nm]
                lat = [r["p95_latency_ms"] for r in rows
                       if r["rate_hz"] == rate and r["policy_key"] == nm]
                d = statistics.fmean(base) - statistics.fmean(arm)          # >0 == cheaper
                lo, hi = bootstrap_ci(base, arm, n=2000)
                pv = bootstrap_p(base, arm, n=2000) if nm != "promote" else 1.0
                pct = d / statistics.fmean(base) * 100
                summary.append({"rate_hz": rate, "policy": nm,
                                "cost_per_req_ms": statistics.fmean(arm),
                                "saving_vs_stock_ms": d, "saving_pct": pct,
                                "ci_lo": lo, "ci_hi": hi, "p": pv,
                                "p95_latency_ms": statistics.fmean(lat)})
                sig = "" if nm == "promote" else ("  *" if pv < 0.05 else "  ns")
                print(f"[e30]   {nm:<16} {statistics.fmean(arm):8.2f} ms/req  "
                      f"saving {pct:+6.1f}%  [{lo:+.2f},{hi:+.2f}] p={pv:.3f}"
                      f"  p95_lat {statistics.fmean(lat):8.1f} ms{sig}")
        save_table(run, "summary", summary)
        print(f"[e30] run_id={run.run_id}")
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
        print(f"ABORT: {e}", file=sys.stderr); sys.exit(2)
