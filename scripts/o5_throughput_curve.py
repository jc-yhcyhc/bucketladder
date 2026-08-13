#!/usr/bin/env python3
"""
O5 — does the ladder win convert into capacity, or only into latency?

Every ladder number in this paper is latency at a fixed, small concurrency, and
the cost of a longer ladder is denominated in KV cache tokens. Those are two
different currencies and the paper never converts them, so a reader deciding
whether to spend memory on shapes has nothing to divide. This measures the
conversion: offered load against sustained goodput and p95 latency, on the
ten-shape default and the fourteen-shape placement ladder, both at the stock 0.92
memory fraction.

REGISTERED PREDICTION (see the config). Padded tokens are real arithmetic, so
removing 1024 of them per prefill step should raise the rate at which the server
saturates -- not merely lower latency under it. If the curves instead coincide and
only the low-load latencies differ, the padding removed was slack the server had
anyway, and the recommendation is latency-only.

WHY OPEN-LOOP. Requests are issued on a Poisson schedule at a target rate and NOT
throttled by completions. A closed-loop harness with N workers cannot saturate a
server -- it self-limits, because a slow server simply gets fewer requests -- and
would draw a flat, reassuring curve no matter what the server did. Open-loop is
what makes a knee visible, and the queue depth that builds past the knee is the
signal, not an artefact.

Goodput is completed requests per second measured over the steady window, so a
request the server never finishes is not counted as served.

Usage:
  python scripts/o5_throughput_curve.py --config configs/o5_throughput_curve.json \\
      --arm default --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import pathlib
import random
import statistics
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _client import complete  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import metrics_available  # noqa: E402


def run_rate(base_url: str, model: str, plen: int, olen: int,
             rate: float, n_requests: int) -> dict:
    """Issue n_requests on a Poisson schedule at `rate`/s, open loop."""
    results: list = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []
    rng = random.Random(1234)

    def one(seed: int) -> None:
        s = complete(base_url, model, plen, olen, seed=seed)
        with lock:
            results.append(s)

    t_start = time.time()
    for i in range(n_requests):
        th = threading.Thread(target=one, args=(i,), daemon=True)
        th.start()
        threads.append(th)
        time.sleep(rng.expovariate(rate))
    issue_done = time.time()
    for th in threads:
        th.join(timeout=180)
    t_end = time.time()

    ok = [s for s in results if s.ok]
    lat = sorted(s.total_ms for s in ok)
    if not lat:
        return {"rate": rate, "completed": 0, "goodput": 0.0, "p50_ms": float("nan"),
                "p95_ms": float("nan"), "issue_s": issue_done - t_start,
                "wall_s": t_end - t_start, "failed": len(results) - len(ok)}
    return {
        "rate": rate,
        "completed": len(ok),
        # Completed requests over the whole wall window, including drain: a
        # server past its knee finishes late, and dividing by the issue window
        # would credit it with throughput it did not have.
        "goodput": len(ok) / (t_end - t_start),
        "p50_ms": lat[len(lat) // 2],
        "p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        "issue_s": issue_done - t_start,
        "wall_s": t_end - t_start,
        "failed": len(results) - len(ok),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--arm", required=True, choices=["default", "gap1024"])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    cfg = dict(load_config(args.config))
    cfg["arm"] = args.arm
    if not metrics_available(args.base_url):
        print(f"[o5] no /metrics at {args.base_url}", file=sys.stderr)
        return 1
    try:
        run = start_run("o5_throughput_curve", cfg, results_root=args.results_root)
    except ControlledVarError as e:
        print(f"[o5] {e}", file=sys.stderr)
        return 2

    status, err = "ok", None
    try:
        rows = []
        print(f"[o5:{args.arm}] {'rate':>6}{'done':>6}{'goodput':>9}"
              f"{'p50 ms':>9}{'p95 ms':>10}{'fail':>6}")
        for rate in cfg["arrival_rates"]:
            r = run_rate(args.base_url, cfg["model"], cfg["prompt_len"],
                         cfg["output_len"], float(rate), cfg["requests_per_rate"])
            r["arm"] = args.arm
            rows.append(r)
            print(f"[o5:{args.arm}] {rate:>6}{r['completed']:>6}{r['goodput']:>9.2f}"
                  f"{r['p50_ms']:>9.0f}{r['p95_ms']:>10.0f}{r['failed']:>6}")
            time.sleep(5)          # let the queue drain between rates
        if not any(r["completed"] for r in rows):
            raise RuntimeError("no completed requests at any rate; arm not measured")
        save_table(run, "throughput", rows)
        print(f"[o5] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
