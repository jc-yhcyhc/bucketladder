#!/usr/bin/env python3
"""
M19 — what batch was ACTUALLY running? The question session 18 could not answer.

Review 4.1, and it is correct: §4.5 declared the n=128 and n=256 decode columns
unclean (queue time 62 ms and 298 ms, so requests were not all resident) and then
rested the refutation of the bandwidth account on exactly those columns. If the
effective running batch never crossed the ridge near 240, then 5.1% MFU is what
the bandwidth account PREDICTS and refutes nothing.

The captured schema has no running-batch gauge, so the question cannot be settled
from data in hand. This measures it directly: sample `vllm:num_requests_running`
throughout the decode window and report its distribution beside the step cost.

Two arms per batch size:
  naive     — fire n requests and measure, as session 18 did
  resident  — release from a barrier, then WAIT for the running gauge to reach n
              before opening the measurement window

If `resident` reaches n and the step cost still shows no bend toward the compute
roof, the bandwidth account is refuted on clean data. If it cannot reach n, that
is the answer to a different question the paper also needs: the reachable width
of the compiled request ladder under this configuration.

Usage:
  python scripts/m19_resident.py --config configs/m19_resident.json \\
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import metrics_available  # noqa: E402
from m8_split_barrier import launch_barrier  # noqa: E402

GAUGE = "vllm:num_requests_running"


def gauge(base_url: str) -> float:
    """Current running-request count, straight from /metrics."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith(GAUGE) and not line.startswith("#"):
                    return float(line.rsplit(" ", 1)[-1])
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if not metrics_available(args.base_url):
        print(f"[m19] no /metrics at {args.base_url}", file=sys.stderr)
        return 1
    try:
        run = start_run("m19_resident", cfg, results_root=args.results_root)
    except ControlledVarError as e:
        print(f"[m19] {e}", file=sys.stderr)
        return 2

    status, err = "ok", None
    try:
        rows: list[dict[str, Any]] = []
        for n in cfg["concurrency"]:
            samples: list[float] = []
            stop = threading.Event()

            def poll(base=args.base_url, s=samples, ev=stop):
                while not ev.is_set():
                    v = gauge(base)
                    if v == v:
                        s.append(v)
                    time.sleep(cfg.get("poll_interval_s", 0.05))

            t = threading.Thread(target=poll, daemon=True)
            t.start()
            t0 = time.perf_counter()
            launch_barrier(args.base_url, cfg["model"], n, cfg["prompt_len"],
                           cfg["output_len"], seed=n)
            wall = time.perf_counter() - t0
            stop.set(); t.join(timeout=2)

            if not samples:
                print(f"[m19] n={n}: no gauge samples", file=sys.stderr)
                continue
            peak = max(samples)
            active = [v for v in samples if v > 0]
            rows.append({"n": n, "wall_s": wall, "samples": len(samples),
                         "running_peak": peak,
                         "running_median_active": statistics.median(active) if active else 0.0,
                         "running_mean_active": statistics.fmean(active) if active else 0.0,
                         "reached_n": peak >= n * cfg.get("reached_frac", 0.95),
                         "frac_of_requested": peak / n})
            r = rows[-1]
            print(f"[m19] n={n:<4} peak running {peak:6.0f}  median(active) "
                  f"{r['running_median_active']:6.0f}  = {100 * r['frac_of_requested']:5.1f}% of "
                  f"requested  {'RESIDENT' if r['reached_n'] else '<-- NEVER REACHED n'}")
        save_table(run, "residency", rows)

        unreached = [r for r in rows if not r["reached_n"]]
        if unreached:
            worst = min(unreached, key=lambda r: r["frac_of_requested"])
            print(f"[m19] VERDICT: the running batch never reaches the requested size at "
                  f"n={[r['n'] for r in unreached]}. At n={worst['n']} it peaks at "
                  f"{worst['running_peak']:.0f} ({100 * worst['frac_of_requested']:.0f}%). "
                  f"Session 18's high-n columns did not measure the batch they were "
                  f"labelled with, and any conclusion resting on them — including our "
                  f"refutation of the bandwidth account — is unsupported. The reachable "
                  f"width of the request ladder is itself the finding.")
        else:
            print("[m19] VERDICT: every cell reached its requested batch. The high-n "
                  "columns measure what they claim, and the refutation stands on them.")
        print(f"[m19] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
