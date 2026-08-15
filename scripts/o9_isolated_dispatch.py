#!/usr/bin/env python3
"""
O9 — a valid placebo above n=2, by making prefills isolated.

§4.4 sweeps concurrency and finds the finer ladder's benefit does not decay. Its
control is prompt 300, chosen because it pads to 512 on both ladders. That holds
only while each request prefills in its own dispatch. Above n=2 the scheduler
admits requests in waves and packs them, the packed step selects the compiled
shape, and the control cell becomes a treated cell — so §4.4's n>=4 rows are
floors on the effect rather than estimates of it, with arm order as their only
control.

The fix is to separate the two things concurrency does. Decode concurrency is what
makes the regime realistic; prefill co-scheduling is what breaks the control. A
launcher that staggers arrivals by more than a prefill's duration keeps each
prefill alone in its step while decodes from earlier requests are still running,
so the per-request ladder mapping holds and prompt 300 is inert again.

This is deliberately less realistic than §4.4's burst arrival. It is not a better
measurement of production; it is the control that bounds §4.4's claim at interval
level rather than as a floor.

REGISTERED PREDICTION. With prefills isolated, the prompt-300 cell returns to a
difference indistinguishable from zero at every concurrency, while 1200 and 3000
keep the reductions §4.3 measured at n=2. If instead the placebo still moves,
isolation has not been achieved and the staggering interval is too short — which
the executed step-size histogram will show directly.

Usage:
  python scripts/o9_isolated_dispatch.py --config configs/o9_isolated_dispatch.json \\
      --arm default --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _client import complete  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import metrics_available  # noqa: E402


def staggered_batch(base_url: str, model: str, plen: int, olen: int,
                    n: int, stagger_s: float, seed: int) -> list[float]:
    """Fire n requests spaced by stagger_s, and collect their latencies.

    The spacing is what isolates the prefills: each request is released after the
    previous one has had time to finish its prefill, while its decode continues.
    """
    out: list[float | None] = [None] * n
    threads: list[threading.Thread] = []

    def one(i: int) -> None:
        s = complete(base_url, model, plen, olen, seed=seed * 100 + i)
        out[i] = s.total_ms if s.ok else None

    for i in range(n):
        t = threading.Thread(target=one, args=(i,))
        t.start()
        threads.append(t)
        if i < n - 1:
            time.sleep(stagger_s)
    for t in threads:
        t.join()
    return [v for v in out if v is not None]


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
    cfg["controlled"] = dict(cfg["controlled"])
    cfg["controlled"]["VLLM_TPU_BUCKET_PADDING_GAP"] = (
        "1024" if args.arm == "gap1024" else "")
    if not metrics_available(args.base_url):
        print(f"[o9] no /metrics at {args.base_url}", file=sys.stderr)
        return 1
    try:
        run = start_run("o9_isolated_dispatch", cfg, results_root=args.results_root)
    except ControlledVarError as e:
        print(f"[o9] {e}", file=sys.stderr)
        return 2

    status, err = "ok", None
    try:
        rows, reps = [], []
        stagger = cfg["stagger_ms"] / 1000.0
        for n in cfg["concurrency_levels"]:
            for plen in cfg["prompt_lens"]:
                lat: list[float] = []
                for rep in range(cfg["repeats"] + cfg["warmup_discard"]):
                    got = staggered_batch(args.base_url, cfg["model"], plen,
                                          cfg["output_len"], n, stagger, rep)
                    if rep < cfg["warmup_discard"] or not got:
                        continue
                    lat.append(statistics.median(got))
                    reps.append({"arm": args.arm, "concurrency": n,
                                 "prompt_len": plen, "rep": rep,
                                 "e2e_ms": lat[-1]})
                if lat:
                    rows.append({"arm": args.arm, "concurrency": n,
                                 "prompt_len": plen,
                                 "e2e_ms_median": statistics.median(lat),
                                 "reps": len(lat)})
                    print(f"[o9:{args.arm}] n={n:<3} plen={plen:<5} "
                          f"e2e {statistics.median(lat):8.1f} ms")
        if not rows:
            raise RuntimeError("no completed requests; arm not measured")
        save_table(run, "latency", rows)
        save_table(run, "latency_reps", reps)
        print(f"[o9] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    raise SystemExit(main())
