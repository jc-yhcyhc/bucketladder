#!/usr/bin/env python3
"""
e40 — holdout validation. Turns simulated predictions into measured results.

plan_v4's rule: **MAPE < 15% per parameter, on hardware the fit never saw.**
Until that passes, e30's numbers are internally consistent predictions and
nothing more.

THE KEY REALISATION, which makes this possible at all:

    Session 3 measured that stock vLLM dispatches whatever is waiting, right
    away — queue time was 0.0 ms at every concurrency level. **So the batch vLLM
    forms is exactly the set of requests that have arrived at it.** Which means
    admission policy can be implemented ENTIRELY CLIENT-SIDE, by choosing when
    to release requests, with no scheduler patch at all.

    Hold 8 requests back and release them together -> vLLM forms a batch of 8.
    Release each on arrival -> vLLM behaves as `promote`.

That has two consequences. It makes every policy in e30 measurable on unmodified
hardware, which is what this script does. And it means the contribution is
*deployable* as a proxy in front of an unmodified server, rather than requiring
a fork — worth stating in the paper.

What is compared: predicted vs measured **TPU-busy ms per request**, from
/metrics deltas taken around EACH BATCH (solidity.md R1).

UNITS TRAP, caught in mock before it reached hardware. vLLM records
`request_prefill_time_seconds` PER REQUEST, and every request in a batch sees
the whole batch's duration. So summing the metric over a run gives sum(n_i *
T_i), while the simulator's cost is sum(T_i) — TPU-busy time. They differ by a
factor of batch size, which would have shown up as a ~70% MAPE and looked like a
broken cost model. Scraping per batch recovers T_i directly, because
delta()["mean_ms"] over one batch IS that batch's duration.

Usage:
  python scripts/e40_holdout.py --config configs/e40_holdout.json --mock
  python scripts/e40_holdout.py --config configs/e40_holdout.json \
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))

from _client import complete, complete_mock  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import MockMetrics, delta, metrics_available, scrape  # noqa: E402
from cost_model import CostModel, padded_batch  # noqa: E402
from policies import ALL_POLICIES, Hybrid  # noqa: E402
from simulator import Simulator  # noqa: E402


def build_policy(name: str, cfg: dict[str, Any]):
    if name == "hybrid":
        return Hybrid(latency_weight=cfg.get("latency_weight", 1.0),
                      max_wait_s=cfg.get("max_wait_s", 0.5))
    cls = ALL_POLICIES[name]
    try:
        return cls(max_wait_s=cfg.get("max_wait_s", 0.5))
    except TypeError:
        return cls()


def drive(base_url: str, model: str, trace, policy, cost: CostModel,
          prompt_len: int, output_len: int, mock_metrics=None,
          scrape_each: bool = True) -> dict[str, Any]:
    """Replay `trace` against a real server, applying `policy` client-side.

    Requests are released in batches exactly when the policy says to dispatch.
    Because stock vLLM never holds anything back, a released group becomes one
    server-side batch — so client-side release timing *is* admission control.
    """
    pending: list = []
    idx = 0
    t0 = time.perf_counter()
    dispatched: list[dict[str, Any]] = []
    # Per-request end-to-end latency, measured from the request's TRACE ARRIVAL
    # time -- not from when the policy chose to release it. That distinction is
    # the whole point: a policy that holds requests back to fill a batch buys
    # its cost saving with delay, and charging from release time would hide
    # exactly the cost being traded away. Without this the headline claim
    # ("saves 23% at no latency cost") is half unmeasured.
    latencies: list[dict[str, Any]] = []

    def elapsed() -> float:
        return time.perf_counter() - t0

    def fire(group) -> None:
        """Release a group simultaneously; they contend for one batch.

        Metrics are scraped around this group alone so the delta's mean is the
        batch's own duration T_i, which is what the simulator models.
        """
        before_b = (mock_metrics.snapshot() if mock_metrics
                    else scrape(base_url)) if scrape_each else None
        threads, results = [], [None] * len(group)

        def one(i, req):
            if mock_metrics is not None:
                s = complete_mock(prompt_len, output_len, ladder=None, staircase=False,
                                  seed=req.rid)
                # Emulate the server faithfully: ONE batch duration, recorded
                # once per request (which is what vLLM's histogram does).
                per = cost.step_cost_ms(len(group), len(group) * prompt_len)
                mock_metrics.record(per / 1000.0, 0.0)
                s.ttft_ms = per
                # Take real time. Without this the mock returns instantly, no
                # arrivals accumulate while the "server" is busy, every batch is
                # size 1, and the harness cannot reproduce queueing at all —
                # which showed up as a 70% error on `promote` that had nothing
                # to do with the cost model.
                time.sleep(per / 1000.0)
            else:
                s = complete(base_url, model, prompt_len, output_len, seed=req.rid)
            results[i] = s

        released_at = elapsed()
        for i, req in enumerate(group):
            th = threading.Thread(target=one, args=(i, req), daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        done_at = elapsed()
        for i, req in enumerate(group):
            latencies.append({
                "rid": req.rid,
                # Held in the client queue because the policy chose to wait.
                "queue_ms": (released_at - req.arrival_s) * 1000.0,
                "service_ms": (done_at - released_at) * 1000.0,
                # What the caller actually experiences.
                "latency_ms": (done_at - req.arrival_s) * 1000.0,
                "ttft_ms": results[i].ttft_ms if results[i] is not None else float("nan"),
                "batch_n": len(group),
            })
        batch_ms = float("nan")
        n_seen, queue_ms = float("nan"), float("nan")
        if scrape_each:
            after_b = mock_metrics.snapshot() if mock_metrics else scrape(base_url)
            d = delta(before_b, after_b)
            dd = d.get("vllm:request_prefill_time_seconds")
            if dd:
                batch_ms = dd["mean_ms"]      # == this batch's duration T_i
                n_seen = dd["count"]
            qq = d.get("vllm:request_queue_time_seconds")
            if qq:
                queue_ms = qq["mean_ms"]
        # SPLIT DIAGNOSTIC. `batch_ms` is the batch's duration only if the whole
        # released group landed in ONE server-side step. If vLLM split it, the
        # delta's mean is the average over several steps while the simulator's
        # cost is their sum, and the APE is then measuring the release mechanism
        # rather than the cost model. Two signals distinguish the cases:
        # `n_seen != n` means requests leaked across the scrape window, and
        # queue_ms > 0 means somebody waited for a later step (e02 measured
        # stock queue time at 0.0 ms, so any positive value here is a split).
        dispatched.append({"n": len(group), "padded_to": padded_batch(len(group), cost.ladder),
                           "t_s": elapsed(), "batch_ms": batch_ms,
                           "n_seen": n_seen, "queue_ms": queue_ms})

    while idx < len(trace) or pending:
        now = elapsed()
        while idx < len(trace) and trace[idx].arrival_s <= now:
            pending.append(trace[idx]); idx += 1
        if not pending:
            nxt = trace[idx].arrival_s
            time.sleep(max(0.0, nxt - elapsed()))
            continue

        next_arrival = trace[idx].arrival_s if idx < len(trace) else None
        d = policy.decide(n_pending=len(pending), now_s=elapsed(),
                          oldest_wait_s=elapsed() - pending[0].arrival_s,
                          next_arrival_s=next_arrival, cost=cost, prompt_len=prompt_len)
        if d.wait_until_s is not None and next_arrival is not None:
            target = min(d.wait_until_s, next_arrival)
            if target > elapsed() + 1e-6:
                time.sleep(min(target - elapsed(), 1.0))
                continue
        take = min(d.dispatch_n or len(pending), len(pending),
                   max(1, cost.max_batched_tokens // max(1, prompt_len)))
        fire(pending[:take])
        pending = pending[take:]

    return {"batches": dispatched, "wall_s": elapsed(), "latencies": latencies}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="override the trace seed. Re-running across seeds is how the "
                         "MEASURED policy comparison gets an interval (solidity.md R4): "
                         "one run per cell is a point estimate and may not be reported.")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["mode"] = "mock" if args.mock else "live"
    if args.seed is not None:
        cfg["seed"] = args.seed
    plen = cfg.get("prompt_len", 512)
    olen = cfg.get("output_len", 1)
    n_req = cfg.get("n_requests", 120)
    cells = cfg.get("cells") or [{"policy": p, "rate_hz": r}
                                 for r in cfg.get("rates_hz", [25, 55])
                                 for p in cfg.get("policies", ["promote", "wait", "hybrid"])]

    run = start_run("e40_holdout", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        cost = CostModel()
        sim = Simulator(cost)
        mock_metrics = MockMetrics() if args.mock else None
        use_metrics = bool(mock_metrics) or metrics_available(args.base_url)
        if not use_metrics:
            print("[e40] /metrics unavailable — cannot validate in TPU-ms. Aborting.",
                  file=sys.stderr)
            return 1

        rows: list[dict[str, Any]] = []
        batch_rows: list[dict[str, Any]] = []
        lat_rows: list[dict[str, Any]] = []
        for cell in cells:
            pname, rate = cell["policy"], cell["rate_hz"]
            trace = sim.make_trace(n_req, rate, plen, seed=cfg.get("seed", 0))

            predicted = sim.run(trace, build_policy(pname, cfg)).cost_per_request_ms

            out = drive(args.base_url, cfg["model"], trace, build_policy(pname, cfg),
                        cost, plen, olen, mock_metrics)
            # TPU-busy per request = sum of per-batch durations / requests.
            busy = [b["batch_ms"] for b in out["batches"] if not math.isnan(b["batch_ms"])]
            measured = (sum(busy) / n_req) if busy else float("nan")

            ape = (abs(measured - predicted) / measured * 100.0
                   if measured and not math.isnan(measured) else float("nan"))

            # Did the client-side release actually produce the batches we asked
            # for? Without this, a failing APE cannot be attributed.
            split = [b for b in out["batches"]
                     if not math.isnan(b["n_seen"]) and b["n_seen"] != b["n"]]
            queued = [b["queue_ms"] for b in out["batches"]
                      if not math.isnan(b["queue_ms"]) and b["queue_ms"] > 0.0]
            split_pct = 100.0 * len(split) / len(out["batches"]) if out["batches"] else float("nan")
            for b in out["batches"]:
                batch_rows.append({"policy": pname, "rate_hz": rate, **b})

            lats = sorted(x["latency_ms"] for x in out["latencies"])
            held = [x["queue_ms"] for x in out["latencies"]]
            p50 = statistics.median(lats) if lats else float("nan")
            p95 = lats[int(0.95 * (len(lats) - 1))] if lats else float("nan")
            for x in out["latencies"]:
                lat_rows.append({"policy": pname, "rate_hz": rate, **x})

            rows.append({"policy": pname, "rate_hz": rate,
                         "predicted_ms_per_req": predicted,
                         "measured_ms_per_req": measured, "ape_pct": ape,
                         "n_requests": n_req, "n_batches": len(out["batches"]),
                         "split_batch_pct": split_pct,
                         "mean_queue_ms": statistics.fmean(queued) if queued else 0.0,
                         # The price of the saving. Reported in the same row so
                         # a cost number can never be quoted without it.
                         "p50_latency_ms": p50, "p95_latency_ms": p95,
                         "mean_held_ms": statistics.fmean(held) if held else 0.0})
            flag = "" if not split else f"   SPLIT {split_pct:.0f}% of batches"
            print(f"[e40] {pname:<10} {rate:>4} req/s   predicted {predicted:7.2f}  "
                  f"measured {measured:7.2f}  APE {ape:5.1f}%   "
                  f"p50 {p50:6.1f} p95 {p95:7.1f} ms{flag}")
        save_table(run, "holdout", rows)
        save_table(run, "batches", batch_rows)
        save_table(run, "latency", lat_rows)

        # Cost and latency, side by side, against stock. A saving quoted without
        # the delay it was bought with is not a result.
        for rate in sorted({r["rate_hz"] for r in rows}):
            base = next((r for r in rows if r["rate_hz"] == rate and r["policy"] == "promote"), None)
            if not base:
                continue
            print(f"[e40] --- {rate} req/s, against stock (promote) ---")
            for r in rows:
                if r["rate_hz"] != rate or r["policy"] == "promote":
                    continue
                d_cost = 100.0 * (1 - r["measured_ms_per_req"] / base["measured_ms_per_req"])
                d_p95 = 100.0 * (r["p95_latency_ms"] / base["p95_latency_ms"] - 1)
                print(f"[e40]   {r['policy']:<8} TPU cost {d_cost:+6.1f}%   p95 latency {d_p95:+7.1f}%")

        # A split rate above a few percent means the APE is testing the release
        # mechanism, not the cost model — report it before the verdict, so a
        # FAIL is never silently attributed to the wrong thing.
        worst_split = max((r["split_batch_pct"] for r in rows
                           if not math.isnan(r["split_batch_pct"])), default=0.0)
        if worst_split > 5.0:
            print(f"[e40] WARNING batches split from their released group in up to "
                  f"{worst_split:.0f}% of dispatches. Client-side release is NOT "
                  f"cleanly controlling batch composition, so the APE below is not "
                  f"a clean test of the cost model.", file=sys.stderr)

        apes = [r["ape_pct"] for r in rows if not math.isnan(r["ape_pct"])]
        if apes:
            mape = statistics.fmean(apes)
            worst = max(apes)
            print(f"[e40] MAPE = {mape:.1f}%   worst cell = {worst:.1f}%")
            verdict = "PASS" if worst < 15.0 else "FAIL"
            print(f"[e40] plan_v4 rule is <15% PER CELL -> {verdict}")
            save_table(run, "verdict", [{"mape_pct": mape, "worst_ape_pct": worst,
                                         "threshold_pct": 15.0, "verdict": verdict}])
            if verdict == "FAIL":
                print("[e40] the cost model must be refit and e30 re-run before its "
                      "numbers may be reported", file=sys.stderr)
        print(f"[e40] run_id={run.run_id}")
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
