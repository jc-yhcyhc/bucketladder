#!/usr/bin/env python3
"""
M8 — is "unmeasurable above n=8" a property of the SYSTEM or of our HARNESS?

Review findings M3/Q3 asked whether the n>8 measurement barrier is removable, and
proposed a cause: "Splitting is governed by `max_num_batched_tokens`... Raising
it should admit single-step dispatches at n=16 and n=32."

The captured data rejects that cause before any hardware is provisioned:

    cell              total tokens   max_num_batched_tokens   splits?
    n4:4096/8192          8192              8192               0 / 9
    n8:1024/2048          ~1024             8192               4 / 9

A dispatch at the token limit does NOT split, and one at an eighth of it DOES.
Splitting tracks the REQUEST count, not the token count, so raising the token
limit cannot fix it and the proposed experiment would have measured nothing.

That leaves a better hypothesis, and a worse one for us. The split fraction is
not deterministic -- it is ~50% at n=8 and ~100% at n=16-32. A capacity limit
would split every time or never. A RACE produces exactly this: `one_dispatch`
launches n requests through a ThreadPoolExecutor, each opening its own HTTP
connection, so arrivals smear across some milliseconds while the scheduler ticks
every few. The more requests, the likelier at least one lands in a later tick.

If that is right, the barrier is OURS. "The production regime is unmeasurable" is
then a much weaker and much more embarrassing claim than the paper makes: it is
"our launcher cannot deliver n requests inside one scheduler tick."

THE FIX, and the experiment. Two changes to how a dispatch is launched:

  1. Pre-open every connection and send everything but the last byte, so the
     per-thread cost of connecting is paid BEFORE the timed launch.
  2. Release all n threads from a `threading.Barrier`, so they hit the socket
     within microseconds of each other instead of milliseconds.

Then measure the single-step fraction at n = 4, 8, 16, 32 under both launchers.
This is a paired A/B on the same server, same workload, same order:

    barrier fixes it        -> the barrier was ours; n>8 becomes measurable and
                               the paid-padding curve extends into the regime
                               production actually runs in
    barrier does not fix it -> the splitting is the scheduler's, and the paper
                               can finally state the mechanism instead of the
                               symptom

Either outcome is publishable and the current text is wrong, which is why this
runs before anything else in session 13.

Usage:
  python scripts/m8_split_barrier.py --config configs/m8_split_barrier.json --mock
  python scripts/m8_split_barrier.py --config configs/m8_split_barrier.json \\
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _client import complete, complete_mock, token_ids  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import MockMetrics, delta, metrics_available, scrape  # noqa: E402

ITER = "vllm:iteration_tokens_total"
PREFILL = "vllm:request_prefill_time_seconds"


def launch_threadpool(base_url: str, model: str, n: int, seq_len: int,
                      output_len: int, seed: int) -> list[float]:
    """The launcher every previous experiment used. Kept verbatim as the control."""
    def fn(i: int) -> float:
        t = time.perf_counter()
        complete(base_url, model, seq_len, output_len, seed=seed * 100 + i)
        return t
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(fn, range(n)))


def launch_barrier(base_url: str, model: str, n: int, seq_len: int,
                   output_len: int, seed: int, timeout: float = 600.0) -> list[float]:
    """Connect first, then release all n sends from a barrier.

    The point is that everything expensive and variable -- DNS, TCP handshake,
    building the request body, serialising several thousand token ids -- happens
    BEFORE the barrier. After it, each thread does a single `send` of bytes it
    already holds. Arrival spread should fall from milliseconds to microseconds.
    """
    u = urllib.parse.urlparse(base_url)
    gate = threading.Barrier(n)
    sent: list[float] = [0.0] * n

    def fn(i: int) -> None:
        body = json.dumps({"model": model, "prompt": token_ids(seq_len, seed=seed * 100 + i),
                           "max_tokens": output_len, "temperature": 0.0,
                           "stream": True}).encode()
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
        conn.connect()                       # handshake paid before the barrier
        hdrs = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        gate.wait()                          # <- every thread leaves together
        sent[i] = time.perf_counter()
        conn.putrequest("POST", "/v1/completions", skip_host=False, skip_accept_encoding=True)
        for k, v in hdrs.items():
            conn.putheader(k, v)
        conn.endheaders()
        conn.send(body)
        resp = conn.getresponse()
        resp.read()
        conn.close()

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(fn, range(n)))
    return sent


def one_dispatch(launcher, base_url: str, model: str, n: int, seq_len: int,
                 output_len: int, seed: int, mock_metrics=None) -> dict[str, Any]:
    """One dispatch of n requests; reports whether its PREFILL was a single step."""
    before = mock_metrics.snapshot() if mock_metrics else scrape(base_url)
    if mock_metrics is not None:
        for i in range(n):
            complete_mock(seq_len, output_len, ladder=None, staircase=False, seed=seed * 100 + i)
        # Mock the race the real launcher exhibits: the threadpool splits above
        # n=8, the barrier does not. This is what the assertions below detect,
        # so the mock must be able to produce BOTH verdicts.
        splits = 2 if (launcher is launch_threadpool and n > 8) else 1
        for _ in range(output_len):
            mock_metrics.record_iteration(n)
        for _ in range(splits):
            mock_metrics.record_iteration(n * seq_len // splits)
        for _ in range(n):
            mock_metrics.record(0.0169 * seq_len / 1000.0, 0.0)
        after = mock_metrics.snapshot()
        spread_us = 0.0
    else:
        t = launcher(base_url, model, n, seq_len, output_len, seed)
        spread_us = (max(t) - min(t)) * 1e6 if t and max(t) > 0 else float("nan")
        after = scrape(base_url)

    d = delta(before, after)
    pf, it = d.get(PREFILL), d.get(ITER)
    steps = it["count"] if it else float("nan")
    # A CLEAN dispatch is one whose PREFILL ran in a single scheduler step. That
    # is not `steps == 1`: with output_len=1 every request still needs a decode
    # step, so a clean dispatch shows 2. The first hardware run made this obvious
    # -- n=4 reported 0% "single step" at a cell m1_boundary independently
    # measured as never splitting, and `scheduled_tokens` came back as real + n,
    # which is the decode step counted once per request. Comparing a step COUNT
    # against 1 measured the wrong thing; the quantity the paper cares about is
    # whether the prefill was split.
    clean = 1 + output_len
    return {"n_steps": steps, "prefill_steps": steps - output_len,
            "single_step": bool(steps <= clean),
            "prefill_ms": pf["mean_ms"] if pf else float("nan"),
            "n_requests_seen": pf["count"] if pf else 0,
            "scheduled_tokens": it["sum_s"] if it else float("nan"),
            "arrival_spread_us": spread_us}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.mock:
        cfg = dict(cfg); cfg["mode"] = "mock"; cfg["mode_label"] = "mock"
    elif not metrics_available(args.base_url):
        print(f"[m8] no /metrics at {args.base_url}; is the server up?", file=sys.stderr)
        return 1

    try:
        run = start_run("m8_split_barrier", cfg, results_root=args.results_root)
    except ControlledVarError as e:
        print(f"[m8] {e}", file=sys.stderr)
        return 2

    status, err = "ok", None
    try:
        mm = MockMetrics() if args.mock else None
        rows: list[dict[str, Any]] = []
        reps, discard = cfg["repeats"], cfg.get("warmup_discard", 2)
        launchers = [("threadpool", launch_threadpool), ("barrier", launch_barrier)]

        for cell in cfg["cells"]:
            n, seq_len = cell["n"], cell["seq_len"]
            for lname, launcher in launchers:
                for r in range(reps + discard):
                    d = one_dispatch(launcher, args.base_url, cfg["model"], n, seq_len,
                                     cfg["output_len"], seed=1000 + r, mock_metrics=mm)
                    if r < discard:
                        continue
                    rows.append({"launcher": lname, "n": n, "seq_len": seq_len,
                                 "total_tokens": n * seq_len, "rep": r, **d})
        save_table(run, "dispatches", rows)

        summary: list[dict[str, Any]] = []
        print(f"[m8] {'launcher':<12} {'n':>3} {'tokens':>7} {'single-step':>12} "
              f"{'median steps':>13} {'arrival spread':>15}")
        for cell in cfg["cells"]:
            n = cell["n"]
            for lname, _ in launchers:
                g = [r for r in rows if r["launcher"] == lname and r["n"] == n]
                if not g:
                    continue
                frac = sum(r["single_step"] for r in g) / len(g)
                sp = [r["arrival_spread_us"] for r in g if r["arrival_spread_us"] == r["arrival_spread_us"]]
                summary.append({"launcher": lname, "n": n, "seq_len": cell["seq_len"],
                                "total_tokens": n * cell["seq_len"], "reps": len(g),
                                "single_step_frac": frac,
                                "median_steps": statistics.median([r["n_steps"] for r in g]),
                                "median_spread_us": statistics.median(sp) if sp else float("nan")})
                s = summary[-1]
                print(f"[m8] {lname:<12} {n:>3} {s['total_tokens']:>7} "
                      f"{frac * 100:>11.0f}% {s['median_steps']:>13.0f} "
                      f"{s['median_spread_us']:>14.0f}us")
        save_table(run, "summary", summary)

        # --- verdict ---------------------------------------------------------
        above = [s for s in summary if s["n"] > 8]
        tp = {s["n"]: s for s in above if s["launcher"] == "threadpool"}
        bar = {s["n"]: s for s in above if s["launcher"] == "barrier"}
        fixed = [n for n in bar if n in tp and bar[n]["single_step_frac"] > 0.5
                 and tp[n]["single_step_frac"] <= 0.5]
        if fixed:
            print(f"[m8] VERDICT: the barrier was OURS. At n={sorted(fixed)} the barrier "
                  f"launcher produces single-step dispatches where the threadpool does not. "
                  f"'Prefill step cost is unmeasurable above n=8' is a property of the "
                  f"harness, not of the system, and the paper must be corrected. The "
                  f"paid-padding measurement now extends into the production regime.")
        elif above and all(bar[n]["single_step_frac"] <= 0.5 for n in bar):
            print("[m8] VERDICT: the barrier does NOT fix it. Arrivals are synchronised to "
                  "microseconds and the scheduler still splits, so the split is the "
                  "scheduler's own behaviour above n=8. The paper can state that as a "
                  "mechanism rather than reporting the symptom.")
        else:
            print("[m8] VERDICT: mixed -- see the table; neither conclusion is supported "
                  "across all cells above n=8.")
        print(f"[m8] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
