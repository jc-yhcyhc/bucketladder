#!/usr/bin/env python3
"""
M10 — is 35.9% padding a property of the STACK, or of the length distribution?

Review finding M5: "Synthetic uniform generation, output_len=1 for most cost
work, no arrival process, no length distribution drawn from a trace, no
burstiness. Consequences: (i) the 35.9% padding figure is a property of the
chosen length distribution, not of the stack... a trace-driven distribution with
continuous batching could plausibly halve or double it. (ii) output_len=1 means
the decode results are measured under a workload with essentially no
decode-dominant steady state. (iii) For a paper about serving, TTFT and ITL
should appear somewhere."

All three are right, and (i) is the one that touches an abstract number. Padding
share under a geometric ladder is decided almost entirely by where a workload's
lengths fall relative to bucket boundaries, so a single fixed-length workload can
land anywhere between best and worst case. Reporting one such number as a stack
characteristic is not defensible.

WHAT THIS MEASURES. The same padding-share instrument as H1 -- per-step
`iteration_tokens_total` bucket edges against real tokens -- but under workloads
that vary in the way that actually moves it:

  fixed-256     the paper's current workload, as the control
  lognormal     sigma from the config; the shape ShareGPT-like prompt lengths
                have, without claiming to BE a trace we do not have
  uniform       maximum spread across the ladder, as an upper-bound probe
  bimodal       short chat turns plus long context, the agentic shape

with Poisson arrivals at a configured rate rather than a closed loop, and
`output_len` large enough that decode has a steady state. Each request's TTFT and
inter-token latency are recorded, so the paper can report the two numbers a
serving paper is expected to report and currently does not.

WHAT IT CANNOT DO. These are parametric families chosen to span the plausible
range, not a production trace -- we have none, and the paper says so. The result
is therefore a RANGE and a sensitivity, not a corrected point estimate. If the
range is wide, the honest reporting of 35.9% is "35.9% under a fixed-256
workload; X-Y% across plausible length distributions," and the abstract must
carry the range.

Usage:
  python scripts/m10_trace_workload.py --config configs/m10_trace_workload.json --mock
  python scripts/m10_trace_workload.py --config configs/m10_trace_workload.json \\
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _client import complete, complete_mock  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import (MockMetrics, bucket_delta, metrics_available,  # noqa: E402
                      scrape_buckets)

ITER = "vllm:iteration_tokens_total"


def draw_lengths(kind: str, k: int, p: dict[str, Any], rng: random.Random) -> list[int]:
    """Prompt lengths from one of four families, clamped to the served range."""
    lo, hi = p["min_len"], p["max_len"]
    out = []
    for _ in range(k):
        if kind == "fixed":
            v = p["fixed_len"]
        elif kind == "lognormal":
            v = int(math.exp(rng.gauss(math.log(p["median_len"]), p["sigma"])))
        elif kind == "uniform":
            v = rng.randint(lo, hi)
        elif kind == "bimodal":
            v = (int(rng.gauss(p["short_len"], p["short_len"] * 0.2)) if rng.random() < p["short_frac"]
                 else int(rng.gauss(p["long_len"], p["long_len"] * 0.2)))
        else:
            raise ValueError(f"unknown length distribution {kind!r}")
        out.append(max(lo, min(hi, v)))
    return out


def ladder_padded(hist: dict[float, float]) -> float:
    """Executed tokens: the reporting bucket edge IS the executed step size.

    Both the token ladder and vLLM's `iteration_tokens_total` bucket edges are
    powers of two, so a step's reporting edge is the size it actually executed
    at. The +Inf bucket is dropped rather than guessed at: it can only be
    populated by a step above the largest compiled shape, which cannot happen.
    """
    return sum(e * c for e, c in hist.items() if e != float("inf"))


def run_arm(kind: str, cfg: dict[str, Any], base_url: str, rng: random.Random,
            mock: MockMetrics | None) -> tuple[list[dict], dict[str, Any]]:
    p = cfg["lengths"]
    n_req, rate = cfg["requests_per_arm"], cfg["arrival_rate_rps"]
    lens = draw_lengths(kind, n_req, p, rng)
    # Poisson arrivals: exponential gaps. A closed loop cannot produce queueing,
    # and queueing is what decides how steps get packed.
    gaps = [rng.expovariate(rate) for _ in range(n_req)]
    before = (mock.bucket_snapshot(ITER) if mock else scrape_buckets(base_url, ITER))

    samples: list[dict[str, Any]] = []
    lock = threading.Lock()
    t0 = time.perf_counter()

    def fire(i: int) -> None:
        due = t0 + sum(gaps[:i + 1])
        d = due - time.perf_counter()
        if d > 0:
            time.sleep(d)
        if mock is not None:
            s = complete_mock(lens[i], cfg["output_len"], ladder=None, staircase=False, seed=i)
            mock.record(s.ttft_ms / 1000.0, (s.total_ms - s.ttft_ms) / 1000.0)
            # Round to the compiled ladder, which is powers of two. Recording the
            # RAW token count would model a stack that pads nothing, so the mock
            # could never exercise the padding arithmetic it exists to check --
            # it reported a negative padding share, which is not a reachable state.
            # The mock must emit the same STEPS a server would, not one per
            # request: a prefill step at the prompt's compiled bucket, then one
            # decode step per output token at the smallest bucket. Recording only
            # the prefill made padded tokens fall BELOW real tokens and reported a
            # negative padding share, which is not a reachable state -- the mock
            # was failing to exercise the arithmetic it exists to check.
            mock.record_iteration(1 << max(4, (lens[i] - 1).bit_length()))
            for _ in range(cfg["output_len"]):
                mock.record_iteration(16)
        else:
            s = complete(base_url, cfg["model"], lens[i], cfg["output_len"], seed=i)
        if not s.ok:
            return
        itl = ((s.total_ms - s.ttft_ms) / max(1, cfg["output_len"] - 1)
               if cfg["output_len"] > 1 else float("nan"))
        with lock:
            samples.append({"arm": kind, "idx": i, "prompt_len": lens[i],
                            "ttft_ms": s.ttft_ms, "e2e_ms": s.total_ms, "itl_ms": itl})

    with ThreadPoolExecutor(max_workers=cfg["max_inflight"]) as pool:
        list(pool.map(fire, range(n_req)))
    wall = time.perf_counter() - t0
    after = (mock.bucket_snapshot(ITER) if mock else scrape_buckets(base_url, ITER))

    hist = bucket_delta(before, after)
    padded = ladder_padded(hist) if hist else float("nan")
    real = sum(lens) + len(samples) * cfg["output_len"]
    ttfts = sorted(s["ttft_ms"] for s in samples)
    itls = sorted(s["itl_ms"] for s in samples if s["itl_ms"] == s["itl_ms"])

    def pct(v, q):
        return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")

    return samples, {
        "arm": kind, "requests": len(samples), "wall_s": wall,
        "achieved_rps": len(samples) / wall if wall else float("nan"),
        "mean_prompt_len": statistics.fmean(lens), "cv_prompt_len":
            (statistics.pstdev(lens) / statistics.fmean(lens)) if len(lens) > 1 else 0.0,
        "tokens_real": real, "tokens_padded": padded,
        "padding_pct_of_executed": (padded - real) / padded * 100 if padded == padded else float("nan"),
        "ttft_p50_ms": pct(ttfts, 0.5), "ttft_p95_ms": pct(ttfts, 0.95),
        "itl_p50_ms": pct(itls, 0.5), "itl_p95_ms": pct(itls, 0.95),
    }


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
        print(f"[m10] no /metrics at {args.base_url}", file=sys.stderr)
        return 1
    try:
        run = start_run("m10_trace_workload", cfg, results_root=args.results_root)
    except ControlledVarError as e:
        print(f"[m10] {e}", file=sys.stderr)
        return 2

    status, err = "ok", None
    try:
        rng = random.Random(cfg["seed"])
        mock = MockMetrics() if args.mock else None
        all_s: list[dict] = []
        summ: list[dict] = []
        for kind in cfg["distributions"]:
            s, agg = run_arm(kind, cfg, args.base_url, rng, mock)
            all_s.extend(s); summ.append(agg)
        save_table(run, "requests", all_s)
        save_table(run, "arms", summ)

        print(f"[m10] {'arm':<11} {'reqs':>5} {'CV':>6} {'rps':>6} {'padding%':>9} "
              f"{'TTFT p50':>9} {'TTFT p95':>9} {'ITL p50':>8} {'ITL p95':>8}")
        for a in summ:
            print(f"[m10] {a['arm']:<11} {a['requests']:>5} {a['cv_prompt_len']:>6.2f} "
                  f"{a['achieved_rps']:>6.2f} {a['padding_pct_of_executed']:>8.1f}% "
                  f"{a['ttft_p50_ms']:>8.1f}ms {a['ttft_p95_ms']:>8.1f}ms "
                  f"{a['itl_p50_ms']:>7.1f}ms {a['itl_p95_ms']:>7.1f}ms")

        vals = [a["padding_pct_of_executed"] for a in summ
                if a["padding_pct_of_executed"] == a["padding_pct_of_executed"]]
        if len(vals) >= 2:
            print(f"[m10] padding share spans {min(vals):.1f}% to {max(vals):.1f}% of executed "
                  f"tokens across {len(vals)} length distributions at the same arrival rate.")
            fixed = next((a for a in summ if a["arm"] == "fixed"), None)
            if fixed and max(vals) - min(vals) > 5:
                print(f"[m10] VERDICT: the review is right. 35.9% is a property of the length "
                      f"distribution, not of the stack. The abstract must report the range, "
                      f"and the control (fixed) sits at "
                      f"{fixed['padding_pct_of_executed']:.1f}% within it.")
            else:
                print("[m10] VERDICT: padding share is stable across length distributions, so "
                      "reporting a single figure is defensible -- state the range anyway.")
        print(f"[m10] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
