#!/usr/bin/env python3
"""
e05 — what makes a dispatch expensive? The per-step token distribution.

THE OPEN QUESTION. Per-dispatch cost is bimodal at n=9..14: two clean modes,
~1.6x apart, at identical concurrency. It is the paper's own subject appearing
in its baseline — whether stock pays the padding looks stochastic — and three
explanations have already been eliminated from captured data:

  total scheduled tokens   IDENTICAL in both modes (n=12: 6156 = 6156)
  step count               identical at n=12 and n=14 (3 = 3)
  drift / warmup / thermal  a runs test matches randomness almost exactly
                            (n=10: 6 runs observed, 6.1 expected), so it is a
                            per-dispatch coin flip, not a slow trend

What remains is HOW the scheduler distributed the same tokens across the same
number of steps. A count cannot distinguish 2 prefill steps + 1 decode from
1 prefill + 2 decode; the histogram BUCKETS of `vllm:iteration_tokens_total`
can, because they record tokens-per-step.

THE HYPOTHESIS THIS TESTS. vLLM pads a step's token count up to a compiled
bucket. Packing 5120 prefill tokens into ONE step pads to 8192 and costs like
8192; splitting them 4096 + 1024 pays two smaller buckets and costs less. If
true, the expensive mode should show a step in a HIGHER token bucket than the
cheap mode, even though both scheduled the same tokens in total.

If that is what the buckets show, the bimodality is not noise: it is the token
ladder being paid or avoided depending on how the scheduler happened to pack the
step. That would make it the same phenomenon the paper is about, observed in the
baseline rather than in the policy — which is a result, not a nuisance.

Usage:
  python scripts/e05_step_shape.py --config configs/e05_step_shape.json --mock
  python scripts/e05_step_shape.py --config configs/e05_step_shape.json \
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _client import complete, complete_mock  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import (MockMetrics, bucket_delta, delta, metrics_available,  # noqa: E402
                      scrape, scrape_buckets)

ITER = "vllm:iteration_tokens_total"
PREFILL = "vllm:request_prefill_time_seconds"


def one_dispatch(base_url: str, model: str, n: int, prompt_len: int, output_len: int,
                 seed: int, mock_metrics=None) -> dict[str, Any]:
    """Fire n requests together; record cost AND the per-step token histogram."""
    if mock_metrics is not None:
        before, before_b = mock_metrics.snapshot(), mock_metrics.bucket_snapshot(ITER)
    else:
        before, before_b = scrape(base_url), scrape_buckets(base_url, ITER)

    def fn(i: int):
        if mock_metrics is not None:
            s = complete_mock(prompt_len, output_len, ladder=None, staircase=False,
                              seed=seed * 100 + i)
            return s
        return complete(base_url, model, prompt_len, output_len, seed=seed * 100 + i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(fn, range(n)))

    if mock_metrics is not None:
        # Mock the hypothesis so the analysis path is exercised offline: on odd
        # seeds pack everything into one big step (expensive), otherwise split.
        big = seed % 2 == 1
        steps = [n * prompt_len] if big else [4096, n * prompt_len - 4096]
        steps = [s for s in steps if s > 0]
        for s in steps:
            mock_metrics.record_iteration(s)
        mock_metrics.record_iteration(n * output_len)
        per = 140.0 if big else 75.0
        for _ in range(n):
            mock_metrics.record(per / 1000.0, 0.0)
        after, after_b = mock_metrics.snapshot(), mock_metrics.bucket_snapshot(ITER)
    else:
        after, after_b = scrape(base_url), scrape_buckets(base_url, ITER)

    d = delta(before, after)
    pf = d.get(PREFILL)
    steps_hist = bucket_delta(before_b, after_b)
    # Largest bucket edge that actually got a step. This is the discriminator:
    # one packed step lands in a high bucket, two split steps do not.
    largest = max((e for e in steps_hist), default=float("nan"))
    return {
        "n": n,
        "prefill_ms": pf["mean_ms"] if pf else float("nan"),
        "n_steps": sum(steps_hist.values()),
        "largest_step_bucket": largest,
        "hist": {str(k): v for k, v in sorted(steps_hist.items())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["mode"] = "mock" if args.mock else "live"
    plen = cfg.get("prompt_len", 512)
    olen = cfg.get("output_len", 1)
    repeats = cfg.get("repeats", 25)
    discard = cfg.get("warmup_discard", 2)
    levels = cfg.get("concurrency", [8, 10, 12, 14, 16])

    run = start_run("e05_step_shape", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        mock_metrics = MockMetrics() if args.mock else None
        if not mock_metrics and not metrics_available(args.base_url):
            print("[e05] /metrics unavailable — this experiment IS a metrics read. Aborting.",
                  file=sys.stderr)
            return 1

        rows: list[dict[str, Any]] = []
        for n in levels:
            for rep in range(-discard, repeats):
                r = one_dispatch(args.base_url, cfg["model"], n, plen, olen, rep, mock_metrics)
                if rep < 0:
                    continue
                rows.append({"repeat": rep, **{k: v for k, v in r.items() if k != "hist"},
                             "hist": str(r["hist"])})
        save_table(run, "dispatches", rows)

        print("[e05] cost mode vs the largest token bucket any step landed in")
        summary: list[dict[str, Any]] = []
        for n in levels:
            sub = [r for r in rows if r["n"] == n and r["prefill_ms"] == r["prefill_ms"]]
            if len(sub) < 4:
                continue
            costs = [r["prefill_ms"] for r in sub]
            mid = (min(costs) + max(costs)) / 2
            lo = [r for r in sub if r["prefill_ms"] < mid]
            hi = [r for r in sub if r["prefill_ms"] >= mid]
            if not lo or not hi:
                print(f"[e05]   n={n:>3}: unimodal ({statistics.median(costs):.1f} ms), no split")
                continue

            def med(rs, k):
                v = [r[k] for r in rs if r[k] == r[k]]
                return statistics.median(v) if v else float("nan")

            row = {"n": n,
                   "lo_n": len(lo), "lo_cost_ms": med(lo, "prefill_ms"),
                   "lo_steps": med(lo, "n_steps"), "lo_largest_bucket": med(lo, "largest_step_bucket"),
                   "hi_n": len(hi), "hi_cost_ms": med(hi, "prefill_ms"),
                   "hi_steps": med(hi, "n_steps"), "hi_largest_bucket": med(hi, "largest_step_bucket")}
            summary.append(row)
            print(f"[e05]   n={n:>3}  LOW  {row['lo_cost_ms']:6.1f} ms  steps {row['lo_steps']:.1f}  "
                  f"largest step bucket {row['lo_largest_bucket']:>8.0f}   (x{len(lo)})")
            print(f"[e05]        HIGH {row['hi_cost_ms']:6.1f} ms  steps {row['hi_steps']:.1f}  "
                  f"largest step bucket {row['hi_largest_bucket']:>8.0f}   (x{len(hi)})")
        save_table(run, "modes", summary)

        # Verdict: does the expensive mode consistently use a bigger token bucket?
        decided = [s for s in summary
                   if s["hi_largest_bucket"] == s["hi_largest_bucket"]
                   and s["lo_largest_bucket"] == s["lo_largest_bucket"]]
        bigger = [s for s in decided if s["hi_largest_bucket"] > s["lo_largest_bucket"]]
        same = [s for s in decided if s["hi_largest_bucket"] == s["lo_largest_bucket"]]
        if decided and len(bigger) >= max(1, len(decided) - 1):
            print("[e05] EXPLAINED: the expensive mode packs tokens into a step that lands in a "
                  "HIGHER compiled token bucket. The bimodality is the token ladder being paid "
                  "or avoided, depending on how the scheduler happened to pack the step.")
        elif len(same) == len(decided) and decided:
            print("[e05] NOT EXPLAINED: both modes use the same token buckets, so identical "
                  "tokens in identically-shaped steps still cost 1.6x apart. The cause is "
                  "outside the scheduler's accounting — host-side stall inside the measured "
                  "prefill window is the remaining candidate, and it would mean "
                  "request_prefill_time_seconds is not a pure TPU-busy measure (solidity.md R1).")
        else:
            print(f"[e05] MIXED: {len(bigger)}/{len(decided)} levels show a higher bucket in the "
                  "expensive mode. Not a clean mechanism; report as unresolved.")
        save_table(run, "verdict", [{"levels": len(decided), "higher_bucket_when_expensive": len(bigger),
                                     "same_bucket": len(same)}])
        print(f"[e05] run_id={run.run_id}")
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
