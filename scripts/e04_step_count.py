#!/usr/bin/env python3
"""
e04 — why is the batch cost bimodal? Counts scheduler steps per dispatch.

e02 at 21 repeats found that for n = 9..14 concurrent requests the per-batch
prefill cost has TWO modes, not one:

    n= 9   low ~87 ms (38% of runs)   high ~138 ms (62%)
    n=10   low ~89 ms (52%)           high ~140 ms (48%)
    n=14   low ~112 ms (5%)           high ~152 ms (95%)

The high mode is what a batch padded up to 16 request-slots should cost. The
low mode continues the n<=8 trend. So on any given dispatch the server either
pays the padding or does not, and the probability that it does rises with n.

That is not a nuisance — **it is the paper's own subject appearing inside the
baseline.** If stock vLLM sometimes avoids the promotion cost by itself, the
admission policy's headline is not "62 ms saved per promotion" but "62 ms
saved on the fraction of dispatches that would have promoted", which is a
different and smaller number. It has to be measured, not assumed.

Two explanations survive e02's own data (which showed all n requests accounted
for, and queue time <= 0.04 ms, ruling out anything waiting for a later batch):

    ONE STEP   all n prefilled together, padded up to the next request bucket.
               Expensive. Should be 1 prefill iteration.
    TWO STEPS  the scheduler splits them, each part fitting a smaller bucket.
               Cheap. Should be 2 prefill iterations.

`vllm:iteration_tokens_total` is a histogram over engine iterations, so its
`_count` delta IS the number of scheduler steps and its `_sum` delta is the
tokens they scheduled. One extra iteration is the entire difference between the
two hypotheses, and nothing else on /metrics separates them.

output_len=1 deliberately: every decode step is also an iteration, so a longer
output buries the one iteration that carries the signal.

Usage:
  python scripts/e04_step_count.py --config configs/e04_step_count.json --mock
  python scripts/e04_step_count.py --config configs/e04_step_count.json \
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _client import complete, complete_mock  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import MockMetrics, delta, metrics_available, scrape  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

ITER = "vllm:iteration_tokens_total"
PREFILL = "vllm:request_prefill_time_seconds"


def one_dispatch(base_url: str, model: str, n: int, prompt_len: int, output_len: int,
                 seed: int, mock_metrics=None) -> dict[str, Any]:
    """Fire n requests simultaneously and measure the steps they took."""
    before = mock_metrics.snapshot() if mock_metrics else scrape(base_url)

    def fn(i: int):
        if mock_metrics is not None:
            s = complete_mock(prompt_len, output_len, ladder=None, staircase=False, seed=seed * 100 + i)
            mock_metrics.record(s.ttft_ms / 1000.0, 0.0)
            return s
        return complete(base_url, model, prompt_len, output_len, seed=seed * 100 + i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(fn, range(n)))

    if mock_metrics is not None:
        # Mock the hypothesis under test so the analysis path is exercised
        # before it meets a live server: split the dispatch above 8 on odd
        # seeds, keep it whole otherwise. Arbitrary, and deliberately so —
        # what is being tested here is that e04 can SEE a varying step count,
        # not what the real server does.
        steps = 2 if (n > 8 and seed % 2 == 1) else 1
        for _ in range(steps):
            mock_metrics.record_iteration(n * prompt_len / steps)
        mock_metrics.record_iteration(n * output_len)  # decode

    after = mock_metrics.snapshot() if mock_metrics else scrape(base_url)
    d = delta(before, after)
    it, pf = d.get(ITER), d.get(PREFILL)
    return {
        "n": n,
        "prefill_ms": pf["mean_ms"] if pf else float("nan"),
        "n_requests": pf["count"] if pf else 0,
        # The discriminator.
        "iterations": it["count"] if it else float("nan"),
        "scheduled_tokens": it["sum_s"] if it else float("nan"),
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
    repeats = cfg.get("repeats", 21)
    discard = cfg.get("warmup_discard", 2)
    levels = cfg.get("concurrency", [8, 9, 10, 12, 14, 16])

    run = start_run("e04_step_count", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        mock_metrics = MockMetrics() if args.mock else None
        if not mock_metrics and not metrics_available(args.base_url):
            print("[e04] /metrics unavailable — this experiment IS a metrics read. Aborting.",
                  file=sys.stderr)
            return 1

        rows: list[dict[str, Any]] = []
        for n in levels:
            for rep in range(-discard, repeats):
                r = one_dispatch(args.base_url, cfg["model"], n, plen, olen, rep, mock_metrics)
                if rep < 0:
                    continue
                rows.append({"repeat": rep, **r})
        save_table(run, "dispatches", rows)

        print("[e04] iterations per dispatch, and the cost that came with each count")
        summary: list[dict[str, Any]] = []
        for n in levels:
            sub = [r for r in rows if r["n"] == n and r["iterations"] == r["iterations"]]
            if not sub:
                continue
            counts = Counter(int(r["iterations"]) for r in sub)
            parts = []
            for it in sorted(counts):
                costs = [r["prefill_ms"] for r in sub if int(r["iterations"]) == it]
                parts.append(f"{it} iter x{counts[it]:<2} -> {statistics.median(costs):6.1f} ms")
                summary.append({"n": n, "iterations": it, "n_dispatches": counts[it],
                                "median_prefill_ms": statistics.median(costs),
                                "median_scheduled_tokens": statistics.median(
                                    [r["scheduled_tokens"] for r in sub if int(r["iterations"]) == it])})
            print(f"[e04]   n={n:>3}:  " + "   |   ".join(parts))
        save_table(run, "by_iterations", summary)

        # --- verdict -------------------------------------------------------
        # The question is NOT "does the step count vary" — an earlier version
        # asked that and declared the mechanism confirmed, which is wrong.
        # A varying step count at a level whose cost is unimodal proves the
        # opposite: the metric moves without the cost moving. The real test is
        # whether cost SEPARATES by step count, and it has to fail on the
        # controls (`control_levels`, unimodal in e02) to mean anything.
        controls = set(cfg.get("control_levels", []))
        spread_rows: list[dict[str, Any]] = []
        for n in levels:
            ss = sorted((s for s in summary if s["n"] == n), key=lambda x: x["iterations"])
            if len(ss) < 2:
                continue
            costs = [s["median_prefill_ms"] for s in ss]
            rel = (max(costs) - min(costs)) / max(costs)
            spread_rows.append({"n": n, "is_control": n in controls,
                                "n_step_counts": len(ss), "cost_spread_frac": rel})
            print(f"[e04]   n={n:>3}: {len(ss)} distinct step counts, "
                  f"cost varies {rel * 100:4.1f}% across them"
                  + ("   <- CONTROL (e02 says unimodal)" if n in controls else ""))

        SEPARATES = 0.25   # a real mode gap in e02 is ~1.5x, i.e. ~35%
        ctrl = [r["cost_spread_frac"] for r in spread_rows if r["is_control"]]
        test = [r["cost_spread_frac"] for r in spread_rows if not r["is_control"]]
        explains = bool(test) and statistics.fmean(test) > SEPARATES and (
            not ctrl or max(ctrl) < SEPARATES)
        if explains:
            print("[e04] MECHANISM CONFIRMED: cost separates by step count on the test "
                  "levels and not on the controls. The cheap mode is the scheduler "
                  "splitting the batch, so stock sometimes avoids the promotion cost "
                  "without being asked to.")
        else:
            why = ("the controls separate too, so step count is tracking something "
                   "other than the cost modes" if ctrl and max(ctrl) >= SEPARATES
                   else "cost does not separate by step count on the test levels")
            print(f"[e04] MECHANISM REJECTED: {why}. Batch splitting does NOT explain "
                  "e02's bimodality; the cause lies inside a single step.")
        save_table(run, "verdict", [{"explains_bimodality": explains,
                                     "mean_test_spread": statistics.fmean(test) if test else float("nan"),
                                     "max_control_spread": max(ctrl) if ctrl else float("nan"),
                                     "threshold": SEPARATES}])
        save_table(run, "spread", spread_rows)
        print(f"[e04] run_id={run.run_id}")
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
