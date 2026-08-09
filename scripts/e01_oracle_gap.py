#!/usr/bin/env python3
"""
e01 — marginal cost of padding. THE load-bearing measurement of this project.

The question, in one line: **when a request of length L is served in a compiled
bucket B > L, does it cost what L costs, or what B costs?**

If it costs what B costs, padding is real, the ladder matters, and there is a
paper. If it costs what L costs, padding is free on this hardware and BOTH the
ladder contribution and the admission contribution die at once. Nothing else in
the plan is worth measuring until this is settled.

Design (notes/session_plan.md, revised): hold ONE bucket and vary occupancy —
serve lengths L = {1.0, 0.9, 0.75, 0.5, 0.25} x B and compare TTFT. The earlier
design compiled an exact-shape ladder over hundreds of trace lengths, which was
hours of XLA warmup to measure a bound. This is ~5 shapes inside an already-warm
ladder, fits in a session, and asks the question directly.

Interpretation, expressed in units of the measured noise floor (e03):
  staircase  TTFT(0.25B) ~= TTFT(B)     -> padding is fully paid. Ladder matters.
  linear     TTFT(0.25B) ~= 0.25 TTFT(B)-> padding is free. Both claims die.
  partial    in between                 -> quantify; the cost model is C(B)-C(L).

A "flatness" statistic makes this a number rather than an eyeball:
    flatness = 1 - (TTFT(L_min) - intercept) / (TTFT(B) - intercept)
  ~1.0 => pure staircase, ~0.0 => pure linear.

Usage:
  python scripts/e01_oracle_gap.py --config configs/e01_marginal_cost.json --mock
  python scripts/e01_oracle_gap.py --config configs/e01_marginal_cost.json \
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _client import Sample, complete, complete_mock, summarise  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from ladder import bucket_for, build_ladder  # noqa: E402

# WARMUP DISCARD. Measured on hardware 2026-08-09: the first request after a
# server start costs ~116 ms against a ~15.4 ms steady state (7.5x), even though
# vLLM has already logged "Application startup complete". Including it puts the
# run-to-run CV at 97%; discarding it gives 1.7%, and discarding 5 gives 0.8%.
# Every cell therefore fires `warmup_discard` unrecorded requests first.


# ACHIEVABLE OCCUPANCY IS BOUNDED BY THE LADDER ITSELF, discovered while
# testing this script. On the power-of-two default ladder, bucket B spans
# (B/2, B], so occupancy can only be varied over a 2x range — 0.5B and 0.25B
# land in *smaller* buckets and would compare two different executables.
# Gap ladders are worse: with gap=512, bucket 4096 spans (3584, 4096], barely
# 0.875-1.0. So the DEFAULT (power-of-two) ladder gives the widest dynamic
# range for this measurement, and e01 should be run against it.
DEFAULT_FRACTIONS = [1.0, 0.95, 0.85, 0.75, 0.65, 0.55]


def occupancy_lengths(bucket: int, fractions: list[float], ladder: list[int]) -> list[int]:
    """Lengths inside `bucket` at the given occupancies.

    Every returned length must map back to `bucket` — otherwise we would be
    comparing across two different executables and measuring nothing. Lengths
    that fall into a smaller bucket are dropped, and the caller is told.
    """
    out: list[int] = []
    for f in fractions:
        L = max(1, int(round(bucket * f)))
        if bucket_for(L, ladder) == bucket:
            out.append(L)
    return sorted(set(out))


def flatness(by_len: dict[int, float], bucket: int) -> float:
    """1.0 = cost independent of true length (staircase); 0.0 = proportional.

    Fits the two extremes through the measured endpoints rather than assuming an
    intercept: the model is cost(L) = a + b*L, and we ask how much of the range
    the shortest request actually saves.
    """
    if len(by_len) < 2:
        return math.nan
    lo_len, hi_len = min(by_len), max(by_len)
    lo, hi = by_len[lo_len], by_len[hi_len]
    if hi <= 0 or hi_len == lo_len:
        return math.nan
    # Proportional prediction for the short request, anchored at the long one.
    predicted_linear = hi * (lo_len / hi_len)
    if hi - predicted_linear == 0:
        return math.nan
    return (lo - predicted_linear) / (hi - predicted_linear)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-linear", action="store_true",
                    help="mock the NULL hypothesis (cost ~ true length) to prove the analysis discriminates")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    config = load_config(args.config)
    config["mode"] = "mock" if args.mock else "live"
    controlled = config["controlled"]
    ladder = build_ladder(controlled["max_model_len"], controlled["VLLM_TPU_BUCKET_PADDING_GAP"])

    buckets = config.get("buckets") or [b for b in ladder if 256 <= b <= 4096]
    fractions = config.get("fractions", DEFAULT_FRACTIONS)
    repeats = config.get("repeats", 5)
    output_len = config.get("output_len", 1)
    discard = config.get("warmup_discard", 2)

    run = start_run("e01_oracle_gap", config, results_root=args.results_root)
    status, err = "ok", None
    try:
        rows: list[dict[str, Any]] = []
        for bucket in buckets:
            lengths = occupancy_lengths(bucket, fractions, ladder)
            if len(lengths) < 2:
                print(f"[e01] bucket {bucket}: <2 usable occupancies, skipping", file=sys.stderr)
                continue
            for L in lengths:
                for rep in range(-discard, repeats):
                    if args.mock:
                        s = complete_mock(L, output_len, ladder=ladder,
                                          staircase=not args.mock_linear, seed=rep)
                    else:
                        s = complete(args.base_url, config["model"], L, output_len, seed=rep)
                    if rep < 0:
                        continue  # warmup, not recorded
                    rows.append({
                        "bucket": bucket, "prompt_len": L, "occupancy": L / bucket,
                        "repeat": rep, **s.as_row(),
                    })
        save_table(run, "samples", rows)

        # --- per-bucket summary + the statistic that decides the project ---
        summary: list[dict[str, Any]] = []
        for bucket in sorted({r["bucket"] for r in rows}):
            by_len: dict[int, float] = {}
            for L in sorted({r["prompt_len"] for r in rows if r["bucket"] == bucket}):
                samples = [Sample(**{k: r[k] for k in Sample.__dataclass_fields__})
                           for r in rows if r["bucket"] == bucket and r["prompt_len"] == L]
                st = summarise(samples)
                by_len[L] = st["median"]
                summary.append({"bucket": bucket, "prompt_len": L,
                                "occupancy": L / bucket, **st})
            f = flatness(by_len, bucket)
            print(f"[e01] bucket {bucket:>5}: "
                  + "  ".join(f"L={L}:{by_len[L]:.1f}ms" for L in sorted(by_len))
                  + f"   flatness={f:.2f}")
        save_table(run, "summary", summary)

        overall = {}
        for bucket in sorted({r["bucket"] for r in rows}):
            by_len = {s["prompt_len"]: s["median"] for s in summary if s["bucket"] == bucket}
            overall[bucket] = flatness(by_len, bucket)
        save_table(run, "flatness", [{"bucket": b, "flatness": v} for b, v in overall.items()])

        vals = [v for v in overall.values() if not math.isnan(v)]
        if vals:
            med = sorted(vals)[len(vals) // 2]
            print(f"[e01] median flatness across buckets = {med:.2f}")
            print("[e01] ~1.0 => padding fully paid (ladder matters); "
                  "~0.0 => padding free (both contributions die)")
        print(f"[e01] run_id={run.run_id}")
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
