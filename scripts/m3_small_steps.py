#!/usr/bin/env python3
"""
M3 — the cost curve below 512 tokens, and whether decomposing a residual wins.

TWO GAPS, both created by the same missing measurement.

`sim/measured_cost_curve.json`'s lowest knot is 512 tokens. Below that the model
scales linearly from the origin, which is an ASSUMPTION and is documented as
one: no dispatch we ever ran had fewer than 512 tokens in it. Every serving
system has a fixed per-step cost — kernel launch, scheduler bookkeeping, host
sync — and linear-from-origin says that cost is zero.

That assumption decides a proposal. Under chunked prefill only the trailing
partial chunk pays padding: a 1808-token residual rounds to 2048 and wastes 240.
But 1808 = 1024 + 512 + 256 + 16 exactly, so it could be issued as four
zero-padding steps instead. On the extrapolated curve that wins:

    C(2048)                = 39.22 ms
    C(1024+512+256+16)     = 37.37 ms      <- better by 1.85 ms

and it wins ONLY because the model prices a 16-token step at 0.41 ms. If a step
costs 3 ms to launch regardless of size, the decomposition loses outright. The
proposal currently rests on the one part of the curve nobody measured.

WHAT THIS DOES.

  1. Measures single-step cost at 16, 32, 64, 128, 256, 512, 1024 real tokens,
     each exactly on a compiled bucket edge so no padding is involved. Extends
     the curve into the region it was extrapolating, and 512/1024 overlap the
     existing knots as a consistency check.
  2. Measures the decomposition DIRECTLY: one dispatch of 1808 tokens against
     four dispatches of 1024/512/256/16, comparing summed server-side prefill
     time. No model in the loop.

Every dispatch is n=1 so the step's token count IS the request length, and the
step count is verified from `iteration_tokens_total` rather than assumed.

Usage:
  python scripts/m3_small_steps.py --config configs/m3_small_steps.json --mock
  python scripts/m3_small_steps.py --config configs/m3_small_steps.json \
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))

from _client import complete, complete_mock  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import MockMetrics, delta, metrics_available, scrape  # noqa: E402
from cost_model import CostModel  # noqa: E402

PREFILL = "vllm:request_prefill_time_seconds"
ITER = "vllm:iteration_tokens_total"


def one(base_url: str, model: str, tokens: int, output_len: int, seed: int,
        mock_metrics=None, mock_fixed_ms: float = 0.0) -> dict[str, Any]:
    """One n=1 dispatch of exactly `tokens` prompt tokens."""
    before = mock_metrics.snapshot() if mock_metrics else scrape(base_url)
    if mock_metrics is not None:
        complete_mock(tokens, output_len, ladder=None, staircase=False, seed=seed)
        # Mock a server with a FIXED per-step cost plus a linear term, which is
        # the hypothesis the real measurement is meant to test.
        per = mock_fixed_ms + 0.0169 * tokens
        mock_metrics.record(per / 1000.0, 0.0)
        mock_metrics.record_iteration(tokens)
        after = mock_metrics.snapshot()
    else:
        complete(base_url, model, tokens, output_len, seed=seed)
        after = scrape(base_url)
    d = delta(before, after)
    pf, it = d.get(PREFILL), d.get(ITER)
    return {"prefill_ms": pf["mean_ms"] if pf else float("nan"),
            "n_steps": it["count"] if it else float("nan")}


def measure(base_url, model, tokens, olen, repeats, discard, mock_metrics, mock_fixed) -> tuple[float, float]:
    costs, steps = [], []
    for rep in range(-discard, repeats):
        r = one(base_url, model, tokens, olen, rep, mock_metrics, mock_fixed)
        if rep < 0:
            continue
        if r["prefill_ms"] == r["prefill_ms"]:
            costs.append(r["prefill_ms"])
        if r["n_steps"] == r["n_steps"]:
            steps.append(r["n_steps"])
    return (statistics.median(costs) if costs else float("nan"),
            statistics.median(steps) if steps else float("nan"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-fixed-ms", type=float, default=0.0,
                    help="per-step fixed cost the mock server charges; used to prove the "
                         "analysis can detect a launch cost the curve currently assumes away")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["mode"] = "mock" if args.mock else "live"
    olen = cfg.get("output_len", 1)
    repeats = cfg.get("repeats", 9)
    discard = cfg.get("warmup_discard", 3)

    run = start_run("m3_small_steps", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        mm = MockMetrics() if args.mock else None
        if not mm and not metrics_available(args.base_url):
            print("[m3] /metrics unavailable — aborting.", file=sys.stderr)
            return 1
        model = CostModel()

        # --- 1. the curve, including below the lowest knot -----------------
        rows: list[dict[str, Any]] = []
        print("[m3] single-step cost at exact bucket sizes (no padding involved)")
        for t in cfg["token_sizes"]:
            c, st = measure(args.base_url, cfg["model"], t, olen, repeats, discard, mm,
                            args.mock_fixed_ms)
            extrap = model.tokens_cost_ms(t)
            rows.append({"tokens": t, "measured_ms": c, "curve_ms": extrap,
                         "n_steps": st, "us_per_token": c / t * 1000 if c == c else float("nan"),
                         "below_lowest_knot": t < 512})
            flag = "  <- was EXTRAPOLATED" if t < 512 else ""
            print(f"[m3]   {t:>5} tok  measured {c:7.2f} ms  curve says {extrap:7.2f}  "
                  f"{c / t * 1000:6.1f} us/tok  steps {st:.1f}{flag}")
        save_table(run, "curve", rows)

        # Fixed per-step cost: intercept of a line through the two smallest
        # measured points. Linear-from-origin claims this is zero.
        small = [r for r in rows if r["measured_ms"] == r["measured_ms"]][:2]
        fixed = float("nan")
        if len(small) == 2:
            (t0, c0), (t1, c1) = (small[0]["tokens"], small[0]["measured_ms"]), \
                                 (small[1]["tokens"], small[1]["measured_ms"])
            slope = (c1 - c0) / (t1 - t0)
            fixed = c0 - slope * t0
            print(f"[m3] implied FIXED per-step cost = {fixed:.2f} ms "
                  f"(the curve's linear-from-origin assumes 0.00)")

        # --- 2. the decomposition, measured rather than modelled -----------
        dec = cfg.get("decomposition")
        verdict_rows = []
        if dec:
            whole, parts = dec["whole_tokens"], dec["parts"]
            c_whole, _ = measure(args.base_url, cfg["model"], whole, olen, repeats, discard, mm,
                                 args.mock_fixed_ms)
            c_parts, part_costs = 0.0, []
            for p in parts:
                c, _ = measure(args.base_url, cfg["model"], p, olen, repeats, discard, mm,
                               args.mock_fixed_ms)
                part_costs.append(c); c_parts += c
            print(f"[m3] decomposition: {whole} tokens as ONE step vs {'+'.join(map(str, parts))}")
            print(f"[m3]   one step   {c_whole:7.2f} ms  (pads to {model.tokens_cost_ms(2048):.0f}-token bucket)")
            print(f"[m3]   decomposed {c_parts:7.2f} ms  = " +
                  " + ".join(f"{c:.2f}" for c in part_costs))
            win = c_whole - c_parts
            print(f"[m3]   -> decomposition is {'BETTER' if win > 0 else 'WORSE'} by {abs(win):.2f} ms "
                  f"({abs(win) / c_whole * 100:.1f}%)")
            verdict_rows.append({"whole_tokens": whole, "parts": str(parts),
                                 "cost_whole_ms": c_whole, "cost_parts_ms": c_parts,
                                 "decomposition_wins": win > 0, "delta_ms": win,
                                 "fixed_step_cost_ms": fixed})
            if win <= 0:
                print("[m3] The proposal is rejected on measurement: per-step cost exceeds the "
                      "padding it avoids. The model said otherwise only because it priced small "
                      "steps by extrapolation.")
        save_table(run, "verdict", verdict_rows)
        print(f"[m3] run_id={run.run_id}")
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
