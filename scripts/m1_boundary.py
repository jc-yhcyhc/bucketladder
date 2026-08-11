#!/usr/bin/env python3
"""
M1 — a designed confirmation that step token padding is not paid at n > 1.

H1 already answered this from captured runs: across 150 dispatches, doubling the
padded token count at fixed real work moved cost by <=2%. But that was a NATURAL
experiment — the scheduler chose the chunk splits, so a lurking variable
correlated with both the split and the cost is not excluded the way
randomisation would exclude it. This is the designed version.

THE STRADDLE. Hold batch size and per-sequence length fixed; move the step's
total token count just across a compiled bucket boundary.

    arm "below"   n=4 x 128 =  512 real tokens -> bucket  512, zero padding
    arm "above"   n=4 x 130 =  520 real tokens -> bucket 1024, 504 padding

Real work differs by **1.6%**. Padded work differs by **100%**.

    cost rises ~2%   -> padding is NOT paid; cost tracks real tokens
    cost rises ~2x   -> padding IS paid; cost tracks the compiled bucket

Per-sequence length is held near-constant (128 vs 130) ON PURPOSE. Comparing,
say, 4x96 against 4x128 would move sequence length by a third, so attention work
would change materially and "proportional to real tokens" would stop being a
clean falsifiable prediction. The arms must differ in almost nothing except
which side of the boundary they land on.

TWO EDGES, so one anomalous cell cannot decide it. Both are far below
`max_num_batched_tokens`, and every dispatch verifies it ran as a SINGLE step
from the `iteration_tokens_total` count delta — the whole point is a step-scoped
test of a step-scoped property, and a dispatch that split silently would
reintroduce exactly the smearing this experiment exists to avoid.

Usage:
  python scripts/m1_boundary.py --config configs/m1_boundary.json --mock
  python scripts/m1_boundary.py --config configs/m1_boundary.json \
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))

from _client import complete, complete_mock  # noqa: F401
from m8_split_barrier import launch_barrier  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import MockMetrics, delta, metrics_available, scrape  # noqa: E402
from ladder import bucket_for, build_ladder  # noqa: E402

PREFILL = "vllm:request_prefill_time_seconds"
ITER = "vllm:iteration_tokens_total"


def one_dispatch(base_url: str, model: str, n: int, seq_len: int, output_len: int,
                 seed: int, mock_metrics=None, mock_pays_padding: bool = False,
                 ladder: list[int] | None = None,
                 synchronised: bool = False) -> dict[str, Any]:
    """One dispatch of n identical requests.

    `synchronised` selects the barrier launcher from M8 instead of the plain
    thread pool. M8 measured why this matters: with the pool, arrivals smear over
    milliseconds and the scheduler splits 100% of dispatches at n=16, so every
    n>8 cell here was discarded as split and the paper reported the quantity as
    unmeasurable. Under a barrier, 40% of n=16 dispatches keep their prefill in
    one step. The launcher is a control, not a detail -- it decides whether this
    experiment has any usable samples in the regime production runs at.
    """
    before = mock_metrics.snapshot() if mock_metrics else scrape(base_url)

    def fn(i: int):
        if mock_metrics is not None:
            return complete_mock(seq_len, output_len, ladder=None, staircase=False,
                                 seed=seed * 100 + i)
        return complete(base_url, model, seq_len, output_len, seed=seed * 100 + i)

    if synchronised and mock_metrics is None:
        launch_barrier(base_url, model, n, seq_len, output_len, seed)
    else:
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(fn, range(n)))

    if mock_metrics is not None:
        real = n * seq_len
        billed = bucket_for(real, ladder) if mock_pays_padding else real
        per = 0.0169 * billed          # 16.9 us/token, the measured rate at n=8
        for _ in range(n):
            mock_metrics.record(per / 1000.0, 0.0)
        mock_metrics.record_iteration(real)
        after = mock_metrics.snapshot()
    else:
        after = scrape(base_url)

    d = delta(before, after)
    pf, it = d.get(PREFILL), d.get(ITER)
    return {"prefill_ms": pf["mean_ms"] if pf else float("nan"),
            "n_requests_seen": pf["count"] if pf else 0,
            "n_steps": it["count"] if it else float("nan"),
            "scheduled_tokens": it["sum_s"] if it else float("nan")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-pays-padding", action="store_true",
                    help="mock a server that DOES pay the padding, to prove the analysis "
                         "can detect it rather than only confirming the expected answer")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["mode"] = "mock" if args.mock else "live"
    controlled = cfg["controlled"]
    ladder = build_ladder(controlled["max_num_batched_tokens"],
                          controlled["VLLM_TPU_BUCKET_PADDING_GAP"])
    olen = cfg.get("output_len", 1)
    repeats = cfg.get("repeats", 3)
    discard = cfg.get("warmup_discard", 2)

    run = start_run("m1_boundary", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        mock_metrics = MockMetrics() if args.mock else None
        if not mock_metrics and not metrics_available(args.base_url):
            print("[m1] /metrics unavailable — aborting.", file=sys.stderr)
            return 1

        rows: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for edge in cfg["edges"]:
            n = edge["n"]
            arms = {}
            for arm in ("below", "above"):
                seq = edge[f"seq_len_{arm}"]
                real = n * seq
                padded = bucket_for(real, ladder)
                costs, splits = [], 0
                for rep in range(-discard, repeats):
                    r = one_dispatch(args.base_url, cfg["model"], n, seq, olen, rep,
                                     mock_metrics, args.mock_pays_padding, ladder,
                                     synchronised=cfg.get("synchronised_launch", False))
                    if rep < 0:
                        continue
                    rows.append({"edge": edge["name"], "arm": arm, "n": n,
                                 "seq_len": seq, "tokens_real": real,
                                 "tokens_padded": padded, "bucket_selected": padded,
                                 "repeat_idx": rep, "step_latency_ms": r["prefill_ms"],
                                 "iteration_tokens_delta": r["scheduled_tokens"],
                                 "n_steps": r["n_steps"]})
                    if r["prefill_ms"] == r["prefill_ms"]:
                        costs.append(r["prefill_ms"])
                    # A prefill split across steps reintroduces the smearing this
                    # experiment exists to avoid, so count it rather than average
                    # over it. output_len adds its own decode step(s).
                    if r["n_steps"] == r["n_steps"] and r["n_steps"] > 1 + olen:
                        splits += 1
                arms[arm] = {"seq": seq, "real": real, "padded": padded,
                             "cost": statistics.median(costs) if costs else float("nan"),
                             "splits": splits, "n": len(costs)}

            b, a = arms["below"], arms["above"]
            real_ratio = a["real"] / b["real"]
            pad_ratio = a["padded"] / b["padded"]
            cost_ratio = a["cost"] / b["cost"]
            results.append({"edge": edge["name"], "n": n,
                            "real_ratio": real_ratio, "padded_ratio": pad_ratio,
                            "cost_ratio": cost_ratio,
                            "cost_below_ms": b["cost"], "cost_above_ms": a["cost"],
                            "splits_below": b["splits"], "splits_above": a["splits"]})
            print(f"[m1] {edge['name']}: n={n}")
            print(f"[m1]   below  {n}x{b['seq']:<4} = {b['real']:>5} real -> bucket {b['padded']:<5} "
                  f"cost {b['cost']:7.2f} ms  (splits {b['splits']}/{b['n']})")
            print(f"[m1]   above  {n}x{a['seq']:<4} = {a['real']:>5} real -> bucket {a['padded']:<5} "
                  f"cost {a['cost']:7.2f} ms  (splits {a['splits']}/{a['n']})")
            print(f"[m1]   real x{real_ratio:.3f}   padded x{pad_ratio:.2f}   "
                  f"-> COST x{cost_ratio:.3f}")
        save_table(run, "dispatches", rows)
        save_table(run, "edges", results)

        split_total = sum(r["splits_below"] + r["splits_above"] for r in results)
        if split_total:
            print(f"[m1] WARNING {split_total} dispatch(es) used more than one prefill step; "
                  "those are smeared and the verdict below is weakened accordingly.",
                  file=sys.stderr)

        # Verdict. "Paid" predicts cost_ratio ~ padded_ratio; "not paid" predicts
        # cost_ratio ~ real_ratio. Score against whichever it lands nearer.
        verdicts = []
        for r in results:
            d_paid = abs(r["cost_ratio"] - r["padded_ratio"])
            d_free = abs(r["cost_ratio"] - r["real_ratio"])
            verdicts.append("paid" if d_paid < d_free else "not_paid")
        paid = verdicts.count("paid")
        print(f"[m1] edges favouring PAID: {paid}/{len(verdicts)}")
        if paid == 0:
            print("[m1] VERDICT: padding is NOT paid at n>1. Cost tracks real tokens across a "
                  "boundary that doubles the padded count — confirming H1's natural experiment "
                  "under randomised assignment.")
        elif paid == len(verdicts):
            print("[m1] VERDICT: padding IS paid at n>1, contradicting H1. The natural "
                  "experiment had a lurking variable and its conclusion must be withdrawn.")
        else:
            print("[m1] VERDICT: edges disagree. Report as unresolved rather than picking one.")
        save_table(run, "verdict", [{"edges_paid": paid, "edges_total": len(verdicts),
                                     "splits": split_total}])
        print(f"[m1] run_id={run.run_id}")
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
