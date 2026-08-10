#!/usr/bin/env python3
"""
e07 — THE GATE. Does a ragged batch cost its padding, or only its tokens?

The project is named for ragged workloads and has never measured one. Every
trace so far gives every request the SAME length (512), so nothing measured to
date bears on raggedness at all. This decides which paper this is, and it should
have run in W0.

THE QUESTION. Take n requests whose lengths differ, holding n and the TOTAL
token count fixed. Three cost models make different predictions:

  PACKED            cost = C(sum of true lengths)
                    Chunked prefill concatenates requests into one step and pads
                    only the step's total token count. Raggedness is then FREE at
                    batch level, and the "ragged" framing dies -- this becomes a
                    batch-size admission paper.

  PER-REQUEST PAD   cost = C(sum of each request's own bucket)
                    Each sequence padded to its own compiled length.

  BATCH PAD         cost = C(n * bucket(longest request))
                    One compiled sequence dimension for the batch, everyone
                    padded up to the longest. The premise the project assumes.

Our own evidence points both ways, which is exactly why it needs measuring.
e01's flatness ~0.97 says a single request pays its whole bucket. But R5 found
`kv_computed` tracks TRUE length, so RPA skips padding inside attention; and
chunked prefill is on, which packs tokens across requests.

DESIGN. Hold n and total tokens fixed, vary only the spread. Uniform cells are
CONTROLS: there, all three models coincide, so a disagreement on those means the
measurement is broken rather than the hypothesis interesting. Ragged cells are
built so the padded total stays within max_num_batched_tokens, keeping the batch
in one scheduler step -- otherwise splitting confounds the comparison.

Usage:
  python scripts/e07_ragged_batch.py --config configs/e07_ragged_batch.json --mock
  python scripts/e07_ragged_batch.py --config configs/e07_ragged_batch.json \
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

from _client import complete, complete_mock  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import (MockMetrics, delta, metrics_available, scrape)  # noqa: E402
from cost_model import CostModel  # noqa: E402
from ladder import bucket_for, build_ladder  # noqa: E402

PREFILL = "vllm:request_prefill_time_seconds"
ITER = "vllm:iteration_tokens_total"


def make_lengths(n: int, total: int, max_len: int, min_len: int = 16) -> list[int]:
    """One long request plus an even split of the remainder.

    Deliberately the simplest ragged shape that isolates the variable: n and the
    total are held exactly, and only the spread moves. A realistic length
    distribution would confound spread with shape.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return [total]
    rest = total - max_len
    if rest < min_len * (n - 1):
        raise ValueError(f"cannot fit {n - 1} requests of >={min_len} into {rest} tokens "
                         f"(n={n}, total={total}, max_len={max_len})")
    base = rest // (n - 1)
    out = [max_len] + [base] * (n - 1)
    out[1] += rest - base * (n - 1)          # put the remainder somewhere exact
    return out


def predictions(lengths: list[int], cost: CostModel, ladder: list[int]) -> dict[str, float]:
    n = len(lengths)
    packed = sum(lengths)
    per_req = sum(bucket_for(x, ladder) for x in lengths)
    batch = n * bucket_for(max(lengths), ladder)
    return {"packed_tokens": packed, "per_request_padded_tokens": per_req,
            "batch_padded_tokens": batch,
            "packed_ms": cost.tokens_cost_ms(packed),
            "per_request_padded_ms": cost.tokens_cost_ms(per_req),
            "batch_padded_ms": cost.tokens_cost_ms(batch)}


def one_dispatch(base_url: str, model: str, lengths: list[int], output_len: int,
                 seed: int, mock_metrics=None, mock_model: str = "packed") -> dict[str, Any]:
    before = mock_metrics.snapshot() if mock_metrics else scrape(base_url)

    def fn(i: int):
        L = lengths[i]
        if mock_metrics is not None:
            return complete_mock(L, output_len, ladder=None, staircase=False, seed=seed * 100 + i)
        return complete(base_url, model, L, output_len, seed=seed * 100 + i)

    with ThreadPoolExecutor(max_workers=len(lengths)) as pool:
        list(pool.map(fn, range(len(lengths))))

    if mock_metrics is not None:
        # Mock whichever hypothesis was requested, so the ANALYSIS is exercised
        # offline against a known answer before it meets hardware.
        from ladder import bucket_for as _b
        lad = build_ladder(8192, "")
        tok = {"packed": sum(lengths),
               "per_request": sum(_b(x, lad) for x in lengths),
               "batch": len(lengths) * _b(max(lengths), lad)}[mock_model]
        per = CostModel().tokens_cost_ms(tok)
        for _ in lengths:
            mock_metrics.record(per / 1000.0, 0.0)
        mock_metrics.record_iteration(tok)
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
    ap.add_argument("--mock-model", default="packed",
                    choices=["packed", "per_request", "batch"],
                    help="which hypothesis the mock server obeys, to prove the analysis "
                         "can tell them apart")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["mode"] = "mock" if args.mock else "live"
    controlled = cfg["controlled"]
    ladder = build_ladder(controlled["max_num_batched_tokens"],
                          controlled["VLLM_TPU_BUCKET_PADDING_GAP"])
    olen = cfg.get("output_len", 1)
    repeats = cfg.get("repeats", 21)
    discard = cfg.get("warmup_discard", 2)
    cells = cfg["cells"]

    run = start_run("e07_ragged_batch", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        cost = CostModel()
        if not cost.knots:
            print("[e07] no measured cost curve; run scripts/refit_cost_model.py --write",
                  file=sys.stderr)
            return 1
        mock_metrics = MockMetrics() if args.mock else None
        if not mock_metrics and not metrics_available(args.base_url):
            print("[e07] /metrics unavailable — aborting.", file=sys.stderr)
            return 1

        rows: list[dict[str, Any]] = []
        summary: list[dict[str, Any]] = []
        for cell in cells:
            n, total, mx = cell["n"], cell["total_tokens"], cell["max_len"]
            lengths = make_lengths(n, total, mx)
            pred = predictions(lengths, cost, ladder)
            is_control = len({bucket_for(x, ladder) for x in lengths}) == 1 and \
                pred["batch_padded_tokens"] == pred["per_request_padded_tokens"]

            costs: list[float] = []
            for rep in range(-discard, repeats):
                r = one_dispatch(args.base_url, cfg["model"], lengths, olen, rep,
                                 mock_metrics, args.mock_model)
                if rep < 0:
                    continue
                rows.append({"n": n, "total_tokens": total, "max_len": mx, "repeat": rep,
                             "lengths": str(lengths), **r})
                if r["prefill_ms"] == r["prefill_ms"]:
                    costs.append(r["prefill_ms"])
            if not costs:
                continue
            obs = statistics.median(costs)
            errs = {k: abs(pred[f"{k}_ms"] - obs) / obs * 100.0
                    for k in ("packed", "per_request_padded", "batch_padded")}
            best = min(errs, key=errs.get)
            summary.append({"n": n, "total_tokens": total, "max_len": mx,
                            "is_control": is_control, "measured_ms": obs,
                            "n_steps": statistics.median([r["n_steps"] for r in rows
                                                          if r["n"] == n and r["max_len"] == mx
                                                          and r["n_steps"] == r["n_steps"]] or [float("nan")]),
                            **pred, **{f"err_{k}_pct": v for k, v in errs.items()},
                            "best_fit": best})
            tag = "  <- CONTROL (all models agree)" if is_control else ""
            print(f"[e07] n={n:<3} total={total:<5} max={mx:<5} lengths={lengths}")
            print(f"[e07]   measured {obs:7.2f} ms |  packed {pred['packed_ms']:7.2f} "
                  f"({errs['packed']:5.1f}%)   per-req-pad {pred['per_request_padded_ms']:7.2f} "
                  f"({errs['per_request_padded']:5.1f}%)   batch-pad {pred['batch_padded_ms']:7.2f} "
                  f"({errs['batch_padded']:5.1f}%)   -> {best}{tag}")
        save_table(run, "dispatches", rows)
        save_table(run, "cells", summary)

        # --- verdict, on the RAGGED cells only ---------------------------
        ragged = [s for s in summary if not s["is_control"]]
        controls = [s for s in summary if s["is_control"]]
        if controls:
            worst_ctrl = max(min(c[f"err_{k}_pct"] for k in
                                 ("packed", "per_request_padded", "batch_padded"))
                             for c in controls)
            print(f"[e07] controls: best-model error up to {worst_ctrl:.1f}% "
                  f"(all three models agree there, so this is measurement noise)")
            if worst_ctrl > 15.0:
                print("[e07] WARNING controls disagree with every model — the measurement is "
                      "suspect and the verdict below should not be trusted.", file=sys.stderr)
        if not ragged:
            print("[e07] no ragged cells; nothing decided", file=sys.stderr)
            return 1

        votes = {k: sum(1 for s in ragged if s["best_fit"] == k)
                 for k in ("packed", "per_request_padded", "batch_padded")}
        winner = max(votes, key=votes.get)
        mean_err = statistics.fmean(s[f"err_{winner}_pct"] for s in ragged)
        print(f"[e07] ragged cells: {votes}  -> {winner} (mean error {mean_err:.1f}%)")
        if winner == "packed":
            print("[e07] VERDICT: raggedness is FREE at batch level. Chunked prefill packs "
                  "requests into a step and pads only the step's token total, so per-request "
                  "padding costs nothing. The 'ragged workload' framing does not survive: what "
                  "remains is admission control over compiled TOKEN-COUNT buckets, which is "
                  "what every measurement so far actually addressed.")
        else:
            print(f"[e07] VERDICT: raggedness is PAID ({winner}). Padding a mixed-length batch "
                  "costs real TPU time, the project's premise holds, and the ladder half of the "
                  "contribution is live.")
        save_table(run, "verdict", [{"winner": winner, "mean_err_pct": mean_err, **votes}])
        print(f"[e07] run_id={run.run_id}")
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
