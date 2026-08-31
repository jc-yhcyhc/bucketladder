#!/usr/bin/env python3
"""Paid-share, with an interval, across all four D2 boundaries at n=1.

MLSys review: "The ~85% paid share at n<=2 rests on one boundary and carries
no interval (§4.2)... This is a cheap measurement -- n=1, all four boundaries,
15 repeats." configs/e14_n1_all_boundaries.json is that measurement; this
turns its captured dispatches into the same (point, CI) form as every other
statistic in this project, via _stats.paid_share_ci.

Clean dispatches only (n_steps <= 1 + output_len), matching
paper_numbers.m14_share's own justification: a split executes more steps and
therefore more padding, so pooling split dispatches into the estimate biases
the share upward -- the ninth failure in §6.

Usage:
  python scripts/e14_analysis.py --results-root results
  python scripts/e14_analysis.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stats import paid_share_ci  # noqa: E402


def clean_arm(rows: list[dict], edge: str, arm: str, output_len: int) -> tuple[list[float], float, float]:
    """(step_latency_ms per clean dispatch, tokens_real, tokens_padded) for one arm."""
    g = [r for r in rows if r["edge"] == edge and r["arm"] == arm
         and r["n_steps"] is not None and r["n_steps"] <= 1 + output_len]
    if not g:
        return [], float("nan"), float("nan")
    return [r["step_latency_ms"] for r in g], g[0]["tokens_real"], g[0]["tokens_padded"]


def analyse(dispatches: list[dict], output_len: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    edges = sorted({r["edge"] for r in dispatches})
    for edge in edges:
        below_vals, real_b, pad_b = clean_arm(dispatches, edge, "below", output_len)
        above_vals, real_a, pad_a = clean_arm(dispatches, edge, "above", output_len)
        if len(below_vals) < 2 or len(above_vals) < 2:
            out[edge] = {"clean_below": len(below_vals), "clean_above": len(above_vals),
                        "point": float("nan"), "ci": (float("nan"), float("nan")),
                        "note": "underpowered after excluding splits"}
            continue
        point, lo, hi = paid_share_ci(below_vals, above_vals, real_b, real_a, pad_b, pad_a)
        out[edge] = {"clean_below": len(below_vals), "clean_above": len(above_vals),
                    "point": point, "ci": (lo, hi)}
    return out


def render(out: dict[str, dict]) -> str:
    L = ["", "Paid share at n=1, by boundary (point [95% CI], clean dispatches)", "-" * 62]
    for edge, d in out.items():
        if d["point"] != d["point"]:
            L.append(f"  {edge:<14} no estimate ({d.get('note', 'nan')}, "
                    f"{d['clean_below']}/{d['clean_above']} clean)")
        else:
            lo, hi = d["ci"]
            ci = "[no CI]" if lo != lo else f"[{lo*100:.1f}%, {hi*100:.1f}%]"
            L.append(f"  {edge:<14} {d['point']*100:6.1f}% {ci:<20} "
                    f"({d['clean_below']}/{d['clean_above']} clean)")
    return "\n".join(L) + "\n"


def load_dispatches(results_root: Path, experiment: str) -> tuple[list[dict], int]:
    """Latest run whose CONFIG identifies as `experiment`, not whose directory
    does: m1_boundary.py hardcodes start_run("m1_boundary", ...), so every
    config that reuses that script -- e14 among them -- lands in
    results/m1_boundary/ together, distinguished only by what their own config
    said its experiment name was. Reading the directory name instead would
    silently analyse whichever config happened to run last."""
    import pyarrow.parquet as pq

    candidates = []
    for d in sorted((results_root / "m1_boundary").glob("m1_boundary__*")):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("config", {}).get("experiment") == experiment:
            candidates.append((meta.get("started_at", ""), d, meta))
    if not candidates:
        return [], 0
    candidates.sort(key=lambda t: t[0])
    _, latest, meta = candidates[-1]
    output_len = meta.get("config", {}).get("output_len", 1)
    return pq.read_table(latest / "dispatches.parquet").to_pylist(), output_len


def _self_test() -> int:
    """Pin clean-dispatch filtering and the underpowered path, no TPU needed."""
    rows = []
    for i in range(15):
        rows.append({"edge": "512/1024", "arm": "below", "n_steps": 1,
                    "tokens_real": 508, "tokens_padded": 512, "step_latency_ms": 50.0 + i % 2})
        rows.append({"edge": "512/1024", "arm": "above", "n_steps": 1,
                    "tokens_real": 516, "tokens_padded": 1024, "step_latency_ms": 92.0 + i % 2})
    # One split dispatch that must be excluded, not averaged in. output_len=1
    # means clean is n_steps <= 2; a split shows MORE steps than that.
    rows.append({"edge": "512/1024", "arm": "below", "n_steps": 3,
                "tokens_real": 508, "tokens_padded": 512, "step_latency_ms": 999.0})
    out = analyse(rows, output_len=1)
    assert out["512/1024"]["clean_below"] == 15, "split dispatch leaked into the clean count"
    assert 0.0 < out["512/1024"]["point"] < 1.0

    # An edge with only one clean dispatch per arm must come back as a stated
    # gap, never a false-confident point-with-no-interval.
    thin = [{"edge": "e2", "arm": "below", "n_steps": 1, "tokens_real": 1, "tokens_padded": 2,
            "step_latency_ms": 10.0},
           {"edge": "e2", "arm": "above", "n_steps": 1, "tokens_real": 2, "tokens_padded": 4,
            "step_latency_ms": 15.0}]
    out2 = analyse(thin, output_len=1)
    assert out2["e2"]["point"] != out2["e2"]["point"]
    assert "underpowered" in out2["e2"]["note"]
    assert "underpowered" in render(out2)
    print("self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--experiment", default="e14_n1_all_boundaries")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    dispatches, output_len = load_dispatches(args.results_root, args.experiment)
    if not dispatches:
        print(f"[e14] no completed run of {args.experiment} under {args.results_root}",
              file=sys.stderr)
        return 1
    out = analyse(dispatches, output_len)
    print(json.dumps(out, indent=2) if args.json else render(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
