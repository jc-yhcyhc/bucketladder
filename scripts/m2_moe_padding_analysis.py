#!/usr/bin/env python3
"""Per-padded-token cost across the three M2 arms, and what the flag recovers.

Arms (see infra/run_m2_arms.sh):
  A  m2_moe_qwen_default   MoE, MOE_ROUTE_PADDING_TO_EXPERT0=0  -- the stock default
  B  m2_moe_qwen_expert0   MoE, MOE_ROUTE_PADDING_TO_EXPERT0=1  -- padding collapsed to expert 0
  C  m2_dense_control      dense reference at the same TP

The quantity of interest is microseconds per padded token, taken across a bucket
edge where real work barely moves and padded work roughly doubles:

    us_per_padded_token = (cost_above_ms - cost_below_ms) * 1000
                          / (tokens_padded_above - tokens_padded_below)

A vs B is the strong comparison -- same model, same slice, same ladder, one env
var -- so it is reported as a paired difference per edge rather than as two
independent means.

Three outcomes are distinguished explicitly, because they are different papers:
  * A ~ B ~ C  : MoE padding costs what dense padding costs; the flag is noise.
  * A > B ~ C  : the default is a real, fixable tax. The flag is the finding.
  * A > B > C  : MoE padding is structurally dearer even mitigated.

Usage:
  python scripts/m2_moe_padding_analysis.py --results-root results
  python scripts/m2_moe_padding_analysis.py --self-test
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = {
    "m2_moe_qwen_default": "A  MoE, flag off (stock)",
    "m2_moe_qwen_expert0": "B  MoE, flag on",
    "m2_dense_control": "C  dense reference",
}


def us_per_padded_token(edge_row: dict) -> float | None:
    """Marginal microseconds per padded token across one bucket edge.

    Returns None rather than a number when the edge cannot support the
    quotient: a non-positive padded-token delta means the edge did not
    straddle a boundary, and dividing by it would manufacture a value.
    """
    d_pad = edge_row.get("padded_delta")
    if not d_pad or d_pad <= 0:
        return None
    return (edge_row["cost_above_ms"] - edge_row["cost_below_ms"]) * 1000.0 / d_pad


def load_arm(root: Path, experiment: str) -> list[dict]:
    """Latest finished run for one arm, as edge rows carrying a padded delta."""
    import pyarrow.parquet as pq

    runs = sorted((root / experiment).glob(f"{experiment}__*"))
    if not runs:
        return []
    latest = runs[-1]
    edges = pq.read_table(latest / "edges.parquet").to_pylist()
    disp = pq.read_table(latest / "dispatches.parquet").to_pylist()

    # edges.parquet carries ratios, not absolute padded counts; recover the
    # delta from the dispatch rows the edge was computed from.
    for e in edges:
        below = [d for d in disp if d["edge"] == e["edge"] and d["arm"] == "below"]
        above = [d for d in disp if d["edge"] == e["edge"] and d["arm"] == "above"]
        if below and above:
            e["padded_delta"] = (statistics.median(d["tokens_padded"] for d in above)
                                 - statistics.median(d["tokens_padded"] for d in below))
        else:
            e["padded_delta"] = None
        e["run_id"] = latest.name
    return edges


def analyse(root: Path) -> dict:
    out: dict = {"arms": {}, "paired": None, "verdict": None}
    per_arm: dict[str, dict[str, float]] = {}

    for exp, label in ARMS.items():
        rows = load_arm(root, exp)
        vals = {r["edge"]: v for r in rows if (v := us_per_padded_token(r)) is not None}
        if vals:
            per_arm[exp] = vals
            out["arms"][exp] = {
                "label": label,
                "run_id": rows[0]["run_id"],
                "per_edge_us": vals,
                "median_us": statistics.median(vals.values()),
            }
        else:
            out["arms"][exp] = {"label": label, "missing": True}

    a, b = per_arm.get("m2_moe_qwen_default"), per_arm.get("m2_moe_qwen_expert0")
    if a and b:
        shared = sorted(set(a) & set(b))
        if shared:
            diffs = [a[e] - b[e] for e in shared]
            med_a = statistics.median(a[e] for e in shared)
            out["paired"] = {
                "edges": shared,
                "per_edge_delta_us": {e: a[e] - b[e] for e in shared},
                "median_delta_us": statistics.median(diffs),
                "recovered_fraction": (statistics.median(diffs) / med_a) if med_a else None,
            }
    return out


def render(out: dict) -> str:
    L = ["", "Per-padded-token cost by arm", "-" * 52]
    for exp, d in out["arms"].items():
        if d.get("missing"):
            L.append(f"  {d['label']:<28} (no completed run)")
        else:
            edges = "  ".join(f"{k}={v:.1f}" for k, v in d["per_edge_us"].items())
            L.append(f"  {d['label']:<28} median {d['median_us']:7.2f} us   [{edges}]")

    p = out.get("paired")
    if p:
        L += ["", "A vs B, paired by edge (same model, same slice, one env var)", "-" * 52,
              f"  median A-B          {p['median_delta_us']:7.2f} us per padded token"]
        if p["recovered_fraction"] is not None:
            L.append(f"  recovered by flag   {p['recovered_fraction'] * 100:6.1f}% of arm A's cost")
        L.append(f"  edges               {', '.join(p['edges'])}")
    else:
        L += ["", "A vs B not computable: both MoE arms need a completed run."]
    return "\n".join(L) + "\n"


def _self_test() -> int:
    """Pin the arithmetic and the refusals, so this is testable with no TPU."""
    assert us_per_padded_token(
        {"cost_below_ms": 10.0, "cost_above_ms": 20.0, "padded_delta": 1000}) == 10.0
    for bad in (0, None, -8):
        assert us_per_padded_token(
            {"cost_below_ms": 10.0, "cost_above_ms": 20.0, "padded_delta": bad}) is None

    out = {"arms": {"m2_moe_qwen_default": {"label": "A", "per_edge_us": {"512/1024": 30.0},
                                            "median_us": 30.0, "run_id": "r"},
                    "m2_moe_qwen_expert0": {"label": "B", "per_edge_us": {"512/1024": 12.0},
                                            "median_us": 12.0, "run_id": "r"}},
           "paired": {"edges": ["512/1024"], "per_edge_delta_us": {"512/1024": 18.0},
                      "median_delta_us": 18.0, "recovered_fraction": 0.6}}
    assert "60.0%" in render(out)
    # A missing arm must degrade to a stated gap, never to a silent zero.
    assert "no completed run" in render({"arms": {"x": {"label": "C", "missing": True}},
                                         "paired": None})
    print("self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    out = analyse(args.results_root)
    print(json.dumps(out, indent=2) if args.json else render(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
