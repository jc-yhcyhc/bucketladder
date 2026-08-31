#!/usr/bin/env python3
"""Reconcile §4.2 (per-request padding vanishes by n=16) against §4.4 (the
ladder is still worth 3.5-12% at n=16), from real per-step data instead of
conjecture.

MLSys review, question 3: the paper's own reconciliation -- per-request
padding vanishes while per-step padding does not -- was declared unmeasurable
because iteration_tokens_total bins on powers of two, coarser than the ladder
spacing. infra/patch_step_logger.py logs the exact pre-histogram numbers;
this turns a captured log of BUCKETLADDER_STEP lines into the one number that
settles it: mean padded-token overhead per step, by concurrency (n_reqs).

If that overhead stays flat or grows with n_reqs while §4.2's per-request
share falls to ~0 by n=16, the reconciliation holds: padding migrates from
individual requests to the packed step rather than disappearing. If it also
falls to ~0, §4.4's benefit needs a different explanation than padding.

Usage:
  ssh <vm> 'grep BUCKETLADDER_STEP /tmp/vllm_serve.log' > /tmp/steps.log
  python scripts/e15_step_reconcile.py /tmp/steps.log
  python scripts/e15_step_reconcile.py --self-test
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

LINE_RE = re.compile(
    r"BUCKETLADDER_STEP n_reqs=(?P<n>\d+) real_tokens=(?P<real>\d+) "
    r"padded_tokens=(?P<padded>\d+)"
)


def parse(lines: list[str]) -> list[dict]:
    """One dict per step. Malformed lines are skipped, not fatal -- a log
    truncated by a server restart mid-write must still yield the steps that
    parsed cleanly."""
    out = []
    for ln in lines:
        m = LINE_RE.search(ln)
        if not m:
            continue
        real, padded = int(m["real"]), int(m["padded"])
        if real <= 0 or padded < real:
            continue  # a race with shutdown can emit a zeroed or inverted line
        out.append({"n_reqs": int(m["n"]), "real_tokens": real, "padded_tokens": padded,
                    "overhead": (padded - real) / real})
    return out


def by_concurrency(steps: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for n in sorted({s["n_reqs"] for s in steps}):
        g = [s["overhead"] for s in steps if s["n_reqs"] == n]
        out[n] = {"n_steps": len(g),
                  "mean_overhead_pct": 100 * statistics.mean(g),
                  "median_overhead_pct": 100 * statistics.median(g)}
    return out


def render(out: dict[int, dict]) -> str:
    L = ["", "Per-step padded-token overhead, by concurrency (from real server steps)",
        "-" * 68,
        f"  {'n_reqs':>7}  {'steps':>7}  {'mean %':>8}  {'median %':>9}"]
    for n, d in out.items():
        L.append(f"  {n:>7}  {d['n_steps']:>7}  {d['mean_overhead_pct']:>7.1f}%  "
                f"{d['median_overhead_pct']:>8.1f}%")
    L += ["",
         "Compare mean % here against §4.2's per-request paid share at the same",
         "n_reqs. Flat or rising here while §4.2 falls to ~0 supports the paper's",
         "reconciliation (padding migrates to the packed step); both falling",
         "together does not, and §4.4's benefit needs a different explanation."]
    return "\n".join(L) + "\n"


def _self_test() -> int:
    lines = (
        ['BUCKETLADDER_STEP n_reqs=1 real_tokens=510 padded_tokens=512'] * 5 +
        ['BUCKETLADDER_STEP n_reqs=16 real_tokens=2000 padded_tokens=2048'] * 5 +
        ['(EngineCore pid=1) some unrelated log line'] +
        ['BUCKETLADDER_STEP n_reqs=16 real_tokens=0 padded_tokens=0']  # must be dropped
    )
    steps = parse(lines)
    assert len(steps) == 10, f"expected 10 clean steps, got {len(steps)}"
    out = by_concurrency(steps)
    assert set(out) == {1, 16}
    assert out[1]["n_steps"] == 5 and out[16]["n_steps"] == 5
    # 512/510 - 1 = 0.392%; 2048/2000 - 1 = 2.4%. Overhead does NOT vanish at
    # n=16 in this synthetic case -- exactly the pattern the reconciliation predicts.
    assert out[1]["mean_overhead_pct"] < out[16]["mean_overhead_pct"]
    assert "reconciliation" in render(out)
    print("self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile", type=Path, nargs="?")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.logfile:
        lines = args.logfile.read_text(errors="replace").splitlines()
    else:
        lines = sys.stdin.read().splitlines()

    steps = parse(lines)
    if not steps:
        print("[e15] no BUCKETLADDER_STEP lines parsed", file=sys.stderr)
        return 1
    out = by_concurrency(steps)
    if args.json:
        import json
        print(json.dumps(out, indent=2))
    else:
        print(render(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
