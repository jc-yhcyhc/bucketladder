#!/usr/bin/env python3
"""
O10 — how much does traffic drift erode the offline ladder rule, before
re-selection recovers it?

MLSys review, targeted question: "The offline ladder-selection rule assumes a
roughly stationary length distribution sampled offline. How sensitive is the
12.1%/5%-prediction-accuracy result to distributional drift between the
sampling window and deployment window, and would you expect the daily/weekly
reselection cadence you propose to actually track realistic traffic
non-stationarity, or is this an open question?"

This is answerable offline, at zero dollars, by reusing exactly the machinery
S4.4 and Appendix S10.11 already describe: sample a length distribution, fit a
ladder against it by dynamic programming over a discretised length axis
(O8's DP, retargeted from BucketServe's relative-waste objective to this
paper's absolute-padded-tokens objective), and price padding at the 35 us/token
constant measured in S4.3. No hardware is needed because the object under test
is the offline MODEL, not a live server -- the same reason S4.4's original
prediction could be checked against measurement without a sweep.

THE EXPERIMENT. Fit a ladder once, against a "sampling window" workload
(median=1200, sigma=0.9, matching S4.4). Then evaluate that SAME, now-stale
ladder against a series of "deployment window" workloads whose median has
drifted by a stated factor, and compare its predicted gain over the stock
ladder against what a ladder freshly re-fit to the drifted distribution would
achieve. The gap between "stale" and "fresh" is the answer to the review
question: how much of the offline rule's benefit is lost to drift before
re-selection, at each drift magnitude.

WHAT THIS DOES NOT SHOW. This is a model-consistency check, not a validation
against measured hardware: nothing here confirms the 35 us/token constant
still holds at a drifted workload's shapes, only that the ladder-choice
machinery degrades in the way the model predicts. Appendix S10.11 already
states the re-selection counter has not been built; this quantifies the
question that counter would need to answer, not the counter itself.

Usage:
  python scripts/o10_drift_sensitivity.py
  python scripts/o10_drift_sensitivity.py --self-test
"""
from __future__ import annotations

import argparse
import bisect
import math
import random
import statistics
import sys


def sample_lengths(n: int, median: int, sigma: float, lo: int, hi: int,
                   seed: int) -> list[int]:
    rng = random.Random(seed)
    return [max(lo, min(hi, int(rng.lognormvariate(math.log(median), sigma))))
            for _ in range(n)]


def gap_ladder(gap: int | None, hi: int = 8192) -> list[int]:
    """The ladder VLLM_TPU_BUCKET_PADDING_GAP produces (S4.7)."""
    if gap is None:
        return [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    lad, x = [16], 16
    while x < hi:
        x = x * 2 if x <= gap else x + gap
        lad.append(min(x, hi))
    return sorted(set(lad))


def pad_to(ladder: list[int], n: int) -> int:
    for s in ladder:
        if s >= n:
            return s
    return ladder[-1]


def padded_tokens(ladder: list[int], lens: list[int]) -> float:
    return statistics.fmean(pad_to(ladder, s) - s for s in lens)


def fit_ladder(lens: list[int], k: int, lo: int = 16, hi: int = 8192,
              step: int = 64) -> list[int]:
    """Globally minimise mean absolute padded tokens over k bucket edges.

    Same DP shape as O8's bucketserve_ladder, retargeted from BucketServe's
    relative-waste cell (Eq 3) to this paper's own objective (S4.4): the sum
    of (b - s) for requests padded up to edge b, not the sum of (1 - s/b).
    """
    cand = list(range(lo, hi + 1, step))
    if cand[-1] != hi:
        cand.append(hi)
    srt = sorted(lens)
    pre = [0.0] * (len(srt) + 1)
    for i, s in enumerate(srt):
        pre[i + 1] = pre[i] + s

    def cell(a: int, b: int) -> float:
        i, j = bisect.bisect_right(srt, a), bisect.bisect_right(srt, b)
        return 0.0 if j <= i else (j - i) * b - (pre[j] - pre[i])

    inf = float("inf")
    dp = [[inf] * (k + 1) for _ in cand]
    back: list[list[int | None]] = [[None] * (k + 1) for _ in cand]
    for ci, c in enumerate(cand):
        dp[ci][1] = cell(lo - 1, c)
    for kk in range(2, k + 1):
        for ci, c in enumerate(cand):
            best, bi = inf, None
            for pi in range(ci):
                if dp[pi][kk - 1] == inf:
                    continue
                v = dp[pi][kk - 1] + cell(cand[pi], c)
                if v < best:
                    best, bi = v, pi
            dp[ci][kk], back[ci][kk] = best, bi
    ci: int | None = len(cand) - 1
    out, kk = [], k
    while kk > 0 and ci is not None:
        out.append(cand[ci])
        ci, kk = back[ci][kk], kk - 1
    return sorted(out)


def gain_ms(ladder: list[int], lens: list[int], base: list[int],
           us_per_token: float) -> float:
    """Predicted latency reduction of `ladder` over the default, on `lens`."""
    return (padded_tokens(base, lens) - padded_tokens(ladder, lens)) * us_per_token / 1000.0


def run(n: int, median: int, sigma: float, k: int, us_per_token: float,
       drifts: list[float], seed: int) -> list[dict]:
    default = gap_ladder(None)
    fit_lens = sample_lengths(n, median, sigma, 64, 7000, seed)
    stale = fit_ladder(fit_lens, k)

    rows = []
    for i, mult in enumerate(drifts):
        drift_median = max(64, int(round(median * mult)))
        deploy_lens = sample_lengths(n, drift_median, sigma, 64, 7000, seed + 1000 + i)
        fresh = fit_ladder(deploy_lens, k)
        g_stale = gain_ms(stale, deploy_lens, default, us_per_token)
        g_fresh = gain_ms(fresh, deploy_lens, default, us_per_token)
        retained = (g_stale / g_fresh * 100.0) if g_fresh > 1e-9 else float("nan")
        rows.append({"drift_x": mult, "deploy_median": drift_median,
                     "stale_gain_ms": g_stale, "fresh_gain_ms": g_fresh,
                     "pct_of_fresh_retained": retained})
    return rows


def render(rows: list[dict], median: int, k: int) -> str:
    L = [f"\nladder fit once at median={median} (K={k}), then evaluated stale",
        "against workloads whose median has drifted by the stated factor:",
        "-" * 68,
        f"{'drift':>7}{'deploy median':>15}{'stale gain':>13}{'fresh gain':>13}{'retained':>11}"]
    for r in rows:
        L.append(f"{r['drift_x']:>6.2f}x{r['deploy_median']:>15}"
                 f"{r['stale_gain_ms']:>11.1f} ms{r['fresh_gain_ms']:>11.1f} ms"
                 f"{r['pct_of_fresh_retained']:>10.0f}%")
    return "\n".join(L) + "\n"


def _self_test() -> int:
    # At zero drift the stale and fresh ladders are fit to the same
    # distribution and should coincide, retaining ~100% of the achievable gain.
    rows = run(n=1500, median=1200, sigma=0.9, k=14, us_per_token=35.0,
              drifts=[1.0], seed=7)
    assert rows[0]["pct_of_fresh_retained"] > 95, rows[0]
    # A larger drift should retain no more than a small drift, monotonically
    # (allowing sampling noise slack rather than requiring strict order).
    rows2 = run(n=1500, median=1200, sigma=0.9, k=14, us_per_token=35.0,
               drifts=[1.0, 3.0], seed=7)
    assert rows2[1]["pct_of_fresh_retained"] <= rows2[0]["pct_of_fresh_retained"] + 15, rows2
    print("self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--median", type=int, default=1200)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--k", type=int, default=14)
    ap.add_argument("--us-per-padded-token", type=float, default=35.0)
    ap.add_argument("--drifts", type=float, nargs="+",
                    default=[0.6, 0.8, 1.0, 1.3, 1.7, 2.5, 4.0])
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    rows = run(args.n, args.median, args.sigma, args.k, args.us_per_padded_token,
              args.drifts, args.seed)
    print(render(rows, args.median, args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
