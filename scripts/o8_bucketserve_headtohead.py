#!/usr/bin/env python3
"""
O8 — BucketServe's own ladder-design objective, run on our workload.

The paper's position on BucketServe has been inferential: it quotes their stated
premise and sets our remeasurement against it. That is weaker than it needs to be,
and a reviewer asked for a head-to-head. Their objective is reimplementable from
the paper, so this runs it.

WHAT BUCKETSERVE SPECIFIES (notes/prior_art.md, from arXiv 2507.17120):

    Eq (2)  Waste_Ratio = (S_max - S_avg) / S_max
    Eq (3)  E[Waste]    = sum_b integral_{L_b}^{U_b} (1 - S/U_b) f(S) dS
    Eq (4)  U_b*        = E[S | L_b <= S < U_b]

Eq (4) is the Lloyd-Max centroid condition, which is a *local* optimality
condition and is what their paper declines to compute, calling it
"computationally expensive to calculate in practice". Rather than iterate to a
local optimum, this solves Eq (3) **globally** by dynamic programming over a
discretised length axis: with K buckets and N candidate edges the cost is
O(K N^2), which is milliseconds here. Their objective therefore gets a better
answer than their own paper proposes, which is the fair way to test it.

WHAT THIS COMPARES. Three ladders against the same sampled workload:

    the stack default          ten exponentially spaced shapes
    the gap lever (S4.7)       what VLLM_TPU_BUCKET_PADDING_GAP can express
    BucketServe DP             their objective, solved globally at equal K

on their metric (waste ratio) and on ours (mean padded tokens per request).

WHY THIS MATTERS TO THE PAPER'S CLAIM. Our refutation concerns the request
dimension and per-request length padding, neither of which carries cost on this
stack. BucketServe's ladder design targets the token dimension, which S4.2 shows
IS paid at small batch. If their boundaries beat the gap lever at equal shape
count, their design contribution stands and the paper must say so; the
disagreement is then about which dimension the padding lives in, not about
whether ladder design is worth doing.

Usage:
  python scripts/o8_bucketserve_headtohead.py
  python scripts/o8_bucketserve_headtohead.py --k 10 14 --emit-ladder 14
"""

from __future__ import annotations

import argparse
import bisect
import math
import random
import statistics


def sample_lengths(n: int, median: int, sigma: float, lo: int, hi: int,
                   seed: int) -> list[int]:
    rng = random.Random(seed)
    return [max(lo, min(hi, int(rng.lognormvariate(math.log(median), sigma))))
            for _ in range(n)]


def gap_ladder(gap: int | None, hi: int = 8192) -> list[int]:
    """The ladder VLLM_TPU_BUCKET_PADDING_GAP produces.

    The rule is not "powers of two, then linear": the stack keeps doubling while
    the doubling step is no larger than the gap, then switches. At gap 256 that
    inserts 768, which the obvious reading does not predict (S4.7).
    """
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


def waste_ratio(ladder: list[int], lens: list[int]) -> float:
    """BucketServe Eq (3): expected 1 - S/U_b, their own objective."""
    return statistics.fmean(1 - s / pad_to(ladder, s) for s in lens)


def padded_tokens(ladder: list[int], lens: list[int]) -> float:
    """Mean padded tokens per request, which S4.3 prices at ~35 us each."""
    return statistics.fmean(pad_to(ladder, s) - s for s in lens)


def bucketserve_ladder(lens: list[int], k: int, lo: int = 16, hi: int = 8192,
                       step: int = 64) -> list[int]:
    """Globally minimise Eq (3) over k bucket edges, by dynamic programming."""
    cand = list(range(lo, hi + 1, step))
    if cand[-1] != hi:
        cand.append(hi)
    srt = sorted(lens)
    pre = [0.0] * (len(srt) + 1)
    for i, s in enumerate(srt):
        pre[i + 1] = pre[i] + s

    def cell(a: int, b: int) -> float:
        """Summed waste for requests in (a, b], each padded up to b."""
        i, j = bisect.bisect_right(srt, a), bisect.bisect_right(srt, b)
        return 0.0 if j <= i else (j - i) - (pre[j] - pre[i]) / b

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--median", type=int, default=1200)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 14])
    ap.add_argument("--us-per-padded-token", type=float, default=35.0,
                    help="measured in S4.3 and confirmed independently in S4.7")
    ap.add_argument("--emit-ladder", type=int, default=None,
                    help="print the BucketServe ladder for this K and exit")
    args = ap.parse_args(argv)

    lens = sample_lengths(args.n, args.median, args.sigma, 64, 7000, args.seed)
    if args.emit_ladder:
        print(",".join(str(x) for x in bucketserve_ladder(lens, args.emit_ladder)))
        return 0

    rows = [("default (exponential)", gap_ladder(None)),
            ("gap 1024 (S4.7)", gap_ladder(1024)),
            ("gap 512", gap_ladder(512))]
    for k in args.k:
        rows.append((f"BucketServe DP, K={k}", bucketserve_ladder(lens, k)))

    base = padded_tokens(gap_ladder(None), lens)
    print(f"workload: n={len(lens)} lognormal median={args.median} sigma={args.sigma}")
    print(f"{'ladder':<24}{'K':>4}{'waste':>9}{'padded tok':>12}{'predicted vs default':>22}")
    for name, lad in rows:
        pt = padded_tokens(lad, lens)
        pred = (base - pt) * args.us_per_padded_token / 1000.0
        print(f"{name:<24}{len(lad):>4}{waste_ratio(lad, lens):>9.4f}{pt:>12.0f}"
              f"{pred:>19.1f} ms")
    for k in args.k:
        print(f"\nBucketServe K={k}: {bucketserve_ladder(lens, k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
