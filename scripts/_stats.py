"""
Intervals, not points (notes/solidity.md, R4).

`bootstrap_ci` and `bootstrap_p` are ported verbatim in behaviour from
`infersim/extract_paper_numbers.py` — same paired-bootstrap CI and same
two-sided permutation test that produced that project's reported p-values.
Borrowing the convention, not the research content, per the README.

What is added here is `flatness_ci`, because this project's headline statistic
is a *ratio of differences* rather than a mean. Propagating an interval through
it analytically is fiddly and easy to get wrong; bootstrapping the whole
statistic is neither.
"""

from __future__ import annotations

import random
import statistics
from typing import Callable, Sequence


def bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    stat_fn: Callable[[Sequence[float]], float] = statistics.mean,
    n: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap CI for stat_fn(a) - stat_fn(b).

    Paired: the same resampled indices are applied to both arms, so anything
    that moves both together (a slow server, a hot neighbour) cancels instead of
    inflating the interval.
    """
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal lengths, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("empty input")
    rng = random.Random(seed)
    n_obs = len(a)
    diffs = []
    for _ in range(n):
        idx = [rng.randint(0, n_obs - 1) for _ in range(n_obs)]
        diffs.append(stat_fn([a[i] for i in idx]) - stat_fn([b[i] for i in idx]))
    diffs.sort()
    return diffs[int(alpha / 2 * n)], diffs[int((1 - alpha / 2) * n)]


def bootstrap_p(a: Sequence[float], b: Sequence[float], n: int = 10000, seed: int = 42) -> float:
    """Two-sided permutation test p-value for a difference in means."""
    rng = random.Random(seed)
    a, b = list(a), list(b)
    obs = abs(statistics.fmean(a) - statistics.fmean(b))
    combined = a + b
    n_a = len(a)
    hits = 0
    for _ in range(n):
        rng.shuffle(combined)
        if abs(statistics.fmean(combined[:n_a]) - statistics.fmean(combined[n_a:])) >= obs:
            hits += 1
    return hits / n


def _flatness_point(lo_vals: Sequence[float], hi_vals: Sequence[float],
                    lo_len: int, hi_len: int) -> float:
    """flatness for one resample. Mirrors e01_oracle_gap.flatness exactly."""
    lo, hi = statistics.median(lo_vals), statistics.median(hi_vals)
    if hi <= 0 or hi_len == lo_len:
        return float("nan")
    predicted_linear = hi * (lo_len / hi_len)
    denom = hi - predicted_linear
    if denom == 0:
        return float("nan")
    return (lo - predicted_linear) / denom


def flatness_ci(
    lo_vals: Sequence[float],
    hi_vals: Sequence[float],
    lo_len: int,
    hi_len: int,
    n: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """(point, ci_lo, ci_hi) for e01's flatness statistic.

    `lo_vals`/`hi_vals` are the per-repeat measurements at the shortest and
    longest occupancy inside one bucket. Resampled independently — they are
    separate cells, not paired observations.

    An interval matters most exactly where the result is interesting: flatness
    near 1.0 with a tight interval is a staircase, flatness near 1.0 with an
    interval spanning 0.5 is an underpowered cell pretending to be one.
    """
    rng = random.Random(seed)
    point = _flatness_point(lo_vals, hi_vals, lo_len, hi_len)
    n_lo, n_hi = len(lo_vals), len(hi_vals)
    if n_lo < 2 or n_hi < 2:
        return point, float("nan"), float("nan")

    vals = []
    for _ in range(n):
        rl = [lo_vals[rng.randint(0, n_lo - 1)] for _ in range(n_lo)]
        rh = [hi_vals[rng.randint(0, n_hi - 1)] for _ in range(n_hi)]
        f = _flatness_point(rl, rh, lo_len, hi_len)
        if f == f:  # drop NaN
            vals.append(f)
    if not vals:
        return point, float("nan"), float("nan")
    vals.sort()
    return point, vals[int(alpha / 2 * len(vals))], vals[int((1 - alpha / 2) * len(vals))]


def fmt_ci(point: float, lo: float, hi: float, unit: str = "") -> str:
    """`0.97 [0.94, 1.01]` — the form every reported number should take."""
    if lo != lo or hi != hi:
        return f"{point:.2f}{unit} [no CI]"
    return f"{point:.2f}{unit} [{lo:.2f}, {hi:.2f}]"
