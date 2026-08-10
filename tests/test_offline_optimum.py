"""
The DP bound is only worth having if it is genuinely a bound.

Its first version was not: it compared policies against the cheapest SAMPLED
frontier point and reported hybrid beating the optimum by 10.4%. These tests
pin the two properties that make the replacement trustworthy — the DP really
finds the minimum, and the Lagrangian bound really lower-bounds it.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "scripts"))

from cost_model import CostModel  # noqa: E402
from e21_offline_optimum import lagrangian_lower_bound, solve  # noqa: E402
from policies import Hybrid, PromoteNow, WaitToFill  # noqa: E402
from simulator import Simulator  # noqa: E402

PLEN = 512


def brute_force(arrivals, cost, lam, max_group):
    """Every contiguous partition, scored directly. Exponential, so tiny n only."""
    n = len(arrivals)
    best = float("inf")
    for cuts in range(1 << max(0, n - 1)):
        groups, start = [], 0
        for i in range(n - 1):
            if cuts >> i & 1:
                groups.append((start, i + 1))
                start = i + 1
        groups.append((start, n))
        if any(b - a > max_group for a, b in groups):
            continue
        finish, total = 0.0, 0.0
        for a, b in groups:
            g = b - a
            c = cost.step_cost_ms(g, g * PLEN)
            t = max(arrivals[b - 1], finish)
            finish = t + c / 1000.0
            lat = sum(finish * 1000.0 - arrivals[m] * 1000.0 for m in range(a, b))
            total += c + lam * lat
        best = min(best, total)
    return best


@pytest.mark.parametrize("lam", [0.0, 0.01, 0.1, 1.0])
def test_dp_matches_brute_force(lam):
    """The DP keeps a Pareto set per prefix rather than discretising time. If
    that pruning ever drops a state it needs, this is where it shows."""
    cost = CostModel()
    arrivals = [r.arrival_s for r in Simulator(cost).make_trace(9, 25, PLEN, seed=3)]
    got = solve(arrivals, cost, PLEN, lam, max_group=16).objective
    want = brute_force(arrivals, cost, lam, max_group=16)
    assert got == pytest.approx(want, rel=1e-9)


def test_zero_lambda_reaches_the_cheapest_possible_cost():
    """With no latency penalty the optimum must hit the minimum per-request cost
    the curve allows.

    This test first asserted "batches of 16, the largest the token budget
    allows", on the assumption that sublinearity makes bigger always better. It
    is not: the measured curve is non-monotone per request, and the cheapest
    batch size is 3 (8.15 ms/req) rather than 16 (9.05). The assumption was the
    test's, not the DP's — and while chasing it, the DP was found exploiting a
    6144-token knot that rested on three observations of a bimodal cell, which
    is why `fit_knots` now requires five.
    """
    cost = CostModel()
    arrivals = [r.arrival_s for r in Simulator(cost).make_trace(48, 25, PLEN, seed=1)]
    plan = solve(arrivals, cost, PLEN, lam=0.0, max_group=16)
    best = min(cost.tokens_cost_ms(PLEN * g) / g for g in range(1, 17))
    assert plan.cost_per_request_ms == pytest.approx(best, rel=0.02)


def test_large_lambda_almost_dispatches_on_arrival():
    """Latency dominating means nothing waits *for company it does not have*.

    Not literally one batch per request: two requests arriving microseconds
    apart are better served together even under a huge latency penalty, because
    batching them finishes both sooner than serialising them. So the assertion
    is "nearly all singletons", not "all".
    """
    cost = CostModel()
    arrivals = [r.arrival_s for r in Simulator(cost).make_trace(20, 5, PLEN, seed=1)]
    plan = solve(arrivals, cost, PLEN, lam=1e6, max_group=16)
    assert plan.n_batches >= 18
    slow = solve(arrivals, cost, PLEN, lam=0.0, max_group=16)
    assert plan.total_latency_ms < slow.total_latency_ms


def test_cost_and_latency_trade_monotonically():
    """Higher lambda must never buy both lower cost and lower latency; if it
    did, the frontier would be wrong rather than merely coarse."""
    cost = CostModel()
    arrivals = [r.arrival_s for r in Simulator(cost).make_trace(40, 25, PLEN, seed=2)]
    plans = [solve(arrivals, cost, PLEN, lam, 16) for lam in (0.0, 0.01, 0.05, 0.3, 2.0)]
    for a, b in zip(plans, plans[1:]):
        assert b.total_cost_ms >= a.total_cost_ms - 1e-9
        assert b.total_latency_ms <= a.total_latency_ms + 1e-9


@pytest.mark.parametrize("policy_cls", [PromoteNow, Hybrid, WaitToFill])
def test_no_policy_beats_the_bound(policy_cls):
    """The property whose violation exposed the first version. A policy is
    feasible for its own latency budget, so the bound must sit at or below its
    cost — for every policy, at every rate."""
    cost = CostModel()
    sim = Simulator(cost)
    lambdas = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 5.0]
    for rate in (10, 25, 55, 90):
        trace = sim.make_trace(40, rate, PLEN, seed=5)
        arrivals = [r.arrival_s for r in trace]
        front = [solve(arrivals, cost, PLEN, lam, 16) for lam in lambdas]
        res = sim.run(trace, policy_cls())
        lb = lagrangian_lower_bound(front, lambdas, res.mean_latency_ms * res.n_requests)
        assert lb <= res.total_cost_ms + 1e-6, (
            f"{policy_cls.__name__} at {rate} req/s beats the bound: "
            f"{res.total_cost_ms:.2f} < {lb:.2f}")
