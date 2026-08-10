#!/usr/bin/env python3
"""
e21 — the offline optimum, by dynamic programming. What `Oracle` was pretending to be.

`sim/policies.py::Oracle` was documented as an upper bound until the refitted
e30 showed `hybrid` beating it at three arrival rates. It is a one-step
lookahead with a fixed horizon: seeing the next arrival is not seeing the
future. So the repo currently supports no claim of the form "hybrid is within
X% of optimal", and that is the gap this script closes.

THE PROBLEM. Given the whole arrival trace in advance, partition it into batches
and choose when to dispatch each, minimising

    total TPU-ms  +  lambda * total latency-ms

A single objective is necessary because the two extremes are degenerate. Cost is
sublinear in tokens (`sim/measured_cost_curve.json`), so minimising cost alone
says "put everything in one maximal batch and never mind the wait"; minimising
latency alone says "dispatch every request the instant it arrives". The
interesting object is the **Pareto frontier** between them, and lambda traces it.

ASSUMPTIONS, both stated rather than buried:
  - Batches are CONTIGUOUS in arrival order. Reordering could in principle do
    better, and is not something a serving system should do — it starves early
    requests. This is FIFO-with-batching, the same discipline every policy here
    obeys, so the comparison is fair.
  - One server, serial batches, no decode. Same model the simulator uses.

THE RECURRENCE. With requests sorted by arrival and `f_j` the finish time of
batch j,

    dispatch time  t_j = max(arrival of the batch's LAST member, f_{j-1})
    f_j            = t_j + C(g * prompt_len) / 1000

State is (prefix length, finish time). Finish time is continuous, so instead of
discretising it — which would make the "optimum" approximate and quietly
unsound as a bound — each prefix keeps its **Pareto frontier** of
(finish_time, objective) pairs. Both coordinates are minimised: an earlier
finish can only help later batches, and a smaller objective is better outright.
That keeps the result exact.

Latency sums come from a prefix-sum table, so extending a batch is O(1):

    sum_{m in batch} (f - a_m) = g * f - (A_k - A_i)

Usage:
  python scripts/e21_offline_optimum.py --config configs/e21_offline_optimum.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))

from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from cost_model import CostModel  # noqa: E402
from policies import ALL_POLICIES, Hybrid  # noqa: E402
from simulator import Request, Simulator  # noqa: E402


@dataclass(frozen=True)
class Plan:
    """One achievable (cost, latency) point and the schedule behind it."""

    objective: float
    total_cost_ms: float
    total_latency_ms: float
    n_batches: int
    n_requests: int

    @property
    def cost_per_request_ms(self) -> float:
        return self.total_cost_ms / self.n_requests

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / self.n_requests


def _pareto(items: list[tuple[float, float, float, float, int]]
            ) -> list[tuple[float, float, float, float, int]]:
    """Keep (finish, objective) pairs that nothing else dominates.

    Sorted by finish ascending; a candidate survives only if its objective beats
    every earlier (i.e. earlier-finishing) survivor. Without this the state
    space grows exponentially in the number of batches.
    """
    items.sort(key=lambda x: (x[0], x[1]))
    out: list[tuple[float, float, float, float, int]] = []
    best = math.inf
    for it in items:
        if it[1] < best - 1e-12:
            out.append(it)
            best = it[1]
    return out


def solve(arrivals: list[float], cost: CostModel, prompt_len: int,
          lam: float, max_group: int) -> Plan:
    """Exact minimum of `total_cost_ms + lam * total_latency_ms`.

    `lam` is in ms of cost per ms of latency summed over requests. lam=0 is
    pure cost (maximal batching); large lam approaches dispatch-on-arrival.
    """
    n = len(arrivals)
    if n == 0:
        return Plan(0.0, 0.0, 0.0, 0, 0)

    # prefix[k] = sum of the first k arrival times, in ms
    prefix = [0.0] * (n + 1)
    for i, a in enumerate(arrivals):
        prefix[i + 1] = prefix[i] + a * 1000.0

    # dp[k] : Pareto frontier of (finish_s, objective, total_cost, total_lat, n_batches)
    dp: list[list[tuple[float, float, float, float, int]]] = [[] for _ in range(n + 1)]
    dp[0] = [(0.0, 0.0, 0.0, 0.0, 0)]

    for k in range(1, n + 1):
        cand: list[tuple[float, float, float, float, int]] = []
        for g in range(1, min(max_group, k) + 1):
            i = k - g
            if not dp[i]:
                continue
            c_ms = cost.step_cost_ms(g, g * prompt_len)
            last_arrival = arrivals[k - 1]          # batch holds i..k-1
            lat_const = prefix[k] - prefix[i]       # sum of arrivals in the batch
            for f_prev, obj_prev, cost_prev, lat_prev, nb in dp[i]:
                t = last_arrival if last_arrival > f_prev else f_prev
                f = t + c_ms / 1000.0
                lat = g * (f * 1000.0) - lat_const  # sum of (finish - arrival), ms
                cand.append((f, obj_prev + c_ms + lam * lat,
                             cost_prev + c_ms, lat_prev + lat, nb + 1))
        dp[k] = _pareto(cand)

    f, obj, tot_cost, tot_lat, nb = min(dp[n], key=lambda x: x[1])
    return Plan(obj, tot_cost, tot_lat, nb, n)


def frontier(arrivals: list[float], cost: CostModel, prompt_len: int,
             lambdas: list[float], max_group: int) -> list[Plan]:
    return [solve(arrivals, cost, prompt_len, lam, max_group) for lam in lambdas]


def lagrangian_lower_bound(front: list[Plan], lambdas: list[float],
                           total_latency_ms: float) -> float:
    """Provable lower bound on the cost of ANY schedule no slower than the policy.

    The obvious approach — take the cheapest sampled frontier point whose
    latency is under the policy's — is wrong, and wrong in the dangerous
    direction. Scalarising by lambda recovers only points on the CONVEX HULL of
    the Pareto set, so when a policy's latency falls between two sampled points
    the search is forced onto the stricter one, which costs more than the true
    optimum at that latency. The first version of this script did that and
    reported hybrid beating the optimum by 10.4% — an impossibility that is
    worth more than the number it replaced, because it says the bound was not a
    bound.

    The dual is exact about this. With g(l) = min_x [cost(x) + l * latency(x)],
    for any schedule x with latency(x) <= L and any l >= 0:

        cost(x)  >=  g(l) - l * latency(x)  >=  g(l) - l * L

    so max_l [g(l) - l*L] lower-bounds every feasible schedule. Taking L to be
    the POLICY's own latency makes the policy itself feasible, which forces the
    bound below the policy's cost and guarantees regret >= 0 by construction.

    The bound can be loose where the frontier is non-convex, and looseness
    OVERSTATES regret — it makes the policy look further from optimal than it
    is. That is the conservative direction for a claim of the form "hybrid is
    within X% of optimal", so the error is on the honest side.
    """
    return max(p.objective - lam * total_latency_ms for p, lam in zip(front, lambdas))


def build_policy(name: str, cfg: dict[str, Any]):
    if name == "hybrid":
        return Hybrid(latency_weight=cfg.get("latency_weight", 1.0),
                      max_wait_s=cfg.get("max_wait_s", 0.5))
    cls = ALL_POLICIES[name]
    try:
        return cls(max_wait_s=cfg.get("max_wait_s", 0.5))
    except TypeError:
        return cls()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    plen = cfg.get("prompt_len", 512)
    n_req = cfg.get("n_requests", 60)
    seeds = cfg.get("seeds", 5)
    rates = cfg.get("rates_hz", [10, 25, 55, 90])
    lambdas = cfg.get("lambdas", [0.0, 0.005, 0.02, 0.05, 0.15, 0.5, 2.0, 10.0])
    pol_names = cfg.get("policies", ["promote", "wait", "hybrid"])

    run = start_run("e21_offline_optimum", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        cost = CostModel()
        if not cost.knots:
            print("[e21] refusing to bound the optimum with the analytic cost model, "
                  "which failed its hardware holdout. Run scripts/refit_cost_model.py --write.",
                  file=sys.stderr)
            return 1
        sim = Simulator(cost)
        max_group = max(1, cost.max_batched_tokens // plen)
        print(f"[e21] DP optimum vs policies. n={n_req} prompt_len={plen} "
              f"max_group={max_group} seeds={seeds} lambdas={len(lambdas)}")

        front_rows: list[dict[str, Any]] = []
        gap_rows: list[dict[str, Any]] = []

        for rate in rates:
            fronts: list[list[Plan]] = []
            pol_pts: dict[str, list[tuple[float, float]]] = {p: [] for p in pol_names}
            for seed in range(seeds):
                trace: list[Request] = sim.make_trace(n_req, rate, plen, seed=seed)
                arrivals = [r.arrival_s for r in trace]
                fronts.append(frontier(arrivals, cost, plen, lambdas, max_group))
                for p in pol_names:
                    res = sim.run(trace, build_policy(p, cfg))
                    # MEAN latency, matching the DP's summed-latency objective.
                    # Comparing a policy's p50 against a mean-latency frontier
                    # is not like-for-like and flatters the policy whenever the
                    # distribution is skewed -- which, for one that deliberately
                    # holds requests back, it always is.
                    pol_pts[p].append((res.cost_per_request_ms, res.mean_latency_ms))

            print(f"[e21] --- {rate} req/s ---")
            for j, lam in enumerate(lambdas):
                c = statistics.fmean(fronts[s][j].cost_per_request_ms for s in range(seeds))
                l = statistics.fmean(fronts[s][j].mean_latency_ms for s in range(seeds))
                b = statistics.fmean(fronts[s][j].n_batches for s in range(seeds))
                front_rows.append({"rate_hz": rate, "lambda": lam, "cost_per_request_ms": c,
                                   "mean_latency_ms": l, "n_batches": b})
                print(f"[e21]   lambda={lam:<6g} optimum  cost {c:6.2f} ms/req   "
                      f"mean latency {l:8.1f} ms   batches {b:5.1f}")

            for p in pol_names:
                pc = statistics.fmean(x[0] for x in pol_pts[p])
                pl = statistics.fmean(x[1] for x in pol_pts[p])
                # Per seed: provable lower bound on the cost of any schedule no
                # slower than this policy on this trace.
                gaps = []
                for s in range(seeds):
                    cost_total = pol_pts[p][s][0] * n_req
                    lat_total = pol_pts[p][s][1] * n_req
                    lb = lagrangian_lower_bound(fronts[s], lambdas, lat_total)
                    if cost_total > 0:
                        gaps.append(100.0 * (cost_total - lb) / cost_total)
                gap = statistics.fmean(gaps) if gaps else float("nan")
                assert not gaps or min(gaps) > -1e-6, (
                    f"negative regret {min(gaps)} for {p}: the bound is not a bound")
                gap_rows.append({"rate_hz": rate, "policy": p, "cost_per_request_ms": pc,
                                 "mean_latency_ms": pl, "regret_pct": gap, "n_seeds": len(gaps)})
                shown = f"{gap:5.1f}%" if gaps else "  n/a"
                print(f"[e21]   {p:<8} cost {pc:6.2f}  mean lat {pl:7.1f}   "
                      f"cost above the bound at its own latency: {shown}")

        save_table(run, "frontier", front_rows)
        save_table(run, "regret", gap_rows)
        print("[e21] regret = how much cheaper an offline scheduler could have been "
              "WITHOUT being slower than the policy. This is the bound `Oracle` never was.")
        print(f"[e21] run_id={run.run_id}")
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
