#!/usr/bin/env python3
"""
O11 — bounding the MoE second-order padding cost from public routing facts,
without hardware.

MLSys review, targeted question: "Given MOE_ROUTE_PADDING_TO_EXPERT0 ships
disabled and fails open, and you could not obtain MoE-capacity hardware, can
you provide even a simulated estimate of the magnitude of the second-order
'padding activates an unnecessary expert' cost, to bound whether this is a
3% or a 60% effect, as you note it could be?"

Appendix S10.6 states the sign of this effect (real) without a magnitude,
because no reading of the source settles it. This computes a magnitude bound
from public architecture facts rather than from source or from hardware:

  - Qwen3-30B-A3B: 128 routed experts, top-8 per token, trained with a
    standard load-balancing auxiliary loss (Qwen3 Technical Report,
    arXiv:2505.09388; corroborated by the HF Qwen3MoE model docs). The
    auxiliary loss is minimised under uniform expert utilisation, which is
    the standard justification (GShard, Switch Transformer, ST-MoE) for
    treating per-token routing as close to a uniform random k-of-E draw as
    a FIRST-ORDER model -- not a claim that real routing is exactly uniform.

THE MODEL. A step with N real tokens, each an independent uniform random
k-of-E draw, touches some set of experts. A (N+1)-th, padded token is also a
uniform random k-of-E draw, independent of the first N. By linearity of
expectation, the expected number of the padded token's k picks that land on
an expert NONE of the N real tokens touched -- a genuinely new matmul group
the step would not otherwise run -- is exactly

    k * P(a specific expert untouched by N real tokens)
  = k * ((E - k) / E) ** N

exactly, under the i.i.d.-uniform model (no approximation once that model is
granted). Dividing by k gives the FRACTION of the padded token's own expert
dispatches that are new groups, which is what "3% or 60%" is a bound on.

WHAT THIS DOES NOT SHOW. This bounds how many NEW GROUPS a padded token
causes to activate, as a function of real batch size -- not the microseconds
that costs, which depends on the fixed per-group launch/weight-load overhead
against the marginal per-token cost inside an already-active group, neither
of which this paper measures. It also assumes i.i.d. uniform routing per
step, which the auxiliary loss trains toward on average but does not
guarantee at the level of one step's real traffic: correlated routing
(similar prompts routing similarly) would lower the new-group fraction below
this estimate; adversarial or narrow-domain traffic could raise it. This is
the mechanism's REACH, not its cost in time, and not a hardware measurement.

Usage:
  python scripts/o11_moe_padding_bound.py
  python scripts/o11_moe_padding_bound.py --self-test
"""
from __future__ import annotations

import argparse
import sys


def new_group_fraction(n_real: int, experts: int, top_k: int) -> float:
    """Exact expected fraction of a padded token's k dispatches that land on
    an expert none of `n_real` i.i.d.-uniform real tokens already touched."""
    return ((experts - top_k) / experts) ** n_real


def crossing_n(target_frac: float, experts: int, top_k: int) -> float:
    """n_real at which the new-group fraction first falls to `target_frac`."""
    import math
    return math.log(target_frac) / math.log((experts - top_k) / experts)


def render(experts: int, top_k: int, ns: list[int]) -> str:
    L = [f"\nQwen3-30B-A3B: E={experts} routed experts, top-{top_k} per token,",
        "i.i.d.-uniform routing model (first-order, from the load-balancing",
        "auxiliary loss's training target -- not a measurement):",
        "-" * 56,
        f"{'n_real (§4.2 batch)':>20}{'new-group fraction':>24}"]
    for n in ns:
        L.append(f"{n:>20}{new_group_fraction(n, experts, top_k) * 100:>23.2f}%")
    c3 = crossing_n(0.03, experts, top_k)
    c60 = crossing_n(0.60, experts, top_k)
    L.append(f"\ncrosses 60% at n_real ~= {c60:.1f}  (§4.2 tests n=8: {new_group_fraction(8, experts, top_k)*100:.1f}%)")
    L.append(f"crosses  3% at n_real ~= {c3:.1f}   (§4.2 tests up to n=16: {new_group_fraction(16, experts, top_k)*100:.1f}%)")
    return "\n".join(L) + "\n"


def _self_test() -> int:
    E, K = 128, 8
    # At n_real=0 nothing is touched yet: every dispatch is new by definition.
    assert abs(new_group_fraction(0, E, K) - 1.0) < 1e-9
    # Monotonically non-increasing in n_real.
    vals = [new_group_fraction(n, E, K) for n in range(0, 40)]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    # The paper's own stated bracket ("three percent... or sixty") should
    # bracket realistic §4.2 batch sizes under this model, not sit far
    # outside the range that model produces.
    assert 0.55 < new_group_fraction(8, E, K) < 0.65, new_group_fraction(8, E, K)
    assert 40 < crossing_n(0.03, E, K) < 70
    # Tends to zero as n_real grows.
    assert new_group_fraction(500, E, K) < 1e-9
    print("self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experts", type=int, default=128,
                    help="total routed experts (Qwen3-30B-A3B: 128)")
    ap.add_argument("--top-k", type=int, default=8,
                    help="experts activated per token (Qwen3-30B-A3B: 8)")
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 54, 64, 96, 128])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    print(render(args.experts, args.top_k, args.ns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
