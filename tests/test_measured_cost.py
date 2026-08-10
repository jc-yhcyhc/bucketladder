"""
The refitted cost model, and the harness overhead that made it validate.

Both were forced by a hardware failure: the analytic ladder-step model returned
MAPE 105.7% on e40 against a 15%-per-cell rule. These tests pin the properties
that mattered, so a future edit cannot quietly undo them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "scripts"))

from cost_model import MEASURED_KNOTS, CostModel  # noqa: E402
from policies import PromoteNow  # noqa: E402
from simulator import Simulator  # noqa: E402


def test_measured_curve_is_present_and_loaded():
    """A missing curve silently falls back to the model that FAILED its holdout.
    That is how a superseded model went on serving predictions for a session."""
    assert MEASURED_KNOTS, "sim/measured_cost_curve.json missing or empty"
    assert len(MEASURED_KNOTS) >= 8


def test_cost_is_sublinear_in_tokens():
    """The whole economics of admission control. A token costs ~25.7 us alone
    and ~16.9 us in a batch of eight; if this inverts, batching stops paying and
    every policy result changes sign."""
    c = CostModel()
    alone = c.tokens_cost_ms(512) / 512
    batched = c.tokens_cost_ms(4096) / 4096
    assert batched < alone
    assert alone / batched > 1.3


def test_batching_beats_splitting_at_equal_work():
    """One batch of 8 must cost less than 8 batches of 1 for the same tokens."""
    c = CostModel()
    assert c.tokens_cost_ms(4096) < 8 * c.tokens_cost_ms(512)


def test_interpolates_between_knots():
    c = CostModel()
    lo, hi = c.tokens_cost_ms(1024), c.tokens_cost_ms(2048)
    mid = c.tokens_cost_ms(1536)
    assert lo < mid < hi


def test_below_the_first_knot_is_floor_plus_linear_not_from_origin():
    """This test previously asserted the OPPOSITE and was wrong.

    It encoded linear-from-origin — `C(256) == C(512)/2` — on the reasoning that
    no measurement existed below 512 tokens so scaling was the conservative
    choice. M3 measured it: there is a 6.11 ms floor per step, and the old rule
    understated a 16-token step by 15x. A test that pins an assumption is only
    as good as the assumption, and this one outlived its evidence by eight
    sessions.
    """
    c = CostModel()
    assert c.tokens_cost_ms(256) > c.tokens_cost_ms(512) / 2


def test_falls_back_to_analytic_form_without_knots():
    """A fresh checkout with no captured runs must still simulate."""
    c = CostModel(knots=())
    assert c.step_cost_ms(8, 4096) > 0
    with pytest.raises(ValueError):
        c.tokens_cost_ms(4096)


def test_client_overhead_enlarges_batches_but_not_tpu_cost():
    """The harness's per-dispatch dead time changes batch COMPOSITION -- more
    arrivals accumulate while the driver is busy -- but the TPU is idle during
    it, and the paper's metric is TPU-busy time. Ignoring it made the simulator
    form batches of 1.22 where hardware formed 2.95, and was the sole cause of
    the one holdout cell that failed."""
    cost = CostModel()
    trace_src = Simulator(cost).make_trace(200, 55, 512, seed=7)

    plain = Simulator(cost, client_overhead_s=0.0).run(trace_src, PromoteNow())
    withoh = Simulator(cost, client_overhead_s=0.024).run(trace_src, PromoteNow())

    assert withoh.n_batches < plain.n_batches          # bigger batches
    assert withoh.cost_per_request_ms < plain.cost_per_request_ms   # hence cheaper
    # Overhead is never charged as compute.
    assert withoh.total_cost_ms < plain.total_cost_ms


def test_overhead_defaults_to_zero():
    """e30 asks what a policy costs intrinsically, not what our harness costs."""
    assert Simulator(CostModel()).client_overhead_s == 0.0


def test_small_steps_carry_the_measured_fixed_cost():
    """M3 measured a 16-token step at 6.10 ms and a 32-token step at 6.08.

    The model used to scale linearly from the origin below its lowest knot,
    pricing a 16-token step at 0.41 ms — 15x too low. That is not a rounding
    issue: it reversed the verdict on decomposing a padded residual into exact
    bucket sizes, which the extrapolation said won by 1.85 ms and which measures
    20.6% worse.
    """
    from cost_model import FIXED_STEP_COST_MS
    c = CostModel()
    assert c.tokens_cost_ms(16) > 5.0
    assert c.tokens_cost_ms(16) == pytest.approx(FIXED_STEP_COST_MS, abs=0.3)
    # Still monotone, and still meets the first measured knot.
    assert c.tokens_cost_ms(16) < c.tokens_cost_ms(256) < c.tokens_cost_ms(512)
    # The floor rule must still meet the first measured knot exactly, whatever
    # its value happens to be — the curve is regenerated from data, so pinning a
    # literal here would break on every refit for no reason.
    assert c.tokens_cost_ms(MEASURED_KNOTS[0][0]) == pytest.approx(MEASURED_KNOTS[0][1], rel=1e-9)


def test_decomposing_a_residual_loses():
    """1808 tokens as one padded step vs 1024+512+256+16. Measured: 42.33 vs
    51.06 ms. The model must now agree in direction, because four steps pay the
    fixed cost four times."""
    c = CostModel()
    whole = c.tokens_cost_ms(2048)
    parts = sum(c.tokens_cost_ms(t) for t in (1024, 512, 256, 16))
    assert parts > whole
