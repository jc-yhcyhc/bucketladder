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


def test_scales_from_origin_below_the_first_knot():
    """No measurement exists under 512 tokens. Clamping flat would make a tiny
    batch cost as much as a real one, which is exactly the quantity the policy
    comparison turns on."""
    c = CostModel()
    assert c.tokens_cost_ms(256) == pytest.approx(c.tokens_cost_ms(512) / 2, rel=1e-9)


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
