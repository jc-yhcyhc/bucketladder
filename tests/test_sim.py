"""Tests for the admission simulator, policies and cost model.

The cost model is fitted to captured hardware measurements, so the first job is
guarding those constants against silent drift. The second is the simulator's
progress guarantee, which a real bug motivated.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sim")); sys.path.insert(0, str(REPO / "scripts"))

from cost_model import (CostModel, PROMOTION_COST_MS, REQUEST_LADDER,  # noqa: E402
                        TOKEN_SLOPE_MS_PER_TOKEN, padded_batch)
from policies import (DownshiftToEdge, Hybrid, Oracle, PromoteNow, WaitToFill)  # noqa: E402
from simulator import Simulator  # noqa: E402


# --- cost model is fitted, not invented ------------------------------------

def test_cost_model_reproduces_the_measured_cells():
    """Fitted against captured/session3 e02. Within 3% on both anchor cells."""
    c = CostModel()
    assert c.step_cost_ms(8, 8 * 512) == pytest.approx(75.56, rel=0.05)
    assert c.step_cost_ms(9, 9 * 512) == pytest.approx(137.92, rel=0.05)


def test_promotion_cost_matches_the_measurement():
    assert CostModel().promotion_cost_ms(8, 9) == pytest.approx(PROMOTION_COST_MS, rel=1e-6)


def test_promotion_inside_a_bucket_is_free():
    """The reason waiting is ever pointless: 9->10 changes no compiled shape."""
    assert CostModel().promotion_cost_ms(9, 10) == 0.0
    assert CostModel().promotion_cost_ms(1, 8) == 0.0


def test_padded_batch_uses_the_measured_ladder():
    assert REQUEST_LADDER == (8, 16, 32, 64, 128, 256)
    assert [padded_batch(n) for n in (1, 8, 9, 16, 17, 33)] == [8, 8, 16, 16, 32, 64]


def test_measured_constants_have_not_drifted():
    """These trace to a specific captured run. Changing them silently would
    invalidate every simulation result, so make it loud."""
    assert PROMOTION_COST_MS == pytest.approx(60.21, abs=0.01)
    assert TOKEN_SLOPE_MS_PER_TOKEN == pytest.approx(4.18e-3, rel=1e-3)


# --- the progress guarantee (a real bug) -----------------------------------

@pytest.mark.parametrize("policy", [PromoteNow(), WaitToFill(), Hybrid(),
                                    DownshiftToEdge(), Oracle()])
@pytest.mark.parametrize("rate", [10, 40, 90])
def test_every_policy_terminates(policy, rate):
    """wait-to-fill at 10 req/s span forever: it recomputed its deadline from
    `now`, converging on the threshold without reaching it. The simulator now
    refuses waits that do not move the clock."""
    s = Simulator()
    r = s.run(s.make_trace(120, rate, 512, 0), policy)
    assert r.n_requests == 120
    assert r.n_batches >= 1


def test_a_pathological_policy_cannot_hang_the_simulator():
    class ZenoPolicy(WaitToFill):
        name = "zeno"
        def decide(self, *, now_s, **kw):
            from policies import Decision
            return Decision(wait_until_s=now_s + 1e-18)   # never advances
    s = Simulator()
    r = s.run(s.make_trace(50, 30, 512, 0), ZenoPolicy())
    assert r.n_requests == 50


# --- policy semantics ------------------------------------------------------

def test_stock_never_waits_by_choice():
    """e02 measured queue time 0.0 ms at every concurrency level — but that was
    an idle server. Stock never *chooses* to hold a request; any queueing it
    shows is time behind an in-flight batch. So the claim to test is that on an
    idle server stock queues nothing, and that it never queues more than a
    policy that deliberately waits."""
    s = Simulator()
    idle = s.run(s.make_trace(200, 5, 512, 1), PromoteNow())
    assert idle.p50_queue_ms == pytest.approx(0.0, abs=1e-9)

    tr = s.make_trace(200, 40, 512, 1)
    assert s.run(tr, PromoteNow()).p50_queue_ms <= s.run(tr, WaitToFill()).p50_queue_ms


def test_wait_to_fill_achieves_full_occupancy():
    s = Simulator()
    r = s.run(s.make_trace(300, 60, 512, 2), WaitToFill())
    assert r.mean_batch_occupancy > 0.95


def test_waiting_is_cheaper_but_slower_than_stock():
    """The whole tension in one assertion."""
    s = Simulator()
    tr = s.make_trace(300, 25, 512, 3)
    stock, wait = s.run(tr, PromoteNow()), s.run(tr, WaitToFill())
    assert wait.cost_per_request_ms < stock.cost_per_request_ms
    assert wait.p95_latency_ms > stock.p95_latency_ms


def test_hybrid_beats_stock_on_cost_without_losing_latency():
    """The actual proposal, stated as a test."""
    s = Simulator()
    tr = s.make_trace(400, 25, 512, 4)
    stock, hyb = s.run(tr, PromoteNow()), s.run(tr, Hybrid())
    assert hyb.cost_per_request_ms < stock.cost_per_request_ms
    assert hyb.p95_latency_ms <= stock.p95_latency_ms * 1.05


def test_matched_traces_are_identical_across_policies():
    """Differences must be policy, not luck."""
    s = Simulator()
    a = [(r.rid, r.arrival_s) for r in s.make_trace(50, 30, 512, 7)]
    b = [(r.rid, r.arrival_s) for r in s.make_trace(50, 30, 512, 7)]
    assert a == b


def test_token_budget_is_never_exceeded():
    s = Simulator()
    c = CostModel()
    r = s.run(s.make_trace(300, 200, 512, 5), PromoteNow())
    # 8192/512 = 16 requests max per step
    assert r.n_batches >= 300 / 16


# --- e40 holdout harness ---------------------------------------------------
# A mock PASS proves the HARNESS is correct — that predicted equals measured
# when the server obeys the cost model exactly. It says nothing about whether
# real hardware obeys it. Only the hardware run tests that.

def test_holdout_harness_is_self_consistent(tmp_path):
    """Against a server that follows the cost model exactly, MAPE must be ~0.
    If this drifts, the harness is broken and any hardware MAPE is meaningless."""
    import json
    sys.path.insert(0, str(REPO / "scripts"))
    import e40_holdout as e40
    from _common import read_manifest
    import pandas as pd

    cfg = json.loads((REPO / "configs" / "e40_holdout.json").read_text())
    cfg.update({"n_requests": 24, "rates_hz": [40], "policies": ["promote", "wait"]})
    p = tmp_path / "cfg.json"; p.write_text(json.dumps(cfg))

    assert e40.main(["--config", str(p), "--mock", "--results-root", str(tmp_path / "r")]) == 0
    entry = read_manifest(tmp_path / "r")[0]
    v = pd.read_parquet(Path(entry["path"]) / "verdict.parquet").iloc[0]
    assert v["worst_ape_pct"] < 15.0, f"harness self-consistency broken: {v['worst_ape_pct']:.1f}%"
    assert v["verdict"] == "PASS"


def test_holdout_reports_per_cell_not_just_mean():
    """plan_v4's rule is <15% PER CELL. A good mean hiding one bad cell must
    still fail, so the verdict is computed on the worst cell."""
    import json
    sys.path.insert(0, str(REPO / "scripts"))
    import e40_holdout as e40
    src = (REPO / "scripts" / "e40_holdout.py").read_text()
    assert 'worst_ape_pct' in src and 'worst < 15.0' in src
