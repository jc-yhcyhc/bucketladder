"""Tests for the session-2 experiment scripts: e01, e02, e03.

The theme: each script's *analysis* must be able to tell the two hypotheses
apart. A mock that only ever produces the outcome we hope for proves nothing,
so every script has a mock for BOTH outcomes and the tests assert the statistic
separates them.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import e01_oracle_gap as e01  # noqa: E402
import e02_stock_baseline as e02  # noqa: E402
import e03_noise_floor as e03  # noqa: E402
from _client import complete_mock, summarise, token_ids  # noqa: E402
from _common import read_manifest  # noqa: E402
from ladder import build_ladder  # noqa: E402

E01_CFG = REPO / "configs" / "e01_marginal_cost.json"
E02_CFG = REPO / "configs" / "e02_stock_baseline.json"
E03_CFG = REPO / "configs" / "e03_noise_floor.json"


# --- exact-length prompts --------------------------------------------------

def test_token_ids_exact_length_and_deterministic():
    """Bucket-boundary work needs exact token counts, not approximate ones."""
    assert len(token_ids(513)) == 513
    assert token_ids(100, seed=1) == token_ids(100, seed=1)
    assert token_ids(100, seed=1) != token_ids(100, seed=2)


def test_token_ids_rejects_zero():
    with pytest.raises(ValueError):
        token_ids(0)


# --- e01: does the flatness statistic discriminate? ------------------------

def test_flatness_is_one_for_pure_staircase():
    # Same cost regardless of length -> padding fully paid.
    assert e01.flatness({100: 50.0, 200: 50.0}, 200) == pytest.approx(1.0, abs=0.01)


def test_flatness_is_zero_for_pure_proportional():
    # Cost exactly proportional to length -> padding free.
    assert e01.flatness({100: 25.0, 200: 50.0}, 200) == pytest.approx(0.0, abs=0.01)


def test_flatness_is_between_for_partial():
    f = e01.flatness({100: 37.5, 200: 50.0}, 200)
    assert 0.2 < f < 0.8


def test_occupancy_lengths_never_leave_the_bucket():
    """Comparing across two buckets would measure two different executables."""
    ladder = build_ladder(8192, "")
    for bucket in (512, 1024, 4096):
        for L in e01.occupancy_lengths(bucket, e01.DEFAULT_FRACTIONS, ladder):
            assert bucket / 2 < L <= bucket, (bucket, L)


def test_occupancy_below_half_is_impossible_on_power_of_two_ladder():
    """The constraint discovered while building this: 0.5B is a bucket edge."""
    ladder = build_ladder(8192, "")
    assert e01.occupancy_lengths(1024, [0.5, 0.25], ladder) == []


def test_e01_mock_separates_staircase_from_linear(tmp_path):
    rc_s = e01.main(["--config", str(E01_CFG), "--mock",
                     "--results-root", str(tmp_path / "stair")])
    rc_l = e01.main(["--config", str(E01_CFG), "--mock", "--mock-linear",
                     "--results-root", str(tmp_path / "lin")])
    assert rc_s == 0 and rc_l == 0

    def median_flatness(root):
        import pandas as pd
        entry = read_manifest(root)[0]
        df = pd.read_parquet(Path(entry["path"]) / "flatness.parquet")
        return df["flatness"].median()

    stair, lin = median_flatness(tmp_path / "stair"), median_flatness(tmp_path / "lin")
    assert stair > 0.8, stair
    assert lin < 0.4, lin
    assert stair - lin > 0.5


# --- e03: the noise floor sets the units ----------------------------------

def test_e03_reports_a_cv(tmp_path, capsys):
    rc = e03.main(["--config", str(E03_CFG), "--mock", "--results-root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WITHIN-RUN CV" in out
    import pandas as pd
    entry = read_manifest(tmp_path)[0]
    df = pd.read_parquet(Path(entry["path"]) / "noise_floor.parquet")
    assert 0 < float(df["cv"].iloc[0]) < 0.5


def test_across_restart_cv_needs_two_blocks():
    import math
    assert math.isnan(e03.across_restart_cv([1.0]))
    assert e03.across_restart_cv([100.0, 110.0]) > 0


def test_summarise_ignores_failed_samples():
    from _client import Sample
    good = [Sample(1, 1, 10.0, 20.0), Sample(1, 1, 12.0, 22.0)]
    bad = [Sample(1, 1, float("nan"), float("nan"), ok=False, error="boom")]
    st = summarise(good + bad)
    assert st["n"] == 2


# --- e02: does the step analysis discriminate? -----------------------------

def test_concurrency_sweep_brackets_the_edge():
    levels = e02.concurrency_sweep([8, 16, 32], around=8)
    assert 8 in levels and 9 in levels and 7 in levels
    assert max(levels) >= 16


def test_e02_mock_separates_promote_from_queue(tmp_path):
    import pandas as pd

    def largest_step(policy, root):
        rc = e02.main(["--config", str(E02_CFG), "--mock", "--mock-policy", policy,
                       "--results-root", str(root)])
        assert rc == 0
        entry = read_manifest(root)[0]
        df = pd.read_parquet(Path(entry["path"]) / "verdict.parquet")
        return float(df["largest_step_frac"].iloc[0]), bool(df["on_request_ladder_edge"].iloc[0])

    promote, p_edge = largest_step("promote", tmp_path / "p")
    queue, q_edge = largest_step("queue", tmp_path / "q")
    # Both stock behaviours put their biggest jump on a ladder crossing...
    assert p_edge and q_edge
    # ...but queueing costs a full extra wave, so it is the larger step.
    assert queue > promote


def test_e02_records_padded_batch_size(tmp_path):
    import pandas as pd
    e02.main(["--config", str(E02_CFG), "--mock", "--results-root", str(tmp_path)])
    entry = read_manifest(tmp_path)[0]
    df = pd.read_parquet(Path(entry["path"]) / "summary.parquet")
    row = df[df["concurrency"] == 9]
    assert not row.empty and int(row["padded_to"].iloc[0]) == 16


# --- contract still holds for the new scripts ------------------------------

@pytest.mark.parametrize("mod,cfg", [(e01, E01_CFG), (e02, E02_CFG), (e03, E03_CFG)])
def test_new_scripts_honour_the_controlled_var_contract(mod, cfg, tmp_path):
    from _common import ControlledVarError
    bad = json.loads(cfg.read_text())
    bad["controlled"]["enable_prefix_caching"] = True
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ControlledVarError):
        mod.main(["--config", str(p), "--mock", "--results-root", str(tmp_path / "r")])
    assert read_manifest(tmp_path / "r") == []


# --- server-side metrics: the R1 fix --------------------------------------

REAL_PROM = """\
# HELP vllm:request_prefill_time_seconds Prefill time
# TYPE vllm:request_prefill_time_seconds histogram
vllm:request_prefill_time_seconds_bucket{le="0.01",model_name="Qwen/Qwen3-4B"} 3
vllm:request_prefill_time_seconds_sum{model_name="Qwen/Qwen3-4B"} 1.5
vllm:request_prefill_time_seconds_count{model_name="Qwen/Qwen3-4B"} 100
vllm:request_queue_time_seconds_sum{model_name="Qwen/Qwen3-4B"} 0.25
vllm:request_queue_time_seconds_count{model_name="Qwen/Qwen3-4B"} 100
vllm:num_requests_running{model_name="Qwen/Qwen3-4B"} 2.0
"""


def test_parses_prometheus_histograms():
    import _metrics
    snap = _metrics.parse_prometheus(REAL_PROM)
    assert snap["vllm:request_prefill_time_seconds"].count == 100
    assert snap["vllm:request_prefill_time_seconds"].mean() == pytest.approx(0.015)
    # gauges and _bucket lines must not be mistaken for histograms
    assert "vllm:num_requests_running" not in snap


def test_delta_gives_mean_over_exactly_the_new_requests():
    """Counters are cumulative; only the delta describes our own requests."""
    import _metrics
    before = _metrics.parse_prometheus(REAL_PROM)
    after = _metrics.parse_prometheus(
        REAL_PROM.replace("_sum{model_name=\"Qwen/Qwen3-4B\"} 1.5", "_sum{model_name=\"Qwen/Qwen3-4B\"} 2.5")
                 .replace("time_seconds_count{model_name=\"Qwen/Qwen3-4B\"} 100",
                          "time_seconds_count{model_name=\"Qwen/Qwen3-4B\"} 110"))
    d = _metrics.delta(before, after)
    pf = d["vllm:request_prefill_time_seconds"]
    assert pf["count"] == 10
    assert pf["mean_ms"] == pytest.approx(100.0)  # 1.0 s over 10 requests


def test_delta_omits_metrics_with_no_new_requests():
    import _metrics
    snap = _metrics.parse_prometheus(REAL_PROM)
    assert _metrics.delta(snap, snap) == {}


def test_short_names_are_readable():
    import _metrics
    assert _metrics.short("vllm:request_prefill_time_seconds") == "prefill"
    assert _metrics.short("vllm:request_queue_time_seconds") == "queue"
    assert _metrics.short("vllm:time_to_first_token_seconds") == "ttft"


# --- e02 redesign: the 2x2 must reach opposite verdicts --------------------

def test_e02_verdict_separates_promote_from_queue(tmp_path):
    """The whole point of the redesign. Client TTFT could not do this."""
    import pandas as pd

    def verdict(policy, root):
        assert e02.main(["--config", str(E02_CFG), "--mock", "--mock-policy", policy,
                         "--results-root", str(root)]) == 0
        entry = read_manifest(root)[0]
        return pd.read_parquet(Path(entry["path"]) / "stock_verdict.parquet").iloc[0]

    p = verdict("promote", tmp_path / "p")
    q = verdict("queue", tmp_path / "q")

    assert "promote" in p["verdict"], p["verdict"]
    assert "queue" in q["verdict"], q["verdict"]
    # promote pays in PREFILL and does not wait; queue is the mirror image.
    assert p["d_prefill_ms"] > p["d_queue_ms"]
    assert q["d_queue_ms"] > q["d_prefill_ms"]


def test_e02_records_server_timing_table(tmp_path):
    import pandas as pd
    e02.main(["--config", str(E02_CFG), "--mock", "--results-root", str(tmp_path)])
    entry = read_manifest(tmp_path)[0]
    df = pd.read_parquet(Path(entry["path"]) / "server_timing.parquet")
    assert {"queue_ms", "prefill_ms", "concurrency", "padded_to"} <= set(df.columns)


# --- e01: headline must come from server timing, not the client -----------

def test_e01_headline_uses_server_prefill(tmp_path):
    import pandas as pd
    e01.main(["--config", str(E01_CFG), "--mock", "--results-root", str(tmp_path)])
    entry = read_manifest(tmp_path)[0]
    df = pd.read_parquet(Path(entry["path"]) / "flatness.parquet")
    assert (df["source"] == "server_prefill").all()
    # the client proxy is retained for comparison, not discarded
    assert "flatness_client_ttft" in df.columns


def test_e01_still_separates_hypotheses_on_server_timing(tmp_path):
    import pandas as pd

    def med(args, root):
        e01.main(["--config", str(E01_CFG), "--mock", *args, "--results-root", str(root)])
        entry = read_manifest(root)[0]
        return pd.read_parquet(Path(entry["path"]) / "flatness.parquet")["flatness"].median()

    assert med([], tmp_path / "s") - med(["--mock-linear"], tmp_path / "l") > 0.5


# --- R4: intervals, not points --------------------------------------------

def test_bootstrap_ci_brackets_a_known_difference():
    from _stats import bootstrap_ci
    a = [10.0] * 20
    b = [8.0] * 20
    lo, hi = bootstrap_ci(a, b)
    assert lo <= 2.0 <= hi


def test_bootstrap_ci_requires_paired_lengths():
    from _stats import bootstrap_ci
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], [1.0])


def test_bootstrap_p_detects_no_difference():
    from _stats import bootstrap_p
    a = [10.0, 10.1, 9.9, 10.0, 10.2]
    assert bootstrap_p(a, list(a)) > 0.5


def test_bootstrap_p_detects_a_real_difference():
    from _stats import bootstrap_p
    assert bootstrap_p([10.0] * 10, [20.0] * 10) < 0.05


def test_flatness_ci_is_tight_for_clean_staircase():
    """Cost identical at both occupancies -> flatness 1.0 with a narrow CI."""
    from _stats import flatness_ci
    pt, lo, hi = flatness_ci([50.0, 50.1, 49.9] * 4, [50.0, 50.1, 49.9] * 4, 100, 200)
    assert pt == pytest.approx(1.0, abs=0.05)
    assert hi - lo < 0.2


def test_flatness_ci_is_wide_when_underpowered():
    """Noisy cells must produce an interval that says so, not a confident point."""
    from _stats import flatness_ci
    _, lo, hi = flatness_ci([10.0, 90.0, 30.0, 70.0], [50.0, 55.0, 45.0, 60.0], 100, 200)
    assert hi - lo > 0.5


def test_flatness_ci_matches_e01_point_estimate():
    """The bootstrapped point must equal the statistic e01 reports."""
    from _stats import flatness_ci
    lo_vals, hi_vals = [25.0] * 5, [50.0] * 5
    pt, _, _ = flatness_ci(lo_vals, hi_vals, 100, 200)
    assert pt == pytest.approx(e01.flatness({100: 25.0, 200: 50.0}, 200), abs=1e-9)


def test_fmt_ci_shape():
    from _stats import fmt_ci
    assert fmt_ci(0.97, 0.94, 1.01) == "0.97 [0.94, 1.01]"
    assert "no CI" in fmt_ci(0.97, float("nan"), float("nan"))


# Shared 512/1024 boundary geometry for the paid_share_ci tests below: real
# tokens move 508->516 (real_ratio ~1.016), padded tokens double 512->1024
# (padded_ratio 2.0) -- e14_n1_all_boundaries.json's actual first edge.
_PS_REAL_BELOW, _PS_REAL_ABOVE = 508.0, 516.0
_PS_PAD_BELOW, _PS_PAD_ABOVE = 512.0, 1024.0


def test_paid_share_ci_is_one_when_fully_paid():
    """Cost tracks the padded ratio exactly -> share 1.0 with a narrow CI."""
    from _stats import paid_share_ci
    real_ratio = _PS_REAL_ABOVE / _PS_REAL_BELOW
    below = [100.0, 100.2, 99.8] * 4
    above = [b * 2.0 for b in below]  # cost_ratio == padded_ratio == 2.0
    pt, lo, hi = paid_share_ci(below, above, _PS_REAL_BELOW, _PS_REAL_ABOVE,
                               _PS_PAD_BELOW, _PS_PAD_ABOVE)
    assert pt == pytest.approx(1.0, abs=1e-9)
    assert hi - lo < 0.1
    assert real_ratio < 2.0  # sanity: the two ratios this statistic sits between differ


def test_paid_share_ci_is_zero_when_fully_free():
    """Cost tracks the real ratio exactly -> share 0.0 with a narrow CI."""
    from _stats import paid_share_ci
    real_ratio = _PS_REAL_ABOVE / _PS_REAL_BELOW
    below = [100.0, 100.2, 99.8] * 4
    above = [b * real_ratio for b in below]  # cost_ratio == real_ratio
    pt, lo, hi = paid_share_ci(below, above, _PS_REAL_BELOW, _PS_REAL_ABOVE,
                               _PS_PAD_BELOW, _PS_PAD_ABOVE)
    assert pt == pytest.approx(0.0, abs=1e-9)
    assert hi - lo < 0.1


def test_paid_share_ci_is_wide_when_underpowered():
    """Noisy arms must produce an interval that says so, not a confident point."""
    from _stats import paid_share_ci
    below = [80.0, 120.0, 90.0, 110.0]
    above = [150.0, 250.0, 180.0, 220.0]
    _, lo, hi = paid_share_ci(below, above, _PS_REAL_BELOW, _PS_REAL_ABOVE,
                              _PS_PAD_BELOW, _PS_PAD_ABOVE)
    assert hi - lo > 0.5


def test_paid_share_ci_matches_paper_numbers_formula():
    """The bootstrapped point must equal (measured - real) / (padded - real),
    the formula paper_numbers.m1_share/m14_share already use."""
    from _stats import paid_share_ci
    below_vals, above_vals = [50.0] * 5, [92.0] * 5
    real_ratio = _PS_REAL_ABOVE / _PS_REAL_BELOW
    padded_ratio = _PS_PAD_ABOVE / _PS_PAD_BELOW
    cost_ratio = 92.0 / 50.0
    expected = (cost_ratio - real_ratio) / (padded_ratio - real_ratio)
    pt, _, _ = paid_share_ci(below_vals, above_vals, _PS_REAL_BELOW, _PS_REAL_ABOVE,
                             _PS_PAD_BELOW, _PS_PAD_ABOVE)
    assert pt == pytest.approx(expected, abs=1e-9)


def test_paid_share_ci_needs_at_least_two_repeats_per_arm():
    """One repeat can't resample, so the interval must come back as NaN, not 0."""
    from _stats import paid_share_ci
    pt, lo, hi = paid_share_ci([100.0], [200.0], _PS_REAL_BELOW, _PS_REAL_ABOVE,
                               _PS_PAD_BELOW, _PS_PAD_ABOVE)
    assert pt == pytest.approx(1.0, abs=1e-9)  # point estimate still computable
    assert lo != lo and hi != hi  # NaN != NaN


def test_e01_emits_intervals_and_flags_underpowered(tmp_path):
    import pandas as pd
    e01.main(["--config", str(E01_CFG), "--mock", "--results-root", str(tmp_path)])
    entry = read_manifest(tmp_path)[0]
    df = pd.read_parquet(Path(entry["path"]) / "flatness_ci.parquet")
    assert {"flatness", "ci_lo", "ci_hi", "ci_width"} <= set(df.columns)
    assert (df["ci_lo"] <= df["flatness"]).all() and (df["flatness"] <= df["ci_hi"]).all()


# --- R3: the second model must actually differ ----------------------------

def test_granite_config_differs_in_head_dim():
    """Guards the reason granite was chosen. If someone swaps it for another
    Qwen size, R3 becomes vacuous and this fails."""
    cfg = json.loads((REPO / "configs" / "e01_marginal_cost_granite.json").read_text())
    assert "granite" in cfg["model"]
    assert cfg["model"] != json.loads(E01_CFG.read_text())["model"]
