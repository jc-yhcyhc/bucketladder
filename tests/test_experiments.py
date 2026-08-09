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
