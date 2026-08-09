"""Tests for scripts/e00_smoke_test.py — the W0b gate script.

Covers the log parser, the controlled-variable audit, and end-to-end runs in
mock mode, so the script is exercised fully before it ever reaches a TPU.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import e00_smoke_test as e00  # noqa: E402
from _common import ControlledVarError, read_manifest  # noqa: E402
from ladder import build_ladder  # noqa: E402

DEFAULT_CFG = REPO / "configs" / "e00_default_ladder.json"
GAP_CFG = REPO / "configs" / "e00_gap512_ladder.json"
BAD_CFG = REPO / "configs" / "e00_BAD_apc_unrecorded.json"


# --- warmup log parsing ----------------------------------------------------

def test_parses_the_prepared_paddings_line():
    """The real format, per hardware capture 2026-08-09."""
    lines = ["(EngineCore pid=1) INFO [utils.py:210] Prepared token paddings: [16, 32, 64]"]
    assert e00.parse_warmup_log(lines) == [16, 32, 64]


def test_falls_back_to_backbone_precompiles():
    """If a future vLLM drops the summary line, per-shape compiles still work."""
    lines = [
        "INFO [compilation_manager.py:133] Precompile worker0 backbone --> {'num_tokens': 64, 'num_reqs': 8}",
        "INFO [compilation_manager.py:133] Precompile worker0 backbone --> {'num_tokens': 16, 'num_reqs': 8}",
        # Non-ladder axes must NOT be picked up:
        "INFO [compilation_manager.py:133] Precompile worker0 sample --> {'num_reqs': 8}",
    ]
    assert e00.parse_warmup_log(lines) == [16, 64]


def test_summary_line_wins_over_precompiles():
    lines = [
        "INFO [utils.py:210] Prepared token paddings: [16, 32]",
        "INFO [compilation_manager.py:133] Precompile worker0 backbone --> {'num_tokens': 999}",
    ]
    assert e00.parse_warmup_log(lines) == [16, 32]


def test_parser_returns_empty_on_junk():
    assert e00.parse_warmup_log(["nothing to see", ""]) == []


def test_mock_log_round_trips_through_parser():
    for gap in ("", 512):
        lines = e00.mock_warmup_log(8192, gap)
        assert e00.parse_warmup_log(lines) == build_ladder(8192, gap)


# --- controlled-variable audit --------------------------------------------

def test_audit_marks_mock_when_no_server():
    cfg = json.loads(DEFAULT_CFG.read_text())
    rows = e00.audit_controlled(cfg, None)
    assert {r["verdict"] for r in rows} == {"mock"}
    assert len(rows) == len(cfg["controlled"])


def test_audit_flags_mismatch():
    # `server` is a flat dict of vLLM's own logged engine config, as returned
    # by parse_server_config — not a nested {"controlled": ...} block.
    cfg = json.loads(DEFAULT_CFG.read_text())
    rows = {r["variable"]: r for r in e00.audit_controlled(cfg, {"enable_prefix_caching": True})}
    assert rows["enable_prefix_caching"]["verdict"] == "MISMATCH"


def test_audit_reports_unverified_rather_than_passing():
    """A variable the server does not report must be visibly unverified, not
    quietly counted as ok."""
    cfg = json.loads(DEFAULT_CFG.read_text())
    rows = {r["variable"]: r for r in e00.audit_controlled(cfg, {"enable_prefix_caching": False})}
    assert rows["enable_prefix_caching"]["verdict"] == "ok"
    assert rows["max_model_len"]["verdict"] == "unverified"


# --- end to end, mock mode -------------------------------------------------

@pytest.mark.parametrize("cfg_path", [DEFAULT_CFG, GAP_CFG])
def test_mock_run_passes_gate(tmp_path, cfg_path, capsys):
    rc = e00.main(["--config", str(cfg_path), "--mock", "--results-root", str(tmp_path)])
    assert rc == 0
    assert "GATE PASSED" in capsys.readouterr().out


def test_mock_run_writes_expected_artifacts(tmp_path):
    e00.main(["--config", str(DEFAULT_CFG), "--mock", "--results-root", str(tmp_path)])

    entries = read_manifest(tmp_path)
    assert len(entries) == 1 and entries[0]["status"] == "ok"

    run_dir = Path(entries[0]["path"])
    assert set(entries[0]["tables"]) == {"ladder", "controlled_audit"}
    assert (run_dir / "ladder.parquet").exists()
    assert (run_dir / "controlled_audit.parquet").exists()

    ladder_json = json.loads((run_dir / "ladder_default.json").read_text())
    assert ladder_json["mode"] == "mock"
    assert ladder_json["ladder"] == build_ladder(8192, "")


def test_gap_config_produces_finer_ladder(tmp_path):
    e00.main(["--config", str(GAP_CFG), "--mock", "--results-root", str(tmp_path / "gap")])
    e00.main(["--config", str(DEFAULT_CFG), "--mock", "--results-root", str(tmp_path / "def")])

    gap = json.loads((Path(read_manifest(tmp_path / "gap")[0]["path"]) / "ladder_default.json").read_text())
    dflt = json.loads((Path(read_manifest(tmp_path / "def")[0]["path"]) / "ladder_default.json").read_text())
    assert len(gap["ladder"]) > len(dflt["ladder"])


def test_bad_config_aborts_before_creating_anything(tmp_path):
    """The deliberate test from plan_v4.md's verification section: a config
    with prefix caching in a non-compliant state must abort at start_run."""
    with pytest.raises(ControlledVarError):
        e00.main(["--config", str(BAD_CFG), "--mock", "--results-root", str(tmp_path)])
    assert read_manifest(tmp_path) == []
    assert not list(tmp_path.glob("**/meta.json"))


def test_gate_fails_on_ladder_mismatch(tmp_path, monkeypatch, capsys):
    """If the observed ladder disagrees with the prediction, the gate must fail
    and the run must be recorded as failed — not pass with a warning."""
    monkeypatch.setattr(e00, "parse_warmup_log", lambda lines: [1, 2, 3])
    rc = e00.main(["--config", str(DEFAULT_CFG), "--mock", "--results-root", str(tmp_path)])
    assert rc == 1
    assert "GATE FAILED" in capsys.readouterr().err
    assert read_manifest(tmp_path)[0]["status"] == "failed"


def test_gate_fails_on_empty_log(tmp_path, monkeypatch):
    monkeypatch.setattr(e00, "parse_warmup_log", lambda lines: [])
    rc = e00.main(["--config", str(DEFAULT_CFG), "--mock", "--results-root", str(tmp_path)])
    assert rc == 1
    assert read_manifest(tmp_path)[0]["status"] == "failed"


def test_captured_log_mode(tmp_path):
    log = tmp_path / "warmup.log"
    log.write_text("\n".join(e00.mock_warmup_log(8192, "")))
    rc = e00.main(
        ["--config", str(DEFAULT_CFG), "--warmup-log", str(log), "--results-root", str(tmp_path / "r")]
    )
    assert rc == 0
    assert read_manifest(tmp_path / "r")[0]["status"] == "ok"


# --- fixture section, now backed by REAL captured output ------------------
# The previous hand-written fixture guessed the log format and guessed wrong;
# it was deleted rather than kept, because a fictional fixture sitting beside
# real logs invites validating against the fiction.

FIXTURE = REPO / "tests" / "fixtures" / "real_v5e4_default.log"


def fixture_lines():
    return FIXTURE.read_text().splitlines()


def test_fixture_ladder_parses_to_powers_of_two():
    assert e00.parse_warmup_log(fixture_lines()) == build_ladder(8192, "")


def test_fixture_engine_config_parses():
    cfg = e00.parse_server_config(fixture_lines())
    assert cfg["tensor_parallel_size"] == 4
    assert cfg["max_seq_len"] == 8192
    assert cfg["enable_prefix_caching"] is False
    assert cfg["enable_chunked_prefill"] is True
    assert cfg["kv_cache_dtype"] == "auto"
    assert cfg["speculative_config"] is None


def test_coercion_of_logged_tokens():
    assert e00._coerce("True") is True
    assert e00._coerce("False") is False
    assert e00._coerce("None") is None
    assert e00._coerce("8192") == 8192
    assert e00._coerce("1.5") == 1.5
    assert e00._coerce("'bfloat16'") == "bfloat16"
    assert e00._coerce("tpu") == "tpu"


def test_audit_against_fixture_matches_shipped_config():
    """The shipped default config must agree with a realistic server log."""
    cfg = json.loads(DEFAULT_CFG.read_text())
    server = e00.parse_server_config(fixture_lines())
    rows = {r["variable"]: r["verdict"] for r in e00.audit_controlled(cfg, server)}

    for name in (
        "tensor_parallel_size",
        "max_model_len",            # via the max_seq_len alias
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "kv_cache_dtype",
        "speculative_model",        # via the speculative_config alias
    ):
        assert rows[name] == "ok", f"{name} -> {rows[name]}"

    # Env vars are structurally unverifiable from the log. They must be
    # reported as such, never silently passed.
    assert rows["VLLM_TPU_BUCKET_PADDING_GAP"] == "unverified"
    assert rows["XLA_FLAGS"] == "unverified"
    # vLLM 0.25 does not echo this one either — a known hole, named not hidden.
    assert rows["max_num_batched_tokens"] == "unverified"


def test_audit_catches_prefix_caching_left_on():
    """The failure this whole contract exists to prevent: config says APC off,
    server actually has it on."""
    cfg = json.loads(DEFAULT_CFG.read_text())
    server = e00.parse_server_config(fixture_lines())
    server["enable_prefix_caching"] = True
    rows = {r["variable"]: r["verdict"] for r in e00.audit_controlled(cfg, server)}
    assert rows["enable_prefix_caching"] == "MISMATCH"


def test_gate_fails_on_controlled_mismatch(tmp_path):
    """A MISMATCH must fail the gate, not just appear in a table."""
    log = tmp_path / "w.log"
    log.write_text(FIXTURE.read_text().replace("enable_prefix_caching=False", "enable_prefix_caching=True"))
    rc = e00.main(["--config", str(DEFAULT_CFG), "--warmup-log", str(log), "--results-root", str(tmp_path / "r")])
    assert rc == 1
    assert read_manifest(tmp_path / "r")[0]["status"] == "failed"


def test_end_to_end_on_fixture_passes(tmp_path, capsys):
    """Closest thing to a real run available without a TPU: real-shaped log,
    real parsers, real audit, real gate."""
    rc = e00.main(["--config", str(DEFAULT_CFG), "--warmup-log", str(FIXTURE), "--results-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "GATE PASSED" in out
    assert "unverifiable against server" in out  # the two env vars
    assert read_manifest(tmp_path)[0]["status"] == "ok"


# --- GROUND TRUTH: real logs captured on hardware 2026-08-09 ---------------
# v5litepod-4 (us-west4-a), tpu_inference 0.25.0, vllm 0.25.0, Qwen3-4B.
# These are the real thing, not hand-written. If a vLLM upgrade changes the log
# format these fail, which is exactly what we want to hear about.

REAL_DEFAULT = REPO / "tests" / "fixtures" / "real_v5e4_default.log"
REAL_GAP512 = REPO / "tests" / "fixtures" / "real_v5e4_gap512.log"


def test_real_default_ladder_matches_prediction():
    """The load-bearing check: hardware emitted exactly exponential_ladder(8192)."""
    obs = e00.parse_warmup_log(REAL_DEFAULT.read_text().splitlines())
    assert obs == build_ladder(8192, "")
    assert obs == [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def test_real_gap512_ladder_matches_prediction():
    """VLLM_TPU_BUCKET_PADDING_GAP=512 produced exactly gap_ladder(8192, 512)."""
    obs = e00.parse_warmup_log(REAL_GAP512.read_text().splitlines())
    assert obs == build_ladder(8192, 512)
    assert len(obs) == 21


def test_real_logs_expose_the_batch_axis_too():
    """Discovered on hardware: vLLM pads the BATCH dimension to its own ladder,
    a second axis the plan had not accounted for."""
    for f in (REAL_DEFAULT, REAL_GAP512):
        p = e00.parse_paddings(f.read_text().splitlines())
        assert p["request"] == [8, 16, 32, 64, 128, 256], f
        assert p["attn request"] == [256], f
        assert "token" in p


def test_gap_changes_token_ladder_but_not_batch_ladder():
    """The env var moves the token axis only — the request ladder is identical
    across both runs. Relevant to how the ladder can actually be manipulated."""
    a = e00.parse_paddings(REAL_DEFAULT.read_text().splitlines())
    b = e00.parse_paddings(REAL_GAP512.read_text().splitlines())
    assert a["token"] != b["token"]
    assert a["request"] == b["request"]


def test_real_engine_config_parses_and_confirms_chunked_prefill():
    """Confirms on hardware the finding that killed L1."""
    cfg = e00.parse_server_config(REAL_DEFAULT.read_text().splitlines())
    assert cfg["enable_chunked_prefill"] is True
    assert cfg["enable_prefix_caching"] is False
    assert cfg["tensor_parallel_size"] == 4
    assert cfg["max_seq_len"] == 8192


def test_real_logs_pass_the_gate(tmp_path):
    for cfg, log in ((DEFAULT_CFG, REAL_DEFAULT), (GAP_CFG, REAL_GAP512)):
        rc = e00.main(["--config", str(cfg), "--warmup-log", str(log),
                       "--results-root", str(tmp_path / log.stem)])
        assert rc == 0, log


def test_guessed_pattern_is_gone():
    """The old assumed format appears nowhere in real output. Pinned so nobody
    reintroduces it from memory."""
    for f in (REAL_DEFAULT, REAL_GAP512):
        assert "Compiling graph for num_tokens" not in f.read_text()
