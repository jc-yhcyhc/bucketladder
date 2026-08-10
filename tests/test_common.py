"""Tests for the traceability contract in scripts/_common.py.

These encode the rules from notes/plan_v4.md that are easy to erode later:
meta.json first, never overwrite, abort-don't-warn on controlled variables,
and a config_hash stable across dict ordering and process restarts.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from _common import (  # noqa: E402
    CONTROLLED_VARS,
    ControlledVarError,
    assert_controlled_vars,
    config_hash,
    finish_run,
    load_config,
    make_run_id,
    read_manifest,
    save_table,
    start_run,
)


def good_controlled(**overrides):
    base = {
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 8192,
        "tensor_parallel_size": 4,
        "speculative_model": None,
        "kv_cache_dtype": "bfloat16",
        "VLLM_TPU_BUCKET_PADDING_GAP": "",
        "max_model_len": 8192,
        "XLA_FLAGS": "",
        # Default False. Added to the contract 2026-08-10 once M2 began sweeping
        # it; every run before that used the default.
        "ATTN_BUCKETIZED_NUM_REQS": False,
    }
    base.update(overrides)
    return base


def good_config(**overrides):
    return {"model": "test", "controlled": good_controlled(**overrides)}


# --- controlled-variable contract: abort, never warn ----------------------

def test_accepts_compliant_config():
    assert_controlled_vars(good_config())


def test_aborts_when_prefix_caching_on():
    with pytest.raises(ControlledVarError, match="enable_prefix_caching"):
        assert_controlled_vars(good_config(enable_prefix_caching=True))


def test_aborts_when_a_variable_is_missing():
    cfg = good_config()
    del cfg["controlled"]["max_model_len"]
    with pytest.raises(ControlledVarError, match="max_model_len: MISSING"):
        assert_controlled_vars(cfg)


def test_aborts_when_controlled_block_absent():
    with pytest.raises(ControlledVarError, match="no 'controlled' block"):
        assert_controlled_vars({"model": "test"})


def test_aborts_on_unknown_controlled_var():
    """An unchecked variable in the block is worse than no variable — it looks
    controlled but nothing verifies it."""
    with pytest.raises(ControlledVarError, match="unknown controlled vars"):
        assert_controlled_vars(good_config(enable_sausages=True))


def test_aborts_on_wrong_tp_size():
    with pytest.raises(ControlledVarError, match="tensor_parallel_size"):
        assert_controlled_vars(good_config(tensor_parallel_size=8))


def test_explicit_vars_accept_any_value_but_must_exist():
    assert_controlled_vars(good_config(enable_chunked_prefill=False))
    assert_controlled_vars(good_config(VLLM_TPU_BUCKET_PADDING_GAP=512))


def test_the_shipped_bad_config_is_actually_rejected():
    """configs/e00_BAD_apc_unrecorded.json exists to be caught. If this test
    ever passes silently the contract has been weakened."""
    cfg = load_config(REPO / "configs" / "e00_BAD_apc_unrecorded.json")
    with pytest.raises(ControlledVarError):
        assert_controlled_vars(cfg)


def test_the_shipped_good_configs_pass():
    for name in ("e00_default_ladder.json", "e00_gap512_ladder.json"):
        assert_controlled_vars(load_config(REPO / "configs" / name))


# --- config hashing --------------------------------------------------------

def test_hash_is_order_invariant():
    a = {"b": 1, "a": {"y": 2, "x": 3}}
    b = {"a": {"x": 3, "y": 2}, "b": 1}
    assert config_hash(a) == config_hash(b)


def test_hash_changes_with_content():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_hash_stable_across_process_restarts():
    """Guards against anyone reaching for hash(), id(), or set ordering."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from _common import config_hash; "
        "print(config_hash({'a': 1, 'b': [1, 2, {'c': 3}], 'd': 'x'}))"
        % str(REPO / "scripts")
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        for _ in range(3)
    }
    assert len(outs) == 1 and outs != {""}


def test_run_id_shape():
    rid = make_run_id("e00_smoke_test", good_config(), when=0)
    assert rid.startswith("e00_smoke_test__1970")
    assert rid.endswith(config_hash(good_config()))


# --- run lifecycle ---------------------------------------------------------

def test_meta_written_before_any_work(tmp_path):
    run = start_run("t", good_config(), results_root=tmp_path)
    # No save_table call yet — meta.json must already exist and say "running".
    assert run.meta_path.exists()
    meta = json.loads(run.meta_path.read_text())
    assert meta["status"] == "running"
    assert meta["config_hash"] == config_hash(good_config())
    assert "git" in meta and "host" in meta


def test_bad_config_leaves_no_directory(tmp_path):
    with pytest.raises(ControlledVarError):
        start_run("t", good_config(enable_prefix_caching=True), results_root=tmp_path)
    assert not (tmp_path / "t").exists()


def test_tables_never_overwritten(tmp_path):
    run = start_run("t", good_config(), results_root=tmp_path)
    save_table(run, "x", [{"a": 1}])
    with pytest.raises(FileExistsError):
        save_table(run, "x", [{"a": 2}])


def test_run_dir_never_reused(tmp_path):
    cfg = good_config()
    run = start_run("t", cfg, results_root=tmp_path)
    with pytest.raises(FileExistsError):
        # Same experiment + same config + same second -> same RUN_ID.
        import _common

        orig = _common.make_run_id
        try:
            _common.make_run_id = lambda *a, **k: run.run_id
            start_run("t", cfg, results_root=tmp_path)
        finally:
            _common.make_run_id = orig


def test_identical_config_yields_identical_parquet_bytes(tmp_path):
    """plan_v4.md verification: re-running an identical config must produce a
    byte-identical table."""
    rows = [{"bucket": 16, "cost": 1.5}, {"bucket": 32, "cost": 2.5}]
    r1 = start_run("t", good_config(), results_root=tmp_path / "a")
    r2 = start_run("t", good_config(), results_root=tmp_path / "b")
    p1 = save_table(r1, "tbl", rows)
    p2 = save_table(r2, "tbl", rows)
    assert p1.read_bytes() == p2.read_bytes()


def test_finish_writes_manifest(tmp_path):
    run = start_run("t", good_config(), results_root=tmp_path)
    save_table(run, "tbl", [{"a": 1}])
    finish_run(run, status="ok")

    meta = json.loads(run.meta_path.read_text())
    assert meta["status"] == "ok"
    assert meta["tables"] == ["tbl"]
    assert meta["duration_sec"] >= 0

    entries = read_manifest(tmp_path)
    assert len(entries) == 1
    assert entries[0]["run_id"] == run.run_id
    assert entries[0]["status"] == "ok"
    assert entries[0]["preempted"] is False


def test_failure_is_recorded_not_swallowed(tmp_path):
    run = start_run("t", good_config(), results_root=tmp_path)
    try:
        raise ValueError("boom")
    except ValueError as e:
        finish_run(run, status="failed", error=e)

    meta = json.loads(run.meta_path.read_text())
    assert meta["status"] == "failed"
    assert meta["error"]["type"] == "ValueError"
    assert "boom" in meta["error"]["message"]
    assert "Traceback" in meta["error"]["traceback"]
    assert read_manifest(tmp_path)[0]["status"] == "failed"


def test_finish_is_idempotent(tmp_path):
    run = start_run("t", good_config(), results_root=tmp_path)
    finish_run(run)
    finish_run(run)
    assert len(read_manifest(tmp_path)) == 1


def test_manifest_accumulates(tmp_path):
    for _ in range(3):
        import time

        run = start_run("t", good_config(), results_root=tmp_path)
        finish_run(run)
        time.sleep(1.01)  # RUN_ID has second resolution
    assert len(read_manifest(tmp_path)) == 3


def test_controlled_vars_list_is_not_silently_shrunk():
    """If someone deletes a controlled variable, this fails loudly. The set is
    the contract; shrinking it is a research decision, not a refactor."""
    assert set(CONTROLLED_VARS) == {
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "max_num_batched_tokens",
        "tensor_parallel_size",
        "speculative_model",
        "kv_cache_dtype",
        "VLLM_TPU_BUCKET_PADDING_GAP",
        "max_model_len",
        "XLA_FLAGS",
        "ATTN_BUCKETIZED_NUM_REQS",
    }


# --- spot preemption: per-run atomicity ------------------------------------
# A spot VM can vanish mid-write. The invariant is that MANIFEST.jsonl is
# appended only by finish_run, so a run with no manifest entry never finished
# no matter what files sit in its directory. Analysis reads the manifest.

def test_interrupted_run_is_not_complete(tmp_path):
    from _common import is_complete

    run = start_run("t", good_config(), results_root=tmp_path)
    save_table(run, "partial", [{"a": 1}])
    # Simulate preemption: process dies here, finish_run never runs.
    assert not is_complete(run.dir)
    assert read_manifest(tmp_path) == []


def test_finished_run_is_complete(tmp_path):
    from _common import is_complete

    run = start_run("t", good_config(), results_root=tmp_path)
    finish_run(run)
    assert is_complete(run.dir)


def test_mark_interrupted_flags_and_records(tmp_path):
    from _common import mark_interrupted_runs

    run = start_run("t", good_config(), results_root=tmp_path)
    save_table(run, "partial", [{"a": 1}])
    # (no finish_run — the VM went away)

    touched = mark_interrupted_runs(tmp_path)
    assert touched == [run.run_id]

    meta = json.loads(run.meta_path.read_text())
    assert meta["status"] == "interrupted"
    assert meta["preempted"] is True

    # Visible in the manifest rather than silently absent.
    entries = read_manifest(tmp_path)
    assert len(entries) == 1
    assert entries[0]["status"] == "interrupted"
    assert entries[0]["preempted"] is True


def test_mark_interrupted_cleans_temp_tables(tmp_path):
    from _common import mark_interrupted_runs

    run = start_run("t", good_config(), results_root=tmp_path)
    stale = run.dir / "half_written.parquet.tmp"
    stale.write_bytes(b"truncated garbage")
    mark_interrupted_runs(tmp_path)
    assert not stale.exists(), "a half-written table must not survive recovery"


def test_mark_interrupted_leaves_finished_runs_alone(tmp_path):
    from _common import mark_interrupted_runs

    run = start_run("t", good_config(), results_root=tmp_path)
    finish_run(run, status="ok")
    assert mark_interrupted_runs(tmp_path) == []
    assert json.loads(run.meta_path.read_text())["status"] == "ok"
    assert len(read_manifest(tmp_path)) == 1


def test_mark_interrupted_is_idempotent(tmp_path):
    from _common import mark_interrupted_runs

    start_run("t", good_config(), results_root=tmp_path)
    assert len(mark_interrupted_runs(tmp_path)) == 1
    assert mark_interrupted_runs(tmp_path) == []
    assert len(read_manifest(tmp_path)) == 1


def test_usable_runs_excludes_preempted_and_dirty(tmp_path):
    """The only function analysis code should use to find results.

    Note this test cannot assume the good run is usable: if the working tree is
    dirty (it usually is mid-development) usable_runs correctly drops it too.
    So assert the exclusions, and assert the inclusion only on a clean tree.
    """
    from _common import mark_interrupted_runs, usable_runs
    import time

    ok = start_run("t", good_config(), results_root=tmp_path)
    finish_run(ok, status="ok")
    time.sleep(1.01)
    start_run("t", good_config(), results_root=tmp_path)  # abandoned mid-run
    mark_interrupted_runs(tmp_path)

    assert len(read_manifest(tmp_path)) == 2, "both runs must be visible in the manifest"

    usable = usable_runs(tmp_path)
    assert all(e["status"] == "ok" and not e["preempted"] for e in usable)
    assert all(not e["git_dirty"] for e in usable)

    was_dirty = json.loads(ok.meta_path.read_text())["git"]["dirty"]
    if not was_dirty:
        assert {e["run_id"] for e in usable} == {ok.run_id}
    else:
        assert usable == [], "dirty tree must exclude even a successful run"


def test_save_table_leaves_no_tmp_file(tmp_path):
    run = start_run("t", good_config(), results_root=tmp_path)
    save_table(run, "x", [{"a": 1}])
    assert list(run.dir.glob("*.tmp")) == []
    assert (run.dir / "x.parquet").exists()
