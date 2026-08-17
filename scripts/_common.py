"""
Traceability contract for every experiment in this repo.

Rules (notes/plan_v4.md, "Traceability contract"):
  1. meta.json is written FIRST, before any work happens. A crashed run still
     leaves evidence of what it was trying to do.
  2. Nothing is ever overwritten. A RUN_ID collision is an error, not a merge.
  3. Every run appends one line to MANIFEST.jsonl on completion (success or not).
  4. start_run / save_table / finish_run are used in try/finally so that a
     failed run is still recorded, with status="failed" and the traceback.
  5. assert_controlled_vars ABORTS on an unrecorded or wrong controlled
     variable. It never warns. A run that cannot prove prefix caching is off
     produces no data.

Nothing here imports jax, vllm, or torch — this module is fully testable on a
laptop with no accelerator, which is the point.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

RESULTS_ROOT = Path(os.environ.get("BUCKETLADDER_RESULTS", "results"))
MANIFEST_NAME = "MANIFEST.jsonl"

# ---------------------------------------------------------------------------
# Controlled variables
# ---------------------------------------------------------------------------
# Every one of these must appear in a config's `controlled` block. Values are
# either a literal that must match exactly, or REQUIRE_EXPLICIT meaning "any
# value, but it must be stated". Missing key -> abort. See plan_v4.md.

REQUIRE_EXPLICIT = "<must-be-stated>"

CONTROLLED_VARS: dict[str, Any] = {
    # Corrupts measured prompt-token counts, which corrupts every padding
    # number in the paper. This is the one that must be off, not merely stated.
    "enable_prefix_caching": False,
    # Determines whether prefill padding exists at all. Swept, so any value,
    # but it must be recorded.
    "enable_chunked_prefill": REQUIRE_EXPLICIT,
    # Sets the chunk size and therefore the residual prefill padding.
    "max_num_batched_tokens": REQUIRE_EXPLICIT,
    # Pinned to the hardware in use. 4 for the default path (v5litepod-4, our
    # quota-certain configuration); 1 if provisioning falls back to the GCE
    # path's ct6e-standard-1t, where Llama-3.1-8B fits one 32 GB v6e chip and
    # there is no tensor-parallel reduction at all. Set BUCKETLADDER_TP to
    # match the hardware; the config must then state that exact value.
    "tensor_parallel_size": int(os.environ.get("BUCKETLADDER_TP", "4")),
    # Changes accepted-token paths entirely.
    "speculative_model": None,
    "kv_cache_dtype": REQUIRE_EXPLICIT,
    # THE independent variable. Empty string means "vLLM default exponential
    # padding (nearest power of two)"; an int means linear buckets 16 ->
    # max_model_len with that gap.
    "VLLM_TPU_BUCKET_PADDING_GAP": REQUIRE_EXPLICIT,
    "max_model_len": REQUIRE_EXPLICIT,
    # The fraction of HBM vLLM may claim. Promoted to a control in session 25,
    # when it turned out to decide whether an experiment can run at all: the
    # 21-shape token ladder dies in warmup with RESOURCE_EXHAUSTED at the stack
    # default of 0.92 (32.50M requested, 12.40M free) on two independently
    # provisioned v5e-4 hosts, and boots only at 0.80. Compiled executables are
    # charged against the same HBM the KV cache is sized to fill, so the ladder
    # and this number are coupled -- which makes an unrecorded value a silent
    # confound between any two arms whose shape counts differ. Every run before
    # session 25 used 0.92: `gpu-memory-utilization` appears in no committed
    # revision of serve_remote.sh and in no config, so the default was in force
    # throughout. REQUIRE_EXPLICIT rather than a literal, because 0.80 is the
    # correct and deliberate value for the ladder-cost experiment.
    "gpu_memory_utilization": REQUIRE_EXPLICIT,
    "XLA_FLAGS": REQUIRE_EXPLICIT,
    # tpu-inference's own env flag, added 2026-08-10. Default False, and when
    # False `get_attn_req_paddings` returns [max_req_size] — ONE bucket — so
    # attention executes at 256 requests whatever the batch size, and the
    # request ladder the warmup log advertises is not the one attention uses.
    #
    # Swept (M2 measures what enabling it costs), so any value, but it must be
    # stated. It is NOT visible in vLLM's engine-config line, so the audit
    # cannot confirm it the way it confirms prefix caching; the check that it
    # took effect is the warmup log, where "Prepared attn request paddings"
    # prints the full ladder instead of [256].
    "ATTN_BUCKETIZED_NUM_REQS": REQUIRE_EXPLICIT,
    # Decides which implementation serves the model, and therefore whether
    # MOE_ROUTE_PADDING_TO_EXPERT0 below has any effect at all. "auto" resolves
    # GptOssForCausalLM and Qwen3MoeForCausalLM to the vLLM implementation via
    # _VLLM_PREFERRED_ARCHITECTURES; the flax_nnx path for the same model has no
    # padding-awareness whatsoever, so the same flag on the same model would
    # measure nothing. Recorded because "auto" is a resolution, not a value.
    "MODEL_IMPL_TYPE": REQUIRE_EXPLICIT,
    # tpu-inference default False. When False, padding tokens carry whatever
    # top_k returned for their garbage rows and are dispatched to k real experts
    # (4 for gpt-oss-20b, 8 for Qwen3-30B-A3B), which can widen the active
    # expert set and slow the gmm. When True, they collapse onto expert 0 at
    # weight 0. This is the independent variable of the M2 MoE arms, so it is
    # swept -- but an unrecorded value would make a MoE-vs-dense padding cost
    # uninterpretable, because the default is the expensive path.
    #
    # It fails OPEN: if query_start_loc cannot be read from the attention
    # metadata the interface logs a warning once and serves with padding routed
    # normally. So the config value is an intent, and the check that it took
    # effect is the absence of "MOE_ROUTE_PADDING_TO_EXPERT0: failed to read
    # num_valid_tokens" in the server log.
    "MOE_ROUTE_PADDING_TO_EXPERT0": REQUIRE_EXPLICIT,
}


class ControlledVarError(RuntimeError):
    """Raised when a controlled variable is missing or wrong. Always fatal."""


def assert_controlled_vars(config: Mapping[str, Any]) -> None:
    """Abort unless every controlled variable is present and correct.

    Deliberately raises rather than warns: a run that cannot prove prefix
    caching is off must produce no data at all, because a warning in a log is
    not something anyone reads six weeks later while writing a paper.
    """
    controlled = config.get("controlled")
    if not isinstance(controlled, Mapping):
        raise ControlledVarError(
            "config has no 'controlled' block; every config must carry one "
            f"with keys: {sorted(CONTROLLED_VARS)}"
        )

    # A control may be DECLARED as an experiment's independent variable. Session
    # 13's tensor-parallel ablation was designed, deployed, and refused by this
    # function -- `tensor_parallel_size: is 2, must be 4` -- because the contract
    # could not tell an undeclared drift from a deliberate, recorded variation.
    # That is the correct default and the wrong outcome: the experiment that
    # would answer the paper's generality question could not run.
    #
    # The declaration is not an escape hatch. It must name the variable and give
    # a reason, both land in `meta.json`, and the guardrail in paper_numbers.py
    # then sees the arms differ in that field and demands any cross-arm claim
    # assert invariance over it explicitly. Relaxing the contract would have lost
    # that; declaring keeps the variation on the record where a claim must
    # confront it.
    declared = config.get("independent_vars")
    if declared is not None and not isinstance(declared, Mapping):
        raise ControlledVarError(
            "'independent_vars' must be a mapping of {variable: reason}, so the "
            f"reason is recorded with the run; got {type(declared).__name__}"
        )
    declared = dict(declared or {})

    problems: list[str] = []
    unknown_declared = sorted(set(declared) - set(CONTROLLED_VARS))
    if unknown_declared:
        problems.append(
            f"independent_vars names {unknown_declared}, which are not controlled "
            "variables — only a control can be declared as an independent variable"
        )
    for name, reason in declared.items():
        if not isinstance(reason, str) or len(reason.strip()) < 20:
            problems.append(
                f"independent_vars[{name!r}] needs a reason of at least 20 characters "
                "explaining why varying it is the point of this experiment"
            )

    for name, expected in CONTROLLED_VARS.items():
        if name not in controlled:
            problems.append(f"{name}: MISSING (must be stated explicitly)")
            continue
        actual = controlled[name]
        if expected is REQUIRE_EXPLICIT or name in declared:
            continue
        if actual != expected:
            problems.append(
                f"{name}: is {actual!r}, must be {expected!r} "
                f"(declare it in 'independent_vars' if varying it is the experiment)"
            )

    unknown = sorted(set(controlled) - set(CONTROLLED_VARS))
    if unknown:
        problems.append(
            f"unknown controlled vars {unknown} — add them to CONTROLLED_VARS "
            "so they are checked, or remove them"
        )

    # A registered prediction must say WHY the lever moves the target. Provenance
    # checks whether two quantities came from the same configuration; they say
    # nothing about whether the lever you turned acts on the quantity you are
    # claiming about. The tenth failure was exactly that: a smaller model was
    # substituted for a quantized one, but weight bytes and per-token flops both
    # scale with parameter count, so intensity = 2P/2P = 1 and the derivative
    # with respect to the lever is zero. Two lines of algebra, available before
    # the session, and not written down because nothing required it.
    if "note_prediction" in config and not config.get("prediction_mechanism"):
        problems.append(
            "note_prediction is present without 'prediction_mechanism': state the target "
            "as a formula in the lever and say why the derivative is nonzero"
        )

    # The twelfth failure: a D3 (request-dimension) ablation was used to license a
    # D2 (token-dimension) recommendation, and a D3 prediction survived the
    # retraction of the account that made it formulable. Provenance cannot see
    # that, and neither can the lever check -- both quantities are real and both
    # levers move something. What is wrong is that they live in different
    # dimensions of the ladder.
    #
    # Every experiment must name which dimension it measures, so a claim
    # combining two can be rejected mechanically rather than by noticing.
    DIMS = {"D1", "D2", "D3", "none"}
    dim = config.get("dimension")
    if dim is None:
        problems.append(
            "config has no 'dimension': name which quantized dimension this measures "
            f"— one of {sorted(DIMS)} (D1 prompt length, D2 tokens/step, D3 requests/step, "
            "'none' for infrastructure and offline re-analysis)"
        )
    elif dim not in DIMS:
        problems.append(f"dimension {dim!r} is not one of {sorted(DIMS)}")

    if problems:
        raise ControlledVarError(
            "controlled-variable contract violated; refusing to run:\n  - "
            + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

def config_hash(config: Mapping[str, Any]) -> str:
    """Stable 12-hex-char hash of a config.

    Must be invariant to dict ordering and stable across process restarts, so
    that re-running an identical config is detectable. sort_keys gives the
    first; avoiding hash() / id() gives the second.
    """
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _git_state(repo: Path) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=repo, capture_output=True, text=True, timeout=10
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    return {
        "commit": commit,
        # Any uncommitted change means results are not reproducible from the
        # commit alone. Recorded, never silently tolerated.
        "dirty": bool(status) if status is not None else None,
    }


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

@dataclass
class Run:
    """Handle for one experiment run. Created by start_run()."""

    run_id: str
    experiment: str
    config: dict[str, Any]
    dir: Path
    started_at: float
    tables: list[str] = field(default_factory=list)
    _finished: bool = False

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"


def make_run_id(experiment: str, config: Mapping[str, Any], when: float | None = None) -> str:
    """RUN_ID = <experiment>__<UTC timestamp>__<config_hash>.

    Sortable, unique, and self-describing: you can tell what a directory is
    without opening it.
    """
    ts = datetime.fromtimestamp(when if when is not None else time.time(), timezone.utc)
    return f"{experiment}__{ts.strftime('%Y%m%dT%H%M%SZ')}__{config_hash(config)}"


def start_run(
    experiment: str,
    config: Mapping[str, Any],
    results_root: Path | str | None = None,
    check_controlled: bool = True,
) -> Run:
    """Create the run directory and write meta.json BEFORE any work happens.

    Raises ControlledVarError before creating anything if the config violates
    the controlled-variable contract — a bad config leaves no directory behind.
    """
    cfg = json.loads(json.dumps(config, default=str))  # deep copy, JSON-safe

    if check_controlled:
        assert_controlled_vars(cfg)

    root = Path(results_root) if results_root is not None else RESULTS_ROOT
    run_id = make_run_id(experiment, cfg)
    run_dir = root / experiment / run_id

    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}\n"
            "Runs are never overwritten. Delete it deliberately if you mean to."
        )
    run_dir.mkdir(parents=True)

    repo = Path(__file__).resolve().parent.parent
    meta = {
        "run_id": run_id,
        "experiment": experiment,
        "config": cfg,
        "config_hash": config_hash(cfg),
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(repo),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        # Set by the harness when a spot VM is reclaimed. Excluded from paper
        # figures, same rule as git dirty.
        "preempted": False,
        "env": {
            k: os.environ.get(k)
            for k in ("VLLM_TPU_BUCKET_PADDING_GAP", "XLA_FLAGS", "TPU_NAME", "TPU_TYPE")
        },
        "tables": [],
    }
    _write_json_atomic(run_dir / "meta.json", meta)

    return Run(
        run_id=run_id,
        experiment=experiment,
        config=cfg,
        dir=run_dir,
        started_at=time.time(),
    )


def save_table(run: Run, name: str, rows: Sequence[Mapping[str, Any]] | Any) -> Path:
    """Write one result table as Parquet, atomically. Never overwrites.

    Atomicity matters because of spot preemption: a VM can vanish mid-write.
    Writing to a temp file and renaming means a preempted run leaves either a
    complete table or no table — never a truncated Parquet that reads as valid
    data. See is_complete() for the other half of this.
    """
    import pandas as pd  # imported lazily so the module is importable without it

    if name.endswith(".parquet"):
        name = name[: -len(".parquet")]
    path = run.dir / f"{name}.parquet"
    if path.exists():
        raise FileExistsError(f"table already exists, refusing to overwrite: {path}")

    df = rows if hasattr(rows, "to_parquet") else pd.DataFrame(list(rows))
    tmp = path.with_suffix(".parquet.tmp")
    # index=False so an identical config yields a byte-identical file.
    df.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic within a filesystem
    run.tables.append(name)
    return path


# ---------------------------------------------------------------------------
# Preemption / crash recovery
# ---------------------------------------------------------------------------
# A spot VM can be reclaimed at any moment. The rule (plan_v4.md) is that a
# preempted run is excluded from paper figures, same as a dirty git tree. That
# requires being able to tell a complete run from an abandoned one, which is
# what these two functions are for.
#
# The invariant: MANIFEST.jsonl is appended only by finish_run. So a run
# directory with no manifest entry was never finished, regardless of what
# files it contains. Read results via the manifest, never by globbing.

def is_complete(run_dir: Path | str) -> bool:
    """True only if this run finished. Never trust a directory listing."""
    meta = Path(run_dir) / "meta.json"
    if not meta.exists():
        return False
    try:
        return json.loads(meta.read_text()).get("status") in ("ok", "failed")
    except (OSError, json.JSONDecodeError):
        return False


def mark_interrupted_runs(
    results_root: Path | str | None = None, reason: str = "preempted"
) -> list[str]:
    """Find runs left in 'running' state and record them as interrupted.

    Run this at the START of every session, before any new work: anything still
    marked 'running' from a previous session died with the VM. Returns the
    run_ids touched.

    Interrupted runs get status='interrupted' and preempted=True, are appended
    to MANIFEST.jsonl so they are visible rather than silently absent, and are
    excluded from paper figures by the same rule as dirty:true.
    """
    root = Path(results_root) if results_root is not None else RESULTS_ROOT
    if not root.exists():
        return []

    touched: list[str] = []
    for meta_path in sorted(root.glob("*/*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("status") != "running":
            continue

        meta["status"] = "interrupted"
        meta["preempted"] = True
        meta["interrupted_reason"] = reason
        meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(meta_path, meta)

        # Remove any half-written temp tables left by the interrupted process.
        for tmp in meta_path.parent.glob("*.tmp"):
            tmp.unlink(missing_ok=True)

        with (root / MANIFEST_NAME).open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "run_id": meta["run_id"],
                        "experiment": meta["experiment"],
                        "status": "interrupted",
                        "config_hash": meta["config_hash"],
                        "started_at": meta["started_at"],
                        "finished_at": meta["finished_at"],
                        "duration_sec": None,
                        "tables": meta.get("tables", []),
                        "git_commit": meta["git"]["commit"],
                        "git_dirty": meta["git"]["dirty"],
                        "preempted": True,
                        "path": str(meta_path.parent),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        touched.append(meta["run_id"])
    return touched


def usable_runs(results_root: Path | str | None = None) -> list[dict[str, Any]]:
    """Manifest entries safe to put in a paper figure.

    Excludes interrupted/preempted runs and dirty git trees. This is the only
    function analysis code should use to find results.
    """
    return [
        e
        for e in read_manifest(results_root)
        if e["status"] == "ok" and not e.get("preempted") and not e.get("git_dirty")
    ]


def finish_run(
    run: Run,
    status: str = "ok",
    error: BaseException | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Finalise meta.json and append to MANIFEST.jsonl. Safe to call twice."""
    if run._finished:
        return
    run._finished = True

    meta = json.loads(run.meta_path.read_text())
    meta["status"] = status
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["duration_sec"] = round(time.time() - run.started_at, 3)
    meta["tables"] = list(run.tables)
    if error is not None:
        meta["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
    if extra:
        meta.update(extra)
    _write_json_atomic(run.meta_path, meta)

    manifest = run.dir.parent.parent / MANIFEST_NAME
    manifest.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "run_id": run.run_id,
        "experiment": run.experiment,
        "status": status,
        "config_hash": meta["config_hash"],
        "started_at": meta["started_at"],
        "finished_at": meta["finished_at"],
        "duration_sec": meta["duration_sec"],
        "tables": meta["tables"],
        "git_commit": meta["git"]["commit"],
        "git_dirty": meta["git"]["dirty"],
        "preempted": meta.get("preempted", False),
        "path": str(run.dir),
    }
    with manifest.open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, obj: Any) -> None:
    """Write via a temp file + rename so a crash never leaves half a meta.json."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path | str) -> dict[str, Any]:
    """Load a JSON config. YAML is deliberately not supported — one less
    dependency on the TPU VM, and configs are small."""
    return json.loads(Path(path).read_text())


def read_manifest(results_root: Path | str | None = None) -> list[dict[str, Any]]:
    root = Path(results_root) if results_root is not None else RESULTS_ROOT
    manifest = root / MANIFEST_NAME
    if not manifest.exists():
        return []
    return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    # Session-start recovery. Run this FIRST every session, before new work:
    #   python scripts/_common.py --mark-interrupted results/
    # Anything still marked 'running' died with a previous VM.
    import argparse

    ap = argparse.ArgumentParser(description="traceability contract utilities")
    ap.add_argument("--mark-interrupted", metavar="RESULTS_DIR", type=Path)
    ap.add_argument("--reason", default="preempted")
    ap.add_argument("--list-usable", metavar="RESULTS_DIR", type=Path)
    a = ap.parse_args()

    if a.mark_interrupted:
        ids = mark_interrupted_runs(a.mark_interrupted, reason=a.reason)
        print(f"marked {len(ids)} interrupted run(s) as preempted")
        for i in ids:
            print(f"  {i}")
    if a.list_usable:
        rows = usable_runs(a.list_usable)
        print(f"{len(rows)} usable run(s) (status=ok, not preempted, clean tree)")
        for r in rows:
            print(f"  {r['run_id']:<55} {','.join(r['tables'])}")
