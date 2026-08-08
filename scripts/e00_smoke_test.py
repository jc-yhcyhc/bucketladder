#!/usr/bin/env python3
"""
e00 — W0b gate: ladder enumeration + controlled-variable audit.

Two jobs, and the run fails if either does:

  1. Enumerate the bucket ladder the server actually compiled, by parsing the
     warmup log, and check it against scripts/ladder.py's prediction. If they
     disagree, our model of the independent variable is wrong and every
     downstream experiment is measuring something we do not understand.

  2. Prove the controlled-variable contract is live: the recorded config must
     match the server's reported config, and a config with prefix caching in an
     unrecorded state must abort at start_run rather than warn.

Runs in three modes:

  --mock            no server, synthetic warmup log. Used by the test suite and
                    by anyone without a TPU. NOTE the mock log is generated FROM
                    ladder.py's prediction, so passing in mock mode proves the
                    parser and comparison are wired up — it does NOT prove real
                    vLLM output matches the regexes. Only a real run does that.
  --warmup-log F    parse a real log captured from a previous server start.
                    This is the mode that first meets reality.
  (default)         read the log at config["warmup_log_path"], written by
                    infra/vm_setup.sh.

Both real modes also parse vLLM's resolved engine-config line out of the same
log and audit it against the recorded controlled variables.

Usage:
  python scripts/e00_smoke_test.py --config configs/e00_default_ladder.json --mock
  python scripts/e00_smoke_test.py --config configs/e00_default_ladder.json \
      --warmup-log /tmp/vllm_warmup.log
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    ControlledVarError,
    finish_run,
    load_config,
    save_table,
    start_run,
)
from ladder import build_ladder  # noqa: E402

# vLLM logs compiled shapes during warmup. Both spellings seen in the wild;
# the parser is deliberately permissive because this is exactly the kind of
# thing that silently changes between releases.
_BUCKET_PATTERNS = [
    re.compile(r"[Cc]ompil\w*.*?num_tokens[=: ]+(\d+)"),
    re.compile(r"[Ww]arm(?:ing )?up.*?shape[=: ]+\((\d+)"),
    re.compile(r"bucket[=: ]+(\d+)"),
]


def parse_warmup_log(lines: Iterable[str]) -> list[int]:
    """Extract compiled bucket sizes from a vLLM TPU warmup log."""
    found: set[int] = set()
    for line in lines:
        for pat in _BUCKET_PATTERNS:
            for m in pat.finditer(line):
                found.add(int(m.group(1)))
    return sorted(found)


def mock_warmup_log(max_model_len: int, padding_gap: Any) -> list[str]:
    """Synthetic log matching what ladder.py predicts.

    Deliberately generated FROM the predicted ladder so the mock path proves
    the parser and the comparison work. It cannot prove the real server agrees
    — only the on-TPU run does that, which is why this script's gate output
    records which mode produced it.
    """
    ladder = build_ladder(max_model_len, padding_gap)
    lines = ["INFO vllm.worker: starting XLA warmup"]
    lines += [
        f"INFO vllm.worker: Compiling graph for num_tokens={b} (batch=1)" for b in ladder
    ]
    lines.append("INFO vllm.worker: warmup complete")
    return lines


# vLLM logs its resolved engine configuration at startup, e.g.
#   INFO ... Initializing an LLM engine (v0.11.0) with config: model='...',
#   tensor_parallel_size=4, enable_prefix_caching=False, max_model_len=8192, ...
# That line is the audit's source of truth. An earlier version of this script
# queried /v1//models for a "controlled" block; that endpoint returns model
# cards and no such block exists, so the audit was a no-op that reported
# everything as "unverified". Parsing the engine config line actually works.
_ENGINE_CONFIG_RE = re.compile(r"with config:\s*(.+)$")
_KV_RE = re.compile(r"(\w+)=('[^']*'|\"[^\"]*\"|[^,\s]+)")

# Our controlled-variable name -> the name(s) vLLM may log it under.
_VLLM_ALIASES: dict[str, tuple[str, ...]] = {
    "enable_prefix_caching": ("enable_prefix_caching",),
    "enable_chunked_prefill": ("enable_chunked_prefill", "chunked_prefill_enabled"),
    "max_num_batched_tokens": ("max_num_batched_tokens",),
    "tensor_parallel_size": ("tensor_parallel_size",),
    "speculative_model": ("speculative_model", "speculative_config"),
    "kv_cache_dtype": ("kv_cache_dtype",),
    "max_model_len": ("max_model_len",),
    # Environment variables — vLLM does not echo these into the engine config
    # line, so they are structurally unverifiable from the log. Reported as
    # such rather than quietly passed.
    "VLLM_TPU_BUCKET_PADDING_GAP": (),
    "XLA_FLAGS": (),
}


def _coerce(raw: str) -> Any:
    """Turn a logged token into a Python value for comparison."""
    s = raw.strip().strip("'\"")
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_server_config(lines: Iterable[str]) -> dict[str, Any]:
    """Extract vLLM's resolved engine config from its startup log."""
    out: dict[str, Any] = {}
    for line in lines:
        m = _ENGINE_CONFIG_RE.search(line)
        if not m:
            continue
        for k, v in _KV_RE.findall(m.group(1)):
            out[k] = _coerce(v)
    return out


def audit_controlled(
    config: dict[str, Any], server: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Compare recorded controlled vars against the server's own reported config.

    `server` is the dict from parse_server_config, or None in mock mode.

    A variable the server does not report is 'unverified', never 'ok'. Those
    are known holes and the paper should name them rather than imply every
    controlled variable was checked.
    """
    rows: list[dict[str, Any]] = []
    for name, recorded in sorted(config["controlled"].items()):
        reported: Any = None
        found = False
        if server is not None:
            for alias in _VLLM_ALIASES.get(name, (name,)):
                if alias in server:
                    reported, found = server[alias], True
                    break

        if server is None:
            verdict = "mock"
        elif not found:
            verdict = "unverified"
        elif reported != recorded:
            verdict = "MISMATCH"
        else:
            verdict = "ok"

        rows.append(
            {
                "variable": name,
                "recorded": json.dumps(recorded),
                "server_reported": json.dumps(reported) if found else None,
                "verdict": verdict,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true", help="no server; synthetic warmup log")
    ap.add_argument("--warmup-log", type=Path, help="parse a captured warmup log")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    config = load_config(args.config)
    mode = "mock" if args.mock else ("log" if args.warmup_log else "live")
    config["mode"] = mode

    # start_run asserts the controlled-variable contract and will raise before
    # creating any directory if the config is non-compliant.
    run = start_run("e00_smoke_test", config, results_root=args.results_root)
    status, err = "ok", None
    try:
        controlled = config["controlled"]
        max_model_len = controlled["max_model_len"]
        padding_gap = controlled["VLLM_TPU_BUCKET_PADDING_GAP"]

        predicted = build_ladder(max_model_len, padding_gap)

        if args.mock:
            log_lines = mock_warmup_log(max_model_len, padding_gap)
        elif args.warmup_log:
            log_lines = args.warmup_log.read_text().splitlines()
        else:
            log_lines = Path(config["warmup_log_path"]).read_text().splitlines()

        observed = parse_warmup_log(log_lines)
        # Mock mode has no real server to audit against; every other mode reads
        # vLLM's own resolved engine config out of the same log.
        server = None if args.mock else parse_server_config(log_lines)

        save_table(
            run,
            "ladder",
            [
                {"index": i, "bucket": b, "source": "predicted"}
                for i, b in enumerate(predicted)
            ]
            + [
                {"index": i, "bucket": b, "source": "observed"}
                for i, b in enumerate(observed)
            ],
        )
        audit_rows = audit_controlled(config, server)
        save_table(run, "controlled_audit", audit_rows)

        # --- gate ---
        problems: list[str] = []
        if not observed:
            problems.append("warmup log yielded no buckets — parser or log is wrong")
        if observed and observed != predicted:
            problems.append(
                f"ladder mismatch: predicted {predicted}, observed {observed}"
            )
        bad = [r for r in audit_rows if r["verdict"] == "MISMATCH"]
        if bad:
            problems.append(f"controlled-variable mismatch: {[r['variable'] for r in bad]}")

        (run.dir / "ladder_default.json").write_text(
            json.dumps({"mode": mode, "ladder": observed or predicted}, indent=2) + "\n"
        )

        if problems:
            status = "failed"
            print("GATE FAILED:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1

        unverified = [r["variable"] for r in audit_rows if r["verdict"] == "unverified"]
        print(f"[e00] mode={mode} buckets={len(observed or predicted)} -> {observed or predicted}")
        if unverified:
            print(f"[e00] WARNING unverifiable against server: {unverified}")
        print(f"[e00] GATE PASSED  run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001 — recorded, then re-raised via finally
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
