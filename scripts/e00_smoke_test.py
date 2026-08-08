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
                    by anyone without a TPU. Exercises every code path except
                    the HTTP call.
  --warmup-log F    parse a real log captured from a previous server start.
  (default)         query a live vLLM server over the OpenAI-compatible API.

Usage:
  python scripts/e00_smoke_test.py --config configs/e00_default_ladder.json --mock
  python scripts/e00_smoke_test.py --config configs/e00_default_ladder.json \
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
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


def fetch_server_config(base_url: str, timeout: float = 30.0) -> dict[str, Any]:
    """Read the server's own view of its configuration."""
    url = base_url.rstrip("/") + "/v1/models"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def audit_controlled(config: dict[str, Any], server: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Compare recorded controlled vars against the server's reported values.

    Any key the server does not expose is reported as 'unverified' rather than
    silently passed — an unverifiable controlled variable is a known hole, and
    the paper should say which ones they are.
    """
    rows: list[dict[str, Any]] = []
    reported = (server or {}).get("controlled", {}) if server else {}
    for name, recorded in sorted(config["controlled"].items()):
        if not server:
            verdict = "mock"
        elif name not in reported:
            verdict = "unverified"
        elif reported[name] != recorded:
            verdict = "MISMATCH"
        else:
            verdict = "ok"
        rows.append(
            {
                "variable": name,
                "recorded": json.dumps(recorded),
                "server_reported": json.dumps(reported.get(name)) if server else None,
                "verdict": verdict,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true", help="no server; synthetic warmup log")
    ap.add_argument("--warmup-log", type=Path, help="parse a captured warmup log")
    ap.add_argument("--base-url", default="http://localhost:8000")
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
            server = None
        elif args.warmup_log:
            log_lines = args.warmup_log.read_text().splitlines()
            server = None
        else:
            log_lines = Path(config["warmup_log_path"]).read_text().splitlines()
            server = fetch_server_config(args.base_url)

        observed = parse_warmup_log(log_lines)

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
