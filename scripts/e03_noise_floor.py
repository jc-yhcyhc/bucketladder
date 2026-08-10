#!/usr/bin/env python3
"""
e03 — noise floor. Runs FIRST in every hardware session.

Nothing downstream is interpretable without this. A "5% gap" or a "flatness of
0.9" means nothing until we know whether repeat-to-repeat variation is 1% or 8%.
Every threshold in e01/e02 is expressed as a multiple of the CV measured here
(notes/session_plan.md).

Measures two different noises, because they are not the same number:

  within-run    repeats against one already-warm server. The floor for
                comparing two lengths in the same e01 sweep.
  across-restart  the server is restarted between blocks. The floor for
                comparing anything measured on different days, which every
                multi-session result does. Expected to be larger, and if it is
                large it constrains the whole experimental design.

Usage:
  python scripts/e03_noise_floor.py --config configs/e03_noise_floor.json --mock
  python scripts/e03_noise_floor.py --config configs/e03_noise_floor.json \
      --base-url http://localhost:8000 --restart-block 0
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _client import complete, complete_mock, summarise  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from ladder import build_ladder  # noqa: E402

# WARMUP DISCARD. Measured on hardware 2026-08-09: the first request after a
# server start costs ~116 ms against a ~15.4 ms steady state (7.5x), even though
# vLLM has already logged "Application startup complete". Including it puts the
# run-to-run CV at 97%; discarding it gives 1.7%, and discarding 5 gives 0.8%.
# Every cell therefore fires `warmup_discard` unrecorded requests first.



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--restart-block", type=int, default=0,
                    help="index of this server restart; run once per restart to get the across-restart floor")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    config = load_config(args.config)
    config["mode"] = "mock" if args.mock else "live"
    config["restart_block"] = args.restart_block
    controlled = config["controlled"]
    ladder = build_ladder(controlled["max_num_batched_tokens"], controlled["VLLM_TPU_BUCKET_PADDING_GAP"])

    probe_len = config.get("probe_len", 512)
    output_len = config.get("output_len", 1)
    repeats = config.get("repeats", 30)
    discard = config.get("warmup_discard", 5)

    run = start_run("e03_noise_floor", config, results_root=args.results_root)
    status, err = "ok", None
    try:
        rows: list[dict[str, Any]] = []
        samples = []
        for rep in range(-discard, repeats):
            if args.mock:
                # Vary the seed per restart block so the mock exhibits a
                # genuine across-restart offset to detect.
                s = complete_mock(probe_len, output_len, ladder=ladder,
                                  seed=rep + 1000 * args.restart_block)
            else:
                s = complete(args.base_url, config["model"], probe_len, output_len, seed=rep)
            if rep < 0:
                continue  # warmup, not recorded — see note at top of file
            samples.append(s)
            rows.append({"restart_block": args.restart_block, "repeat": rep, **s.as_row()})
        save_table(run, "samples", rows)

        st = summarise(samples)
        save_table(run, "noise_floor", [{"restart_block": args.restart_block,
                                         "probe_len": probe_len, **st}])

        print(f"[e03] block={args.restart_block} n={st['n']} probe_len={probe_len}")
        print(f"[e03]   median={st['median']:.2f} ms  mean={st['mean']:.2f}  sd={st['stdev']:.2f}")
        print(f"[e03]   WITHIN-RUN CV = {st['cv'] * 100:.2f}%")
        print(f"[e03]   -> a difference is only credible above ~{3 * st['cv'] * 100:.1f}% (3x CV)")
        if args.restart_block == 0:
            print("[e03] NOTE this is the WITHIN-RUN floor only. solidity.md R2 requires the")
            print("[e03]   ACROSS-RESTART floor before any headline effect is believed:")
            print("[e03]   restart the server and re-run with --restart-block 1, then 2,")
            print("[e03]   and combine with across_restart_cv().")
        print(f"[e03] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


def across_restart_cv(block_medians: list[float]) -> float:
    """CV of the per-restart medians — the floor for cross-session comparison.

    Fed the `median` column of each block's noise_floor table. Kept as a plain
    function so it can be applied later, offline, once several blocks exist.
    """
    vals = [v for v in block_medians if not math.isnan(v)]
    if len(vals) < 2:
        return math.nan
    mean = statistics.fmean(vals)
    return statistics.stdev(vals) / mean if mean else math.nan


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ControlledVarError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        sys.exit(2)
