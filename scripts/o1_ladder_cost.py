#!/usr/bin/env python3
"""
O1 — what does a shape cost, and what does it buy? The paper's one optimisation.

Every measurement in this work says run-time padding is close to free and that
shape coverage is paid at warmup. That is a characterisation, and a
characterisation implies an action nobody has quantified: **if padding is free
and shapes cost warmup, the ladder should be as COARSE as the workload tolerates,
not as fine as the stack can compile.**

`VLLM_TPU_BUCKET_PADDING_GAP` is the lever. Unset, vLLM's TPU backend compiles an
exponential token ladder of 10 shapes; set to 512 it compiles a linear ladder of
21. A finer ladder pads less — the whole premise of length bucketing — so if the
premise were true, the 21-shape ladder should serve faster. If this paper is
right, it should serve identically and cost roughly twice the warmup.

    fine ladder faster        -> padding is paid after all, somewhere we have not
                                 looked, and the paper's central claim is wrong
    fine ladder same speed,
    more warmup               -> the coarse ladder strictly dominates, and the
                                 saving is the paper's first positive result

MEASURED PER ARM:
  * time from process start to the server accepting requests (warmup)
  * number of compiled token shapes, parsed from the warmup log
  * steady-state latency at matched load, so "no latency cost" is a measurement
    rather than an assumption

WHY THIS IS THE RIGHT OPTIMISATION TO TRY. It follows from the finding rather
than being bolted on, it needs no patch to vLLM — only a documented environment
variable — and it is falsifiable in the direction that would hurt: if the fine
ladder wins on latency, the paper's premise collapses.

Usage:
  python scripts/o1_ladder_cost.py --config configs/o1_ladder_cost.json \\
      --base-url http://localhost:8000 --arm default --warmup-seconds 344
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _client import complete  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import metrics_available  # noqa: E402


def ladder_from_log(path: pathlib.Path) -> tuple[list[int], list[int]]:
    """The compiled token and request ladders the server printed at boot."""
    tok: list[int] = []
    req: list[int] = []
    if not path.exists():
        return tok, req
    txt = path.read_text(errors="replace")
    m = re.search(r"Prepared token paddings: \[([^\]]*)\]", txt)
    if m:
        tok = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
    m = re.search(r"Prepared request paddings: \[([^\]]*)\]", txt)
    if m:
        req = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
    return tok, req


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--arm", required=True, choices=["default", "gap512"])
    ap.add_argument("--warmup-seconds", type=float, required=True,
                    help="measured externally: process start -> /health 200")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    cfg = dict(load_config(args.config))
    cfg["arm"] = args.arm
    if not metrics_available(args.base_url):
        print(f"[o1] no /metrics at {args.base_url}", file=sys.stderr)
        return 1
    try:
        run = start_run("o1_ladder_cost", cfg, results_root=args.results_root)
    except ControlledVarError as e:
        print(f"[o1] {e}", file=sys.stderr)
        return 2

    status, err = "ok", None
    try:
        tok, req = ladder_from_log(pathlib.Path(cfg["warmup_log_path"]))
        print(f"[o1:{args.arm}] warmup {args.warmup_seconds:.1f} s   "
              f"{len(tok)} token shapes {tok[:4]}...{tok[-2:]}   "
              f"{len(req)} request shapes")

        # Steady-state latency at matched load, so "no latency cost" is measured.
        rows = []
        for plen in cfg["prompt_lens"]:
            lat = []
            for rep in range(cfg["repeats"] + cfg["warmup_discard"]):
                with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
                    out = list(pool.map(
                        lambda i: complete(args.base_url, cfg["model"], plen,
                                           cfg["output_len"], seed=rep * 100 + i),
                        range(cfg["concurrency"])))
                if rep < cfg["warmup_discard"]:
                    continue
                ok = [s for s in out if s.ok]
                bad = [s for s in out if not s.ok]
                if bad and rep == cfg["warmup_discard"]:
                    # Report loudly. A previous run produced an empty latency
                    # table that was indistinguishable from "no requests were
                    # made", because failures were discarded in silence.
                    print(f"[o1:{args.arm}]   prompt_len={plen} "
                          f"{len(bad)}/{len(out)} FAILED: {bad[0].error!r}",
                          file=sys.stderr)
                if ok:
                    lat.append(statistics.median([s.total_ms for s in ok]))
            if lat:
                rows.append({"arm": args.arm, "prompt_len": plen,
                             "e2e_ms_median": statistics.median(lat),
                             "e2e_ms_min": min(lat), "reps": len(lat)})
                print(f"[o1:{args.arm}]   prompt_len={plen:<5} "
                      f"e2e {statistics.median(lat):8.1f} ms")
        if not rows:
            # Never return success with an empty headline table.
            raise RuntimeError(
                "no latency rows: every request failed at every prompt length. "
                "See the per-cell errors above; the arm is not measured.")
        save_table(run, "latency", rows)
        save_table(run, "ladder", [{
            "arm": args.arm, "warmup_s": args.warmup_seconds,
            "n_token_shapes": len(tok), "n_request_shapes": len(req),
            "token_shapes": json.dumps(tok), "request_shapes": json.dumps(req),
            "warmup_s_per_token_shape": args.warmup_seconds / len(tok) if tok else float("nan"),
        }])
        print(f"[o1] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
