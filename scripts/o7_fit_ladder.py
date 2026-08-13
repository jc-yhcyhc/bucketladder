#!/usr/bin/env python3
"""
O7 — choosing a ladder from a length distribution, and testing the choice.

§4.9 recommends placing compiled entries where the workload straddles them, and
demonstrates it with a placement chosen by knowing two prompt lengths in advance.
That is an existence proof, not a method. A method has to start from a length
distribution, pick a ladder without looking at latency, and then be right.

So: sample a lognormal workload, compute expected padded tokens per request for
each ladder the stack can compile, and predict the latency reduction from the
per-padded-token cost §4.9 already measured (34.9-35.3 us at this concurrency).
The prediction is quantitative and registered in the config -- 10.4, 16.2 and
19.0 ms against the default for gaps 1024, 512 and 256 -- so this can fail
against a number rather than against a direction.

TWO THINGS THIS GUARDS.

The ladder rule is verified, not assumed. Predicted padding comes from "powers of
two to 1024, then 1024 + k*gap", inferred from three boot logs. Each arm re-reads
the ladder the server actually printed and **aborts** if it differs, because a
prediction computed from the wrong ladder would be checked against the right one
and quietly disagree.

The workload is replayed, not resampled. The same sampled lengths run against
every arm in the same order, so arms are paired on the workload and differ only in
which shapes exist. Resampling per arm would put the variance of a heavy-tailed
distribution straight into the comparison.

Usage:
  python scripts/o7_fit_ladder.py --config configs/o7_fit_ladder.json \\
      --gap 1024 --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _client import complete  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import metrics_available  # noqa: E402


def predicted_ladder(gap: int | None) -> list[int]:
    if gap is None:
        return [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    lad = [16, 32, 64, 128, 256, 512, 1024]
    x = 1024 + gap
    while x < 8192:
        lad.append(x)
        x += gap
    lad.append(8192)
    return lad


def actual_ladder(path: pathlib.Path) -> list[int]:
    if not path.exists():
        return []
    m = re.search(r"Prepared token paddings: \[([^\]]*)\]", path.read_text(errors="replace"))
    return [int(x) for x in m.group(1).replace(" ", "").split(",") if x] if m else []


def pad_to(lad: list[int], n: int) -> int:
    for s in lad:
        if s >= n:
            return s
    return lad[-1]


def sample_lengths(w: dict) -> list[int]:
    rng = random.Random(w["seed"])
    out = []
    while len(out) < w["n_prompts"]:
        v = int(rng.lognormvariate(math.log(w["median_len"]), w["sigma"]))
        out.append(max(w["min_len"], min(w["max_len"], v)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--gap", required=True,
                    help="'none' for the default exponential ladder, else the gap")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    gap = None if args.gap.lower() in ("none", "null", "") else int(args.gap)
    cfg = dict(load_config(args.config))
    cfg["arm"] = f"gap{gap}" if gap else "default"
    cfg["controlled"] = dict(cfg["controlled"])
    cfg["controlled"]["VLLM_TPU_BUCKET_PADDING_GAP"] = str(gap) if gap else ""

    if not metrics_available(args.base_url):
        print(f"[o7] no /metrics at {args.base_url}", file=sys.stderr)
        return 1

    want = predicted_ladder(gap)
    got = actual_ladder(pathlib.Path(cfg["warmup_log_path"]))
    if not got:
        print("[o7] could not read the compiled ladder from the warmup log; "
              "refusing to check a prediction against an unknown ladder",
              file=sys.stderr)
        return 3
    if got != want:
        print(f"[o7] LADDER MISMATCH — predicted padding was computed from\n"
              f"     {want}\n  but the server compiled\n     {got}\n"
              "  The offline prediction does not describe this arm. Aborting rather "
              "than reporting a comparison against the wrong model.", file=sys.stderr)
        return 4

    try:
        run = start_run("o7_fit_ladder", cfg, results_root=args.results_root)
    except ControlledVarError as e:
        print(f"[o7] {e}", file=sys.stderr)
        return 2

    status, err = "ok", None
    try:
        lens = sample_lengths(cfg["workload"])
        pads = [pad_to(got, n) - n for n in lens]
        mean_pad = statistics.mean(pads)
        print(f"[o7:{cfg['arm']}] {len(got)} shapes   mean padded "
              f"{mean_pad:.0f} tok/req over {len(lens)} prompts")

        n = cfg["concurrency"]
        lat: list[float] = []
        reps: list[dict] = []
        for rep in range(cfg["repeats"] + cfg["warmup_discard"]):
            batch: list[float] = []
            for i in range(0, len(lens), n):
                grp = lens[i:i + n]
                with ThreadPoolExecutor(max_workers=len(grp)) as pool:
                    out = list(pool.map(
                        lambda t: complete(args.base_url, cfg["model"], t[1],
                                           cfg["output_len"], seed=rep * 1000 + t[0]),
                        list(enumerate(grp))))
                ok = [s for s in out if s.ok]
                if not ok and rep == cfg["warmup_discard"]:
                    print(f"[o7:{cfg['arm']}] group at {i} all failed: "
                          f"{out[0].error!r}", file=sys.stderr)
                batch.extend(s.total_ms for s in ok)
            if rep < cfg["warmup_discard"]:
                continue
            if batch:
                lat.append(statistics.mean(batch))
                reps.append({"arm": cfg["arm"], "rep": rep, "mean_ms": lat[-1],
                             "n_ok": len(batch)})
        if not lat:
            raise RuntimeError("no completed requests; arm not measured")
        row = {"arm": cfg["arm"], "gap": gap if gap else 0,
               "n_shapes": len(got), "mean_padded_tokens": mean_pad,
               "mean_ms": statistics.mean(lat), "reps": len(lat)}
        save_table(run, "fit", [row])
        save_table(run, "fit_reps", reps)
        print(f"[o7:{cfg['arm']}] mean e2e {statistics.mean(lat):8.1f} ms "
              f"over {len(lat)} replays")
        print(f"[o7] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
