#!/usr/bin/env python3
"""
O3 — where does the finer ladder stop paying?

§4.9 measured a finer token ladder winning at two concurrent requests: -8.7% and
-12.5% end-to-end at prompt lengths that straddle ladder entries, bought with
+53% cold startup and 15.0% of KV cache capacity. That is a recommendation with a
hole in it, because it is measured at one concurrency and the whole argument for
*why* it should stop paying is a statement about concurrency: §4.3 finds the paid
share of token padding at 23.1% of nominal at batch 4, falling to indistinguishable
from zero by 16. A deployment cannot act on "it pays at low concurrency" without
knowing where low ends.

So sweep it. Same two ladders, same prompt lengths, concurrency 1 -> 16.

REGISTERED PREDICTION. The gap512-minus-default difference is negative at n=1-2,
shrinks monotonically in magnitude, and crosses zero somewhere between n=4 and
n=16, tracking §4.3's paid-share curve. If instead the benefit persists at n=16,
the paid-share curve and this measurement disagree and one of them is wrong. If
it is already zero at n=4, the §4.9 recommendation is narrower than stated and
applies only to near-serial traffic.

WHAT CHANGES ABOVE n=2, AND WHY THE PLACEBO MATTERS MORE HERE. At n<=2 each
prefill is its own dispatch (300..3000 tokens, all under max_num_batched_tokens
= 8192), so a request's own length picks its compiled shape and the ladder
mapping in §4.9 is exact. Above that, chunked prefill packs several requests into
one step and the shape is chosen for the PACKED step, not the request -- so the
per-request "pads to" column stops being meaningful and the arms can no longer be
told apart by arithmetic on prompt length alone. The placebo cell carries the
design through: prompt 300 pads to 512 on both ladders at every concurrency, so
whatever it shows at a given n is that n's arm-level offset, and the treated
cells are read against it. This is why the placebo is swept too rather than
measured once.

Usage:
  python scripts/o3_concurrency_crossover.py --config configs/o3_concurrency_crossover.json \\
      --arm default --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _client import complete  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import metrics_available  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--arm", required=True, choices=["default", "gap512"])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    cfg = dict(load_config(args.config))
    cfg["arm"] = args.arm
    if not metrics_available(args.base_url):
        print(f"[o3] no /metrics at {args.base_url}", file=sys.stderr)
        return 1
    try:
        run = start_run("o3_concurrency_crossover", cfg, results_root=args.results_root)
    except ControlledVarError as e:
        print(f"[o3] {e}", file=sys.stderr)
        return 2

    status, err = "ok", None
    try:
        reps: list[dict] = []
        rows: list[dict] = []
        for n in cfg["concurrency_levels"]:
            for plen in cfg["prompt_lens"]:
                lat: list[float] = []
                for rep in range(cfg["repeats"] + cfg["warmup_discard"]):
                    with ThreadPoolExecutor(max_workers=n) as pool:
                        out = list(pool.map(
                            lambda i: complete(args.base_url, cfg["model"], plen,
                                               cfg["output_len"], seed=rep * 100 + i),
                            range(n)))
                    if rep < cfg["warmup_discard"]:
                        continue
                    ok = [s for s in out if s.ok]
                    bad = [s for s in out if not s.ok]
                    if bad and rep == cfg["warmup_discard"]:
                        print(f"[o3:{args.arm}] n={n} plen={plen} "
                              f"{len(bad)}/{len(out)} FAILED: {bad[0].error!r}",
                              file=sys.stderr)
                    if ok:
                        # Median over the concurrent requests of one repeat: the
                        # batch is the unit of observation, since they share a step.
                        lat.append(statistics.median([s.total_ms for s in ok]))
                        reps.append({"arm": args.arm, "concurrency": n,
                                     "prompt_len": plen, "rep": rep,
                                     "e2e_ms": lat[-1]})
                if lat:
                    rows.append({"arm": args.arm, "concurrency": n, "prompt_len": plen,
                                 "e2e_ms_median": statistics.median(lat),
                                 "e2e_ms_min": min(lat), "reps": len(lat)})
                    print(f"[o3:{args.arm}] n={n:<3} plen={plen:<5} "
                          f"e2e {statistics.median(lat):8.1f} ms")
        if not rows:
            raise RuntimeError(
                "no latency rows: every request failed at every cell. See the "
                "per-cell errors above; the arm is not measured.")
        save_table(run, "latency", rows)
        save_table(run, "latency_reps", reps)
        print(f"[o3] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
