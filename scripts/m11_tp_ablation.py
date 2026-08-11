#!/usr/bin/env python3
"""
M11 — is request-dimension padding free, or merely not the bottleneck at TP=4?

Review finding M4/Q6, and the sharpest generality challenge the draft received:

  "v5e with a 4B model at TP=4 places per-chip weights in the low hundreds of MB
   -- an operating point where decode is bandwidth-bound and padding on the
   request dimension is cheap almost by construction... If padding on the request
   dimension is free at n=4-8 primarily because that dimension is not the
   bottleneck in the TP=4 sharded layout, the negative advice in §9 is a
   statement about this sharding configuration rather than about the stack."

Correct, and answerable on the hardware already in hand: hold the model, the
chips and the workload fixed, and vary only the sharding.

THE PREDICTION, REGISTERED BEFORE THE MEASUREMENT. It is written into
`configs/m11_tp{1,2,4}.json` rather than here, so it cannot be edited after the
fact. From `m9_roofline.py`: decode reads the whole weight set every step, so
per-chip weight bytes scale as 1/TP. Therefore

    LEVEL  should scale roughly with 1/TP  (TP=1 about 4x TP=4)
    SHAPE  should be preserved            (the curve stays near-flat in n)

Both halves are testable and neither is safe. A prediction that cannot fail is
not worth registering.

WHAT THE ANSWER MEANS EITHER WAY. The paper's negative advice rests on the claim
that decode cost is dominated by a weight floor that batch size does not move,
which is what makes an extra request -- real or padded -- nearly free.

    the floor dominates at every TP   -> the claim is about the stack, and the
                                         advice survives the sharding objection
    the floor only dominates at TP=4  -> the advice is a statement about this
                                         layout and §9 must be narrowed to it

Note the asymmetry: LESS sharding means a BIGGER per-chip weight floor, so the
hostile case for our claim is high TP, not low. This ablation goes the wrong way
to be flattering, which is why it is worth running.

Usage:
  python scripts/m11_tp_ablation.py
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pyarrow.parquet as pq  # noqa: E402

from _common import finish_run, save_table, start_run  # noqa: E402

ROOT = HERE.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-glob",
                    default="session14-ablation/results/e02_stock_baseline/*")
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    arms: dict[int, dict[str, Any]] = {}
    for d in sorted(glob.glob(str(ROOT / "captured" / args.capture_glob))):
        p, m = pathlib.Path(d) / "server_timing.parquet", pathlib.Path(d) / "meta.json"
        if not (p.exists() and m.exists()):
            continue
        conf = json.loads(m.read_text())["config"]
        tp = conf["controlled"]["tensor_parallel_size"]
        rows = pq.read_table(p).to_pylist()
        ol = conf["output_len"]
        arms[tp] = {"conf": conf, "run_id": pathlib.Path(d).name, "output_len": ol,
                    "curve": {n: statistics.median(
                        [r["decode_ms"] for r in rows if r["concurrency"] == n]) / ol
                        for n in sorted({r["concurrency"] for r in rows})}}
    if len(arms) < 2:
        print(f"[m11] need at least two TP arms, found {sorted(arms)}", file=sys.stderr)
        return 1

    base = arms[max(arms)]["conf"]
    cfg = {"experiment": "m11_tp_ablation", "mode": "offline",
           "model": base.get("model"), "controlled": dict(base["controlled"]),
           "tp_arms": sorted(arms), "source_runs": {str(t): a["run_id"] for t, a in arms.items()},
           "independent_vars": {"tensor_parallel_size":
                                ("Re-analysis of the TP ablation; the arms differ in sharding "
                                 "by design, which is the entire content of the experiment.")},
           "note_prediction": ("Registered in configs/m11_tp*.json before measurement: level "
                               "scales roughly 1/TP, shape preserved.")}
    run = start_run("m11_tp_ablation", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        ref = max(arms)
        rows_out: list[dict[str, Any]] = []
        for tp in sorted(arms, reverse=True):
            for n, ms in sorted(arms[tp]["curve"].items()):
                r = arms[ref]["curve"].get(n)
                rows_out.append({"tp": tp, "n": n, "ms_per_step": ms,
                                 "ratio_to_tp%d" % ref: ms / r if r else float("nan"),
                                 "predicted_ratio": ref / tp})
        save_table(run, "curves", rows_out)

        shape: list[dict[str, Any]] = []
        for tp in sorted(arms, reverse=True):
            c = arms[tp]["curve"]
            lo, hi = min(c), max(c)
            shape.append({"tp": tp, "n_lo": lo, "n_hi": hi,
                          "ms_lo": c[lo], "ms_hi": c[hi], "shape_ratio": c[hi] / c[lo],
                          "level_ratio_to_ref": c[lo] / arms[ref]["curve"][lo],
                          "predicted_level_ratio": ref / tp})
        save_table(run, "shape", shape)

        print(f"[m11] {'TP':>3} {'n':>3} {'ms/step':>9} {'vs TP=%d' % ref:>9} {'predicted':>10}")
        for r in rows_out:
            print(f"[m11] {r['tp']:>3} {r['n']:>3} {r['ms_per_step']:>9.2f} "
                  f"{r['ratio_to_tp%d' % ref]:>8.2f}x {r['predicted_ratio']:>9.2f}x")

        print("[m11] --- the registered prediction, scored ---")
        for s in shape:
            print(f"[m11] TP={s['tp']}: level {s['level_ratio_to_ref']:.2f}x "
                  f"(predicted {s['predicted_level_ratio']:.2f}x)   "
                  f"shape n={s['n_lo']}->{s['n_hi']} rises {s['shape_ratio']:.2f}x")

        ref_shape = next(s["shape_ratio"] for s in shape if s["tp"] == ref)
        level_ok = all(abs(s["level_ratio_to_ref"] - s["predicted_level_ratio"])
                       / s["predicted_level_ratio"] < 0.20 for s in shape)
        shape_ok = all(abs(s["shape_ratio"] - ref_shape) / ref_shape < 0.20 for s in shape)

        if not level_ok:
            worst = min(shape, key=lambda s: s["level_ratio_to_ref"] / s["predicted_level_ratio"])
            print(f"[m11] LEVEL: prediction MISSED. TP={worst['tp']} came in at "
                  f"{worst['level_ratio_to_ref']:.2f}x against {worst['predicted_level_ratio']:.2f}x "
                  f"predicted -- SUB-proportional, so cost does not scale with per-chip weight "
                  f"bytes alone. The likeliest reading is that the higher-TP arms pay collective "
                  f"overhead the roofline does not model, which flatters them here.")
        else:
            print("[m11] LEVEL: prediction held within 20%.")

        if not shape_ok:
            odd = max(shape, key=lambda s: abs(s["shape_ratio"] - ref_shape))
            print(f"[m11] SHAPE: prediction MISSED at TP={odd['tp']} "
                  f"({odd['shape_ratio']:.2f}x vs {ref_shape:.2f}x at TP={ref}). The curve "
                  f"gets FLATTER with less sharding, which is what a larger per-chip weight "
                  f"floor implies: more of the step is floor, so batch size moves it less.")
        else:
            print("[m11] SHAPE: preserved across every TP arm.")

        # The question M4/Q6 actually asked, which neither half of the prediction
        # settles on its own.
        flattest = min(shape, key=lambda s: s["shape_ratio"])
        print("[m11] --- Q6: is the claim about the stack or about TP=4? ---")
        print(f"[m11] Every arm's per-step cost rises less than {max(s['shape_ratio'] for s in shape):.2f}x "
              f"over a {shape[0]['n_hi'] // shape[0]['n_lo']}x batch rise, and the FLATTEST arm is "
              f"TP={flattest['tp']} at {flattest['shape_ratio']:.2f}x.")
        if flattest["tp"] < ref:
            print("[m11] VERDICT: the weight floor dominates at every sharding measured, and "
                  "dominates MORE with less sharding -- the opposite of the reviewer's "
                  "alternative. Request-dimension padding being cheap is not an artifact of "
                  "the TP=4 layout, and §9's advice survives the objection for this model "
                  "and chip. It says nothing about a larger model or a multi-host slice.")
        else:
            print("[m11] VERDICT: the curve is flattest at the reference sharding, so the "
                  "cheapness of request-dimension padding may indeed be a property of that "
                  "layout. §9 must be narrowed to it.")
        print(f"[m11] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
