#!/usr/bin/env python3
"""
M7 — confidence intervals on every load-bearing number, and Q1's disagreement.

Review finding M7: "Statistical reporting is thin for a paper whose entire
contribution is measurement. Exactly one p-value appears. No confidence intervals
accompany 85%, 24%, 16%, 35.9%, 5.23%, 2.4x, 13x, or the boundary-wise paid
shares. SS4.6 establishes a decode run-to-run spread of up to 4% -- which is
larger than several differences the paper treats as signal."

Correct, and acting on it is not cosmetic: it changes what SS4.1 is entitled to
say. Bootstrapping the medians it rests on shows the n=8 decode cell is far
noisier than the others, and the interval on the quantity the argument turns on
does not exclude what the argument rejects.

Also answers Q1, which asked why SS4.1 and SS4.5 disagree by 31% on the same
n=8 -> n=16 decode ratio. They do not disagree; they are different workloads, and
the paper failed to say so adjacent to either table. Worse, SS4.1 itself pooled
two runs that disagree about n=9 by 20% -- found while answering the review, and
the same error class as the other six.

The reported quantity is the POSITION of n=9 between n=8 and n=16:

    0%   -> n=9 costs what n=8 costs      (attention not padded to 16)
    100% -> n=9 costs what n=16 costs     (attention padded to 16, the premise)

That is the right statistic because it is what the claim is about, and because
it is scale-free -- it does not care that the two workloads have different
absolute costs, which is precisely the confound Q1 identified.

Usage:
  python scripts/m7_intervals.py
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import random
import statistics
import sys
from typing import Any, Sequence

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pyarrow.parquet as pq  # noqa: E402

from _common import finish_run, save_table, start_run  # noqa: E402

B = 10000
SEED = 20260811


def boot_median(v: Sequence[float], rng: random.Random) -> tuple[float, float, float]:
    if len(v) < 2:
        return (statistics.median(v) if v else float("nan"), float("nan"), float("nan"))
    d = sorted(statistics.median(rng.choices(list(v), k=len(v))) for _ in range(B))
    return statistics.median(v), d[int(0.025 * B)], d[int(0.975 * B)]


def load(glb: str, table: str = "server_timing"):
    for d in sorted(glob.glob(str(pathlib.Path(__file__).resolve().parent.parent / "captured" / glb))):
        p, m = pathlib.Path(d) / f"{table}.parquet", pathlib.Path(d) / "meta.json"
        if p.exists() and m.exists():
            yield (pathlib.Path(d).name, json.loads(m.read_text())["config"],
                   pq.read_table(p).to_pylist())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    rng = random.Random(SEED)
    # Inherit the controlled block from the run whose cells this re-analyses, then
    # state the one variable that DOES vary across the arms. The contract refuses
    # a bare note, which is the correct behaviour: an analysis spanning arms that
    # differ must name the difference rather than wave at it.
    base = next((c for _r, c, _rows in load("session4-qwen3/results/e02_stock_baseline/*")), None)
    if base is None:
        print("[m7] no source run to inherit controlled vars from", file=sys.stderr)
        return 1
    controlled = dict(base.get("controlled", {}))
    controlled.setdefault("ATTN_BUCKETIZED_NUM_REQS", False)
    cfg = {"experiment": "m7_intervals", "mode": "offline", "bootstrap_resamples": B,
           "seed": SEED, "controlled": controlled, "model": base.get("model"),
           "note_varies": ("ATTN_BUCKETIZED_NUM_REQS is False in three of the four arms and "
                           "True in the fourth BY DESIGN -- comparing them is the point. Every "
                           "other controlled variable is identical across the arms; prompt_len "
                           "and output_len are workload, not controlled vars, and differ between "
                           "the SS4.1 and SS4.5 runs, which is what Q1 asked about."),
           "note_scope": ("Bootstrap over run-to-run repeats within a single captured run. "
                          "It quantifies measurement noise, NOT between-run drift, which "
                          "the arms comparison below reports separately.")}
    run = start_run("m7_intervals", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        # --- the three runs SS4.1's argument has been drawn from ---------------
        arms = [
            ("session4-dense-r3", "session4-qwen3/results/e02_stock_baseline/*", 3, None),
            ("session4-dense-r21", "session4-qwen3/results/e02_stock_baseline/*", 21, None),
            ("session7-m2-flagoff", "session7-m1m2/results/e02_stock_baseline/*", 21, False),
            ("session7-m2-flagon", "session7-m1m2/results/e02_stock_baseline/*", 21, True),
        ]
        cells: list[dict[str, Any]] = []
        positions: list[dict[str, Any]] = []
        for tag, glb, reps, flag in arms:
            for rid, conf, rows in load(glb):
                if conf.get("repeats") != reps:
                    continue
                if flag is not None and conf["controlled"].get("ATTN_BUCKETIZED_NUM_REQS") is not flag:
                    continue
                g = {n: [r["decode_ms"] for r in rows if r["concurrency"] == n] for n in (8, 9, 16)}
                if not all(g.values()):
                    continue
                for n in (8, 9, 16):
                    med, lo, hi = boot_median(g[n], rng)
                    cells.append({"arm": tag, "run_id": rid, "n": n, "n_obs": len(g[n]),
                                  "prompt_len": conf["prompt_len"], "output_len": conf["output_len"],
                                  "median_ms": med, "ci_lo_ms": lo, "ci_hi_ms": hi,
                                  "ci_width_pct": (hi - lo) / med * 100 if med else float("nan")})
                pos = []
                for _ in range(B):
                    m8 = statistics.median(rng.choices(g[8], k=len(g[8])))
                    m9 = statistics.median(rng.choices(g[9], k=len(g[9])))
                    m16 = statistics.median(rng.choices(g[16], k=len(g[16])))
                    if m16 != m8:
                        pos.append((m9 - m8) / (m16 - m8) * 100)
                pos.sort()
                positions.append({"arm": tag, "run_id": rid, "n_obs": len(g[9]),
                                  "prompt_len": conf["prompt_len"], "output_len": conf["output_len"],
                                  "position_pct": pos[len(pos) // 2],
                                  "ci_lo_pct": pos[int(0.025 * len(pos))],
                                  "ci_hi_pct": pos[int(0.975 * len(pos))],
                                  "excludes_full_padding": pos[int(0.975 * len(pos))] < 100.0,
                                  "excludes_zero_padding": pos[int(0.025 * len(pos))] > 0.0})
                break
        save_table(run, "decode_cells", cells)
        save_table(run, "n9_position", positions)

        print("[m7] --- SS4.1: decode cells with bootstrap 95% CIs ---")
        print(f"[m7] {'arm':<22} {'n':>3} {'obs':>4} {'median':>9} {'95% CI':>20} {'width':>7}")
        for c in cells:
            print(f"[m7] {c['arm']:<22} {c['n']:>3} {c['n_obs']:>4} {c['median_ms']:>8.2f}ms "
                  f"[{c['ci_lo_ms']:>7.2f},{c['ci_hi_ms']:>7.2f}] {c['ci_width_pct']:>6.1f}%")

        wide = [c for c in cells if c["ci_width_pct"] > 15 and c["n_obs"] >= 21]
        if wide:
            ns = sorted({c["n"] for c in wide})
            print(f"[m7] NOTE: at 21 repeats the wide cells are n={ns}. SS4.6 reported "
                  f"decode run-to-run spread as 1.00-1.04x; these cells are far wider "
                  f"than that, so that figure does not describe every cell.")

        print("[m7] --- where n=9 sits between n=8 and n=16 (0%=unpadded, 100%=padded to 16) ---")
        for p in positions:
            print(f"[m7] {p['arm']:<22} plen={p['prompt_len']:<5} olen={p['output_len']:<3} "
                  f"{p['position_pct']:>6.0f}%  95% CI [{p['ci_lo_pct']:>6.0f}%,{p['ci_hi_pct']:>6.0f}%]")
        good = [p for p in positions if p["n_obs"] >= 21]
        if good:
            if all(p["excludes_full_padding"] for p in good):
                print("[m7] Every 21-repeat arm EXCLUDES full padding to the next bucket, "
                      "which is what the compiled-shape premise predicts. That much holds.")
            if not any(p["excludes_zero_padding"] for p in good):
                print("[m7] No arm excludes zero padding either -- consistent with the claim, "
                      "but the interval is wide enough that this comparison ALONE cannot "
                      "carry SS4.1. The load-bearing evidence is the paired flag on/off "
                      "experiment and the source reading; this is corroboration, and the "
                      "paper must present it as such.")
            spread = max(p["position_pct"] for p in good) - min(p["position_pct"] for p in good)
            print(f"[m7] between-arm spread of the point estimate: {spread:.0f} pp across "
                  f"{len(good)} runs measuring the same quantity.")

        # --- Q1: the two tables are different workloads ----------------------
        q1: list[dict[str, Any]] = []
        for tag, glb, reps in (("SS4.1", "session7-m1m2/results/e02_stock_baseline/*", 21),
                               ("SS4.5", "session12-regime/results/e02_stock_baseline/*", 0)):
            for rid, conf, rows in load(glb):
                if conf.get("repeats", 0) < reps:
                    continue
                if tag == "SS4.5" and conf.get("output_len") != 64:
                    continue
                if tag == "SS4.1" and conf["controlled"].get("ATTN_BUCKETIZED_NUM_REQS"):
                    continue
                g = {n: [r["decode_ms"] for r in rows if r["concurrency"] == n] for n in (8, 16)}
                if not all(g.values()):
                    continue
                ol = conf["output_len"]
                m8, m16 = statistics.median(g[8]), statistics.median(g[16])
                q1.append({"section": tag, "run_id": rid, "prompt_len": conf["prompt_len"],
                           "output_len": ol, "ms_per_step_n8": m8 / ol,
                           "ms_per_step_n16": m16 / ol, "ratio_n8_to_n16": m16 / m8})
                break
        save_table(run, "q1_configs", q1)
        print("[m7] --- Q1: why SS4.1 and SS4.5 report different n=8->n=16 ratios ---")
        for r in q1:
            print(f"[m7] {r['section']:<6} prompt_len={r['prompt_len']:<5} output_len={r['output_len']:<3} "
                  f"n=8 {r['ms_per_step_n8']:5.2f} ms/step  n=16 {r['ms_per_step_n16']:5.2f}  "
                  f"ratio {r['ratio_n8_to_n16']:.2f}x")
        if len(q1) == 2:
            a, b_ = q1
            print(f"[m7] Not a contradiction: the workloads differ. prompt_len "
                  f"{a['prompt_len']} vs {b_['prompt_len']} means the KV read per decode step "
                  f"differs by {a['prompt_len'] / b_['prompt_len']:.1f}x, and decode attention "
                  f"scales with it. The defect is that neither table stated its configuration.")
        print(f"[m7] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
