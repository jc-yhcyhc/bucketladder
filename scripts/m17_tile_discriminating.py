#!/usr/bin/env python3
"""
M17 — a matmul microbenchmark that can actually fail. M15 could not.

M15 timed one jitted matmul per call and reported 142.9 us at M=1, flat to
143.6 us at M=256. Two things were wrong with it, and the second is worse.

WRONG THING ONE: it measured dispatch, not hardware. The qkv projection holds
7.86 MB of weights per chip. At v5e's 819 GB/s that is 9.6 us, so 142.9 us sits
15x above the bandwidth floor at an implied **55 GB/s -- 7% of peak**, and an
order of magnitude below what the full server achieves while doing strictly more
work. A single jitted op with a host round-trip and `block_until_ready` lands in
exactly that range. The "per row" column was `constant / M`, which is division.

WRONG THING TWO: it could not have discriminated. For that shape the memory floor
(9.6 us) and compute at M=256 (2 * 256 * 2560 * 1536 = 2.0 GFLOP, 10.2 us at
peak) cross near M ~= 240. So **bandwidth alone predicts a flat curve across the
entire sweep**, and tile padding predicts flat only to M ~= 8. Tiling's
prediction is a subset of bandwidth's over M in [1, 256]; no outcome in that
range separates them. A test whose two hypotheses make the same prediction
everywhere it looks is not a test.

That is a class-two error -- instrument definition -- caught this time before it
reached the paper, but only because a reviewer checked an absolute number against
the hardware. Hence the rule this script enforces: **every cell reports its
implied GB/s and FLOP/s, and a cell below a utilisation floor is refused rather
than interpreted.**

THE DESIGN. Two arms, because one cannot do it.

  ARM A -- real shapes, amortised. K matmuls against the same resident weights
  inside one `jit`, so dispatch is divided by K and per-iteration time reflects
  the hardware. This measures the bandwidth-vs-compute crossover and is a direct
  test of m13's frontier: the curve should be flat to M ~= 240 and rise after.

  ARM B -- small resident weights, amortised. Shrink the weight matrix until the
  bandwidth floor is far below the compute time, so bandwidth is NOT the binding
  constraint and MXU tile structure is the only thing left. Sweep M through the
  compute-bound region. Here, and only here, a tile knee is identifiable.

  Arm A flat to ~240 -> bandwidth story confirmed, frontier corroborated
  Arm B knee at M=k   -> tile width is k; if k=4 it is a live explanation for
                         the n=4 convergence, and otherwise it is not

Usage:
  python scripts/m17_tile_discriminating.py --config configs/m17_tile.json
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import finish_run, load_config, save_table, start_run  # noqa: E402


def timed(fn, x, reps: int) -> float:
    for _ in range(3):
        fn(x).block_until_ready()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(x).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    run = start_run("m17_tile_discriminating", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        hw = cfg["hardware"]
        peak, bw = hw["peak_bf16_flops_per_chip"], hw["hbm_bw_bytes_per_s_per_chip"]
        floor = cfg["utilisation_floor_pct"]
        rows: list[dict[str, Any]] = []

        for arm in cfg["arms"]:
            K_, N_ = arm["K"], arm["N"]
            iters = arm["iters"]
            w = jax.random.normal(jax.random.PRNGKey(0), (K_, N_), dtype=jnp.bfloat16)
            wbytes = K_ * N_ * 2

            def build(M):
                z = jnp.zeros((M, N_), dtype=jnp.float32)

                def f(x, w=w, z=z, iters=iters):
                    def body(_, acc):
                        return acc + (x @ w).astype(jnp.float32)
                    return jax.lax.fori_loop(0, iters, body, z)
                return jax.jit(f)

            print(f"[m17] === arm {arm['name']}: {K_}x{N_} ({wbytes / 1e6:.2f} MB), "
                  f"{iters} iters/call ===")
            for M in arm["m_values"]:
                x = jax.random.normal(jax.random.PRNGKey(M), (M, K_), dtype=jnp.bfloat16)
                total = timed(build(M), x, cfg["repeats"])
                per_iter = total / iters
                fl = 2 * M * K_ * N_
                gbs = wbytes / per_iter / 1e9
                tfs = fl / per_iter / 1e12
                util = max(100 * gbs * 1e9 / bw, 100 * tfs * 1e12 / peak)
                rows.append({"arm": arm["name"], "M": M, "K": K_, "N": N_,
                             "iters": iters, "us_per_iter": per_iter * 1e6,
                             "implied_GBs": gbs, "implied_TFLOPs": tfs,
                             "utilisation_pct": util,
                             "bw_floor_us": wbytes / bw * 1e6,
                             "compute_us": fl / peak * 1e6,
                             "usable": util >= floor})
                r = rows[-1]
                print(f"[m17]   M={M:<4} {r['us_per_iter']:8.2f} us/iter  "
                      f"{gbs:7.1f} GB/s  {tfs:6.2f} TFLOP/s  util {util:5.1f}%"
                      f"{'' if r['usable'] else '   <- BELOW FLOOR, not interpreted'}")
        save_table(run, "cells", rows)

        print("[m17] --- verdicts ---")
        verdicts: list[dict[str, Any]] = []
        for arm in cfg["arms"]:
            g = sorted([r for r in rows if r["arm"] == arm["name"]], key=lambda r: r["M"])
            usable = [r for r in g if r["usable"]]
            if len(usable) < 3:
                print(f"[m17] {arm['name']}: only {len(usable)} cells above the {floor}% "
                      f"utilisation floor — REFUSING to interpret. This is the check M15 "
                      f"lacked; without it a dispatch-bound curve reads as a hardware result.")
                verdicts.append({"arm": arm["name"], "knee_M": -1, "usable_cells": len(usable),
                                 "interpretable": False})
                continue
            knee = None
            for i in range(1, len(usable)):
                if usable[i]["us_per_iter"] > usable[i - 1]["us_per_iter"] * cfg["rise_threshold"]:
                    knee = usable[i]["M"]
                    break
            verdicts.append({"arm": arm["name"], "knee_M": knee if knee else -1,
                             "usable_cells": len(usable), "interpretable": True,
                             "peak_util_pct": max(r["utilisation_pct"] for r in usable)})
            print(f"[m17] {arm['name']}: knee at M="
                  f"{knee if knee else '>' + str(usable[-1]['M'])}, "
                  f"peak utilisation {max(r['utilisation_pct'] for r in usable):.0f}%")
        save_table(run, "verdicts", verdicts)

        b = next((v for v in verdicts if v["arm"] == "tile_probe"), None)
        if b and b["interpretable"]:
            k = b["knee_M"]
            if k == 4:
                print("[m17] Arm B knees at M=4. Tile padding IS a live explanation for the "
                      "n=4 convergence.")
            elif k > 0:
                print(f"[m17] Arm B knees at M={k}, not 4. Tile structure is real and lands "
                      f"elsewhere, so it does not explain the n=4 convergence — and this time "
                      f"the sweep was in a regime where a knee at 4 WOULD have been visible.")
            else:
                print("[m17] Arm B shows no knee in the compute-bound region, so there is no "
                      "tile-width effect to explain n=4 at any M measured.")
        print(f"[m17] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
