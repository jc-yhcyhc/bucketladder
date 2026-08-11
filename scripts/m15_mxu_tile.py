#!/usr/bin/env python3
"""
M15 — is the n=4 convergence an MXU tiling effect? The last live hypothesis.

Review Q1, and the first concrete mechanism anyone has proposed for the one
result this project has never explained:

  "Have you tested the tiling hypothesis directly — that at small batch the
   matmul's M dimension is padded to the MXU tile, so additional sequences are
   free until M reaches the tile boundary, after which the curve changes
   character? An isolated matmul microbenchmark at M = 1, 2, 4, 8, 16, 32 with
   the model's actual weight shapes is an off-server experiment; does the
   resulting M-padding curve match your paid-share curve?"

This is a better hypothesis than anything we generated, and it survives what
killed the others. The operator profile (M12) showed no category of device time
moving discontinuously at n=4 — but a tile-padding effect would NOT show up as a
category shift. The same matmul kernel runs either way; it just stops being
padded. A breakdown by operator class is blind to it by construction.

THE PREDICTION, REGISTERED HERE BEFORE THE MEASUREMENT.

  If tiling explains it: time per matmul is FLAT from M=1 up to the tile
  boundary, then rises roughly linearly. The per-sequence cost therefore falls
  as 1/M inside the tile and flattens outside it, and the knee sits at the tile
  width. For the paid-share curve to be explained by this, the knee must land
  at M=4.

  If tiling does not explain it: either the curve is smooth with no knee, or the
  knee sits somewhere else entirely (8, 16, 128 are the plausible TPU tile
  widths, and 128 is the most likely for a v5e MXU). A knee at 128 would say
  tiling is real but irrelevant to n=4.

WHY THIS IS CHEAP AND CLEAN. No server, no vLLM, no scheduler. Just the model's
actual weight shapes multiplied by an M-row activation, timed on the same chip.
Every confound this project has fought — chunked prefill, request padding,
arrival races, split dispatches — is absent by construction.

Usage:
  python scripts/m15_mxu_tile.py --config configs/m15_mxu_tile.json
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    run = start_run("m15_mxu_tile", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        a = cfg["arch"]
        tp = cfg["controlled"]["tensor_parallel_size"]
        H = a["hidden_size"]
        # The two shapes that dominate a decode step, sharded as the model is.
        shapes = {
            "qkv_proj": (H, (a["num_attention_heads"] + 2 * a["num_key_value_heads"])
                         * a["head_dim"] // tp),
            "mlp_up": (H, a["intermediate_size"] // tp),
            "mlp_down": (a["intermediate_size"] // tp, H),
        }
        reps, warmup = cfg["repeats"], cfg.get("warmup", 5)
        rows: list[dict[str, Any]] = []

        for name, (K, N) in shapes.items():
            key = jax.random.PRNGKey(0)
            w = jax.random.normal(key, (K, N), dtype=jnp.bfloat16)
            fn = jax.jit(lambda x, w=w: x @ w)
            for M in cfg["m_values"]:
                x = jax.random.normal(jax.random.PRNGKey(M), (M, K), dtype=jnp.bfloat16)
                for _ in range(warmup):
                    fn(x).block_until_ready()
                ts = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    fn(x).block_until_ready()
                    ts.append((time.perf_counter() - t0) * 1e6)
                med = statistics.median(ts)
                rows.append({"matmul": name, "M": M, "K": K, "N": N,
                             "us_median": med, "us_min": min(ts),
                             "us_per_row": med / M,
                             "flops": 2 * M * K * N})
                print(f"[m15] {name:<10} M={M:<4} {med:8.1f} us  {med / M:8.2f} us/row")
        save_table(run, "matmuls", rows)

        # --- where is the knee? ----------------------------------------------
        print("[m15] --- where does per-row cost stop falling? ---")
        knees: list[dict[str, Any]] = []
        for name in shapes:
            g = sorted([r for r in rows if r["matmul"] == name], key=lambda r: r["M"])
            # Inside a tile, total time is flat, so per-row cost halves per doubling.
            # Outside it, total time grows and per-row cost flattens. The knee is the
            # first M where total time rises materially over the previous point.
            knee = None
            for i in range(1, len(g)):
                if g[i]["us_median"] > g[i - 1]["us_median"] * cfg["rise_threshold"]:
                    knee = g[i]["M"]
                    break
            flat_ratio = g[-1]["us_median"] / g[0]["us_median"]
            knees.append({"matmul": name, "knee_M": knee if knee else -1,
                          "total_time_ratio_over_range": flat_ratio,
                          "M_lo": g[0]["M"], "M_hi": g[-1]["M"]})
            print(f"[m15]   {name:<10} knee at M={knee if knee else '>%d' % g[-1]['M']}"
                  f"   total time M={g[0]['M']}->{g[-1]['M']} rises {flat_ratio:.2f}x")
        save_table(run, "knees", knees)

        found = [k["knee_M"] for k in knees if k["knee_M"] > 0]
        if found and all(k == 4 for k in found):
            print("[m15] VERDICT: the knee is at M=4 in every matmul. Tile padding is a "
                  "live explanation for the n=4 convergence, and the paper gains the "
                  "mechanism it has been missing.")
        elif found:
            print(f"[m15] VERDICT: knees at M={sorted(set(found))}, not 4. Tile padding is "
                  f"real but does not land where the convergence does, so it does not "
                  f"explain it. The hypothesis is tested and rejected rather than left open.")
        else:
            print(f"[m15] VERDICT: no knee within M<={max(r['M'] for r in rows)} — total "
                  f"matmul time never rises by the threshold, so these shapes are "
                  f"latency-bound across the whole range and tile padding cannot explain "
                  f"a change at n=4.")
        print(f"[m15] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
