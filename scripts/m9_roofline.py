#!/usr/bin/env python3
"""
M9 — the mechanism the paper measured around: arithmetic intensity vs batch size.

Review finding M1, the highest-value one: "Every result here is wall-time deltas
from /metrics. The central finding is that padded slots are free at n=4-8 and
expensive at n=1-2, which is almost certainly an arithmetic-intensity statement...
The paper never reports achieved HBM bandwidth, MXU utilisation, or an operator
breakdown, so it cannot distinguish this from competing explanations."

Correct. What the review assumed, though, is that answering it needs a profiler.
Most of it does not. Achieved HBM bandwidth and MFU are

    bytes moved per step / measured step time        and
    FLOPs per step      / measured step time / peak

and both numerators are arithmetic on published model dimensions, not
measurements. The step times are already captured. So the roofline is computable
offline, today, for zero dollars -- and it is the part of M1 that carries the
explanation. Only the operator breakdown (which KERNEL changes at n=4) needs
xprof, and that is scoped separately in `m10_profile.py`.

WHAT THIS DECIDES. If decode at n=1-2 sits at the memory-bandwidth roof, the
per-step cost is a fixed weight-load floor that batch size cannot amortise away,
and padding a request dimension into that floor is free -- which is exactly what
§4.1 measures and never explains. If the measured time instead sits far from
both roofs, the bandwidth story is wrong and the paper should not tell it.

    measured ~= bytes/peak_bw at low n, pulling away as n grows
        -> memory-bound floor confirmed; padding hides under it until compute
           catches up, and the paper gains its mechanism
    measured far above both roofs everywhere
        -> neither roof binds; the cost is overhead (dispatch, collectives) and
           the arithmetic-intensity explanation must be dropped

HONESTY ABOUT WHAT THIS IS NOT. A roofline built from published dimensions is a
model, not a measurement. It cannot see kernel choice, tiling, ICI collective
time, or a fused-vs-unfused matmul. It is reported as an upper bound on
efficiency and a lower bound on time, and where the measurement beats the model
the model is wrong, not the hardware.

Usage:
  python scripts/m9_roofline.py --config configs/m9_roofline.json
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

from _common import finish_run, load_config, save_table, start_run  # noqa: E402

ROOT = HERE.parent


def decode_times(glb: str, output_len: int) -> tuple[dict[int, float], str, int]:
    """Median decode ms/step by batch size, from a captured regime-map run."""
    for d in sorted(glob.glob(str(ROOT / "captured" / glb))):
        p, m = pathlib.Path(d) / "server_timing.parquet", pathlib.Path(d) / "meta.json"
        if not (p.exists() and m.exists()):
            continue
        conf = json.loads(m.read_text())["config"]
        if conf.get("output_len") != output_len:
            continue
        rows = pq.read_table(p).to_pylist()
        ns = sorted({r["concurrency"] for r in rows})
        return ({n: statistics.median([r["decode_ms"] for r in rows if r["concurrency"] == n])
                 / output_len for n in ns}, pathlib.Path(d).name, conf["prompt_len"])
    return {}, "", 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    a, hw = cfg["arch"], cfg["hardware"]
    tp = cfg["controlled"]["tensor_parallel_size"]
    times, src, plen = decode_times(cfg["source_glob"], cfg["source_output_len"])
    if not times:
        print(f"[m9] no captured decode times at {cfg['source_glob']}", file=sys.stderr)
        return 1
    cfg = dict(cfg); cfg["source_run"] = src; cfg["source_prompt_len"] = plen

    run = start_run("m9_roofline", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        L, H = a["num_hidden_layers"], a["hidden_size"]
        hd, nkv, nh = a["head_dim"], a["num_key_value_heads"], a["num_attention_heads"]
        inter, vocab = a["intermediate_size"], a["vocab_size"]
        dt = a["dtype_bytes"]

        # Parameters, counted rather than quoted, so the number is auditable.
        q = H * nh * hd; kv = 2 * H * nkv * hd; o = nh * hd * H
        mlp = 3 * H * inter                      # gate, up, down (SwiGLU)
        per_layer = q + kv + o + mlp
        params = L * per_layer + vocab * H       # tied embedding counted once
        params_chip = params / tp
        peak_flops = hw["peak_bf16_flops_per_chip"]
        peak_bw = hw["hbm_bw_bytes_per_s_per_chip"]

        rows: list[dict[str, Any]] = []
        for n, ms in sorted(times.items()):
            ctx = plen + cfg["source_output_len"] // 2      # mean context during decode
            # Bytes read per chip per decode step: all weights, plus this batch's KV.
            w_bytes = params_chip * dt
            kv_bytes = 2 * n * ctx * (nkv / tp) * hd * L * a["kv_dtype_bytes"]
            byt = w_bytes + kv_bytes
            # FLOPs per chip: 2*params per token for the matmuls, plus attention.
            fl = 2 * params_chip * n + 2 * 2 * n * ctx * (nh / tp) * hd * L
            t = ms / 1000.0
            t_bw, t_fl = byt / peak_bw, fl / peak_flops
            rows.append({
                "n": n, "ms_per_step": ms, "context_tokens": ctx,
                "bytes_per_chip": byt, "weight_bytes_frac": w_bytes / byt,
                "flops_per_chip": fl, "arithmetic_intensity": fl / byt,
                "achieved_bw_GBs": byt / t / 1e9,
                "bw_utilisation_pct": (byt / t) / peak_bw * 100,
                "achieved_TFLOPs": fl / t / 1e12, "mfu_pct": (fl / t) / peak_flops * 100,
                "roof_bw_ms": t_bw * 1000, "roof_flops_ms": t_fl * 1000,
                "roof_ms": max(t_bw, t_fl) * 1000,
                "measured_over_roof": ms / (max(t_bw, t_fl) * 1000),
                "bound_by": "memory" if t_bw > t_fl else "compute",
            })
        save_table(run, "roofline", rows)

        print(f"[m9] {params / 1e9:.2f}B params, TP={tp} -> {params_chip / 1e9:.2f}B/chip, "
              f"{params_chip * dt / 1e9:.2f} GB of weights read every decode step")
        print(f"[m9] peak {peak_flops / 1e12:.0f} TFLOP/s, {peak_bw / 1e9:.0f} GB/s per chip")
        print(f"[m9] source {src}, prompt_len={plen}")
        print(f"[m9] {'n':>3} {'ms/step':>8} {'AI':>7} {'BW GB/s':>9} {'BW%':>6} "
              f"{'TFLOP/s':>8} {'MFU%':>6} {'roof ms':>8} {'x roof':>7} {'bound':>7}")
        for r in rows:
            print(f"[m9] {r['n']:>3} {r['ms_per_step']:>8.2f} {r['arithmetic_intensity']:>7.1f} "
                  f"{r['achieved_bw_GBs']:>9.1f} {r['bw_utilisation_pct']:>5.1f}% "
                  f"{r['achieved_TFLOPs']:>8.2f} {r['mfu_pct']:>5.2f}% {r['roof_ms']:>8.2f} "
                  f"{r['measured_over_roof']:>7.2f} {r['bound_by']:>7}")

        lo = [r for r in rows if r["n"] <= 2]
        hi = [r for r in rows if r["n"] >= 16]
        if lo and hi:
            print(f"[m9] weights are {statistics.fmean([r['weight_bytes_frac'] for r in lo]) * 100:.0f}% "
                  f"of bytes moved at n<=2 and "
                  f"{statistics.fmean([r['weight_bytes_frac'] for r in hi]) * 100:.0f}% at n>=16.")
            if all(r["bound_by"] == "memory" for r in rows):
                print("[m9] Every batch size is memory-bound by this model: the step reads the "
                      "whole weight set regardless of n, so cost is dominated by a floor that "
                      "batch size does not move. That is the mechanism behind §4.5's 2.4x "
                      "step-cost rise over a 32x batch rise, and behind padding on the request "
                      "dimension being free -- there is nothing to pay until the floor is left.")
            r1 = next((r for r in rows if r["n"] == 1), None)
            if r1 and r1["measured_over_roof"] > 2:
                print(f"[m9] CAVEAT: at n=1 the measurement is {r1['measured_over_roof']:.1f}x the "
                      f"roof, so most of that step is NOT bandwidth -- it is fixed overhead the "
                      f"roofline cannot see. The bandwidth story explains the SHAPE of the curve, "
                      f"not the level, and the paper must say so.")
        print(f"[m9] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
