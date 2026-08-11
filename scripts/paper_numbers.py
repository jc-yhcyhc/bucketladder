#!/usr/bin/env python3
"""
Every number in the paper, recomputed from captured runs and diffed against the text.

WHY. `notes/paper_draft.md` carries roughly forty specific figures, and most were
transcribed by hand out of terminal output across eight hardware sessions.
Nothing guaranteed they matched the parquet files they came from.

That is worth checking here more than it usually would be, because this
project's error record is specifically about inference from numbers: a cost
model built on a three-repeat median of a bimodal cell, a "provable bound" that
a policy beat, `Oracle` documented as an upper bound it was not, and a headline
of "padding is free" written three days before measuring that 6-10% of it is
paid. Every one was caught, and every one was a number reasoned about
incorrectly rather than a number typed incorrectly — which is exactly what a
transcription pass would surface if it had happened too.

WHAT THIS IS. The `claim_id` indirection plan_v3 called for and the repo never
built. Each claim names the run it comes from and how it is derived, so a number
in the paper can be traced to a `run_id` without human memory in the loop. It
emits `paper_numbers.parquet` and fails loudly on any mismatch.

A claim that cannot be recomputed is reported as UNVERIFIED rather than passed —
silence is not agreement.

Usage:
  python scripts/paper_numbers.py
  python scripts/paper_numbers.py --write        # also emit the parquet
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import pathlib
import statistics
import sys
from typing import Any, Callable

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))

import pyarrow.parquet as pq  # noqa: E402

ROOT = HERE.parent


# --- loaders ---------------------------------------------------------------

def runs(pattern: str, table: str) -> list[tuple[str, dict, list[dict]]]:
    """(run_id, config, rows) for every captured run matching `pattern`."""
    out = []
    for d in sorted(glob.glob(str(ROOT / "captured" / pattern))):
        p, m = pathlib.Path(d) / f"{table}.parquet", pathlib.Path(d) / "meta.json"
        if not (p.exists() and m.exists()):
            continue
        out.append((pathlib.Path(d).name, json.loads(m.read_text())["config"],
                    pq.read_table(p).to_pylist()))
    return out


def e01_flatness(model: str, bucket: int) -> tuple[float, str]:
    """Median flatness at `bucket`, over the GOOD e01 runs for `model`.

    Session 3's first Qwen3 run opened its metrics window before the warmup
    requests, putting a 7.5x first-request outlier inside the delta and yielding
    flatness 2.79 at bucket 256. It is excluded by the same rule that caught it:
    flatness above 1.1 is not physically meaningful — cost cannot fall faster
    than proportionally with length — so a run exhibiting it is broken.
    """
    vals, ids = [], []
    for rid, cfg, rows in runs("*/results/e01_oracle_gap/*", "flatness"):
        if cfg["model"] != model:
            continue
        by = {r["bucket"]: r["flatness"] for r in rows}
        if any(v > 1.1 for v in by.values() if v == v):
            continue
        if bucket in by and by[bucket] == by[bucket]:
            vals.append(by[bucket]); ids.append(rid)
    if not vals:
        return float("nan"), ""
    return statistics.median(vals), ";".join(ids)


def e07_spread_pooled(n: int, max_len: int) -> tuple[float, str]:
    """Median measured cost across the two SPREAD runs, which are replicates.

    Pinning the run matters. Session 6 ran e07 twice on the same server: a 5-cell
    gate and a 12-cell spread sweep, and the same cell differs between them by up
    to 7%. The paper's table is the spread sweep, pooled with its
    chunked-prefill-off twin (§4.1 showed the setting is irrelevant, making them
    replicates of one quantity). An earlier version of this audit silently picked
    whichever run sorted first and reported three spurious mismatches.
    """
    vals, ids = [], []
    for rid, cfg, rows in runs("session6-gate/results/e07_ragged_batch/*", "cells"):
        if len(rows) <= 5:          # the gate run, not the sweep
            continue
        for r in rows:
            if r["n"] == n and r["max_len"] == max_len:
                vals.append(r["measured_ms"]); ids.append(rid)
    if not vals:
        return float("nan"), ""
    return statistics.median(vals), ";".join(ids)


def e07_penalty(n: int, max_len: int, packed: float = 69.08) -> tuple[float, str]:
    """Penalty over the packed prediction: (measured / predicted - 1) x 100.

    NOT the absolute percentage error the cells table stores, which divides by
    the measurement instead of the prediction. Same two numbers, different
    denominators, and mixing them produced a 28.0-vs-21.9 "mismatch" that was
    purely a definition clash.
    """
    v, src = e07_spread_pooled(n, max_len)
    return (100 * (v / packed - 1), src) if v == v else (float("nan"), src)


def e02_dense(session: str, n: int, col: str, min_repeats: int = 21) -> tuple[float, str]:
    for rid, cfg, rows in runs(f"{session}/results/e02_stock_baseline/*", "server_timing"):
        if cfg.get("repeats", 0) < min_repeats:
            continue
        sub = [r[col] for r in rows if r["concurrency"] == n]
        if sub:
            return statistics.median(sub), rid
    return float("nan"), ""


def m2_arm(flag: bool, n: int, col: str) -> tuple[float, str]:
    for rid, cfg, rows in runs("session7-m1m2/results/e02_stock_baseline/*", "server_timing"):
        if cfg.get("concurrency") != [8, 9, 16]:
            continue
        if cfg["controlled"].get("ATTN_BUCKETIZED_NUM_REQS") is not flag:
            continue
        sub = [r[col] for r in rows if r["concurrency"] == n]
        if sub:
            return statistics.median(sub), rid
    return float("nan"), ""


def m1_edge(name: str, field: str) -> tuple[float, str]:
    for rid, cfg, rows in runs("session7-m1m2/results/m1_boundary/*", "edges"):
        for r in rows:
            if r["edge"] == name:
                return r[field], rid
    return float("nan"), ""


def s13(exp: str, table: str):
    """Rows from a session-13 experiment, by name."""
    for _rid, _cfg, rows in runs(f"session13-review/results/{exp}/*", table):
        return rows
    return []


def m10_pad(arm: str) -> tuple[float, str]:
    """Padded share of executed tokens for one length-distribution arm."""
    for r in s13("m10_trace_workload", "arms"):
        if r["arm"] == arm:
            return r["padding_pct_of_executed"], "m10"
    return float("nan"), ""


def m8_split(launcher: str, n: int) -> tuple[float, str]:
    """% of dispatches whose PREFILL split, from raw step counts.

    Recomputed from n_steps rather than read from the stored `single_step`
    column, because that column was written by the criterion §6 reports as the
    eighth failure. Reading it back would launder the error into the paper.
    """
    rows = [r for r in s13("m8_split_barrier", "dispatches")
            if r["launcher"] == launcher and r["n"] == n]
    if not rows:
        return float("nan"), ""
    clean = 1 + 1                      # output_len=1: one prefill step + one decode
    return 100 * sum(r["n_steps"] > clean for r in rows) / len(rows), "m8"


def m9_roof(n: int, field: str) -> tuple[float, str]:
    for _rid, _cfg, rows in runs("../results/review/m9_roofline/*", "roofline"):
        for r in rows:
            if r["n"] == n:
                return r[field], "m9"
    return float("nan"), ""


def m6_ab(n: int, field: str) -> tuple[float, str]:
    for _rid, _cfg, rows in runs("../results/review/m6_lens_ablation/*", "summary_by_n"):
        for r in rows:
            if r["n"] == n:
                return r[field], "m6"
    return float("nan"), ""

def m1_share(edge: str) -> float:
    """Paid share of nominal padding at a boundary, as a FRACTION, at n=4.

    Defined once here because the review found it used inline three times and
    defined nowhere: (measured - real) / (padded - real), i.e. where the measured
    cost ratio falls between the ratio real work predicts and the ratio the
    compiled-shape premise predicts. 0 = padding free, 1 = padding fully paid.
    """
    for _rid, _cfg, rows in runs("session8-m3m4/results/m1_boundary/*", "edges"):
        for r in rows:
            if r["edge"] == f"n4:{edge}" and r["splits_below"] == 0 and r["splits_above"] == 0:
                return (r["cost_ratio"] - r["real_ratio"]) / (r["padded_ratio"] - r["real_ratio"])
    return float("nan")


def m14_share(edge: str) -> tuple[float, str]:
    """Paid padding share at n=16, from CLEAN dispatches only.

    Recomputed here from raw per-dispatch rows rather than read from the run's
    `edges` table, because the run that produced that table pooled split
    dispatches into the median -- the ninth failure in §6, and the one the
    paper's own §2 said could not happen. Reading the stored summary back would
    launder it. A split executes more steps and therefore more padding, so
    pooling biases this share upward.
    """
    for rid, cfg, rows in runs("session14-ablation/results/m1_boundary/*", "dispatches"):
        olen = cfg["output_len"]
        arms = {}
        for arm in ("below", "above"):
            g = [r for r in rows if r["edge"] == edge and r["arm"] == arm
                 and r["n_steps"] == r["n_steps"] and r["n_steps"] <= 1 + olen
                 and r["step_latency_ms"] == r["step_latency_ms"]]
            if not g:
                return float("nan"), ""
            arms[arm] = (statistics.median([r["step_latency_ms"] for r in g]),
                         g[0]["tokens_real"], g[0]["tokens_padded"])
        b, a = arms["below"], arms["above"]
        rr, pr = a[1] / b[1], a[2] / b[2]
        return 100 * ((a[0] / b[0]) - rr) / (pr - rr), rid
    return float("nan"), ""

def m12_cat(n: int, cat: str) -> tuple[float, str]:
    """Share of TPU:0 XLA-Ops device time in one operator category."""
    import sys as _s
    _s.path.insert(0, str(HERE))
    from m12_profile import parse_trace, summarise  # noqa: PLC0415
    f = ROOT / "captured" / "session15-profile" / "traces" / "prof" / f"n{n}" \
        / "t1v-n-736b28ee-w-0.trace.json.gz"
    if not f.exists():
        return float("nan"), ""
    cats = summarise(parse_trace(f))
    tot = sum(cats.values())
    return (100 * cats.get(cat, 0.0) / tot if tot else float("nan")), "m12"

def m13_front(ctx: int, field: str) -> tuple[float, str]:
    for _rid, _cfg, rows in runs("../results/review/m13_frontier/*", "frontier"):
        for r in rows:
            if r["context_tokens"] == ctx:
                return r[field], "m13"
    return float("nan"), ""

def m8_clean(edge: str) -> tuple[float, str]:
    """Paid share at n=8 from CLEAN dispatches only (session 8 raw rows)."""
    for rid, cfg, rows in runs("session8-m3m4/results/m1_boundary/*", "dispatches"):
        olen = cfg["output_len"]
        arms = {}
        for arm in ("below", "above"):
            g = [r for r in rows if r["edge"] == edge and r["arm"] == arm
                 and r["n_steps"] == r["n_steps"] and r["n_steps"] <= 1 + olen
                 and r["step_latency_ms"] == r["step_latency_ms"]]
            if not g:
                return float("nan"), ""
            arms[arm] = (statistics.median([r["step_latency_ms"] for r in g]),
                         g[0]["tokens_real"], g[0]["tokens_padded"])
        b, a = arms["below"], arms["above"]
        rr, pr = a[1] / b[1], a[2] / b[2]
        return 100 * ((a[0] / b[0]) - rr) / (pr - rr), rid
    return float("nan"), ""

def m15_flat(matmul: str) -> tuple[float, str]:
    """Total matmul time at M=256 relative to M=1. 1.0 means M is free."""
    for rid, _cfg, rows in runs("session16-mechanism/results/m15_mxu_tile/*", "matmuls"):
        g = {r["M"]: r["us_median"] for r in rows if r["matmul"] == matmul}
        if 1 in g and 256 in g:
            return g[256] / g[1], rid
    return float("nan"), ""


def m16_share(edge: str) -> tuple[float, str]:
    for rid, _cfg, rows in runs("session16-mechanism/results/m1_boundary/*", "edges"):
        for r in rows:
            if r["edge"] == edge:
                return (100 * (r["cost_ratio"] - r["real_ratio"])
                        / (r["padded_ratio"] - r["real_ratio"])), rid
    return float("nan"), ""

def h1() -> tuple[dict, str]:
    """Recompute the headroom directly rather than trusting a stored table."""
    tot_r = tot_p = 0
    ratios: list[float] = []
    rid_used = ""
    for rid, cfg, rows in runs("session5/results/e05_step_shape/*", "dispatches"):
        plen, olen = cfg["prompt_len"], cfg["output_len"]
        rid_used = rid
        for r in rows:
            if r["prefill_ms"] != r["prefill_ms"]:
                continue
            real = r["n"] * (plen + olen)
            pad = sum(float(k) * v for k, v in ast.literal_eval(r["hist"]).items())
            tot_r += real; tot_p += pad
            ratios.append(pad / real - 1.0)
    ratios.sort()
    return ({"ratio": tot_p / tot_r, "pct_padding": 100 * (tot_p - tot_r) / tot_p,
             "mean": 100 * statistics.fmean(ratios),
             "p95": 100 * ratios[int(0.95 * (len(ratios) - 1))]}, rid_used)


def m6_slope(n: int) -> tuple[float, str]:
    """Median slope over CLEAN (single-step) cells at batch size n, session 12."""
    for rid, cfg, rows in runs("session12-regime/results/m5_lens_form/*", "fits"):
        cl = [r["slope_us_per_token"] for r in rows if r["n"] == n and r["splits"] == 0]
        if cl:
            return statistics.median(cl), rid
    return float("nan"), ""


def m6_clean_cells(n: int) -> tuple[float, str]:
    """How many cells at batch size n yielded a single-step dispatch.

    Zero at n>=8 is the session-12 result: prefill step cost cannot be isolated
    at serving batch sizes with a /metrics-delta instrument, because the
    scheduler splits every dispatch.
    """
    for rid, cfg, rows in runs("session12-regime/results/m5_lens_form/*", "fits"):
        at = [r for r in rows if r["n"] == n]
        if at:
            return float(sum(1 for r in at if r["splits"] == 0)), rid
    return float("nan"), ""


def m6_decode(n: int, per_seq: bool = False) -> tuple[float, str]:
    """Per-decode-step cost at batch size n (output_len=64)."""
    for rid, cfg, rows in runs("session12-regime/results/e02_stock_baseline/*", "server_timing"):
        if cfg.get("output_len") != 64:
            continue
        v = [r["decode_ms"] for r in rows if r["concurrency"] == n]
        if v:
            step = statistics.median(v) / cfg["output_len"]
            return (step / n * 1000.0 if per_seq else step), rid
    return float("nan"), ""


def m6_lens_mape() -> tuple[float, str]:
    for rid, cfg, rows in runs("session12-regime/results/m5_lens_form/*", "verdict"):
        if rows:
            return rows[0]["holdout_mape_pct"], rid
    return float("nan"), ""


def curve_cost(tokens: int) -> float:
    d = json.loads((ROOT / "sim" / "measured_cost_curve.json").read_text())
    return dict((int(t), c) for t, c in d["knots_tokens_ms"])[tokens]


# --- the claims ------------------------------------------------------------
# (claim_id, section, what it says, stated value, tolerance, recompute)

Claim = tuple[str, str, str, float, float, Callable[[], tuple[float, str]]]

CLAIMS: list[Claim] = [
    # --- D1, the ragged-batch gate -----------------------------------------
    ("D1.control", "4.1", "uniform controls match all three models to 1.9%", 1.9, 0.3,
     lambda: (max(min(c[f"err_{k}_pct"] for k in ("packed", "per_request_padded", "batch_padded"))
                  for c in runs("session6-gate/results/e07_ragged_batch/*", "cells")[-1][2]
                  if c["is_control"]), "e07")),
    # The four pooled D1 claims combine the chunked-prefill-ON and -OFF spread
    # runs. e08 measured that the setting does not change the result (packed
    # still wins 8/10 ragged cells, batch padding still rejected by 75-579%),
    # which is what makes them replicates rather than different conditions. The
    # guardrail flagged them until that was stated, which is the point of it.
    ("D1.n8.max512", "4.1", "n=8 max=512 pooled median ms", 68.47, 0.3,
     lambda: e07_spread_pooled(8, 512), ("controlled",)),
    ("D1.n8.max1024", "4.1", "n=8 max=1024 pooled median ms", 73.38, 0.3,
     lambda: e07_spread_pooled(8, 1024), ("controlled",)),
    ("D1.n8.max3900", "4.1", "n=8 max=3900 pooled median ms", 88.28, 0.3,
     lambda: e07_spread_pooled(8, 3900), ("controlled",)),
    ("D1.pen.max3072", "4.1", "penalty vs packed at max=3072", 29.4, 0.6,
     lambda: e07_penalty(8, 3072), ("controlled",)),
    ("D1.pen.max3900", "4.1", "penalty vs packed at max=3900", 27.8, 0.6,
     lambda: e07_penalty(8, 3900), ("controlled",)),
    ("D1.reject", "4.1", "batch-padded rejected by 44-618%", 618.0, 5.0,
     lambda: (max(r["err_batch_padded_pct"]
                  for _, _, rows in runs("session6-gate/results/e07_ragged_batch/*", "cells")
                  for r in rows if not r["is_control"]), "e07-all")),

    # --- D2, the flatness staircase ----------------------------------------
    ("D2.qwen.512", "4.2", "Qwen3-4B flatness @512", 1.00, 0.03, lambda: e01_flatness("Qwen/Qwen3-4B", 512)),
    ("D2.qwen.1024", "4.2", "Qwen3-4B flatness @1024", 0.96, 0.03, lambda: e01_flatness("Qwen/Qwen3-4B", 1024)),
    ("D2.qwen.2048", "4.2", "Qwen3-4B flatness @2048", 0.91, 0.03, lambda: e01_flatness("Qwen/Qwen3-4B", 2048)),
    ("D2.qwen.4096", "4.2", "Qwen3-4B flatness @4096", 0.81, 0.03, lambda: e01_flatness("Qwen/Qwen3-4B", 4096)),
    ("D2.tiny.512", "4.2", "TinyLlama flatness @512", 0.90, 0.03,
     lambda: e01_flatness("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 512)),
    ("D2.tiny.2048", "4.2", "TinyLlama flatness @2048", 0.82, 0.03,
     lambda: e01_flatness("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2048)),
    ("D2.smol.2048", "4.2", "SmolLM2 flatness @2048", 0.73, 0.03,
     lambda: e01_flatness("HuggingFaceTB/SmolLM2-1.7B-Instruct", 2048)),
    ("D2.smol.4096", "4.2", "SmolLM2 flatness @4096", 0.54, 0.03,
     lambda: e01_flatness("HuggingFaceTB/SmolLM2-1.7B-Instruct", 4096)),

    # --- D3, decode does not pay the batch ladder --------------------------
    ("D3.decode.n8", "4.3", "decode @n=8 (flag off)", 53.3, 1.5, lambda: e02_dense("session4-qwen3", 8, "decode_ms")),
    ("D3.decode.n9", "4.3", "decode @n=9 (flag off)", 51.4, 1.5, lambda: e02_dense("session4-qwen3", 9, "decode_ms")),
    ("D3.decode.n16", "4.3", "decode @n=16 (flag off)", 91.8, 1.5, lambda: e02_dense("session4-qwen3", 16, "decode_ms")),

    # --- M2, the attention flag --------------------------------------------
    ("M2.off.n9.decode", "4.3", "M2 decode n=9 flag off", 61.8, 1.0, lambda: m2_arm(False, 9, "decode_ms")),
    ("M2.on.n9.decode", "4.3", "M2 decode n=9 flag on", 61.8, 1.0, lambda: m2_arm(True, 9, "decode_ms")),
    ("M2.off.n16.e2e", "4.3", "M2 e2e n=16 flag off", 254.2, 2.0, lambda: m2_arm(False, 16, "e2e_ms")),
    ("M2.on.n16.e2e", "4.3", "M2 e2e n=16 flag on", 254.6, 2.0, lambda: m2_arm(True, 16, "e2e_ms")),

    # --- H1, the padding ceiling -------------------------------------------
    ("H1.ratio", "5", "padded/real over 150 dispatches", 1.56, 0.02, lambda: (h1()[0]["ratio"], h1()[1])),
    ("H1.pct", "5", "% of executed tokens that are padding", 35.9, 0.5, lambda: (h1()[0]["pct_padding"], h1()[1])),
    ("H1.mean", "5", "mean per-dispatch padding ratio %", 56.8, 1.0, lambda: (h1()[0]["mean"], h1()[1])),
    ("H1.p95", "5", "p95 per-dispatch padding ratio %", 99.6, 1.0, lambda: (h1()[0]["p95"], h1()[1])),

    # --- session 13: the review response ------------------------------------
    ("M10.pad.fixed", "4.4", "padded share, fixed-256 workload %", 51.0, 0.5, lambda: m10_pad("fixed")),
    ("M10.pad.uniform", "4.4", "padded share, uniform workload %", 27.3, 0.5, lambda: m10_pad("uniform")),
    ("M10.pad.lognormal", "4.4", "padded share, lognormal workload %", 38.4, 0.5, lambda: m10_pad("lognormal")),
    ("M10.pad.bimodal", "4.4", "padded share, bimodal workload %", 32.7, 0.5, lambda: m10_pad("bimodal")),
    ("M8.tp.n8", "4.3", "prefill split %, old launcher n=8", 20.0, 0.1, lambda: m8_split("threadpool", 8)),
    ("M8.bar.n8", "4.3", "prefill split %, barrier n=8", 0.0, 0.1, lambda: m8_split("barrier", 8)),
    ("M8.tp.n16", "4.3", "prefill split %, old launcher n=16", 100.0, 0.1, lambda: m8_split("threadpool", 16)),
    ("M8.bar.n16", "4.3", "prefill split %, barrier n=16", 60.0, 0.1, lambda: m8_split("barrier", 16)),
    ("M8.bar.n32", "4.3", "prefill split %, barrier n=32", 100.0, 0.1, lambda: m8_split("barrier", 32)),
    ("M9.bw.n1", "4.5", "HBM bandwidth utilisation at n=1 %", 64.9, 0.3, lambda: m9_roof(1, "bw_utilisation_pct")),
    ("M9.bw.n32", "4.5", "HBM bandwidth utilisation at n=32 %", 31.4, 0.3, lambda: m9_roof(32, "bw_utilisation_pct")),
    ("M9.mfu.n32", "4.5", "MFU at n=32 %", 3.65, 0.05, lambda: m9_roof(32, "mfu_pct")),
    ("M9.wfrac.n1", "4.5", "weight share of bytes moved at n=1 %", 99.0, 1.0,
     lambda: (lambda v, s: (v * 100, s))(*m9_roof(1, "weight_bytes_frac"))),
    ("M6.const.n1", "4.2", "constant-only MAPE at n=1 %", 0.96, 0.05, lambda: m6_ab(1, "const_mape_pct")),
    ("M6.const.n4", "4.2", "constant-only MAPE at n=4 %", 14.80, 0.1, lambda: m6_ab(4, "const_mape_pct")),
    ("M6.lens.n4", "4.2", "LENS MAPE at n=4 %", 19.77, 0.1, lambda: m6_ab(4, "lens_mape_pct")),

    # --- session 14: the regime the paper called unmeasurable ---------------
    ("M14.share.e1", "4.3", "paid padding share at n=16, 512/1024 %", -15.4, 2.0,
     lambda: m14_share("n16:512/1024")),
    ("M14.share.e2", "4.3", "paid padding share at n=16, 1024/2048 %", -2.7, 2.0,
     lambda: m14_share("n16:1024/2048")),
    ("M14.share.e3", "4.3", "paid padding share at n=16, 2048/4096 %", 0.5, 2.0,
     lambda: m14_share("n16:2048/4096")),

    # --- session 15: the operator breakdown (review Q2) ---------------------
    ("M12.attn.n1", "4.5", "attention share of device time at n=1 %", 6.8, 0.3,
     lambda: m12_cat(1, "attention")),
    ("M12.attn.n16", "4.5", "attention share of device time at n=16 %", 34.2, 0.3,
     lambda: m12_cat(16, "attention")),
    ("M12.matmul.n1", "4.5", "matmul share of device time at n=1 %", 78.5, 0.3,
     lambda: m12_cat(1, "matmul")),
    ("M12.matmul.n16", "4.5", "matmul share of device time at n=16 %", 51.4, 0.3,
     lambda: m12_cat(16, "matmul")),
    ("M12.coll.n4", "4.5", "collective share of device time at n=4 %", 13.9, 0.3,
     lambda: m12_cat(4, "collective")),

    # --- paid share at n=8, recomputed with splits EXCLUDED -----------------
    # The 16% previously reported pooled split dispatches, which the ninth
    # failure showed biases the share upward. These are the clean-only values.
    ("M14.share.n8.e2", "4.3", "paid share n=8, 2048/4096, clean only %", 21.0, 2.0,
     lambda: m8_clean("n8:2048/4096")),
    ("M14.share.n8.e3", "4.3", "paid share n=8, 4096/8192, clean only %", 14.3, 2.0,
     lambda: m8_clean("n8:4096/8192")),

    # --- session 16: the mechanism at its purest, and a failed prediction ---
    # RETRACTED 2026-08-11. These timed per-dispatch overhead, not the weight
    # load: 7.86 MB of qkv weights at 142.9 us implies 55 GB/s, 7% of peak. The
    # sweep also could not discriminate tiling from bandwidth, because for this
    # shape both predict a flat curve over M in [1, 256]. Replaced by m17.
    # ("M15.flat.qkv", "4.5", "qkv matmul time M=256 / M=1", 1.00, 0.10, ...),
    # ("M15.flat.mlp", "4.5", "mlp_up matmul time M=256 / M=1", 1.08, 0.10, ...),
    ("M16.tiny.n4", "4.5", "TinyLlama paid share n=4, 1024/2048 %", 13.4, 1.5,
     lambda: m16_share("n4:1024/2048")),
    ("M16.tiny.n8", "4.5", "TinyLlama paid share n=8, 1024/2048 %", 5.9, 1.5,
     lambda: m16_share("n8:1024/2048")),

    # --- the analytic frontier (review M1/Q3) -------------------------------
    ("M13.margin.256", "4.5", "margin to ridge at 256-token context %", 10.0, 1.0,
     lambda: m13_front(256, "margin_pct")),
    ("M13.margin.8192", "4.5", "margin to ridge at 8192-token context %", 96.0, 1.0,
     lambda: m13_front(8192, "margin_pct")),
    ("M13.mfu256.top", "4.5", "MFU at ladder top, 256-token context %", 49.0, 1.0,
     lambda: m13_front(256, "mfu_at_ladder_top_pct")),

    # --- M1, the randomised straddle ---------------------------------------
    ("M1.e1.cost", "5", "edge 512/1024 cost ratio", 1.110, 0.01, lambda: m1_edge("512/1024", "cost_ratio")),
    ("M1.e2.cost", "5", "edge 1024/2048 cost ratio", 1.070, 0.01, lambda: m1_edge("1024/2048", "cost_ratio")),
    ("M1.e1.share", "5", "edge 512/1024 share of padding paid %", 9.6, 0.6,
     lambda: (lambda c, r, p: (100 * (c - r) / (p - r), "m1"))(
         m1_edge("512/1024", "cost_ratio")[0], m1_edge("512/1024", "real_ratio")[0],
         m1_edge("512/1024", "padded_ratio")[0])),
    ("M1.e2.share", "5", "edge 1024/2048 share of padding paid %", 6.3, 0.6,
     lambda: (lambda c, r, p: (100 * (c - r) / (p - r), "m1"))(
         m1_edge("1024/2048", "cost_ratio")[0], m1_edge("1024/2048", "real_ratio")[0],
         m1_edge("1024/2048", "padded_ratio")[0])),

    # --- cost curve --------------------------------------------------------
    # RETIRED 2026-08-10: encodes the below-512 floor rule M5 invalidated.
    # ("CM.c512", "6", "cost of a 512-token step", 13.15, 0.05, lambda: (curve_cost(512), "curve")),
    # RETIRED 2026-08-10: encodes the below-512 floor rule M5 invalidated.
    # ("CM.c4096", "6", "cost of a 4096-token step", 69.08, 0.05, lambda: (curve_cost(4096), "curve")),
    # RETIRED 2026-08-10: encodes the below-512 floor rule M5 invalidated.
    # ("CM.c8192", "6", "cost of an 8192-token step", 144.75, 0.05, lambda: (curve_cost(8192), "curve")),
    # RETIRED 2026-08-10: encodes the below-512 floor rule M5 invalidated.
    # ("CM.us1", "6", "us/token at n=1", 25.7, 0.2, lambda: (curve_cost(512) / 512 * 1000, "curve")),
    # RETIRED 2026-08-10: encodes the below-512 floor rule M5 invalidated.
    # ("CM.us8", "6", "us/token at n=8", 16.9, 0.2, lambda: (curve_cost(4096) / 4096 * 1000, "curve")),
    # --- session 12: what the paper now leads with -------------------------
    ("M6.slope.n1", "4", "prefill slope at n=1 (flat, a staircase)", 1.6, 0.8, lambda: m6_slope(1)),
    ("M6.slope.n2", "4", "prefill slope at n=2 (flat)", 0.8, 0.8, lambda: m6_slope(2)),
    ("M6.slope.n4", "4", "prefill slope at n=4 (linear)", 17.2, 1.5, lambda: m6_slope(4)),
    ("M6.clean.n4", "4", "clean single-step cells at n=4", 3.0, 0.5, lambda: m6_clean_cells(4)),
    ("M6.clean.n8", "4", "clean single-step cells at n=8 (NONE)", 0.0, 0.5, lambda: m6_clean_cells(8)),
    ("M6.clean.n16", "4", "clean single-step cells at n=16 (NONE)", 0.0, 0.5, lambda: m6_clean_cells(16)),
    ("M6.clean.n32", "4", "clean single-step cells at n=32 (NONE)", 0.0, 0.5, lambda: m6_clean_cells(32)),
    ("M6.decode.n1", "4", "decode ms per step at n=1", 3.80, 0.15, lambda: m6_decode(1)),
    ("M6.decode.n32", "4", "decode ms per step at n=32", 9.13, 0.30, lambda: m6_decode(32)),
    ("M6.decodeseq.n1", "4", "decode us/step/seq at n=1", 3801.5, 60.0, lambda: m6_decode(1, True)),
    ("M6.decodeseq.n32", "4", "decode us/step/seq at n=32", 285.2, 10.0, lambda: m6_decode(32, True)),
    ("M6.lens.mape", "9", "LENS holdout MAPE on TPU", 5.23, 0.3, lambda: m6_lens_mape()),
]


# --- the guardrail --------------------------------------------------------
# Six of this project's errors share one cause: a quantity measured under one
# configuration used under another. The first version of this rule said "no
# derivation may combine quantities measured at different BATCH SIZES", which
# would not have caught session 4's failure -- a cost model fitted at
# output_len=8 and run at output_len=1. The general form costs the same to
# implement and catches the whole class.
#
# For any claim derived from more than one run, diff the runs' configs and
# require the claim to name every differing field as one it asserts invariance
# over. Resolving to be careful is not a control; this is.
# Diff EVERY config key, not a chosen list. A whitelist missed the axis that
# caused the error it was built for: batch size is not a top-level field -- it
# lives inside `edges` for the straddle experiments and is implicit (n=1) in
# M3's `token_sizes`. Any whitelist someone writes will omit the field that
# matters next time, so the default is "everything differs until asserted", and
# free text is the only exemption.
IGNORED_FIELDS = {"description", "mode", "mode_label", "warmup_log_path"}


def run_config(run_id: str) -> dict[str, Any] | None:
    for d in glob.glob(str(ROOT / "captured" / "*" / "results" / "*" / run_id)):
        m = pathlib.Path(d) / "meta.json"
        if m.exists():
            return json.loads(m.read_text()).get("config")
    return None


def differing_fields(run_ids: list[str]) -> list[str]:
    cfgs = [c for c in (run_config(r) for r in run_ids) if c]
    if len(cfgs) < 2:
        return []
    keys = set()
    for c in cfgs:
        keys |= {k for k in c if not k.startswith("note_") and k not in IGNORED_FIELDS}
    out = []
    for f in sorted(keys):
        vals = {json.dumps(c.get(f), sort_keys=True, default=str) for c in cfgs}
        if len(vals) > 1:
            out.append(f)
    return out


def check_invariance(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report claims that combine runs differing in an unasserted config field."""
    flags = []
    for c in claims:
        ids = [x for x in str(c.get("source_run", "")).split(";") if x and "__" in x]
        if len(ids) < 2:
            continue
        diff = differing_fields(ids)
        unasserted = [f for f in diff if f not in c.get("invariant_over", ())]
        if unasserted:
            flags.append({"claim_id": c["claim_id"], "n_runs": len(ids),
                          "unasserted_fields": ",".join(unasserted)})
    return flags


# Retired claims, kept so the guardrail can be shown working on the error it was
# built for. The crossover combined a step cost measured at n=1 with a padding
# fraction measured at n=4; both authors of this project built that equation,
# and both noticed one turn too late.
RETIRED = [
    # WITHDRAWN 2026-08-11 on review finding M2/Q5. "~4-9% of execution is
    # recoverable" was the paper's headline actionable number and appeared in the
    # abstract, §4.4, §7 and §9. It multiplies a padding SHARE measured under a
    # fixed-256 workload by a PAID share measured at n=4 under boundary-straddling
    # near-fixed lengths. Registering it as a claim -- which it never was, because
    # it lived only in prose -- made the guardrail evaluate it, and it flagged
    # `concurrency` and `prompt_len`: the two axes the reviewer named independently.
    #
    # Invariance over `concurrency` cannot be asserted, because §4.3's own finding
    # is that the paid share MOVES with it: ~85% at n=1-2, 10-25% at n=4-8. The
    # derivation therefore commits the exact error §6 names as this project's root
    # cause, in the paper that names it.
    #
    # It is withdrawn rather than corrected. A defensible replacement needs the
    # padding share and the paid share measured under ONE workload at ONE batch
    # size, which is what m10_trace_workload.py and the m8 barrier re-runs are for.
    # Until then the paper states the two measured quantities separately and does
    # not multiply them.
    {"claim_id": "S7.recoverable.lo(RETIRED)", "section": "7",
     "claim": "~4% of execution recoverable (low end) -- WITHDRAWN, cross-configuration",
     "source_run": "e05_step_shape__20260810T051309Z__f7f642b502fb;"
                   "m1_boundary__20260810T182212Z__0c4087987fd2",
     "invariant_over": ()},
    {"claim_id": "S7.recoverable.hi(RETIRED)", "section": "7",
     "claim": "~9% of execution recoverable (high end) -- WITHDRAWN, cross-configuration",
     "source_run": "e05_step_shape__20260810T051309Z__f7f642b502fb;"
                   "m1_boundary__20260810T182212Z__0c4087987fd2",
     "invariant_over": ()},
    {"claim_id": "S5.crossover(RETIRED)", "section": "5",
     "claim": "step-for-alignment crosses at ~2048 tokens",
     "source_run": "m3_small_steps__20260810T182131Z__9b0dc2baf9ad;m1_boundary__20260810T182212Z__0c4087987fd2",
     "invariant_over": ()},
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    rows: list[dict[str, Any]] = []
    bad, unver = 0, 0
    for claim in CLAIMS:
        cid, sec, what, stated, tol, fn = claim[:6]
        inv = claim[6] if len(claim) > 6 else ()
        try:
            got, src = fn()
        except Exception as exc:  # noqa: BLE001
            got, src = float("nan"), f"ERROR: {type(exc).__name__}"
        if got != got:
            verdict = "UNVERIFIED"; unver += 1
        elif abs(got - stated) <= tol:
            verdict = "ok"
        else:
            verdict = "MISMATCH"; bad += 1
        rows.append({"claim_id": cid, "section": sec, "claim": what,
                     "stated": stated, "recomputed": got, "tolerance": tol,
                     "delta": got - stated if got == got else float("nan"),
                     "verdict": verdict, "source_run": src, "invariant_over": inv})
        mark = {"ok": "  ", "MISMATCH": "!!", "UNVERIFIED": "??"}[verdict]
        print(f"{mark} {cid:<20} §{sec:<4} stated {stated:>8.2f}  recomputed "
              f"{got:>8.2f}  {verdict}")

    print(f"\n{len(rows)} claims: {len(rows) - bad - unver} ok, {bad} MISMATCH, {unver} UNVERIFIED")

    flags = check_invariance(rows + RETIRED)
    print(f"\n--- invariance guardrail: {len(flags)} claim(s) combine runs with "
          f"unasserted config differences ---")
    for f in flags:
        print(f"  !! {f['claim_id']:<28} {f['n_runs']} runs differ in: {f['unasserted_fields']}")
    if not flags:
        print("  (none)")
    live = [f for f in flags if "RETIRED" not in f["claim_id"]]
    if live:
        bad += len(live)
    if args.write:
        import pyarrow as pa
        out = ROOT / "results" / "paper_numbers.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        pa.parquet.write_table(pa.Table.from_pylist(rows), out) if hasattr(pa, "parquet") else \
            pq.write_table(pa.Table.from_pylist(rows), out)
        print(f"wrote {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
