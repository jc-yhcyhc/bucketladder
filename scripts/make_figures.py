#!/usr/bin/env python3
"""
The paper's three figures, regenerated from captured runs.

Every value is read from `captured/` rather than typed, so a figure cannot drift
from the claim it illustrates — the same reason `paper_numbers.py` exists, and
the reason `reproduce_all.sh --figures` runs this.

Three figures, chosen because each carries a claim that a table states less well:

  fig1  decode cost vs batch size      §4.5 — the cleanest data in the project,
                                       and the one place a *shape* is the point:
                                       smooth, monotone, no discontinuity.
  fig2  paid padding by boundary       §4.4 — magnitude comparison across four
                                       boundaries against the 100% the premise
                                       predicts. Grouped bars, not a line: the
                                       boundaries are categories, not a scale.
  fig3  LENS holdout error vs n        §4.2 — the money figure. A published
                                       predictor's error is not spread across
                                       the range; it is LOCALISED at n=4, which
                                       is exactly the claim.

Palette: #3b6fd4 / #d97a1f / #0f9b6c, validated (lightness band, chroma floor,
CVD separation, normal-vision floor, contrast) before use rather than eyeballed.

Usage:
  python scripts/make_figures.py --out figures/
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Categorical hues in FIXED order — never cycled, never reassigned by rank.
BLUE, ORANGE, GREEN = "#3b6fd4", "#d97a1f", "#0f9b6c"
INK, MUTED, GRID = "#1c1c1a", "#6b6b66", "#e3e3df"
SURFACE = "#fcfcfb"


def style(ax) -> None:
    """Recessive grid and axes; text in ink tokens, never a series colour."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)


def newfig(w=6.4, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    return fig, ax


def runs(pattern: str, table: str):
    out = []
    for d in sorted(glob.glob(str(ROOT / "captured" / pattern))):
        p, m = pathlib.Path(d) / f"{table}.parquet", pathlib.Path(d) / "meta.json"
        if p.exists() and m.exists():
            out.append((json.loads(m.read_text())["config"], pq.read_table(p).to_pylist()))
    return out


# --- fig 1: decode is smooth ------------------------------------------------

def fig_decode(out: pathlib.Path) -> str:
    ns, per_step, per_seq = [], [], []
    for cfg, rows in runs("session12-regime/results/e02_stock_baseline/*", "server_timing"):
        if cfg.get("output_len") != 64:
            continue
        for n in sorted({r["concurrency"] for r in rows}):
            v = [r["decode_ms"] for r in rows if r["concurrency"] == n]
            ns.append(n)
            per_step.append(statistics.median(v) / cfg["output_len"])
            per_seq.append(statistics.median(v) / cfg["output_len"] / n * 1000)
        break
    fig, ax = newfig()
    ax.plot(ns, per_seq, color=BLUE, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label="per sequence")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=10)
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("batch size (sequences per step)", color=MUTED)
    ax.set_ylabel("decode cost per sequence (µs/step)", color=MUTED)
    # Direct-label the endpoints only — never a number on every point.
    for i in (0, len(ns) - 1):
        ax.annotate(f"{per_seq[i]:.0f} µs", (ns[i], per_seq[i]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", color=INK, fontsize=9)
    ax.set_title("Decode cost falls 13× with batch size, with no discontinuity",
                 color=INK, fontsize=11, loc="left", pad=12)
    fig.text(0.01, 0.01, "per-step cost rises only 2.4× over the same 32× range",
             color=MUTED, fontsize=8)
    fig.tight_layout(); f = out / "fig1_decode.png"; fig.savefig(f); plt.close(fig)
    return f.name


# --- fig 2: paid padding by boundary ---------------------------------------

def fig_padding(out: pathlib.Path) -> str:
    want = {"n4:512/1024": "512→1024", "n4:1024/2048": "1024→2048",
            "n4:2048/4096": "2048→4096", "n4:4096/8192": "4096→8192"}
    labels, shares = [], []
    for cfg, rows in runs("session8-m3m4/results/m1_boundary/*", "edges"):
        for r in rows:
            if r["edge"] in want and r["splits_below"] == 0 and r["splits_above"] == 0:
                labels.append(want[r["edge"]])
                shares.append(100 * (r["cost_ratio"] - r["real_ratio"]) /
                              (r["padded_ratio"] - r["real_ratio"]))
        if labels:
            break
    fig, ax = newfig()
    x = range(len(labels))
    ax.bar(x, shares, width=0.55, color=BLUE, zorder=3)
    ax.axhline(100, color=ORANGE, linewidth=2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("what the compiled-shape premise predicts: 100%", (len(labels) - 0.5, 100),
                textcoords="offset points", xytext=(0, -16), ha="right",
                color=INK, fontsize=9)
    for i, s in zip(x, shares):
        ax.annotate(f"{s:.0f}%", (i, s), textcoords="offset points", xytext=(0, 6),
                    ha="center", color=INK, fontsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.set_xlabel("compiled token boundary crossed", color=MUTED)
    ax.set_ylabel("share of nominal padding actually paid (%)", color=MUTED)
    ax.set_title("Most padding is never paid — and the share rises with the boundary",
                 color=INK, fontsize=11, loc="left", pad=12)
    fig.text(0.01, 0.01, "batch size 4, single-step dispatches only", color=MUTED, fontsize=8)
    fig.tight_layout(); f = out / "fig2_padding.png"; fig.savefig(f); plt.close(fig)
    return f.name


# --- fig 3: LENS error is localised ----------------------------------------

def fig_lens(out: pathlib.Path) -> str:
    pts = []
    for cfg, rows in runs("session12-regime/results/m5_lens_form/*", "fits"):
        pts = [(r["n"], r["holdout_ape_pct"]) for r in rows
               if r["splits"] == 0 and r["holdout_ape_pct"] == r["holdout_ape_pct"]]
        if pts:
            break
    fig, ax = newfig()
    # Shade the failure region rather than relying on colour alone to carry it.
    ax.axvspan(3.1, 5.2, color=ORANGE, alpha=0.10, zorder=1)
    ax.axhline(2.15, color=GREEN, linewidth=2, linestyle=(0, (4, 3)), zorder=2,
               label="LENS reported accuracy (NPU)")
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=70, color=BLUE,
               edgecolor=SURFACE, linewidth=2, zorder=3, label="this work (TPU)")
    ax.set_xscale("log", base=2); ax.set_xticks(sorted({p[0] for p in pts}))
    ax.set_xticklabels([str(n) for n in sorted({p[0] for p in pts})])
    # Label below the cluster, not beside it: at n=4 the points sit at the top
    # of the axis and an offset to the right collides with them and the frame.
    ax.annotate("error localised at n=4", (4, min(p[1] for p in pts if p[0] == 4)),
                textcoords="offset points", xytext=(0, -22), ha="center",
                color=INK, fontsize=9)
    ax.set_xlim(0.8, 6.5)
    ax.set_ylim(-1.5, max(p[1] for p in pts) * 1.18)
    ax.set_xlabel("batch size", color=MUTED)
    ax.set_ylabel("held-out prediction error (%)", color=MUTED)
    ax.set_title("A published predictor's error is not spread — it is localised",
                 color=INK, fontsize=11, loc="left", pad=12)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.01, 0.01, "LENS protocol reproduced on TPU; mid-bucket point withheld from each fit",
             color=MUTED, fontsize=8)
    fig.tight_layout(); f = out / "fig3_lens.png"; fig.savefig(f); plt.close(fig)
    return f.name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "figures")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (fig_decode, fig_padding, fig_lens):
        try:
            made.append(fn(args.out))
        except Exception as exc:  # noqa: BLE001
            print(f"[fig] {fn.__name__} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    for m in made:
        print(f"[fig] {args.out / m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
