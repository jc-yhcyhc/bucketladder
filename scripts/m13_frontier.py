#!/usr/bin/env python3
"""
M13 — where does free padding END? The bound the paper asserts around instead of.

Review M1/Q3, and the sharpest criticism of draft 3:

  "Decode is measured to n=32 ... the compiled request ladder runs to 256. The
   paper's own MFU series (0.27% -> 1.68% -> 3.65% at n=1/8/32) roughly doubles
   per doubling of batch, which extrapolates to tens of percent by n=128-256 --
   i.e. out of the memory-bound regime that makes padding free. 'By n=16 it is
   free' is therefore measured at one, possibly two points before the mechanism
   is expected to weaken ... The paper needs either measurements at n>=64 or an
   explicit analytic bound: given the weight floor, KV bytes per step, and peak
   HBM BW, at what (batch, context) does the paid share become nonzero?"

That bound is derivable from bytes and flops already counted by `m9_roofline.py`,
and the review is right that asserting freedom at the edge of the measured range
is the paper's weakest move. This computes the frontier instead.

THE DERIVATION. Per chip, per decode step, ignoring nothing that scales:

    bytes(n, L) = W + n * L * kv_per_token          W = all weights, every step
    flops(n, L) = 2 * P * n + attention(n, L)       P = params per chip

A step is memory-bound while `bytes/BW > flops/peak_flops`. Padding a request
slot adds neither KV bytes (RPA's padded slots hold no blocks) nor real tokens,
so while the step is memory-bound the padded slot rides inside a floor that
batch size does not move -- which is the paper's mechanism. It stops being free
when the compute term catches the memory term.

Setting the two equal and solving for n gives the frontier. With attention flops
small relative to the projection matmuls at these sizes, the crossover sits near

    n* ~= peak_flops / BW / 2      ... the classic ridge point, in tokens/step

and the KV term moves it down as context grows, because KV bytes scale with n*L
while weight bytes do not.

WHAT MAKES THIS FALSIFIABLE. The frontier is a prediction about a regime we did
not measure, stated as a number rather than a hedge. If a later session measures
paid padding at n=64 or n=128 and finds it nonzero well below the frontier, this
bound is wrong and the mechanism is incomplete. It is written to be checkable.

Usage:
  python scripts/m13_frontier.py --config configs/m9_roofline.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import finish_run, load_config, save_table, start_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=pathlib.Path,
                    default=HERE.parent / "configs" / "m9_roofline.json")
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    src = load_config(args.config)
    a, hw = src["arch"], src["hardware"]
    tp = src["controlled"]["tensor_parallel_size"]

    cfg = {"experiment": "m13_frontier", "dimension": "none", "mode": "offline", "model": src.get("model"),
           "controlled": src["controlled"], "arch": a, "hardware": hw,
           "note_derivation": ("Analytic frontier from the same byte and flop accounting as "
                               "m9_roofline; no new measurement. Answers review Q3, which asked "
                               "for a bound rather than an assertion of freedom at n>=16.")}
    run = start_run("m13_frontier", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        L_, H = a["num_hidden_layers"], a["hidden_size"]
        hd, nkv, nh = a["head_dim"], a["num_key_value_heads"], a["num_attention_heads"]
        inter, vocab, dt = a["intermediate_size"], a["vocab_size"], a["dtype_bytes"]
        q = H * nh * hd; kvp = 2 * H * nkv * hd; o = nh * hd * H
        params = L_ * (q + kvp + o + 3 * H * inter) + vocab * H
        P = params / tp
        W = P * dt
        peak, bw = hw["peak_bf16_flops_per_chip"], hw["hbm_bw_bytes_per_s_per_chip"]
        kv_per_tok = 2 * (nkv / tp) * hd * L_ * a["kv_dtype_bytes"]

        ridge = peak / bw          # flops per byte at which the two roofs meet
        print(f"[m13] weights/chip {W / 1e9:.2f} GB; ridge point {ridge:.0f} FLOP/byte")
        print(f"[m13] KV per token/chip {kv_per_tok / 1024:.1f} KiB")

        rows: list[dict[str, Any]] = []
        for ctx in (256, 1024, 4096, 8192):
            frontier = None
            prev_bound = None
            for n in range(1, 4097):
                byt = W + n * ctx * kv_per_tok
                fl = 2 * P * n + 2 * 2 * n * ctx * (nh / tp) * hd * L_
                bound = "memory" if byt / bw > fl / peak else "compute"
                if prev_bound == "memory" and bound == "compute":
                    frontier = n
                    break
                prev_bound = bound
            # Where the weight floor stops dominating the byte budget, which is the
            # separate question of when a padded slot stops being cheap to CARRY.
            n_half = W / (ctx * kv_per_tok)
            # "Never crosses" is the wrong thing to report on its own. As n grows the
            # weight term is amortised away and the intensity tends to a LIMIT set by
            # the per-sequence terms alone. If that limit sits just under the ridge,
            # the regime is marginal at the top of the ladder even though it never
            # formally crosses -- which is what the review's MFU extrapolation is
            # actually detecting, and it is right to.
            flops_per_seq = 2 * P + 2 * 2 * ctx * (nh / tp) * hd * L_
            bytes_per_seq = ctx * kv_per_tok
            limit = flops_per_seq / bytes_per_seq
            # Utilisation at the top of the compiled request ladder.
            n_top = 256
            byt_top = W + n_top * bytes_per_seq
            fl_top = 2 * P * n_top + 2 * 2 * n_top * ctx * (nh / tp) * hd * L_
            t_top = max(byt_top / bw, fl_top / peak)
            rows.append({"context_tokens": ctx,
                         "frontier_n": frontier if frontier else -1,
                         "n_where_kv_equals_weights": n_half,
                         "limit_flops_per_byte": limit,
                         "ridge_flops_per_byte": ridge,
                         "margin_pct": 100 * (1 - limit / ridge),
                         "mfu_at_ladder_top_pct": 100 * (fl_top / t_top) / peak,
                         "weights_gb": W / 1e9,
                         "kv_bytes_per_seq": bytes_per_seq})
        save_table(run, "frontier", rows)

        print(f"[m13] {'context':>8} {'frontier n*':>12} {'KV=W at n':>11} "
              f"{'limit FLOP/byte':>16} {'margin to ridge':>16} {'MFU at n=256':>13}")
        for r in rows:
            f = r["frontier_n"]
            print(f"[m13] {r['context_tokens']:>8} "
                  f"{(str(f) if f > 0 else '>4096'):>12} "
                  f"{r['n_where_kv_equals_weights']:>11.0f} "
                  f"{r['limit_flops_per_byte']:>16.0f} "
                  f"{r['margin_pct']:>15.0f}% {r['mfu_at_ladder_top_pct']:>12.0f}%")

        ladder_top = 256
        worst = min((r["frontier_n"] for r in rows if r["frontier_n"] > 0), default=-1)
        print("[m13] --- what this bounds ---")
        tight = min(rows, key=lambda r: r["margin_pct"])
        if worst < 0:
            print(f"[m13] No formal crossover within n<=4096: the intensity rises toward a LIMIT "
                  f"rather than through the ridge, because KV bytes grow with n alongside the "
                  f"flops. The whole compiled ladder (top {ladder_top}) stays nominally "
                  f"memory-bound.")
            print(f"[m13] But the margin is not comfortable everywhere. At context "
                  f"{tight['context_tokens']} the limit is {tight['limit_flops_per_byte']:.0f} "
                  f"FLOP/byte against a ridge of {ridge:.0f} -- only "
                  f"{tight['margin_pct']:.0f}% of headroom -- and MFU at the top of the ladder "
                  f"reaches {tight['mfu_at_ladder_top_pct']:.0f}%.")
            print(f"[m13] The review's extrapolation is therefore CORRECT in direction: free "
                  f"padding is comfortable at long context and marginal at short context and "
                  f"high batch. The paper must say 'free across the ladder at the contexts "
                  f"measured, with the margin narrowing to {tight['margin_pct']:.0f}% at "
                  f"{tight['context_tokens']}-token context', not 'free'.")
        else:
            print(f"[m13] The memory-bound regime ends at n*={worst} in the worst case measured. "
                  f"The compiled request ladder tops out at {ladder_top}.")
            if worst > ladder_top:
                print(f"[m13] n* > {ladder_top}, so the whole ladder sits inside the free regime "
                      f"and no reachable batch size on this stack leaves it.")
            else:
                print(f"[m13] n* <= {ladder_top}: part of the ladder is OUTSIDE the free regime, "
                      f"and the paper must qualify its advice to n < {worst}.")
        print("[m13] FALSIFIABLE: measure the paid share at n=64 or n=128. If it is nonzero "
              "well below the frontier above, this bound is wrong and the mechanism is "
              "incomplete.")
        print(f"[m13] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
