#!/usr/bin/env python3
"""
O1 analysis — does a finer token ladder buy anything, and what does it cost?

The experiment has a placebo built into it, and that is the point of the design.
Of the four prompt lengths, two (300, 600) pad to the SAME compiled shape on both
ladders, and two (1200, 3000) pad to a smaller shape on the fine ladder only:

    prompt   default ladder   gap512 ladder   padded tokens saved
      300         512              512                0     <- placebo
      600        1024             1024                0     <- placebo
     1200        2048             1536              512
     3000        4096             3072             1024

So the placebo cells measure everything that differs between two server boots
EXCEPT padding: process state, cache warmth, run order, ordinary variance. If the
effect is real, it appears in the two treated cells and not in the two placebo
cells, and its size tracks tokens saved rather than tracking prompt length. If
instead all four cells move together, what we are looking at is an arm-level
offset -- a slower boot -- and there is no padding effect to report.

Arms are separate boots of vLLM, so the two-sample bootstrap is the unpaired one;
see the note in _stats.bootstrap_ci_unpaired for why pairing here would be an
artefact of loop order rather than a property of the data.

Usage:
  python scripts/o1_analyze.py                    # all runs under results/
  python scripts/o1_analyze.py --results-root results
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _stats import bootstrap_ci_unpaired, bootstrap_p, fmt_ci  # noqa: E402

# Where the two ladders coincide, and where they do not. Derived from the ladders
# the servers actually printed, not assumed: see the assertion in main().
DEFAULT_LADDER = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
GAP512_LADDER = [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584,
                 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]


def pad_to(ladder: list[int], n: int) -> int:
    for s in ladder:
        if s >= n:
            return s
    return ladder[-1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=pathlib.Path, default=pathlib.Path("results"))
    args = ap.parse_args(argv)

    try:
        import pandas as pd
    except ImportError:
        print("[o1a] pandas required", file=sys.stderr)
        return 1

    root = args.results_root / "o1_ladder_cost"
    reps_files = sorted(root.glob("*/latency_reps.parquet"))
    ladder_files = sorted(root.glob("*/ladder.parquet"))
    if not reps_files:
        print(f"[o1a] no latency_reps.parquet under {root} — only runs made after "
              "per-repeat persistence was added can carry a CI", file=sys.stderr)
        return 1

    reps = pd.concat([pd.read_parquet(f) for f in reps_files], ignore_index=True)
    ladders = (pd.concat([pd.read_parquet(f) for f in ladder_files], ignore_index=True)
               if ladder_files else None)

    # --- warmup, the cost side -------------------------------------------
    if ladders is not None:
        print("=== Warmup: what the shapes cost ===")
        print(f"{'arm':<9}{'order':<7}{'shapes':>7}{'warmup_s':>10}{'s/shape':>9}")
        for _, r in ladders.sort_values(["arm", "order"]).iterrows():
            print(f"{r['arm']:<9}{str(r.get('order', '-')):<7}"
                  f"{int(r['n_token_shapes']):>7}{r['warmup_s']:>10.0f}"
                  f"{r['warmup_s_per_token_shape']:>9.1f}")
        print()

    # --- latency, the benefit side ---------------------------------------
    print("=== Latency: what the shapes buy (gap512 - default; negative = fine ladder faster) ===")
    print(f"{'prompt':>7}{'pad_def':>9}{'pad_512':>9}{'saved':>7}"
          f"{'default':>10}{'gap512':>9}{'delta_ms':>10}  {'95% CI':<22}{'p':>7}  kind")

    treated_effect: list[tuple[int, float]] = []
    placebo_effect: list[float] = []
    for plen in sorted(reps["prompt_len"].unique()):
        d = reps[(reps.prompt_len == plen) & (reps.arm == "default")]["e2e_ms"].tolist()
        g = reps[(reps.prompt_len == plen) & (reps.arm == "gap512")]["e2e_ms"].tolist()
        if not d or not g:
            continue
        pd_, pg = pad_to(DEFAULT_LADDER, plen), pad_to(GAP512_LADDER, plen)
        saved = pd_ - pg
        delta = statistics.fmean(g) - statistics.fmean(d)
        lo, hi = bootstrap_ci_unpaired(g, d)
        p = bootstrap_p(g, d)
        kind = "PLACEBO" if saved == 0 else "treated"
        (placebo_effect.append(delta) if saved == 0
         else treated_effect.append((saved, delta)))
        print(f"{plen:>7}{pd_:>9}{pg:>9}{saved:>7}"
              f"{statistics.fmean(d):>10.1f}{statistics.fmean(g):>9.1f}{delta:>10.1f}"
              f"  {f'[{lo:+.1f}, {hi:+.1f}]':<22}{p:>7.4f}  {kind}")

    # --- the discriminating test ------------------------------------------
    print("\n=== Does the effect track padded tokens, or is it an arm-level offset? ===")
    if placebo_effect:
        print(f"placebo cells (0 tokens saved): mean delta "
              f"{statistics.fmean(placebo_effect):+.1f} ms")
    for saved, delta in sorted(treated_effect):
        # delta is in MILLISECONDS, so ms/token x 1000 is MICROseconds/token.
        # Mislabelling this as ns put a figure 1000x too small into a draft.
        print(f"treated cell, {saved:>4} tokens saved: delta {delta:+7.1f} ms "
              f"-> {1000 * delta / saved:+7.1f} us per padded token")
    # The placebo cells are not exactly zero, and with intervals this tight the
    # offset is resolvable rather than lost in noise. Whatever it is -- server
    # state, a slightly warmer cache -- it is by construction NOT padding, since
    # those cells pad identically on both ladders. So subtract it: the corrected
    # figure is the padding term with the arm-level offset removed.
    if placebo_effect and len(treated_effect) >= 2:
        offset = statistics.fmean(placebo_effect)
        print(f"\n=== Placebo-corrected (subtracting the {offset:+.1f} ms arm offset) ===")
        for saved, delta in sorted(treated_effect):
            print(f"{saved:>4} tokens saved: {delta - offset:+7.1f} ms "
                  f"-> {1000 * (delta - offset) / saved:+7.1f} us per padded token")
        corr = [1000 * (dl - offset) / sv for sv, dl in treated_effect]
        cspread = (max(corr) - min(corr)) / abs(statistics.fmean(corr))
        print(f"corrected agreement: {cspread * 100:.0f}% spread")

    # Scale check. A padded token cannot plausibly cost more than a real one, so
    # deriving the real-token cost in the SAME units puts the padded figure next
    # to something that bounds it. Printing a derived rate with no comparable
    # neighbour is how a per-token cost reached a draft 1000x too small, labelled
    # ns when the arithmetic was ms/token x 1000 = us/token.
    lens = sorted(reps["prompt_len"].unique())
    if len(lens) >= 2:
        lo, hi = lens[0], lens[-1]
        dlo = reps[(reps.prompt_len == lo) & (reps.arm == "default")]["e2e_ms"]
        dhi = reps[(reps.prompt_len == hi) & (reps.arm == "default")]["e2e_ms"]
        if len(dlo) and len(dhi):
            real_us = 1000 * (dhi.mean() - dlo.mean()) / (hi - lo)
            print(f"\nscale check: a REAL prefill token costs {real_us:.1f} us "
                  f"({lo}->{hi} tokens on the default arm)")
            if treated_effect:
                pad_us = abs(statistics.fmean(
                    [1000 * dl / sv for sv, dl in treated_effect]))
                print(f"             a PADDED token costs {pad_us:.1f} us "
                      f"= {100 * pad_us / real_us:.0f}% of a real one")
                if pad_us > real_us * 1.5:
                    print("             WARNING: a padded token cannot cost more "
                          "than a real one — check units and the ladder mapping")

    if len(treated_effect) >= 2:
        per_tok = [1000 * dl / sv for sv, dl in treated_effect]
        spread = (max(per_tok) - min(per_tok)) / abs(statistics.fmean(per_tok))
        print(f"\nper-token cost agreement across treated cells: {spread * 100:.0f}% spread")
        print("Independent cells agreeing on cost-per-padded-token is the signature")
        print("of a padding effect; an arm-level offset would instead show a")
        print("CONSTANT delta and therefore a per-token cost that scales as 1/saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
