#!/usr/bin/env python3
"""
O6 — does the ladder recommendation survive prefix caching?

Prefix caching is off and asserted in every other experiment here, and it is ON
by default in production vLLM. That is not a cosmetic difference for this paper:
caching removes already-computed prefix tokens from the prefill, so the step
lands on a *different compiled shape* than the request's length would suggest.
§4.9 recommends placing ladder entries against the prompt length distribution; if
caching shifts which shape a prompt actually uses, the recommendation has to be
stated against the *uncached* prefill length instead, and a reader running with
the production default would otherwise place their entries in the wrong place.

The workload has to have something to cache. `token_ids` gives a different
sequence per seed, so the obvious harness produces requests with nothing in
common and caching does nothing to it — a result that would say nothing about
production. Here every request shares a fixed prefix and varies only its tail,
which is the shape of a system prompt or a few-shot preamble.

FOUR ARMS, because the question is an interaction rather than a main effect:

    default ladder,  caching off   <- the regime every other section measures
    default ladder,  caching on    <- the production default
    gap1024 ladder,  caching off   <- §4.9's placement win, as measured there
    gap1024 ladder,  caching on    <- does that win survive?

REGISTERED PREDICTION. With a 2048-token shared prefix and a 3000-token prompt,
caching should cut the prefill to about 952 tokens, which pads to 1024 on BOTH
ladders — the two ladders differ only at 3072 against 4096. So the placement
benefit at this prompt length should largely VANISH when caching is on, not
because the mechanism is wrong but because caching moves the step off the entry
the placement was chosen for. If the benefit instead survives intact, then the
prefill is not being shortened the way the cache-hit accounting implies, and the
mechanism in §4.9 needs re-examining.

Either outcome sharpens the recommendation: place entries against the length
distribution *the server actually prefills*, which caching changes.

Usage:
  python scripts/o6_prefix_caching.py --config configs/o6_prefix_caching.json \\
      --arm default_apc_on --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import re  # noqa: E402
import urllib.request  # noqa: E402

from _client import complete  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import metrics_available  # noqa: E402

ARMS = ["default_apc_off", "default_apc_on", "gap1024_apc_off", "gap1024_apc_on"]

# `_metrics.scrape` returns histogram snapshots; this counter is a plain scalar,
# so read the endpoint directly rather than bending that helper's return shape.
CACHED_RE = re.compile(r"^vllm:prompt_tokens_cached[^ {]*(?:\{[^}]*\})?\s+([0-9.e+-]+)",
                       re.M)


def cached_tokens(base_url: str) -> float:
    """Prompt tokens vLLM reports as served from cache. NaN if not exported.

    This is the arm's own proof that caching did something. Without it, an
    'APC on' arm that silently failed to enable caching -- a flag not taking, a
    workload with no shared prefix -- would look exactly like a true null.
    """
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=10) as r:
            body = r.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return float("nan")
    vals = [float(m.group(1)) for m in CACHED_RE.finditer(body)]
    return sum(vals) if vals else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    cfg = dict(load_config(args.config))
    cfg["arm"] = args.arm
    cfg["controlled"] = dict(cfg["controlled"])
    cfg["controlled"]["enable_prefix_caching"] = args.arm.endswith("_apc_on")
    cfg["controlled"]["VLLM_TPU_BUCKET_PADDING_GAP"] = (
        "1024" if args.arm.startswith("gap1024") else "")
    if not metrics_available(args.base_url):
        print(f"[o6] no /metrics at {args.base_url}", file=sys.stderr)
        return 1
    try:
        run = start_run("o6_prefix_caching", cfg, results_root=args.results_root)
    except ControlledVarError as e:
        print(f"[o6] {e}", file=sys.stderr)
        return 2

    status, err = "ok", None
    try:
        rows, reps = [], []
        for plen in cfg["prompt_lens"]:
            pre = cfg["prefix_len"] if plen > cfg["prefix_len"] else 0
            c0 = cached_tokens(args.base_url)
            lat = []
            for rep in range(cfg["repeats"] + cfg["warmup_discard"]):
                with ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
                    out = list(pool.map(
                        lambda i: complete(args.base_url, cfg["model"], plen,
                                           cfg["output_len"], seed=rep * 100 + i,
                                           prefix_len=pre),
                        range(cfg["concurrency"])))
                if rep < cfg["warmup_discard"]:
                    continue
                ok = [s for s in out if s.ok]
                bad = [s for s in out if not s.ok]
                if bad and rep == cfg["warmup_discard"]:
                    print(f"[o6:{args.arm}] plen={plen} {len(bad)}/{len(out)} "
                          f"FAILED: {bad[0].error!r}", file=sys.stderr)
                if ok:
                    lat.append(statistics.median([s.total_ms for s in ok]))
                    reps.append({"arm": args.arm, "prompt_len": plen, "rep": rep,
                                 "e2e_ms": lat[-1], "prefix_len": pre})
            c1 = cached_tokens(args.base_url)
            if lat:
                rows.append({"arm": args.arm, "prompt_len": plen, "prefix_len": pre,
                             "e2e_ms_median": statistics.median(lat),
                             "reps": len(lat), "cached_delta": c1 - c0})
                print(f"[o6:{args.arm}] plen={plen:<5} prefix={pre:<5} "
                      f"e2e {statistics.median(lat):8.1f} ms   "
                      f"cached+{c1 - c0:.0f}")
        if not rows:
            raise RuntimeError("no latency rows; arm not measured")
        save_table(run, "latency", rows)
        save_table(run, "latency_reps", reps)
        print(f"[o6] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
