#!/usr/bin/env python3
"""
I1 — does bucket-aligned step packing pay, and does it change any output?

Run once per server mode (default / aligned) against the SAME build, with
`BUCKETLADDER_ALIGN` deciding which. The two runs are compared offline.

CORRECTNESS FIRST. The patch defers prefill tokens between steps. Chunked
prefill already does that constantly, so it should be invisible in the output —
but "should be" is not a measurement. This sends a fixed set of greedy
(temperature 0) completions and records a hash of each generated text. If the
hashes differ between modes the patch is wrong, and **no timing number from
either mode means anything**. That check runs first and its result is stored
alongside the timings so the two can never be read apart.

THEN THE A/B. A Poisson trace at several rates, measuring:

  TPU-busy per request   server-side, from /metrics deltas (the paper's unit)
  p50 / p99 latency      client-side, arrival to completion

Both sides are reported. The optimisation defers work, so it *must* be checked
for added tail latency; a throughput number on its own would be the one-sided
report the plan warned about.

Usage:
  python scripts/i1_eval.py --config configs/i1_eval.json --mode default \\
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sim"))

from _client import token_ids  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from _metrics import delta, metrics_available, scrape  # noqa: E402
from simulator import Simulator  # noqa: E402
from cost_model import CostModel  # noqa: E402

PREFILL = "vllm:request_prefill_time_seconds"


def greedy(base_url: str, model: str, prompt_len: int, out_len: int, seed: int,
           timeout: float = 120.0) -> str:
    """One deterministic completion. temperature=0 so any text difference is the
    patch's fault, not sampling."""
    body = json.dumps({"model": model, "prompt": token_ids(prompt_len, seed=seed),
                       "max_tokens": out_len, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["text"]


def run_rate(base_url: str, model: str, rate: float, n_req: int, prompt_len: int,
             out_len: int, seed: int) -> dict[str, Any]:
    """Replay a Poisson trace; measure server TPU-busy and client latency."""
    trace = Simulator(CostModel()).make_trace(n_req, rate, prompt_len, seed=seed)
    before = scrape(base_url)
    lat: list[float] = []
    lock = threading.Lock()
    t0 = time.perf_counter()

    def fire(rid: int, arrival: float) -> None:
        d = arrival - (time.perf_counter() - t0)
        if d > 0:
            time.sleep(d)
        s = time.perf_counter()
        try:
            greedy(base_url, model, prompt_len, out_len, rid)
        except Exception:  # noqa: BLE001
            return
        with lock:
            lat.append((time.perf_counter() - s) * 1000.0 + max(0.0, -d) * 1000.0)

    ths = [threading.Thread(target=fire, args=(r.rid, r.arrival_s), daemon=True) for r in trace]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    d = delta(before, scrape(base_url)).get(PREFILL)
    lat.sort()
    return {"rate_hz": rate, "n_ok": len(lat), "wall_s": wall,
            # sum over requests / n = TPU-busy per request, the paper's unit
            "tpu_ms_per_req": (d["sum_s"] * 1000.0 / len(lat)) if d and lat else float("nan"),
            "p50_latency_ms": statistics.median(lat) if lat else float("nan"),
            "p99_latency_ms": lat[int(0.99 * (len(lat) - 1))] if lat else float("nan"),
            "throughput_req_s": len(lat) / wall if wall else float("nan")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=["default", "aligned"])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg["mode_label"] = args.mode
    cfg["mode"] = "live"
    plen, olen = cfg.get("prompt_len", 512), cfg.get("output_len", 16)
    n_req, rates = cfg.get("n_requests", 60), cfg.get("rates_hz", [25, 55])
    reps = cfg.get("repeats", 3)

    run = start_run("i1_eval", cfg, results_root=args.results_root)
    status, err = "ok", None
    try:
        if not metrics_available(args.base_url):
            print("[i1] /metrics unavailable — aborting.", file=sys.stderr)
            return 1

        # --- correctness gate ------------------------------------------
        checks = []
        for seed in cfg.get("correctness_seeds", list(range(12))):
            for pl in cfg.get("correctness_prompt_lens", [128, 520, 1032, 2056]):
                txt = greedy(args.base_url, cfg["model"], pl, cfg.get("correctness_out", 24), seed)
                checks.append({"seed": seed, "prompt_len": pl,
                               "sha": hashlib.sha256(txt.encode()).hexdigest()[:16],
                               "n_chars": len(txt)})
        save_table(run, "correctness", checks)
        print(f"[i1] correctness: {len(checks)} greedy completions hashed "
              f"(prompt lens {cfg.get('correctness_prompt_lens', [])} chosen to straddle buckets)")

        # --- performance ------------------------------------------------
        rows = []
        for rate in rates:
            for rep in range(reps):
                r = run_rate(args.base_url, cfg["model"], rate, n_req, plen, olen, 900 + rep)
                rows.append({"mode": args.mode, "repeat_idx": rep, **r})
                print(f"[i1] {args.mode:<8} {rate:>3} req/s rep{rep}  "
                      f"TPU {r['tpu_ms_per_req']:6.2f} ms/req  "
                      f"p50 {r['p50_latency_ms']:7.1f}  p99 {r['p99_latency_ms']:7.1f}  "
                      f"thru {r['throughput_req_s']:5.1f} req/s")
        save_table(run, "perf", rows)
        print(f"[i1] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ControlledVarError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        sys.exit(2)
