#!/usr/bin/env python3
"""
e02 — what does stock tpu-inference already do when a bucket saturates?

The gate that decides whether the admission contribution exists. If the shipped
scheduler already behaves like our `hybrid` policy, there is nothing to propose
and the paper narrows to ladder design alone (notes/session_plan.md, "the two
gates are independent").

Session 1 sharpened this. vLLM announces TWO ladders:

    Prepared token paddings:   [16, 32, ..., 8192]      <- sequence axis
    Prepared request paddings: [8, 16, 32, 64, 128, 256] <- BATCH axis

So "the bucket a request needs" has a batch component that is itself quantised,
and `VLLM_TPU_BUCKET_PADDING_GAP` does NOT move it (verified: identical across
both session-1 runs). That makes the request ladder the interesting axis for
admission: with 9 concurrent requests, does the server pad the batch to 16 and
eat the waste, or hold requests back to fill 8 exactly?

REDESIGNED 2026-08-09 after the first version came back inconclusive. That
version swept concurrency and watched client TTFT, which **conflates queueing
with batch padding** — at n>8 some requests are simply waiting, and TTFT cannot
say which. Predictably the largest step landed where no edge was crossed, and
the step that did cross 16->32 came back *faster*.

The fix is not more samples, it is a better observable. vLLM already separates
the two server-side:

    vllm:request_queue_time_seconds     arrival -> first scheduled  (WAITING)
    vllm:request_prefill_time_seconds   prefill phase duration      (COMPUTING)

So the confounded question splits into two clean ones, and the answer is a
2x2 rather than a judgement call:

  prefill UP at edge, queue flat   -> the batch is padded and the server PAYS.
                                      Stock behaves like `promote`.
  queue UP, prefill flat           -> the server WAITS to fill the batch.
                                      Stock behaves like `queue`; promote is ours.
  both up                          -> mixed; report both and quantify.
  both flat                        -> crossing the edge costs nothing, and the
                                      batch axis is not where the paper lives.

Usage:
  python scripts/e02_stock_baseline.py --config configs/e02_stock_baseline.json --mock
  python scripts/e02_stock_baseline.py --config configs/e02_stock_baseline.json \
      --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _client import Sample, complete, complete_mock, summarise  # noqa: E402
from _metrics import MockMetrics, delta, metrics_available, scrape, short  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from ladder import build_ladder  # noqa: E402

# Observed on hardware 2026-08-09; re-read from the log rather than trusting
# this constant once e00 has run in the same session.
DEFAULT_REQUEST_LADDER = [8, 16, 32, 64, 128, 256]

# WARMUP DISCARD. Measured on hardware 2026-08-09: the first request after a
# server start costs ~116 ms against a ~15.4 ms steady state (7.5x), even though
# vLLM has already logged "Application startup complete". Including it puts the
# run-to-run CV at 97%; discarding it gives 1.7%, and discarding 5 gives 0.8%.
# Every cell therefore fires `warmup_discard` unrecorded requests first.



def concurrency_sweep(request_ladder: list[int], around: int = 8, span: int = 2) -> list[int]:
    """Concurrency levels bracketing a request-ladder edge.

    Dense around the edge, because the whole question is whether anything
    happens exactly there.
    """
    lo = max(1, around - span)
    hi = around + span
    pts = set(range(lo, hi + 1))
    pts.add(around)
    nxt = next((b for b in request_ladder if b > around), None)
    if nxt:
        pts.update({nxt - 1, nxt, min(nxt + span, nxt * 2)})
    return sorted(p for p in pts if p >= 1)


def fire_concurrent(n: int, fn) -> list[Sample]:
    """Issue n requests simultaneously; they must arrive together to contend."""
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(lambda i: fn(i), range(n)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-policy", choices=["queue", "promote"], default="promote",
                    help="which stock behaviour to simulate, to prove the analysis discriminates")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--results-root", type=Path, default=None)
    args = ap.parse_args(argv)

    config = load_config(args.config)
    config["mode"] = "mock" if args.mock else "live"
    controlled = config["controlled"]
    ladder = build_ladder(controlled["max_model_len"], controlled["VLLM_TPU_BUCKET_PADDING_GAP"])

    req_ladder = config.get("request_ladder", DEFAULT_REQUEST_LADDER)
    edge = config.get("edge", 8)
    prompt_len = config.get("prompt_len", 512)
    output_len = config.get("output_len", 8)
    repeats = config.get("repeats", 3)
    discard = config.get("warmup_discard", 1)
    levels = config.get("concurrency") or concurrency_sweep(req_ladder, around=edge)

    run = start_run("e02_stock_baseline", config, results_root=args.results_root)
    status, err = "ok", None
    try:
        mock_metrics = MockMetrics() if args.mock else None
        use_metrics = bool(mock_metrics) or metrics_available(args.base_url)
        if not use_metrics:
            print("[e02] WARNING /metrics unavailable — falling back to client TTFT, "
                  "which CANNOT separate queueing from padding. Result will be "
                  "inconclusive by construction.", file=sys.stderr)

        rows: list[dict[str, Any]] = []
        server_rows: list[dict[str, Any]] = []
        for n in levels:
            for rep in range(-discard, repeats):
                before = (mock_metrics.snapshot() if mock_metrics
                          else scrape(args.base_url)) if use_metrics else None
                if args.mock:
                    # promote: batch padded up to the next request bucket, so
                    #          per-request cost jumps at the edge.
                    # queue:   excess requests wait a full batch time.
                    nxt = next((b for b in req_ladder if b >= n), req_ladder[-1])
                    def one(i, n=n, rep=rep, nxt=nxt):
                        s = complete_mock(prompt_len, output_len, ladder=ladder, seed=rep * 100 + i)
                        base = s.ttft_ms
                        if args.mock_policy == "promote":
                            # padded batch -> PREFILL grows, no waiting
                            prefill = base * nxt / max(1, min(n, nxt))
                            queue = base * 0.05
                        else:
                            # server waits for a full batch -> QUEUE grows
                            prefill = base
                            queue = base * (math.ceil(n / req_ladder[0]) - 1) + base * 0.05
                        mock_metrics.record(prefill / 1000.0, queue / 1000.0)
                        s.ttft_ms = prefill + queue
                        return s
                    samples = fire_concurrent(n, one)
                else:
                    samples = fire_concurrent(
                        n, lambda i, rep=rep: complete(
                            args.base_url, config["model"], prompt_len, output_len,
                            seed=rep * 100 + i))
                if use_metrics:
                    after = mock_metrics.snapshot() if mock_metrics else scrape(args.base_url)
                    d = delta(before, after)
                    if rep >= 0:
                        server_rows.append({
                            "concurrency": n, "repeat": rep,
                            "padded_to": next((b for b in req_ladder if b >= n), None),
                            **{f"{short(k)}_ms": v["mean_ms"] for k, v in d.items()},
                            "n_requests_seen": max((v["count"] for v in d.values()), default=0),
                        })
                if rep < 0:
                    continue  # warmup, not recorded
                for i, s in enumerate(samples):
                    rows.append({"concurrency": n, "repeat": rep, "req_index": i,
                                 "at_edge": n == edge, **s.as_row()})
        save_table(run, "samples", rows)

        summary: list[dict[str, Any]] = []
        for n in levels:
            sub = [Sample(**{k: r[k] for k in Sample.__dataclass_fields__})
                   for r in rows if r["concurrency"] == n]
            st = summarise(sub)
            summary.append({"concurrency": n, "padded_to":
                            next((b for b in req_ladder if b >= n), None), **st})
        save_table(run, "summary", summary)

        print(f"[e02] prompt_len={prompt_len} output_len={output_len} "
              f"request_ladder={req_ladder}")
        prev = None
        for s in summary:
            jump = "" if prev is None or math.isnan(prev) else f"  ({(s['median']/prev - 1)*100:+.0f}%)"
            edge_mark = " <- ladder edge" if s["concurrency"] in req_ladder else ""
            print(f"[e02]   n={s['concurrency']:>3} -> batch {s['padded_to']:>3}: "
                  f"TTFT median {s['median']:.1f} ms{jump}{edge_mark}")
            prev = s["median"]

        if server_rows:
            save_table(run, "server_timing", server_rows)
            print("[e02] --- SERVER-SIDE (the measurement that separates the two) ---")
            base_q = base_p = None
            for n in levels:
                sub = [r for r in server_rows if r["concurrency"] == n]
                if not sub:
                    continue
                q = sorted(r.get("queue_ms", float("nan")) for r in sub)[len(sub) // 2]
                pf = sorted(r.get("prefill_ms", float("nan")) for r in sub)[len(sub) // 2]
                if base_q is None:
                    base_q, base_p = q, pf
                print(f"[e02]   n={n:>3} -> batch {sub[0]['padded_to']:>3}: "
                      f"queue {q:7.1f} ms ({q - base_q:+7.1f})   "
                      f"prefill {pf:7.1f} ms ({pf - base_p:+7.1f})")
            lo = [r for r in server_rows if r["concurrency"] == min(levels)]
            hi = [r for r in server_rows if r["concurrency"] == max(levels)]
            if lo and hi:
                dq = (sum(r.get("queue_ms", 0) for r in hi) / len(hi)
                      - sum(r.get("queue_ms", 0) for r in lo) / len(lo))
                dp = (sum(r.get("prefill_ms", 0) for r in hi) / len(hi)
                      - sum(r.get("prefill_ms", 0) for r in lo) / len(lo))
                verdict = ("promote (server pays padding)" if dp > dq else
                           "queue (server waits)" if dq > dp else "ambiguous")
                print(f"[e02] VERDICT: dqueue={dq:+.1f} ms  dprefill={dp:+.1f} ms  -> stock looks like {verdict!r}")
                save_table(run, "stock_verdict", [{"d_queue_ms": dq, "d_prefill_ms": dp,
                                                   "verdict": verdict}])

        # Largest relative step, and whether it sits on a ladder edge.
        # An edge is CROSSED when the padded batch size changes between two
        # adjacent concurrency levels. Deriving it from padded_to is robust;
        # arithmetic on n is not (the step lands at the first n ABOVE a ladder
        # value, which is not always ladder_value + 1 once levels are sparse).
        steps = [
            (summary[i]["median"] / summary[i - 1]["median"] - 1,
             summary[i]["concurrency"],
             summary[i]["padded_to"] != summary[i - 1]["padded_to"])
            for i in range(1, len(summary))
            if summary[i - 1]["median"] and not math.isnan(summary[i - 1]["median"])
        ]
        if steps:
            big, at, on_edge = max(steps, key=lambda t: t[0])
            print(f"[e02] largest step {big*100:+.0f}% at n={at} "
                  f"({'CROSSES' if on_edge else 'does not cross'} a request-ladder edge)")
            edge_steps = [d for d, _, e in steps if e]
            other_steps = [d for d, _, e in steps if not e]
            if edge_steps and other_steps:
                print(f"[e02] median step ACROSS edges = {sorted(edge_steps)[len(edge_steps)//2]*100:+.0f}%"
                      f" vs WITHIN a bucket = {sorted(other_steps)[len(other_steps)//2]*100:+.0f}%")
            print("[e02] compare against the e03 noise floor before believing any step")
            save_table(run, "verdict", [{"largest_step_frac": big, "at_concurrency": at,
                                         "on_request_ladder_edge": on_edge}])
        print(f"[e02] run_id={run.run_id}")
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
