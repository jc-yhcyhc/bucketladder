#!/usr/bin/env python3
"""
M12 — what changes at n=4? An operator breakdown, which is the last thing the
review asked for that we have not measured.

Review Q2: "What is the achieved HBM bandwidth utilisation and MXU/MFU at
n=1, 2, 4, 8 and 16 during decode, and what does an xprof operator breakdown show
changing at n=4? Specifically: does the fraction of step time in attention versus
the projection/MLP matmuls, or the fraction in ICI collectives, shift
discontinuously there? If a single profiling session resolves the paper's one
unexplained result, it seems worth roughly $3."

The bandwidth and MFU half was answered offline by `m9_roofline.py`. This is the
half that genuinely needs a profiler: the roofline knows how many bytes and flops
a step *should* need, and nothing about which kernels the compiler actually
emitted or how long each ran.

THE OPEN QUESTION. Three independent observations converge at n=4 — the
within-bucket slope, the paid-padding fraction, and LENS's failure — and three
drafts have said "we could not identify what changes there." If some category of
device time moves discontinuously at n=4 while its neighbours move smoothly, that
is the answer. If every category scales smoothly, the convergence is not visible
at operator granularity and the paper should stop implying a mechanism exists to
be found.

HOW IT WORKS. `tpu_inference`'s worker exposes `jax.profiler.start_trace` through
vLLM's `/start_profile` and `/stop_profile` endpoints when the server is started
with `VLLM_TORCH_PROFILER_DIR` set (read from tpu_worker.py at 0.25.0, not
guessed). We bracket a fixed decode workload at each batch size, then aggregate
device-side event durations by operator category.

WHY --parse-only EXISTS. Trace parsing is where this kind of script is usually
wrong, and being wrong about it on a running TPU costs $4.80/hr. The trace files
are pulled once; every subsequent iteration on the parser runs offline at $0
against the same bytes. This is the same discipline `capture.sh` prints after
every session, applied to the one experiment whose analysis is genuinely
uncertain in advance.

Usage:
  python scripts/m12_profile.py --config configs/m12_profile.json \\
      --base-url http://localhost:8000 --trace-dir /tmp/prof     # on the VM
  python scripts/m12_profile.py --parse-only captured/*/traces   # offline, $0
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import pathlib
import re
import sys
import time
import urllib.request
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _client import complete  # noqa: E402
from _common import ControlledVarError, finish_run, load_config, save_table, start_run  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

# Operator categories, matched against the emitted kernel names in order. The
# names XLA emits are fusion labels, not framework ops, so these are patterns
# over what actually appears in a TPU trace rather than a tidy taxonomy.
CATEGORIES = [
    ("attention", re.compile(r"ragged|paged|attention|flash|attn", re.I)),
    ("collective", re.compile(r"all-reduce|all_reduce|reduce-scatter|all-gather|"
                              r"collective|ici|cross-replica", re.I)),
    ("matmul", re.compile(r"fusion|dot|matmul|conv|gemm|einsum", re.I)),
    ("copy", re.compile(r"copy|transpose|reshape|bitcast|dynamic-slice|"
                        r"dynamic-update-slice|concatenate", re.I)),
]


def categorise(name: str) -> str:
    for label, pat in CATEGORIES:
        if pat.search(name):
            return label
    return "other"


def post(base_url: str, path: str, timeout: float = 120.0) -> int:
    req = urllib.request.Request(base_url.rstrip("/") + path, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except Exception as exc:  # noqa: BLE001
        print(f"[m12] {path} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0


def parse_trace(path: pathlib.Path) -> list[dict[str, Any]]:
    """Chrome-trace events from one jax.profiler trace file.

    Only complete events ('ph':'X') carry a duration, and only device threads
    matter -- host-side Python tracing would otherwise dominate the totals and
    measure our own client rather than the TPU.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        blob = json.load(fh)
    events = blob.get("traceEvents", blob if isinstance(blob, list) else [])
    # Map pid/tid -> thread name so device lanes can be told from host lanes.
    names: dict[tuple[Any, Any], str] = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") in ("thread_name", "process_name"):
            names[(e.get("pid"), e.get("tid"))] = str(e.get("args", {}).get("name", ""))
    out = []
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        lane = names.get((e.get("pid"), e.get("tid")), "") + " " + \
            names.get((e.get("pid"), None), "")
        if not re.search(r"tpu|device|xla|core", lane, re.I):
            continue
        out.append({"name": str(e.get("name", "")), "dur_us": float(e["dur"]), "lane": lane})
    return out


def summarise(events: list[dict[str, Any]]) -> dict[str, float]:
    by = collections.defaultdict(float)
    for e in events:
        by[categorise(e["name"])] += e["dur_us"]
    return dict(by)


def do_parse(roots: list[str]) -> int:
    """Aggregate every pulled trace, grouped by the batch size in its directory."""
    rows: list[dict[str, Any]] = []
    for root in roots:
        for f in sorted(glob.glob(f"{root}/**/*trace.json*", recursive=True)):
            p = pathlib.Path(f)
            m = re.search(r"\bn(\d+)\b", str(p))
            if not m:
                continue
            n = int(m.group(1))
            ev = parse_trace(p)
            if not ev:
                continue
            cats = summarise(ev)
            total = sum(cats.values())
            rows.append({"n": n, "file": p.name, "events": len(ev),
                         "total_device_us": total,
                         **{f"{k}_pct": 100 * v / total for k, v in cats.items()},
                         **{f"{k}_us": v for k, v in cats.items()}})
    if not rows:
        print("[m12] no parsable traces found. Directories must contain the batch "
              "size as 'n<N>' so each trace can be attributed.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["n"])
    labels = [c[0] for c in CATEGORIES] + ["other"]
    print(f"[m12] {'n':>3} {'events':>8} {'total us':>11} " +
          " ".join(f"{c:>11}" for c in labels))
    for r in rows:
        print(f"[m12] {r['n']:>3} {r['events']:>8} {r['total_device_us']:>11.0f} " +
              " ".join(f"{r.get(c + '_pct', 0.0):>10.1f}%" for c in labels))

    # The discontinuity test: for each category, is the n=2->4 step in its share
    # unlike the steps on either side of it? A mechanism visible at operator
    # granularity should show up as exactly that.
    print("[m12] --- does any category move discontinuously at n=4? ---")
    ns = [r["n"] for r in rows]
    found = []
    for c in labels:
        share = {r["n"]: r.get(c + "_pct", 0.0) for r in rows}
        deltas = {ns[i + 1]: share[ns[i + 1]] - share[ns[i]] for i in range(len(ns) - 1)}
        if 4 not in deltas or len(deltas) < 3:
            continue
        at4 = deltas[4]
        others = [v for k, v in deltas.items() if k != 4]
        typical = max(abs(v) for v in others) if others else 0.0
        flag = abs(at4) > 2.0 and abs(at4) > 2 * typical
        print(f"[m12]   {c:<11} share changes {at4:+6.1f} pp into n=4, "
              f"largest change elsewhere {typical:5.1f} pp{'   <-- DISCONTINUOUS' if flag else ''}")
        if flag:
            found.append((c, at4))
    if found:
        print("[m12] VERDICT: " + ", ".join(f"{c} ({d:+.1f} pp)" for c, d in found) +
              " moves discontinuously at n=4. This is a candidate mechanism for the "
              "convergence §4.3 reports and cannot explain.")
    else:
        print("[m12] VERDICT: every category's share moves smoothly through n=4. The "
              "convergence is NOT visible at operator granularity, and the paper "
              "should say that rather than implying an unfound mechanism exists.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-only", nargs="+", metavar="DIR")
    ap.add_argument("--config", type=pathlib.Path)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--trace-dir", default="/tmp/prof",
                    help="where traces are collected INTO, one subdir per batch size")
    ap.add_argument("--profiler-dir", default="/tmp/vllm_prof",
                    help="VLLM_TORCH_PROFILER_DIR the server was started with; every "
                         "session writes here and files are moved out per batch size")
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    args = ap.parse_args(argv)

    if args.parse_only:
        return do_parse(args.parse_only)
    if not args.config:
        ap.error("--config is required unless --parse-only is given")

    cfg = load_config(args.config)
    try:
        run = start_run("m12_profile", cfg, results_root=args.results_root)
    except ControlledVarError as e:
        print(f"[m12] {e}", file=sys.stderr)
        return 2

    status, err = "ok", None
    try:
        rows: list[dict[str, Any]] = []
        # The server writes every session into ONE directory -- profile_dir is
        # fixed at worker construction from VLLM_TORCH_PROFILER_DIR. Traces are
        # therefore moved out into a per-batch-size directory after each stop,
        # since the parser attributes a trace to a cell by its path and would
        # otherwise pool all five cells into whichever name it matched first.
        pdir = pathlib.Path(args.profiler_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        seen = {str(f) for f in pdir.rglob("*") if f.is_file()}
        for n in cfg["concurrency"]:
            tdir = pathlib.Path(args.trace_dir) / f"n{n}"
            tdir.mkdir(parents=True, exist_ok=True)
            # Warm the shape before tracing, so compilation is not in the trace.
            with ThreadPoolExecutor(max_workers=n) as pool:
                list(pool.map(lambda i: complete(args.base_url, cfg["model"],
                                                 cfg["prompt_len"], cfg["output_len"],
                                                 seed=i), range(n)))
            started = post(args.base_url, "/start_profile")
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:
                list(pool.map(lambda i: complete(args.base_url, cfg["model"],
                                                 cfg["prompt_len"], cfg["output_len"],
                                                 seed=1000 + i), range(n)))
            wall = time.perf_counter() - t0
            stopped = post(args.base_url, "/stop_profile")
            time.sleep(cfg.get("flush_seconds", 5))
            moved = 0
            for f in sorted(pdir.rglob("*")):
                if not f.is_file() or str(f) in seen:
                    continue
                seen.add(str(f))
                dest = tdir / f.name
                if dest.exists():
                    dest = tdir / f"{f.parent.name}_{f.name}"
                f.replace(dest)
                moved += 1
            rows.append({"n": n, "wall_s": wall, "start_status": started,
                         "stop_status": stopped, "trace_dir": str(tdir),
                         "trace_files": moved})
            print(f"[m12] n={n:<3} start={started} stop={stopped} "
                  f"wall={wall:.2f}s traces={moved}")
            if moved == 0:
                print(f"[m12]   WARNING no new trace files appeared under {pdir}",
                      file=sys.stderr)
        save_table(run, "sessions", rows)
        if not any(r["start_status"] == 200 for r in rows):
            print("[m12] no profiling session started. Was the server launched with "
                  "VLLM_TORCH_PROFILER_DIR set?", file=sys.stderr)
        print(f"[m12] traces under {args.trace_dir}; parse offline with --parse-only")
        print(f"[m12] run_id={run.run_id}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, err = "failed", exc
        raise
    finally:
        finish_run(run, status=status, error=err)


if __name__ == "__main__":
    sys.exit(main())
