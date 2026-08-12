#!/usr/bin/env python3
"""
G1 — the GPU control. Is the batch dimension a PAID quantity on GPU?

§8 of the paper asserts an architectural distinction and does not measure it:
CUDA-graph capture makes the batch dimension a paid quantity on GPU in a way a
compiled TPU step does not, because a graph is captured per shape and replayed,
so an unseen batch size costs a capture rather than riding inside a compiled
step. That single unmeasured caveat is what caps the paper at "TPU note" rather
than a statement about the class of optimisations BucketServe and LAPS belong to.

THE SYMMETRY THAT MAKES THIS A CONTROL, NOT A NEW EXPERIMENT. vLLM's CUDA path
captures graphs for a fixed set of batch sizes and **pads a batch up to the next
captured size** — which is structurally the same thing as the TPU request ladder
in §4.1. So the identical question can be asked of both stacks with the identical
instrument:

    take a batch size that sits just above a ladder/capture entry,
    and ask whether it costs what the entry below costs, or what the entry
    above costs.

On TPU the answer is "the entry below" — padding is free, because Ragged Paged
Attention does no work for padded request slots (§4.5, established by cutting the
compiled slot count 32x for a -0.9% change). If the GPU answer is "the entry
above", the paper has a two-column table instead of an assertion.

THREE MEASUREMENTS, in decreasing order of how much they carry:

  A. PADDING PAID?  Decode at n = 8, 9, 16 with graphs on. vLLM pads 9 up to the
     next captured size. If cost(9) ~= cost(16), the batch dimension is paid; if
     cost(9) ~= cost(8), it is not, and the architectural claim in §8 is wrong.

  B. THE EAGER CONTROL. The same sweep with --enforce-eager, which captures no
     graphs at all. This separates "padding to a captured shape costs" from "this
     batch size is simply more expensive" -- exactly the control §4.1 runs on TPU
     by toggling ATTN_BUCKETIZED_NUM_REQS. Without it, arm A alone cannot
     distinguish the two.

  C. WHAT A CAPTURE COSTS. Startup time against the number of captured sizes.
     This is the quantity BucketServe and LAPS are managing when they say "the
     number of graphs must be limited", and on TPU the analogous quantity is XLA
     warmup, which we already know is 5-30 min for the first bucket.

WHAT EACH OUTCOME COSTS, stated carefully, because an earlier version of this
docstring overstated it. If A shows cost(9) ~= cost(8) with graphs on, GPU batch
padding is free too. That withdraws exactly ONE comparative sentence in §8 -- that
capture makes the batch dimension paid in a way it is not on TPU. It does not
touch §4.1 (the printed ladder is not the executed one), §4.5 (RPA does no work
for padded slots, established by cutting compiled slots 32x), §4.3, or §4.2: each
rests on its own TPU measurement and no GPU result bears on them.

That outcome would in fact make the paper STRONGER. A padding premise false on
both architectures is a claim about the optimisation family BucketServe and LAPS
belong to, rather than about one accelerator -- more reach than §8 currently
claims, at the cost of an architectural explanation for a difference that turned
out not to exist. Both outcomes are publishable; only one of them is the one we
guessed.

Usage:
  python scripts/g1_cuda_graph.py --arm graphs --out /tmp/g1_graphs.json
  python scripts/g1_cuda_graph.py --arm eager  --out /tmp/g1_eager.json
  python scripts/g1_cuda_graph.py --report /tmp/g1_graphs.json /tmp/g1_eager.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time


def measure(arm: str, model: str, ns: list[int], prompt_len: int,
            output_len: int, repeats: int) -> dict:
    """Decode cost per step at each batch size, in-process (no HTTP server)."""
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    t0 = time.perf_counter()
    llm = LLM(model=model, enforce_eager=(arm == "eager"),
              max_model_len=2048, gpu_memory_utilization=0.85,
              enable_prefix_caching=False)
    startup_s = time.perf_counter() - t0

    sp = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
    prompt = [1234] * prompt_len
    out: dict[str, object] = {"arm": arm, "model": model, "startup_s": startup_s,
                              "prompt_len": prompt_len, "output_len": output_len,
                              "cells": []}
    for n in ns:
        reqs = [{"prompt_token_ids": prompt}] * n
        llm.generate(reqs, sp)                      # warm this shape
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            llm.generate(reqs, sp)
            ts.append((time.perf_counter() - t) * 1000.0)
        med = statistics.median(ts)
        out["cells"].append({"n": n, "wall_ms": med, "ms_per_step": med / output_len,
                             "ms_per_step_per_seq": med / output_len / n,
                             "samples": ts})
        print(f"[g1:{arm}] n={n:<4} {med:9.1f} ms  {med / output_len:7.3f} ms/step",
              flush=True)
    return out


def report(paths: list[str]) -> int:
    """The comparison the paper needs, with the TPU number beside it."""
    arms = {}
    for p in paths:
        d = json.loads(pathlib.Path(p).read_text())
        arms[d["arm"]] = d
    if "graphs" not in arms:
        print("[g1] need at least the 'graphs' arm", file=sys.stderr)
        return 1

    print(f"\n[g1] {'arm':>8} {'startup s':>10} " +
          " ".join(f"n={c['n']:<8}" for c in arms['graphs']['cells']))
    for a, d in arms.items():
        cells = " ".join(f"{c['ms_per_step']:9.3f} " for c in d["cells"])
        print(f"[g1] {a:>8} {d['startup_s']:>10.1f} {cells}")

    # The load-bearing statistic: where n=9 sits between n=8 and n=16.
    # 0% = costs what 8 costs (padding free); 100% = costs what 16 costs (paid).
    print(f"\n[g1] {'arm':>8} {'position of n=9 between n=8 and n=16':>40}")
    pos = {}
    for a, d in arms.items():
        c = {x["n"]: x["ms_per_step"] for x in d["cells"]}
        if not {8, 9, 16} <= set(c):
            continue
        if c[16] == c[8]:
            continue
        p = (c[9] - c[8]) / (c[16] - c[8]) * 100
        pos[a] = p
        print(f"[g1] {a:>8} {p:>39.0f}%")

    print("\n[g1] --- verdict ---")
    if "graphs" in pos:
        g = pos["graphs"]
        e = pos.get("eager")
        if g > 50:
            print(f"[g1] With graphs, n=9 sits at {g:.0f}% of the way to n=16: the batch "
                  f"dimension IS a paid quantity on GPU.")
            if e is not None and e < 50:
                print(f"[g1] With --enforce-eager it sits at {e:.0f}%, so the cost is the "
                      f"CAPTURE PADDING and not the batch size itself. This is the control "
                      f"that makes the comparison sound.")
            elif e is not None:
                print(f"[g1] But eager also sits at {e:.0f}%, so the cost is NOT specific to "
                      f"graph capture and the architectural claim needs rethinking.")
            print("[g1] TPU comparison: the same statistic is -5%/-3%/-3% (§4.1), i.e. "
                  "padding to the next request-ladder entry is free there.")
        else:
            print(f"[g1] With graphs, n=9 sits at {g:.0f}% of the way to n=16 — GPU batch "
                  f"padding is ALSO cheap.")
            print("[g1] What this withdraws is ONE SENTENCE in §8: that capture makes the "
                  "batch dimension paid *in a way it is not on TPU*. That comparative claim "
                  "dies; nothing else does.")
            print("[g1] What survives, because each rests on its own TPU measurement: §4.1's "
                  "printed-vs-executed ladder, §4.5's ragged-skipping mechanism, §4.3's "
                  "token-dimension paid share, §4.2's LENS ablation.")
            print("[g1] And this outcome makes the paper STRONGER, not weaker: if the padding "
                  "premise is false on GPU too, the negative result is about the optimisation "
                  "family rather than about one accelerator, which is more reach than §8 "
                  "currently claims.")
    if "eager" in arms and "graphs" in arms:
        d = arms["graphs"]["startup_s"] - arms["eager"]["startup_s"]
        print(f"[g1] Graph capture costs {d:+.1f} s of startup — the quantity BucketServe "
              f"and LAPS manage when they limit the number of graphs.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["graphs", "eager"])
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--ns", default="8,9,16")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--report", nargs="+")
    args = ap.parse_args(argv)

    if args.report:
        return report(args.report)
    if not args.arm or not args.out:
        ap.error("--arm and --out are required unless --report is given")
    d = measure(args.arm, args.model, [int(x) for x in args.ns.split(",")],
                args.prompt_len, args.output_len, args.repeats)
    args.out.write_text(json.dumps(d, indent=2))
    print(f"[g1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
