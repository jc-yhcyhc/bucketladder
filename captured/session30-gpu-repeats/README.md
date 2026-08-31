# Session 30 — GPU control, 20 repeats/cell

MLSys review: "An L4 is ~$0.30/hr; run 20 repeats per cell and report the
interval." Ran on a fresh L4 GCE instance (g2-standard-4 + nvidia-l4,
us-central1-a), vLLM 0.25.0, TinyLlama-1.1B-Chat-v1.0 -- same model and
instrument as the original single-measurement run in §4.9 (then §8).

  scripts/g1_cuda_graph.py --arm graphs --repeats 20 --out g1_graphs_r20.json
  scripts/g1_cuda_graph.py --arm eager  --repeats 20 --out g1_eager_r20.json
  scripts/g1_analyze_repeats.py g1_graphs_r20.json g1_eager_r20.json

Result (bootstrap 95% CI, median-based, 10000 resamples):

  graphs: position of n=9 between n=8 and n=16 = 16.0% [15.0%, 17.1%]
  eager:  position of n=9 between n=8 and n=16 =  0.7% [-9.6%, 11.1%]

Both exclude "fully paid" (100%). Only eager's interval includes "fully
free" (0%); graphs' does not -- at this power, graphs shows a real, small,
resolvable batch-padding cost, and eager is statistically indistinguishable
from zero. The original n=1 point estimates (17%, 1%) undersold graphs'
interval width and correctly called eager free, though without evidence
that could rule out graphs also being free.

VM billed ~65 minutes (~$0.93) before teardown. Setup needed three fixes
not in any existing image: python3-venv, python3-dev (Python.h), and
build-essential (cc1plus) -- documented in infra/create_gpu.sh's own
history rather than repeated here.
