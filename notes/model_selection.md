# Model selection for R3 (a second attention shape)

`notes/solidity.md` R3 requires any structural finding — the staircase, and
especially the large-bucket flatness gradient — to appear in more than one
attention configuration, or be explicitly scoped to the one it was measured on.

## The constraint nobody mentions: the tested list is attention-homogeneous

Fetched `config.json` for every plausible candidate, 2026-08-09:

| Model | heads | kv | GQA | head_dim | sliding window | gated |
|---|---|---|---|---|---|---|
| Qwen3-4B *(current)* | 32 | 8 | 4:1 | **128** | – | no |
| Qwen3-8B | 32 | 8 | 4:1 | **128** | – | no |
| Qwen3-0.6B | 16 | 8 | 2:1 | **128** | – | no |
| Qwen3-30B-A3B | 32 | 4 | 8:1 | **128** | – | no |
| Mistral-7B-v0.3 | 32 | 8 | 4:1 | **128** | – | no |
| Llama-3.1-8B | 32 | 8 | 4:1 | **128** | – | **yes** |
| Phi-4-mini | 24 | 8 | 3:1 | **128** | 262144 (inert at 8k) | no |
| **granite-3.1-2b** | 32 | 8 | **4:1** | **64** | – | no |
| **SmolLM2-1.7B** | 32 | 32 | **1:1 (MHA)** | **64** | – | no |
| OLMo-2-7B | 32 | 32 | 1:1 (MHA) | 128 | – | no |

**Every model on `tpu-inference`'s tested list is head_dim=128 with GQA.**
Replicating inside that list would vary parameter count and nothing structural,
which is not what R3 asks for. `head_dim` is the axis that matters most here:
RPA's tiling is organised around it, and RPA v2 supported *only* 128 — so if the
large-bucket gradient is a tiling artifact, head_dim is where it should move.

## Choice

**Superseded 2026-08-10 — see "What actually happened" below.**

**Primary: `ibm-granite/granite-3.1-2b-instruct`** — ungated, small (fast to
load and compile), and a **single-variable** change from Qwen3-4B: head_dim 64
instead of 128, GQA ratio held at 4:1. That isolates the axis of interest.

**Optional third: `HuggingFaceTB/SmolLM2-1.7B-Instruct`** — head_dim 64 *and*
full MHA. A stronger contrast, but confounded across two factors, so it is a
follow-up rather than the control.

## What actually happened, and the control that replaces it

granite-3.1-2b **failed to load at TP=4** (`IndivisibleError`), so the planned
single-variable run never happened. Session 3 fell back to the confounded
option, and it produced a real but ambiguous result:

| Model | head_dim | GQA | flatness @ 4096 |
|---|---|---|---|
| Qwen3-4B | 128 | 4:1 | **0.81** |
| SmolLM2-1.7B | **64** | **1:1** | **0.54** |

The gradient moved, which is the R3 finding — the staircase is
architecture-dependent, not universal. But **two variables moved together**, so
this cannot say which one caused it. That matters for the paper's scope: if
`head_dim` drives it, the effect tracks RPA's tiling and generalises by tile
shape; if the GQA ratio drives it, it tracks KV width and generalises by
attention *type*. Those are different claims about who this work applies to.

**The decomposing control: `Qwen/Qwen1.5-4B-Chat`** (`configs/e01_marginal_cost_qwen15.json`).
Holds head_dim at 128 and moves only the GQA ratio to 1:1 MHA, in the same
family at the same parameter count.

| | head_dim | GQA | reads |
|---|---|---|---|
| result ≈ 0.81 (like Qwen3) | 128 | 1:1 | **head_dim drives it** — GQA is irrelevant |
| result ≈ 0.54 (like SmolLM2) | 128 | 1:1 | **GQA drives it** — head_dim is irrelevant |
| result in between | 128 | 1:1 | both contribute; report the decomposition |

Every outcome is informative, which is the property a control should have.

**Why not `allenai/OLMo-2-1124-7B-Instruct`**, the previously identified
candidate: it is also head_dim 128 + MHA and would work in principle, but
`max_position_embeddings` is **4096**. Bucket 4096 at occupancy 1.0 needs 4096
prompt tokens plus an output token — 4097 of context — so the single most
informative cell, the one where Qwen3 and SmolLM2 diverge most, would have had
to be shrunk to fit. Qwen1.5-4B has 32768 positions and runs the *identical*
fractions as every other e01 run, which keeps the comparison clean.

Not perfectly single-variable: Qwen1.5-4B has 40 layers to Qwen3-4B's 36, and
Qwen3 adds QK-norm. Layer count scales cost but should not change the *shape* of
the occupancy curve, which is what flatness measures. Stated as a limitation
rather than hidden.

TP=4 divisibility checked in advance this time, since that is what killed
granite: 20 attention heads / 4 = 5, intermediate 6912 / 4 = 1728.

**Fallback: `gemma-3-4b-it`** once the gated HF access lands. Gemma-3 is the
most different thing available (head_dim 256, interleaved sliding-window
attention) *and* is on the tested list — the best of both, but it needs the
approval that session 1 was designed to route around.

## The risk, stated plainly

granite and SmolLM2 are **not on `tpu-inference`'s tested list**. They may fail
to load, fall back off the RPA kernel, or hit an unimplemented path. If that
happens the finding is "we could not test R3 on this stack", which is itself
worth knowing and worth exactly one session to discover — but the run must be
treated as exploratory, and Gemma-3 remains the principled answer once access
exists.
