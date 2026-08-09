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

**Primary: `ibm-granite/granite-3.1-2b-instruct`** — ungated, small (fast to
load and compile), and a **single-variable** change from Qwen3-4B: head_dim 64
instead of 128, GQA ratio held at 4:1. That isolates the axis of interest.

**Optional third: `HuggingFaceTB/SmolLM2-1.7B-Instruct`** — head_dim 64 *and*
full MHA. A stronger contrast, but confounded across two factors, so it is a
follow-up rather than the control.

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
