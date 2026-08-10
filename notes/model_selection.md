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

### The hard constraint discovered 2026-08-10: the JAX registry is short

`tpu-inference` 0.25.0 implements these architectures natively
(`tpu_inference/models/jax/`, cross-checked against the loader's registry):

```
LlamaForCausalLM   Qwen3ForCausalLM   Qwen3MoeForCausalLM   Gemma4ForCausalLM
GptOssForCausalLM  DeepseekV3ForCausalLM   Llama4ForCausalLM   Eagle3LlamaForCausalLM
Qwen2ForCausalLM (BROKEN, see below)      DFlashForCausalLM
```

That is the real menu. It explains earlier results retroactively: SmolLM2 and
granite are both `LlamaForCausalLM`, which is why SmolLM2 loaded at all, and
`allenai/OLMo-2-*` is `Olmo2ForCausalLM` — **not on the list**, so the OLMo-2
plan would have failed at engine start regardless of its context length.

**`Qwen2ForCausalLM` is unservable on this version.** `models/jax/qwen2.py:401`
reads `model_config.hf_config.text_config.hidden_size`, but `text_config` exists
only on *multimodal* Qwen2 configs. Every text-only Qwen2 model — Qwen1.5 at any
size, Qwen2-7B — raises `AttributeError: 'Qwen2Config' object has no attribute
'text_config'` before it ever touches the TPU. This is a bug in the stack, not
in the configuration. It cost one server boot to find.

### The control that survives: `princeton-nlp/Sheared-LLaMA-1.3B`

`configs/e01_marginal_cost_sheared.json`. `LlamaForCausalLM`, head_dim **128**,
full MHA, 16 heads / 16 kv, intermediate 5504 — all divisible by 4.

It is a **better** control than the Qwen1.5 plan, because the comparison it
anchors changes exactly one thing:

| | architecture | head_dim | GQA | flatness @ 4096 |
|---|---|---|---|---|
| SmolLM2-1.7B-Instruct | Llama | **64** | MHA | 0.54 |
| **Sheared-LLaMA-1.3B** | Llama | **128** | MHA | ? |
| Qwen3-4B | Qwen3 | 128 | 4:1 | 0.81 |

Against SmolLM2 the only difference is head_dim, within one architecture at a
comparable size — a true single-variable contrast, which Qwen1.5 could not have
given (it would have changed the GQA ratio while also changing family and layer
count). Against Qwen3-4B it isolates the GQA ratio, with family as a residual
confound that must be stated.

| outcome | reads |
|---|---|
| ≈ 0.81, like Qwen3 | **head_dim drives the gradient**; GQA does not |
| ≈ 0.54, like SmolLM2 | **head_dim does not**; the cause is GQA or family |
| in between | both contribute; report the decomposition |

Every outcome is informative, which is the property a control should have, and
the readings were written down before the run.

**Cost of the 4096-position limit.** `max_position_embeddings` is 4096, so
bucket 4096 at occupancy 1.0 would need 4097 tokens of context and be rejected.
The top fraction is therefore 0.99 (4055 tokens), which still maps to bucket
4096. Flatness is computed from actual lengths, so this is accounted for rather
than assumed away — but it is a real difference from the SmolLM2 and Qwen3 runs
and travels with the number.

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
