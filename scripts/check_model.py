#!/usr/bin/env python3
"""
Can this model actually be served? Answers in seconds, from a laptop, for $0.

Session 4 lost two server boots — about 25 minutes of billed v5e-4 — to two
model properties that are both readable from the HuggingFace API before
anything is provisioned:

  1. `Qwen/Qwen1.5-4B-Chat` is `Qwen2ForCausalLM`. tpu-inference 0.25.0's
     `models/jax/qwen2.py` reads `hf_config.text_config.hidden_size`, which
     exists only on multimodal Qwen2 configs, so every text-only Qwen2 model
     dies at engine start with AttributeError.
  2. `princeton-nlp/Sheared-LLaMA-1.3B` has the right architecture and the
     right attention geometry, and ships only `pytorch_model.bin`. The JAX
     loader requires `*.safetensors` and raises "Cannot find any *.safetensors
     files" after the weights have already been downloaded.

Neither failure is visible in `config.json`'s architecture field alone, and
both are fatal. The earlier `granite-3.1-2b` loss (IndivisibleError at TP=4)
is a third instance of the same class.

What this checks, in the order that costs the least to be wrong about:

    architecture   is it in tpu-inference's JAX registry at all?
    weights        are there *.safetensors, or only .bin?
    sharding       do heads / kv-heads / intermediate divide by TP?
    context        does max_position_embeddings fit the intended max_model_len?
    attention      head_dim and GQA ratio, reported so a control model can be
                   chosen on the axis that is actually meant to vary

Usage:
  python scripts/check_model.py NousResearch/Llama-2-7b-chat-hf
  python scripts/check_model.py Qwen/Qwen3-4B --tp 4 --max-model-len 8192
  python scripts/check_model.py A B C          # compare candidates
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

# Read off `tpu_inference/models/jax/` and the loader registry on the VM,
# 2026-08-10, tpu-inference 0.25.0. Kept as data because it is a property of a
# pinned dependency version, not a fact about the world — re-derive it after any
# upgrade with:
#   ls venv/lib/python3.11/site-packages/tpu_inference/models/jax/
JAX_REGISTRY = {
    "LlamaForCausalLM",
    "Llama4ForCausalLM",
    "Eagle3LlamaForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Gemma4ForCausalLM",
    "GptOssForCausalLM",
    "DeepseekV3ForCausalLM",
    "DFlashForCausalLM",
    "Qwen2ForCausalLM",   # present but BROKEN, see BROKEN below
}

# In the registry, resolvable, and still fatal.
BROKEN = {
    "Qwen2ForCausalLM": ("models/jax/qwen2.py reads hf_config.text_config, which exists only on "
                         "multimodal Qwen2 configs; text-only Qwen2 models raise AttributeError "
                         "at engine start"),
}

HF = "https://huggingface.co"


def _get_json(url: str, timeout: float = 20.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "bucketladder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch(model: str) -> tuple[dict[str, Any], list[str]]:
    cfg = _get_json(f"{HF}/{model}/raw/main/config.json")
    info = _get_json(f"{HF}/api/models/{model}")
    files = [s.get("rfilename", "") for s in info.get("siblings", [])]
    return cfg, files


def check(model: str, tp: int, max_model_len: int) -> dict[str, Any]:
    try:
        cfg, files = fetch(model)
    except urllib.error.HTTPError as e:
        return {"model": model, "ok": False, "fatal": [f"HuggingFace returned HTTP {e.code} "
                                                       f"({'gated or private' if e.code in (401, 403) else 'not found'})"]}
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return {"model": model, "ok": False, "fatal": [f"could not reach HuggingFace: {e}"]}
    return evaluate(model, cfg, files, tp, max_model_len)


def evaluate(model: str, cfg: dict[str, Any], files: list[str],
             tp: int, max_model_len: int) -> dict[str, Any]:
    """The decision, separated from the fetch so it is testable offline.

    Every rule here was learned from a specific failed boot; the tests pin them
    against the exact configurations that failed.
    """
    fatal: list[str] = []
    warn: list[str] = []

    arch = (cfg.get("architectures") or ["?"])[0]
    if arch not in JAX_REGISTRY:
        fatal.append(f"architecture {arch} is not in tpu-inference's JAX registry "
                     f"(supported: {', '.join(sorted(JAX_REGISTRY - set(BROKEN)))})")
    elif arch in BROKEN:
        fatal.append(f"architecture {arch} is in the registry but BROKEN: {BROKEN[arch]}")

    if not any(f.endswith(".safetensors") for f in files):
        fatal.append("no *.safetensors — the JAX loader cannot read pytorch_model.bin, and it "
                     "fails only AFTER downloading the weights")

    hidden = cfg.get("hidden_size")
    heads = cfg.get("num_attention_heads")
    kv = cfg.get("num_key_value_heads", heads)
    inter = cfg.get("intermediate_size")
    head_dim = cfg.get("head_dim") or (hidden // heads if hidden and heads else None)

    for name, v in (("num_attention_heads", heads), ("num_key_value_heads", kv),
                    ("intermediate_size", inter)):
        if v is not None and v % tp != 0:
            fatal.append(f"{name}={v} is not divisible by TP={tp} (IndivisibleError at load)")

    maxpos = cfg.get("max_position_embeddings")
    if maxpos and maxpos < max_model_len:
        warn.append(f"max_position_embeddings={maxpos} < intended max_model_len={max_model_len}; "
                    f"serve with --max-model-len {maxpos} and note that a bucket at full "
                    f"occupancy needs one more token of context than its size")

    return {"model": model, "ok": not fatal, "fatal": fatal, "warn": warn,
            "arch": arch, "head_dim": head_dim, "heads": heads, "kv": kv,
            "gqa": f"{heads // kv}:1" if heads and kv else "?",
            "attn": "MHA" if heads == kv else "GQA",
            "maxpos": maxpos, "layers": cfg.get("num_hidden_layers"),
            "safetensors": sum(1 for f in files if f.endswith(".safetensors"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args(argv)

    results = [check(m, args.tp, args.max_model_len) for m in args.models]
    worst = 0
    for r in results:
        mark = "OK  " if r["ok"] else "FAIL"
        print(f"[{mark}] {r['model']}")
        if r.get("arch"):
            print(f"         {r['arch']}  head_dim={r['head_dim']}  "
                  f"{r['attn']} {r['heads']}/{r['kv']} ({r['gqa']})  "
                  f"layers={r['layers']}  maxpos={r['maxpos']}  "
                  f"safetensors={r['safetensors']}")
        for f in r.get("fatal", []):
            print(f"    FATAL  {f}")
            worst = 1
        for w in r.get("warn", []):
            print(f"    warn   {w}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
