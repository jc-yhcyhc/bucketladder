"""
The model preflight is pinned against the boots that actually failed.

Each case below is a configuration this project really tried on a v5e-4. Two of
them cost a server boot each (~[redacted] and ~12 minutes) before the rule existed;
a third, OLMo-2, was caught by the checker before it was ever booted. If a
future dependency bump changes these answers, that is a real signal and this
test should fail loudly rather than be relaxed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_model import evaluate  # noqa: E402

ST = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
BIN = ["pytorch_model.bin"]


def cfg(arch, hidden=4096, heads=32, kv=None, inter=11008, maxpos=8192, layers=32, head_dim=None):
    c = {"architectures": [arch], "hidden_size": hidden, "num_attention_heads": heads,
         "num_key_value_heads": heads if kv is None else kv, "intermediate_size": inter,
         "max_position_embeddings": maxpos, "num_hidden_layers": layers}
    if head_dim:
        c["head_dim"] = head_dim
    return c


def test_llama_with_safetensors_passes():
    """NousResearch/Llama-2-7b-chat-hf — the R3 control that actually served."""
    r = evaluate("llama2", cfg("LlamaForCausalLM", maxpos=4096), ST, tp=4, max_model_len=4096)
    assert r["ok"], r["fatal"]
    assert r["head_dim"] == 128 and r["attn"] == "MHA"


def test_qwen2_architecture_is_rejected_even_though_it_is_registered():
    """Qwen1.5-4B-Chat. Present in the JAX registry, and fatal anyway — the
    reason a bare 'is the architecture supported' check is not enough."""
    r = evaluate("qwen15", cfg("Qwen2ForCausalLM", hidden=2560, heads=20, inter=6912, maxpos=32768),
                 ST, tp=4, max_model_len=8192)
    assert not r["ok"]
    assert any("BROKEN" in f and "text_config" in f for f in r["fatal"])


def test_missing_safetensors_is_fatal():
    """Sheared-LLaMA-1.3B. Right architecture, right attention geometry, and
    the loader fails only AFTER downloading the weights."""
    r = evaluate("sheared", cfg("LlamaForCausalLM", hidden=2048, heads=16, inter=5504, maxpos=4096),
                 BIN, tp=4, max_model_len=4096)
    assert not r["ok"]
    assert any("safetensors" in f for f in r["fatal"])


def test_unregistered_architecture_is_fatal():
    """OLMo-2-7B. Caught before it was ever booted."""
    r = evaluate("olmo2", cfg("Olmo2ForCausalLM", maxpos=4096), ST, tp=4, max_model_len=4096)
    assert not r["ok"]
    assert any("not in tpu-inference's JAX registry" in f for f in r["fatal"])


def test_indivisible_heads_are_fatal():
    """granite-3.1-2b's failure mode, IndivisibleError, reduced to arithmetic."""
    r = evaluate("granite", cfg("LlamaForCausalLM", hidden=2048, heads=32, kv=6, inter=8192),
                 ST, tp=4, max_model_len=4096)
    assert not r["ok"]
    assert any("num_key_value_heads=6" in f and "TP=4" in f for f in r["fatal"])


def test_short_context_warns_but_does_not_block():
    """A 4096-position model is usable; it just needs max_model_len lowered and
    the top occupancy cell shortened. That is a caveat, not a blocker."""
    r = evaluate("llama2", cfg("LlamaForCausalLM", maxpos=4096), ST, tp=4, max_model_len=8192)
    assert r["ok"]
    assert any("max_position_embeddings=4096" in w for w in r["warn"])


@pytest.mark.parametrize("heads,kv,expect", [(32, 32, "MHA"), (32, 8, "GQA"), (16, 8, "GQA")])
def test_attention_type_reported(heads, kv, expect):
    """R3 needs models chosen on the attention axis, so the checker has to
    report head_dim and the GQA ratio, not merely pass or fail."""
    r = evaluate("m", cfg("LlamaForCausalLM", heads=heads, kv=kv), ST, tp=4, max_model_len=4096)
    assert r["attn"] == expect
