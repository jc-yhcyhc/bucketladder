"""Tests for scripts/ladder.py — the paper's independent variable.

The load-bearing test is test_documented_vllm_example: it pins our model of the
ladder to the exact example in vLLM's TPU docs. If that ever fails, our model
of the independent variable is wrong and everything downstream is measuring
something we do not understand.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ladder import (  # noqa: E402
    bucket_for,
    build_ladder,
    chunked_prefill_shapes,
    exponential_ladder,
    gap_ladder,
    padding_fraction,
    padding_tokens,
    prefill_padding_tokens,
)


def test_documented_vllm_example():
    """docs.vllm.ai TPU page: max_model_len=512, padding_gap=64 gives exactly
    [16, 32, 64, 128, 192, 256, 320, 384, 448, 512]."""
    assert gap_ladder(512, 64) == [16, 32, 64, 128, 192, 256, 320, 384, 448, 512]


def test_exponential_is_powers_of_two():
    assert exponential_ladder(2048) == [16, 32, 64, 128, 256, 512, 1024, 2048]


def test_build_ladder_dispatch():
    assert build_ladder(512, "") == exponential_ladder(512)
    assert build_ladder(512, None) == exponential_ladder(512)
    assert build_ladder(512, 64) == gap_ladder(512, 64)
    assert build_ladder(512, "64") == gap_ladder(512, 64)


def test_ladder_is_sorted_unique_and_bounded():
    for gap in ("", 64, 128, 512):
        lad = build_ladder(4096, gap)
        assert lad == sorted(lad), gap
        assert len(lad) == len(set(lad)), gap
        assert lad[-1] == 4096, gap
        assert lad[0] == 16, gap


def test_gap_ladder_finer_than_exponential():
    """A smaller gap means more buckets — the compile-budget tradeoff both
    BucketServe and LAPS name but neither optimises."""
    assert len(build_ladder(8192, 128)) > len(build_ladder(8192, 512))
    assert len(build_ladder(8192, 512)) > len(build_ladder(8192, ""))


def test_bucket_for_picks_smallest_fitting():
    lad = [16, 32, 64, 128]
    assert bucket_for(1, lad) == 16
    assert bucket_for(16, lad) == 16
    assert bucket_for(17, lad) == 32
    assert bucket_for(128, lad) == 128


def test_bucket_for_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds the largest bucket"):
        bucket_for(129, [16, 32, 64, 128])


def test_padding_tokens_worst_case_under_half():
    """Power-of-two padding wastes just under 50% of the padded shape."""
    lad = exponential_ladder(4096)
    worst = max(padding_tokens(L, lad) / bucket_for(L, lad) for L in range(17, 4097))
    assert worst < 0.5


def test_padding_fraction_zero_on_exact_hits():
    lad = [16, 32, 64]
    assert padding_fraction([16, 32, 64], lad) == 0.0


def test_padding_fraction_between_zero_and_one():
    lad = exponential_ladder(2048)
    frac = padding_fraction([17, 100, 513, 1025], lad)
    assert 0.0 < frac < 1.0


# --- chunked prefill: the finding that killed L1 -------------------------

def test_chunked_shapes_split_correctly():
    assert chunked_prefill_shapes(10_000, 8192) == [8192, 1808]
    assert chunked_prefill_shapes(8192, 8192) == [8192]
    assert chunked_prefill_shapes(100, 8192) == [100]


def test_full_chunk_of_power_of_two_pads_to_nothing():
    lad = exponential_ladder(8192)
    assert padding_tokens(8192, lad) == 0


def test_chunked_prefill_collapses_padding():
    """The kill_condition.md arithmetic, pinned as a test.

    10k-token prompt: ~64% waste unchunked, ~2.4% chunked at 8192.
    """
    lad = exponential_ladder(16384)
    unchunked = prefill_padding_tokens(10_000, lad, chunk=None)
    chunked = prefill_padding_tokens(10_000, lad, chunk=8192)

    assert unchunked == 16384 - 10_000        # 6384
    assert chunked == 2048 - 1808             # 240
    assert unchunked / 10_000 == pytest.approx(0.638, abs=0.01)
    assert chunked / 10_000 == pytest.approx(0.024, abs=0.001)
    assert chunked < unchunked / 20


@pytest.mark.parametrize("length", [1, 17, 500, 8192, 8193, 50_000])
def test_chunked_never_worse_than_unchunked(length):
    lad = exponential_ladder(65536)
    assert prefill_padding_tokens(length, lad, 8192) <= prefill_padding_tokens(length, lad, None)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        gap_ladder(512, 0)
    with pytest.raises(ValueError):
        exponential_ladder(0)
    with pytest.raises(ValueError):
        chunked_prefill_shapes(100, 0)
