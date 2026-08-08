"""
The bucket ladder: the paper's independent variable.

vLLM TPU pads every request up to one of N precompiled shapes, using one of two
schemes (docs.vllm.ai TPU configuration page):

  exponential (default)  pad to the nearest power of two
  bucket padding         VLLM_TPU_BUCKET_PADDING_GAP=g -> buckets start at 16,
                         end at max_model_len, increment by g

This module reproduces both schemes so ladders can be enumerated, compared and
optimised with no TPU present. The on-hardware check in e00_smoke_test.py
verifies that the real server's warmup log matches what this module predicts —
if they ever disagree, the model here is wrong and everything downstream is too.

Reference: notes/plan_v4.md, notes/prior_art.md section 1.
"""

from __future__ import annotations

from typing import Iterable, Sequence

MIN_BUCKET = 16


def exponential_ladder(max_model_len: int, min_bucket: int = MIN_BUCKET) -> list[int]:
    """vLLM's default TPU padding: nearest power of two.

    Worst case this wastes just under 50% of the padded shape, which is why the
    default ladder plausibly leaves real headroom (plan_v4.md, oracle gap).
    """
    if max_model_len < 1:
        raise ValueError(f"max_model_len must be >= 1, got {max_model_len}")
    buckets: list[int] = []
    b = min_bucket
    while b < max_model_len:
        buckets.append(b)
        b *= 2
    buckets.append(max_model_len)
    return buckets


def gap_ladder(max_model_len: int, padding_gap: int, min_bucket: int = MIN_BUCKET) -> list[int]:
    """VLLM_TPU_BUCKET_PADDING_GAP: linear buckets from 16 to max_model_len.

    Matches the documented example exactly: max_model_len=512, gap=64 gives
    [16, 32, 64, 128, 192, 256, 320, 384, 448, 512]. Note vLLM ramps
    exponentially up to the first multiple of the gap, then goes linear.
    """
    if padding_gap < 1:
        raise ValueError(f"padding_gap must be >= 1, got {padding_gap}")
    if max_model_len < 1:
        raise ValueError(f"max_model_len must be >= 1, got {max_model_len}")

    buckets: list[int] = []
    b = min_bucket
    while b < min(padding_gap, max_model_len):
        buckets.append(b)
        b *= 2

    b = padding_gap
    while b < max_model_len:
        buckets.append(b)
        b += padding_gap

    buckets.append(max_model_len)
    # dedupe, preserve order
    seen: set[int] = set()
    out: list[int] = []
    for x in buckets:
        if x not in seen and x <= max_model_len:
            seen.add(x)
            out.append(x)
    return out


def build_ladder(max_model_len: int, padding_gap: int | str | None = None) -> list[int]:
    """Dispatch on the VLLM_TPU_BUCKET_PADDING_GAP value.

    "" / None -> exponential (vLLM's default). An int -> gap ladder.
    """
    if padding_gap in (None, "", "none"):
        return exponential_ladder(max_model_len)
    return gap_ladder(max_model_len, int(padding_gap))


def bucket_for(length: int, ladder: Sequence[int]) -> int:
    """Smallest bucket that fits `length` — what the request is padded up to."""
    for b in ladder:
        if length <= b:
            return b
    raise ValueError(
        f"length {length} exceeds the largest bucket {ladder[-1]}; "
        "such a request cannot be served by this ladder"
    )


def padding_tokens(length: int, ladder: Sequence[int]) -> int:
    """Wasted token slots for one request. NOTE: token counts are NOT the
    paper's cost metric — cost is C(B) - C(L) on a measured superlinear curve
    (plan_v4.md, and the critique of BucketServe's token-based objective).
    This function exists for baselines and sanity checks only."""
    return bucket_for(length, ladder) - length


def padding_fraction(lengths: Iterable[int], ladder: Sequence[int]) -> float:
    """Expected padding fraction under a ladder — the honest replacement for
    v2's invented 'raggedness index' (see design_review_v2.md)."""
    total_len = 0
    total_pad = 0
    for L in lengths:
        total_len += L
        total_pad += padding_tokens(L, ladder)
    if total_len == 0:
        return 0.0
    return total_pad / (total_len + total_pad)


def chunked_prefill_shapes(length: int, chunk: int) -> list[int]:
    """Prefill shapes actually submitted when chunked prefill is enabled.

    This is the finding that killed L1 (notes/kill_condition.md): a full chunk
    is submitted at exactly `chunk` tokens, and only the final partial chunk is
    padded against the ladder. With chunk=8192 (a power of two) a full chunk
    pads to nothing at all.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    shapes = [chunk] * (length // chunk)
    rem = length % chunk
    if rem:
        shapes.append(rem)
    return shapes or [0]


def prefill_padding_tokens(
    length: int, ladder: Sequence[int], chunk: int | None
) -> int:
    """Prefill padding with and without chunked prefill.

    chunk=None reproduces the pre-gate assumption (whole prompt -> one bucket).
    Passing a chunk size reproduces what vLLM V1 actually does by default.
    """
    if chunk is None:
        return padding_tokens(length, ladder)
    return sum(padding_tokens(s, ladder) for s in chunked_prefill_shapes(length, chunk))
