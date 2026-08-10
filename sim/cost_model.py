"""
Cost model, fitted to measurements — not assumed.

Every constant here traces to a captured run, and each carries how well it is
supported. That matters because the simulator's conclusions are only as good as
this file, and two of these parameters are much better supported than the rest.

Provenance: captured/session3/results/e02_stock_baseline (v5litepod-4,
Qwen3-4B, tpu_inference 0.25.0, prompt_len=512, 2026-08-09).

    n=6  pad= 8  tokens=3072  prefill= 55.85 ms
    n=7  pad= 8  tokens=3584  prefill= 74.05
    n=8  pad= 8  tokens=4096  prefill= 75.56
    n=9  pad=16  tokens=4608  prefill=137.92   <- crosses the request-ladder edge
    n=10 pad=16  tokens=5120  prefill=139.72
    n=15 pad=16  tokens=7680  prefill=150.76
    n=16 pad=16  tokens=8192  prefill=167.25   <- saturates max_num_batched_tokens
    n=18 pad=32  tokens=9216  prefill=171.86   <- saturates; needs 2 steps

KNOWN CONFOUND: at prompt_len=512, n=16 puts exactly 8192 tokens in flight,
which is `max_num_batched_tokens`. Cells at or above that are excluded from all
fits — they measure the token budget, not the request ladder.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- WELL SUPPORTED -------------------------------------------------------
# The 8->9 discontinuity. Both cells sit well below the token budget and differ
# by one request, so the jump is attributable to the batch padding 8 -> 16.
PROMOTION_COST_MS = 60.21
# Marginal cost of real work, from the pad=16 cells below the budget (3 points).
TOKEN_SLOPE_MS_PER_TOKEN = 4.18e-3

# The ratio that motivates the whole admission question: the 9th request needs
# 2.14 ms of its own work and triggers 60.21 ms of padding.
PROMOTION_TO_WORK_RATIO = 28.1

# --- WEAKLY SUPPORTED — treat as uncertain, sweep in sensitivity ----------
# Within pad=8 the implied slope is ~19 ms/1k tokens, 4.6x the pad=16 slope.
# Three points, one of which (n=7) looks anomalous. Not used for headline
# numbers; recorded so nobody silently assumes a single global slope.
PAD8_SLOPE_MS_PER_TOKEN = 1.92e-2
SLOPE_UNCERTAINTY = (4.18e-3, 1.92e-2)

# Batch-size ladder vLLM reported on hardware (identical across both session-1
# runs and unaffected by VLLM_TPU_BUCKET_PADDING_GAP).
REQUEST_LADDER = (8, 16, 32, 64, 128, 256)
MAX_NUM_BATCHED_TOKENS = 8192


def padded_batch(n: int, ladder: tuple[int, ...] = REQUEST_LADDER) -> int:
    """The batch size actually compiled for, given n admitted requests."""
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}")
    for b in ladder:
        if n <= b:
            return b
    return ladder[-1]


@dataclass(frozen=True)
class CostModel:
    """Prefill cost of one scheduling step.

    cost = base_per_slot * padded_batch  +  token_slope * total_tokens

    The first term is what padding costs: it depends on the *padded* batch, not
    the number of real requests, which is precisely the finding e02 produced.
    The second is work you pay regardless.
    """

    base_per_slot_ms: float = PROMOTION_COST_MS / 8.0   # 8->16 adds 8 slots
    token_slope_ms: float = TOKEN_SLOPE_MS_PER_TOKEN
    ladder: tuple[int, ...] = REQUEST_LADDER
    max_batched_tokens: int = MAX_NUM_BATCHED_TOKENS

    def step_cost_ms(self, n_requests: int, total_tokens: int) -> float:
        return (self.base_per_slot_ms * padded_batch(n_requests, self.ladder)
                + self.token_slope_ms * total_tokens)

    def promotion_cost_ms(self, n_from: int, n_to: int) -> float:
        """Extra cost of admitting up to n_to instead of stopping at n_from.

        Zero when both land in the same compiled batch — the case where
        admitting more is free and waiting is pointless.
        """
        return self.base_per_slot_ms * (padded_batch(n_to, self.ladder)
                                        - padded_batch(n_from, self.ladder))

    def fits_token_budget(self, n_requests: int, prompt_len: int) -> bool:
        return n_requests * prompt_len <= self.max_batched_tokens
