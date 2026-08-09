"""
Minimal vLLM measurement client. No dependencies beyond the stdlib.

Follows the shape of infersim/calibration/measure_throughput_grid_vllm.py —
plain urllib against the OpenAI-compatible API, plus a mock mode so every
experiment is testable with no server.

The one thing that matters here and did not matter in infersim: **exact prompt
token counts**. This project measures behaviour at bucket boundaries, so
"roughly 512 tokens" is useless — a request of 513 tokens lands in a different
compiled executable than one of 512. vLLM's /v1/completions accepts a list of
token IDs as `prompt`, which sidesteps the tokenizer entirely and gives exact
lengths. That is what `prompt_token_ids` is for.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence


@dataclass
class Sample:
    """One request's measurement."""

    prompt_len: int          # exact input tokens requested
    output_len: int          # max_tokens requested
    ttft_ms: float           # time to first token — the prefill-latency proxy
    total_ms: float          # full request wall-clock
    ok: bool = True
    error: str = ""

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Exact-length prompts
# ---------------------------------------------------------------------------

def token_ids(n: int, seed: int = 0, vocab_lo: int = 1000, vocab_hi: int = 20000) -> list[int]:
    """`n` deterministic token IDs, avoiding special/low IDs.

    Deterministic so that repeated runs of an identical config produce identical
    requests — required by the byte-identical-Parquet rule in plan_v4.md.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = random.Random(seed)
    return [rng.randint(vocab_lo, vocab_hi) for _ in range(n)]


# ---------------------------------------------------------------------------
# Real client
# ---------------------------------------------------------------------------

def complete(
    base_url: str,
    model: str,
    prompt_len: int,
    output_len: int = 1,
    seed: int = 0,
    timeout: float = 600.0,
) -> Sample:
    """One streaming completion of an exactly-`prompt_len`-token prompt.

    output_len defaults to 1: for prefill-cost measurement we want TTFT to
    dominate and decode to contribute as little as possible.
    """
    url = base_url.rstrip("/") + "/v1/completions"
    body = json.dumps(
        {
            "model": model,
            "prompt": token_ids(prompt_len, seed=seed),
            "max_tokens": output_len,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )

    t0 = time.perf_counter()
    ttft: float | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                if ttft is None:
                    # First streamed chunk == first token produced.
                    ttft = (time.perf_counter() - t0) * 1000.0
        total = (time.perf_counter() - t0) * 1000.0
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return Sample(prompt_len, output_len, math.nan, math.nan, ok=False, error=str(e))

    if ttft is None:
        return Sample(prompt_len, output_len, math.nan, total, ok=False, error="no tokens streamed")
    return Sample(prompt_len, output_len, ttft, total)


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------
# The mock deliberately implements the STAIRCASE hypothesis: cost depends on the
# padded bucket, not the true length. That is the hypothesis e01 exists to test,
# so mock-mode results must never be read as evidence for it — they only prove
# the harness records and analyses correctly. Set `staircase=False` to get the
# null (cost proportional to true length) and check the analysis can tell them
# apart, which tests/test_e01 does.

def complete_mock(
    prompt_len: int,
    output_len: int = 1,
    ladder: Sequence[int] | None = None,
    staircase: bool = True,
    noise_cv: float = 0.02,
    seed: int = 0,
) -> Sample:
    from ladder import bucket_for  # local import; keeps this module import-light

    rng = random.Random((prompt_len, output_len, seed).__hash__() & 0xFFFFFFFF)
    billed = bucket_for(prompt_len, ladder) if (staircase and ladder) else prompt_len
    # ~13k prefill tokens/s, superlinear in length (attention is quadratic-ish).
    base_ms = billed / 13.0 + (billed ** 2) / 4.0e5 + 12.0
    ttft = base_ms * rng.gauss(1.0, noise_cv)
    total = ttft + output_len * 8.0 * rng.gauss(1.0, noise_cv)
    return Sample(prompt_len, output_len, ttft, total)


# ---------------------------------------------------------------------------
# Repetition / summary
# ---------------------------------------------------------------------------

def repeat(fn, n: int) -> list[Sample]:
    return [fn(i) for i in range(n)]


def summarise(samples: Iterable[Sample], field: str = "ttft_ms") -> dict[str, float]:
    """Median, IQR and coefficient of variation over successful samples.

    CV is the number the noise floor produces, and every downstream threshold is
    expressed as a multiple of it (notes/session_plan.md).
    """
    vals = [getattr(s, field) for s in samples if s.ok and not math.isnan(getattr(s, field))]
    if not vals:
        return {"n": 0, "median": math.nan, "mean": math.nan, "stdev": math.nan, "cv": math.nan}
    mean = statistics.fmean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": mean,
        "stdev": stdev,
        "cv": (stdev / mean) if mean else math.nan,
        "min": min(vals),
        "max": max(vals),
    }


def wait_for_server(base_url: str, timeout: float = 3600.0, poll: float = 15.0) -> bool:
    """Block until /health answers. XLA warmup can take tens of minutes."""
    url = base_url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(poll)
    return False
