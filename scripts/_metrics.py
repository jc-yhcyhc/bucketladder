"""
Server-side timing, scraped from vLLM's Prometheus endpoint.

Why this exists (notes/solidity.md, R1): the paper's claims are about **cost** —
TPU-seconds — and a client stopwatch measures network RTT + HTTP + tokenizer +
queueing + compute. Fine for a kill check, not fine for a headline number.
vLLM already measures the pieces separately:

    vllm:request_prefill_time_seconds   histogram  prefill phase duration
    vllm:request_decode_time_seconds    histogram  decode phase duration
    vllm:request_queue_time_seconds     histogram  arrival -> first scheduled
    vllm:time_to_first_token_seconds    histogram  arrival -> first token
    vllm:e2e_request_latency_seconds    histogram  arrival -> done

`request_queue_time_seconds` is the one that fixes e02. Client TTFT under
concurrency conflates waiting with computing, so it cannot say whether crossing
a batch-padding edge *costs* anything. Queue time and prefill time separate
exactly those two, so the question becomes answerable:

    prefill time rises at the edge, queue time flat  -> padding is being paid
    queue time rises, prefill time flat              -> the server is waiting
    both flat                                        -> the edge costs nothing

**Method: deltas, not snapshots.** A histogram's `_sum` and `_count` are
cumulative over the server's lifetime. Scraping before and after a measurement
block and dividing the deltas gives the exact mean over precisely the requests
we issued — no contamination from warmup or from other cells.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

# Prometheus text format: name{label="v",...} value
_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)\s*$")

# The metrics this project actually uses. Others are ignored to keep snapshots
# small and comparisons obvious.
TIMING_METRICS = (
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_inference_time_seconds",
    # NOT a time. Discovered on hardware 2026-08-09 and kept because it is the
    # most direct mechanism evidence available (solidity.md R5): the count of KV
    # tokens actually computed during prefill. If it tracks the PADDED bucket,
    # the hardware is genuinely computing the padding; if it tracks the TRUE
    # length, RPA skips the work and the cost must be elsewhere. Either answer
    # converts "flatness is 0.97" from a correlation into a mechanism.
    "vllm:request_prefill_kv_computed_tokens",
    # ALSO not a time. A histogram over ENGINE ITERATIONS, so its `_count` delta
    # is the number of scheduler steps that ran, and its `_sum` is the tokens
    # they scheduled. Added 2026-08-10 to explain e02's bimodality: at n=9..14
    # the per-batch cost has two modes (~90 ms and ~140 ms) with the expensive
    # one taking over as n rises, and the two candidate explanations —
    # "one padded step" versus "two smaller steps" — differ by exactly one
    # iteration. Nothing else exposed on /metrics separates them.
    "vllm:iteration_tokens_total",
)


@dataclass
class HistoSnapshot:
    """Cumulative sum/count of one histogram at one instant."""

    name: str
    total: float   # _sum
    count: float   # _count

    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")


def parse_prometheus(text: str) -> dict[str, HistoSnapshot]:
    """Extract `_sum`/`_count` for every timing histogram present.

    Labels are ignored: these experiments serve one model on one server, so all
    series for a metric belong to it. If a second model is ever served
    concurrently this must start filtering on model_name.
    """
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, value = m.group("name"), m.group("value")
        try:
            v = float(value)
        except ValueError:
            continue
        if name.endswith("_sum"):
            sums[name[: -len("_sum")]] = sums.get(name[: -len("_sum")], 0.0) + v
        elif name.endswith("_count"):
            base = name[: -len("_count")]
            counts[base] = counts.get(base, 0.0) + v

    out: dict[str, HistoSnapshot] = {}
    for base in set(sums) | set(counts):
        if base in TIMING_METRICS:
            out[base] = HistoSnapshot(base, sums.get(base, 0.0), counts.get(base, 0.0))
    return out


def scrape(base_url: str, timeout: float = 30.0) -> dict[str, HistoSnapshot]:
    url = base_url.rstrip("/") + "/metrics"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return parse_prometheus(resp.read().decode())


# --- histogram buckets ------------------------------------------------------
# `_sum` and `_count` give the mean and nothing about shape. That is enough for
# a timing histogram and not enough for `iteration_tokens_total`, where the
# question is how the scheduler DISTRIBUTED tokens across steps. e04 showed the
# per-dispatch cost is bimodal with identical total scheduled tokens and, at
# n=12, an identical step COUNT in both modes — and a count cannot tell 2
# prefill + 1 decode from 1 prefill + 2 decode. The bucket edges can.

_BUCKET = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)_bucket\{(?P<labels>[^}]*)\}\s+(?P<value>[^\s]+)\s*$')
_LE = re.compile(r'le="(?P<le>[^"]+)"')


def parse_buckets(text: str, metric: str) -> dict[float, float]:
    """Cumulative bucket counts for one histogram, keyed by upper edge.

    Prometheus buckets are cumulative (`le` = "less than or equal"), so
    differencing adjacent edges gives the per-bucket population. Returned
    cumulative because the deltas must be taken against another SNAPSHOT first;
    differencing edges before differencing snapshots would mix the two.
    """
    out: dict[float, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "_bucket{" not in line:
            continue
        m = _BUCKET.match(line)
        if not m or m.group("name") != metric:
            continue
        le = _LE.search(m.group("labels"))
        if not le:
            continue
        edge = float("inf") if le.group("le") in ("+Inf", "Inf") else float(le.group("le"))
        try:
            out[edge] = out.get(edge, 0.0) + float(m.group("value"))
        except ValueError:
            continue
    return out


def scrape_buckets(base_url: str, metric: str, timeout: float = 30.0) -> dict[float, float]:
    url = base_url.rstrip("/") + "/metrics"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return parse_buckets(resp.read().decode(), metric)


def bucket_delta(before: dict[float, float], after: dict[float, float]) -> dict[float, float]:
    """Per-bucket (not cumulative) population added between two snapshots.

    Returns {upper_edge: count} for the steps that ran in between — i.e. the
    distribution of tokens per scheduler step over exactly this dispatch.
    """
    edges = sorted(set(before) | set(after))
    cum = [(e, after.get(e, 0.0) - before.get(e, 0.0)) for e in edges]
    out: dict[float, float] = {}
    prev = 0.0
    for e, c in cum:
        n = c - prev
        if n > 0:
            out[e] = n
        prev = c
    return out


def delta(before: dict[str, HistoSnapshot], after: dict[str, HistoSnapshot]) -> dict[str, dict[str, float]]:
    """Per-metric mean over exactly the requests issued between two scrapes.

    Returns {metric: {"sum_s", "count", "mean_s", "mean_ms"}}. A metric whose
    count did not advance is omitted — reporting a mean over zero requests would
    be worse than reporting nothing.
    """
    out: dict[str, dict[str, float]] = {}
    for name, aft in after.items():
        bef = before.get(name)
        d_count = aft.count - (bef.count if bef else 0.0)
        d_sum = aft.total - (bef.total if bef else 0.0)
        if d_count <= 0:
            continue
        out[name] = {
            "sum_s": d_sum,
            "count": d_count,
            "mean_s": d_sum / d_count,
            "mean_ms": d_sum / d_count * 1000.0,
        }
    return out


def short(name: str) -> str:
    """`vllm:request_prefill_time_seconds` -> `prefill`, for readable tables."""
    n = name.removeprefix("vllm:").removesuffix("_seconds")
    n = n.removeprefix("request_").removesuffix("_time")
    return {"time_to_first_token": "ttft", "e2e_request_latency": "e2e",
            "prefill_kv_computed_tokens": "kv_tokens"}.get(n, n)


def metrics_available(base_url: str) -> bool:
    try:
        return bool(scrape(base_url))
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

# vLLM's default histogram edges for iteration_tokens_total are powers of two
# over the batched-token range. Mirrored here so the mock's bucket arithmetic is
# the same shape as the server's.
MOCK_BUCKET_EDGES = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, float("inf"))


class MockMetrics:
    """Accumulates fake server-side timings so experiments run with no server.

    Mirrors the real semantics: cumulative counters that the caller snapshots
    before and after. Feed it per-request (prefill_s, queue_s) and it behaves
    like the endpoint.
    """

    def __init__(self) -> None:
        self._sum: dict[str, float] = {m: 0.0 for m in TIMING_METRICS}
        self._count: dict[str, float] = {m: 0.0 for m in TIMING_METRICS}
        self._iterations: list[float] = []

    def record(self, prefill_s: float, queue_s: float = 0.0, decode_s: float = 0.0) -> None:
        for name, v in (
            ("vllm:request_prefill_time_seconds", prefill_s),
            ("vllm:request_queue_time_seconds", queue_s),
            ("vllm:request_decode_time_seconds", decode_s),
            ("vllm:time_to_first_token_seconds", prefill_s + queue_s),
            ("vllm:e2e_request_latency_seconds", prefill_s + queue_s + decode_s),
        ):
            self._sum[name] += v
            self._count[name] += 1

    def record_iteration(self, tokens: float) -> None:
        """One engine step. Separate from `record` because iterations and
        requests are different populations — n requests can share one step, and
        telling those apart is the entire point of e04."""
        self._sum["vllm:iteration_tokens_total"] += tokens
        self._count["vllm:iteration_tokens_total"] += 1
        self._iterations.append(tokens)

    def snapshot(self) -> dict[str, HistoSnapshot]:
        return {m: HistoSnapshot(m, self._sum[m], self._count[m])
                for m in TIMING_METRICS if self._count[m] > 0}

    def bucket_snapshot(self, metric: str) -> dict[float, float]:
        """Cumulative bucket counts, mirroring Prometheus\'s `le` semantics so
        the mock exercises the same differencing code the live path uses."""
        edges = list(MOCK_BUCKET_EDGES)
        out: dict[float, float] = {}
        for e in edges:
            out[e] = float(sum(1 for v in self._iterations if v <= e))
        return out
