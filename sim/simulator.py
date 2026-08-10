"""
Discrete-event simulator for batch admission against a compiled batch ladder.

The question it exists to answer, which session 3 sharpened:

    Stock tpu-inference ALWAYS promotes — it dispatches whatever is waiting and
    pays to pad the batch up to the next compiled size. e02 measured that at
    60.2 ms, against 2.14 ms of real work for the request that triggered it: a
    28x ratio. So: when should it have waited instead?

Waiting is not free either — every request already queued pays the delay. The
crossover depends on arrival rate, and that is exactly what a simulator is for.

Shape borrowed from `infersim` (event loop, policy-as-ABC-with-hooks, one
result dataclass); no research content is shared. Cost parameters come from
`sim/cost_model.py`, which is fitted to captured measurements.
"""

from __future__ import annotations

import heapq
import random
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cost_model import CostModel, padded_batch
from policies import Decision

if TYPE_CHECKING:
    from policies import AdmissionPolicy


@dataclass
class Request:
    rid: int
    arrival_s: float
    prompt_len: int
    dispatched_s: float | None = None
    completed_s: float | None = None

    @property
    def queue_delay_ms(self) -> float:
        if self.dispatched_s is None:
            return float("nan")
        return (self.dispatched_s - self.arrival_s) * 1000.0

    @property
    def latency_ms(self) -> float:
        if self.completed_s is None:
            return float("nan")
        return (self.completed_s - self.arrival_s) * 1000.0


@dataclass
class Batch:
    """One dispatched scheduling step."""

    requests: list[Request]
    start_s: float
    cost_ms: float
    padded_to: int

    @property
    def n(self) -> int:
        return len(self.requests)

    @property
    def wasted_slots(self) -> int:
        """Compiled slots paid for and not used — the padding waste."""
        return self.padded_to - self.n


@dataclass
class SimResult:
    policy: str
    arrival_rate_hz: float
    seed: int
    n_requests: int
    n_batches: int
    total_cost_ms: float
    wasted_slot_fraction: float
    p50_latency_ms: float
    p95_latency_ms: float
    p50_queue_ms: float
    mean_batch_occupancy: float
    # TPU-seconds is the paper's unit; at fixed hardware it is proportional to
    # busy time, so total_cost_ms IS the cost signal.
    cost_per_request_ms: float

    def as_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Simulator:
    """Single-server batch scheduler with a compiled batch ladder.

    Deliberately simple: one prefill queue, one server, no decode. The
    admission decision lives entirely in the policy, so policies are compared
    on identical arrival traces (matched-trace evaluation).
    """

    # Any requested wait shorter than this is treated as "dispatch now".
    # 1 us is far below the ~10 ms granularity of anything being modelled.
    MIN_WAIT_S = 1e-6

    def __init__(self, cost: CostModel | None = None,
                 client_overhead_s: float = 0.0) -> None:
        """`client_overhead_s` is dead time between one dispatch completing and
        the next being issued — a property of whatever DRIVES the server, not of
        the server.

        Default 0: e30 asks what a policy costs intrinsically. But e40's harness
        scrapes /metrics around every batch to get per-batch cost, and that
        scrape plus HTTP and thread spawn measures **22.6-24.9 ms per dispatch**
        (median, from `promote`'s own timings on the holdout runs — `promote`
        never waits, so its whole inter-dispatch gap beyond compute is
        overhead). At 55 req/s, with arrivals 18.2 ms apart, that silently adds
        ~1.4 requests to every batch: measured mean batch size was 2.95 where
        the simulator produced 1.22.

        So validating against e40 requires simulating e40, overhead included.
        The value is measured from a policy that cannot wait, not tuned to make
        the holdout pass.

        Consequence for the headline, which is worth stating in the paper: the
        overhead enlarges batches for EVERY policy, and stock benefits most
        because it dispatches most often. The measured saving of hybrid over
        stock is therefore a **lower bound** on the saving against a stock
        server driven without that overhead.
        """
        self.cost = cost or CostModel()
        self.client_overhead_s = client_overhead_s

    def make_trace(self, n: int, rate_hz: float, prompt_len: int, seed: int) -> list[Request]:
        """Poisson arrivals. Same seed -> same trace, so policies are compared
        on identical inputs rather than on independent draws."""
        rng = random.Random(seed)
        t = 0.0
        out = []
        for i in range(n):
            t += rng.expovariate(rate_hz)
            out.append(Request(i, t, prompt_len))
        return out

    def run(self, trace: list[Request], policy: "AdmissionPolicy") -> SimResult:
        requests = [Request(r.rid, r.arrival_s, r.prompt_len) for r in trace]
        pending: list[Request] = []
        arrivals = list(requests)
        heapq.heapify(ai := [(r.arrival_s, r.rid) for r in arrivals])
        by_id = {r.rid: r for r in requests}

        now = 0.0
        server_free_at = 0.0
        batches: list[Batch] = []

        while ai or pending:
            # Advance to the next event: an arrival, or the server freeing up.
            if not pending:
                now = max(now, ai[0][0])
            while ai and ai[0][0] <= now:
                _, rid = heapq.heappop(ai)
                pending.append(by_id[rid])

            if not pending:
                continue

            now = max(now, server_free_at)
            while ai and ai[0][0] <= now:
                _, rid = heapq.heappop(ai)
                pending.append(by_id[rid])

            next_arrival = ai[0][0] if ai else None
            decision = policy.decide(
                n_pending=len(pending),
                now_s=now,
                oldest_wait_s=now - pending[0].arrival_s,
                next_arrival_s=next_arrival,
                cost=self.cost,
                prompt_len=pending[0].prompt_len,
            )

            if decision.wait_until_s is not None and next_arrival is not None:
                # Hold for more arrivals, but never past the next one — the
                # decision is re-made each time the queue grows.
                target = min(decision.wait_until_s, next_arrival)
                # PROGRESS GUARANTEE. A policy that recomputes its deadline from
                # the current time can converge on it without ever reaching it:
                # `wait_until = now + (max_wait - oldest_wait)` lands a float
                # epsilon short of the threshold, `>=` never fires, and the loop
                # spins forever advancing by 1e-16. Caught with wait-to-fill at
                # 10 req/s. Rather than patch one policy, refuse to honour a
                # wait that does not move the clock: dispatch instead.
                if target <= now + self.MIN_WAIT_S:
                    decision = Decision(dispatch_n=len(pending))
                else:
                    now = target
                    server_free_at = max(server_free_at, now)
                    continue

            take = min(decision.dispatch_n or len(pending), len(pending))
            # Never exceed the token budget in one step. Computed directly:
            # decrementing one at a time is O(pending), which is fine until the
            # server is overloaded and the queue runs to thousands, at which
            # point it dominates the whole simulation.
            budget_max = max(1, self.cost.max_batched_tokens // max(1, pending[0].prompt_len))
            take = min(take, budget_max)

            batch_reqs = pending[:take]
            pending = pending[take:]
            tokens = sum(r.prompt_len for r in batch_reqs)
            cost_ms = self.cost.step_cost_ms(len(batch_reqs), tokens)
            for r in batch_reqs:
                r.dispatched_s = now
                r.completed_s = now + cost_ms / 1000.0
            batches.append(Batch(batch_reqs, now, cost_ms,
                                 padded_batch(len(batch_reqs), self.cost.ladder)))
            # Compute finishes, then the driver spends `client_overhead_s`
            # before it can issue the next dispatch. Requests keep arriving
            # throughout, which is exactly why it changes batch composition.
            # It is NOT added to cost_ms: the TPU is idle during it, and the
            # paper's metric is TPU-busy time.
            server_free_at = now + cost_ms / 1000.0 + self.client_overhead_s
            now = server_free_at

        return self._summarise(policy, requests, batches)

    def _summarise(self, policy, requests: list[Request], batches: list[Batch]) -> SimResult:
        lat = sorted(r.latency_ms for r in requests if r.completed_s is not None)
        q = sorted(r.queue_delay_ms for r in requests if r.dispatched_s is not None)
        total = sum(b.cost_ms for b in batches)
        slots = sum(b.padded_to for b in batches)
        wasted = sum(b.wasted_slots for b in batches)
        return SimResult(
            policy=policy.name,
            arrival_rate_hz=getattr(policy, "_rate_hz", float("nan")),
            seed=getattr(policy, "_seed", -1),
            n_requests=len(requests),
            n_batches=len(batches),
            total_cost_ms=total,
            wasted_slot_fraction=wasted / slots if slots else 0.0,
            p50_latency_ms=statistics.median(lat) if lat else float("nan"),
            p95_latency_ms=lat[int(0.95 * (len(lat) - 1))] if lat else float("nan"),
            p50_queue_ms=statistics.median(q) if q else float("nan"),
            mean_batch_occupancy=(statistics.fmean(b.n / b.padded_to for b in batches)
                                  if batches else float("nan")),
            cost_per_request_ms=total / len(requests) if requests else float("nan"),
        )
