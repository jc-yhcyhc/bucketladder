"""
Admission policies: what to do when the compiled batch you need is not full.

`stock` is not a strawman — session 3 measured that vLLM dispatches whatever is
waiting and pays to pad the batch up, with queue time 0.0 ms at every
concurrency level tested. So `stock` IS `promote`, and the paper's question is
not "should we promote?" but **"stock always promotes; when is that wrong?"**

Policy-as-ABC-with-hooks follows `infersim/policies/capacity/base.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from cost_model import CostModel, padded_batch


@dataclass
class Decision:
    """Dispatch now with `dispatch_n`, or hold until `wait_until_s`."""

    dispatch_n: int | None = None
    wait_until_s: float | None = None


class AdmissionPolicy(ABC):
    name = "abstract"

    @abstractmethod
    def decide(self, *, n_pending: int, now_s: float, oldest_wait_s: float,
               next_arrival_s: float | None, cost: CostModel,
               prompt_len: int) -> Decision:
        ...


class PromoteNow(AdmissionPolicy):
    """Dispatch everything waiting, immediately. **This is stock vLLM.**

    Measured, not assumed: e02 found queue time 0.0 ms at n=6..18 while prefill
    time stepped at the ladder edge.
    """

    name = "promote (stock)"

    def decide(self, *, n_pending, **_) -> Decision:
        return Decision(dispatch_n=n_pending)


class WaitToFill(AdmissionPolicy):
    """Never pay for an unused slot: hold until the compiled batch is full.

    The opposite extreme. Optimal on cost, potentially terrible on latency, and
    it can stall indefinitely at low arrival rates — hence `max_wait_s`, without
    which this policy is not merely bad but unbounded.
    """

    name = "wait-to-fill"

    def __init__(self, max_wait_s: float = 0.5) -> None:
        self.max_wait_s = max_wait_s

    def decide(self, *, n_pending, now_s, oldest_wait_s, next_arrival_s, cost, **_) -> Decision:
        target = padded_batch(n_pending, cost.ladder)
        if n_pending >= target or next_arrival_s is None:
            return Decision(dispatch_n=n_pending)
        if oldest_wait_s >= self.max_wait_s:
            return Decision(dispatch_n=n_pending)  # deadline wins over efficiency
        return Decision(wait_until_s=now_s + (self.max_wait_s - oldest_wait_s))


class Hybrid(AdmissionPolicy):
    """Wait only while waiting is cheaper than the padding it avoids.

    The actual proposal. Dispatching n requests when n has just crossed a ladder
    edge wastes (padded_to - n) slots; the model prices that. Waiting costs
    every queued request its delay. Hold iff

        padding_cost_ms  >  n_pending * expected_wait_ms * latency_weight

    `latency_weight` converts queueing delay into the same currency as compute,
    and is the one free parameter — swept in e30 rather than tuned.
    """

    name = "hybrid"

    def __init__(self, latency_weight: float = 1.0, max_wait_s: float = 0.5) -> None:
        self.latency_weight = latency_weight
        self.max_wait_s = max_wait_s

    def decide(self, *, n_pending, now_s, oldest_wait_s, next_arrival_s, cost, **_) -> Decision:
        target = padded_batch(n_pending, cost.ladder)
        if n_pending >= target or next_arrival_s is None:
            return Decision(dispatch_n=n_pending)
        if oldest_wait_s >= self.max_wait_s:
            return Decision(dispatch_n=n_pending)

        wasted_slots = target - n_pending
        padding_ms = cost.base_per_slot_ms * wasted_slots
        wait_ms = max(0.0, (next_arrival_s - now_s) * 1000.0)
        delay_ms = n_pending * wait_ms * self.latency_weight
        if padding_ms > delay_ms:
            return Decision(wait_until_s=next_arrival_s)
        return Decision(dispatch_n=n_pending)


class DownshiftToEdge(AdmissionPolicy):
    """Dispatch only a *lower* ladder edge's worth, leaving the rest queued.

    The move neither stock nor wait-to-fill can make. With 9 pending, stock pays
    for 16 slots; this dispatches 8 (a perfectly full compiled batch) and lets
    the 9th wait for company. Cheap when the 9th is about to have company, and
    it never stalls, because it always dispatches something.
    """

    name = "downshift"

    def decide(self, *, n_pending, cost, **_) -> Decision:
        target = padded_batch(n_pending, cost.ladder)
        if n_pending == target:
            return Decision(dispatch_n=n_pending)
        lower = [b for b in cost.ladder if b < target]
        if not lower:
            return Decision(dispatch_n=n_pending)
        return Decision(dispatch_n=lower[-1])


class Oracle(AdmissionPolicy):
    """Upper bound: knows the future, waits only when a full batch is imminent.

    Not implementable; it exists so the gap between hybrid and achievable is
    visible rather than assumed.
    """

    name = "oracle"

    def __init__(self, horizon_s: float = 0.05) -> None:
        self.horizon_s = horizon_s

    def decide(self, *, n_pending, now_s, next_arrival_s, cost, **_) -> Decision:
        target = padded_batch(n_pending, cost.ladder)
        if n_pending >= target or next_arrival_s is None:
            return Decision(dispatch_n=n_pending)
        if next_arrival_s - now_s <= self.horizon_s:
            return Decision(wait_until_s=next_arrival_s)
        return Decision(dispatch_n=n_pending)


ALL_POLICIES = {
    "promote": PromoteNow,
    "wait": WaitToFill,
    "hybrid": Hybrid,
    "downshift": DownshiftToEdge,
    "oracle": Oracle,
}
