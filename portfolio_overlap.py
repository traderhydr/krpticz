"""portfolio_overlap.py -- post-hoc portfolio-concurrency diagnostic for
backtest_engine.py's independently-simulated per-symbol trades.

`BacktestEngine.run_many()` simulates every symbol completely independently
-- no shared clock, no shared position count -- then merges the resulting
reports statistically. That's correct for per-symbol signal quality, but it
means a multi-symbol/multi-coin run's raw trade list implicitly assumes
UNLIMITED concurrent capital: if 20 symbols signal in the same 15m bar,
today's engine happily "opens" all 20 in the simulation.

This module does NOT change that -- re-architecting BacktestEngine into a
single time-synchronized event loop across the whole universe (a real,
enforced max_open_trades constraint) is a substantially larger, separate
piece of work. Instead, this replays the ALREADY-SIMULATED, already-closed
trade list chronologically and asks: under a portfolio-level concurrency
cap, which of these trades would never have gotten a slot? It reports the
breach frequency and the resulting EV/R impact, so a max_open_trades value
can be picked with real numbers behind it -- without touching simulation
correctness or introducing the risk of a from-scratch portfolio engine.

Known approximations (read before trusting the numbers):
  - A trade that doesn't get a slot is treated as REJECTED (never taken,
    contributes zero), not DELAYED to whenever a slot frees up. A real
    system might instead wait and enter later at a different, unmodeled
    price -- this diagnostic doesn't attempt to model that; it answers "how
    much of the current trade list would a hard cap have discarded," not
    "what would a real capacity-constrained bot's P&L have been."
  - Priority among trades that open on the EXACT same bar timestamp is
    ker_ratio descending (higher trend efficiency wins the slot); a trade
    with no recorded ker_ratio sorts last. Non-simultaneous trades are
    never reordered -- an earlier-opened trade always keeps its slot over
    a later one regardless of either one's ker_ratio, matching how a real
    portfolio can't retroactively evict an already-open position for a
    higher-conviction latecomer.
  - Each symbol's trades already reflect that symbol's OWN, unconstrained
    entry ladder/fills -- rejecting a trade here doesn't free up anything
    for a different, not-actually-simulated entry to have taken its place.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import groupby


@dataclass(frozen=True)
class OverlapResult:
    max_open_trades: int
    n_total: int
    n_admitted: int
    n_rejected: int
    rejected_sum_R: float
    admitted_sum_R: float
    admitted_ev_R: float
    baseline_ev_R: float
    max_concurrent_observed: int


def _priority_key(trade: dict) -> float:
    """Sort key for same-timestamp admission ties: higher ker_ratio first
    (more negative key = higher priority); a trade with no recorded
    ker_ratio sorts last, not first -- missing conviction data is not a
    reason to prefer it over a trade that has some."""
    ker = trade.get("ker_ratio")
    return -ker if ker is not None else float("inf")


def simulate_capacity_constraint(trades: list[dict], max_open_trades: int) -> tuple[list[dict], list[dict]]:
    """Greedy, chronological admission-control replay -- see module
    docstring for what this is and isn't modeling.

    Requires each trade dict to have non-None "opened_at_ts"/"closed_at_ts"
    (a trade missing either, e.g. never actually opened, is silently
    dropped from both outputs -- there is nothing to place on a timeline).
    Trades sharing an identical opened_at_ts are ranked against each other
    by `_priority_key` before slot admission; otherwise processed strictly
    in open-time order. A trade is ADMITTED if, after evicting every
    already-open trade whose own close time is at-or-before this trade's
    open time, a free slot remains; otherwise it's REJECTED.

    Returns (admitted, rejected), both lists of the original trade dicts,
    each in the order they were decided (open-time order, ties broken by
    priority).
    """
    if max_open_trades <= 0:
        raise ValueError(f"max_open_trades must be positive, got {max_open_trades}")

    valid = [t for t in trades if t.get("opened_at_ts") is not None and t.get("closed_at_ts") is not None]
    ordered = sorted(valid, key=lambda t: t["opened_at_ts"])

    admitted: list[dict] = []
    rejected: list[dict] = []
    open_heap: list[tuple[int, int]] = []  # (closed_at_ts, id(trade)) -- min-heap by close time
    open_ids: set[int] = set()

    for open_ts, group in groupby(ordered, key=lambda t: t["opened_at_ts"]):
        batch = sorted(group, key=_priority_key)
        for trade in batch:
            while open_heap and open_heap[0][0] <= open_ts:
                _, tid = heapq.heappop(open_heap)
                open_ids.discard(tid)
            if len(open_ids) < max_open_trades:
                tid = id(trade)
                heapq.heappush(open_heap, (trade["closed_at_ts"], tid))
                open_ids.add(tid)
                admitted.append(trade)
            else:
                rejected.append(trade)
    return admitted, rejected


def _max_concurrent(trades: list[dict]) -> int:
    """Peak simultaneously-open trade count, with a close processed before
    an open at the exact same timestamp (matches `simulate_capacity_constraint`'s
    own eviction-before-admission order at a tied timestamp)."""
    events: list[tuple[int, int]] = []
    for t in trades:
        events.append((t["opened_at_ts"], 1))
        events.append((t["closed_at_ts"], -1))
    events.sort(key=lambda e: (e[0], e[1]))  # -1 (close) sorts before +1 (open) at an equal ts
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def overlap_report(trades: list[dict], max_open_trades_grid: list[int]) -> list[OverlapResult]:
    """One `OverlapResult` per cap in `max_open_trades_grid`, all measured
    against the same baseline (every trade with a real `r_multiple`)."""
    valid = [t for t in trades if t.get("r_multiple") is not None and t.get("opened_at_ts") is not None and t.get("closed_at_ts") is not None]
    baseline_ev = (sum(t["r_multiple"] for t in valid) / len(valid)) if valid else float("nan")
    peak_concurrent = _max_concurrent(valid) if valid else 0

    results = []
    for cap in max_open_trades_grid:
        admitted, rejected = simulate_capacity_constraint(valid, cap)
        admitted_r = [t["r_multiple"] for t in admitted]
        rejected_r = [t["r_multiple"] for t in rejected]
        results.append(OverlapResult(
            max_open_trades=cap, n_total=len(valid), n_admitted=len(admitted), n_rejected=len(rejected),
            rejected_sum_R=round(sum(rejected_r), 4) if rejected_r else 0.0,
            admitted_sum_R=round(sum(admitted_r), 4) if admitted_r else 0.0,
            admitted_ev_R=round(sum(admitted_r) / len(admitted_r), 4) if admitted_r else float("nan"),
            baseline_ev_R=round(baseline_ev, 4) if valid else float("nan"),
            max_concurrent_observed=peak_concurrent,
        ))
    return results
