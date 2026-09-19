"""grid_resimulation.py -- exit-geometry grid re-simulation harness (Step
10.11 follow-up: "model the geometry before touching code").

Answers a narrower, empirically-grounded version of the question the
earlier analytical sweep couldn't: holding entries fixed, which SL/TP1/
TP2/weight combination actually performs best against REAL historical
price action -- not an assumed win rate.

Two-phase design:

  1. Generate ONE canonical set of trade entries by running the REAL,
     unmodified production stack (`VectorizedSignalCache`'s DirectionEngine/
     RegimeFilter gates, then the real `TradeLifecycleManager.open_trade()`,
     then the real, unmodified `PositionState.update()` using CURRENT
     production defaults) over each symbol's history exactly the way
     `backtest_engine.simulate_symbol` does. This is "the existing
     baseline" -- same filters, same entries, same filled-VWAP evolution,
     same single-position-per-symbol gating -- so every geometry in the
     grid sees IDENTICAL entries. FREEZE each trade's entry ladder (levels,
     weights) and its exact fill sequence (which tier filled, at what
     price, on what bar) -- entry fills depend only on price touching a
     resting ladder level, never on exit geometry, so this is a faithful
     freeze, not an approximation.

  2. For every candidate exit geometry, replay ONLY the exit side (SL,
     TP1/TP2 re-anchored to the real filled VWAP every bar exactly like
     risk_manager.py's Step 10.11 fix, the breakeven buffer, a trailing
     runner for whatever weight remains, and an EMA20 time-decay check
     against a swept `bars_held >= time_decay_bars` cutoff -- see
     TIME_DECAY_BARS_GRID) against the real OHLC path from each frozen
     trade's signal bar forward. Win rate, payoff, and drawdown are
     OUTPUTS of this replay, not inputs -- unlike the earlier analytical
     model, which had to assume a fixed win rate because it had no price
     paths to test against.

Known simplifications (stated once here, not scattered through the code):
  - The runner's trailing-stop distance (`RUNNER_TRAIL_ATR_MULT`) is a
    tunable ASSUMPTION, not calibrated from data -- Step 10.10 removed
    KRYPTIC's runner tier entirely, so there is no real production
    precedent for how one behaves in this market. Change the constant
    below and re-run to see how sensitive the ranking is to it.
  - Each frozen trade's exit replay is bounded to
    `[signal_index, next_trade's signal_index - 1]` (or end of data for
    the last trade) so results from different geometries never overlap
    into a neighboring trade's territory. A geometry that would still be
    open at that boundary is closed at the boundary bar's close, flagged
    "WINDOW_END" -- watch the WINDOW_END count in the output; a large one
    means SL/TP combinations near the wide end of the grid are getting
    cut off before naturally resolving and their stats are less reliable.
  - Entry-tier cancellation-on-TP-fire (production's
    `ladder.cancel_unfilled_orders`) is replicated per-geometry: once a
    geometry's OWN TP1 or TP2 fires, no further frozen fill events are
    applied for that geometry from the following bar onward, mirroring
    the real ladder-protection rule without needing to re-simulate entry
    fills per geometry.
  - Friction is applied as `FRICTION_R_AT_SL_2_20 * (2.20 / sl_atr)` R,
    subtracted from every trade's raw price-action R-multiple, win or
    loss. FRICTION_R_AT_SL_2_20 is a median (fees_paid + slippage_paid) /
    risk_dollars measurement rescaled to what it would be at a 2.20 ATR
    SL, since the $ cost is roughly fixed while the R-unit it's divided
    by shrinks/grows with sl_atr. Step 11.3 RECALIBRATED this constant
    from 0.093 to 0.132 (see "CONFIRMED SECOND BUG" below) -- treat any
    number from a round before Step 11.3 as computed against the STALE,
    too-low 0.093 constant, and every EV/WR figure in this file's own
    grid history as needing a re-run before being trusted again.

CONFIRMED BUG, fixed after Round 3 (see Round 4 below): the scratch-win
breakeven buffer was implemented as a side effect of TP1 firing, not as
the INDEPENDENT threshold check risk_manager.py's real step 2b actually
is (`SCRATCH_WIN_TRIGGER_ATR_MULT`, fixed at 0.60, checked every bar
regardless of tp1_fired). This diverges from production for any swept
`tp1_atr != 0.60` -- which is most of what Rounds 1-3 tested -- since the
real stop can lock in near breakeven well before (or after) a TP1 set
away from 0.60 ever fires, something those rounds' replay never modeled.
Found by comparing this tool's own prediction against a REAL production
backtest_engine.py run at the Round-3-recommended defaults: the tool
predicted EV +0.03-0.04R; the real run came back EV -0.0177R. Rounds 1-3's
specific numbers, and especially Round 3's "tp1=0.70 beats 0.60" finding,
should be treated as unreliable pending a re-run under the fix below.

This repo's own dev sandbox has no outbound route to any exchange and no
cached OHLCV -- run this on a machine with real market data access.

Usage:
    python3 -m grid_resimulation --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 180

Grid history:
  Round 1 -- sl_atr 1.20-2.20, tp1_atr 1.10-1.50, tp2_atr 2.20-2.80: all
    144 combinations NEGATIVE EV (best -0.0665R). sl_atr dominated, still
    improving toward its own ceiling (2.20); tp1_atr was a secondary,
    also-still-improving lever; tp2_atr/weights were noise (<=0.005R
    spread).
  Round 2 -- sl_atr 2.20-3.40, tp1_atr 0.70-1.10, tp2_atr 2.80-3.20: 24/96
    combinations turned POSITIVE (best +0.0128R, sl=3.00/tp1=0.70/20%/
    tp2=3.20/50%/runner=30%). This is the first round where sl_atr showed
    an INTERIOR peak (2.6: 0.0061 -> 3.0: 0.0102 -> 3.4: 0.0070 at
    tp1=0.70) rather than still climbing toward the grid edge -- ~3.0 ATR
    looks like the real optimum, not an artifact of an under-explored
    boundary. tp1_atr was STILL monotonically improving toward its floor
    (0.70) with no peak found yet. tp2_atr/weights remained noise.
    IMPORTANT: even the best config's edge is razor-thin -- payoff ratio
    only 0.298 (avg_win 0.20R vs avg_loss -0.67R), needing a 77.05%
    breakeven win rate against an actual 78.52% -- a 1.5-point margin.
    That's not a robust edge; it's one bad sample away from flipping
    negative. If exit-geometry tuning alone can't do much better than
    this, the entry side (deliberately held fixed by this script's
    design) is the more likely place left to find real edge.
  Round 3 -- narrows sl_atr to bracket the round-2 peak (2.80-3.20) and
    continues extending tp1_atr down (0.50-0.70) since it hadn't peaked;
    tp1_w fixed at 0.20 (consistently better both rounds); tp2_atr/tp2_w
    kept to 2x2 confirmation points, still expected to be noise. Best:
    +0.041R (sl=2.80/tp1=0.70/20%/tp2=2.80/50%/runner=30%) -- but this
    number is UNRELIABLE, see the breakeven-buffer bug above: found only
    after this round, by comparing the tool's prediction for this exact
    config against a real production run, which came back EV -0.0177R.
  Round 4 -- fixes the breakeven-buffer bug above, then sweeps
    scratch_trigger_atr_mult itself for the first time
    ([0.60, 0.80, 1.00, 1.20, 1.50] plus TP1_GATED) x scratch_offset_atr_mult
    ([0.15, 0.25]) x sl_atr (2.60-3.00) x tp1_atr (0.70-1.10) x tp2_atr
    (2.80/3.20). RESULT REVERSED THE WORKING HYPOTHESIS: mean EV fell
    monotonically as the trigger loosened (0.60: +0.046R -> 0.80: +0.027R
    -> 1.00: +0.005R -> 1.20: -0.022R -> 1.50: -0.027R; TP1_GATED: +0.010R,
    in between). The production default (0.60) -- the exact value blamed
    for the 73.6% BE-closure rate -- was the BEST value tested, not the
    problem. Mechanism: those BE closures were never evidence the trigger
    was too tight; they reflect tp2 (2.80-3.20 ATR) simply being out of
    reach for most trades regardless of when the stop tightens. Loosening
    the trigger doesn't make price travel further -- it just leaves the
    position exposed to the wide original stop for longer, so a favorable
    move that reverses erases the gain (or turns it into a real loss)
    instead of locking in a small win. Win rate fell right alongside EV as
    the trigger loosened (72.6% -> 60.3%), confirming this directly.
    scratch_offset_atr_mult=0.25 also cleanly beat 0.15 (mean +0.050R vs
    +0.042R) -- covers round-trip fees better on the trades that do
    scratch. Within the winning trigger=0.60 group, tp1_atr finally
    PLATEAUED (0.90: +0.0480R, 1.10: +0.0480R, essentially tied) rather
    than still climbing to an edge, and tp2_atr=2.80 continued to edge out
    3.20. Best: +0.057R (sl=2.60/tp1=1.10/20%/tp2=2.80/50%/runner=30%/
    trigger=0.60/offset=0.25), WR 84.94%, payoff 0.259, a 5.54-point
    margin over its own 79.40% breakeven rate -- the widest margin any
    round has produced, on the most reliable numbers yet (window_end_pct
    0.39%, the lowest of any round). Still short of the +0.15R target
    (38% of the way there). One loose end: sl_atr still favored the
    tightest tested value (2.60) with EV still falling monotonically
    toward it (2.60: +0.049R -> 2.80: +0.048R -> 3.00: +0.042R) -- three
    rounds of drift (3.00 -> 2.80 -> 2.60) with no interior peak found yet.
  Round 5 -- narrows sl_atr to [2.20, 2.40, 2.60] to check for a
    real floor, with every other dimension now fixed at Round 4's settled
    values (tp1_atr=1.10/20%, tp2_atr=2.80/50%/runner=30%,
    scratch_trigger_atr_mult=0.60, scratch_offset_atr_mult=0.25) rather
    than swept -- a small, fast, focused 3-combination run. Confirmed
    sl_atr=2.60 as a genuine interior peak, bracketed on both sides
    across two rounds (2.6 beats 2.8 beats 3.0 in Round 4; 2.6 beats 2.4
    beats 2.2 here). This Round-5 locked-in geometry shipped to
    production (risk_manager.py, Step 11.2) and was verified against a
    real 180-day, 263-trade production backtest.

CONFIRMED SECOND BUG (Step 11.3): that Step 11.2 production verification
came back EV -0.0086R -- still negative. 72.6% of real trades (191/263)
closed via the breakeven-buffer stop, essentially unchanged from Step
11.0's 73.6% (so the wider tp1_atr/tp2_atr from Rounds 4-5 barely
mattered -- most trades never got near them either way), and 46% of
THOSE breakeven-labeled closes (88/191) were net LOSSES, some as deep as
-0.64R. Root cause: FRICTION_R_AT_SL_2_20 (0.093) understated real
round-trip fees+slippage by ~40%. Measured directly off that same
production backtest's own trade log: avg (fees_paid + slippage_paid) /
risk_dollars = ~0.111R at sl_atr=2.60 -- LARGER than
scratch_offset_atr_mult=0.25's own ~0.096R gross margin at that same
sl_atr (0.25/2.60). The "guaranteed small win" scratch-win mechanic was,
on average, a guaranteed small LOSS once real costs were included --
every prior round's EV/WR numbers were computed against a friction
assumption that was too optimistic by roughly 40%, in the same direction
as (and compounding) the Round-1-3 breakeven-timing bug. Fixed by
recalibrating FRICTION_R_AT_SL_2_20 to 0.132 (= 0.111 * 2.60/2.20, so
the constant now reproduces the MEASURED 0.111R at sl_atr=2.60 instead
of inventing a lower one) and, directly in risk_manager.py rather than
waiting on a re-swept grid, widening breakeven_buffer_atr_mult to 0.35
(from 0.25 -- SCRATCH_OFFSET_GRID below updated to match) so its ~0.135R
gross margin clears the measured ~0.111R friction with ~0.024R to
spare, instead of running at a net deficit. This is a direct arithmetic
fix, NOT a new sweep round -- every number in Rounds 1-5 above was
computed under the stale 0.093 constant and should be treated as
optimistic pending a full re-run with the corrected constant.

  Round 6 (Step 11.3, current) -- sweeps `time_decay_bars` in
    [3, 4, 5, 6] (production default 6) with every other dimension held
    at the Step 11.3 settled values above. Motivated by the Step 11.2
    backtest's TIME_DECAY bucket: 24 trades, -$1,435 combined, avg
    -0.596R, uniform across all three symbols -- the single largest
    R-loss bucket in that backtest, bigger than the friction-arithmetic
    bug this round's own sibling fix addresses. Hypothesis under test:
    closing a stalled, momentum-against position sooner (3-5 bars instead
    of 6) prevents some of that -$1,435 without giving up trades that
    would have recovered. The per-geometry output now reports the
    TIME_DECAY bucket's own n/avg_R/sum_R separately from the overall
    ev_R, specifically so this question is answerable directly rather
    than inferred from the aggregate. NOT yet implemented: the separate
    "0.30 ATR favorable-excursion within 3 bars" early-stagnation idea
    raised alongside this request -- that's a distance-based exit
    condition, not a bar-count one, and needs its own GeometryParams
    field and its own sweep dimension if it's wanted next.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest_engine import (
    _REQUIRED_CANDLE_KEYS,
    HistoricalDataLoader,
    VectorizedSignalCache,
    funding_asof,
)
from risk_manager import PositionState, TradeLifecycleManager

log = logging.getLogger(__name__)

FRICTION_R_AT_SL_2_20 = 0.132  # Step 11.3 recalibration -- see module docstring's "CONFIRMED SECOND BUG"
RUNNER_TRAIL_ATR_MULT = 2.0  # ASSUMPTION -- see module docstring.
TP1_W = 0.20                 # fixed (not swept) -- matches production's Step 11.0 default, settled by prior rounds
TP2_W = 0.50                 # fixed (not swept) -- ditto
RUNNER_W = round(1.0 - TP1_W - TP2_W, 6)

# Round 5 settled sl_atr=2.60 (a genuine interior peak, bracketed on both
# sides), tp1_atr=1.10/tp2_atr=2.80 (plateaued in Round 4), and
# scratch_trigger_atr_mult=0.60 (loosening it monotonically WORSENED EV,
# reversing the original hypothesis). scratch_offset_atr_mult is bumped
# from Round 4/5's 0.25 to 0.35 for Step 11.3's friction recalibration
# (see module docstring) -- 0.25's gross margin no longer clears the
# corrected, higher friction constant above. Every value below is a
# fixed, single-value "grid" rather than a swept range -- re-run this
# with the corrected FRICTION_R_AT_SL_2_20 before trusting the resulting
# EV/WR as more than a sanity check on the hand-computed Step 11.3 fix.
SL_GRID = [2.60]
TP1_GRID = [1.10]
TP2_GRID = [2.80]
SCRATCH_TRIGGER_GRID: list[float | None] = [0.60]
SCRATCH_OFFSET_GRID = [0.35]

# Round 6 (Step 11.3, TIME_DECAY investigation): TIME_DECAY exits were the
# single largest loss bucket in the Step 11.2 verification backtest (24
# trades, -$1,435 combined, avg -0.596R, uniform across all 3 symbols --
# a stalled position that never reached TP1 within `time_decay_bars` bars
# AND whose close was already on the wrong side of ema20). This sweeps
# how tight that cutoff can go: production's default is 6 bars (1.5h on
# 15m candles); testing 3/4/5 asks how much of that -$1,435 a faster exit
# would have prevented, and whether the average decay loss shrinks toward
# -0.20/-0.30R as hypothesized, or whether a tighter cutoff just trades
# fewer, smaller decay losses for MORE of them (exiting trades that would
# have recovered). This does NOT implement the separate "0.30 ATR
# favorable-excursion" early-stagnation idea also raised alongside this
# sweep -- that's a different exit condition entirely (distance-based,
# not bar-count-based) and would need its own dedicated sweep dimension.
TIME_DECAY_BARS_GRID = [3, 4, 5, 6]


@dataclass(frozen=True)
class GeometryParams:
    sl_atr: float
    tp1_atr: float
    tp1_w: float
    tp2_atr: float
    tp2_w: float
    runner_w: float
    # Defaults match production's own (pre-Round-4) values -- INDEPENDENT
    # mode at the 0.60 ATR threshold -- so every call site written before
    # Round 4 keeps behaving exactly as already verified, without needing
    # to name these explicitly.
    scratch_trigger_atr_mult: float | None = 0.60  # None = TP1_GATED mode; see SCRATCH_TRIGGER_GRID
    scratch_offset_atr_mult: float = 0.15
    time_decay_bars: int = 6  # matches production's default; see TIME_DECAY_BARS_GRID


@dataclass(frozen=True)
class FrozenTrade:
    """One baseline-generated trade, frozen at the entry side so every
    exit geometry replays against identical entries. See module docstring
    for exactly what "frozen" means and why it's a faithful freeze.

    fill_events: (bar_offset_from_signal, tier_index, fill_price),
        chronological. offset 0 means "already filled as of the signal
        bar itself" (an immediate market Entry 1 prefilled at ladder
        CONSTRUCTION time -- see entry_ladder.py's `_formulate_ladder`
        and risk_manager.py's Step 10.11 fix docstring for why this can't
        just be read off `PositionState.execution_log`). offset N>=1
        means "filled during the Nth bar after the signal bar", i.e.
        absolute row `signal_index + N`.
    """
    symbol: str
    direction: str
    signal_index: int
    signal_ts: int
    window_end_index: int
    expected_vwap: float
    atr_at_signal: float
    ladder_levels: tuple[float, ...]
    ladder_weights: tuple[float, ...]
    fill_events: tuple[tuple[int, int, float], ...]


def build_grid() -> list[GeometryParams]:
    grid = []
    for sl, tp1, tp2, scratch_trigger, scratch_offset, time_decay_bars in itertools.product(
        SL_GRID, TP1_GRID, TP2_GRID, SCRATCH_TRIGGER_GRID, SCRATCH_OFFSET_GRID, TIME_DECAY_BARS_GRID,
    ):
        grid.append(GeometryParams(
            sl_atr=sl, tp1_atr=tp1, tp1_w=TP1_W, tp2_atr=tp2, tp2_w=TP2_W, runner_w=RUNNER_W,
            scratch_trigger_atr_mult=scratch_trigger, scratch_offset_atr_mult=scratch_offset,
            time_decay_bars=time_decay_bars,
        ))
    return grid


# ---------------------------------------------------------------------------
# Phase 1: canonical baseline entries (real, unmodified production stack)
# ---------------------------------------------------------------------------

def _extract_frozen_trade(symbol: str, signal_index: int, signal_ts: int, window_end_index: int,
                           position: PositionState, diagnostics: dict, baseline_mgr: TradeLifecycleManager) -> FrozenTrade | None:
    ladder = position.ladder
    expected_vwap = diagnostics["expected_vwap"]
    initial_sl = diagnostics["initial_sl"]
    atr_at_signal = abs(expected_vwap - initial_sl) / baseline_mgr.initial_sl_atr_mult

    fill_events: list[tuple[int, int, float]] = []
    logged_tiers = {e["tier_index"] for e in position.execution_log if e["type"] == "ENTRY_FILL"}
    for i, filled in enumerate(ladder.fills):
        if filled and ladder.weights[i] > 0 and i not in logged_tiers:
            # Construction-time prefill (immediate market Entry 1) -- never
            # appears in execution_log; see class docstring.
            fill_events.append((0, i, ladder.levels[i]))
    for e in position.execution_log:
        if e["type"] == "ENTRY_FILL":
            fill_events.append((e["bar_index"] + 1, e["tier_index"], e["price"]))
    fill_events.sort(key=lambda t: t[0])
    if not fill_events:
        return None  # nothing ever filled -- not a real trade (matches production's phantom-trade guard)

    return FrozenTrade(
        symbol=symbol, direction=position.direction, signal_index=signal_index, signal_ts=signal_ts,
        window_end_index=window_end_index, expected_vwap=expected_vwap, atr_at_signal=atr_at_signal,
        ladder_levels=tuple(ladder.levels), ladder_weights=tuple(ladder.weights), fill_events=tuple(fill_events),
    )


def generate_frozen_trades(
    symbol: str, ltf_df: pd.DataFrame, btc_df: pd.DataFrame, funding_series: pd.Series | None,
    baseline_mgr: TradeLifecycleManager, *, buffer_capacity: int = 1000,
) -> tuple[list[FrozenTrade], VectorizedSignalCache]:
    """Replays `ltf_df` through the exact production decision stack (same
    call sequence as `backtest_engine.simulate_symbol`, minus the
    DryRunHarness bookkeeping this doesn't need) using `baseline_mgr`'s
    OWN exit geometry to decide when each position closes and the next
    signal search resumes -- i.e. "identical to the existing baseline"."""
    entry_engine = baseline_mgr.entry_ladder_engine
    cache = VectorizedSignalCache(
        ltf_df, btc_df, direction_engine=entry_engine.direction_engine, regime_filter=entry_engine.regime_filter,
        funding_series=funding_series,
    )
    n = len(cache.ltf_df)
    raw: list[tuple[int, PositionState, dict]] = []
    position: PositionState | None = None

    for i in range(n):
        bar = {k: cache.ltf_df[k].iat[i] if k in cache.ltf_df.columns else None for k in _REQUIRED_CANDLE_KEYS}
        bar["ts"] = int(bar["ts"])

        if position is not None and not position.closed:
            enriched = {**bar, **cache.enrich(i)}
            position.update(enriched, htf_bar=cache.htf_bar(i))
            if position.closed:
                position = None
            continue

        direction = cache.is_candidate(i)
        if direction is None:
            continue

        sub_ltf = cache.ltf_df.iloc[max(0, i + 1 - buffer_capacity): i + 1].reset_index(drop=True)
        sub_btc = cache.btc_df_upto(i, buffer_capacity=buffer_capacity)
        sub_htf = cache.htf_df_upto(i, buffer_capacity=buffer_capacity)
        funding = funding_asof(funding_series, bar["ts"])
        new_position, diagnostics = baseline_mgr.open_trade(sub_ltf, sub_btc, funding, htf_df=sub_htf)
        if new_position is not None and new_position.direction == direction:
            position = new_position
            raw.append((i, new_position, diagnostics))

    frozen: list[FrozenTrade] = []
    for k, (sig_idx, pos, diag) in enumerate(raw):
        window_end = (raw[k + 1][0] - 1) if k + 1 < len(raw) else (n - 1)
        signal_ts = int(cache.ltf_df["ts"].iat[sig_idx])
        ft = _extract_frozen_trade(symbol, sig_idx, signal_ts, window_end, pos, diag, baseline_mgr)
        if ft is not None:
            frozen.append(ft)
    return frozen, cache


# ---------------------------------------------------------------------------
# Phase 2: per-geometry exit replay
# ---------------------------------------------------------------------------

def replay_exit(
    trade: FrozenTrade, *, atr_arr: np.ndarray, ema20_arr: np.ndarray,
    high_arr: np.ndarray, low_arr: np.ndarray, close_arr: np.ndarray, params: GeometryParams,
) -> dict | None:
    sign = 1.0 if trade.direction == "LONG" else -1.0
    initial_sl = trade.expected_vwap - sign * params.sl_atr * trade.atr_at_signal
    current_sl = initial_sl

    fills_by_offset: dict[int, list[tuple[int, float]]] = {}
    for offset, tier_i, price in trade.fill_events:
        fills_by_offset.setdefault(offset, []).append((tier_i, price))

    filled_tiers: dict[int, float] = {}
    tp1_fired = tp2_fired = breakeven_moved = runner_active = entries_cancelled_from_next_bar = False
    runner_extreme = None
    realized_weight = 0.0
    realized_pnl = 0.0
    exit_reason: str | None = None
    closed = False
    bars_held = 0
    window_len = trade.window_end_index - trade.signal_index

    def filled_weight_total() -> float:
        return sum(trade.ladder_weights[t] for t in filled_tiers)

    def filled_vwap() -> float | None:
        if not filled_tiers:
            return None
        tw = filled_weight_total()
        if tw <= 0:
            return None
        return sum(trade.ladder_weights[t] * filled_tiers[t] for t in filled_tiers) / tw

    def open_size() -> float:
        return max(0.0, filled_weight_total() - realized_weight)

    def close_amount(amount: float, price: float) -> None:
        nonlocal realized_weight, realized_pnl
        vwap = filled_vwap()
        if vwap is None or amount <= 0:
            return
        realized_weight += amount
        realized_pnl += (price - vwap) * amount * sign

    for tier_i, price in fills_by_offset.get(0, []):
        filled_tiers[tier_i] = price

    for offset in range(1, window_len + 1):
        abs_row = trade.signal_index + offset
        bars_held += 1
        hi, lo, close = float(high_arr[abs_row]), float(low_arr[abs_row]), float(close_arr[abs_row])
        atr = float(atr_arr[abs_row])
        ema20 = float(ema20_arr[abs_row])

        # 1. Stop-loss first, pre-bar-open size, pessimistic same-bar
        # resolution. `current_sl` also IS the runner's trailing stop once
        # ratcheted (see step 3b) -- if the runner phase already started
        # on a prior bar, a hit here is really the trail catching up a bar
        # later, not the original hard stop, so label it accordingly for
        # accurate reporting (the R math is identical either way).
        pre_open = open_size()
        if pre_open > 0:
            sl_hit = (lo <= current_sl) if sign > 0 else (hi >= current_sl)
            if sl_hit:
                close_amount(pre_open, current_sl)
                exit_reason = "RUNNER_TRAIL" if runner_active else "SL"
                closed = True
                break

        # 2. Frozen entry fills -- suppressed once THIS geometry's own TP
        # fired on a prior bar (ladder-protection cancellation).
        if not entries_cancelled_from_next_bar:
            for tier_i, price in fills_by_offset.get(offset, []):
                filled_tiers[tier_i] = price

        vwap = filled_vwap()
        atr_ok = not np.isnan(atr) and atr > 0

        # 2a. Re-anchor TP1/TP2 to the real filled vwap every bar.
        tp1_level = (vwap + sign * params.tp1_atr * atr) if (vwap is not None and atr_ok and not tp1_fired) else None
        tp2_level = (vwap + sign * params.tp2_atr * atr) if (vwap is not None and atr_ok and not tp2_fired) else None

        def _move_breakeven_to(new_vwap: float) -> None:
            nonlocal current_sl, breakeven_moved
            candidate = new_vwap + sign * params.scratch_offset_atr_mult * atr
            if (candidate > current_sl) if sign > 0 else (candidate < current_sl):
                current_sl = candidate
            breakeven_moved = True

        # 2b. Active scratch-win buffer, INDEPENDENT mode: a fixed-ATR
        # threshold check against `scratch_trigger_atr_mult`, exactly
        # like risk_manager.py's own step 2b -- NOT a side effect of TP1
        # firing. Runs every bar regardless of tp1_fired: for any
        # tp1_atr != scratch_trigger_atr_mult, this threshold can be
        # reached well before (or after) TP1's own level, and production
        # (in this mode) moves the stop the moment ITS OWN threshold is
        # hit, independent of TP1. Skipped entirely in TP1_GATED mode
        # (`scratch_trigger_atr_mult is None`) -- see step 3 below instead.
        if (
            params.scratch_trigger_atr_mult is not None
            and not breakeven_moved and open_size() > 0 and vwap is not None and atr_ok
        ):
            threshold = params.scratch_trigger_atr_mult * atr
            reached = (hi >= vwap + threshold) if sign > 0 else (lo <= vwap - threshold)
            if reached:
                _move_breakeven_to(vwap)

        tp_fired_this_bar = False

        # 3. Passive TP1/TP2 checks.
        if tp1_level is not None:
            hit = (hi >= tp1_level) if sign > 0 else (lo <= tp1_level)
            if hit:
                tp1_fired = True
                tp_fired_this_bar = True
                amt = min(params.tp1_w, open_size())
                if amt > 0:
                    close_amount(amt, tp1_level)
                # TP1_GATED mode: the buffer moves ONLY as a side effect
                # of TP1 actually firing, using whatever vwap holds at
                # that moment -- Rounds 1-3's original mechanic,
                # deliberately reintroduced here as an explicit,
                # selectable comparison point (see SCRATCH_TRIGGER_GRID).
                if params.scratch_trigger_atr_mult is None and not breakeven_moved and vwap is not None and atr_ok:
                    _move_breakeven_to(vwap)
        if tp2_level is not None and open_size() > 0:
            hit = (hi >= tp2_level) if sign > 0 else (lo <= tp2_level)
            if hit:
                tp2_fired = True
                tp_fired_this_bar = True
                amt = min(params.tp2_w, open_size())
                if amt > 0:
                    close_amount(amt, tp2_level)
        if tp_fired_this_bar:
            entries_cancelled_from_next_bar = True

        # 3b. Runner phase: once TP1 AND TP2 have both fired and weight
        # remains, trail an ATR chandelier stop (ratchets favorably only).
        if tp1_fired and tp2_fired and open_size() > 0 and atr_ok:
            if not runner_active:
                runner_active = True
                runner_extreme = close
            runner_extreme = max(runner_extreme, hi) if sign > 0 else min(runner_extreme, lo)
            trail_stop = runner_extreme - sign * RUNNER_TRAIL_ATR_MULT * atr
            if (trail_stop > current_sl) if sign > 0 else (trail_stop < current_sl):
                current_sl = trail_stop
            trail_hit = (lo <= current_sl) if sign > 0 else (hi >= current_sl)
            if trail_hit:
                close_amount(open_size(), current_sl)
                exit_reason = "RUNNER_TRAIL"
                closed = True
                break

        # 3c. Time-decay invalidation -- swept via params.time_decay_bars
        # since Round 6 (Step 11.3); see TIME_DECAY_BARS_GRID.
        if not tp1_fired and open_size() > 0 and bars_held >= params.time_decay_bars and not np.isnan(ema20):
            against = (close < ema20) if sign > 0 else (close > ema20)
            if against:
                close_amount(open_size(), close)
                exit_reason = "TIME_DECAY"
                closed = True
                break

        # 4. Fully closed: nothing open and no pending frozen fill left.
        pending = any(o > offset for o in fills_by_offset) and not entries_cancelled_from_next_bar
        if open_size() <= 1e-9 and not pending:
            exit_reason = exit_reason or ("TP2" if tp2_fired else "TP1")
            closed = True
            break

    if not closed:
        remaining = open_size()
        if remaining > 0:
            close_amount(remaining, float(close_arr[trade.signal_index + window_len]))
        exit_reason = "WINDOW_END"

    fw = filled_weight_total()
    if fw <= 0:
        return None
    risk_amount = abs(trade.expected_vwap - initial_sl) * fw
    r_raw = realized_pnl / risk_amount if risk_amount > 0 else 0.0
    friction_r = FRICTION_R_AT_SL_2_20 * (2.20 / params.sl_atr)
    return {
        "symbol": trade.symbol, "signal_ts": trade.signal_ts, "direction": trade.direction,
        "r_multiple": r_raw - friction_r, "exit_reason": exit_reason, "bars_held": bars_held, "filled_weight": fw,
    }


# ---------------------------------------------------------------------------
# Sweep driver + reporting
# ---------------------------------------------------------------------------

_ArrBundle = tuple[list[FrozenTrade], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def run_sweep(frozen_by_symbol: dict[str, _ArrBundle], grid: list[GeometryParams]) -> pd.DataFrame:
    rows = []
    for params in grid:
        results: list[dict] = []
        for symbol, (trades, atr_arr, ema20_arr, high_arr, low_arr, close_arr) in frozen_by_symbol.items():
            for t in trades:
                r = replay_exit(t, atr_arr=atr_arr, ema20_arr=ema20_arr, high_arr=high_arr, low_arr=low_arr, close_arr=close_arr, params=params)
                if r is not None:
                    results.append(r)
        if not results:
            continue
        results.sort(key=lambda r: r["signal_ts"])
        r_values = np.array([r["r_multiple"] for r in results])
        wins = r_values[r_values > 0]
        losses = r_values[r_values <= 0]
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_win, gross_loss = float(wins.sum()), float(abs(losses.sum()))

        equity = np.cumsum(r_values)
        running_max = np.maximum.accumulate(equity)
        max_dd = float((running_max - equity).max()) if len(equity) else 0.0

        window_end_count = sum(1 for r in results if r["exit_reason"] == "WINDOW_END")
        # Time-decay-specific breakout (Round 6 / Step 11.3): the overall
        # ev_R/win_rate_pct alone can't show whether a tighter
        # time_decay_bars cutoff actually shrinks the decay bucket's own
        # loss, or just trades fewer/bigger decay losses for more/smaller
        # ones -- report that bucket's own count and R directly.
        decay_r = np.array([r["r_multiple"] for r in results if r["exit_reason"] == "TIME_DECAY"])
        rows.append(dict(
            sl_atr=params.sl_atr, tp1_atr=params.tp1_atr, tp1_w=params.tp1_w,
            tp2_atr=params.tp2_atr, tp2_w=params.tp2_w, runner_w=params.runner_w,
            scratch_trigger_atr_mult=("TP1_GATED" if params.scratch_trigger_atr_mult is None else params.scratch_trigger_atr_mult),
            scratch_offset_atr_mult=params.scratch_offset_atr_mult,
            time_decay_bars=params.time_decay_bars,
            n_trades=len(results), win_rate_pct=round(100.0 * len(wins) / len(r_values), 2),
            avg_win_R=round(avg_win, 4), avg_loss_R=round(avg_loss, 4),
            payoff_ratio=round(avg_win / abs(avg_loss), 4) if avg_loss != 0 else np.nan,
            profit_factor=round(gross_win / gross_loss, 4) if gross_loss > 0 else np.nan,
            ev_R=round(float(r_values.mean()), 4), max_dd_R=round(max_dd, 4),
            window_end_pct=round(100.0 * window_end_count / len(results), 2),
            n_time_decay=len(decay_r),
            time_decay_avg_R=round(float(decay_r.mean()), 4) if len(decay_r) else np.nan,
            time_decay_sum_R=round(float(decay_r.sum()), 4) if len(decay_r) else 0.0,
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df  # no frozen trades produced a result under any geometry -- nothing to sort
    return df.sort_values("ev_R", ascending=False).reset_index(drop=True)


async def main() -> None:  # pragma: no cover -- real-network entrypoint, not exercised by tests
    parser = argparse.ArgumentParser(description="KRYPTIC exit-geometry grid re-simulation")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--warmup-days", type=int, default=10)
    parser.add_argument("--cache-dir", default="./.backtest_cache")
    parser.add_argument("--out", default="grid_resimulation_results.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 86400 * 1000
    warmup_ms = args.warmup_days * 86400 * 1000

    loader = HistoricalDataLoader(cache_dir=args.cache_dir)
    baseline_mgr = TradeLifecycleManager()  # current production defaults (Step 10.11) -- defines the baseline entry gating
    btc_df = await loader.load("BTCUSDT", "4h", start_ms - warmup_ms, end_ms)

    frozen_by_symbol: dict[str, _ArrBundle] = {}
    total_trades = 0
    for symbol in args.symbols.split(","):
        symbol = symbol.strip().upper()
        ltf_df = await loader.load(symbol, "15m", start_ms - warmup_ms, end_ms)
        funding = await loader.load_funding(symbol, start_ms - warmup_ms, end_ms)
        trades, cache = generate_frozen_trades(symbol, ltf_df, btc_df, funding, baseline_mgr)
        trades = [t for t in trades if t.signal_ts >= start_ms]
        frozen_by_symbol[symbol] = (
            trades, cache.atr14.to_numpy(), cache.ema20.to_numpy(),
            cache.ltf_df["high"].to_numpy(), cache.ltf_df["low"].to_numpy(), cache.ltf_df["close"].to_numpy(),
        )
        total_trades += len(trades)
        print(f"{symbol}: {len(trades)} baseline trade(s) generated for replay")
    await loader.aclose()

    grid = build_grid()
    print(f"\nSweeping {len(grid)} geometr(y/ies) x {total_trades} frozen trade(s)...")
    t0 = time.time()
    results = run_sweep(frozen_by_symbol, grid)
    print(f"Done in {time.time() - t0:.1f}s -- {len(results)} geometr(y/ies) produced results.\n")

    results.to_csv(args.out, index=False)
    print(f"Wrote {len(results)} row(s) to {args.out}\n")

    cols = ["sl_atr", "tp1_atr", "tp2_atr", "scratch_trigger_atr_mult", "scratch_offset_atr_mult", "n_trades",
            "win_rate_pct", "avg_win_R", "avg_loss_R", "payoff_ratio", "profit_factor", "ev_R", "max_dd_R"]
    print("=== TOP 5 by empirical EV ===")
    print(results[cols].head(5).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    asyncio.run(main())
