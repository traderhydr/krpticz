"""risk_manager.py -- trade lifecycle & position exit manager (KRYPTIC Step 5,
restructured for Step 10.10's high-probability target geometry).

`TradeLifecycleManager` turns an `EntryLadder` (from entry_ladder.py) into a
fully-specified trade: an initial ATR-scaled hard stop and a 2-target take-
profit plan, both priced directly off the ladder's own theoretical VWAP
(no more Fibonacci-leg-derived targets) -- then hands back a `PositionState`,
the bar-by-bar state machine that tracks fills, stop/TP execution, the
active scratch-win stop migration, and time-decay invalidation.

Step 10.10 replaced the prior 4-target-plus-runner geometry (Steps 5/10.6/
10.8) with a deliberately simpler, tighter, higher-win-rate design: a wide
2.20 ATR noise stop, TP1 at 0.75 ATR (60% of size) and TP2 at 1.50 ATR
(remaining 40%) -- both measured off the REAL filled VWAP, re-anchored on
every new entry fill, not estimated once from an impulse leg. TP3/TP4/the
dynamic runner no longer exist (their list slots stay reserved -- `None`
price, 0 weight -- so `PositionState`'s 5-element `tp_levels`/`tp_weights`
shape doesn't have to change everywhere that already assumes it).

Step 10.11 fixed a residual gap in that re-anchoring: it only ran when a
NEW entry tier filled during `update()`'s bar loop, but an immediate
market Entry 1 fills at ladder-construction time, before `PositionState`
even exists -- so its fill never went through that path, leaving TP1/TP2
priced off the theoretical, all-4-tiers `expected_vwap` forever whenever
price never retraced far enough to fill a second tier. For a LONG,
`expected_vwap` sits BELOW a lone Entry-1 fill (it blends in deeper,
unfilled retracement levels); at 0.75 ATR out, that construction-time TP1
could land BELOW the real filled VWAP -- an inverted target that a normal
retracement would touch at a loss while still recording a "TP1 hit."
Step 10.11 makes `_revalidate_tp_geometry` run every bar (not just on a
fresh fill), widens TP2 to 1.80 ATR, floors TP1's distance at 1.10 ATR
(`max(tp1_atr_mult, tp1_min_atr_mult) * atr`), and adds a hard invariant
clamp forcing TP1 back to the profitable side of the real filled VWAP if
it's ever found otherwise.

Step 11.0 replaces the geometry's actual numbers with the output of a
real-price-path grid re-simulation (grid_resimulation.py, 3 rounds against
405 real BTCUSDT/ETHUSDT/SOLUSDT trades over 180 days), not another guess:
initial_sl 2.80 ATR (widened from 2.20), tp1 0.60 ATR (tightened from
0.75) at 20% weight (down from 60%), tp2 3.20 ATR (widened from 1.80) at
50% weight, and a NEW 30%-weight trailing-runner tier for what's left --
the first time this module has had a genuine third tier since Step 10.10
removed the Step 5/10.8 runner entirely. The re-simulation swept sl_atr
and tp1_atr on real historical price action and found real interior
optima for both (not edge-of-grid artifacts -- see the module's own git
history for the 3 rounds' numbers); tp2_atr and the weight split were
statistically indistinguishable from each other (<=0.005R spread) and
were fixed at whichever tested value scored marginally highest. The
runner's ATR chandelier-trail distance (`runner_trail_atr_mult`, default
2.0) is the SAME fixed assumption the re-simulation used throughout its
own runner modeling -- it was never itself swept, so treat it as
"consistent with what was validated," not "independently optimized."

READ THIS BEFORE TRUSTING THIS GEOMETRY AS "GOOD": the re-simulation's
own best result was EV = +0.0235R with a payoff ratio of only 0.2366 --
its win rate (83.46%) cleared the 80.87% breakeven rate by just 2.6
percentage points. That is a real, empirically-measured edge over 405
real trades, not a fabricated one -- but it is a thin one, plausible to
erode from fees drifting, a regime shift, or plain sampling noise. This
step's own conclusion (see the chat history around this commit) was that
further exit-geometry tuning had hit diminishing returns and the
remaining edge, if there is more to find, is on the ENTRY side (Step
11.0's own next section) -- not that this geometry is a finished,
robust edge on its own.

Step 11.2 revisits that conclusion once, after Step 11.0's own squeeze/
exhaustion entry filters (regime_filter.py/entry_ladder.py) went live and
a real production backtest under the LOCKED Step 11.0 geometry came back
EV = -0.0177R -- negative, with 73.6% of trades closing at breakeven.
Root-caused (grid_resimulation.py's own git history) to
`breakeven_buffer_atr_mult` (then 0.15) and `scratch_win_trigger_atr_mult`
(0.60) never having been swept at all, and to a real bug in the
re-simulation tool's own breakeven logic that had been silently
distorting every prior round's numbers for any tp1_atr != 0.60. Two more
re-simulation rounds, WITH that bug fixed and the scratch-win mechanic
finally swept for the first time, produced a genuinely different, better
answer: sl_atr 2.60 (tightened back down from 2.80 -- confirmed a real
interior peak, bracketed on both sides across two separate rounds, not
an edge-of-grid artifact), tp1_atr 1.10 (widened back up from 0.60) at
the same 20% weight, tp2_atr 2.80 (tightened from 3.20) at the same 50%/
30%-runner split, and `breakeven_buffer_atr_mult` widened to 0.25 (from
0.15 -- clears round-trip fees better on trades that do scratch).
`scratch_win_trigger_atr_mult` itself stayed at 0.60: loosening it was
the original hypothesis (rescue trades being "choked" before reaching
tp2) and the re-simulation showed that hypothesis was BACKWARDS --
loosening it monotonically WORSENED EV (0.60: +0.046R -> 1.50: -0.027R)
because most trades were never going to reach the far tp2 regardless of
when the stop tightened, so loosening only left more of them exposed to
the wide original stop for longer. This configuration's own best
re-simulated result: EV +0.057R, WR 84.94%, a 5.54-point margin over its
79.40% breakeven rate -- the widest margin any round has produced, still
short of a formal target but the strongest, most-validated result this
geometry has had. Verify with a real 180-day production backtest before
trusting it further; the sandbox this was developed in has no market
data access, so every number above came from the user's own machine.

Step 11.3: that verification backtest came back (263 real trades,
BTCUSDT/ETHUSDT/SOLUSDT, 180 days) at EV = -0.0086R -- still negative,
nowhere near the +0.05R target, though roughly half of Step 11.0's
-0.0177R regression. Root cause this time was NOT the SL/TP1/TP2
distances: 72.6% of trades (191/263) still closed via the breakeven-
buffer stop before ever nearing TP1 -- essentially unchanged from Step
11.0's 73.6%, meaning the wider TP1/TP2 barely mattered because most
trades never got anywhere near them. The real finding was in that
breakeven bucket itself: 46% of it (88/191) closed at a NET LOSS (as
low as -0.64R), because grid_resimulation.py's friction constant
(`FRICTION_R_AT_SL_2_20`, then 0.093) underestimated real round-trip
fees+slippage by ~40% -- the real production figure, measured directly
off this same backtest's Trades sheet, averaged ~0.111R of friction at
sl_atr=2.60, which is LARGER than the 0.25-ATR buffer's ~0.096R gross
margin at that same sl_atr. The "guaranteed small win" scratch-win
mechanic was, on average, a guaranteed small LOSS once real execution
costs were included -- pure arithmetic, not a modeling subtlety.
Step 11.3 recalibrates `FRICTION_R_AT_SL_2_20` to 0.132 (from 0.093,
so the tool's own friction assumption at sl_atr=2.60 now matches the
measured ~0.111R instead of understating it) and widens
`breakeven_buffer_atr_mult` to 0.35 (from 0.25 -- 0.35/2.60 = 0.135R
gross margin, clearing the measured ~0.111R friction with roughly
+0.024R of net cushion left over, instead of running at a net deficit).
This is a DIRECT ARITHMETIC FIX (buffer margin > measured friction),
not a re-run of the grid sweep -- the sweep itself needs to be re-run
with the corrected friction constant before its own EV/WR predictions
can be trusted again for anything beyond this one hand-computed value.
Even if it holds, this does NOT fix the underlying problem: 72.6% of
entries stalling before ever reaching TP1 is an ENTRY/regime-quality
issue, not an exit-parameter one -- see the chat history around this
commit for the Step 11.3 entry-side investigation this pivots to next.

Same conventions as the rest of this KRYPTIC track: every input DataFrame
is assumed to contain only fully CLOSED candles. `PositionState.update()`
is deliberately a pure state machine over pre-computed per-bar values
(passed in via `bar` dict) rather than something that owns a rolling
indicator pipeline itself -- the caller computes EMA20/ATR14 ONCE,
vectorized, over the whole history via indicators.py, and feeds each bar's
already-computed scalar values in one at a time. Doing it any other way
(recomputing a rolling window from scratch on every bar) would be both
slower and a much easier place to accidentally leak future data into a
"current" reading.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

import indicators as ind
from entry_ladder import EntryLadder, EntryLadderEngine

log = logging.getLogger(__name__)

Direction = Literal["LONG", "SHORT"]

TP_WEIGHTS: tuple[float, float, float, float, float] = (0.20, 0.50, 0.0, 0.0, 0.0)
"""Step 11.2 geometry, locked in from grid_resimulation.py's real-price-path
sweep: TP1 20% at 1.10 ATR, TP2 50% at 2.80 ATR, both off the real filled
VWAP -- the remaining 30% (`PositionState.runner_weight`, NOT part of this
tuple) goes to the new ATR chandelier-trail runner tier once both TP1 and
TP2 have fired. TP1+TP2 deliberately do NOT sum to 1.0 anymore (0.70,
not 1.0) -- `PositionState.__post_init__` validates
`sum(tp_weights) + runner_weight == 1.0` instead. Slots 3/4/5 stay
reserved (`None` price, 0 weight) rather than removed, so the 5-element
list shape this module and its callers have always assumed doesn't need
to change -- the runner is NOT tp_levels[2]/tp_weights[2] (a fixed price
target doesn't fit a dynamically-trailing exit; see `runner_weight`/
`runner_trail_atr_mult`/`runner_active`/`runner_extreme` instead)."""


@dataclass
class PositionState:
    """Bar-by-bar execution state machine for one open (or opening) trade.

    Sizes/weights throughout are fractions of the ladder's full intended
    position size (the same convention entry_ladder.py uses), not dollar
    amounts or contract counts -- this module doesn't know the account's
    equity or leverage, only relative sizing. `realized_pnl` is therefore a
    price-weighted, size-fraction-scaled figure (sum of
    (exit_price - weighted_avg_entry) * closed_size_fraction, sign-adjusted
    for direction), not a dollar P&L.

    Args:
        direction: "LONG" or "SHORT".
        ladder: The EntryLadder this position is scaling into (owns entry
            fill state; this class only reads/drives it, never replaces it).
        initial_sl: The hard stop computed at trade construction (kept for
            reference; never mutated after construction) -- Step 10.10:
            `expected_vwap -/+ 2.20 * ATR14` (the theoretical full-ladder
            VWAP at construction time, since nothing has filled yet).
        current_sl: The ACTIVE stop -- starts equal to initial_sl, migrates
            once (see `scratch_win_trigger_atr_mult`) to a small-guaranteed-
            win level once price has moved far enough in profit.
        tp_levels: [tp1, tp2, tp3, tp4, tp5] target prices. tp3/tp4/tp5 are
            always None -- there is no third static price target (see
            `TP_WEIGHTS`'s own docstring for why the list still has 5
            slots, and `runner_weight` below for what actually happens to
            the size these targets don't account for).
        tp_weights: Position-size fraction per TP tier (default TP_WEIGHTS,
            [0.20, 0.50, 0.0, 0.0, 0.0] -- Step 11.0 -- sums to 0.70, NOT
            1.0; the remainder is `runner_weight`, below).
        tp_fills: Per-tier completion state (True once that tier has fired,
            even if -- see `update` -- there was no size left open to
            actually close when it did).
        breakeven_moved: Whether the Step 10.10 active scratch-win stop
            migration has already run (guards against re-triggering on a
            later bar). Kept under its Step 10.6 name for cross-module
            compatibility (live_runner.py/backtest_engine.py both key off
            `breakeven_moved`/`BREAKEVEN_MOVE` to label a stop-out "BE" in
            reports) even though the migrated stop is now always a small
            WIN, never a bare breakeven -- see `breakeven_buffer_atr_mult`.
        realized_weight: Cumulative position-size fraction closed so far
            (via SL, TP, or time-decay invalidation).
        realized_pnl: Cumulative price-weighted PnL (see class docstring).
        closed: True once no more size can ever open or close (open_size is
            0 AND every entry tier is either filled or cancelled -- i.e.
            there is no live limit order that could still add size later).
        execution_log: Every event this position has ever recorded, oldest
            first. Each entry is a dict with at least "bar_index" and
            "type" (ENTRY_FILL, SL_HIT, TP_HIT, TP_HIT_NO_SIZE,
            BREAKEVEN_MOVE, CANCEL, TIME_DECAY_EXIT, TP_CLAMPED,
            TP1_INVARIANT_CLAMP, RUNNER_EXIT -- Step 11.0).
        breakeven_buffer_atr_mult: ATR multiple beyond the real filled VWAP
            the Step 10.10 scratch-win stop migrates to once triggered
            (default 0.35, Step 11.3 -- widened from Step 11.2's 0.25
            after a real production backtest showed that 0.25's ~0.096R
            gross margin at sl_atr=2.60 was actually SMALLER than the
            ~0.111R of real round-trip fees+slippage measured on that
            same backtest, i.e. the "guaranteed small win" was a
            guaranteed small LOSS on average. 0.35 ATR gives ~0.135R
            gross margin, clearing that measured friction with ~0.024R
            to spare) -- guarantees the stop-out, if it comes, clears
            round-trip taker fees and slippage as a small win rather
            than landing exactly at breakeven or worse.
        scratch_win_trigger_atr_mult: How far price must move in profit
            from the real filled VWAP, in ATR14, before the scratch-win
            stop migration fires (default 0.60, Step 10.10 -- re-swept in
            Step 11.2 and confirmed unchanged; loosening it was the
            original hypothesis for the -0.0177R production regression
            and the re-simulation showed that hypothesis was BACKWARDS,
            see the module docstring's Step 11.2 section).
        tp1_atr_mult / tp2_atr_mult: TP1/TP2's fixed distance from the real
            filled VWAP, in ATR14 (defaults 1.10 / 2.80 -- Step 11.2's
            real-price-path-optimized values, replacing Step 11.0's 0.60
            / 3.20). Not just a construction-time estimate --
            `_revalidate_tp_geometry` re-anchors both to this exact
            multiple of the REAL filled VWAP EVERY bar (Step 10.11; was
            only on a fresh fill under Step 10.10 -- see
            `update_entry_fill`'s docstring for why that missed a
            construction-time-prefilled tier), since the whole point of
            this design is "always exactly this many ATRs from wherever
            the real average entry lands," not a one-time guess.
        tp1_min_atr_mult: Floor under TP1's distance from the real filled
            VWAP, in ATR14 (default 0.30, unchanged since Step 11.0; well
            below the current 1.10 `tp1_atr_mult` so it never binds --
            kept low rather than removed so a future re-tuning of
            `tp1_atr_mult` toward 0 wouldn't silently lose this safety
            net). The effective distance is
            `max(tp1_atr_mult, tp1_min_atr_mult) * atr`, and also the
            distance the hard invariant clamp forces TP1 to if it is ever
            found on the wrong side of the real filled VWAP.
        runner_weight: Step 11.0 -- position-size fraction (default 0.30)
            reserved for the trailing-runner tier: whatever's left after
            TP1 and TP2's own weights. Unlike tp_weights, this is NOT a
            fixed price target -- see `runner_trail_atr_mult`.
        runner_trail_atr_mult: ATR chandelier-trail distance (default 2.0)
            for the runner tier, once active (see `update`'s runner step).
            The trail only ever ratchets favorably (LONG: up only; SHORT:
            down only), exactly like the breakeven-buffer migration above.
            This exact value is what grid_resimulation.py's sweep used
            throughout its own runner modeling -- it was held fixed, not
            itself swept, so "consistent with what was validated" is the
            right level of confidence, not "independently optimized."
        runner_active: Whether the runner's trailing stop has started
            tracking yet (both TP1 and TP2 have fired and size remains).
        runner_extreme: The best price (LONG: highest high; SHORT: lowest
            low) seen since the runner phase began -- the anchor the
            trailing stop trails behind. None until `runner_active`.
        time_decay_bars: If TP1 still hasn't fired after this many bars
            (default 6, i.e. 1.5h on 15m candles) AND momentum has flipped
            (see `update`'s `ema20` bar field), the position is closed out
            at market rather than left to drift toward the wide stop --
            see `update`'s TIME_DECAY_EXIT step.
    """

    direction: Direction
    ladder: EntryLadder
    initial_sl: float
    current_sl: float
    tp_levels: list[float | None]
    tp_weights: list[float] = field(default_factory=lambda: list(TP_WEIGHTS))
    tp_fills: list[bool] = field(default_factory=lambda: [False] * 5)
    breakeven_moved: bool = False
    realized_weight: float = 0.0
    realized_pnl: float = 0.0
    closed: bool = False
    execution_log: list[dict] = field(default_factory=list)
    bars_processed: int = 0
    breakeven_buffer_atr_mult: float = 0.35
    scratch_win_trigger_atr_mult: float = 0.60
    tp1_atr_mult: float = 1.10
    tp1_min_atr_mult: float = 0.30
    tp2_atr_mult: float = 2.80
    time_decay_bars: int = 6
    runner_weight: float = 0.30
    runner_trail_atr_mult: float = 2.0
    runner_active: bool = False
    runner_extreme: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("LONG", "SHORT"):
            raise ValueError('direction must be "LONG" or "SHORT"')
        for name, seq, expected_len in (
            ("tp_levels", self.tp_levels, 5), ("tp_weights", self.tp_weights, 5), ("tp_fills", self.tp_fills, 5),
        ):
            if len(seq) != expected_len:
                raise ValueError(f"{name} must have exactly {expected_len} entries, got {len(seq)}")
        if self.tp_levels[4] is not None:
            raise ValueError("tp_levels[4] must be None -- no dynamic runner PRICE target (see runner_weight)")
        total_weight = sum(self.tp_weights) + self.runner_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"tp_weights + runner_weight must sum to 1.0, got {total_weight}")

    @property
    def filled_entry_weight(self) -> float:
        """Total position-size fraction that has ever filled from the entry ladder."""
        return sum(w for w, f in zip(self.ladder.weights, self.ladder.fills) if f)

    @property
    def open_size(self) -> float:
        """Currently open position-size fraction (filled entries minus everything closed so far)."""
        return max(0.0, self.filled_entry_weight - self.realized_weight)

    @property
    def weighted_avg_entry(self) -> float | None:
        """Blended average entry price across filled entry tiers (delegates to the ladder; None if nothing has filled)."""
        return self.ladder.calculate_vwap()

    def _log(self, bar_index: int, kind: str, **kwargs) -> dict:
        entry = {"bar_index": bar_index, "type": kind, **kwargs}
        self.execution_log.append(entry)
        return entry

    def _close_weight(self, bar_index: int, kind: str, amount: float, price: float, **extra) -> None:
        avg_entry = self.weighted_avg_entry
        if avg_entry is None or amount <= 0:
            return
        sign = 1.0 if self.direction == "LONG" else -1.0
        pnl = (price - avg_entry) * amount * sign
        self.realized_weight += amount
        self.realized_pnl += pnl
        self._log(bar_index, kind, price=price, closed_weight=amount, pnl=pnl, realized_pnl_total=self.realized_pnl, **extra)

    def update_entry_fill(self, bar_index: int, newly_filled_tiers: list[int]) -> None:
        """Log newly-filled entry tiers.

        Re-anchoring TP1/TP2 no longer happens here -- see `update`'s
        Step 2a. It used to be gated on "a tier filled THIS bar," which
        left a hole: an entry tier can also arrive already filled at
        POSITION CONSTRUCTION time (a market-order Entry 1 on a fresh
        breakout -- see entry_ladder.py's `_formulate_ladder`), and that
        fill never flows through `ladder.update_fills()`/this method at
        all. TP1/TP2 would then stay priced off the construction-time
        `expected_vwap` (the full 4-tier theoretical average, pulled well
        below/above the real one-tier fill for a LONG/SHORT) forever --
        Step 10.11's "inverted TP1" bug: a target computed on the wrong
        side of the REAL filled VWAP that price could reach on a pure
        retracement, exiting at a loss while still logged as "TP1 hit."
        Step 10.11 fixes this by re-anchoring unconditionally every bar
        instead (see `update`), so a pre-filled tier is corrected on the
        very first bar processed, before that bar's own TP check runs.

        Args:
            bar_index: Current bar's index, for the execution log.
            newly_filled_tiers: Tier indices that just filled this bar (from
                `ladder.update_fills`).
        """
        for tier_i in newly_filled_tiers:
            self._log(bar_index, "ENTRY_FILL", tier_index=tier_i, price=self.ladder.levels[tier_i], weight=self.ladder.weights[tier_i])

    def _revalidate_tp_geometry(self, bar_index: int, *, atr: float | None) -> None:
        """Re-anchor TP1/TP2 to the REAL filled VWAP, every bar (Step 10.11).

        TP1 = filled_vwap +/- max(tp1_atr_mult, tp1_min_atr_mult) * atr --
        a floor under TP1's distance, not just a fixed multiple, so a
        `tp1_atr_mult` smaller than the floor would always yield at least
        `tp1_min_atr_mult` ATRs of real distance (Step 11.2's 1.10 is well
        above the 0.30 floor, so it never binds in practice).
        TP2 = filled_vwap +/- tp2_atr_mult * atr, unfloored (no observed
        inversion risk at 2.80 ATR's much larger distance).

        Then a hard invariant clamp, independent of the distance math
        above: TP1 must be strictly on the profitable side of filled_vwap
        (LONG: `tp1 > filled_vwap`; SHORT: `tp1 < filled_vwap`) or it's
        forced to `filled_vwap +/- tp1_min_atr_mult * atr`. The floor
        already guarantees this in the normal case (a positive distance in
        the direction of `sign` can never land on the wrong side) -- this
        is defense-in-depth against a future change to the distance
        formula reintroducing an inversion silently.

        Not a "clamp only if invalid" rule for the base case -- both
        targets are DEFINED as a fixed ATR distance from the real filled
        VWAP, so any VWAP change moves them with it, even if the old level
        was still technically profitable. Runs every bar (not just on a
        fresh fill) specifically to catch a tier that filled at
        construction time -- see `update_entry_fill`'s docstring."""
        filled_vwap = self.weighted_avg_entry
        if filled_vwap is None or atr is None or atr <= 0:
            return
        sign = 1.0 if self.direction == "LONG" else -1.0

        def _set_tp(i: int, new_level: float) -> None:
            old_level = self.tp_levels[i]
            if new_level == old_level:
                return
            self.tp_levels[i] = new_level
            self._log(bar_index, "TP_CLAMPED", tier_index=i, old_level=old_level, new_level=new_level, filled_vwap=filled_vwap)

        if not self.tp_fills[0] and self.tp_levels[0] is not None:
            tp1_dist = max(self.tp1_atr_mult, self.tp1_min_atr_mult) * atr
            _set_tp(0, filled_vwap + sign * tp1_dist)

        if not self.tp_fills[1] and self.tp_levels[1] is not None:
            _set_tp(1, filled_vwap + sign * self.tp2_atr_mult * atr)

        if not self.tp_fills[0] and self.tp_levels[0] is not None:
            inverted = (self.tp_levels[0] <= filled_vwap) if self.direction == "LONG" else (self.tp_levels[0] >= filled_vwap)
            if inverted:
                # Hardcoded 1.10 ATR, deliberately NOT `self.tp1_min_atr_mult`
                # (nor `self.tp1_atr_mult`, even though Step 11.2's default
                # happens to also be 1.10 -- this constant must stay fixed
                # regardless of either field's configured value): this clamp
                # exists to catch the distance math above being wrong,
                # including a misconfigured tp1_atr_mult/tp1_min_atr_mult --
                # reusing either field here would let a bad config defeat
                # both layers at once.
                old_level = self.tp_levels[0]
                self.tp_levels[0] = filled_vwap + sign * 1.10 * atr
                self._log(
                    bar_index, "TP1_INVARIANT_CLAMP",
                    old_level=old_level, new_level=self.tp_levels[0], filled_vwap=filled_vwap,
                )

    def update(self, bar: dict, htf_bar: dict | None = None) -> list[dict]:
        """Advance the state machine by one closed bar. Returns this bar's new log entries.

        Args:
            bar: {"high", "low", "close", "volume"} for one closed candle
                (required; "open" not needed here), plus these OPTIONAL
                pre-computed indicator values -- omit any of them to simply
                skip the check that needs it, rather than raising:
                  "atr": float (indicators.average_true_range(..., 14) for
                      this bar) -- used by TP1/TP2 re-anchoring and the
                      scratch-win buffer.
                  "ema20": float (indicators.ema(close, 20) for this bar)
                      -- used only by the time-decay invalidation check.
            htf_bar: Accepted for backward compatibility with callers built
                for the pre-Step-10.10 TP3 HTF-FVG-mitigation mechanism --
                that mechanism no longer exists (there is no TP3), so this
                is now unused; passing it (or not) has no effect.

        Returns:
            The list of execution_log entries appended during this call
            (empty if the position is already closed, or nothing happened).

        Execution order within one bar (all using this bar's OHLC, so this
        is bar-resolution, not tick-resolution -- see the SL-first rule
        below for how that ambiguity is resolved deterministically):
            1. Stop-loss, checked against the size open as of the START of
               this bar (a same-bar entry fill can never "un-hit" an
               already-active stop) -- the conservative convention this
               repo's own risk_guard.py already uses for the same reason.
               A hit closes ALL open size immediately and ends the bar.
            2. New entry fills (via `ladder.update_fills`), re-anchoring
               TP1/TP2 to the freshly recalculated real filled VWAP.
            2b. Active scratch-win buffer: once price has moved
               `scratch_win_trigger_atr_mult` ATRs in profit from the real
               filled VWAP, `current_sl` migrates (once) to
               `filled_vwap +/- breakeven_buffer_atr_mult * ATR` -- a
               guaranteed small win if later stopped out, never a bare
               breakeven or worse.
            3. Passive TP1/TP2 checks, in ascending order, against whatever
               is open at each check (a single wide bar can trigger both).
               Either one firing cancels any still-resting entry tier --
               once the position is in take-profit mode, a later
               retracement must never fill a resting order at worse
               momentum. TP1+TP2 weights sum to 0.70 (Step 11.0), not 1.0
               -- reaching TP2 leaves `runner_weight` still open, not zero.
            3b. Trailing runner (Step 11.0): once BOTH TP1 and TP2 have
               fired and size remains (i.e. exactly `runner_weight`, in
               the normal case), an ATR chandelier stop trails behind the
               best price seen since -- ratcheting `current_sl` favorably
               only, same convention as the breakeven buffer. A same-bar
               hit closes immediately here; a hit on a LATER bar is caught
               by step 1 above instead (see that step's own comment for
               why it's then labeled RUNNER_EXIT, not SL_HIT).
            3c. Time-decay invalidation: if TP1 still hasn't fired after
               `time_decay_bars` bars and this bar's close is on the wrong
               side of `ema20`, close out at market rather than let a
               stalled trade drift toward the wide stop.
            4. Fully-closed detection.
        """
        if self.closed:
            return []
        self.bars_processed += 1
        idx = self.bars_processed - 1
        log_start = len(self.execution_log)

        hi, lo, close = float(bar["high"]), float(bar["low"]), float(bar["close"])

        # 1. Stop-loss first, against pre-bar open size. `current_sl` also
        # IS the runner's trailing stop once ratcheted (step 3b) -- if the
        # runner phase already started on a prior bar, a hit here is
        # really the trail catching up a bar later, not the original hard
        # stop, so label it accordingly (the R math is identical either way).
        pre_bar_open_size = self.open_size
        if pre_bar_open_size > 0:
            sl_hit = (lo <= self.current_sl) if self.direction == "LONG" else (hi >= self.current_sl)
            if sl_hit:
                self._close_weight(idx, "RUNNER_EXIT" if self.runner_active else "SL_HIT", pre_bar_open_size, self.current_sl)
                self.closed = True
                return self.execution_log[log_start:]

        # 2. New entry fills.
        self.update_entry_fill(idx, self.ladder.update_fills(bar))

        # 2a. Re-anchor TP1/TP2 to the real filled VWAP -- every bar, not
        # only bars with a fresh fill, so a tier already filled at
        # construction time (e.g. an immediate market Entry 1) is
        # corrected on the very first bar processed, before this bar's own
        # TP check below can act on a stale, potentially inverted level.
        self._revalidate_tp_geometry(idx, atr=bar.get("atr"))

        # 2b. Active scratch-win buffer.
        if not self.breakeven_moved and self.open_size > 0:
            avg_entry = self.weighted_avg_entry
            atr_val = bar.get("atr")
            if avg_entry is not None and atr_val is not None and atr_val > 0:
                threshold = self.scratch_win_trigger_atr_mult * atr_val
                buffer = self.breakeven_buffer_atr_mult * atr_val
                if self.direction == "LONG":
                    reached = hi >= avg_entry + threshold
                    candidate_sl = avg_entry + buffer
                else:
                    reached = lo <= avg_entry - threshold
                    candidate_sl = avg_entry - buffer
                if reached:
                    improved = (candidate_sl > self.current_sl) if self.direction == "LONG" else (candidate_sl < self.current_sl)
                    if improved:
                        self.current_sl = candidate_sl
                    self.breakeven_moved = True
                    self._log(idx, "BREAKEVEN_MOVE", new_sl=self.current_sl, filled_vwap=avg_entry)

        # 3. Passive TP1/TP2 checks.
        for i in range(4):
            if self.tp_fills[i] or self.tp_levels[i] is None:
                continue
            level = self.tp_levels[i]
            hit = (hi >= level) if self.direction == "LONG" else (lo <= level)
            if not hit:
                continue
            self.tp_fills[i] = True
            close_amount = min(self.tp_weights[i], self.open_size)
            if close_amount <= 0:
                self._log(idx, "TP_HIT_NO_SIZE", tier_index=i, price=level)
            else:
                self._close_weight(idx, "TP_HIT", close_amount, level, tier_index=i)
            # Ladder protection: either target firing puts the position in
            # take-profit mode -- cancel any still-resting entry tier so a
            # later retracement can never fill at worse momentum than what
            # already justified taking profit.
            for cancelled in self.ladder.cancel_unfilled_orders(current_price=close, trigger_tp2=True):
                self._log(idx, "CANCEL", **cancelled)

        # 3b. Trailing runner (Step 11.0): once TP1 AND TP2 have both
        # fired and weight remains -- the runner_weight, if both fired
        # cleanly -- trail an ATR chandelier stop behind the best price
        # seen since, ratcheting favorably only (never loosens).
        if self.tp_fills[0] and self.tp_fills[1] and self.open_size > 0:
            atr_val = bar.get("atr")
            if atr_val is not None and atr_val > 0:
                sign = 1.0 if self.direction == "LONG" else -1.0
                if not self.runner_active:
                    self.runner_active = True
                    self.runner_extreme = close
                self.runner_extreme = max(self.runner_extreme, hi) if self.direction == "LONG" else min(self.runner_extreme, lo)
                trail_stop = self.runner_extreme - sign * self.runner_trail_atr_mult * atr_val
                improved = (trail_stop > self.current_sl) if self.direction == "LONG" else (trail_stop < self.current_sl)
                if improved:
                    self.current_sl = trail_stop
                trail_hit = (lo <= self.current_sl) if self.direction == "LONG" else (hi >= self.current_sl)
                if trail_hit:
                    self._close_weight(idx, "RUNNER_EXIT", self.open_size, self.current_sl)
                    self.closed = True
                    return self.execution_log[log_start:]

        # 3c. Time-decay invalidation.
        if not self.tp_fills[0] and self.open_size > 0 and self.bars_processed >= self.time_decay_bars:
            ema20 = bar.get("ema20")
            if ema20 is not None:
                against = (close < ema20) if self.direction == "LONG" else (close > ema20)
                if against:
                    self._close_weight(idx, "TIME_DECAY_EXIT", self.open_size, close)
                    self.closed = True
                    for cancelled in self.ladder.cancel_unfilled_orders(current_price=close, trigger_tp2=True):
                        self._log(idx, "CANCEL", **cancelled)
                    return self.execution_log[log_start:]

        # 4. Fully closed: nothing open, and no pending entry order could ever add more.
        no_pending_entries = all(f or c for f, c in zip(self.ladder.fills, self.ladder.cancelled))
        if self.open_size <= 1e-9 and no_pending_entries:
            self.closed = True

        return self.execution_log[log_start:]


class TradeLifecycleManager:
    """Builds a fully-specified PositionState from an EntryLadder.

    Args:
        entry_ladder_engine: Injected EntryLadderEngine (default: a fresh one).
        atr_length: ATR length for SL/TP sizing (default 14).
        initial_sl_atr_mult: `initial_sl` distance from `expected_vwap`, in
            ATR14 (default 2.60, Step 11.2 -- tightened back down from
            Step 11.0's 2.80; grid_resimulation.py's re-simulation
            re-confirmed 2.60 as a genuine interior optimum, bracketed on
            both sides across two separate rounds, not an edge-of-grid
            artifact -- see the module docstring's Step 11.2 section). A
            deliberately wide "noise stop" meant to sit outside ordinary
            15m chop, trading a bigger loss on the (now rarer) full
            stop-outs for a much higher hit rate on the (now closer)
            targets below.
        tp1_atr_mult / tp2_atr_mult: TP1/TP2's distance from
            `expected_vwap` at construction (defaults 1.10 / 2.80, Step
            11.2 -- both real-price-path-optimized, replacing Step 11.0's
            0.60 / 3.20). `PositionState`'s own copy of these same values
            re-anchors both to the REAL filled VWAP every bar at runtime
            -- see `_revalidate_tp_geometry`.
        tp1_min_atr_mult: Floor under TP1's real-filled-VWAP distance, in
            ATR14 (default 0.30, unchanged since Step 11.0 -- well below
            the current 1.10 `tp1_atr_mult` so it never binds) -- passed
            straight through to each PositionState.
        runner_trail_atr_mult: ATR chandelier-trail distance for the
            Step 11.0 runner tier (default 2.0, the same fixed assumption
            grid_resimulation.py's sweep used throughout -- see
            `PositionState.runner_trail_atr_mult`'s own docstring for why
            that's "consistent with what was validated," not
            "independently optimized") -- passed straight through.
        breakeven_buffer_atr_mult: Passed straight through to each
            PositionState for the Step 10.10 active scratch-win stop
            migration's target margin beyond the real filled VWAP
            (default 0.35, Step 11.3 -- widened from 0.25 after a real
            production backtest showed 0.25's gross margin was smaller
            than measured real friction, i.e. a net loss on average --
            see the module docstring's Step 11.3 section for the numbers).
        scratch_win_trigger_atr_mult: Passed straight through -- how far
            price must move in profit before the scratch-win migration
            fires (default 0.60 -- re-swept in Step 11.2 and confirmed
            unchanged; see the module docstring's Step 11.2 section for
            why loosening it made things worse, not better).
        time_decay_bars: Passed straight through -- bars TP1 has to fire
            before an unmoved position gets invalidated on an EMA20
            momentum flip (default 6).
    """

    def __init__(
        self,
        *,
        entry_ladder_engine: EntryLadderEngine | None = None,
        atr_length: int = 14,
        initial_sl_atr_mult: float = 2.60,
        tp1_atr_mult: float = 1.10,
        tp1_min_atr_mult: float = 0.30,
        tp2_atr_mult: float = 2.80,
        runner_trail_atr_mult: float = 2.0,
        breakeven_buffer_atr_mult: float = 0.35,
        scratch_win_trigger_atr_mult: float = 0.60,
        time_decay_bars: int = 6,
    ) -> None:
        self.entry_ladder_engine = entry_ladder_engine or EntryLadderEngine()
        self.atr_length = atr_length
        self.initial_sl_atr_mult = initial_sl_atr_mult
        self.tp1_atr_mult = tp1_atr_mult
        self.tp1_min_atr_mult = tp1_min_atr_mult
        self.tp2_atr_mult = tp2_atr_mult
        self.runner_trail_atr_mult = runner_trail_atr_mult
        self.breakeven_buffer_atr_mult = breakeven_buffer_atr_mult
        self.scratch_win_trigger_atr_mult = scratch_win_trigger_atr_mult
        self.time_decay_bars = time_decay_bars

    def open_trade(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame,
        funding_rate: float | None = None,
        htf_df: pd.DataFrame | None = None,
    ) -> tuple[PositionState | None, dict]:
        """Build a ladder (via EntryLadderEngine), then price the Step
        11.0 geometry directly off its theoretical VWAP: 2 static targets
        (TP1/TP2) plus a trailing runner for whatever weight is left.

        Returns:
            (position, diagnostics). `position` is None if the entry
            ladder itself couldn't be built; `diagnostics["reason"]`
            explains why. On success, `diagnostics` also carries
            `initial_sl`, `expected_vwap`, `risk_atr_multiple` (always
            exactly `initial_sl_atr_mult` by construction now), and
            `tp_levels`.
        """
        ladder, ladder_diag = self.entry_ladder_engine.build_ladder(df, btc_df, funding_rate, htf_df=htf_df)
        diagnostics: dict = {"opened": False, "reason": None, "ladder_diagnostics": ladder_diag}
        if ladder is None:
            diagnostics["reason"] = ladder_diag["reason"]
            return None, diagnostics

        direction = ladder.direction
        high, low, close = df["high"], df["low"], df["close"]
        atr = ind.average_true_range(high, low, close, length=self.atr_length)
        last_atr = atr.iloc[-1]
        if pd.isna(last_atr) or last_atr <= 0:
            diagnostics["reason"] = "ATR not computable for SL/TP sizing"
            return None, diagnostics
        last_atr = float(last_atr)

        expected_vwap = sum(level * weight for level, weight in zip(ladder.levels, ladder.weights))
        sign = 1.0 if direction == "LONG" else -1.0
        initial_sl = expected_vwap - sign * self.initial_sl_atr_mult * last_atr

        # Step 10.11: if a tier is already filled at construction time (an
        # immediate market Entry 1 -- see entry_ladder.py's
        # `_formulate_ladder`), price TP1/TP2 off the REAL filled VWAP from
        # the start, not the theoretical `expected_vwap` -- the latter
        # blends in still-unfilled, deeper retracement tiers and, for a
        # LONG, sits BELOW a lone Entry-1 fill (mirror for SHORT), which
        # could put TP1 on the wrong side of the real average entry before
        # `PositionState.update()` even runs its first bar. When nothing
        # has filled yet, `expected_vwap` is the only anchor available; the
        # first real fill's `_revalidate_tp_geometry` call corrects it.
        tp_anchor_vwap = ladder.calculate_vwap() if ladder.has_active_fills else expected_vwap
        tp1_dist = max(self.tp1_atr_mult, self.tp1_min_atr_mult) * last_atr
        tp1 = tp_anchor_vwap + sign * tp1_dist
        tp2 = tp_anchor_vwap + sign * self.tp2_atr_mult * last_atr
        risk_atr_multiple = abs(expected_vwap - initial_sl) / last_atr
        diagnostics.update(initial_sl=initial_sl, expected_vwap=expected_vwap, risk_atr_multiple=risk_atr_multiple)

        position = PositionState(
            direction=direction, ladder=ladder,
            initial_sl=initial_sl, current_sl=initial_sl,
            tp_levels=[tp1, tp2, None, None, None],
            breakeven_buffer_atr_mult=self.breakeven_buffer_atr_mult,
            scratch_win_trigger_atr_mult=self.scratch_win_trigger_atr_mult,
            tp1_atr_mult=self.tp1_atr_mult, tp1_min_atr_mult=self.tp1_min_atr_mult, tp2_atr_mult=self.tp2_atr_mult,
            runner_trail_atr_mult=self.runner_trail_atr_mult,
            time_decay_bars=self.time_decay_bars,
        )
        diagnostics["opened"] = True
        diagnostics["tp_levels"] = list(position.tp_levels)
        return position, diagnostics
