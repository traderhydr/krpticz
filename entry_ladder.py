"""entry_ladder.py -- 4-tier scale-in entry engine for KRYPTIC (Step 4).

`EntryLadderEngine` turns a qualifying setup into a concrete 4-price scale-in
ladder: an immediate/shallow tier, an OTE (Optimal Trade Entry) Fibonacci
tier, a Fair-Value-Gap-midpoint tier gated by order flow, and a deep
liquidity-sweep/volatility tier. It only builds a ladder once both
prerequisite engines agree there's something to trade:

    1. RegimeFilter.evaluate_market_conditions(...) -> allowed == True
    2. DirectionEngine.get_directional_bias(...) -> "LONG" or "SHORT"

(DirectionEngine necessarily runs first here, even though the spec's own
prose lists RegimeFilter first: RegimeFilter's `direction` parameter -- see
regime_filter.py -- is only known once DirectionEngine has picked a side, so
there is no other viable evaluation order.)

Then it locates a qualifying impulse leg (a confirmed swing-low-to-swing-high
run for LONG, swing-high-to-swing-low for SHORT, or -- when no such
structural leg is available yet -- a single breakout expansion candle) and
prices the four tiers off it.

Same conventions as the rest of this KRYPTIC track: every input DataFrame
is assumed to contain only fully CLOSED candles, and "price" means "the
last closed candle's close" unless stated otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

import indicators as ind
from directional_bias import DirectionEngine
from regime_filter import RegimeFilter

Direction = Literal["LONG", "SHORT"]

LADDER_WEIGHTS: tuple[float, float, float, float] = (0.40, 0.30, 0.20, 0.10)
"""All 4 tiers active, front-loaded toward the shallower fills (matching
ZENITH's/GEM's own ENTRY_WEIGHTS front-loading, for a consistent ladder
shape across all three engines): Entry 1 dominant (breakout/FVG-proximal,
0.40), Entry 2 a shallow 0.382-0.50 retracement (0.30), Entry 3 a deeper
0.618-0.705 retracement (0.20), Entry 4 the origin-sweep/ATR-band tier
(0.10) -- previously 0-weighted ("effectively disabled") since this
profile doesn't wait for a full liquidity sweep back to the impulse
origin; kept active now so a real liquidity sweep back to origin still
adds size instead of being ignored."""
ENTRY3_TIER_INDEX = 2  # the bar-delta-gated tier, fixed by the 4-tier spec


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}")


@dataclass(frozen=True)
class ImpulseLeg:
    """A directional price swing used to anchor the ladder's Fibonacci/FVG math.

    origin_index/terminal_index are POSITIONAL (iloc-style) indices into the
    DataFrame the leg was found in -- the actual bar where each extreme
    printed, not a pandas index label.
    """

    origin_price: float
    origin_index: int
    terminal_price: float
    terminal_index: int
    source: Literal["swing_pivots", "expansion_candle"]
    fresh_break_on_last_bar: bool


@dataclass
class EntryLadder:
    """A 4-tier scale-in order ladder and its live fill/cancellation state.

    Args:
        direction: "LONG" or "SHORT".
        levels: [entry_1, entry_2, entry_3, entry_4] prices.
        weights: Position-size fraction per tier (default LADDER_WEIGHTS,
            [0.15, 0.25, 0.35, 0.25] -- sums to 1.0).
        fills: Per-tier fill state. Entry 1 may already be True at
            construction time if it was built as an immediate market order
            (see EntryLadderEngine._formulate_ladder); all others start
            unfilled.
        cancelled: Per-tier cancellation state (see `cancel_unfilled_orders`).
            Not part of the original 4-field spec, but needed to make
            cancellation idempotent across repeated calls.
        entry_types: Human-readable label per tier (e.g. "market_breakout",
            "limit_fvg_outer", "market_fallback_no_fvg", "entry2_fib",
            "entry3_fib", "origin_sweep", "atr_band") -- diagnostic only.
    """

    direction: Direction
    levels: list[float]
    weights: list[float] = field(default_factory=lambda: list(LADDER_WEIGHTS))
    fills: list[bool] = field(default_factory=lambda: [False, False, False, False])
    cancelled: list[bool] = field(default_factory=lambda: [False, False, False, False])
    entry_types: list[str] = field(default_factory=lambda: ["", "", "", ""])

    def __post_init__(self) -> None:
        if self.direction not in ("LONG", "SHORT"):
            raise ValueError('direction must be "LONG" or "SHORT"')
        for name, seq in (("levels", self.levels), ("weights", self.weights), ("fills", self.fills), ("cancelled", self.cancelled)):
            if len(seq) != 4:
                raise ValueError(f"{name} must have exactly 4 entries, got {len(seq)}")
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {sum(self.weights)}")

    @property
    def has_active_fills(self) -> bool:
        """Whether any tier that actually carries position size (`weight >
        0`) has filled -- as opposed to a tier merely being marked
        `fills[i] = True` (e.g. the calibrated [0.40, 0.35, 0.25, 0.0]
        ladder's 0-weight Entry 4, "effectively disabled" per
        LADDER_WEIGHTS' own docstring). `update_fills`'s direction-aware,
        one-sided touch check makes a real (positive-weight) fill always
        accompany a deep one now (see its own docstring), so this should
        never be False while `fills` has any True in it in practice -- kept
        as an explicit, cheap, independently-checkable invariant rather
        than trusting that fact silently."""
        return any(f and w > 0 for w, f in zip(self.weights, self.fills))

    def calculate_vwap(self) -> float | None:
        """Volume(size)-weighted average entry price across FILLED tiers only.

        "Volume" here means each tier's ladder weight (the fraction of total
        position size it represents), not exchange trade volume -- this is
        the blended average entry price a scale-in position actually holds.

        Returns:
            The weighted average price, or None if `has_active_fills` is
            False (no tier has filled yet, or -- structurally prevented by
            `update_fills` now, but still guarded here defensively --
            every filled tier carries 0 weight). Dividing by a zero total
            weight previously crashed here; this treats that state exactly
            like "nothing has filled yet", which is what it actually is,
            economically.
        """
        if not self.has_active_fills:
            return None
        filled = [(lvl, w) for lvl, w, f in zip(self.levels, self.weights, self.fills) if f]
        total_weight = sum(w for _, w in filled)
        return sum(lvl * w for lvl, w in filled) / total_weight

    def update_fills(self, bar: dict, *, require_bar_delta_for_entry3: bool = True) -> list[int]:
        """Check one new closed bar against every unfilled, uncancelled tier.

        Args:
            bar: {"high": float, "low": float, "close": float, "volume": float}
                for one closed candle (a "open" key is not needed).
            require_bar_delta_for_entry3: If True (default), tier index
                `ENTRY3_TIER_INDEX` (Entry 3, the 0.618-0.705 retracement
                tier) only fills when price traded through its level AND that
                same bar's Close-Location-Value order-flow proxy
                (`indicators.bar_delta`'s formula, computed inline here for
                a single bar) is supportive of the trade direction --
                positive for LONG, negative for SHORT. Entries 1, 2, and 4
                fill on price alone.

        Returns:
            Ascending list of tier indices (0-3) newly filled by this bar.

        Fill test is direction-aware and one-sided, not "does this bar's
        own high/low straddle the level": a resting LONG limit buy at
        `level` fills once `bar.low <= level` -- price reached down to or
        past it, whether or not this SPECIFIC bar's high also happens to
        still be above it (SHORT is the mirror: `bar.high >= level`).
        Checking `low <= level <= high` instead (requiring the level to sit
        WITHIN this one bar's own range) used to let a single sharp bar gap
        straight past shallower tiers -- its high/low never straddling
        them -- while still registering a fill on a deeper one alone. Since
        levels are always constructed monotonically (Entry 1 shallowest ..
        Entry 4 deepest), the fixed check guarantees that filling any tier
        also fills every shallower one in the SAME bar (Entry 3's own extra
        bar_delta gate can still block it alone; Entries 1/2/4 carry no
        such gate), which is what makes a real, non-zero-weight fill always
        accompany a deep one -- see EntryLadder.calculate_vwap()'s
        docstring for the crash this was closing off.
        """
        hi, lo, close, volume = float(bar["high"]), float(bar["low"]), float(bar["close"]), float(bar["volume"])
        rng = hi - lo
        delta = volume * (2.0 * (close - lo) / rng - 1.0) if rng > 0 else 0.0

        newly_filled: list[int] = []
        for i, (level, filled, cancelled) in enumerate(zip(self.levels, self.fills, self.cancelled)):
            if filled or cancelled:
                continue
            touched = (lo <= level) if self.direction == "LONG" else (hi >= level)
            if not touched:
                continue
            if i == ENTRY3_TIER_INDEX and require_bar_delta_for_entry3:
                supportive = delta > 0 if self.direction == "LONG" else delta < 0
                if not supportive:
                    continue
            self.fills[i] = True
            newly_filled.append(i)
        return newly_filled

    def cancel_unfilled_orders(self, current_price: float, trigger_tp2: bool) -> list[dict]:
        """Cancel every still-unfilled, not-already-cancelled tier once TP2 triggers.

        This is the fix for the "ghost limit order" failure mode the spec
        describes: price can rally straight to TP2 using only the tiers that
        DID fill, without ever pulling back far enough to reach Entry 3/4.
        Left resting, those stale limit orders could fill much later on an
        unrelated retrace, opening a position with no relationship to the
        original setup. `trigger_tp2=True` clears all of them at once.

        Args:
            current_price: Latest price. Recorded on each cancelled tier's
                returned diagnostic (distance from that tier's level) for
                logging/audit purposes -- it does not gate WHICH tiers get
                cancelled. `trigger_tp2` is the sole trigger: a tier resting
                unfilled simply because price hasn't reached it yet is
                exactly the ghost order this method exists to clear once the
                trade has already achieved TP2 without it.
            trigger_tp2: Whether TP2 has been reached on the position built
                from whichever tiers filled.

        Returns:
            List of {"tier_index", "level", "distance_from_current"} for
            each tier newly cancelled by this call (empty if trigger_tp2 is
            False or there was nothing left to cancel).
        """
        if not trigger_tp2:
            return []
        newly_cancelled = []
        for i, (level, filled, cancelled) in enumerate(zip(self.levels, self.fills, self.cancelled)):
            if filled or cancelled:
                continue
            self.cancelled[i] = True
            newly_cancelled.append({"tier_index": i, "level": level, "distance_from_current": current_price - level})
        return newly_cancelled


class EntryLadderEngine:
    """Builds a 4-tier EntryLadder from OHLCV data, gated on regime + direction.

    Args:
        regime_filter: Injected RegimeFilter (default: a fresh `RegimeFilter()`).
        direction_engine: Injected DirectionEngine (default: a fresh `DirectionEngine()`).
        swing_left / swing_right: Swing pivot window used to find the
            structural impulse leg on `df`'s own (working) timeframe
            (default 5/5 -- more responsive than DirectionEngine's HTF
            default of 2/2, since this is entry timing, not trend bias).
        leg_displacement_atr_mult: Optional ATR-multiple displacement filter
            for the "fresh breakout/CHoCH" check that drives Entry 1's
            market-order path (default 0.0 = any break counts).
        atr_length: ATR length used for the displacement filter, the
            expansion-candle fallback, and Entry 4's ATR band (default 14).
        expansion_lookback: How many of the most recent bars to scan for a
            qualifying breakout expansion candle when no structural impulse
            leg is available (default 10).
        expansion_atr_mult: A bar's (high-low) range must be at least this
            many multiples of ATR to count as an expansion candle (default 1.5).
        entry2_retracement_low / entry2_retracement_high: Entry 2's
            Fibonacci retracement band (default 0.382/0.50 -- the
            high-frequency shallow-fill calibration; a deeper 0.705/0.790
            OTE band was this parameter's original value). Entry 2 is
            priced at the band's midpoint.
        entry3_retracement_low / entry3_retracement_high: Entry 3's
            Fibonacci retracement band (default 0.618/0.705). Entry 3 is
            priced at the band's midpoint -- a pure retracement level, not
            tied to any FVG.
        atr_band_mult: ATR multiple for Entry 4's volatility-buffer
            alternative (default 2.0, i.e. close -/+ 2*ATR14).
        max_leg_age_bars: A swing-pivot-based leg whose more recent pivot
            (by the bar it actually printed on, not its confirmation bar)
            is older than this many bars is rejected as stale (default 50)
            -- it falls back to the expansion-candle path instead of
            anchoring the ladder to structure with no bearing on the
            current move. See `_leg_passes_sanity_checks`.
        leg_proximity_atr_mult: How far price is allowed to have moved past
            the leg's far edge before the leg is considered disconnected
            from current price (default 1.0 ATR14). See
            `_leg_passes_sanity_checks`.
        leg_min_displacement_atr_mult: Minimum leg range (high-low), in
            ATR14, below which the leg is too small to build a meaningful
            retracement ladder off of (default 1.0). See
            `_leg_passes_sanity_checks`.
        exhaustion_max_percentile: Step 11.0 candle-exhaustion filter.
            Rejects a FRESH-BREAKOUT market entry (Entry 1 priced via
            `leg.fresh_break_on_last_bar` -- either a confirmed BOS/CHoCH
            on the last bar, or the expansion-candle fallback) whose
            trigger bar's close sits past this fraction of its own
            impulse leg's [low, high] range (>1.0 means "beyond the leg's
            own far edge by that fraction of the leg's size" -- see
            `_trigger_bar_close_location`, whose value is NOT capped at
            1.0). Only applies to the market-order path; a resting
            FVG-limit or Fibonacci-retracement entry is unaffected (those
            are never "chasing" a trigger bar by construction).

            Default 1.5, deliberately LOOSE, not the "upper quartile"
            (0.75) the concept's own name suggests -- checking this
            repo's own proven bullish cross-check fixture
            (test_backtest_engine.py's `make_candles(900, seed=1)`) found
            its one real, production-default-qualifying signal already
            had a trigger-bar CLV of 1.18: a confirmed BOS/CHoCH fires
            some bars AFTER the swing pivot that defines the leg, so by
            the time the break is confirmed, price has typically ALREADY
            moved past the leg's own high -- that's a normal confirmation
            lag, not necessarily true exhaustion, and a literal quartile
            cutoff would have silently rejected essentially every clean
            breakout signal. This is the SAME failure mode `stretch_atr_max`
            hit in Step 10.8 and `squeeze_max_percentile` was just found to
            risk too (see regime_filter.py's own docstring) -- treat this
            default as equally unvalidated until checked the same way: a
            real backtest, watching total_trades before and after.
        long_conviction_min: Step 11.3 candle-conviction gate, LONG ONLY.
            Rejects ANY long signal (not just a fresh-breakout market
            entry -- unlike `exhaustion_max_percentile` above, this
            doesn't need an impulse leg) whose trigger bar's own close
            sits below this fraction of ITS OWN [low, high] range
            (`_signal_bar_conviction`; 1.0 = closed exactly at the bar's
            high). entry_diagnostics.py's real-price-path profile of 260
            frozen trades under the Step 11.3 settled exit geometry found
            LONG outcomes separating cleanly and monotonically by this
            metric -- EV climbing from +0.071R unfiltered to +0.176R /
            profit factor 2.95 at a 0.70 cutoff (n=46 kept) -- while the
            identical metric showed NO separating power for SHORT
            outcomes at any threshold (SHORT's own baseline EV was
            already negative, -0.021R, independent of conviction); this
            gate is LONG-only by design, not a directional oversight.

            Default is 0.40, NOT that empirically-best-looking 0.70,
            following the exact same precedent as `stretch_atr_max`
            (Step 10.8) and `exhaustion_max_percentile`/
            `squeeze_max_percentile` above: checking 0.70 against this
            repo's own canonical bullish cross-check fixture
            (test_backtest_engine.py's `make_candles(900, seed=1)`)
            found its ONE real, all-other-gates-qualifying LONG signal
            has a conviction of only 0.4425 -- 0.70 would have zeroed
            out the single known-good signal this whole test suite
            cross-checks against. 0.40 isn't just "loose enough to
            survive that check" as a compromise, though -- on the same
            260-trade real data, it independently keeps EV at +0.111R /
            profit factor 2.03 while retaining 97/136 (71%) of LONG
            volume and only 2 real winners rejected, vs. 46/136 (34%)
            volume and +0.176R at 0.70. Both are real points on the same
            monotonic curve; 0.40 trades some of the peak EV for a much
            larger surviving sample and for not being contradicted by
            the one fixture this whole test suite already treats as
            "definitely a real signal." Confirm with a real production
            backtest (watching total_trades before/after, same as every
            other gate in this file) before ever raising this toward 0.70."""

    def __init__(
        self,
        *,
        regime_filter: RegimeFilter | None = None,
        direction_engine: DirectionEngine | None = None,
        swing_left: int = 5,
        swing_right: int = 5,
        leg_displacement_atr_mult: float = 0.0,
        atr_length: int = 14,
        expansion_lookback: int = 10,
        expansion_atr_mult: float = 1.5,
        entry2_retracement_low: float = 0.382,
        entry2_retracement_high: float = 0.50,
        entry3_retracement_low: float = 0.618,
        entry3_retracement_high: float = 0.705,
        atr_band_mult: float = 2.0,
        max_leg_age_bars: int = 50,
        leg_proximity_atr_mult: float = 1.0,
        leg_min_displacement_atr_mult: float = 1.0,
        exhaustion_max_percentile: float = 1.5,
        long_conviction_min: float = 0.40,
    ) -> None:
        self.regime_filter = regime_filter or RegimeFilter()
        self.direction_engine = direction_engine or DirectionEngine()
        self.swing_left = swing_left
        self.swing_right = swing_right
        self.leg_displacement_atr_mult = leg_displacement_atr_mult
        self.atr_length = atr_length
        self.expansion_lookback = expansion_lookback
        self.expansion_atr_mult = expansion_atr_mult
        self.entry2_retracement_low = entry2_retracement_low
        self.entry2_retracement_high = entry2_retracement_high
        self.entry3_retracement_low = entry3_retracement_low
        self.entry3_retracement_high = entry3_retracement_high
        self.atr_band_mult = atr_band_mult
        self.max_leg_age_bars = max_leg_age_bars
        self.leg_proximity_atr_mult = leg_proximity_atr_mult
        self.leg_min_displacement_atr_mult = leg_min_displacement_atr_mult
        self.exhaustion_max_percentile = exhaustion_max_percentile
        self.long_conviction_min = long_conviction_min

    # -- impulse leg discovery ----------------------------------------------

    def _leg_passes_sanity_checks(self, leg: ImpulseLeg, direction: Direction, current_price: float, atr: float, n_bars: int) -> bool:
        """Reject a candidate leg that's stale, too small, or disconnected
        from current price -- the hotfix for `_find_impulse_leg` picking a
        distant/historical swing pair whose retracement math has no
        relationship to where price actually is right now (which showed up
        downstream as a TP1 on the wrong side of the entry price).

        Applied uniformly to BOTH the swing-pivot and expansion-candle
        candidates (the latter is far less likely to fail this in practice,
        since it's already confined to a short recent lookback, but running
        the same check on both paths is cheap and removes a class of
        degenerate edge cases rather than trusting one path's construction
        to make it structurally impossible).
        """
        age_bars = (n_bars - 1) - max(leg.origin_index, leg.terminal_index)
        if age_bars > self.max_leg_age_bars:
            return False

        leg_low = min(leg.origin_price, leg.terminal_price)
        leg_high = max(leg.origin_price, leg.terminal_price)
        if leg_high - leg_low < self.leg_min_displacement_atr_mult * atr:
            return False

        margin = self.leg_proximity_atr_mult * atr
        if direction == "LONG":
            if not (leg_low < current_price):
                return False
            if not (current_price <= leg_high + margin):
                return False
        else:
            if not (leg_high > current_price):
                return False
            if not (current_price >= leg_low - margin):
                return False
        return True

    def _find_impulse_leg(self, df: pd.DataFrame, direction: Direction) -> ImpulseLeg | None:
        high, low, close = df["high"], df["low"], df["close"]
        n = len(df)
        min_bars = self.swing_left + 2 * self.swing_right + 2
        fresh_break = False

        atr_series = ind.average_true_range(high, low, close, length=self.atr_length)
        last_atr = atr_series.iloc[-1]
        current_price = float(close.iloc[-1])
        atr_available = not pd.isna(last_atr) and last_atr > 0

        if n >= min_bars:
            pivots = ind.detect_swing_pivots(high, low, left_bars=self.swing_left, right_bars=self.swing_right)
            atr_for_displacement = atr_series if self.leg_displacement_atr_mult > 0 else None
            events = ind.detect_structure_breaks(
                close, pivots.pivot_high, pivots.pivot_low,
                atr=atr_for_displacement, displacement_atr_mult=self.leg_displacement_atr_mult,
            )
            fresh_break = bool(
                (events.bos_up.iloc[-1] or events.choch_up.iloc[-1]) if direction == "LONG"
                else (events.bos_down.iloc[-1] or events.choch_down.iloc[-1])
            )

            if direction == "LONG":
                terminal_price_s, terminal_src_s = pivots.pivot_high.dropna(), pivots.pivot_high_source_index.dropna()
                origin_price_s, origin_src_s = pivots.pivot_low.dropna(), pivots.pivot_low_source_index.dropna()
            else:
                terminal_price_s, terminal_src_s = pivots.pivot_low.dropna(), pivots.pivot_low_source_index.dropna()
                origin_price_s, origin_src_s = pivots.pivot_high.dropna(), pivots.pivot_high_source_index.dropna()

            if len(terminal_price_s) and len(origin_price_s):
                terminal_price = float(terminal_price_s.iloc[-1])
                terminal_src_idx = terminal_src_s.iloc[-1]
                valid_origin_mask = origin_src_s < terminal_src_idx
                if valid_origin_mask.any():
                    origin_price = float(origin_price_s[valid_origin_mask].iloc[-1])
                    origin_src_idx = origin_src_s[valid_origin_mask].iloc[-1]
                    candidate = ImpulseLeg(
                        origin_price=origin_price, origin_index=int(df.index.get_loc(origin_src_idx)),
                        terminal_price=terminal_price, terminal_index=int(df.index.get_loc(terminal_src_idx)),
                        source="swing_pivots", fresh_break_on_last_bar=fresh_break,
                    )
                    if atr_available and self._leg_passes_sanity_checks(candidate, direction, current_price, float(last_atr), n):
                        return candidate
                    # Stale, too small, or disconnected from current price --
                    # fall through to the expansion-candle path below rather
                    # than anchor the ladder to irrelevant structure.

        # Fallback: most recent qualifying breakout expansion candle.
        lookback = min(self.expansion_lookback, n)
        opens = df["open"] if "open" in df.columns else close.shift(1)
        for i in range(n - 1, n - 1 - lookback, -1):
            if i < 1 or pd.isna(atr_series.iloc[i]) or atr_series.iloc[i] <= 0:
                continue
            bar_range = high.iloc[i] - low.iloc[i]
            if bar_range < self.expansion_atr_mult * atr_series.iloc[i]:
                continue
            is_bullish = close.iloc[i] > opens.iloc[i]
            candidate = None
            if direction == "LONG" and is_bullish:
                candidate = ImpulseLeg(
                    origin_price=float(low.iloc[i]), origin_index=i,
                    terminal_price=float(high.iloc[i]), terminal_index=i,
                    source="expansion_candle", fresh_break_on_last_bar=(i == n - 1),
                )
            elif direction == "SHORT" and not is_bullish:
                candidate = ImpulseLeg(
                    origin_price=float(high.iloc[i]), origin_index=i,
                    terminal_price=float(low.iloc[i]), terminal_index=i,
                    source="expansion_candle", fresh_break_on_last_bar=(i == n - 1),
                )
            if candidate is not None and atr_available and self._leg_passes_sanity_checks(candidate, direction, current_price, float(last_atr), n):
                return candidate
        return None

    # -- ladder pricing ------------------------------------------------------

    def _formulate_ladder(self, df: pd.DataFrame, direction: Direction, leg: ImpulseLeg) -> tuple[EntryLadder, dict | None]:
        high, low, close = df["high"], df["low"], df["close"]
        last_close = float(close.iloc[-1])

        fvg_events = ind.detect_fair_value_gaps(high, low)
        wanted_direction = "bullish" if direction == "LONG" else "bearish"
        active = ind.active_fvg_zones_asof(fvg_events, as_of_index=df.index[-1])
        active = active[active["direction"] == wanted_direction]

        fvg_row = None
        if len(active):
            leg_start_label = df.index[min(leg.origin_index, leg.terminal_index)]
            leg_end_label = df.index[max(leg.origin_index, leg.terminal_index)]
            within_leg = active[(active["formed_at"] >= leg_start_label) & (active["formed_at"] <= leg_end_label)]
            pool = within_leg if len(within_leg) else active
            fvg_row = pool.iloc[-1]

        # Entry 1: immediate market order on a fresh breakout, else a resting
        # limit at the FVG's outer (shallower, first-touched) edge, else a
        # plain market fallback when neither condition is available.
        if leg.fresh_break_on_last_bar:
            entry1, entry1_type, entry1_prefilled = last_close, "market_breakout", True
        elif fvg_row is not None:
            entry1 = float(fvg_row["upper"] if direction == "LONG" else fvg_row["lower"])
            entry1_type, entry1_prefilled = "limit_fvg_outer", False
        else:
            entry1, entry1_type, entry1_prefilled = last_close, "market_fallback_no_fvg", True

        # Entry 2: shallow retracement band midpoint (0.382-0.50 by default).
        entry2_band = ind.fibonacci_levels(
            leg.origin_price, leg.terminal_price,
            retracements=(self.entry2_retracement_low, self.entry2_retracement_high), extensions=(),
        )["retracements"]
        entry2 = (entry2_band[self.entry2_retracement_low] + entry2_band[self.entry2_retracement_high]) / 2.0

        # Entry 3: deeper retracement band midpoint (0.618-0.705 by default)
        # -- a pure retracement level, independent of any FVG (unlike Entry 1,
        # which does use one when available).
        entry3_band = ind.fibonacci_levels(
            leg.origin_price, leg.terminal_price,
            retracements=(self.entry3_retracement_low, self.entry3_retracement_high), extensions=(),
        )["retracements"]
        entry3 = (entry3_band[self.entry3_retracement_low] + entry3_band[self.entry3_retracement_high]) / 2.0
        entry3_type = "entry3_fib"

        # Entry 4: the deeper of the impulse-origin sweep level and the 2.0
        # ATR volatility band -- whichever offers more room, since either one
        # alone can be too shallow (a tight origin) or too wide (in a
        # low-volatility regime) on its own.
        atr = ind.average_true_range(high, low, close, length=self.atr_length)
        last_atr = atr.iloc[-1]
        if pd.isna(last_atr):
            entry4, entry4_type = leg.origin_price, "origin_sweep_only_no_atr"
        else:
            bands = ind.atr_dynamic_bands(close, atr, multiplier=self.atr_band_mult)
            band_price = float(bands["lower_band"].iloc[-1] if direction == "LONG" else bands["upper_band"].iloc[-1])
            if direction == "LONG":
                entry4 = min(leg.origin_price, band_price)
            else:
                entry4 = max(leg.origin_price, band_price)
            entry4_type = "origin_sweep" if entry4 == leg.origin_price else "atr_band"

        ladder = EntryLadder(direction=direction, levels=[entry1, entry2, entry3, entry4])
        ladder.entry_types = [entry1_type, "entry2_fib", entry3_type, entry4_type]
        if entry1_prefilled:
            ladder.fills[0] = True

        fvg_info = None
        if fvg_row is not None:
            fvg_info = {
                "direction": fvg_row["direction"],
                "upper": float(fvg_row["upper"]),
                "lower": float(fvg_row["lower"]),
                "consequent_encroachment": float(fvg_row["consequent_encroachment"]),
                "formed_at": fvg_row["formed_at"],
            }
        return ladder, fvg_info

    # -- public API ------------------------------------------------------------

    def build_ladder(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame,
        funding_rate: float | None = None,
        htf_df: pd.DataFrame | None = None,
    ) -> tuple[EntryLadder | None, dict]:
        """Evaluate prerequisites and, if met, build a 4-tier EntryLadder.

        Args:
            df: Working-timeframe closed-candle OHLCV DataFrame ("open",
                "high", "low", "close", "volume", plus a "ts" epoch-ms
                column or a DatetimeIndex -- see directional_bias.py).
            btc_df: BTCUSDT closed-candle OHLCV for RegimeFilter's macro
                beta gate (see regime_filter.py).
            funding_rate: Current funding rate for RegimeFilter's crowd
                sentiment gate, or None if unavailable.
            htf_df: Optional pre-built higher-timeframe OHLC DataFrame,
                passed through to DirectionEngine for its HTF structure gate.

        Returns:
            (ladder, diagnostics). `ladder` is None whenever a prerequisite
            fails or no qualifying impulse leg/expansion candle is found;
            `diagnostics["reason"]` says why. `diagnostics` always includes
            the full DirectionEngine and (once direction is known)
            RegimeFilter diagnostic dicts, plus the located impulse leg and
            FVG info once a ladder is built:
                {
                    "built": bool, "reason": str | None,
                    "direction": "LONG" | "SHORT" | "NEUTRAL",
                    "bias": {...DirectionEngine diagnostics...},
                    "regime": {...RegimeFilter diagnostics...} | None,
                    "impulse_leg": {...} | None,
                    "fvg": {...} | None,
                }

        Raises:
            ValueError: if `df` is missing required columns.
        """
        _require_columns(df, ["high", "low", "close", "volume"], "df")

        bias, bias_diag = self.direction_engine.get_directional_bias(df, htf_df=htf_df)
        diagnostics: dict = {
            "built": False, "reason": None,
            "direction": bias, "bias": bias_diag, "regime": None,
            "impulse_leg": None, "fvg": None,
        }
        if bias == "NEUTRAL":
            diagnostics["reason"] = "directional bias is NEUTRAL"
            return None, diagnostics

        allowed, regime_diag = self.regime_filter.evaluate_market_conditions(df, btc_df, funding_rate, direction=bias)
        diagnostics["regime"] = regime_diag
        if not allowed:
            diagnostics["reason"] = f"regime filter blocked: {regime_diag['failed_gates']}"
            return None, diagnostics

        if bias == "LONG":
            conviction = self._signal_bar_conviction(df, bias)
            diagnostics["signal_bar_conviction"] = conviction
            if conviction < self.long_conviction_min:
                diagnostics["reason"] = (
                    f"LONG signal bar conviction too low: close sits at {conviction:.2f} of its own "
                    f"[low, high] range, below the {self.long_conviction_min} minimum"
                )
                return None, diagnostics

        leg = self._find_impulse_leg(df, bias)
        if leg is None:
            diagnostics["reason"] = "no qualifying impulse leg or breakout expansion candle found"
            return None, diagnostics

        diagnostics["impulse_leg"] = {
            "origin_price": leg.origin_price, "origin_index": leg.origin_index,
            "terminal_price": leg.terminal_price, "terminal_index": leg.terminal_index,
            "source": leg.source, "fresh_break_on_last_bar": leg.fresh_break_on_last_bar,
        }

        if leg.fresh_break_on_last_bar:
            clv = self._trigger_bar_close_location(df, leg)
            diagnostics["impulse_leg"]["trigger_close_location"] = clv
            exhausted = (clv > self.exhaustion_max_percentile) if bias == "LONG" else (clv < 1.0 - self.exhaustion_max_percentile)
            if exhausted:
                diagnostics["reason"] = (
                    f"fresh breakout trigger bar is exhausted -- close sits at the "
                    f"{clv:.2f} percentile of its own impulse leg's range, past the "
                    f"{self.exhaustion_max_percentile} quartile cutoff for {bias}"
                )
                return None, diagnostics

        ladder, fvg_info = self._formulate_ladder(df, bias, leg)
        diagnostics["fvg"] = fvg_info
        diagnostics["built"] = True
        return ladder, diagnostics

    @staticmethod
    def _signal_bar_conviction(df: pd.DataFrame, direction: Direction) -> float:
        """Where the trigger (last) bar's own close sits within ITS OWN
        [low, high] range -- 0.0 = at the bar's low, 1.0 = at the bar's
        high for LONG, mirrored for SHORT (1.0 = closed at the bar's
        low). Distinct from `_trigger_bar_close_location` below, which
        measures the same bar's close against the IMPULSE LEG's range,
        not the bar's own, and only runs for a fresh-breakout market
        entry -- this runs for every LONG signal regardless of entry
        path, since that's exactly what entry_diagnostics.py profiled
        against real outcomes (see `long_conviction_min`'s docstring)."""
        last = df.iloc[-1]
        high, low, close = float(last["high"]), float(last["low"]), float(last["close"])
        rng = high - low
        if rng <= 0:
            return 0.5
        return (close - low) / rng if direction == "LONG" else (high - close) / rng

    @staticmethod
    def _trigger_bar_close_location(df: pd.DataFrame, leg: ImpulseLeg) -> float:
        """Where the last (trigger) bar's close sits within `leg`'s own
        [low, high] range, as a fraction: 0.0 = at the leg's low, 1.0 = at
        the leg's high (can exceed [0, 1] if price has already run past
        the leg's far edge -- `_leg_passes_sanity_checks`'
        `leg_proximity_atr_mult` allows up to 1.0 ATR of that). For a
        single-candle expansion-candle leg (origin/terminal ARE that
        candle's own low/high), this is exactly that candle's own
        close-location-value -- "did this candle close near its own high"."""
        last_close = float(df["close"].iloc[-1])
        leg_low = min(leg.origin_price, leg.terminal_price)
        leg_high = max(leg.origin_price, leg.terminal_price)
        if leg_high <= leg_low:
            return 0.5  # degenerate range (shouldn't happen post leg_min_displacement_atr_mult) -- neutral, don't reject on it
        return (last_close - leg_low) / (leg_high - leg_low)
