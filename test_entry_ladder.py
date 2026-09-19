"""test_entry_ladder.py -- unit tests for entry_ladder.py's Step 11.0
candle-exhaustion filter.

Only new Step 11.0 behavior is covered here -- entry_ladder.py's
pre-existing impulse-leg-discovery/FVG/retracement-pricing logic has no
dedicated test file of its own (exercised only indirectly, via
end-to-end fixtures in test_engine.py/test_live_runner.py/
test_multi_strategy.py/test_backtest_engine.py); backfilling full
coverage for that pre-existing surface is out of scope for this change.

Run with:  python3 -m unittest test_entry_ladder -v
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from entry_ladder import EntryLadderEngine, ImpulseLeg
from test_engine import AlwaysAllowRegimeFilter, make_candles


class TriggerBarCloseLocationTests(unittest.TestCase):
    """`_trigger_bar_close_location`: where the trigger bar's close sits
    within the impulse leg's own [low, high] range, as a 0..1 fraction
    (can exceed that range if price has already run past the leg's far
    edge -- `_leg_passes_sanity_checks` allows up to 1.0 ATR of that)."""

    @staticmethod
    def _df_with_close(close: float) -> pd.DataFrame:
        return pd.DataFrame({"close": pd.Series([100.0, 101.0, close])})

    def test_close_at_leg_low_is_zero(self) -> None:
        leg = ImpulseLeg(origin_price=100.0, origin_index=0, terminal_price=110.0, terminal_index=1, source="swing_pivots", fresh_break_on_last_bar=True)
        clv = EntryLadderEngine._trigger_bar_close_location(self._df_with_close(100.0), leg)
        self.assertAlmostEqual(clv, 0.0, places=6)

    def test_close_at_leg_high_is_one(self) -> None:
        leg = ImpulseLeg(origin_price=100.0, origin_index=0, terminal_price=110.0, terminal_index=1, source="swing_pivots", fresh_break_on_last_bar=True)
        clv = EntryLadderEngine._trigger_bar_close_location(self._df_with_close(110.0), leg)
        self.assertAlmostEqual(clv, 1.0, places=6)

    def test_close_at_midpoint_is_half(self) -> None:
        leg = ImpulseLeg(origin_price=100.0, origin_index=0, terminal_price=110.0, terminal_index=1, source="swing_pivots", fresh_break_on_last_bar=True)
        clv = EntryLadderEngine._trigger_bar_close_location(self._df_with_close(105.0), leg)
        self.assertAlmostEqual(clv, 0.5, places=6)

    def test_close_past_leg_high_exceeds_one(self) -> None:
        leg = ImpulseLeg(origin_price=100.0, origin_index=0, terminal_price=110.0, terminal_index=1, source="swing_pivots", fresh_break_on_last_bar=True)
        clv = EntryLadderEngine._trigger_bar_close_location(self._df_with_close(112.0), leg)
        self.assertAlmostEqual(clv, 1.2, places=6)

    def test_direction_agnostic_uses_min_max_of_origin_terminal(self) -> None:
        # A SHORT leg's origin (swing high) > terminal (swing low) --
        # the function must not assume origin < terminal.
        leg = ImpulseLeg(origin_price=110.0, origin_index=0, terminal_price=100.0, terminal_index=1, source="swing_pivots", fresh_break_on_last_bar=True)
        clv = EntryLadderEngine._trigger_bar_close_location(self._df_with_close(100.0), leg)
        self.assertAlmostEqual(clv, 0.0, places=6)

    def test_degenerate_zero_range_returns_neutral_half(self) -> None:
        leg = ImpulseLeg(origin_price=100.0, origin_index=0, terminal_price=100.0, terminal_index=1, source="swing_pivots", fresh_break_on_last_bar=True)
        clv = EntryLadderEngine._trigger_bar_close_location(self._df_with_close(100.0), leg)
        self.assertAlmostEqual(clv, 0.5, places=6)


class ExhaustionFilterTests(unittest.TestCase):
    """build_ladder()'s Step 11.0 exhaustion gate: rejects a FRESH-BREAKOUT
    market entry whose trigger bar already closed past the exhaustion
    quartile of its own impulse leg's range. `_find_impulse_leg` is
    patched to a controlled ImpulseLeg so the exact CLV is known --
    isolating this gate from entry_ladder.py's own (separately exercised
    elsewhere) leg-discovery logic."""

    def setUp(self) -> None:
        # A proven 4-way bullish confluence (see test_engine.py) -- real
        # DirectionEngine output, real RegimeFilter bypassed via the stub.
        full = make_candles(371, direction=1, seed=1)
        self.df = pd.DataFrame(full)
        self.btc_df = self.df.copy()
        self.last_close = float(self.df["close"].iloc[-1])
        # Explicit 0.75 here, not the (deliberately looser) class default
        # -- see EntryLadderEngine's own docstring for why 0.75 would
        # reject real signals in production and isn't used as the default.
        # long_conviction_min=0.0 disables the separate Step 11.3 conviction
        # gate so these tests stay isolated to exhaustion-gate behavior --
        # this fixture's own trigger bar doesn't reliably clear 0.70 on its
        # OWN candle range (a different, unrelated denominator from the
        # leg-relative CLV this class is about).
        self.engine = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter(), exhaustion_max_percentile=0.75, long_conviction_min=0.0)

    def test_fresh_breakout_closing_at_the_leg_high_is_rejected(self) -> None:
        leg = ImpulseLeg(origin_price=self.last_close - 10.0, origin_index=300, terminal_price=self.last_close,
                          terminal_index=369, source="swing_pivots", fresh_break_on_last_bar=True)
        with patch.object(self.engine, "_find_impulse_leg", return_value=leg):
            ladder, diag = self.engine.build_ladder(self.df, self.btc_df, funding_rate=0.02)
        self.assertIsNone(ladder)
        self.assertFalse(diag["built"])
        self.assertIn("exhausted", diag["reason"])
        self.assertAlmostEqual(diag["impulse_leg"]["trigger_close_location"], 1.0, places=6)

    def test_fresh_breakout_closing_at_the_leg_low_is_not_rejected(self) -> None:
        leg = ImpulseLeg(origin_price=self.last_close, origin_index=300, terminal_price=self.last_close + 10.0,
                          terminal_index=369, source="swing_pivots", fresh_break_on_last_bar=True)
        with patch.object(self.engine, "_find_impulse_leg", return_value=leg):
            ladder, diag = self.engine.build_ladder(self.df, self.btc_df, funding_rate=0.02)
        self.assertIsNotNone(ladder)
        self.assertTrue(diag["built"])

    def test_below_quartile_cutoff_passes(self) -> None:
        # last_close sits at exactly the 70th percentile of a 0..10 range
        # (leg_low + 7.0) -- below this test's 0.75 cutoff.
        leg = ImpulseLeg(origin_price=self.last_close - 7.0, origin_index=300, terminal_price=self.last_close + 3.0,
                          terminal_index=369, source="swing_pivots", fresh_break_on_last_bar=True)
        with patch.object(self.engine, "_find_impulse_leg", return_value=leg):
            ladder, diag = self.engine.build_ladder(self.df, self.btc_df, funding_rate=0.02)
        self.assertIsNotNone(ladder)
        self.assertAlmostEqual(diag["impulse_leg"]["trigger_close_location"], 0.70, places=6)

    def test_non_market_entry_leg_is_never_exhaustion_checked(self) -> None:
        """A leg NOT sourced from a fresh breakout (fresh_break_on_last_bar
        False -- Entry 1 will be a resting FVG/Fibonacci order, never a
        market order) must build even with a close deep in the same
        'exhausted' zone that would reject a fresh-breakout leg."""
        leg = ImpulseLeg(origin_price=self.last_close - 10.0, origin_index=300, terminal_price=self.last_close,
                          terminal_index=369, source="swing_pivots", fresh_break_on_last_bar=False)
        with patch.object(self.engine, "_find_impulse_leg", return_value=leg):
            ladder, diag = self.engine.build_ladder(self.df, self.btc_df, funding_rate=0.02)
        self.assertIsNotNone(ladder)
        self.assertTrue(diag["built"])
        self.assertNotIn("trigger_close_location", diag["impulse_leg"])

    def test_custom_exhaustion_max_percentile_is_respected(self) -> None:
        leg = ImpulseLeg(origin_price=self.last_close - 5.0, origin_index=300, terminal_price=self.last_close + 5.0,
                          terminal_index=369, source="swing_pivots", fresh_break_on_last_bar=True)  # CLV = 0.5
        loose = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter(), exhaustion_max_percentile=0.90, long_conviction_min=0.0)
        tight = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter(), exhaustion_max_percentile=0.40, long_conviction_min=0.0)
        with patch.object(loose, "_find_impulse_leg", return_value=leg):
            loose_ladder, _ = loose.build_ladder(self.df, self.btc_df, funding_rate=0.02)
        with patch.object(tight, "_find_impulse_leg", return_value=leg):
            tight_ladder, _ = tight.build_ladder(self.df, self.btc_df, funding_rate=0.02)
        self.assertIsNotNone(loose_ladder, "0.5 CLV must pass a loose 0.90 cutoff")
        self.assertIsNone(tight_ladder, "0.5 CLV must fail a tight 0.40 cutoff")


class SignalBarConvictionTests(unittest.TestCase):
    """build_ladder()'s Step 11.3 LONG-only conviction gate: rejects ANY
    long signal (not gated on fresh_break_on_last_bar, unlike the
    exhaustion filter above) whose trigger bar's own close sits below
    `long_conviction_min` of its OWN [low, high] range."""

    @staticmethod
    def _df_with_conviction(conviction: float, direction: int = 1, rng: float = 10.0, n: int = 371) -> pd.DataFrame:
        """Same proven bullish/bearish confluence fixture as
        ExhaustionFilterTests, with only the last bar's high/low widened
        around its EXISTING close (left untouched, so the real
        DirectionEngine's bias computation -- which also reads that same
        close -- isn't disturbed) to hit an exact target conviction."""
        rows = make_candles(n, direction=direction, seed=1)
        df = pd.DataFrame(rows)
        close = float(df["close"].iloc[-1])
        if direction == 1:
            low, high = close - conviction * rng, close + (1 - conviction) * rng
        else:
            high, low = close + conviction * rng, close - (1 - conviction) * rng
        df.loc[df.index[-1], ["high", "low"]] = [high, low]
        return df

    def test_direct_math_long(self) -> None:
        df = pd.DataFrame({"high": [110.0], "low": [100.0], "close": [107.0]})
        self.assertAlmostEqual(EntryLadderEngine._signal_bar_conviction(df, "LONG"), 0.7, places=6)

    def test_direct_math_short_is_mirrored(self) -> None:
        df = pd.DataFrame({"high": [110.0], "low": [100.0], "close": [103.0]})
        self.assertAlmostEqual(EntryLadderEngine._signal_bar_conviction(df, "SHORT"), 0.7, places=6)

    def test_degenerate_zero_range_returns_neutral_half(self) -> None:
        df = pd.DataFrame({"high": [100.0], "low": [100.0], "close": [100.0]})
        self.assertAlmostEqual(EntryLadderEngine._signal_bar_conviction(df, "LONG"), 0.5, places=6)

    def test_low_conviction_long_signal_is_rejected_without_needing_a_leg(self) -> None:
        df = self._df_with_conviction(0.2)  # below the 0.40 default
        engine = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter())
        ladder, diag = engine.build_ladder(df, df, funding_rate=0.02)
        self.assertIsNone(ladder)
        self.assertIn("conviction", diag["reason"])
        self.assertAlmostEqual(diag["signal_bar_conviction"], 0.2, places=6)
        self.assertIsNone(diag["impulse_leg"], "must reject before ever calling _find_impulse_leg")

    def test_high_conviction_long_signal_passes_the_gate(self) -> None:
        df = self._df_with_conviction(0.9)  # above the 0.40 default
        engine = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter())
        _, diag = engine.build_ladder(df, df, funding_rate=0.02)
        self.assertAlmostEqual(diag["signal_bar_conviction"], 0.9, places=6)
        self.assertNotIn("conviction", diag["reason"] or "")

    def test_default_does_not_exceed_the_canonical_fixtures_conviction(self) -> None:
        """Regression: an earlier 0.70 default would have rejected this
        repo's own canonical real-signal fixture (test_backtest_engine's
        make_candles(900, seed=1)), whose one real, all-other-gates-
        qualifying LONG bar has a conviction of 0.4425 -- see
        long_conviction_min's own docstring for the full story, and
        test_backtest_engine.VectorizedCrossCheckTests for the full
        pipeline's own end-to-end proof that fixture still trades."""
        engine = EntryLadderEngine()
        self.assertLessEqual(engine.long_conviction_min, 0.4425)

    def test_custom_threshold_is_respected(self) -> None:
        df = self._df_with_conviction(0.4)
        loose = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter(), long_conviction_min=0.30)
        tight = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter(), long_conviction_min=0.50)
        loose_ladder, _ = loose.build_ladder(df, df, funding_rate=0.02)
        tight_ladder, _ = tight.build_ladder(df, df, funding_rate=0.02)
        self.assertIsNotNone(loose_ladder, "0.4 conviction must pass a loose 0.30 minimum")
        self.assertIsNone(tight_ladder, "0.4 conviction must fail a tight 0.50 minimum")

    def test_short_signals_are_never_gated_by_long_conviction_min(self) -> None:
        """The gate is LONG-only by design (entry_diagnostics.py found no
        separating power for SHORT outcomes at any threshold) -- a SHORT
        signal with a "bad" raw conviction score must never be rejected
        by this gate."""
        df = self._df_with_conviction(0.1, direction=-1)  # would fail any real LONG-style minimum
        engine = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter())
        _, diag = engine.build_ladder(df, df, funding_rate=0.02)
        self.assertNotIn("signal_bar_conviction", diag)


if __name__ == "__main__":
    unittest.main()
