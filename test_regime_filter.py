"""test_regime_filter.py -- tests for regime_filter.py's Step 10.8
stretch/exhaustion gate and Step 11.0's squeeze/compression gate.

Run with:  python3 -m unittest test_regime_filter -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from regime_filter import RegimeFilter


def _flat_df(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes)
    return pd.DataFrame({"high": close + 0.3, "low": close - 0.3, "close": close, "volume": pd.Series([1000.0] * len(close))})


class StretchGateTests(unittest.TestCase):
    """Step 10.8: reject a candidate whose close already sits more than
    `stretch_atr_max` ATRs away from its own baseline EMA -- the 180-day
    BTC/ETH/SOL backtest showed entries past this threshold winning only
    45.5% of the time vs. the overall 54.8% (buying exhaustion, not
    ignition)."""

    def test_passes_when_close_is_near_baseline_ema(self) -> None:
        # A flat series: EMA converges to price, ATR stays small and
        # constant -- close sits right on its own EMA, stretch ~ 0.
        df = _flat_df([100.0] * 80)
        rf = RegimeFilter()
        gate = rf._stretch_gate(df)
        self.assertTrue(gate.passed, gate.detail)
        self.assertAlmostEqual(gate.value, 0.0, places=6)

    def test_fails_when_close_has_run_away_from_baseline_ema(self) -> None:
        # A long flat run establishes a stable, low-ATR baseline EMA, then
        # one sharp final bar jumps far away from it -- ATR barely reacts
        # (Wilder smoothing lags a single new bar), so the jump reads as
        # many ATRs of stretch, exactly the "already expanded, not just
        # starting" breakout this gate exists to reject.
        df = _flat_df([100.0] * 79 + [130.0])
        rf = RegimeFilter()
        gate = rf._stretch_gate(df)
        self.assertFalse(gate.passed, gate.detail)
        self.assertGreater(gate.value, rf.stretch_atr_max)

    def test_fails_safe_during_warmup(self) -> None:
        df = _flat_df([100.0, 101.0, 99.0])  # far too few bars for ATR14/EMA50
        rf = RegimeFilter()
        gate = rf._stretch_gate(df)
        self.assertFalse(gate.passed)
        self.assertIsNone(gate.value)

    def test_custom_stretch_atr_max_is_respected(self) -> None:
        df = _flat_df([100.0] * 79 + [102.0])  # a small, modest jump
        loose = RegimeFilter(stretch_atr_max=10.0)
        tight = RegimeFilter(stretch_atr_max=0.01)
        self.assertTrue(loose._stretch_gate(df).passed)
        self.assertFalse(tight._stretch_gate(df).passed)

    def test_evaluate_market_conditions_includes_stretch_gate(self) -> None:
        df = _flat_df([100.0] * 79 + [130.0])
        btc_df = pd.DataFrame({"close": pd.Series(np.linspace(50.0, 150.0, 80))})
        rf = RegimeFilter()
        allowed, diag = rf.evaluate_market_conditions(df, btc_df, funding_rate=None, direction="LONG")
        self.assertIn("stretch", diag["gates"])
        self.assertIn("stretch", diag["failed_gates"])
        self.assertFalse(allowed)


class SqueezeGateTests(unittest.TestCase):
    """Step 11.0: requires the current Bollinger Band width to sit at or
    below `squeeze_max_percentile` of its own trailing `squeeze_lookback`-
    bar history -- i.e. volatility must have contracted relative to
    itself recently. Uses small custom bb_length/lookback (not the
    production defaults) purely to keep test fixtures short; see
    RegimeFilter's own docstring for why the production default
    (0.85) is deliberately loose and still unvalidated against real data."""

    def test_passes_when_current_width_is_the_most_compressed_in_its_window(self) -> None:
        # 20 oscillating bars (wide BB width throughout) followed by a
        # flat tail -- the last bar's 5-bar window is dead flat (width 0),
        # the narrowest point in the whole trailing 20-bar history.
        oscillating = [100.0 + 10.0 * ((-1) ** i) for i in range(20)]
        flat_tail = [100.0] * 6
        df = pd.DataFrame({"close": pd.Series(oscillating + flat_tail, dtype=float)})
        rf = RegimeFilter(squeeze_bb_length=5, squeeze_lookback=20, squeeze_max_percentile=0.35)
        gate = rf._squeeze_gate(df)
        self.assertTrue(gate.passed, gate.detail)
        self.assertAlmostEqual(gate.value, 0.10, places=6)

    def test_fails_when_current_width_is_already_expanded(self) -> None:
        # 20 flat bars (near-zero BB width throughout) followed by a sharp
        # whipsaw tail -- the last bar's 5-bar window is now the widest
        # point in the trailing history, exactly the "already expanding,
        # not compressed" case this gate exists to reject.
        flat_head = [100.0] * 20
        whipsaw_tail = [100.0, 110.0, 90.0, 115.0, 85.0, 120.0]
        df = pd.DataFrame({"close": pd.Series(flat_head + whipsaw_tail, dtype=float)})
        rf = RegimeFilter(squeeze_bb_length=5, squeeze_lookback=20, squeeze_max_percentile=0.35)
        gate = rf._squeeze_gate(df)
        self.assertFalse(gate.passed, gate.detail)
        self.assertAlmostEqual(gate.value, 1.0, places=6)

    def test_fails_safe_during_warmup(self) -> None:
        df = pd.DataFrame({"close": pd.Series([100.0] * 10, dtype=float)})  # far too few bars
        rf = RegimeFilter(squeeze_bb_length=5, squeeze_lookback=20)
        gate = rf._squeeze_gate(df)
        self.assertFalse(gate.passed)
        self.assertIsNone(gate.value)

    def test_custom_squeeze_max_percentile_is_respected(self) -> None:
        flat_head = [100.0] * 20
        whipsaw_tail = [100.0, 110.0, 90.0, 115.0, 85.0, 120.0]
        df = pd.DataFrame({"close": pd.Series(flat_head + whipsaw_tail, dtype=float)})
        loose = RegimeFilter(squeeze_bb_length=5, squeeze_lookback=20, squeeze_max_percentile=1.0)
        tight = RegimeFilter(squeeze_bb_length=5, squeeze_lookback=20, squeeze_max_percentile=0.35)
        self.assertTrue(loose._squeeze_gate(df).passed)
        self.assertFalse(tight._squeeze_gate(df).passed)

    def test_evaluate_market_conditions_includes_squeeze_gate(self) -> None:
        flat_head = [100.0] * 20
        whipsaw_tail = [100.0, 110.0, 90.0, 115.0, 85.0, 120.0]
        df = _flat_df(flat_head + whipsaw_tail)
        btc_df = pd.DataFrame({"close": pd.Series(np.linspace(50.0, 150.0, 80))})
        rf = RegimeFilter(squeeze_bb_length=5, squeeze_lookback=20, squeeze_max_percentile=0.35)
        allowed, diag = rf.evaluate_market_conditions(df, btc_df, funding_rate=None, direction="LONG")
        self.assertIn("squeeze", diag["gates"])
        self.assertIn("squeeze", diag["failed_gates"])
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
