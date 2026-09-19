"""test_entry_diagnostics.py -- unit + smoke tests for entry_diagnostics.py.

Covers:
  1. `_signal_features`'s candle-shape/volume math in isolation.
  2. `_fill_depth`'s tier-count/fill-pct math in isolation.
  3. `_truncate_fills`'s offset-cutoff counterfactual.
  4. `bucket_summary`/`threshold_whatif`'s aggregation math on a
     hand-built profile DataFrame (no need to run the real stack for
     these -- they're pure pandas over already-profiled rows).
  5. A smoke test proving `profile_trades` + `fill_truncation_whatif` run
     end-to-end against the real production stack on a small synthetic
     series without crashing.

Run with:  python3 -m unittest test_entry_diagnostics -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from entry_diagnostics import (
    SETTLED_GEOMETRY,
    _fill_depth,
    _htf_features,
    _signal_features,
    _truncate_fills,
    bucket_summary,
    fill_truncation_whatif,
    profile_trades,
    symbol_direction_summary,
    symbol_exclusion_whatif,
    threshold_whatif,
    threshold_whatif_by_direction,
)
from grid_resimulation import FrozenTrade, generate_frozen_trades
from risk_manager import TradeLifecycleManager
from test_backtest_engine import make_candles, make_monotonic_trend


def _make_trade(direction="LONG", expected_vwap=100.0, atr_at_signal=2.0,
                 fill_events=((0, 0, 100.0),), ladder_weights=(1.0, 0.0, 0.0, 0.0),
                 window_end_index=10, signal_index=0) -> FrozenTrade:
    return FrozenTrade(
        symbol="TESTUSDT", direction=direction, signal_index=signal_index, signal_ts=1_700_000_000_000,
        window_end_index=window_end_index, expected_vwap=expected_vwap, atr_at_signal=atr_at_signal,
        ladder_levels=(100.0, 95.0, 90.0, 85.0), ladder_weights=ladder_weights, fill_events=fill_events,
    )


class _FakeCache:
    """Minimal stand-in for VectorizedSignalCache: only the two attributes
    `_signal_features` actually reads."""

    def __init__(self, ltf_df: pd.DataFrame, rvol: pd.Series) -> None:
        self.ltf_df = ltf_df
        self.rvol = rvol


class _FakeHtfCache:
    """Minimal stand-in for VectorizedSignalCache: only the attributes
    `_htf_features` actually reads."""

    def __init__(self, htf_row_idx, htf_trend_by_bucket, ema_fast, ema_slow, atr14) -> None:
        self.htf_row_idx = htf_row_idx
        self.htf_trend_by_bucket = htf_trend_by_bucket
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr14 = atr14


class SignalFeaturesTests(unittest.TestCase):
    def test_long_close_location_and_body_to_range(self) -> None:
        # bar: open=100, high=110, low=95, close=108 -- range=15, body=8
        df = pd.DataFrame({"open": [100.0], "high": [110.0], "low": [95.0], "close": [108.0]})
        cache = _FakeCache(df, pd.Series([2.5]))
        feats = _signal_features(cache, 0, "LONG")
        self.assertAlmostEqual(feats["body_to_range"], 8.0 / 15.0, places=6)
        self.assertAlmostEqual(feats["close_location"], (108.0 - 95.0) / 15.0, places=6)
        self.assertAlmostEqual(feats["vol_ratio"], 2.5, places=6)

    def test_short_close_location_is_mirrored(self) -> None:
        # SHORT: close_location = (high-close)/(high-low) -- close near the LOW is favorable.
        df = pd.DataFrame({"open": [100.0], "high": [110.0], "low": [95.0], "close": [97.0]})
        cache = _FakeCache(df, pd.Series([1.0]))
        feats = _signal_features(cache, 0, "SHORT")
        self.assertAlmostEqual(feats["close_location"], (110.0 - 97.0) / 15.0, places=6)

    def test_zero_range_bar_yields_nan_not_a_crash(self) -> None:
        df = pd.DataFrame({"open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]})
        cache = _FakeCache(df, pd.Series([1.0]))
        feats = _signal_features(cache, 0, "LONG")
        self.assertTrue(np.isnan(feats["body_to_range"]))
        self.assertTrue(np.isnan(feats["close_location"]))

    def test_nan_vol_ratio_passthrough(self) -> None:
        df = pd.DataFrame({"open": [100.0], "high": [110.0], "low": [95.0], "close": [105.0]})
        cache = _FakeCache(df, pd.Series([float("nan")]))
        feats = _signal_features(cache, 0, "LONG")
        self.assertTrue(np.isnan(feats["vol_ratio"]))


class HtfFeaturesTests(unittest.TestCase):
    def test_age_zero_when_bucket_differs_from_previous(self) -> None:
        cache = _FakeHtfCache(
            htf_row_idx=np.array([0, 1, 2, 3]),
            htf_trend_by_bucket=np.array([1, 1, -1, -1]),
            ema_fast=pd.Series([101.0] * 4), ema_slow=pd.Series([100.0] * 4), atr14=pd.Series([2.0] * 4),
        )
        feats = _htf_features(cache, 2)  # bucket 2's trend (-1) just changed from bucket 1's (1)
        self.assertEqual(feats["htf_break_age_buckets"], 0)

    def test_age_counts_consecutive_matching_buckets(self) -> None:
        cache = _FakeHtfCache(
            htf_row_idx=np.array([0, 1, 2, 3]),
            htf_trend_by_bucket=np.array([1, 1, 1, 1]),
            ema_fast=pd.Series([101.0] * 4), ema_slow=pd.Series([100.0] * 4), atr14=pd.Series([2.0] * 4),
        )
        feats = _htf_features(cache, 3)
        self.assertEqual(feats["htf_break_age_buckets"], 3)

    def test_negative_bucket_index_yields_none_age(self) -> None:
        cache = _FakeHtfCache(
            htf_row_idx=np.array([-1]),
            htf_trend_by_bucket=np.array([1]),
            ema_fast=pd.Series([101.0]), ema_slow=pd.Series([100.0]), atr14=pd.Series([2.0]),
        )
        feats = _htf_features(cache, 0)
        self.assertIsNone(feats["htf_break_age_buckets"])

    def test_ema_spread_atr_is_unsigned_ratio(self) -> None:
        cache = _FakeHtfCache(
            htf_row_idx=np.array([0]), htf_trend_by_bucket=np.array([1]),
            ema_fast=pd.Series([103.0]), ema_slow=pd.Series([100.0]), atr14=pd.Series([2.0]),
        )
        feats = _htf_features(cache, 0)
        self.assertAlmostEqual(feats["ema_spread_atr"], 1.5, places=6)

    def test_zero_or_missing_atr_yields_nan_spread(self) -> None:
        cache = _FakeHtfCache(
            htf_row_idx=np.array([0]), htf_trend_by_bucket=np.array([1]),
            ema_fast=pd.Series([103.0]), ema_slow=pd.Series([100.0]), atr14=pd.Series([0.0]),
        )
        feats = _htf_features(cache, 0)
        self.assertTrue(np.isnan(feats["ema_spread_atr"]))


class SymbolDirectionSummaryTests(unittest.TestCase):
    def test_groups_by_symbol_and_direction(self) -> None:
        df = pd.DataFrame([
            {"symbol": "BTCUSDT", "direction": "SHORT", "r_multiple": 0.5, "htf_break_age_buckets": 2, "ema_spread_atr": -0.3},
            {"symbol": "BTCUSDT", "direction": "SHORT", "r_multiple": 0.3, "htf_break_age_buckets": 4, "ema_spread_atr": -0.5},
            {"symbol": "SOLUSDT", "direction": "SHORT", "r_multiple": -1.0, "htf_break_age_buckets": 0, "ema_spread_atr": -0.1},
        ])
        summary = symbol_direction_summary(df)
        self.assertEqual(summary.loc[("BTCUSDT", "SHORT"), "n"], 2)
        self.assertAlmostEqual(summary.loc[("BTCUSDT", "SHORT"), "avg_R"], 0.4, places=6)
        self.assertAlmostEqual(summary.loc[("SOLUSDT", "SHORT"), "sum_R"], -1.0, places=6)


class SymbolExclusionWhatifTests(unittest.TestCase):
    def test_excluding_the_dragging_symbol_changes_the_sign(self) -> None:
        df = pd.DataFrame([
            {"symbol": "BTCUSDT", "direction": "SHORT", "r_multiple": 1.0},
            {"symbol": "BTCUSDT", "direction": "SHORT", "r_multiple": 0.5},
            {"symbol": "SOLUSDT", "direction": "SHORT", "r_multiple": -3.0},
            {"symbol": "LONGROW", "direction": "LONG", "r_multiple": 5.0},  # must not leak into SHORT stats
        ])
        result = symbol_exclusion_whatif(df, "SHORT", ["SOLUSDT"])
        self.assertAlmostEqual(result["baseline"]["sum_R"], -1.5, places=6)
        self.assertAlmostEqual(result["excl_SOLUSDT"]["sum_R"], 1.5, places=6)
        self.assertEqual(result["excl_SOLUSDT"]["n"], 2)
        self.assertAlmostEqual(result["excl_all_listed"]["sum_R"], 1.5, places=6)


class FillDepthTests(unittest.TestCase):
    def test_single_tier_fill(self) -> None:
        trade = _make_trade(fill_events=((0, 0, 100.0),), ladder_weights=(0.40, 0.35, 0.25, 0.0))
        depth = _fill_depth(trade)
        self.assertEqual(depth["n_tiers_hit"], 1)
        self.assertAlmostEqual(depth["fill_pct"], 40.0, places=6)

    def test_multi_tier_fill(self) -> None:
        trade = _make_trade(fill_events=((0, 0, 100.0), (2, 1, 95.0)), ladder_weights=(0.40, 0.35, 0.25, 0.0))
        depth = _fill_depth(trade)
        self.assertEqual(depth["n_tiers_hit"], 2)
        self.assertAlmostEqual(depth["fill_pct"], 75.0, places=6)

    def test_full_ladder_fill_is_100_pct(self) -> None:
        trade = _make_trade(fill_events=((0, 0, 100.0), (1, 1, 95.0), (2, 2, 90.0)), ladder_weights=(0.40, 0.35, 0.25, 0.0))
        depth = _fill_depth(trade)
        self.assertEqual(depth["n_tiers_hit"], 3)
        self.assertAlmostEqual(depth["fill_pct"], 100.0, places=6)


class TruncateFillsTests(unittest.TestCase):
    def test_offset_zero_prefill_always_kept(self) -> None:
        trade = _make_trade(fill_events=((0, 0, 100.0), (5, 1, 95.0)))
        truncated = _truncate_fills(trade, max_offset=1)
        self.assertEqual(truncated.fill_events, ((0, 0, 100.0),))

    def test_fills_within_cutoff_are_kept(self) -> None:
        trade = _make_trade(fill_events=((0, 0, 100.0), (1, 1, 95.0), (2, 2, 90.0)))
        truncated = _truncate_fills(trade, max_offset=1)
        self.assertEqual(truncated.fill_events, ((0, 0, 100.0), (1, 1, 95.0)))

    def test_none_max_offset_is_a_no_op(self) -> None:
        trade = _make_trade(fill_events=((0, 0, 100.0), (5, 1, 95.0)))
        self.assertIs(_truncate_fills(trade, None), trade)

    def test_all_fills_beyond_cutoff_leaves_empty_tuple(self) -> None:
        trade = _make_trade(fill_events=((3, 1, 95.0), (4, 2, 90.0)))
        truncated = _truncate_fills(trade, max_offset=1)
        self.assertEqual(truncated.fill_events, ())


class BucketSummaryTests(unittest.TestCase):
    def test_groups_and_orders_by_sum_r_ascending(self) -> None:
        df = pd.DataFrame([
            {"exit_reason": "TP1", "r_multiple": 0.8, "body_to_range": 0.7, "close_location": 0.9, "vol_ratio": 3.0, "fill_pct": 40.0, "n_tiers_hit": 1},
            {"exit_reason": "TP1", "r_multiple": 0.6, "body_to_range": 0.6, "close_location": 0.8, "vol_ratio": 2.8, "fill_pct": 40.0, "n_tiers_hit": 1},
            {"exit_reason": "SL", "r_multiple": -1.1, "body_to_range": 0.2, "close_location": 0.3, "vol_ratio": 1.2, "fill_pct": 90.0, "n_tiers_hit": 3},
        ])
        summary = bucket_summary(df)
        self.assertEqual(list(summary.index), ["SL", "TP1"], "SL's negative sum_R must sort first (ascending)")
        self.assertEqual(summary.loc["TP1", "n"], 2)
        self.assertAlmostEqual(summary.loc["TP1", "avg_R"], 0.7, places=6)
        self.assertAlmostEqual(summary.loc["SL", "avg_fill_pct"], 90.0, places=6)

    def test_empty_dataframe_returns_empty(self) -> None:
        self.assertTrue(bucket_summary(pd.DataFrame()).empty)


class ThresholdWhatifTests(unittest.TestCase):
    def test_higher_threshold_rejects_more_and_flags_good_vs_bad(self) -> None:
        df = pd.DataFrame([
            {"exit_reason": "TP1", "r_multiple": 0.8, "vol_ratio": 3.0},
            {"exit_reason": "SL", "r_multiple": -1.1, "vol_ratio": 1.0},
            {"exit_reason": "TIME_DECAY", "r_multiple": -0.6, "vol_ratio": 1.2},
            {"exit_reason": "TP2", "r_multiple": 0.9, "vol_ratio": 1.8},  # a real winner just above the lower threshold
        ])
        result = threshold_whatif(df, "vol_ratio", [1.5, 2.0], higher_is_better=True)
        row_1_5 = result[result.threshold == 1.5].iloc[0]
        self.assertEqual(row_1_5["n_kept"], 2)  # TP1(3.0), TP2(1.8)
        self.assertEqual(row_1_5["bad_rejected"], 2)  # SL(1.0), TIME_DECAY(1.2)
        self.assertEqual(row_1_5["good_rejected"], 0)
        row_2_0 = result[result.threshold == 2.0].iloc[0]
        self.assertEqual(row_2_0["n_kept"], 1)  # TP1(3.0) only -- TP2(1.8) now also rejected
        self.assertEqual(row_2_0["good_rejected"], 1)  # TP2 -- the cost of the tighter threshold
        self.assertEqual(row_2_0["bad_rejected"], 2)  # SL, TIME_DECAY

    def test_nan_column_rows_are_excluded_from_the_base_population(self) -> None:
        df = pd.DataFrame([
            {"exit_reason": "TP1", "r_multiple": 0.8, "vol_ratio": 3.0},
            {"exit_reason": "SL", "r_multiple": -1.1, "vol_ratio": float("nan")},
        ])
        result = threshold_whatif(df, "vol_ratio", [1.5], higher_is_better=True)
        self.assertEqual(result.iloc[0]["n_kept"] + result.iloc[0]["n_rejected"], 1)

    def test_runner_trail_counts_as_a_good_rejection(self) -> None:
        """Regression: _GOOD_EXIT_REASONS used to say "RUNNER", which
        never matches replay_exit's actual "RUNNER_TRAIL" label -- a
        rejected runner-tier winner silently vanished from good_rejected."""
        df = pd.DataFrame([
            {"exit_reason": "RUNNER_TRAIL", "r_multiple": 0.9, "vol_ratio": 1.0},
            {"exit_reason": "TP1", "r_multiple": 0.8, "vol_ratio": 3.0},
        ])
        result = threshold_whatif(df, "vol_ratio", [1.5], higher_is_better=True)
        self.assertEqual(result.iloc[0]["good_rejected"], 1)

    def test_profit_factor_kept(self) -> None:
        df = pd.DataFrame([
            {"exit_reason": "TP1", "r_multiple": 0.8, "vol_ratio": 3.0},
            {"exit_reason": "TP2", "r_multiple": 0.4, "vol_ratio": 3.0},
            {"exit_reason": "SL", "r_multiple": -0.6, "vol_ratio": 3.0},
        ])
        result = threshold_whatif(df, "vol_ratio", [1.5], higher_is_better=True)
        self.assertAlmostEqual(result.iloc[0]["profit_factor_kept"], (0.8 + 0.4) / 0.6, places=6)

    def test_track_reasons_adds_per_reason_kept_columns(self) -> None:
        df = pd.DataFrame([
            {"exit_reason": "TIME_DECAY", "r_multiple": -0.6, "vol_ratio": 3.0},
            {"exit_reason": "TIME_DECAY", "r_multiple": -0.5, "vol_ratio": 1.0},
            {"exit_reason": "TP2", "r_multiple": 0.9, "vol_ratio": 3.0},
        ])
        result = threshold_whatif(df, "vol_ratio", [1.5], track_reasons=["TIME_DECAY", "TP2"])
        self.assertEqual(result.iloc[0]["TIME_DECAY_kept"], 1)
        self.assertEqual(result.iloc[0]["TP2_kept"], 1)


class ThresholdWhatifByDirectionTests(unittest.TestCase):
    def test_splits_into_one_table_per_direction(self) -> None:
        df = pd.DataFrame([
            {"direction": "LONG", "exit_reason": "TP2", "r_multiple": 0.8, "vol_ratio": 3.0},
            {"direction": "LONG", "exit_reason": "SL", "r_multiple": -0.1, "vol_ratio": 1.0},
            {"direction": "SHORT", "exit_reason": "TIME_DECAY", "r_multiple": -0.6, "vol_ratio": 3.0},
        ])
        tables = threshold_whatif_by_direction(df, "vol_ratio", [1.5])
        self.assertEqual(set(tables), {"LONG", "SHORT"})
        self.assertEqual(tables["LONG"].iloc[0]["n_kept"], 1)
        self.assertEqual(tables["SHORT"].iloc[0]["n_kept"], 1)


class EndToEndSmokeTests(unittest.TestCase):
    """Proves profile_trades + fill_truncation_whatif run against the REAL
    production stack without crashing and produce a sane, correctly-shaped
    result -- not a claim about what a real 180-day/3-symbol run would show
    (this repo's sandbox has no market data access)."""

    def test_small_synthetic_run_produces_well_formed_tables(self) -> None:
        n = 900  # same verified fixture size grid_resimulation's own smoke test uses
        btc_rows = make_monotonic_trend(n, direction=1, seed=99, slope=0.05)
        btc_df = pd.DataFrame(btc_rows)
        ltf_df = pd.DataFrame(make_candles(n, direction=1, seed=1))

        mgr = TradeLifecycleManager()
        trades, cache = generate_frozen_trades("TESTUSDT", ltf_df, btc_df, None, mgr)
        frozen_by_symbol = {"TESTUSDT": (trades, cache)}

        df = profile_trades(frozen_by_symbol, geometry=SETTLED_GEOMETRY)
        if len(trades) == 0:
            self.assertTrue(df.empty)
            return

        expected_cols = {"symbol", "signal_ts", "direction", "body_to_range", "close_location", "vol_ratio",
                          "htf_break_age_buckets", "ema_spread_atr",
                          "n_tiers_hit", "fill_pct", "r_multiple", "exit_reason", "bars_held", "filled_weight"}
        self.assertEqual(expected_cols, set(df.columns))
        for _, row in df.iterrows():
            self.assertGreaterEqual(row["n_tiers_hit"], 1)
            self.assertGreaterEqual(row["fill_pct"], 0.0)
            self.assertLessEqual(row["fill_pct"], 100.0)

        summary = bucket_summary(df)
        self.assertEqual(summary["n"].sum(), len(df))

        trunc = fill_truncation_whatif(frozen_by_symbol, max_offsets=[1, None])
        self.assertLessEqual(len(trunc), 2)
        for _, row in trunc.iterrows():
            self.assertGreaterEqual(row["n_trades"], 0)


if __name__ == "__main__":
    unittest.main()
