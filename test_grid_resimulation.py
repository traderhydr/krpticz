"""test_grid_resimulation.py -- unit + smoke tests for grid_resimulation.py.

Covers:
  1. `replay_exit`'s exit-side state machine in isolation (hand-built
     FrozenTrade + synthetic bar arrays): SL-first pessimistic resolution,
     TP1/TP2 re-anchoring, the breakeven buffer, the trailing runner
     (including the same-current_sl "trail catches up a bar later" label
     fix), ladder-protection fill cancellation, time-decay, and the
     window-end fallback.
  2. `_extract_frozen_trade`'s construction-time-prefill capture (the
     exact Step 10.11 bug class) using the REAL EntryLadder/PositionState.
  3. `build_grid`'s combinatorics.
  4. A smoke test proving `generate_frozen_trades` + `run_sweep` run
     end-to-end against the real production stack on a small synthetic
     series without crashing and produce a sane, correctly-shaped result.

Run with:  python3 -m unittest test_grid_resimulation -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from entry_ladder import EntryLadder
from grid_resimulation import (
    FRICTION_R_AT_SL_2_20,
    FrozenTrade,
    GeometryParams,
    _extract_frozen_trade,
    build_grid,
    generate_frozen_trades,
    replay_exit,
    run_sweep,
)
from risk_manager import PositionState, TradeLifecycleManager
from test_backtest_engine import make_candles, make_monotonic_trend


def _make_trade(direction="LONG", expected_vwap=100.0, atr_at_signal=2.0,
                 fill_events=((0, 0, 100.0),), ladder_weights=(1.0, 0.0, 0.0, 0.0),
                 window_end_index=10, signal_index=0) -> FrozenTrade:
    return FrozenTrade(
        symbol="TESTUSDT", direction=direction, signal_index=signal_index, signal_ts=1_700_000_000_000,
        window_end_index=window_end_index, expected_vwap=expected_vwap, atr_at_signal=atr_at_signal,
        ladder_levels=(100.0, 95.0, 90.0, 85.0), ladder_weights=ladder_weights, fill_events=fill_events,
    )


class ReplayExitSlFirstTests(unittest.TestCase):
    def test_sl_hit_closes_full_filled_weight_with_friction_applied(self) -> None:
        trade = _make_trade(window_end_index=1)
        n = 2
        high = np.array([0.0, 101.0])
        low = np.array([0.0, 95.0])   # <= initial_sl (96) -- hits
        close = np.array([0.0, 96.0])
        atr = np.array([2.0, 2.0])
        ema20 = np.full(n, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=1.0, tp1_w=0.3, tp2_atr=2.0, tp2_w=0.3, runner_w=0.4)

        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        self.assertEqual(result["exit_reason"], "SL")
        # initial_sl = 100 - 2.0*2.0 = 96; risk_amount = |100-96|*1.0 = 4
        # raw R = (96-100)*1.0/4 = -1.0; friction = FRICTION_R_AT_SL_2_20*(2.20/2.0)
        expected_friction = FRICTION_R_AT_SL_2_20 * (2.20 / 2.0)
        self.assertAlmostEqual(result["r_multiple"], -1.0 - expected_friction, places=6)

    def test_same_bar_sl_and_tp_resolves_sl_first(self) -> None:
        trade = _make_trade(window_end_index=1)
        high = np.array([0.0, 200.0])  # would also clear any TP
        low = np.array([0.0, 90.0])    # <= initial_sl (96)
        close = np.array([0.0, 150.0])
        atr = np.array([2.0, 2.0])
        ema20 = np.full(2, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=1.0, tp1_w=0.3, tp2_atr=2.0, tp2_w=0.3, runner_w=0.4)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        self.assertEqual(result["exit_reason"], "SL", "a bar clearing both must resolve SL first (pessimistic)")


class ReplayExitTpAndRunnerTests(unittest.TestCase):
    def test_tp1_tp2_runner_full_sequence(self) -> None:
        """Hand-computed 4-bar winning sequence: TP1 fires bar1, TP2 fires
        bar2 (runner phase begins same bar), the trail ratchets bar3, and
        a bar4 pullback triggers the SL-first check using the ratcheted
        trail level -- correctly labeled RUNNER_TRAIL, not SL."""
        trade = _make_trade(window_end_index=4)
        # index 0 is an unused placeholder (offset starts at 1).
        high = np.array([0.0, 103.0, 106.0, 110.0, 108.0])
        low = np.array([0.0, 101.0, 104.0, 107.0, 104.0])
        close = np.array([0.0, 102.0, 105.0, 108.0, 105.0])
        atr = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        ema20 = np.full(5, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=1.0, tp1_w=0.3, tp2_atr=2.0, tp2_w=0.3, runner_w=0.4)

        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)

        # bar1: vwap=100, tp1=102 -> hit, closes 0.3 @ 102, pnl=0.6; BE buffer -> sl=100+0.15*2=100.3
        # bar2: tp2=104 -> hit, closes 0.3 @ 104, pnl=1.2; runner starts, extreme=max(105,106)=106, trail=106-4=102 (>100.3, ratchets); low=104 no hit
        # bar3: extreme=max(106,110)=110, trail=110-4=106 (>102, ratchets); low=107, no hit
        # bar4: SL-first check uses sl=106 (set end of bar3); low=104<=106 -> hit, closes remaining 0.4 @ 106, pnl=2.4
        # total pnl = 0.6+1.2+2.4=4.2; risk_amount=|100-96|*1.0=4; raw R=1.05
        expected_friction = FRICTION_R_AT_SL_2_20 * (2.20 / 2.0)
        self.assertEqual(result["exit_reason"], "RUNNER_TRAIL")
        self.assertAlmostEqual(result["r_multiple"], 1.05 - expected_friction, places=6)
        self.assertAlmostEqual(result["filled_weight"], 1.0, places=6)

    def test_breakeven_buffer_only_moves_once_and_never_loosens(self) -> None:
        trade = _make_trade(window_end_index=3)
        high = np.array([0.0, 103.0, 103.0, 200.0])
        low = np.array([0.0, 101.0, 90.0, 90.0])  # bar2 would blow through the ORIGINAL sl (96) but not the BE-moved one (100.3)
        close = np.array([0.0, 102.0, 91.0, 91.0])
        atr = np.array([2.0, 2.0, 2.0, 2.0])
        ema20 = np.full(4, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=1.0, tp1_w=0.3, tp2_atr=5.0, tp2_w=0.3, runner_w=0.4)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        # bar1: TP1 fires @102 (0.3 closed), BE buffer moves sl to 100.3.
        # bar2: low=90 breaches the BE-moved stop (100.3) -- must close the
        # REMAINING 0.7 at 100.3, not at the original 96.
        self.assertEqual(result["exit_reason"], "SL")
        pnl = (102.0 - 100.0) * 0.3 + (100.3 - 100.0) * 0.7
        risk_amount = abs(100.0 - 96.0) * 1.0
        expected_friction = FRICTION_R_AT_SL_2_20 * (2.20 / 2.0)
        self.assertAlmostEqual(result["r_multiple"], pnl / risk_amount - expected_friction, places=6)

    def test_independent_mode_breakeven_buffer_can_fire_before_tp1(self) -> None:
        """Regression for a confirmed real bug (fixed pre-Round-4): in
        INDEPENDENT mode (the default), the scratch-win buffer must
        trigger off its OWN `scratch_trigger_atr_mult` threshold, never
        as a side effect of TP1 firing. With tp1_atr set wide (1.5), a
        bar that clears the 0.60 threshold but NOT tp1's own 1.5 level
        must still move the stop -- and a later bar breaching that early,
        TP1-independent stop must exit at it, not ride the full original SL."""
        trade = _make_trade(window_end_index=2)
        # bar1: high 101.5 clears vwap(100)+0.60*2=101.2, but not tp1's
        # own vwap+1.5*2=103 -- TP1 must NOT fire, but the buffer must.
        high = np.array([0.0, 101.5, 101.0])
        low = np.array([0.0, 99.5, 99.0])
        close = np.array([0.0, 101.0, 99.5])
        atr = np.array([2.0, 2.0, 2.0])
        ema20 = np.full(3, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=1.5, tp1_w=0.3, tp2_atr=5.0, tp2_w=0.3, runner_w=0.4)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        # bar2: low=99 breaches the early buffer-moved stop (100.3), NOT
        # the original initial_sl (96) -- and TP1 never fired.
        self.assertEqual(result["exit_reason"], "SL")
        risk_amount = abs(100.0 - 96.0) * 1.0
        expected_friction = FRICTION_R_AT_SL_2_20 * (2.20 / 2.0)
        expected_r = (100.3 - 100.0) * 1.0 / risk_amount - expected_friction
        self.assertAlmostEqual(result["r_multiple"], expected_r, places=6)

    def test_tp1_gated_mode_never_moves_breakeven_before_tp1_fires(self) -> None:
        """Round 4: TP1_GATED mode (`scratch_trigger_atr_mult=None`) is
        the deliberately-reintroduced alternative to independent mode --
        the exact SAME bars as the independent-mode test above must now
        produce a DIFFERENT outcome, since the buffer never moves without
        TP1 actually firing (which it never does here, tp1_atr=1.5 is
        never reached): the position rides untouched to WINDOW_END."""
        trade = _make_trade(window_end_index=2)
        high = np.array([0.0, 101.5, 101.0])
        low = np.array([0.0, 99.5, 99.0])
        close = np.array([0.0, 101.0, 99.5])
        atr = np.array([2.0, 2.0, 2.0])
        ema20 = np.full(3, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=1.5, tp1_w=0.3, tp2_atr=5.0, tp2_w=0.3, runner_w=0.4,
                                 scratch_trigger_atr_mult=None, scratch_offset_atr_mult=0.15)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        self.assertEqual(result["exit_reason"], "WINDOW_END")
        risk_amount = abs(100.0 - 96.0) * 1.0
        expected_friction = FRICTION_R_AT_SL_2_20 * (2.20 / 2.0)
        expected_r = (99.5 - 100.0) * 1.0 / risk_amount - expected_friction
        self.assertAlmostEqual(result["r_multiple"], expected_r, places=6)

    def test_tp1_gated_mode_moves_breakeven_as_soon_as_tp1_fires(self) -> None:
        """TP1_GATED mode still needs to actually move the buffer once
        TP1 DOES fire -- just never before."""
        trade = _make_trade(window_end_index=2)
        high = np.array([0.0, 103.0, 101.0])  # bar1 clears tp1's own 1.0*2=102 level
        low = np.array([0.0, 101.0, 99.0])
        close = np.array([0.0, 102.5, 99.5])
        atr = np.array([2.0, 2.0, 2.0])
        ema20 = np.full(3, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=1.0, tp1_w=0.3, tp2_atr=5.0, tp2_w=0.3, runner_w=0.4,
                                 scratch_trigger_atr_mult=None, scratch_offset_atr_mult=0.15)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        # bar2: low=99 breaches the TP1-triggered buffer stop (100.3), not the original initial_sl (96).
        self.assertEqual(result["exit_reason"], "SL")
        pnl = (102.0 - 100.0) * 0.3 + (100.3 - 100.0) * 0.7
        risk_amount = abs(100.0 - 96.0) * 1.0
        expected_friction = FRICTION_R_AT_SL_2_20 * (2.20 / 2.0)
        self.assertAlmostEqual(result["r_multiple"], pnl / risk_amount - expected_friction, places=6)

    def test_ladder_protection_cancels_pending_fill_after_tp1_fires(self) -> None:
        """A frozen fill scheduled for a LATER bar must be dropped once
        this geometry's own TP1 already fired -- mirrors production's
        cancel_unfilled_orders."""
        trade = _make_trade(
            fill_events=((0, 0, 100.0), (2, 1, 95.0)),  # tier 1 was going to fill on offset 2
            ladder_weights=(0.5, 0.5, 0.0, 0.0), window_end_index=3,
        )
        high = np.array([0.0, 103.0, 101.0, 101.0])
        low = np.array([0.0, 99.0, 94.0, 94.0])  # bar2's low would have touched tier 1's level (95) if not cancelled
        close = np.array([0.0, 100.0, 96.0, 96.0])
        atr = np.array([2.0, 2.0, 2.0, 2.0])
        ema20 = np.full(4, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=1.0, tp1_w=0.5, tp2_atr=10.0, tp2_w=0.3, runner_w=0.2)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        # TP1 (0.3 weight of the eventually-filled 0.5) fires at bar1 --
        # only tier 0 (0.5 weight) has filled by then, so amt=min(0.5,0.5)=0.5 fully closes it.
        self.assertAlmostEqual(result["filled_weight"], 0.5, places=6, msg="tier 1's later fill must be cancelled, not applied")

    def test_time_decay_closes_stalled_position_on_momentum_flip(self) -> None:
        trade = _make_trade(window_end_index=7)
        n = 8
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)   # never near SL(96) or TP1(way above)
        close = np.full(n, 99.5)  # below ema20 (100) -- unfavorable for LONG
        atr = np.full(n, 2.0)
        ema20 = np.full(n, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=5.0, tp1_w=0.3, tp2_atr=8.0, tp2_w=0.3, runner_w=0.4)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        self.assertEqual(result["exit_reason"], "TIME_DECAY")
        self.assertEqual(result["bars_held"], 6, "must fire on exactly the 6th bar (the default time_decay_bars), not before")

    def test_time_decay_bars_is_swept_not_fixed(self) -> None:
        """Round 6 (Step 11.3): time_decay_bars is now a GeometryParams
        field, not the old hardcoded module constant -- a tighter cutoff
        must fire the check that many bars sooner, given the same
        stalled/against-momentum bar sequence throughout."""
        trade = _make_trade(window_end_index=7)
        n = 8
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        close = np.full(n, 99.5)  # below ema20 (100) -- unfavorable for LONG, every bar
        atr = np.full(n, 2.0)
        ema20 = np.full(n, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=5.0, tp1_w=0.3, tp2_atr=8.0, tp2_w=0.3, runner_w=0.4, time_decay_bars=3)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        self.assertEqual(result["exit_reason"], "TIME_DECAY")
        self.assertEqual(result["bars_held"], 3, "must fire on exactly the configured 3rd bar, not the old hardcoded 6th")

    def test_window_end_closes_remaining_size_at_last_close(self) -> None:
        trade = _make_trade(window_end_index=2)
        high = np.array([0.0, 101.0, 101.0])
        low = np.array([0.0, 99.0, 99.0])
        close = np.array([0.0, 100.5, 103.0])
        atr = np.array([2.0, 2.0, 2.0])
        ema20 = np.full(3, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=5.0, tp1_w=0.3, tp2_atr=8.0, tp2_w=0.3, runner_w=0.4)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        self.assertEqual(result["exit_reason"], "WINDOW_END")
        risk_amount = abs(100.0 - 96.0) * 1.0
        expected_friction = FRICTION_R_AT_SL_2_20 * (2.20 / 2.0)
        self.assertAlmostEqual(result["r_multiple"], (103.0 - 100.0) * 1.0 / risk_amount - expected_friction, places=6)

    def test_construction_time_prefill_at_offset_zero_is_applied_before_bar_loop(self) -> None:
        trade = _make_trade(fill_events=((0, 0, 100.0),), window_end_index=1)
        high = np.array([0.0, 102.0])
        low = np.array([0.0, 99.0])
        close = np.array([0.0, 101.0])
        atr = np.array([2.0, 2.0])
        ema20 = np.full(2, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=5.0, tp1_w=0.3, tp2_atr=8.0, tp2_w=0.3, runner_w=0.4)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        self.assertAlmostEqual(result["filled_weight"], 1.0, places=6)

    def test_zero_fills_returns_none(self) -> None:
        trade = _make_trade(fill_events=(), window_end_index=1)
        atr = np.array([2.0, 2.0])
        ema20 = np.full(2, 100.0)
        high = np.array([0.0, 101.0])
        low = np.array([0.0, 99.0])
        close = np.array([0.0, 100.0])
        params = GeometryParams(sl_atr=2.0, tp1_atr=1.0, tp1_w=0.3, tp2_atr=2.0, tp2_w=0.3, runner_w=0.4)
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        self.assertIsNone(result)

    def test_short_direction_is_symmetric(self) -> None:
        trade = _make_trade(direction="SHORT", window_end_index=1)
        high = np.array([0.0, 101.0])
        low = np.array([0.0, 95.0])
        close = np.array([0.0, 96.0])
        atr = np.array([2.0, 2.0])
        ema20 = np.full(2, 100.0)
        params = GeometryParams(sl_atr=2.0, tp1_atr=1.0, tp1_w=0.3, tp2_atr=2.0, tp2_w=0.3, runner_w=0.4)
        # initial_sl = 100 + 2*2 = 104 (SHORT); high(101) never reaches it -- no SL.
        result = replay_exit(trade, atr_arr=atr, ema20_arr=ema20, high_arr=high, low_arr=low, close_arr=close, params=params)
        self.assertNotEqual(result["exit_reason"], "SL")


class ExtractFrozenTradeTests(unittest.TestCase):
    """The exact Step 10.11 bug class: a construction-time-prefilled tier
    (an immediate market Entry 1) never appears in execution_log, so it
    must be recovered directly from the ladder's post-construction state."""

    def test_construction_time_prefill_is_captured_at_offset_zero(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 90.0, 80.0, 70.0], fills=[True, False, False, False])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=80.0, current_sl=80.0,
                             tp_levels=[110.0, 120.0, None, None, None])
        mgr = TradeLifecycleManager()
        diag = {"expected_vwap": 91.5, "initial_sl": 80.0}
        ft = _extract_frozen_trade("TESTUSDT", signal_index=5, signal_ts=123, window_end_index=20, position=pos, diagnostics=diag, baseline_mgr=mgr)
        self.assertIsNotNone(ft)
        self.assertIn((0, 0, 100.0), ft.fill_events)

    def test_real_new_fill_is_captured_at_offset_bar_index_plus_one(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=80.0, current_sl=80.0,
                             tp_levels=[110.0, 120.0, None, None, None])
        pos.update({"high": 101.0, "low": 99.0, "close": 100.0, "volume": 500, "atr": 2.0})  # bar_index 0 -> fills tier 0
        mgr = TradeLifecycleManager()
        diag = {"expected_vwap": 100.0 * 0.40 + 95.0 * 0.35 + 90.0 * 0.25, "initial_sl": 80.0}
        ft = _extract_frozen_trade("TESTUSDT", signal_index=5, signal_ts=123, window_end_index=20, position=pos, diagnostics=diag, baseline_mgr=mgr)
        self.assertIsNotNone(ft)
        self.assertIn((1, 0, 100.0), ft.fill_events)

    def test_never_filled_ladder_returns_none(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=80.0, current_sl=80.0,
                             tp_levels=[110.0, 120.0, None, None, None])
        mgr = TradeLifecycleManager()
        diag = {"expected_vwap": 92.75, "initial_sl": 80.0}
        ft = _extract_frozen_trade("TESTUSDT", signal_index=5, signal_ts=123, window_end_index=20, position=pos, diagnostics=diag, baseline_mgr=mgr)
        self.assertIsNone(ft)


class BuildGridTests(unittest.TestCase):
    def test_every_combo_weights_sum_to_one(self) -> None:
        grid = build_grid()
        self.assertGreater(len(grid), 0)
        for g in grid:
            self.assertAlmostEqual(g.tp1_w + g.tp2_w + g.runner_w, 1.0, places=6)
            self.assertGreater(g.runner_w, 0.0)

    def test_expected_combo_count(self) -> None:
        # Round 6 (Step 11.3): 1 SL x 1 TP1 x 1 TP2 x 1 scratch_trigger x
        # 1 scratch_offset x 4 time_decay_bars = 4 -- every dimension
        # except time_decay_bars is settled to a single fixed value.
        grid = build_grid()
        self.assertEqual(len(grid), 4)

    def test_grid_uses_step_11_3_settled_values(self) -> None:
        grid = build_grid()
        for g in grid:
            self.assertEqual(g.sl_atr, 2.60)
            self.assertEqual(g.tp1_atr, 1.10)
            self.assertEqual(g.tp2_atr, 2.80)
            self.assertEqual(g.scratch_trigger_atr_mult, 0.60)
            self.assertEqual(g.scratch_offset_atr_mult, 0.35)
        self.assertEqual(sorted(g.time_decay_bars for g in grid), [3, 4, 5, 6])


class EndToEndSmokeTests(unittest.TestCase):
    """Proves generate_frozen_trades + run_sweep run against the REAL
    production stack without crashing and produce a sane result -- not a
    claim about what a real 180-day/3-symbol run would show (this repo's
    sandbox has no market data access; see grid_resimulation.py's own
    module docstring)."""

    def test_small_synthetic_run_produces_a_well_formed_result_table(self) -> None:
        n = 900  # verified fixture size that reliably produces trades under real production gates
        btc_rows = make_monotonic_trend(n, direction=1, seed=99, slope=0.05)
        btc_df = pd.DataFrame(btc_rows)
        ltf_df = pd.DataFrame(make_candles(n, direction=1, seed=1))

        mgr = TradeLifecycleManager()
        trades, cache = generate_frozen_trades("TESTUSDT", ltf_df, btc_df, None, mgr)

        frozen_by_symbol = {
            "TESTUSDT": (
                trades, cache.atr14.to_numpy(), cache.ema20.to_numpy(),
                cache.ltf_df["high"].to_numpy(), cache.ltf_df["low"].to_numpy(), cache.ltf_df["close"].to_numpy(),
            )
        }
        small_grid = [
            GeometryParams(sl_atr=1.20, tp1_atr=1.10, tp1_w=0.20, tp2_atr=2.20, tp2_w=0.40, runner_w=0.40),
            GeometryParams(sl_atr=2.20, tp1_atr=1.50, tp1_w=0.30, tp2_atr=2.80, tp2_w=0.50, runner_w=0.20),
            GeometryParams(sl_atr=2.20, tp1_atr=1.50, tp1_w=0.30, tp2_atr=2.80, tp2_w=0.50, runner_w=0.20,
                           scratch_trigger_atr_mult=None, scratch_offset_atr_mult=0.25),  # TP1_GATED mode
        ]
        results = run_sweep(frozen_by_symbol, small_grid)

        expected_cols = {"sl_atr", "tp1_atr", "tp1_w", "tp2_atr", "tp2_w", "runner_w",
                          "scratch_trigger_atr_mult", "scratch_offset_atr_mult", "time_decay_bars", "n_trades",
                          "win_rate_pct", "avg_win_R", "avg_loss_R", "payoff_ratio", "profit_factor", "ev_R", "max_dd_R",
                          "window_end_pct", "n_time_decay", "time_decay_avg_R", "time_decay_sum_R"}
        if len(trades) == 0:
            self.assertEqual(len(results), 0)
            return
        self.assertEqual(expected_cols, set(results.columns))
        self.assertLessEqual(len(results), len(small_grid))
        for _, row in results.iterrows():
            self.assertGreaterEqual(row["n_trades"], 1)
            self.assertGreaterEqual(row["win_rate_pct"], 0.0)
            self.assertLessEqual(row["win_rate_pct"], 100.0)
            self.assertGreaterEqual(row["max_dd_R"], 0.0)


if __name__ == "__main__":
    unittest.main()
