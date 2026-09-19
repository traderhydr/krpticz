"""test_risk_manager.py -- unit tests for risk_manager.py's target geometry
(Step 10.10/10.11/11.0/11.2).

Covers:
  1. Construction-time SL/TP1/TP2 pricing: fixed ATR multiples off the
     ladder's theoretical expected_vwap (TradeLifecycleManager).
  2. Runtime TP1/TP2 re-anchoring to the REAL filled vwap on every entry
     fill (PositionState._revalidate_tp_geometry).
  3. Ladder protection: either target firing cancels remaining unfilled
     entry tiers.
  4. The active scratch-win stop migration.
  5. Time-decay invalidation.
  6. Step 11.0's trailing-runner tier: activation, ratchet-only trailing,
     same-bar vs. next-bar exit labeling, and the weight-sum invariant.
  7. Two still-relevant regression suites carried over unchanged from
     earlier steps: the zero-weight/gap-through phantom-position fix, and
     the "price runs to a target before any entry fills" phantom-position
     fix -- neither depends on how TP1/TP2 are priced.

Run with:  python3 -m unittest test_risk_manager -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import indicators as ind
from entry_ladder import EntryLadder
from risk_manager import PositionState, TradeLifecycleManager


class _FixedLadderEngine:
    """Returns a fixed, pre-built ladder -- isolates these tests from
    EntryLadderEngine's own (already independently tested) leg-selection
    logic. Step 10.10 no longer prices anything off the impulse leg, so
    this fixture doesn't need to fabricate one."""

    def __init__(self, ladder: EntryLadder) -> None:
        self._ladder = ladder

    def build_ladder(self, df, btc_df, funding_rate, htf_df=None):
        diag = {"built": True, "reason": None, "direction": self._ladder.direction, "impulse_leg": None, "fvg": None}
        return self._ladder, diag


class TradeLifecycleManagerGeometryTests(unittest.TestCase):
    """Step 10.10: initial_sl/tp1/tp2 are fixed ATR multiples off the
    ladder's theoretical expected_vwap -- no Fibonacci-leg pricing."""

    def setUp(self) -> None:
        n = 30
        close = pd.Series(np.linspace(98.0, 100.0, n))
        self.df = pd.DataFrame({"high": close + 0.3, "low": close - 0.3, "close": close, "volume": pd.Series([1000.0] * n)})
        self.btc_df = pd.DataFrame({"close": pd.Series(np.linspace(100, 300, 60))})
        self.atr = float(ind.average_true_range(self.df["high"], self.df["low"], self.df["close"], length=14).iloc[-1])

    @staticmethod
    def _expected_vwap(ladder: EntryLadder) -> float:
        return sum(level * weight for level, weight in zip(ladder.levels, ladder.weights))

    def test_long_geometry_is_fixed_atr_multiples_off_expected_vwap(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        mgr = TradeLifecycleManager(entry_ladder_engine=_FixedLadderEngine(ladder))
        pos, diag = mgr.open_trade(self.df, self.btc_df, funding_rate=0.0)
        self.assertTrue(diag["opened"], diag["reason"])

        expected_vwap = self._expected_vwap(ladder)
        self.assertAlmostEqual(diag["expected_vwap"], expected_vwap, places=6)
        self.assertAlmostEqual(pos.initial_sl, expected_vwap - 2.60 * self.atr, places=6)
        # tp1_atr_mult (1.10) is above the Step 11.0 tp1_min_atr_mult
        # floor (0.30), so tp1_atr_mult itself governs.
        self.assertAlmostEqual(pos.tp_levels[0], expected_vwap + 1.10 * self.atr, places=6)
        self.assertAlmostEqual(pos.tp_levels[1], expected_vwap + 2.80 * self.atr, places=6)
        self.assertIsNone(pos.tp_levels[2])
        self.assertIsNone(pos.tp_levels[3])
        self.assertIsNone(pos.tp_levels[4])
        self.assertAlmostEqual(diag["risk_atr_multiple"], 2.60, places=6)
        self.assertEqual(list(pos.tp_weights), [0.20, 0.50, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(pos.runner_weight, 0.30, places=6)

    def test_short_geometry_is_symmetric(self) -> None:
        ladder = EntryLadder(direction="SHORT", levels=[100.0, 105.0, 110.0, 115.0])
        mgr = TradeLifecycleManager(entry_ladder_engine=_FixedLadderEngine(ladder))
        pos, diag = mgr.open_trade(self.df, self.btc_df, funding_rate=0.0)
        self.assertTrue(diag["opened"], diag["reason"])

        expected_vwap = self._expected_vwap(ladder)
        self.assertAlmostEqual(pos.initial_sl, expected_vwap + 2.60 * self.atr, places=6)
        self.assertAlmostEqual(pos.tp_levels[0], expected_vwap - 1.10 * self.atr, places=6)
        self.assertAlmostEqual(pos.tp_levels[1], expected_vwap - 2.80 * self.atr, places=6)

    def test_custom_atr_multiples_are_respected(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        mgr = TradeLifecycleManager(
            entry_ladder_engine=_FixedLadderEngine(ladder),
            initial_sl_atr_mult=3.0, tp1_atr_mult=1.0, tp1_min_atr_mult=1.10, tp2_atr_mult=2.0,
        )
        pos, diag = mgr.open_trade(self.df, self.btc_df, funding_rate=0.0)
        expected_vwap = self._expected_vwap(ladder)
        self.assertAlmostEqual(pos.initial_sl, expected_vwap - 3.0 * self.atr, places=6)
        # tp1_atr_mult (1.0) is still below tp1_min_atr_mult (1.10) -- floor wins.
        self.assertAlmostEqual(pos.tp_levels[0], expected_vwap + 1.10 * self.atr, places=6)
        self.assertAlmostEqual(pos.tp_levels[1], expected_vwap + 2.0 * self.atr, places=6)
        self.assertAlmostEqual(diag["risk_atr_multiple"], 3.0, places=6)

    def test_atr_not_computable_rejects_the_setup(self) -> None:
        close = pd.Series([100.0, 101.0, 99.0])  # far too few bars for ATR14
        df = pd.DataFrame({"high": close + 0.3, "low": close - 0.3, "close": close, "volume": pd.Series([1000.0] * 3)})
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        mgr = TradeLifecycleManager(entry_ladder_engine=_FixedLadderEngine(ladder))
        pos, diag = mgr.open_trade(df, self.btc_df, funding_rate=0.0)
        self.assertIsNone(pos)
        self.assertIn("ATR", diag["reason"])


class TpReanchoringTests(unittest.TestCase):
    """PositionState._revalidate_tp_geometry: TP1/TP2 are re-anchored to
    EXACTLY tp1_atr_mult/tp2_atr_mult ATRs off the REAL filled vwap on
    every new entry fill -- not just clamped when they'd otherwise be
    invalid. That's the whole point of "priced off the real filled VWAP,"
    not a one-time construction-time guess."""

    def test_tp1_tp2_reanchor_to_real_vwap_on_first_fill_long(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        # Deliberately wrong construction-time guesses -- the real fill
        # must override them completely, not just nudge them into validity.
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=70.0, current_sl=70.0,
                             tp_levels=[999.0, 999.0, None, None, None])
        log = pos.update({"high": 101.0, "low": 99.0, "close": 100.0, "volume": 500, "atr": 5.0})
        clamps = {e["tier_index"]: e for e in log if e["type"] == "TP_CLAMPED"}
        self.assertEqual(set(clamps), {0, 1})
        # tp1_atr_mult defaults to 1.10, above the 0.30 floor -- tp1_atr_mult itself governs.
        self.assertAlmostEqual(pos.tp_levels[0], 100.0 + 1.10 * 5.0, places=6)
        self.assertAlmostEqual(pos.tp_levels[1], 100.0 + 2.80 * 5.0, places=6)

    def test_reanchor_short_symmetric(self) -> None:
        ladder = EntryLadder(direction="SHORT", levels=[100.0, 105.0, 110.0, 115.0])
        pos = PositionState(direction="SHORT", ladder=ladder, initial_sl=120.0, current_sl=120.0,
                             tp_levels=[1.0, 1.0, None, None, None])
        pos.update({"high": 101.0, "low": 99.0, "close": 100.0, "volume": 500, "atr": 5.0})
        self.assertAlmostEqual(pos.tp_levels[0], 100.0 - 1.10 * 5.0, places=6)
        self.assertAlmostEqual(pos.tp_levels[1], 100.0 - 2.80 * 5.0, places=6)

    def test_reanchor_tracks_vwap_shift_across_progressive_fills(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=60.0, current_sl=60.0,
                             tp_levels=[105.0, 110.0, None, None, None])
        pos.update({"high": 101.0, "low": 99.0, "close": 100.0, "volume": 500, "atr": 4.0})
        self.assertAlmostEqual(pos.tp_levels[0], 100.0 + 1.10 * 4.0, places=6)

        # Entry 2 fills too, pulling the real vwap down -- TP1/TP2 must
        # track it exactly, not just stay "still valid."
        log2 = pos.update({"high": 96.0, "low": 94.0, "close": 95.0, "volume": 500, "atr": 4.0})
        clamps2 = [e for e in log2 if e["type"] == "TP_CLAMPED"]
        self.assertEqual(len(clamps2), 2)
        new_vwap = pos.weighted_avg_entry
        self.assertLess(new_vwap, 100.0)
        self.assertAlmostEqual(pos.tp_levels[0], new_vwap + 1.10 * 4.0, places=6)
        self.assertAlmostEqual(pos.tp_levels[1], new_vwap + 2.80 * 4.0, places=6)

    def test_already_filled_tier_is_never_reanchored(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=60.0, current_sl=60.0,
                             tp_levels=[105.0, 110.0, None, None, None])
        pos.tp_fills[0] = True  # pretend TP1 already fired
        log = pos.update({"high": 101.0, "low": 99.0, "close": 100.0, "volume": 500, "atr": 4.0})
        clamps = {e["tier_index"] for e in log if e["type"] == "TP_CLAMPED"}
        self.assertEqual(clamps, {1})
        self.assertEqual(pos.tp_levels[0], 105.0, "a filled tier's price must never be touched again")

    def test_missing_atr_skips_revalidation_without_crashing(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=60.0, current_sl=60.0,
                             tp_levels=[105.0, 110.0, None, None, None])
        log = pos.update({"high": 101.0, "low": 99.0, "close": 100.0, "volume": 500})  # no "atr" key
        self.assertFalse(any(e["type"] == "TP_CLAMPED" for e in log))
        self.assertEqual(pos.tp_levels[0], 105.0, "left as-is when ATR isn't available to compute a re-anchor")


class InvertedTp1RegressionTests(unittest.TestCase):
    """Step 10.11: the exact bug reported from a real 180-day backtest --
    97 trades closed at "TP1" with negative R. Root cause: Step 10.10's
    `_revalidate_tp_geometry` only ran on a bar where `ladder.update_fills`
    reported a NEW fill -- but Entry 1 can also arrive already filled at
    LADDER CONSTRUCTION time (an immediate market order on a fresh
    breakout -- see entry_ladder.py's `_formulate_ladder`, which sets
    `ladder.fills[0] = True` before `PositionState` even exists). That
    fill never flows through `update_fills`, so it never triggered a
    re-anchor: TP1/TP2 stayed priced off the construction-time
    `expected_vwap`, which blends in still-unfilled, deeper retracement
    tiers and, for a LONG, sits BELOW a lone Entry-1 fill (mirror for
    SHORT) -- putting TP1 on the LOSING side of the real filled VWAP."""

    def test_construction_time_prefilled_market_entry_is_corrected_on_first_bar(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 90.0, 80.0, 70.0],
                              fills=[True, False, False, False])  # Entry 1 prefilled at 100, as a market order would be
        # Reproduce the OLD bug's construction-time TP1: priced off
        # expected_vwap (0.40*100 + 0.35*90 + 0.25*80 = 91.5), not the real
        # filled vwap (100 -- only Entry 1 has filled). At the old 0.75 ATR
        # distance (ATR=2), that's 93.0 -- BELOW the real average entry of
        # 100.0, i.e. inverted: reachable by a pure retracement at a loss.
        old_buggy_tp1 = 91.5 + 0.75 * 2.0
        self.assertLess(old_buggy_tp1, 100.0, "sanity check: the old construction-time TP1 really was inverted")
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=80.0, current_sl=80.0,
                             tp_levels=[old_buggy_tp1, 105.0, None, None, None])
        # First bar processed: no NEW entry fill (tier 0 was already filled
        # at construction), so the Step 10.10 gate (`if not
        # newly_filled_tiers: return`) would have skipped re-anchoring
        # forever. Step 10.11 re-anchors every bar instead.
        log = pos.update({"high": 100.5, "low": 99.5, "close": 100.0, "volume": 500, "atr": 2.0})
        clamp = next(e for e in log if e["type"] == "TP_CLAMPED" and e["tier_index"] == 0)
        self.assertEqual(clamp["old_level"], old_buggy_tp1)
        self.assertAlmostEqual(pos.tp_levels[0], 100.0 + 1.10 * 2.0, places=6)
        self.assertGreater(pos.tp_levels[0], pos.weighted_avg_entry, "TP1 must never sit on the losing side of the real filled VWAP")

    def test_open_trade_anchors_tp_to_real_vwap_when_entry1_is_prefilled(self) -> None:
        """End-to-end: TradeLifecycleManager.open_trade() itself must not
        reintroduce the bug by pricing TP1/TP2 off expected_vwap when the
        ladder it received already carries an active fill."""
        n = 30
        close = pd.Series(np.linspace(98.0, 100.0, n))
        df = pd.DataFrame({"high": close + 0.3, "low": close - 0.3, "close": close, "volume": pd.Series([1000.0] * n)})
        btc_df = pd.DataFrame({"close": pd.Series(np.linspace(100, 300, 60))})
        ladder = EntryLadder(direction="LONG", levels=[100.0, 90.0, 80.0, 70.0],
                              fills=[True, False, False, False])
        mgr = TradeLifecycleManager(entry_ladder_engine=_FixedLadderEngine(ladder))
        pos, diag = mgr.open_trade(df, btc_df, funding_rate=0.0)
        self.assertTrue(diag["opened"], diag["reason"])
        real_vwap = ladder.calculate_vwap()
        self.assertEqual(real_vwap, 100.0)
        self.assertGreater(pos.tp_levels[0], real_vwap, "TP1 must be priced off the real filled vwap, not expected_vwap, from construction")

    def test_hard_invariant_clamp_forces_tp1_to_correct_side_long(self) -> None:
        """Defense-in-depth: even a pathologically misconfigured distance
        formula (both tp1_atr_mult and tp1_min_atr_mult negative) must not
        be able to leave TP1 on the losing side of the real filled VWAP --
        the hard invariant clamp forces it back using its own hardcoded
        1.10 ATR distance, independent of either configurable field."""
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=60.0, current_sl=60.0,
                             tp_levels=[105.0, 110.0, None, None, None],
                             tp1_atr_mult=-5.0, tp1_min_atr_mult=-5.0)
        log = pos.update({"high": 101.0, "low": 99.0, "close": 100.0, "volume": 500, "atr": 2.0})
        invariant_clamp = next(e for e in log if e["type"] == "TP1_INVARIANT_CLAMP")
        self.assertAlmostEqual(invariant_clamp["new_level"], 100.0 + 1.10 * 2.0, places=6)
        self.assertGreater(pos.tp_levels[0], pos.weighted_avg_entry)

    def test_hard_invariant_clamp_forces_tp1_to_correct_side_short(self) -> None:
        ladder = EntryLadder(direction="SHORT", levels=[100.0, 105.0, 110.0, 115.0])
        pos = PositionState(direction="SHORT", ladder=ladder, initial_sl=140.0, current_sl=140.0,
                             tp_levels=[95.0, 90.0, None, None, None],
                             tp1_atr_mult=-5.0, tp1_min_atr_mult=-5.0)
        log = pos.update({"high": 101.0, "low": 99.0, "close": 100.0, "volume": 500, "atr": 2.0})
        invariant_clamp = next(e for e in log if e["type"] == "TP1_INVARIANT_CLAMP")
        self.assertAlmostEqual(invariant_clamp["new_level"], 100.0 - 1.10 * 2.0, places=6)
        self.assertLess(pos.tp_levels[0], pos.weighted_avg_entry)

    def test_tp2_distance_is_2_80_atr(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        mgr = TradeLifecycleManager(entry_ladder_engine=_FixedLadderEngine(ladder))
        n = 30
        close = pd.Series(np.linspace(98.0, 100.0, n))
        df = pd.DataFrame({"high": close + 0.3, "low": close - 0.3, "close": close, "volume": pd.Series([1000.0] * n)})
        btc_df = pd.DataFrame({"close": pd.Series(np.linspace(100, 300, 60))})
        atr = float(ind.average_true_range(df["high"], df["low"], df["close"], length=14).iloc[-1])
        pos, diag = mgr.open_trade(df, btc_df, funding_rate=0.0)
        expected_vwap = sum(level * weight for level, weight in zip(ladder.levels, ladder.weights))
        self.assertAlmostEqual(pos.tp_levels[1], expected_vwap + 2.80 * atr, places=6)


class TpLadderProtectionTests(unittest.TestCase):
    """Either TP1 or TP2 firing puts the position in take-profit mode --
    any still-resting entry tier must be cancelled immediately, so a
    later retracement can never fill a tier at worse momentum than what
    already justified taking profit."""

    def test_tp1_hit_cancels_remaining_unfilled_entry_tiers(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=70.0, current_sl=70.0,
                             tp_levels=[105.0, 110.0, None, None, None])
        # Entry 1 (100, 0.40) and Entry 2 (95, 0.35) both fill on a
        # gap-through bar (0.75 total, real vwap = 97.667) -- re-anchoring
        # (see TpReanchoringTests) immediately repriced TP1/TP2 to
        # 97.667 + 1.10*2 = 99.867 and 97.667 + 2.80*2 = 103.267. This bar's
        # high (100.0) clears the first but not the second, so ONLY TP1
        # fires, leaving Entry 3 (90, 0.25) still resting -- which must
        # get cancelled even though price never pulled back to it.
        log = pos.update({"high": 100.0, "low": 94.0, "close": 99.5, "volume": 500, "atr": 2.0})
        cancel_tiers = {e["tier_index"] for e in log if e["type"] == "CANCEL"}
        self.assertEqual(cancel_tiers, {2, 3})
        self.assertEqual(pos.ladder.cancelled, [False, False, True, True])
        self.assertAlmostEqual(pos.open_size, 0.75 - 0.20, places=6)  # tier1+2 filled (0.75) minus TP1's 0.20 weight
        self.assertFalse(pos.closed)

    def test_later_retracement_cannot_fill_a_tier_cancelled_by_tp1(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=70.0, current_sl=70.0,
                             tp_levels=[105.0, 110.0, None, None, None])
        pos.update({"high": 100.0, "low": 94.0, "close": 99.5, "volume": 500, "atr": 2.0})
        self.assertEqual(pos.ladder.fills, [True, True, False, False])
        self.assertFalse(pos.closed)

        # A later bar retraces all the way down through Entry 3/4's
        # levels -- they must NOT fill now that TP1 already cancelled them.
        log2 = pos.update({"high": 100.0, "low": 84.0, "close": 90.0, "volume": 500, "atr": 2.0})
        self.assertEqual([e for e in log2 if e["type"] == "ENTRY_FILL"], [])
        self.assertEqual(pos.ladder.fills, [True, True, False, False])

    def test_tp1_and_tp2_together_exhaust_a_partially_filled_position(self) -> None:
        """If only Entry 1 (0.40) has filled, TP1 (0.20 weight) and TP2
        (0.50 weight) together exceed it -- each close_amount is capped at
        whatever's actually open, and with every other tier cancelled at
        the same time, the position is fully closed in this one bar with
        nothing left for the runner tier to trail."""
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=70.0, current_sl=70.0,
                             tp_levels=[110.0, 120.0, None, None, None])
        pos.update({"high": 111.0, "low": 99.0, "close": 110.5, "volume": 500, "atr": 2.0})
        self.assertTrue(pos.closed)
        self.assertEqual(pos.open_size, 0.0)

    def test_tp2_firing_leaves_the_runner_open_not_fully_closed(self) -> None:
        """Step 11.0: TP1 + TP2 weights sum to 0.70, NOT 1.0 anymore --
        once TP2 also fires (here, on the same wide bar as TP1), whatever
        weight the runner tier is due (proportional to what actually
        filled) is still open, not zero. This bar's low (94.0) is also
        low enough to immediately trigger the runner's freshly-set
        breakeven-buffer-turned-trail stop, so the position DOES end up
        closed by the end of this same bar -- but via a real RUNNER_EXIT,
        not because TP1+TP2 alone summed to the full position."""
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=70.0, current_sl=70.0,
                             tp_levels=[105.0, 110.0, None, None, None])
        # Entry 1+2 fill (vwap=97.667); high=105 clears both re-anchored
        # targets (99.867 and 103.267), so TP1 closes 0.20 of the 0.75
        # filled and TP2 closes another 0.50 (a REAL TP_HIT, not
        # TP_HIT_NO_SIZE), leaving 0.05 for the runner -- which this same
        # bar's low (94.0) immediately stops out.
        log = pos.update({"high": 105.0, "low": 94.0, "close": 104.0, "volume": 500, "atr": 2.0})
        self.assertTrue(any(e["type"] == "TP_HIT" and e.get("tier_index") == 0 for e in log))
        self.assertTrue(any(e["type"] == "TP_HIT" and e.get("tier_index") == 1 for e in log))
        self.assertTrue(any(e["type"] == "RUNNER_EXIT" for e in log), "the 0.05 runner remainder must be closed via the trail, not left dangling")
        self.assertTrue(pos.closed)
        self.assertAlmostEqual(pos.open_size, 0.0, places=9)


class RunnerTrailTests(unittest.TestCase):
    """Step 11.0's new trailing-runner tier: only activates once TP1 AND
    TP2 have both fired, ratchets an ATR chandelier stop favorably only
    (never loosens, even if a later bar's ATR would otherwise compute a
    worse level), and labels a hit RUNNER_EXIT -- whether caught same-bar
    by the runner step itself or a bar later by the ordinary step-1
    SL-first check (see that step's own comment for why)."""

    @staticmethod
    def _post_tp1_tp2_position(current_sl: float = 95.0) -> PositionState:
        """A ladder fully filled at a flat vwap of 100.0 (three equal-price
        tiers, so the exact weighting doesn't matter), with TP1 (0.20) and
        TP2 (0.50) already recorded as fired and closed -- leaving exactly
        the 0.30 runner_weight open. breakeven_moved=True so the separate
        scratch-win buffer mechanic can't interfere with these tests."""
        ladder = EntryLadder(direction="LONG", levels=[100.0, 100.0, 100.0, 100.0], fills=[True, True, True, False])
        return PositionState(
            direction="LONG", ladder=ladder, initial_sl=90.0, current_sl=current_sl,
            tp_levels=[101.2, 106.4, None, None, None], tp_fills=[True, True, False, False, False],
            realized_weight=0.70, breakeven_moved=True,
        )

    def test_runner_not_active_until_both_tp1_and_tp2_have_fired(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 100.0, 100.0, 100.0], fills=[True, True, True, False])
        pos = PositionState(
            direction="LONG", ladder=ladder, initial_sl=90.0, current_sl=95.0,
            tp_levels=[101.2, 999.0, None, None, None], tp_fills=[True, False, False, False, False],
            realized_weight=0.20, breakeven_moved=True,
        )
        log = pos.update({"high": 105.0, "low": 103.0, "close": 104.0, "volume": 500, "atr": 2.0})
        self.assertFalse(pos.runner_active)
        self.assertEqual(pos.current_sl, 95.0, "no trailing should happen before TP2 has also fired")
        self.assertFalse(any(e["type"] == "RUNNER_EXIT" for e in log))

    def test_runner_activates_ratchets_and_a_later_bar_hit_is_labeled_runner_exit(self) -> None:
        pos = self._post_tp1_tp2_position(current_sl=95.0)

        pos.update({"high": 105.0, "low": 103.0, "close": 104.0, "volume": 500, "atr": 2.0})
        self.assertTrue(pos.runner_active)
        self.assertAlmostEqual(pos.current_sl, 101.0, places=6)  # extreme=105, trail=105-2*2=101 (> initial 95)

        pos.update({"high": 110.0, "low": 107.0, "close": 108.0, "volume": 500, "atr": 2.0})
        self.assertAlmostEqual(pos.current_sl, 106.0, places=6)  # extreme=110, trail=110-2*2=106 (> 101)
        self.assertFalse(pos.closed)

        # A reversal bar breaches the trail SET AT THE END OF THE PRIOR
        # bar -- caught by step 1's ordinary SL-first check, not step 3b's
        # own same-bar check, exercising the labeling fix directly.
        log3 = pos.update({"high": 105.0, "low": 103.0, "close": 104.0, "volume": 500, "atr": 2.0})
        exit_events = [e for e in log3 if e["type"] == "RUNNER_EXIT"]
        self.assertEqual(len(exit_events), 1)
        self.assertFalse(any(e["type"] == "SL_HIT" for e in log3), "a runner-phase stop-out must never be labeled SL_HIT")
        self.assertEqual(exit_events[0]["price"], 106.0)
        self.assertTrue(pos.closed)
        self.assertAlmostEqual(pos.open_size, 0.0, places=9)

    def test_runner_never_loosens_even_if_a_later_bars_atr_would_compute_a_worse_level(self) -> None:
        pos = self._post_tp1_tp2_position(current_sl=95.0)

        pos.update({"high": 105.0, "low": 103.0, "close": 104.0, "volume": 500, "atr": 2.0})
        self.assertAlmostEqual(pos.current_sl, 101.0, places=6)

        # Extreme doesn't advance (high 104 < prior extreme 105) AND ATR
        # jumps to 3.0 -- naively recomputing trail = 105 - 3.0*2 = 99,
        # WORSE than the already-ratcheted 101. The stop must stay at 101,
        # and a same-bar low breaching 101 (but not 99) must still trigger.
        log2 = pos.update({"high": 104.0, "low": 100.0, "close": 102.0, "volume": 500, "atr": 3.0})
        exit_events = [e for e in log2 if e["type"] == "RUNNER_EXIT"]
        self.assertEqual(len(exit_events), 1)
        self.assertEqual(exit_events[0]["price"], 101.0, "must exit at the never-loosened 101, not a recomputed-but-worse 99")

    def test_weight_sum_invariant_includes_runner_weight(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        with self.assertRaises(ValueError):
            PositionState(
                direction="LONG", ladder=ladder, initial_sl=90.0, current_sl=90.0,
                tp_levels=[105.0, 110.0, None, None, None],
                tp_weights=[0.20, 0.50, 0.0, 0.0, 0.0], runner_weight=0.20,  # sums to 0.90, not 1.0
            )


class ScratchWinBufferTests(unittest.TestCase):
    """Step 10.10: once price has moved scratch_win_trigger_atr_mult ATRs
    in profit from the real filled vwap, current_sl migrates (once) to
    filled_vwap +/- breakeven_buffer_atr_mult * ATR -- a guaranteed small
    win if later stopped out, never a bare breakeven or worse, and never
    loosens an already-better stop."""

    def test_long_scratch_win_triggers_and_guarantees_a_small_win(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=70.0, current_sl=70.0,
                             tp_levels=[200.0, 210.0, None, None, None])  # far away -- won't fire this bar
        # Entry 1 fills at 100; price then reaches 100 + 0.60*2 = 101.2 in profit.
        log = pos.update({"high": 101.5, "low": 99.5, "close": 101.0, "volume": 500, "atr": 2.0})
        moves = [e for e in log if e["type"] == "BREAKEVEN_MOVE"]
        self.assertEqual(len(moves), 1)
        self.assertTrue(pos.breakeven_moved)
        self.assertAlmostEqual(pos.current_sl, 100.0 + 0.35 * 2.0, places=6)

    def test_scratch_win_never_loosens_an_already_better_stop(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=70.0, current_sl=101.0,  # already better than the 100.7 candidate
                             tp_levels=[200.0, 210.0, None, None, None])
        log = pos.update({"high": 101.5, "low": 99.5, "close": 101.0, "volume": 500, "atr": 2.0})
        moves = [e for e in log if e["type"] == "BREAKEVEN_MOVE"]
        self.assertEqual(len(moves), 1)
        self.assertEqual(pos.current_sl, 101.0, "an already-tighter stop must not be loosened by the migration")
        self.assertTrue(pos.breakeven_moved)

        # One-shot: a later bar must not re-trigger the migration.
        log2 = pos.update({"high": 105.0, "low": 103.0, "close": 104.0, "volume": 500, "atr": 2.0})
        self.assertEqual([e for e in log2 if e["type"] == "BREAKEVEN_MOVE"], [])

    def test_short_scratch_win_symmetric(self) -> None:
        ladder = EntryLadder(direction="SHORT", levels=[100.0, 105.0, 110.0, 115.0])
        pos = PositionState(direction="SHORT", ladder=ladder, initial_sl=130.0, current_sl=130.0,
                             tp_levels=[10.0, 5.0, None, None, None])
        log = pos.update({"high": 100.5, "low": 98.5, "close": 99.0, "volume": 500, "atr": 2.0})
        moves = [e for e in log if e["type"] == "BREAKEVEN_MOVE"]
        self.assertEqual(len(moves), 1)
        self.assertAlmostEqual(pos.current_sl, 100.0 - 0.35 * 2.0, places=6)

    def test_no_migration_before_threshold_reached(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=70.0, current_sl=70.0,
                             tp_levels=[200.0, 210.0, None, None, None])
        # Only 0.5 in profit, short of the 0.60*2=1.2 threshold.
        log = pos.update({"high": 100.5, "low": 99.5, "close": 100.0, "volume": 500, "atr": 2.0})
        self.assertEqual([e for e in log if e["type"] == "BREAKEVEN_MOVE"], [])
        self.assertEqual(pos.current_sl, 70.0)

    def test_no_migration_without_any_fill(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0], fills=[False, False, False, False])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=70.0, current_sl=70.0,
                             tp_levels=[200.0, 210.0, None, None, None])
        log = pos.update({"high": 101.0, "low": 100.5, "close": 100.8, "volume": 500, "atr": 2.0})
        self.assertEqual([e for e in log if e["type"] == "BREAKEVEN_MOVE"], [])


class TimeDecayInvalidationTests(unittest.TestCase):
    """Step 10.10: if TP1 hasn't fired within time_decay_bars bars and
    this bar's close is on the wrong side of ema20, close out at market
    rather than let a stalled trade drift toward the wide 2.20 ATR stop."""

    @staticmethod
    def _stalled_position(direction: str = "LONG", time_decay_bars: int = 3) -> PositionState:
        if direction == "LONG":
            ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0], fills=[True, False, False, False])
            return PositionState(direction="LONG", ladder=ladder, initial_sl=70.0, current_sl=70.0,
                                  tp_levels=[200.0, 210.0, None, None, None], time_decay_bars=time_decay_bars)
        ladder = EntryLadder(direction="SHORT", levels=[100.0, 105.0, 110.0, 115.0], fills=[True, False, False, False])
        return PositionState(direction="SHORT", ladder=ladder, initial_sl=130.0, current_sl=130.0,
                              tp_levels=[10.0, 5.0, None, None, None], time_decay_bars=time_decay_bars)

    def test_long_closes_at_market_on_momentum_flip_after_time_decay_bars(self) -> None:
        pos = self._stalled_position("LONG", time_decay_bars=3)
        flat_bar = {"high": 100.5, "low": 99.5, "close": 100.0, "volume": 500, "atr": 2.0, "ema20": 99.0}
        pos.update(flat_bar)  # bar 1
        pos.update(flat_bar)  # bar 2
        # bar 3: bars_processed reaches time_decay_bars, close now BELOW ema20.
        log = pos.update({"high": 100.0, "low": 97.0, "close": 97.5, "volume": 500, "atr": 2.0, "ema20": 99.0})
        exit_events = [e for e in log if e["type"] == "TIME_DECAY_EXIT"]
        self.assertEqual(len(exit_events), 1)
        self.assertTrue(pos.closed)
        self.assertEqual(pos.open_size, 0.0)

    def test_no_exit_if_tp1_already_fired(self) -> None:
        pos = self._stalled_position("LONG", time_decay_bars=1)
        pos.tp_fills[0] = True
        log = pos.update({"high": 100.0, "low": 97.0, "close": 97.5, "volume": 500, "atr": 2.0, "ema20": 99.0})
        self.assertEqual([e for e in log if e["type"] == "TIME_DECAY_EXIT"], [])

    def test_no_exit_before_time_decay_bars_elapsed(self) -> None:
        pos = self._stalled_position("LONG", time_decay_bars=6)
        log = pos.update({"high": 100.0, "low": 97.0, "close": 97.5, "volume": 500, "atr": 2.0, "ema20": 99.0})
        self.assertEqual([e for e in log if e["type"] == "TIME_DECAY_EXIT"], [])
        self.assertFalse(pos.closed)

    def test_no_exit_if_momentum_still_favorable(self) -> None:
        pos = self._stalled_position("LONG", time_decay_bars=1)
        log = pos.update({"high": 101.0, "low": 99.5, "close": 100.5, "volume": 500, "atr": 2.0, "ema20": 99.0})  # close ABOVE ema20
        self.assertEqual([e for e in log if e["type"] == "TIME_DECAY_EXIT"], [])
        self.assertFalse(pos.closed)

    def test_short_symmetric(self) -> None:
        pos = self._stalled_position("SHORT", time_decay_bars=1)
        # close ABOVE ema20 is against a SHORT.
        log = pos.update({"high": 103.0, "low": 99.0, "close": 102.5, "volume": 500, "atr": 2.0, "ema20": 101.0})
        exit_events = [e for e in log if e["type"] == "TIME_DECAY_EXIT"]
        self.assertEqual(len(exit_events), 1)
        self.assertTrue(pos.closed)

    def test_missing_ema20_skips_the_check(self) -> None:
        pos = self._stalled_position("LONG", time_decay_bars=1)
        log = pos.update({"high": 100.0, "low": 97.0, "close": 97.5, "volume": 500, "atr": 2.0})  # no ema20
        self.assertEqual([e for e in log if e["type"] == "TIME_DECAY_EXIT"], [])
        self.assertFalse(pos.closed)


class ZeroWeightPhantomPositionTests(unittest.TestCase):
    """Regression, two layers (carried over unchanged -- neither depends
    on how TP1/TP2 are priced):

    1. Root cause: EntryLadder.update_fills used to check whether a level
       fell WITHIN one specific bar's own high/low range (`lo <= level <=
       hi`), not "has price reached at least this far" -- so a single fast
       bar could gap straight past the shallower tiers' levels (never
       straddling them) while still landing on Entry 4's alone. With the
       calibrated [0.40, 0.35, 0.25, 0.0] ladder, Entry 4 carries 0 weight,
       so calculate_vwap()'s total_weight was 0 -- a ZeroDivisionError
       that crashed a real 6-month live-network backtest (SOLUSDT/
       DOGEUSDT/ETHUSDT) the first time this code path met real market
       data. Fixed with a direction-aware, one-sided touch check (`lo <=
       level` for LONG, `hi >= level` for SHORT) -- since levels are
       always monotonically ordered shallow-to-deep, touching a deep level
       now provably also touches every shallower one in the SAME bar, so a
       real (positive-weight) fill always accompanies a deep one. These
       first two tests prove that directly: the exact gap that used to
       strand Entry 4 alone now fills Entry 1/2 (and, direction-mirrored,
       Entry 3) right alongside it.
    2. Defense-in-depth: `has_active_fills`/`calculate_vwap()` still
       report "nothing meaningfully filled" if a ladder somehow ends up
       with ONLY zero-weight tiers marked filled, independent of whether
       `update_fills` can still reach that state.
    """

    def test_long_gap_through_bar_fills_shallow_tiers_alongside_the_deep_one(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])  # default weights: entry4 = 0.0
        # [83, 87] never straddles entry1/2/3 (100/95/90) the way the OLD
        # check required -- this is exactly the gap that used to strand
        # Entry 4 alone. bar_delta here is negative (close near the low),
        # so Entry 3 specifically stays blocked by its own extra gate --
        # proving Entry 1/2 fill on the one-sided check ALONE, not because
        # every tier's gate happened to agree.
        newly_filled = ladder.update_fills({"high": 87.0, "low": 83.0, "close": 84.0, "volume": 500})
        self.assertEqual(newly_filled, [0, 1, 3])
        self.assertEqual(ladder.fills, [True, True, False, True])
        self.assertTrue(ladder.has_active_fills)
        vwap = ladder.calculate_vwap()
        self.assertIsNotNone(vwap)
        self.assertAlmostEqual(vwap, (100.0 * 0.40 + 95.0 * 0.35) / 0.75, places=6)

    def test_short_gap_through_bar_fills_shallow_tiers_alongside_the_deep_one(self) -> None:
        ladder = EntryLadder(direction="SHORT", levels=[85.0, 90.0, 95.0, 100.0])  # ascending ordering for SHORT
        # [98, 102] blows straight past entry1/2 (85/90) the way the OLD
        # check required them to be independently straddled. close near
        # the LOW makes bar_delta supportive for SHORT (delta < 0), so
        # Entry 3 (95) fills alongside everything else here.
        newly_filled = ladder.update_fills({"high": 102.0, "low": 98.0, "close": 98.5, "volume": 500})
        self.assertEqual(newly_filled, [0, 1, 2, 3])
        self.assertTrue(ladder.has_active_fills)
        self.assertIsNotNone(ladder.calculate_vwap())

    def test_has_active_fills_and_vwap_stay_none_if_only_a_zero_weight_tier_is_ever_marked_filled(self) -> None:
        """Defense-in-depth: hand-construct the pathological state directly
        (bypassing update_fills, which can no longer reach it) to prove
        the guard itself is still correct on its own, independent of the
        fix above."""
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0], fills=[False, False, False, True])
        self.assertFalse(ladder.has_active_fills)
        self.assertIsNone(ladder.calculate_vwap(), "a fill with zero total weight must report 'nothing meaningfully filled', not divide by zero")

    def test_position_state_update_does_not_crash_and_opens_real_weight_on_gap_through_bar(self) -> None:
        """The exact crash path -- update_entry_fill -> _revalidate_tp_geometry
        -> weighted_avg_entry -> calculate_vwap() -- now resolves to a
        genuinely-filled position instead of a phantom one."""
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=80.0, current_sl=80.0,
                             tp_levels=[110.0, 115.0, None, None, None])
        log = pos.update({"high": 87.0, "low": 83.0, "close": 84.0, "volume": 500, "atr": 2.0})  # must not raise
        filled_tiers = {e["tier_index"] for e in log if e["type"] == "ENTRY_FILL"}
        self.assertEqual(filled_tiers, {0, 1, 3})
        self.assertIsNotNone(pos.weighted_avg_entry)
        self.assertAlmostEqual(pos.open_size, 0.75, places=6)  # 0.40 + 0.35 + 0.0 (entry4)


class TpHitBeforeAnyEntryFillTests(unittest.TestCase):
    """A second, distinct phantom-position path from the same family as
    `ZeroWeightPhantomPositionTests` above (carried over unchanged), found
    auditing a real 180-day backtest's exported trade CSV: a chunk of
    "closed trades" had every financial field (avg_entry, r_multiple,
    fees, exit_reason) blank.

    Root cause: the passive TP1/TP2 check (`PositionState.update`, step 3)
    tests each TP level purely against this bar's high/low -- it never
    checks whether any entry has actually filled, and cancels remaining
    unfilled tiers unconditionally once its level is touched. So a
    resting-limit ladder that never gets a pullback before price runs
    straight up to a TP level gets every entry tier cancelled in one
    shot -- satisfying `no_pending_entries` -- while `open_size` is
    trivially 0 (nothing ever filled), which marks the position `closed`
    with zero risk ever taken and zero PnL ever possible.

    That half of the bug is legitimate cleanup (a setup that never
    triggered its entries before running past its own take-profit target
    is dead and should stop blocking the symbol) -- the actual defect is
    downstream, in `DryRunHarness.on_position_closed`, which used to
    record this as a full trade anyway. This test proves the PositionState
    side of the scenario is real and reachable; the harness-side fix and
    its own regression test live in test_live_runner.py.
    """

    def test_price_running_to_tp_before_any_pullback_closes_with_zero_filled_weight(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[95.0, 90.0, 85.0, 80.0], fills=[False, False, False, False])
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=75.0, current_sl=75.0,
                             tp_levels=[105.0, 110.0, None, None, None])
        # Price never dips into the entry ladder (low stays at 100, above
        # every entry level) -- it runs straight up through TP1 and TP2.
        log = pos.update({"high": 112.0, "low": 100.0, "close": 111.0, "volume": 500, "atr": 2.0})

        self.assertEqual([e for e in log if e["type"] == "ENTRY_FILL"], [])
        self.assertIsNone(pos.weighted_avg_entry)
        self.assertEqual(pos.open_size, 0.0)
        self.assertTrue(any(e["type"] == "CANCEL" for e in log))
        self.assertTrue(pos.closed, "no pending entries + open_size==0 marks it closed even though nothing ever filled")


if __name__ == "__main__":
    unittest.main()
