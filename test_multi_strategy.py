"""test_multi_strategy.py -- integration & concurrency tests for
multi_strategy_manager.py (KRYPTIC Step 9).

Uses THREE KrypticEngine instances (under different strategy_name values)
to stand in for KRYPTIC/ZENITH/GEM -- this repo's actual ZENITH/GEM
strategies predate this KRYPTIC track and use a different architecture
(see multi_strategy_manager.py's module docstring for why). These tests
prove the HARNESS's own coordination logic (data-fetch dedup, hedge-mode
isolation, targeted cancellation, margin arbitration) -- they say nothing
about ZENITH/GEM's actual trading logic, which this module never touches.

Run with:  python3 -m unittest test_multi_strategy -v
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_collector import MockExchangeStream
from engine import KrypticEngine
from entry_ladder import EntryLadder, EntryLadderEngine
from multi_strategy_manager import (
    MultiStrategyManager, OrderPayload, RiskArbiter,
    make_cl_ord_id, order_side_for, pos_side_for, short_symbol,
)
from risk_manager import PositionState, TradeLifecycleManager


class AlwaysAllowRegimeFilter:
    """Stub isolating these tests from RegimeFilter's own independently
    tested gates (Step 2) -- this suite is about the multi-strategy
    harness, not re-proving the regime gate works."""

    def evaluate_market_conditions(self, df, btc_df, funding_rate, *, direction):
        return True, {"allowed": True, "direction": direction, "failed_gates": [], "gates": {}, "stub": True}


def make_candles(n, direction, seed=1, wiggle_amp=8.0, slope=0.5, start_ms=1_700_000_000_000):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 100 + direction * t * slope
    wiggle = wiggle_amp * np.sin(t / 6.0) + rng.normal(0, 0.3, n)
    close = base + wiggle
    high = close + np.abs(rng.normal(0.5, 0.1, n))
    low = close - np.abs(rng.normal(0.5, 0.1, n))
    opens = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(1000, 5000, n)
    step = 900_000
    start_ms = (start_ms // step) * step
    return [
        {"ts": start_ms + i * step, "open": float(opens[i]), "high": float(high[i]),
         "low": float(low[i]), "close": float(close[i]), "volume": float(vol[i])}
        for i in range(n)
    ]


class OrderTaggingTests(unittest.TestCase):
    def test_cl_ord_id_matches_spec_examples(self) -> None:
        self.assertEqual(make_cl_ord_id("KRYPTIC", "LONG", "BTCUSDT", "E1", 1726510000), "KRP_L_BTC_E1_1726510000")
        self.assertEqual(make_cl_ord_id("ZENITH", "SHORT", "BTCUSDT", "E2", 1726510000), "ZEN_S_BTC_E2_1726510000")

    def test_pos_side_and_order_side_mapping(self) -> None:
        self.assertEqual(pos_side_for("LONG"), "long")
        self.assertEqual(pos_side_for("SHORT"), "short")
        self.assertEqual(order_side_for("LONG", is_exit=False), "buy")
        self.assertEqual(order_side_for("LONG", is_exit=True), "sell")
        self.assertEqual(order_side_for("SHORT", is_exit=False), "sell")
        self.assertEqual(order_side_for("SHORT", is_exit=True), "buy")

    def test_short_symbol_strips_quote_currency(self) -> None:
        self.assertEqual(short_symbol("BTCUSDT"), "BTC")
        self.assertEqual(short_symbol("ETHUSDT"), "ETH")


class SharedOrderBookTests(unittest.TestCase):
    def setUp(self) -> None:
        from multi_strategy_manager import SharedOrderBook
        self.book = SharedOrderBook()
        self.book.place(OrderPayload("KRP_L_BTC_E1_100", "KRYPTIC", "BTCUSDT", "long", "buy", "E1", 100.0, 0.15))
        self.book.place(OrderPayload("KRP_L_BTC_E2_100", "KRYPTIC", "BTCUSDT", "long", "buy", "E2", 95.0, 0.25))
        self.book.place(OrderPayload("ZEN_S_BTC_E1_100", "ZENITH", "BTCUSDT", "short", "sell", "E1", 110.0, 0.5))

    def test_targeted_cancel_by_strategy_and_side(self) -> None:
        cancelled = self.book.cancel_by_strategy_and_side("KRYPTIC", "LONG", "BTCUSDT")
        self.assertEqual({o.cl_ord_id for o in cancelled}, {"KRP_L_BTC_E1_100", "KRP_L_BTC_E2_100"})
        remaining = self.book.open_orders()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].cl_ord_id, "ZEN_S_BTC_E1_100")

    def test_cancel_does_not_touch_other_strategy_or_opposite_side(self) -> None:
        self.book.place(OrderPayload("KRP_S_BTC_E1_100", "KRYPTIC", "BTCUSDT", "short", "sell", "E1", 120.0, 0.15))
        self.book.cancel_by_strategy_and_side("KRYPTIC", "LONG", "BTCUSDT")
        remaining_ids = {o.cl_ord_id for o in self.book.open_orders()}
        self.assertEqual(remaining_ids, {"ZEN_S_BTC_E1_100", "KRP_S_BTC_E1_100"})


class RiskArbiterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arbiter = RiskArbiter(account_equity=10000.0, global_risk_ceiling_pct=6.0,
                                    strategy_budgets_pct={"ZENITH": 35.0, "GEM": 35.0, "KRYPTIC": 30.0})

    def test_approves_within_budget_and_ceiling(self) -> None:
        approved, reason = self.arbiter.request_margin("KRYPTIC", 200.0)
        self.assertTrue(approved)
        self.assertIsNone(reason)

    def test_global_ceiling_blocks_even_within_strategy_budget(self) -> None:
        self.arbiter.request_margin("KRYPTIC", 200.0)
        approved, reason = self.arbiter.request_margin("KRYPTIC", 500.0)  # 700 > 6% of 10000 = 600
        self.assertFalse(approved)
        self.assertIn("global portfolio risk ceiling", reason)

    def test_strategy_budget_blocks_independent_of_global_ceiling(self) -> None:
        arbiter = RiskArbiter(account_equity=10000.0, global_risk_ceiling_pct=50.0, strategy_budgets_pct={"KRYPTIC": 10.0})
        approved, reason = arbiter.request_margin("KRYPTIC", 2000.0)  # 2000 > 10% of 10000 = 1000
        self.assertFalse(approved)
        self.assertIn("strategy budget exceeded", reason)

    def test_release_margin_frees_capacity(self) -> None:
        self.arbiter.request_margin("KRYPTIC", 200.0)
        self.arbiter.release_margin("KRYPTIC", 200.0)
        self.assertEqual(self.arbiter.allocated_for("KRYPTIC"), 0.0)
        approved, _ = self.arbiter.request_margin("KRYPTIC", 500.0)
        self.assertTrue(approved)

    def test_unlisted_strategy_has_zero_budget(self) -> None:
        approved, reason = self.arbiter.request_margin("UNKNOWN", 1.0)
        self.assertFalse(approved)
        self.assertIn("strategy budget exceeded", reason)


class MultiStrategyManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def _make_krptic_pipeline(self, exchange, symbol="BTCUSDT", strategy_name="KRYPTIC", warmup_bars=370):
        entry_engine = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter())
        trade_manager = TradeLifecycleManager(entry_ladder_engine=entry_engine)
        return KrypticEngine(exchange=exchange, symbol=symbol, trade_manager=trade_manager,
                              strategy_name=strategy_name, state_dir=self.tmpdir, warmup_bars=warmup_bars)

    async def test_shared_data_bus_deduplicates_rest_calls(self) -> None:
        warm = make_candles(370, direction=1, seed=1)

        class CountingExchange(MockExchangeStream):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.historical_calls = 0

            async def historical_klines(self, symbol, interval, limit):
                self.historical_calls += 1
                return await super().historical_klines(symbol, interval, limit)

        exchange = CountingExchange(historical=warm, live=[], funding_rate=0.02, btc_historical=warm)
        kryptic_raw = self._make_krptic_pipeline(exchange, strategy_name="KRYPTIC")
        zenith_raw = self._make_krptic_pipeline(exchange, strategy_name="ZENITH")

        arbiter = RiskArbiter(account_equity=10000.0, global_risk_ceiling_pct=50.0,
                               strategy_budgets_pct={"KRYPTIC": 30.0, "ZENITH": 35.0})
        msm = MultiStrategyManager(exchange=exchange, arbiter=arbiter, state_dir=self.tmpdir, warmup_bars=370)
        msm.register_engine(kryptic_raw)
        msm.register_engine(zenith_raw)

        await msm.warmup_all()

        self.assertLessEqual(exchange.historical_calls, 2, "two engines on the same symbol must not each fetch independently")
        self.assertEqual(len(kryptic_raw.ltf_buffer), 370)
        self.assertEqual(len(zenith_raw.ltf_buffer), 370)

    async def test_dual_side_simultaneous_execution_independent_state_no_collision(self) -> None:
        """The spec's headline concurrency scenario: KRYPTIC Long + ZENITH
        Short on the same symbol, processed by the same shared bar, with
        fully independent fills and correctly-scoped, distinct state files."""
        warm = [{"ts": 1_700_000_000_000 + i * 900_000, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0} for i in range(310)]
        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.0)
        kryptic_raw = self._make_krptic_pipeline(exchange, strategy_name="KRYPTIC", warmup_bars=300)
        zenith_raw = self._make_krptic_pipeline(exchange, strategy_name="ZENITH", warmup_bars=300)

        arbiter = RiskArbiter(account_equity=10000.0, global_risk_ceiling_pct=50.0,
                               strategy_budgets_pct={"KRYPTIC": 30.0, "ZENITH": 35.0})
        msm = MultiStrategyManager(exchange=exchange, arbiter=arbiter, state_dir=self.tmpdir, warmup_bars=300)
        managed_kryptic = msm.register_engine(kryptic_raw)
        managed_zenith = msm.register_engine(zenith_raw)
        await msm.warmup_all()

        # tp1/tp2_atr_mult deliberately overridden far beyond default (0.75/
        # 1.50) -- Step 10.10 re-anchors TP1/TP2 to the REAL filled vwap on
        # every new fill (see TpReanchoringTests), and this bar's gap-through
        # fill drops that real vwap well below the level these targets were
        # constructed with; at the DEFAULT multiples the re-anchored TP1
        # would already sit inside this bar's range and fire immediately,
        # which isn't what this test is checking (independent per-strategy
        # state, not TP-hit mechanics -- that's covered in test_risk_manager.py).
        kryptic_ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        kryptic_pos = PositionState(direction="LONG", ladder=kryptic_ladder, initial_sl=80.0, current_sl=80.0,
                                     tp_levels=[105.0, 110.0, None, None, None], tp1_atr_mult=10.0, tp2_atr_mult=15.0)
        kryptic_raw.active_position = kryptic_pos
        managed_kryptic._rescope_for_direction("LONG")
        kryptic_raw._persist(reason="test_setup")

        zenith_ladder = EntryLadder(direction="SHORT", levels=[102.0, 105.0, 110.0, 115.0])
        zenith_pos = PositionState(direction="SHORT", ladder=zenith_ladder, initial_sl=120.0, current_sl=120.0,
                                    tp_levels=[90.0, 85.0, None, None, None], tp1_atr_mult=10.0, tp2_atr_mult=15.0)
        zenith_raw.active_position = zenith_pos
        managed_zenith._rescope_for_direction("SHORT")
        zenith_raw._persist(reason="test_setup")

        bar = {"ts": warm[-1]["ts"] + 900_000, "open": 100.0, "high": 101.0, "low": 94.0, "close": 96.0, "volume": 500}
        msm.on_candle_close("BTCUSDT", bar)

        self.assertEqual(kryptic_pos.ladder.fills[:2], [True, True])
        self.assertFalse(zenith_pos.ladder.fills[0])  # SHORT entry1=102 never touched (bar high=101)

        kryptic_path = Path(self.tmpdir) / "state_kryptic_btcusdt_long.json"
        zenith_path = Path(self.tmpdir) / "state_zenith_btcusdt_short.json"
        self.assertTrue(kryptic_path.exists())
        self.assertTrue(zenith_path.exists())
        self.assertEqual(kryptic_raw.state_store.load_position().direction, "LONG")
        self.assertEqual(zenith_raw.state_store.load_position().direction, "SHORT")

    async def test_full_order_lifecycle_tagging_registration_fill(self) -> None:
        """Drives a real (non-injected) trade open through the harness,
        proving clOrdId tagging, RiskArbiter approval, order-book
        registration, and closure-driven order removal all wire together.

        This fixture's Entry 1 is a prefilled market breakout, and its
        very next bar's high always exceeds TP1 (TP1's construction-time
        floor sits deliberately just above Entry 1 -- see
        TradeLifecycleManager._calculate_tp_ladder), so under the Step
        10.6 TP rebalance ([0.20, 0.30, 0.30, 0.0, 0.20]) TP1 + the
        dynamic runner exactly exhaust Entry 1's own weight in that same
        bar -- the position fully closes there, cancelling every other
        still-resting entry order along with it, rather than a second
        tier independently filling first."""
        full = make_candles(371, direction=1, seed=1)
        warm, signal_bar = full[:370], full[370]
        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.02, btc_historical=warm)
        kryptic_raw = self._make_krptic_pipeline(exchange, strategy_name="KRYPTIC")

        arbiter = RiskArbiter(account_equity=10000.0, global_risk_ceiling_pct=50.0, strategy_budgets_pct={"KRYPTIC": 30.0})
        msm = MultiStrategyManager(exchange=exchange, arbiter=arbiter, state_dir=self.tmpdir, warmup_bars=370)
        managed = msm.register_engine(kryptic_raw)
        await msm.warmup_all()

        msm.on_candle_close("BTCUSDT", signal_bar)
        pos = kryptic_raw.active_position
        self.assertIsNotNone(pos)

        open_orders = msm.order_book.open_orders(strategy_id="KRYPTIC")
        self.assertGreaterEqual(len(open_orders), 1)
        for o in open_orders:
            self.assertTrue(o.cl_ord_id.startswith("KRP_L_BTC_"))
            self.assertEqual(o.pos_side, "long")
        self.assertEqual(len(msm.telemetry.of_kind("SETUP_APPROVED")), 1)
        self.assertGreater(msm.arbiter.allocated_for("KRYPTIC"), 0)

        e1 = pos.ladder.levels[0]
        last_close = signal_bar["close"]
        fill_bar = {"ts": signal_bar["ts"] + 900_000, "open": last_close, "high": max(last_close, e1) + 1,
                    "low": min(last_close, e1) - 0.5, "close": last_close, "volume": 500}
        msm.on_candle_close("BTCUSDT", fill_bar)

        # Nothing is left resting: the fill-then-immediate-cascade above
        # closes the position entirely, and every still-pending entry gets
        # cancelled along with it (the runner-exit ghost-order fix).
        remaining = msm.order_book.open_orders(strategy_id="KRYPTIC")
        self.assertEqual(len(remaining), 0)
        self.assertTrue(pos.closed)
        self.assertGreaterEqual(len(msm.telemetry.of_kind("GHOST_ORDER_CANCEL")), 1)
        self.assertEqual(len(msm.telemetry.of_kind("POSITION_CLOSED")), 1)

    async def test_trade_open_does_not_leave_stale_pending_state_file(self) -> None:
        """Regression: `KrypticEngine.on_bar_close` (engine.py, unmodified
        here) persists its OWN 'trade_opened' snapshot the instant a trade
        opens, using whatever `state_store` this wrapper currently holds --
        which, at that exact moment, is still the "pending" placeholder
        file from `__init__`, since `_handle_trade_opened` only rescopes to
        the real direction-scoped file AFTER that engine call already
        returned. Before `_clear_stale_pending_state` (added alongside
        Step 10's live_runner.py, whose own state-corruption test caught
        this), that placeholder file was left holding a stale, non-None
        copy of the position forever -- a false "there's an active
        position here" on-disk read for anything inspecting it between
        trades."""
        full = make_candles(371, direction=1, seed=1)
        warm, signal_bar = full[:370], full[370]
        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.02, btc_historical=warm)
        kryptic_raw = self._make_krptic_pipeline(exchange, strategy_name="KRYPTIC")

        arbiter = RiskArbiter(account_equity=10000.0, global_risk_ceiling_pct=50.0, strategy_budgets_pct={"KRYPTIC": 30.0})
        msm = MultiStrategyManager(exchange=exchange, arbiter=arbiter, state_dir=self.tmpdir, warmup_bars=370)
        msm.register_engine(kryptic_raw)
        await msm.warmup_all()

        msm.on_candle_close("BTCUSDT", signal_bar)
        self.assertIsNotNone(kryptic_raw.active_position)

        pending_path = Path(self.tmpdir) / "state_kryptic_btcusdt_pending.json"
        self.assertTrue(pending_path.exists(), "the engine's own premature persist should have created this placeholder file")
        self.assertIsNone(
            json.loads(pending_path.read_text())["position"],
            "the placeholder file must be cleared back to flat once the real direction-scoped file takes over",
        )

        long_path = Path(self.tmpdir) / "state_kryptic_btcusdt_long.json"
        self.assertIsNotNone(json.loads(long_path.read_text())["position"], "the actual position must live at the correctly-scoped file")

    async def test_namespace_cancellation_preserves_other_strategy_orders(self) -> None:
        """The spec's exact scenario: dropping KRYPTIC's unfilled LONG
        tiers (via TP2's cancel_unfilled_orders) must leave ZENITH's
        resting SHORT limit orders on the same symbol completely untouched."""
        warm = [{"ts": 1_700_000_000_000 + i * 900_000, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0} for i in range(310)]
        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.0)
        kryptic_raw = self._make_krptic_pipeline(exchange, strategy_name="KRYPTIC", warmup_bars=300)
        zenith_raw = self._make_krptic_pipeline(exchange, strategy_name="ZENITH", warmup_bars=300)

        arbiter = RiskArbiter(account_equity=10000.0, global_risk_ceiling_pct=50.0,
                               strategy_budgets_pct={"KRYPTIC": 30.0, "ZENITH": 35.0})
        msm = MultiStrategyManager(exchange=exchange, arbiter=arbiter, state_dir=self.tmpdir, warmup_bars=300)
        managed_kryptic = msm.register_engine(kryptic_raw)
        managed_zenith = msm.register_engine(zenith_raw)
        await msm.warmup_all()

        entry_ts = warm[-1]["ts"] + 900_000
        kryptic_ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        kryptic_pos = PositionState(direction="LONG", ladder=kryptic_ladder, initial_sl=80.0, current_sl=80.0,
                                     tp_levels=[105.0, 110.0, 115.0, 120.0, None])
        kryptic_raw.active_position = kryptic_pos
        managed_kryptic._rescope_for_direction("LONG")
        managed_kryptic._entry_bar_ts = entry_ts
        for i, (level, weight) in enumerate(zip(kryptic_ladder.levels, kryptic_ladder.weights)):
            msm.order_book.place(OrderPayload(f"KRP_L_BTC_E{i + 1}_{entry_ts}", "KRYPTIC", "BTCUSDT", "long", "buy", f"E{i + 1}", level, weight))

        zenith_ladder = EntryLadder(direction="SHORT", levels=[102.0, 108.0, 112.0, 116.0])
        zenith_pos = PositionState(direction="SHORT", ladder=zenith_ladder, initial_sl=120.0, current_sl=120.0,
                                    tp_levels=[95.0, 90.0, 85.0, 80.0, None])
        zenith_raw.active_position = zenith_pos
        managed_zenith._rescope_for_direction("SHORT")
        for i, (level, weight) in enumerate(zip(zenith_ladder.levels, zenith_ladder.weights)):
            msm.order_book.place(OrderPayload(f"ZEN_S_BTC_E{i + 1}_{entry_ts}", "ZENITH", "BTCUSDT", "short", "sell", f"E{i + 1}", level, weight))

        self.assertEqual(len(msm.order_book.open_orders()), 8)

        fill_and_tp_bar = {"ts": entry_ts + 900_000, "open": 100.0, "high": 111.0, "low": 94.0, "close": 110.5, "volume": 1000}
        managed_kryptic.on_bar_close(fill_and_tp_bar)

        remaining = {o.cl_ord_id for o in msm.order_book.open_orders()}
        self.assertEqual({cid for cid in remaining if cid.startswith("KRP_")}, set())
        self.assertEqual(len({cid for cid in remaining if cid.startswith("ZEN_")}), 4)

        ghost_events = msm.telemetry.of_kind("GHOST_ORDER_CANCEL")
        self.assertEqual(len(ghost_events), 1)
        self.assertTrue(all(cid.startswith("KRP_") for cid in ghost_events[0].detail["order_book_cancelled"]))

    async def test_margin_exhaustion_clean_rejection(self) -> None:
        """Global portfolio risk ceiling reached -> the trade the underlying
        engine would have opened is cleanly discarded: no position, no
        registered orders, no allocated margin, an explicit rejection log."""
        full = make_candles(371, direction=1, seed=1)
        warm, signal_bar = full[:370], full[370]
        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.02, btc_historical=warm)
        kryptic_raw = self._make_krptic_pipeline(exchange, strategy_name="KRYPTIC")

        arbiter = RiskArbiter(account_equity=10000.0, global_risk_ceiling_pct=0.5, strategy_budgets_pct={"KRYPTIC": 30.0})
        msm = MultiStrategyManager(exchange=exchange, arbiter=arbiter, state_dir=self.tmpdir, warmup_bars=370)
        managed = msm.register_engine(kryptic_raw, risk_per_trade_pct=1.0)  # 1% of equity > the 0.5% global ceiling
        await msm.warmup_all()

        managed.on_bar_close(signal_bar)

        self.assertIsNone(kryptic_raw.active_position)
        self.assertEqual(msm.order_book.open_orders(), [])
        self.assertEqual(msm.arbiter.allocated_for("KRYPTIC"), 0.0)
        rejections = msm.telemetry.of_kind("SETUP_REJECTED")
        self.assertEqual(len(rejections), 1)
        self.assertIn("margin", rejections[0].detail["reason"])

    async def test_position_close_releases_margin_and_cancels_leftover_orders(self) -> None:
        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        ladder.fills[0] = True
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=92.0, current_sl=92.0,
                             tp_levels=[105.0, 110.0, 115.0, 120.0, None])
        warm = [{"ts": 1_700_000_000_000 + i * 900_000, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0} for i in range(310)]
        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.0)
        kryptic_raw = self._make_krptic_pipeline(exchange, strategy_name="KRYPTIC", warmup_bars=300)
        arbiter = RiskArbiter(account_equity=10000.0, global_risk_ceiling_pct=50.0, strategy_budgets_pct={"KRYPTIC": 30.0})
        msm = MultiStrategyManager(exchange=exchange, arbiter=arbiter, state_dir=self.tmpdir, warmup_bars=300)
        managed = msm.register_engine(kryptic_raw)
        await msm.warmup_all()

        kryptic_raw.active_position = pos
        managed._rescope_for_direction("LONG")
        managed._allocated_margin = 100.0
        arbiter.request_margin("KRYPTIC", 100.0)
        self.assertEqual(arbiter.allocated_for("KRYPTIC"), 100.0)

        sl_bar = {"ts": warm[-1]["ts"] + 900_000, "open": 100.0, "high": 100.5, "low": 85.0, "close": 86.0, "volume": 500}
        managed.on_bar_close(sl_bar)

        self.assertIsNone(kryptic_raw.active_position)
        self.assertEqual(arbiter.allocated_for("KRYPTIC"), 0.0, "margin must be released back to the arbiter on close")
        self.assertEqual(len(msm.telemetry.of_kind("POSITION_CLOSED")), 1)


if __name__ == "__main__":
    unittest.main()
