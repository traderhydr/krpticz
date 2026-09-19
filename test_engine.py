"""test_engine.py -- integration tests for KrypticEngine (KRYPTIC Step 8).

Uses the stdlib's unittest (this repo has no test runner dependency yet;
IsolatedAsyncioTestCase needs nothing beyond Python 3.8+) and
data_collector.MockExchangeStream to exercise KrypticEngine end-to-end
without any real network access.

Run with:  python3 -m unittest test_engine -v
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_collector import MockExchangeStream
from engine import KrypticEngine
from entry_ladder import EntryLadder, EntryLadderEngine
from risk_manager import PositionState, TradeLifecycleManager


class AlwaysAllowRegimeFilter:
    """Stub isolating KrypticEngine/EntryLadderEngine's own logic from
    RegimeFilter's independently-tested gates (Step 2) -- these tests are
    about the orchestrator, not re-proving the regime gate works."""

    def evaluate_market_conditions(self, df, btc_df, funding_rate, *, direction):
        return True, {"allowed": True, "direction": direction, "failed_gates": [], "gates": {}, "stub": True}


def make_candles(n, direction, seed=1, wiggle_amp=8.0, slope=0.5, start_ms=1_700_000_000_000):
    """Trend + oscillation synthetic OHLCV, matching the recipe used
    throughout this KRYPTIC track's own test suites (a pure monotonic ramp
    has no real swing structure -- see indicators.py's test history)."""
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


class KrypticEngineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        entry_engine = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter())
        self.trade_manager = TradeLifecycleManager(entry_ladder_engine=entry_engine)

    def _make_engine(self, exchange, symbol="TESTUSDT", strategy_name="KRYPTIC", warmup_bars=370) -> KrypticEngine:
        return KrypticEngine(
            exchange=exchange, symbol=symbol, trade_manager=self.trade_manager,
            strategy_name=strategy_name, state_dir=self.tmpdir, warmup_bars=warmup_bars,
        )

    async def test_cold_boot_scoped_filename_and_tagging(self) -> None:
        warm = make_candles(370, direction=1, seed=1)
        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.02, btc_historical=warm)
        engine = self._make_engine(exchange)
        await engine.warmup()

        self.assertIsNone(engine.active_position)
        self.assertEqual(engine.state_path.name, "state_kryptic_testusdt.json")
        self.assertEqual(engine.strategy_id, "KRYPTIC")

    async def test_full_lifecycle_cold_boot_signal_fill_partial_tp_crash_reboot_recovery_resume(self) -> None:
        """The spec's headline scenario: Cold Boot -> Warmup -> Signal ->
        Fills -> Partial TP -> Crash -> Reboot & Recovery -> Resumed
        execution, verified via exact reconstruction of every piece of
        position state across the simulated crash."""
        full = make_candles(371, direction=1, seed=1)  # this exact seed/length is a proven 4-way bullish confluence
        warm, signal_bar = full[:370], full[370]

        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.02, btc_historical=warm)
        engine = self._make_engine(exchange)
        await engine.warmup()

        # -- Signal --
        result = engine.on_bar_close(signal_bar)
        self.assertTrue(result.trade_opened)
        pos = engine.active_position
        self.assertIsNotNone(pos)
        self.assertTrue(engine.state_path.exists(), "state must persist immediately on opening a trade")
        raw = engine.state_store.load_raw()
        self.assertEqual(raw["extra"]["strategy_id"], "KRYPTIC")
        self.assertEqual(raw["extra"]["reason"], "trade_opened")

        # -- Fill + Partial TP (one bar producing real events, so it's persisted) --
        # High is capped at the midpoint between TP1 and the next TP above it
        # (not just "past e1 by a fixed amount") -- Step 10.8's TP1 distance
        # floor can clamp TP1 upward on this same bar's fill, and the
        # midpoint comfortably clears that clamp while staying well short of
        # TP2/TP3, so this bar hits ONLY TP1 (a genuine partial exit) rather
        # than also reaching a deeper tier and fully closing the position in
        # one bar -- which would defeat the point of this crash-mid-partial-
        # fill scenario. The bar's close is ALSO deliberately pulled back to
        # the midpoint between entry_1 and last_close (rather than reusing
        # last_close outright, or snapping to entry_1) -- either extreme is
        # far enough from this fixture's own EMA50/SuperTrend line to also
        # trip the TP5 dynamic runner's exhaustion check on the very same
        # bar (ema_stretch or supertrend_flip respectively), which would
        # close the remaining 0.25 open size too and, same as above, defeat
        # the "still open when it crashes" scenario this test exists for.
        last_close = signal_bar["close"]
        e1 = pos.ladder.levels[0]
        next_tp_above = min(lvl for lvl in pos.tp_levels[1:4] if lvl is not None)
        tp1_midpoint = pos.tp_levels[0] + (next_tp_above - pos.tp_levels[0]) * 0.5
        high = min(max(last_close, e1, tp1_midpoint), next_tp_above - 0.01)
        bar_close = (last_close + e1) / 2.0
        fill_and_tp_bar = {
            "ts": signal_bar["ts"] + 900_000, "open": last_close,
            "high": high, "low": min(last_close, e1) - 0.5,
            "close": bar_close, "volume": 500,
        }
        r1 = engine.on_bar_close(fill_and_tp_bar)
        self.assertTrue(r1.position_events, "expected this bar to fill an entry and/or hit a TP")
        self.assertFalse(pos.closed, "this bar must leave the position partially open, not fully closed")

        pre_crash = self._snapshot(pos, engine)

        # -- Crash: drop the engine entirely --
        del engine

        # -- Reboot & Recovery --
        engine2 = self._make_engine(exchange)
        await engine2.warmup()
        self.assertIsNotNone(engine2.active_position, "crash recovery failed to restore a position")
        post_recovery = self._snapshot(engine2.active_position, engine2)
        self.assertEqual(post_recovery, pre_crash)

        # -- Resumed execution: the recovered engine keeps processing bars normally --
        next_bar = {
            "ts": fill_and_tp_bar["ts"] + 900_000, "open": last_close,
            "high": last_close + 1, "low": last_close - 1, "close": last_close, "volume": 300,
        }
        engine2.on_bar_close(next_bar)  # must not raise
        self.assertTrue(
            engine2.state_store.load_position() is not None or engine2.active_position is None,
            "engine must still have consistent state after resuming",
        )

    async def test_position_close_clears_scoped_state_file(self) -> None:
        """Injects a controlled, nearly-closed position (bypassing
        open_trade -- this isolates the engine's own persistence-on-close
        wiring from risk_manager.py's TP/leg selection, already proven
        independently) and drives it to a stop-loss close, confirming the
        scoped state file is cleared and a post-mortem is recorded."""
        warm = make_candles(310, direction=1, seed=2)
        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.0, btc_historical=warm)
        engine = self._make_engine(exchange, warmup_bars=300)
        await engine.warmup()

        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        ladder.fills[0] = True
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=92.0, current_sl=92.0,
                             tp_levels=[105.0, 110.0, 115.0, 120.0, None])
        engine.active_position = pos
        engine._persist(reason="test_setup")
        self.assertTrue(engine.state_path.exists())

        last_ts = warm[-1]["ts"]
        sl_bar = {"ts": last_ts + 900_000, "open": 100.0, "high": 100.5, "low": 85.0, "close": 86.0, "volume": 500}
        result = engine.on_bar_close(sl_bar)

        self.assertTrue(result.position_closed_this_bar)
        self.assertIsNone(engine.active_position)
        self.assertEqual(len(engine.closed_position_reports), 1)
        self.assertEqual(engine.closed_position_reports[0]["strategy_id"], "KRYPTIC")
        raw = engine.state_store.load_raw()
        self.assertIsNone(raw["position"], "scoped state file must show a flat position after close")

    async def test_reconnect_replay_catches_missed_tp_and_persists(self) -> None:
        """Heartbeat-triggered reconnect: GapFillReplayer must replay
        missed candles through the active position, backfill the candle
        buffer, and persist the resulting state -- verified with a
        controlled position (see note in the prior test) and a 3-bar gap
        where the middle candle pierces TP1."""
        hist = [
            {"ts": 1_700_000_000_000 + i * 900_000, "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000.0}
            for i in range(310)
        ]
        exchange = MockExchangeStream(historical=hist, live=[], funding_rate=0.0)
        engine = self._make_engine(exchange, warmup_bars=300)
        await engine.warmup()

        ladder = EntryLadder(direction="LONG", levels=[100.0, 95.0, 90.0, 85.0])
        # Both Entry 1 and Entry 2 pre-filled (0.75 total) -- TP1's 0.60
        # weight then only partially closes the position (leaving 0.15
        # open), rather than exceeding the filled size and closing it
        # outright, matching this test's own "still open after TP1" intent.
        ladder.fills[0] = True
        ladder.fills[1] = True
        pos = PositionState(direction="LONG", ladder=ladder, initial_sl=92.0, current_sl=92.0,
                             tp_levels=[105.0, 110.0, None, None, None])
        engine.active_position = pos
        engine._persist(reason="test_setup")
        buffer_len_before = len(engine.ltf_buffer)

        missed = [
            {"ts": hist[-1]["ts"] + 900_000, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.5, "volume": 400},
            {"ts": hist[-1]["ts"] + 1_800_000, "open": 101.5, "high": 106.0, "low": 101.0, "close": 105.5, "volume": 600},  # pierces TP1
            {"ts": hist[-1]["ts"] + 2_700_000, "open": 105.5, "high": 107.0, "low": 104.0, "close": 106.0, "volume": 500},
        ]
        exchange._historical = hist + missed

        await engine._reconnect_and_gapfill()

        self.assertEqual(len(engine.ltf_buffer) - buffer_len_before, 3)
        self.assertEqual(engine._last_candle_ts, missed[-1]["ts"])
        self.assertTrue(pos.tp_fills[0])
        self.assertFalse(pos.closed)
        raw = engine.state_store.load_raw()
        self.assertTrue(raw["position"]["tp_fills"][0])
        self.assertEqual(raw["extra"]["reason"], "gap_replay")

    async def test_strategy_isolation_does_not_touch_other_strategy_state(self) -> None:
        """A pre-existing state file for a DIFFERENT strategy (same symbol,
        same directory) must survive completely untouched, byte-for-byte,
        across KrypticEngine's own boot/warmup/trade-open cycle."""
        warm = make_candles(370, direction=1, seed=1)
        signal_bar = make_candles(371, direction=1, seed=1)[370]

        zenith_path = Path(self.tmpdir) / "state_zenith_testusdt.json"
        zenith_content = '{"schema_version": 1, "symbol": "TESTUSDT", "position": {"fake": "ZENITH data"}}'
        zenith_path.write_text(zenith_content)

        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.02, btc_historical=warm)
        engine = self._make_engine(exchange, strategy_name="KRYPTIC")
        await engine.warmup()
        engine.on_bar_close(signal_bar)

        self.assertNotEqual(engine.state_path, zenith_path)
        self.assertEqual(zenith_path.read_text(), zenith_content)

    async def test_strategy_isolation_different_symbols_distinct_files(self) -> None:
        warm = make_candles(370, direction=1, seed=1)
        exchange_a = MockExchangeStream(historical=warm, live=[], funding_rate=0.0, btc_historical=warm)
        exchange_b = MockExchangeStream(historical=warm, live=[], funding_rate=0.0, btc_historical=warm)

        engine_a = self._make_engine(exchange_a, symbol="TESTUSDT")
        engine_b = self._make_engine(exchange_b, symbol="ETHUSDT")
        await engine_a.warmup()
        await engine_b.warmup()

        self.assertEqual(engine_a.state_path.name, "state_kryptic_testusdt.json")
        self.assertEqual(engine_b.state_path.name, "state_kryptic_ethusdt.json")
        self.assertNotEqual(engine_a.state_path, engine_b.state_path)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _snapshot(position: PositionState, engine: KrypticEngine) -> dict:
        return {
            "fills": list(position.ladder.fills),
            "cancelled": list(position.ladder.cancelled),
            "tp_fills": list(position.tp_fills),
            "current_sl": position.current_sl,
            "vwap": position.weighted_avg_entry,
            "realized_pnl": position.realized_pnl,
            "realized_weight": position.realized_weight,
            "execution_log": list(position.execution_log),
            "last_candle_ts": engine._last_candle_ts,
        }


if __name__ == "__main__":
    unittest.main()
