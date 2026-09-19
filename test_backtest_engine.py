"""test_backtest_engine.py -- tests for backtest_engine.py (KRYPTIC Step 10.5).

No real network anywhere in this file (see backtest_engine.py's own module
docstring for why: this sandbox has no outbound route to any exchange).
`HistoricalDataLoader` is tested against an injected fake page-fetcher;
`simulate_symbol`'s speed/correctness claim is tested by cross-checking it,
bar-by-bar, against `data_collector.MarketDataPipeline` -- the actual,
unmodified, slower production dispatcher -- on the same synthetic data,
proving the vectorized fast path produces IDENTICAL trades.

Run with:  python3 -m unittest test_backtest_engine -v
"""
from __future__ import annotations

import asyncio
import csv
import shutil
import tempfile
import time
import unittest

import numpy as np
import openpyxl
import pandas as pd

from backtest_engine import BacktestEngine, BacktestReport, HistoricalDataLoader, _load_symbol_data, discover_universe, simulate_symbol
from data_collector import MarketDataPipeline, MockExchangeStream
from entry_ladder import EntryLadderEngine
from live_runner import DryRunHarness
from risk_manager import TradeLifecycleManager


def make_candles(n, direction, seed=1, wiggle_amp=10.0, slope=0.08, period=40.0, start_ms=1_700_000_000_000, step_ms=900_000):
    """A trend + oscillation synthetic OHLCV series, tuned (wider swings
    relative to slope, a longer oscillation period) to actually clear the
    REAL, calibrated RegimeFilter/DirectionEngine's thresholds often enough
    to produce trades within a modest bar count -- unlike the make_candles
    recipe test_engine.py/test_multi_strategy.py/test_live_runner.py use,
    which those files pair with an `AlwaysAllowRegimeFilter` stub
    specifically because it does NOT reliably clear the real gates. This
    module's tests exercise the real, unmodified RegimeFilter throughout
    (that's the whole point of the cross-check below), so it needs data
    the real gates actually pass. seed=1/direction=1/n=900 is a verified
    3-trade fixture under production defaults (see the cross-check test)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = 100 + direction * t * slope
    wiggle = wiggle_amp * np.sin(t / period) + rng.normal(0, 0.5, n)
    close = base + wiggle
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    opens = np.concatenate([[close[0]], close[:-1]])
    vol = rng.uniform(1000, 5000, n)
    start_ms = (start_ms // step_ms) * step_ms
    return [
        {"ts": start_ms + i * step_ms, "open": float(opens[i]), "high": float(high[i]),
         "low": float(low[i]), "close": float(close[i]), "volume": float(vol[i])}
        for i in range(n)
    ]


def make_monotonic_trend(n, direction, seed=7, slope=1.0, start_ms=1_700_000_000_000, step_ms=900_000):
    """A cleanly, consistently trending series with no oscillation -- used
    only as the BTC macro-beta benchmark in the cross-check test, so
    close-vs-EMA agrees on a direction for the whole window regardless of
    exactly which row a caller happens to read (sidesteps a real but
    separate design question -- how "as-of" BTC alignment should work for
    a benchmark series that isn't being live-refreshed on a timer, see
    backtest_engine.py's module docstring -- that this test isn't about)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = 100 + direction * t * slope + rng.normal(0, 0.05, n)
    high = close + 0.1
    low = close - 0.1
    opens = np.concatenate([[close[0]], close[:-1]])
    vol = np.full(n, 2000.0)
    start_ms = (start_ms // step_ms) * step_ms
    return [
        {"ts": start_ms + i * step_ms, "open": float(opens[i]), "high": float(high[i]),
         "low": float(low[i]), "close": float(close[i]), "volume": float(vol[i])}
        for i in range(n)
    ]


def _naive_reference_trades(ltf_rows, btc_rows, trade_manager, *, funding_rate=None):
    """The slow, unmodified production dispatcher (`MarketDataPipeline`),
    driven one bar at a time from bar 0 -- no bulk warmup, no MockExchangeStream
    history split -- so it evaluates the exact same set of bars as
    `simulate_symbol` does (every bar, from the start), the fair baseline
    this whole cross-check needs."""
    exchange = MockExchangeStream(historical=[], live=[], funding_rate=funding_rate, btc_historical=btc_rows)
    pipeline = MarketDataPipeline(exchange=exchange, symbol="TESTUSDT", trade_manager=trade_manager, warmup_bars=len(btc_rows))
    import asyncio
    asyncio.run(pipeline.warmup())  # loads btc_buffer fully; ltf_buffer stays empty (historical=[])

    opens, closes = [], []
    for bar in ltf_rows:
        had_position = pipeline.position is not None
        result = pipeline.on_candle_close(bar)
        if result.trade_opened:
            pos = pipeline.position
            opens.append((bar["ts"], pos.direction, tuple(round(x, 6) for x in pos.ladder.levels)))
        if had_position and result.position_closed_this_bar:
            closed = pipeline.closed_positions[-1]
            closes.append((bar["ts"], round(closed.realized_pnl, 6)))
    return opens, closes


class HistoricalDataLoaderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    @staticmethod
    def _fake_pages(rows: list[dict]):
        """A fetch_page(client, symbol, interval, start_time_ms, limit)
        fake that paginates a fixed in-memory row list -- no network."""
        async def fetch_page(client, symbol, interval, start_time_ms, limit):
            candidates = [r for r in rows if r["ts"] >= start_time_ms]
            return candidates[:limit]
        return fetch_page

    async def test_load_fetches_and_caches(self) -> None:
        rows = make_candles(50, direction=1, seed=1)
        loader = HistoricalDataLoader(cache_dir=self.tmpdir, fetch_page=self._fake_pages(rows), page_limit=1500)
        df = await loader.load("BTCUSDT", "15m", rows[0]["ts"], rows[-1]["ts"])
        self.assertEqual(len(df), 50)
        self.assertTrue((loader._cache_path("BTCUSDT", "15m")).exists())

    async def test_load_reuses_cache_without_refetching(self) -> None:
        rows = make_candles(50, direction=1, seed=1)
        calls = {"n": 0}

        async def counting_fetch_page(client, symbol, interval, start_time_ms, limit):
            calls["n"] += 1
            candidates = [r for r in rows if r["ts"] >= start_time_ms]
            return candidates[:limit]

        loader = HistoricalDataLoader(cache_dir=self.tmpdir, fetch_page=counting_fetch_page)
        await loader.load("BTCUSDT", "15m", rows[0]["ts"], rows[-1]["ts"])
        first_call_count = calls["n"]
        self.assertGreater(first_call_count, 0)

        df2 = await loader.load("BTCUSDT", "15m", rows[0]["ts"], rows[-1]["ts"])
        self.assertEqual(calls["n"], first_call_count, "a fully-cached range must not trigger any new fetch_page call")
        self.assertEqual(len(df2), 50)

    async def test_load_extends_cache_for_newer_range(self) -> None:
        rows = make_candles(80, direction=1, seed=1)
        loader = HistoricalDataLoader(cache_dir=self.tmpdir, fetch_page=self._fake_pages(rows))
        await loader.load("BTCUSDT", "15m", rows[0]["ts"], rows[39]["ts"])
        df_full = await loader.load("BTCUSDT", "15m", rows[0]["ts"], rows[-1]["ts"])
        self.assertEqual(len(df_full), 80)
        self.assertEqual(int(df_full["ts"].iloc[-1]), rows[-1]["ts"])

    async def test_load_funding_disabled_returns_empty_series(self) -> None:
        loader = HistoricalDataLoader(cache_dir=self.tmpdir, fetch_funding_page=None)
        series = await loader.load_funding("BTCUSDT", 0, 10_000_000_000)
        self.assertEqual(len(series), 0)

    async def test_load_funding_caches_and_aligns(self) -> None:
        from backtest_engine import funding_asof
        funding_rows = [{"ts": 1_700_000_000_000 + i * 8 * 3600 * 1000, "rate": 0.01 * i} for i in range(5)]

        async def fake_funding_page(client, symbol, start_time_ms, limit):
            candidates = [r for r in funding_rows if r["ts"] >= start_time_ms]
            return candidates[:limit]

        loader = HistoricalDataLoader(cache_dir=self.tmpdir, fetch_funding_page=fake_funding_page)
        series = await loader.load_funding("BTCUSDT", funding_rows[0]["ts"], funding_rows[-1]["ts"])
        self.assertEqual(len(series), 5)
        # as-of a timestamp strictly BETWEEN two prints must return the EARLIER one, never the later.
        mid_ts = funding_rows[2]["ts"] + 1000
        self.assertAlmostEqual(funding_asof(series, mid_ts), funding_rows[2]["rate"])
        self.assertIsNone(funding_asof(series, funding_rows[0]["ts"] - 1))

    async def test_default_fetch_funding_page_binance_clamps_limit_to_1000(self) -> None:
        """Regression: Binance's fundingRate endpoint's `limit` caps at
        1000 (unlike klines' 1500) -- passing the loader's default 1500
        straight through made the real endpoint reject the request with a
        {"code":...,"msg":...} error object instead of a list, which then
        crashed on `row["fundingTime"]` (iterating a dict yields its
        string keys) with a cryptic `TypeError: string indices must be
        integers`. Confirmed against the real Binance API (see the earlier
        conversation this fix responds to) -- this test verifies the
        client-side clamp without hitting the network."""
        from backtest_engine import default_fetch_funding_page_binance

        captured = {}

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return []

        class _FakeClient:
            async def get(self, url, params, headers, timeout):
                captured["limit"] = params["limit"]
                return _FakeResp()

        await default_fetch_funding_page_binance(_FakeClient(), "BTCUSDT", 0, 1500)
        self.assertLessEqual(captured["limit"], 1000)

    async def test_default_fetch_funding_page_binance_raises_clear_error_on_rejected_request(self) -> None:
        """A non-list response body (Binance's error envelope) must raise
        a clear RuntimeError naming the rejection, not crash inside the
        row-parsing list comprehension."""
        from backtest_engine import default_fetch_funding_page_binance

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"code": -1130, "msg": "Data sent for parameter 'limit' is not valid."}

        class _FakeClient:
            async def get(self, url, params, headers, timeout):
                return _FakeResp()

        with self.assertRaises(RuntimeError) as ctx:
            await default_fetch_funding_page_binance(_FakeClient(), "BTCUSDT", 0, 1500)
        self.assertIn("rejected", str(ctx.exception))

    async def test_load_funding_degrades_gracefully_on_fetch_failure(self) -> None:
        """A funding fetch failure must not take down the whole backtest --
        funding is a supplementary signal (RegimeFilter treats it as
        pass-through when missing) everywhere else in this codebase."""
        async def failing_funding_page(client, symbol, start_time_ms, limit):
            raise RuntimeError("Binance fundingRate request rejected: simulated")

        loader = HistoricalDataLoader(cache_dir=self.tmpdir, fetch_funding_page=failing_funding_page)
        series = await loader.load_funding("BTCUSDT", 1_700_000_000_000, 1_700_100_000_000)  # must not raise
        self.assertEqual(len(series), 0)


class VectorizedCrossCheckTests(unittest.TestCase):
    """The correctness backbone of this whole module: proves the fast,
    vectorized-prefilter path (`simulate_symbol`) makes IDENTICAL trade
    decisions to the slow, unmodified, bar-by-bar production dispatcher on
    the same data -- i.e. that skipping most `open_trade()` calls never
    changes which bars actually open a trade."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_matches_production_dispatcher_on_bullish_fixture(self) -> None:
        n = 900
        ltf_rows = make_candles(n, direction=1, seed=1)
        btc_rows = make_monotonic_trend(n, direction=1, seed=99)

        # Two independently-constructed TradeLifecycleManagers, both with
        # pure production defaults (no test stubs) -- the real RegimeFilter,
        # the real DirectionEngine, the real calibrated EntryLadderEngine.
        tm_reference = TradeLifecycleManager()
        tm_fast = TradeLifecycleManager()

        ref_opens, ref_closes = _naive_reference_trades(ltf_rows, btc_rows, tm_reference, funding_rate=0.0)

        ltf_df = pd.DataFrame(ltf_rows)
        btc_df = pd.DataFrame(btc_rows)
        harness = DryRunHarness(state_dir=self.tmpdir)
        fast_trades, _signals_generated = simulate_symbol("TESTUSDT", ltf_df, btc_df, trade_manager=tm_fast, harness=harness, funding_series=None)

        self.assertEqual(len(fast_trades), len(ref_opens), "trade COUNT must match the unmodified production dispatcher exactly")
        self.assertGreater(len(fast_trades), 0, "the proven bullish fixture must actually produce at least one trade in both paths")

        for fast, (ref_ts, ref_direction, ref_levels) in zip(fast_trades, ref_opens):
            self.assertEqual(fast["opened_at_ts"], ref_ts)
            self.assertEqual(fast["direction"], ref_direction)

        for fast, (ref_ts, ref_pnl) in zip(fast_trades, ref_closes):
            self.assertEqual(fast["closed_at_ts"], ref_ts)
            self.assertAlmostEqual(fast["idealized_pnl"], ref_pnl, places=6)

    def test_fast_path_is_meaningfully_faster_than_naive_replay(self) -> None:
        """Not a strict correctness test, but the whole point of this
        module's design (see its docstring) -- confirms the vectorized
        prefilter actually avoids most `open_trade()` calls, not just that
        it happens to produce the same answer some other slow way."""
        n = 900
        ltf_rows = make_candles(n, direction=1, seed=2)
        btc_rows = make_monotonic_trend(n, direction=1, seed=99)
        ltf_df = pd.DataFrame(ltf_rows)
        btc_df = pd.DataFrame(btc_rows)

        harness = DryRunHarness(state_dir=self.tmpdir)
        t0 = time.time()
        simulate_symbol("TESTUSDT", ltf_df, btc_df, trade_manager=TradeLifecycleManager(), harness=harness)
        fast_elapsed = time.time() - t0

        t0 = time.time()
        _naive_reference_trades(ltf_rows, btc_rows, TradeLifecycleManager(), funding_rate=0.0)
        naive_elapsed = time.time() - t0

        self.assertLess(fast_elapsed, naive_elapsed, "the vectorized path should be faster, not just equally correct")


class BacktestReportTests(unittest.TestCase):
    """Hand-constructed trade dicts (matching live_runner.PaperTradeRecord's
    shape exactly) so every metric formula can be verified precisely,
    independent of the simulation loop."""

    @staticmethod
    def _trade(*, r, net_pnl, duration_seconds, closed_at_ts, opened_at_ts=0, tier_indices=(0,)):
        return {
            "strategy_id": "KRYPTIC", "symbol": "BTCUSDT", "direction": "LONG",
            "opened_at_ts": opened_at_ts, "closed_at_ts": closed_at_ts, "duration_seconds": duration_seconds,
            "filled_weight": 1.0, "avg_entry": 100.0, "idealized_pnl": net_pnl, "net_pnl": net_pnl,
            "total_fees": 0.1, "total_slippage_cost": 0.05, "r_multiple": r, "pct_return": net_pnl,
            "fills": [{"kind": "ENTRY", "event_type": "ENTRY_FILL", "tier_index": ti, "raw_price": 100.0,
                       "fill_price": 100.0, "weight": 0.5, "is_market": False, "fee_amount": 0.01,
                       "slippage_cost": 0.0, "bar_index": 0} for ti in tier_indices],
        }

    def test_win_loss_breakeven_classification_and_win_rate(self) -> None:
        trades = [
            self._trade(r=1.5, net_pnl=15.0, duration_seconds=3600, closed_at_ts=1000),
            self._trade(r=-1.0, net_pnl=-10.0, duration_seconds=3600, closed_at_ts=2000),
            self._trade(r=0.02, net_pnl=0.2, duration_seconds=3600, closed_at_ts=3000),  # within breakeven epsilon
        ]
        report = BacktestReport.from_trades("BTCUSDT", trades)
        self.assertEqual(report.total_trades, 3)
        self.assertEqual((report.wins, report.losses, report.breakevens), (1, 1, 1))
        self.assertAlmostEqual(report.win_rate_pct, 50.0)  # 1 win / (1 win + 1 loss), breakeven excluded

    def test_profit_factor_and_expectancy(self) -> None:
        trades = [
            self._trade(r=2.0, net_pnl=20.0, duration_seconds=3600, closed_at_ts=1000),
            self._trade(r=-1.0, net_pnl=-10.0, duration_seconds=3600, closed_at_ts=2000),
        ]
        report = BacktestReport.from_trades("BTCUSDT", trades)
        self.assertAlmostEqual(report.profit_factor, 2.0)  # 20 gross win / 10 gross loss
        self.assertAlmostEqual(report.expectancy_r, 0.5)   # mean(2.0, -1.0)

    def test_profit_factor_infinite_with_no_losses(self) -> None:
        trades = [self._trade(r=1.0, net_pnl=10.0, duration_seconds=3600, closed_at_ts=1000)]
        report = BacktestReport.from_trades("BTCUSDT", trades)
        self.assertEqual(report.profit_factor, float("inf"))

    def test_max_drawdown_from_equity_curve(self) -> None:
        # +1R, then -0.6R, then +0.5R at 1% risk/trade: equity 100 -> 101 -> 100.394 -> 100.897
        trades = [
            self._trade(r=1.0, net_pnl=1.0, duration_seconds=3600, closed_at_ts=1000),
            self._trade(r=-0.6, net_pnl=-0.6, duration_seconds=3600, closed_at_ts=2000),
            self._trade(r=0.5, net_pnl=0.5, duration_seconds=3600, closed_at_ts=3000),
        ]
        report = BacktestReport.from_trades("BTCUSDT", trades, starting_equity=100.0, risk_per_trade_pct=1.0)
        peak = 100.0 * 1.01
        trough = peak * (1 - 0.006)
        expected_dd = (peak - trough) / peak * 100.0
        self.assertAlmostEqual(report.max_drawdown_pct, expected_dd, places=6)

    def test_entry_tier_hit_rates(self) -> None:
        trades = [
            self._trade(r=1.0, net_pnl=1.0, duration_seconds=3600, closed_at_ts=1000, tier_indices=(0,)),
            self._trade(r=1.0, net_pnl=1.0, duration_seconds=3600, closed_at_ts=2000, tier_indices=(0, 1)),
            self._trade(r=1.0, net_pnl=1.0, duration_seconds=3600, closed_at_ts=3000, tier_indices=(0, 1, 2)),
        ]
        report = BacktestReport.from_trades("BTCUSDT", trades)
        self.assertAlmostEqual(report.entry_tier_hit_rate_pct["entry_1"], 100.0)
        self.assertAlmostEqual(report.entry_tier_hit_rate_pct["entry_2"], 200.0 / 3.0)
        self.assertAlmostEqual(report.entry_tier_hit_rate_pct["entry_3"], 100.0 / 3.0)
        self.assertAlmostEqual(report.entry_tier_hit_rate_pct["entry_4"], 0.0)

    def test_sample_size_flag(self) -> None:
        trades = [self._trade(r=1.0, net_pnl=1.0, duration_seconds=3600, closed_at_ts=1000)]
        report = BacktestReport.from_trades("BTCUSDT", trades, sample_size_target=300)
        self.assertFalse(report.meets_sample_size)
        self.assertIn("NOT MET", report.format_summary())

    def test_empty_trades_reports_zero_without_crashing(self) -> None:
        report = BacktestReport.from_trades("BTCUSDT", [])
        self.assertEqual(report.total_trades, 0)
        self.assertEqual(report.win_rate_pct, 0.0)
        self.assertIsNone(report.expectancy_r)
        self.assertIsInstance(report.format_summary(), str)

    def test_combine_merges_multiple_symbols(self) -> None:
        r1 = BacktestReport.from_trades("BTCUSDT", [self._trade(r=1.0, net_pnl=1.0, duration_seconds=3600, closed_at_ts=1000)])
        r2 = BacktestReport.from_trades("ETHUSDT", [self._trade(r=-1.0, net_pnl=-1.0, duration_seconds=3600, closed_at_ts=2000)])
        combined = BacktestReport.combine([r1, r2])
        self.assertEqual(combined.total_trades, 2)
        self.assertIn("BTCUSDT", combined.label)
        self.assertIn("ETHUSDT", combined.label)

    def test_sharpe_sortino_none_when_insufficient_history(self) -> None:
        trades = [self._trade(r=1.0, net_pnl=1.0, duration_seconds=3600, closed_at_ts=1000)]
        report = BacktestReport.from_trades("BTCUSDT", trades)
        self.assertIsNone(report.sharpe_ratio)
        self.assertIsNone(report.sortino_ratio)


class InstitutionalReportExportTests(unittest.TestCase):
    """The institutional multi-tab Excel/CSV report: run a real
    `BacktestEngine` over the same proven bullish fixture
    `VectorizedCrossCheckTests` uses (so trades carry REAL setup
    diagnostics captured from `TradeLifecycleManager.open_trade()`, not
    hand-constructed stand-ins), then verify the funnel-stat invariant and
    both export formats' actual on-disk shape."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def _run(self) -> BacktestReport:
        n = 900
        ltf_df = pd.DataFrame(make_candles(n, direction=1, seed=1))
        btc_df = pd.DataFrame(make_monotonic_trend(n, direction=1, seed=99))
        engine = BacktestEngine(state_dir=self.tmpdir)
        report = engine.run_symbol("TESTUSDT", ltf_df, btc_df)
        self.assertGreater(report.total_trades, 0, "the proven bullish fixture must produce at least one real trade")
        return report

    def test_funnel_stats_invariant(self) -> None:
        report = self._run()
        self.assertGreaterEqual(report.signals_generated, report.setups_opened)
        self.assertEqual(report.setups_opened, report.total_trades + report.expired_unfilled)
        self.assertEqual(report.symbols, ["TESTUSDT"])
        self.assertIn("entry_ladder_weights", report.strategy_config)
        self.assertIn("ker_min", report.strategy_config)

    def test_export_csv_matches_trade_count_and_schema(self) -> None:
        from backtest_engine import _TRADE_ROW_FIELDS

        report = self._run()
        path = f"{self.tmpdir}/trades.csv"
        report.export_csv(path)

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), report.total_trades)
        self.assertEqual(list(rows[0].keys()), _TRADE_ROW_FIELDS)
        # score/leverage are ZENITH/GEM-only concepts KRYPTIC doesn't
        # compute -- must be blank, never fabricated (see _build_notes_rows).
        self.assertEqual(rows[0]["score"], "")
        self.assertEqual(rows[0]["leverage"], "")
        self.assertNotEqual(rows[0]["entry_avg_filled"], "")
        self.assertNotEqual(rows[0]["sl"], "")

    def test_export_excel_has_all_five_tabs_with_expected_row_counts(self) -> None:
        report = self._run()
        path = f"{self.tmpdir}/report.xlsx"
        report.export_excel(path)

        wb = openpyxl.load_workbook(path)
        self.assertEqual(wb.sheetnames, ["Summary", "Trades", "Equity Curve", "Monthly", "Notes"])

        trades_ws = wb["Trades"]
        self.assertEqual(trades_ws.max_row - 1, report.total_trades)  # -1 for the header row

        equity_ws = wb["Equity Curve"]
        self.assertEqual(equity_ws.max_row - 1, report.total_trades)

        summary_values = {row[0].value for row in wb["Summary"].iter_rows() if row[0].value}
        self.assertIn("Strategy profile", summary_values)
        self.assertIn("Entry Ladder Weights", summary_values)
        self.assertIn("Signals generated", summary_values)

        notes_ws = wb["Notes"]
        self.assertGreater(notes_ws.max_row, 1)


class VerificationBacktestTests(unittest.TestCase):
    """Requirement 4: a 90-day slice across 3 volatile pairs, using
    production's REAL (calibrated) RegimeFilter/DirectionEngine/
    EntryLadderEngine defaults -- not a test stub -- with the summary table
    displayed. This does NOT assert total_trades >= 300: over a single
    90-day, single-symbol-per-run window with an honestly-gated production
    regime filter, that sample size is not a realistic target (300+ trades
    would mean a fresh signal every ~7 hours, sustained, on one symbol) --
    the report displays the real count against that target and flags it
    plainly rather than the test pretending otherwise. What IS asserted is
    internal consistency of the report (win/loss/BE partition sums to the
    trade count, equity curve length, drawdown bounds) for however many
    trades the honestly-gated simulation actually produces.
    """

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def test_90_day_three_pair_backtest_summary(self) -> None:
        n = 90 * 96  # 90 days of 15m bars
        btc_rows = make_monotonic_trend(n, direction=1, seed=99, slope=0.05)
        btc_df = pd.DataFrame(btc_rows)

        symbol_seeds = {"SOLUSDT": (1, 1.2), "DOGEUSDT": (2, 1.0), "ETHUSDT": (3, 0.8)}
        engine = BacktestEngine(
            trade_manager_factory=lambda: TradeLifecycleManager(entry_ladder_engine=EntryLadderEngine()),
            state_dir=self.tmpdir,
        )
        data = {}
        for symbol, (seed, wiggle) in symbol_seeds.items():
            rows = make_candles(n, direction=1, seed=seed, wiggle_amp=10.0 * wiggle, slope=0.06)
            data[symbol] = (pd.DataFrame(rows), btc_df, None)

        t0 = time.time()
        reports = engine.run_many(data)
        elapsed = time.time() - t0
        print(f"\n[90-day 3-pair verification backtest] wall clock: {elapsed:.1f}s")

        self.assertIn("COMBINED", reports)
        for symbol in symbol_seeds:
            self.assertIn(symbol, reports)
            report = reports[symbol]
            report.print_summary()
            self.assertEqual(report.wins + report.losses + report.breakevens, report.total_trades)
            self.assertGreaterEqual(report.max_drawdown_pct, 0.0)
            self.assertLessEqual(report.max_drawdown_pct, 100.0)
            # `_compute` only seeds a base equity point once there's at
            # least one real (non-phantom) closed trade to hang start_ts
            # off of -- a symbol that produces zero trades in this
            # honestly-gated 90-day window legitimately has an empty curve.
            expected_curve_len = report.total_trades + 1 if report.total_trades > 0 else 0
            self.assertEqual(len(report.equity_curve), expected_curve_len)

        combined = reports["COMBINED"]
        combined.print_summary()
        self.assertEqual(combined.total_trades, sum(reports[s].total_trades for s in symbol_seeds))
        print(f"[90-day 3-pair verification backtest] combined total trades: {combined.total_trades} (target 300+: {'MET' if combined.meets_sample_size else 'NOT MET'})")

        # Runtime sanity: this is the entire point of "vectorized" -- keep
        # a generous bound (the naive per-bar production dispatcher took
        # ~140s for a NINETY-DAY SINGLE symbol in this same sandbox; three
        # symbols at the same length must stay well under that combined).
        self.assertLess(elapsed, 120.0, "a 90-day, 3-symbol backtest should complete in well under the naive per-bar replay's single-symbol time")


class LoadSymbolDataTests(unittest.IsolatedAsyncioTestCase):
    """`_load_symbol_data`: isolates one symbol's fetch failure so it can't
    take down a multi-hundred-symbol --all-coins run (regression -- a real
    run hit exactly this on a Binance klines 400 for a symbol
    exchangeInfo listed as tradeable)."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    async def test_a_failing_symbol_returns_none_instead_of_raising(self) -> None:
        rows = make_candles(50, direction=1, seed=1)

        async def flaky_fetch_page(client, symbol, interval, start_time_ms, limit):
            if symbol == "BADUSDT":
                raise RuntimeError("400 Bad Request")
            candidates = [r for r in rows if r["ts"] >= start_time_ms]
            return candidates[:limit]

        loader = HistoricalDataLoader(cache_dir=self.tmpdir, fetch_page=flaky_fetch_page)
        btc_df = pd.DataFrame(rows)
        sem = asyncio.Semaphore(2)

        symbol, ltf_df, funding = await _load_symbol_data(loader, sem, "BADUSDT", btc_df, rows[0]["ts"], rows[-1]["ts"])
        self.assertEqual(symbol, "BADUSDT")
        self.assertIsNone(ltf_df)
        self.assertIsNone(funding)

    async def test_a_healthy_symbol_alongside_a_failing_one_still_succeeds(self) -> None:
        rows = make_candles(50, direction=1, seed=1)

        async def flaky_fetch_page(client, symbol, interval, start_time_ms, limit):
            if symbol == "BADUSDT":
                raise RuntimeError("400 Bad Request")
            candidates = [r for r in rows if r["ts"] >= start_time_ms]
            return candidates[:limit]

        loader = HistoricalDataLoader(cache_dir=self.tmpdir, fetch_page=flaky_fetch_page)
        btc_df = pd.DataFrame(rows)
        sem = asyncio.Semaphore(2)

        results = await asyncio.gather(
            _load_symbol_data(loader, sem, "BADUSDT", btc_df, rows[0]["ts"], rows[-1]["ts"]),
            _load_symbol_data(loader, sem, "GOODUSDT", btc_df, rows[0]["ts"], rows[-1]["ts"]),
        )
        by_symbol = {symbol: (ltf_df, funding) for symbol, ltf_df, funding in results}
        self.assertIsNone(by_symbol["BADUSDT"][0])
        self.assertIsNotNone(by_symbol["GOODUSDT"][0])
        self.assertEqual(len(by_symbol["GOODUSDT"][0]), 50)


class DiscoverUniverseTests(unittest.IsolatedAsyncioTestCase):
    """`discover_universe`: thin glue over exchanges.py's own `load_universe`
    and live_runner.py's `filter_and_rank_universe` -- no real network (a
    fake `load_universe_fn` is injected), and no re-implementation of the
    filter/rank/exclusion logic itself (already covered by
    test_live_runner.py's own UniverseFilterConfig tests)."""

    @staticmethod
    def _fake_load_universe(tickers: list[dict]):
        async def load_universe_fn(client, min_quote_vol):
            return {"tickers": tickers}
        return load_universe_fn

    async def test_applies_volume_floor_and_top_n(self) -> None:
        tickers = [
            {"symbol": "BTCUSDT", "quote_vol": 500_000_000.0},
            {"symbol": "ETHUSDT", "quote_vol": 300_000_000.0},
            {"symbol": "DOGEUSDT", "quote_vol": 5_000_000.0},  # below the floor
        ]
        symbols = await discover_universe(
            min_quote_volume_usd=20_000_000.0, top_n=10,
            load_universe_fn=self._fake_load_universe(tickers),
        )
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT"])

    async def test_top_n_caps_the_result_even_when_all_pass_the_floor(self) -> None:
        tickers = [{"symbol": f"COIN{i}USDT", "quote_vol": float(100 - i) * 1_000_000} for i in range(10)]
        symbols = await discover_universe(min_quote_volume_usd=0.0, top_n=3, load_universe_fn=self._fake_load_universe(tickers))
        self.assertEqual(len(symbols), 3)
        self.assertEqual(symbols, ["COIN0USDT", "COIN1USDT", "COIN2USDT"])  # highest quote_vol first

    async def test_stablecoin_and_leveraged_tokens_excluded(self) -> None:
        tickers = [
            {"symbol": "BTCUSDT", "quote_vol": 500_000_000.0},
            {"symbol": "USDCUSDT", "quote_vol": 500_000_000.0},  # stablecoin pair
            {"symbol": "BTCUPUSDT", "quote_vol": 500_000_000.0},  # leveraged token
        ]
        symbols = await discover_universe(min_quote_volume_usd=0.0, top_n=10, load_universe_fn=self._fake_load_universe(tickers))
        self.assertEqual(symbols, ["BTCUSDT"])


if __name__ == "__main__":
    unittest.main()
