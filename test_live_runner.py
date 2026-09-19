"""test_live_runner.py -- tests for live_runner.py (KRYPTIC Step 10).

Uses the stdlib's unittest, `data_collector.MockExchangeStream`, and small
hand-written fakes for `httpx.AsyncClient` -- no real network access
anywhere in this file. Per live_runner.py's own module docstring, the real
`LiveExchangeStream.stream_klines` WebSocket path (the one genuinely new,
unverifiable piece) is NOT exercised here; everything else -- universe
filtering, the fee/slippage paper ledger and its crash recovery, the REST
adapter's retry/parsing logic, and a full paper-trading run through
`LiveRunner` -- is.

Run with:  python3 -m unittest test_live_runner -v
"""
from __future__ import annotations

import shutil
import tempfile
import unittest

import httpx
import numpy as np

from data_collector import MockExchangeStream
from entry_ladder import EntryLadder, EntryLadderEngine
from live_runner import (
    ConsoleDashboard, DryRunHarness, FeeSlippageModel, LiveExchangeStream, LiveRunner,
    UniverseFilterConfig, UniverseScanner, filter_and_rank_universe, is_excluded_symbol, spread_pct,
)
from multi_strategy_manager import RiskArbiter
from risk_manager import PositionState, TradeLifecycleManager


class AlwaysAllowRegimeFilter:
    """Stub isolating these tests from RegimeFilter's own independently
    tested gates (Step 2) -- see test_engine.py/test_multi_strategy.py for
    the same pattern this repeats verbatim."""

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


# ---------------------------------------------------------------------------
# 2. Universe filtering
# ---------------------------------------------------------------------------

class UniverseFilterTests(unittest.TestCase):
    def test_is_excluded_symbol_stablecoin_and_leveraged(self) -> None:
        self.assertTrue(is_excluded_symbol("USDCUSDT"))
        self.assertTrue(is_excluded_symbol("BUSDUSDT"))
        self.assertTrue(is_excluded_symbol("BTCUPUSDT"))
        self.assertTrue(is_excluded_symbol("ETHBEARUSDT"))
        self.assertTrue(is_excluded_symbol("BTC3LUSDT"))
        self.assertFalse(is_excluded_symbol("BTCUSDT"))
        self.assertFalse(is_excluded_symbol("ETHUSDT"))

    def test_spread_pct(self) -> None:
        self.assertAlmostEqual(spread_pct(99.95, 100.05), 0.1, places=6)
        self.assertIsNone(spread_pct(None, 100.0))
        self.assertIsNone(spread_pct(100.0, None))
        self.assertIsNone(spread_pct(0.0, 100.0))
        self.assertIsNone(spread_pct(100.0, 99.0))  # crossed book

    def test_filter_and_rank_universe_volume_spread_exclusion_and_ranking(self) -> None:
        tickers = [
            {"symbol": "BTCUSDT", "quote_vol": 900_000_000, "bid": 100.0, "ask": 100.02},   # passes
            {"symbol": "ETHUSDT", "quote_vol": 600_000_000, "bid": 100.0, "ask": 100.01},   # passes
            {"symbol": "LOWVOLUSDT", "quote_vol": 1_000_000, "bid": 1.0, "ask": 1.001},       # below volume floor
            {"symbol": "WIDEUSDT", "quote_vol": 700_000_000, "bid": 100.0, "ask": 101.0},    # spread too wide
            {"symbol": "USDCUSDT", "quote_vol": 800_000_000},                                 # stablecoin, excluded
            {"symbol": "BTCUPUSDT", "quote_vol": 800_000_000},                                # leveraged token, excluded
            {"symbol": "NOBBOUSDT", "quote_vol": 500_000_000},                                # no bid/ask -- passes anyway
        ]
        ranked = filter_and_rank_universe(tickers, UniverseFilterConfig(top_n=20))
        symbols = [t["symbol"] for t in ranked]
        self.assertEqual(symbols, ["BTCUSDT", "ETHUSDT", "NOBBOUSDT"])  # ranked by quote_vol descending

    def test_filter_and_rank_universe_caps_at_top_n(self) -> None:
        tickers = [{"symbol": f"SYM{i}USDT", "quote_vol": 100_000_000 + i} for i in range(30)]
        ranked = filter_and_rank_universe(tickers, UniverseFilterConfig(top_n=15))
        self.assertEqual(len(ranked), 15)
        self.assertEqual(ranked[0]["symbol"], "SYM29USDT")  # highest quote_vol first


# ---------------------------------------------------------------------------
# UniverseScanner
# ---------------------------------------------------------------------------

class UniverseScannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_scan_once_fires_qualify_and_disqualify_hooks(self) -> None:
        calls = {"tickers": 0}
        universes = [
            [{"symbol": "BTCUSDT", "quote_vol": 900_000_000}, {"symbol": "ETHUSDT", "quote_vol": 800_000_000}],
            [{"symbol": "BTCUSDT", "quote_vol": 900_000_000}, {"symbol": "SOLUSDT", "quote_vol": 700_000_000}],
        ]

        async def fetch_tickers():
            result = universes[min(calls["tickers"], len(universes) - 1)]
            calls["tickers"] += 1
            return result

        qualified, disqualified = [], []

        async def on_qualified(symbol):
            qualified.append(symbol)

        async def on_disqualified(symbol):
            disqualified.append(symbol)

        scanner = UniverseScanner(fetch_tickers=fetch_tickers, on_symbol_qualified=on_qualified, on_symbol_disqualified=on_disqualified)

        added, removed = await scanner.scan_once()
        self.assertEqual(added, {"BTCUSDT", "ETHUSDT"})
        self.assertEqual(removed, set())
        self.assertEqual(sorted(qualified), ["BTCUSDT", "ETHUSDT"])

        added, removed = await scanner.scan_once()
        self.assertEqual(added, {"SOLUSDT"})
        self.assertEqual(removed, {"ETHUSDT"})
        self.assertEqual(disqualified, ["ETHUSDT"])
        self.assertEqual(scanner.qualifying_symbols, {"BTCUSDT", "SOLUSDT"})
        self.assertEqual(scanner.scan_count, 2)

    async def test_scan_once_keeps_previous_universe_on_fetch_failure(self) -> None:
        async def fetch_tickers():
            raise RuntimeError("REST down")

        scanner = UniverseScanner(fetch_tickers=fetch_tickers)
        scanner.qualifying_symbols = {"BTCUSDT"}
        added, removed = await scanner.scan_once()
        self.assertEqual(added, set())
        self.assertEqual(removed, set())
        self.assertEqual(scanner.qualifying_symbols, {"BTCUSDT"})  # unchanged


# ---------------------------------------------------------------------------
# 3. Fee/slippage model
# ---------------------------------------------------------------------------

class FeeSlippageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fees = FeeSlippageModel(maker_fee_pct=0.02, taker_fee_pct=0.05, market_slippage_pct=0.02)

    def test_fee_amount_maker_vs_taker(self) -> None:
        self.assertAlmostEqual(self.fees.fee_amount(100.0, 2.0, is_market=False), 100.0 * 2.0 * 0.02 / 100.0)
        self.assertAlmostEqual(self.fees.fee_amount(100.0, 2.0, is_market=True), 100.0 * 2.0 * 0.05 / 100.0)

    def test_slipped_price_all_four_combinations(self) -> None:
        # LONG entry (market breakout/fallback fill): worse = pay MORE.
        self.assertAlmostEqual(self.fees.slipped_price("LONG", 100.0, is_exit=False), 100.02)
        # SHORT entry: worse = sell for LESS.
        self.assertAlmostEqual(self.fees.slipped_price("SHORT", 100.0, is_exit=False), 99.98)
        # LONG exit (SL/runner): worse = sell for LESS.
        self.assertAlmostEqual(self.fees.slipped_price("LONG", 100.0, is_exit=True), 99.98)
        # SHORT exit: worse = buy back for MORE.
        self.assertAlmostEqual(self.fees.slipped_price("SHORT", 100.0, is_exit=True), 100.02)


# ---------------------------------------------------------------------------
# DryRunHarness
# ---------------------------------------------------------------------------

def _make_single_tier_position(direction="LONG", level=100.0, initial_sl=92.0, tp1=110.0, entry_type="market_breakout", weight=1.0) -> PositionState:
    other_weight = (1.0 - weight) / 3.0
    ladder = EntryLadder(
        direction=direction, levels=[level, level - 5, level - 10, level - 15],
        weights=[weight, other_weight, other_weight, other_weight],
        entry_types=[entry_type, "limit_fvg_outer", "entry3_fib", "atr_band"],
    )
    ladder.fills[0] = True
    return PositionState(
        direction=direction, ladder=ladder, initial_sl=initial_sl, current_sl=initial_sl,
        tp_levels=[tp1, tp1 + 5, tp1 + 10, tp1 + 15, None],
    )


class DryRunHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.fees = FeeSlippageModel(maker_fee_pct=0.02, taker_fee_pct=0.05, market_slippage_pct=0.02)

    def test_market_entry_and_limit_tp_fee_and_slippage(self) -> None:
        """Single-tier ladder (weight 1.0 on the immediate market fill) so
        the whole trade's PnL/fee/slippage math can be hand-verified
        independently of DryRunHarness's own implementation."""
        pos = _make_single_tier_position(level=100.0, initial_sl=92.0, tp1=110.0, weight=1.0)
        harness = DryRunHarness(state_dir=self.tmpdir, fees=self.fees)

        open_bar = {"ts": 1_700_000_000_000}
        harness.on_trade_opened(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, bar=open_bar)

        # -- hand-computed expectations for the market entry fill --
        expected_entry_fill = 100.0 + 100.0 * 0.02 / 100.0  # LONG market entry: worse = higher
        expected_entry_fee = expected_entry_fill * 1.0 * 0.05 / 100.0  # taker
        expected_entry_slippage = abs(expected_entry_fill - 100.0) * 1.0

        key = ("KRYPTIC", "BTCUSDT", "LONG")
        trade = harness._open[key]
        self.assertAlmostEqual(trade.avg_entry, expected_entry_fill)
        self.assertAlmostEqual(trade.total_fees, expected_entry_fee)
        self.assertAlmostEqual(trade.total_slippage_cost, expected_entry_slippage)

        # -- a TP hit closing all size (a resting limit order: no slippage) --
        tp_event = [{"type": "TP_HIT", "tier_index": 0, "price": 110.0, "closed_weight": 1.0, "bar_index": 1}]
        harness.on_bar_events(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, events=tp_event)

        expected_tp_fee = 110.0 * 1.0 * 0.02 / 100.0  # maker
        expected_gross_pnl = (110.0 - expected_entry_fill) * 1.0
        close_bar = {"ts": open_bar["ts"] + 2 * 900_000}
        record = harness.on_position_closed(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, closed_bar=close_bar)

        self.assertIsNotNone(record)
        self.assertAlmostEqual(record.total_fees, expected_entry_fee + expected_tp_fee)
        self.assertAlmostEqual(record.total_slippage_cost, expected_entry_slippage)  # TP added none
        self.assertAlmostEqual(record.net_pnl, expected_gross_pnl - (expected_entry_fee + expected_tp_fee))
        self.assertAlmostEqual(record.duration_seconds, 1800.0)  # 2 * 900_000ms
        self.assertIsNotNone(record.r_multiple)
        self.assertIsNotNone(record.pct_return)
        self.assertEqual(len(harness.closed_trades), 1)
        self.assertNotIn(key, harness._open)

    def test_untracked_position_events_and_close_are_noop(self) -> None:
        pos = _make_single_tier_position()
        harness = DryRunHarness(state_dir=self.tmpdir, fees=self.fees)
        # never called on_trade_opened for this position
        harness.on_bar_events(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, events=[{"type": "TP_HIT", "tier_index": 0, "price": 110.0, "closed_weight": 1.0, "bar_index": 0}])
        record = harness.on_position_closed(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, closed_bar={"ts": 0})
        self.assertIsNone(record)
        self.assertEqual(harness.closed_trades, [])

    def test_position_closed_without_any_fill_is_dropped_not_recorded(self) -> None:
        """TP2's cancel_unfilled_orders() fires on price alone, regardless of
        whether any entry tier has ever filled -- so a setup that never
        pulls back into the ladder before price runs straight past the TP2
        level ends up with every entry cancelled and the position marked
        `closed`, despite zero size ever having opened. That must NOT be
        recorded as a trade (see the `filled_weight <= 0` guard in
        `on_position_closed`): no risk was ever taken and no PnL was ever
        possible, so counting it would inflate total_trades and the
        breakeven bucket with a zero-risk, zero-PnL no-op."""
        pos = _make_single_tier_position(entry_type="limit_fvg_outer")
        pos.ladder.fills[0] = False  # resting limit order, never filled
        harness = DryRunHarness(state_dir=self.tmpdir, fees=self.fees)

        open_bar = {"ts": 1_700_000_000_000}
        harness.on_trade_opened(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, bar=open_bar)
        key = ("KRYPTIC", "BTCUSDT", "LONG")
        self.assertIn(key, harness._open)
        self.assertEqual(harness._open[key].filled_weight, 0.0)

        close_bar = {"ts": open_bar["ts"] + 900_000}
        record = harness.on_position_closed(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, closed_bar=close_bar)

        self.assertIsNone(record)
        self.assertEqual(harness.closed_trades, [])
        self.assertNotIn(key, harness._open)
        # Institutional report's funnel counters (Signals generated vs.
        # Filled trades vs. Expired/Cancelled setups): this setup opened
        # and then expired unfilled -- both counters must reflect that.
        self.assertEqual(harness.trades_opened_count, 1)
        self.assertEqual(harness.trades_expired_unfilled_count, 1)

    def test_on_trade_opened_captures_setup_diagnostics_for_the_report(self) -> None:
        """`diagnostics` (TradeLifecycleManager.open_trade()'s own return
        value) flows through on_trade_opened -> on_position_closed into the
        PaperTradeRecord fields the institutional Excel/CSV report reads
        (expected_vwap, tp_levels, ker_ratio, rvol, htf_aligned,
        risk_atr_multiple, initial_sl, be_price)."""
        pos = _make_single_tier_position(direction="LONG", level=100.0, initial_sl=92.0, tp1=110.0, weight=1.0)
        harness = DryRunHarness(state_dir=self.tmpdir, fees=self.fees)
        diagnostics = {
            "expected_vwap": 99.5, "risk_atr_multiple": 1.8, "tp_levels": [110.0, 115.0, 120.0, 130.0, None],
            "ladder_diagnostics": {
                "regime": {"gates": {"chop": {"value": 0.42}, "liquidity_volume": {"value": 1.35}}},
                "bias": {"conditions": {"htf_structure": {"passed_long": True, "passed_short": False}}},
            },
        }
        open_bar = {"ts": 1_700_000_000_000}
        harness.on_trade_opened(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, bar=open_bar, diagnostics=diagnostics)

        trade = harness._open[("KRYPTIC", "BTCUSDT", "LONG")]
        self.assertEqual(trade.expected_vwap, 99.5)
        self.assertEqual(trade.risk_atr_multiple, 1.8)
        self.assertEqual(trade.tp_levels, (110.0, 115.0, 120.0, 130.0))
        self.assertEqual(trade.ker_ratio, 0.42)
        self.assertEqual(trade.rvol, 1.35)
        self.assertTrue(trade.htf_aligned)

        tp_event = [{"type": "TP_HIT", "tier_index": 0, "price": 110.0, "closed_weight": 1.0, "bar_index": 1}]
        harness.on_bar_events(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, events=tp_event)
        close_bar = {"ts": open_bar["ts"] + 2 * 900_000}
        record = harness.on_position_closed(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, closed_bar=close_bar)

        self.assertEqual(record.initial_sl, 92.0)
        self.assertEqual(record.expected_vwap, 99.5)
        self.assertEqual(record.tp_levels, (110.0, 115.0, 120.0, 130.0))
        self.assertEqual(record.ker_ratio, 0.42)
        self.assertEqual(record.rvol, 1.35)
        self.assertTrue(record.htf_aligned)
        self.assertEqual(record.bars_held, 2)  # last EXIT fill's bar_index (1) + 1
        self.assertIsNone(record.be_price)  # TP1 closed it outright -- breakeven never moved

    def test_crash_recovery_reconstructs_open_trade_and_closed_history(self) -> None:
        pos = _make_single_tier_position(level=100.0, initial_sl=92.0, tp1=110.0, weight=1.0)
        harness1 = DryRunHarness(state_dir=self.tmpdir, fees=self.fees)
        open_bar = {"ts": 1_700_000_000_000}
        harness1.on_trade_opened(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, bar=open_bar)

        # -- "crash": build a brand-new harness pointed at the same dir --
        harness2 = DryRunHarness(state_dir=self.tmpdir, fees=self.fees)
        key = ("KRYPTIC", "BTCUSDT", "LONG")
        self.assertIn(key, harness2._open)
        self.assertAlmostEqual(harness2._open[key].avg_entry, harness1._open[key].avg_entry)
        self.assertAlmostEqual(harness2._open[key].total_fees, harness1._open[key].total_fees)
        self.assertEqual(len(harness2._open[key].fills), 1)

        # -- resume on the recovered harness and close the trade --
        tp_event = [{"type": "TP_HIT", "tier_index": 0, "price": 110.0, "closed_weight": 1.0, "bar_index": 1}]
        harness2.on_bar_events(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, events=tp_event)
        close_bar = {"ts": open_bar["ts"] + 900_000}
        record = harness2.on_position_closed(strategy_id="KRYPTIC", symbol="BTCUSDT", position=pos, closed_bar=close_bar)
        self.assertIsNotNone(record)

        # -- a THIRD harness must see the closed trade and an empty open set --
        harness3 = DryRunHarness(state_dir=self.tmpdir, fees=self.fees)
        self.assertEqual(len(harness3.closed_trades), 1)
        self.assertEqual(harness3._open, {})
        raw = harness3.ledger_store.load_raw()
        self.assertIsNotNone(raw, "ledger file must be valid JSON after the close -- no state corruption")
        self.assertEqual(raw["extra"]["open_trades"], {})


# ---------------------------------------------------------------------------
# 1. LiveExchangeStream REST (no real network -- injected fake httpx client)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Queues of canned responses keyed by a URL substring, consumed in
    order (the last response in a queue repeats once exhausted)."""

    def __init__(self, responses_by_url_substr: dict) -> None:
        self._responses = {k: list(v) for k, v in responses_by_url_substr.items()}
        self.calls: list[str] = []

    async def get(self, url, **kwargs):
        self.calls.append(url)
        for substr, queue in self._responses.items():
            if substr in url:
                if len(queue) > 1:
                    return queue.pop(0)
                return queue[0]
        raise AssertionError(f"unexpected URL in test: {url}")


class LiveExchangeStreamRestTests(unittest.IsolatedAsyncioTestCase):
    async def test_historical_klines_binance_retries_then_succeeds(self) -> None:
        good_payload = [[1_700_000_000_000, "100.0", "101.0", "99.0", "100.5", "1000.0"]]
        client = _FakeAsyncClient({"/fapi/v1/klines": [_FakeResponse(500), _FakeResponse(200, good_payload)]})
        stream = LiveExchangeStream(venue="binance", client=client, max_retries=2, base_delay=0.001)

        rows = await stream.historical_klines("BTCUSDT", "15m", 300)

        self.assertEqual(len(client.calls), 2)  # one failure, one retry that succeeded
        self.assertEqual(rows, [{"ts": 1_700_000_000_000, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0}])

    async def test_historical_klines_binance_exhausts_retries_and_raises(self) -> None:
        client = _FakeAsyncClient({"/fapi/v1/klines": [_FakeResponse(500)]})
        stream = LiveExchangeStream(venue="binance", client=client, max_retries=1, base_delay=0.001)
        with self.assertRaises(httpx.HTTPStatusError):
            await stream.historical_klines("BTCUSDT", "15m", 300)
        self.assertEqual(len(client.calls), 2)  # initial attempt + 1 retry

    async def test_historical_klines_blofin_retries_on_api_error_code(self) -> None:
        good_payload = {"code": "0", "data": [[1_700_000_000_000, "100.0", "101.0", "99.0", "100.5", "1000.0"]]}
        bad_payload = {"code": "50011", "msg": "rate limited"}
        client = _FakeAsyncClient({"/api/v1/market/candles": [_FakeResponse(200, bad_payload), _FakeResponse(200, good_payload)]})
        stream = LiveExchangeStream(venue="blofin", client=client, max_retries=2, base_delay=0.001)

        rows = await stream.historical_klines("BTCUSDT", "15m", 200)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(rows[0]["close"], 100.5)

    async def test_funding_rate_blofin_always_none_without_network_call(self) -> None:
        client = _FakeAsyncClient({})
        stream = LiveExchangeStream(venue="blofin", client=client)
        rate = await stream.funding_rate("BTCUSDT")
        self.assertIsNone(rate)
        self.assertEqual(client.calls, [])

    async def test_funding_rate_binance_success(self) -> None:
        client = _FakeAsyncClient({"/fapi/v1/premiumIndex": [_FakeResponse(200, {"lastFundingRate": "0.0001"})]})
        stream = LiveExchangeStream(venue="binance", client=client, max_retries=1, base_delay=0.001)
        rate = await stream.funding_rate("BTCUSDT")
        self.assertAlmostEqual(rate, 0.01)

    async def test_funding_rate_never_raises_on_persistent_failure(self) -> None:
        client = _FakeAsyncClient({"/fapi/v1/premiumIndex": [_FakeResponse(500)]})
        stream = LiveExchangeStream(venue="binance", client=client, max_retries=1, base_delay=0.001)
        rate = await stream.funding_rate("BTCUSDT")  # must not raise
        self.assertIsNone(rate)

    async def test_funding_rate_never_raises_on_malformed_body(self) -> None:
        client = _FakeAsyncClient({"/fapi/v1/premiumIndex": [_FakeResponse(200, None)]})  # r.json() -> None -> .get() blows up
        stream = LiveExchangeStream(venue="binance", client=client, max_retries=1, base_delay=0.001)
        rate = await stream.funding_rate("BTCUSDT")  # must not raise
        self.assertIsNone(rate)


# ---------------------------------------------------------------------------
# 5. LiveRunner -- full paper-trading smoke test
# ---------------------------------------------------------------------------

class LiveRunnerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    async def test_paper_trade_full_lifecycle_and_clean_shutdown(self) -> None:
        """The spec's headline smoke test: mock live ticks drive a real
        (non-injected) trade open, tier fills, and a full close through
        LiveRunner + DryRunHarness, with fee/slippage-adjusted PnL computed
        and the paper ledger left in a clean, reloadable state -- plus the
        universe scanner's qualify-then-safely-disqualify lifecycle."""
        full = make_candles(371, direction=1, seed=1)  # proven 4-way bullish confluence on bar 370
        warm, signal_bar = full[:370], full[370]
        exchange = MockExchangeStream(historical=warm, live=[], funding_rate=0.02, btc_historical=warm)

        entry_engine = EntryLadderEngine(regime_filter=AlwaysAllowRegimeFilter())
        trade_manager = TradeLifecycleManager(entry_ladder_engine=entry_engine)
        arbiter = RiskArbiter(account_equity=10_000.0, global_risk_ceiling_pct=50.0, strategy_budgets_pct={"KRYPTIC": 30.0})

        async def fetch_tickers():
            return [{"symbol": "BTCUSDT", "quote_vol": 900_000_000.0}]

        console_lines: list[str] = []
        runner = LiveRunner(
            exchange=exchange, arbiter=arbiter, fetch_tickers=fetch_tickers,
            trade_manager_factory=lambda: trade_manager, state_dir=self.tmpdir,
            engine_kwargs={"warmup_bars": 370}, console=ConsoleDashboard(stream=console_lines.append),
        )

        added, _ = await runner.scanner.scan_once()
        self.assertEqual(added, {"BTCUSDT"})
        self.assertIn("BTCUSDT", runner._managed)
        self.assertTrue(any("UNIVERSE" in line for line in console_lines))
        managed = runner._managed["BTCUSDT"]  # kept across the whole test -- handle_bar auto-sweeps the dict entry once flat+disqualified

        # -- Signal: a real trade opens through the full production path --
        result = runner.handle_bar("BTCUSDT", signal_bar)
        self.assertTrue(result.trade_opened)
        pos = managed.active_position
        self.assertIsNotNone(pos)
        key = ("KRYPTIC", "BTCUSDT", pos.direction)
        self.assertIn(key, runner.harness._open, "DryRunHarness must record the paper trade the moment it opens")

        # -- Disqualify mid-trade: must NOT tear the engine down while a
        #    position is open (spec point 2's "terminal state" requirement) --
        await runner._on_symbol_disqualified("BTCUSDT")
        runner.sweep_pending_removals()
        self.assertIn("BTCUSDT", runner._managed, "an open position must survive a disqualification sweep")

        # -- Fill remaining entry tiers (same recipe test_multi_strategy.py
        #    uses). Under Step 10.10's TP_WEIGHTS ([0.60, 0.40, 0.0, 0.0,
        #    0.0]), TP1's 0.60 weight alone already exceeds the single
        #    filled tier's own 0.40 weight -- so this one wide bar both
        #    fills entry1 AND fully closes the position via TP1 outright,
        #    cancelling the still-pending entries 2-4 along the way (the
        #    same ladder-protection cancellation TP1/TP2 both trigger). --
        e1 = pos.ladder.levels[0]
        last_close = signal_bar["close"]
        fill_bar = {
            "ts": signal_bar["ts"] + 900_000, "open": last_close, "high": max(last_close, e1) + 1,
            "low": min(last_close, e1) - 0.5, "close": last_close, "volume": 500,
        }
        result2 = runner.handle_bar("BTCUSDT", fill_bar)
        self.assertTrue(pos.closed)
        self.assertTrue(result2.position_closed_this_bar)
        # handle_bar sweeps pending removals on every call, so this same
        # call both closes the position AND -- because it was already
        # disqualified above -- unregisters it from runner._managed.
        self.assertIsNone(managed.active_position)
        self.assertNotIn("BTCUSDT", runner._managed, "a disqualified symbol must be swept the moment its position goes flat")
        self.assertTrue(any("CLOSED" in line for line in console_lines))

        # -- Paper ledger: one closed, fee/slippage-adjusted trade --
        self.assertEqual(len(runner.harness.closed_trades), 1)
        record_dict = runner.harness.closed_trades[0]
        self.assertGreater(record_dict["total_fees"], 0.0)
        # Both this fill's entry and TP1 exit happen to be resting LIMIT
        # orders for this fixture's seed (no dynamic runner anymore to
        # guarantee a market-priced, slipped exit regardless) -- slippage
        # is a valid, non-negative computed value, not necessarily > 0.
        self.assertGreaterEqual(record_dict["total_slippage_cost"], 0.0)
        self.assertIsNotNone(record_dict["r_multiple"])
        self.assertIsNotNone(record_dict["pct_return"])
        self.assertAlmostEqual(record_dict["duration_seconds"], 900_000 / 1000.0)
        self.assertNotIn(key, runner.harness._open)

        # -- Clean shutdown: the paper ledger reloads identically from disk --
        raw = runner.harness.ledger_store.load_raw()
        self.assertIsNotNone(raw)
        self.assertEqual(raw["extra"]["open_trades"], {})
        self.assertEqual(len(raw["extra"]["closed_trades"]), 1)
        reloaded = DryRunHarness(state_dir=self.tmpdir)
        self.assertEqual(len(reloaded.closed_trades), 1)
        self.assertEqual(reloaded._open, {})

        # -- The engine's own state file must also show a clean, flat close --
        engine_raw = managed.engine.state_store.load_raw()
        self.assertIsNone(engine_raw["position"])

        # -- A further sweep is a safe no-op --
        runner.sweep_pending_removals()
        self.assertNotIn("BTCUSDT", runner._managed)


if __name__ == "__main__":
    unittest.main()
