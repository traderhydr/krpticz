"""test_exchange_reconciler.py -- tests for exchange_reconciler.py.

No real network access anywhere in this file: BinanceSignedClient's own
HTTP calls are exercised only via `_sign()`'s deterministic HMAC math, and
ExchangeReconciler's reconciliation logic is tested against a small
hand-written fake standing in for BinanceSignedClient (matching this
repo's existing test_live_runner.py convention).

Run with:  python3 -m unittest test_exchange_reconciler -v
"""
from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
import unittest
from pathlib import Path

from exchange_reconciler import (
    BinanceSignedClient,
    ExchangeReconciler,
    PermissionError_,
    RealFill,
    _sign,
    _split_increasing_reducing,
    _weighted_avg,
    attribution_status,
)


class SignTests(unittest.TestCase):
    def test_matches_a_hand_computed_hmac_sha256(self) -> None:
        secret = "test-secret"
        query = "symbol=BTCUSDT&timestamp=1700000000000"
        expected = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        self.assertEqual(_sign(secret, query), expected)

    def test_different_queries_produce_different_signatures(self) -> None:
        secret = "test-secret"
        self.assertNotEqual(_sign(secret, "a=1"), _sign(secret, "a=2"))


class AttributionStatusTests(unittest.TestCase):
    def test_no_open_trades_is_none(self) -> None:
        self.assertEqual(attribution_status({}, "BTCUSDT", "LONG"), "NONE")

    def test_exactly_one_is_confident(self) -> None:
        trades = {"k1": {"symbol": "BTCUSDT", "side": "LONG"}}
        self.assertEqual(attribution_status(trades, "BTCUSDT", "LONG"), "CONFIDENT")

    def test_two_engines_same_symbol_and_side_is_ambiguous(self) -> None:
        trades = {
            "k1": {"symbol": "BTCUSDT", "side": "LONG", "engine": "ZENITH"},
            "k2": {"symbol": "BTCUSDT", "side": "LONG", "engine": "KRYPTIC"},
        }
        self.assertEqual(attribution_status(trades, "BTCUSDT", "LONG"), "AMBIGUOUS")

    def test_opposite_side_on_same_symbol_does_not_count_as_ambiguous(self) -> None:
        """Hedge mode keeps LONG/SHORT as separate buckets -- a LONG from
        one engine and a SHORT from another on the same symbol are each
        independently CONFIDENT, not ambiguous with each other."""
        trades = {
            "k1": {"symbol": "BTCUSDT", "side": "LONG", "engine": "ZENITH"},
            "k2": {"symbol": "BTCUSDT", "side": "SHORT", "engine": "GEM"},
        }
        self.assertEqual(attribution_status(trades, "BTCUSDT", "LONG"), "CONFIDENT")
        self.assertEqual(attribution_status(trades, "BTCUSDT", "SHORT"), "CONFIDENT")

    def test_different_symbols_never_collide(self) -> None:
        trades = {
            "k1": {"symbol": "BTCUSDT", "side": "LONG"},
            "k2": {"symbol": "ETHUSDT", "side": "LONG"},
        }
        self.assertEqual(attribution_status(trades, "BTCUSDT", "LONG"), "CONFIDENT")


class WeightedAvgTests(unittest.TestCase):
    def test_empty_list_returns_none(self) -> None:
        self.assertIsNone(_weighted_avg([]))

    def test_single_fill_returns_its_own_price(self) -> None:
        fills = [RealFill(side="BUY", price=100.0, qty=1.0, time_ms=1, position_side="LONG")]
        self.assertEqual(_weighted_avg(fills), 100.0)

    def test_weights_by_quantity(self) -> None:
        fills = [
            RealFill(side="BUY", price=100.0, qty=3.0, time_ms=1, position_side="LONG"),
            RealFill(side="BUY", price=110.0, qty=1.0, time_ms=2, position_side="LONG"),
        ]
        # (100*3 + 110*1) / 4 = 102.5
        self.assertAlmostEqual(_weighted_avg(fills), 102.5, places=6)

    def test_zero_total_quantity_returns_none(self) -> None:
        fills = [RealFill(side="BUY", price=100.0, qty=0.0, time_ms=1, position_side="LONG")]
        self.assertIsNone(_weighted_avg(fills))


class SplitIncreasingReducingTests(unittest.TestCase):
    def test_long_buys_increase_sells_reduce(self) -> None:
        fills = [
            RealFill(side="BUY", price=100.0, qty=1.0, time_ms=1, position_side="LONG"),
            RealFill(side="SELL", price=105.0, qty=1.0, time_ms=2, position_side="LONG"),
        ]
        increasing, reducing = _split_increasing_reducing(fills, "LONG", "LONG")
        self.assertEqual([f.side for f in increasing], ["BUY"])
        self.assertEqual([f.side for f in reducing], ["SELL"])

    def test_short_sells_increase_buys_reduce(self) -> None:
        fills = [
            RealFill(side="SELL", price=100.0, qty=1.0, time_ms=1, position_side="SHORT"),
            RealFill(side="BUY", price=95.0, qty=1.0, time_ms=2, position_side="SHORT"),
        ]
        increasing, reducing = _split_increasing_reducing(fills, "SHORT", "SHORT")
        self.assertEqual([f.side for f in increasing], ["SELL"])
        self.assertEqual([f.side for f in reducing], ["BUY"])

    def test_filters_out_the_other_positionside_bucket(self) -> None:
        """Hedge mode: a SHORT-bucket fill must never leak into a LONG
        trade's reconciliation, even on the same symbol."""
        fills = [
            RealFill(side="BUY", price=100.0, qty=1.0, time_ms=1, position_side="LONG"),
            RealFill(side="SELL", price=200.0, qty=5.0, time_ms=1, position_side="SHORT"),
        ]
        increasing, reducing = _split_increasing_reducing(fills, "LONG", "LONG")
        self.assertEqual(len(increasing), 1)
        self.assertEqual(len(reducing), 0)


class VerifyReadOnlyTests(unittest.IsolatedAsyncioTestCase):
    async def test_raises_on_withdrawals_enabled(self) -> None:
        client = BinanceSignedClient("key", "secret")

        async def fake_signed_get(_client, _base, _path, _params):
            return {"enableReading": True, "enableWithdrawals": True}
        client._signed_get = fake_signed_get
        client._sync_clock = lambda _client: _noop()

        with self.assertRaises(PermissionError_):
            await client.verify_read_only(object())

    async def test_raises_on_margin_trading_enabled(self) -> None:
        client = BinanceSignedClient("key", "secret")

        async def fake_signed_get(_client, _base, _path, _params):
            return {"enableReading": True, "enableWithdrawals": False, "enableSpotAndMarginTrading": True}
        client._signed_get = fake_signed_get
        client._sync_clock = lambda _client: _noop()

        with self.assertRaises(PermissionError_):
            await client.verify_read_only(object())

    async def test_passes_for_a_clean_read_only_key(self) -> None:
        client = BinanceSignedClient("key", "secret")

        async def fake_signed_get(_client, _base, _path, _params):
            return {"enableReading": True, "enableWithdrawals": False, "enableSpotAndMarginTrading": False, "enableMargin": False}
        client._signed_get = fake_signed_get
        client._sync_clock = lambda _client: _noop()

        result = await client.verify_read_only(object())
        self.assertTrue(result["enableReading"])


async def _noop():
    return None


class _FakeSigner:
    """Stands in for BinanceSignedClient in ExchangeReconciler tests --
    returns canned fills/positions instead of making real HTTP calls."""

    def __init__(self, fills_by_symbol: dict[str, list[RealFill]], positions_by_symbol: dict[str, list[dict]]) -> None:
        self._fills = fills_by_symbol
        self._positions = positions_by_symbol

    async def user_trades(self, _client, symbol, _start_ms):
        return self._fills.get(symbol, [])

    async def position_risk(self, _client, symbol):
        return self._positions.get(symbol, [])


class ExchangeReconcilerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = Path(self.tmpdir) / "reconcile_log.jsonl"

    def test_rejects_one_way_position_mode(self) -> None:
        with self.assertRaises(NotImplementedError):
            ExchangeReconciler(_FakeSigner({}, {}), position_mode="one_way", log_path=self.log_path)

    async def test_ambiguous_when_two_engines_share_symbol_and_side(self) -> None:
        signer = _FakeSigner({}, {})
        reconciler = ExchangeReconciler(signer, log_path=self.log_path)

        class _FakeRisk:
            trades = {
                "k1": {"symbol": "BTCUSDT", "side": "LONG", "engine": "ZENITH", "last_ts": 1_700_000_000_000},
                "k2": {"symbol": "BTCUSDT", "side": "LONG", "engine": "KRYPTIC", "last_ts": 1_700_000_000_000},
            }

        results = await reconciler.run_cycle(object(), _FakeRisk())
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.status == "AMBIGUOUS" for r in results))

    async def test_no_real_fills_yet_when_binance_shows_nothing(self) -> None:
        signer = _FakeSigner(fills_by_symbol={}, positions_by_symbol={})
        reconciler = ExchangeReconciler(signer, log_path=self.log_path)

        class _FakeRisk:
            trades = {
                "k1": {
                    "symbol": "ETHUSDT", "side": "LONG", "engine": "GEM", "last_ts": 1_700_000_000_000,
                    "eavg": None, "fill_pct": 0.0, "risk_px": 5.0,
                },
            }

        results = await reconciler.run_cycle(object(), _FakeRisk())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "NO_REAL_FILLS_YET")

    async def test_open_status_with_real_fills_but_position_still_active(self) -> None:
        fills = {"ETHUSDT": [RealFill(side="BUY", price=100.0, qty=2.0, time_ms=1_700_000_001_000, position_side="LONG")]}
        positions = {"ETHUSDT": [{"positionSide": "LONG", "positionAmt": "2.0"}]}
        signer = _FakeSigner(fills, positions)
        reconciler = ExchangeReconciler(signer, log_path=self.log_path)

        class _FakeRisk:
            trades = {
                "k1": {
                    "symbol": "ETHUSDT", "side": "LONG", "engine": "GEM", "last_ts": 1_700_000_000_000,
                    "eavg": 101.0, "fill_pct": 100.0, "risk_px": 5.0, "r_realized": 0.0,
                },
            }

        results = await reconciler.run_cycle(object(), _FakeRisk())
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.status, "OPEN")
        self.assertAlmostEqual(r.real_eavg, 100.0, places=6)
        self.assertIsNotNone(r.discrepancy())
        self.assertAlmostEqual(r.discrepancy()["eavg_delta_pct"], (100.0 - 101.0) / 101.0 * 100.0, places=4)

    async def test_closed_status_computes_real_r_from_prices_only(self) -> None:
        """Real R must be derivable purely from prices (entry/exit/risk_px)
        -- never from Binance's dollar PnL, which this module can't scale
        correctly without knowing the real position's contract size."""
        fills = {
            "ETHUSDT": [
                RealFill(side="BUY", price=100.0, qty=2.0, time_ms=1_700_000_001_000, position_side="LONG"),
                RealFill(side="SELL", price=110.0, qty=2.0, time_ms=1_700_000_005_000, position_side="LONG"),
            ]
        }
        positions = {"ETHUSDT": [{"positionSide": "LONG", "positionAmt": "0.0"}]}
        signer = _FakeSigner(fills, positions)
        reconciler = ExchangeReconciler(signer, log_path=self.log_path)

        class _FakeRisk:
            trades = {
                "k1": {
                    "symbol": "ETHUSDT", "side": "LONG", "engine": "ZENITH", "last_ts": 1_700_000_000_000,
                    "eavg": 101.0, "fill_pct": 100.0, "risk_px": 5.0, "r_realized": 1.5,
                },
            }

        results = await reconciler.run_cycle(object(), _FakeRisk())
        r = results[0]
        self.assertEqual(r.status, "CLOSED")
        self.assertAlmostEqual(r.real_eavg, 100.0, places=6)
        self.assertAlmostEqual(r.real_exit, 110.0, places=6)
        # (110 - 100) / 5 = 2.0 R, independent of real position size.
        self.assertAlmostEqual(r.real_r, 2.0, places=6)
        self.assertAlmostEqual(r.discrepancy()["r_delta"], 2.0 - 1.5, places=6)

    async def test_short_side_real_r_is_mirrored(self) -> None:
        fills = {
            "BTCUSDT": [
                RealFill(side="SELL", price=100.0, qty=1.0, time_ms=1_700_000_001_000, position_side="SHORT"),
                RealFill(side="BUY", price=90.0, qty=1.0, time_ms=1_700_000_005_000, position_side="SHORT"),
            ]
        }
        positions = {"BTCUSDT": [{"positionSide": "SHORT", "positionAmt": "0.0"}]}
        signer = _FakeSigner(fills, positions)
        reconciler = ExchangeReconciler(signer, log_path=self.log_path)

        class _FakeRisk:
            trades = {
                "k1": {
                    "symbol": "BTCUSDT", "side": "SHORT", "engine": "KRYPTIC", "last_ts": 1_700_000_000_000,
                    "eavg": 100.0, "fill_pct": 100.0, "risk_px": 5.0, "r_realized": 2.0,
                },
            }

        results = await reconciler.run_cycle(object(), _FakeRisk())
        r = results[0]
        self.assertEqual(r.status, "CLOSED")
        # SHORT: (entry - exit) / risk_px = (100 - 90) / 5 = 2.0R
        self.assertAlmostEqual(r.real_r, 2.0, places=6)

    async def test_writes_jsonl_log(self) -> None:
        signer = _FakeSigner({}, {})
        reconciler = ExchangeReconciler(signer, log_path=self.log_path)

        class _FakeRisk:
            trades = {
                "k1": {"symbol": "BTCUSDT", "side": "LONG", "engine": "ZENITH", "last_ts": 1_700_000_000_000},
                "k2": {"symbol": "BTCUSDT", "side": "LONG", "engine": "KRYPTIC", "last_ts": 1_700_000_000_000},
            }

        await reconciler.run_cycle(object(), _FakeRisk())
        self.assertTrue(self.log_path.exists())
        lines = self.log_path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)
        row = json.loads(lines[0])
        self.assertEqual(row["status"], "AMBIGUOUS")


if __name__ == "__main__":
    unittest.main()
