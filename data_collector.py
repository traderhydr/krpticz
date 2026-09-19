"""data_collector.py -- live market data ingestion pipeline (KRYPTIC Step 6).

`MarketDataPipeline` bridges an exchange's data feed to `TradeLifecycleManager`
and `PositionState`: REST warmup on boot, a live closed-candle stream,
periodic funding/BTC-benchmark refresh, a heartbeat watchdog, and the
`on_candle_close` dispatcher that drives the whole strategy stack.

Exchange access is duck-typed, not tied to one vendor. Anything with this
shape works:

    class ExchangeAdapter:
        async def historical_klines(self, symbol: str, interval: str, limit: int) -> list[dict]:
            "Closed candles only, oldest first, each {'ts','open','high','low','close','volume'}."
        async def funding_rate(self, symbol: str) -> float | None:
            "Current funding rate, or None if unavailable -- never raises."
        async def stream_klines(self, symbol: str, interval: str, on_message: Callable[[dict], None]) -> None:
            "Runs until cancelled, calling on_message(msg) for every message
            (open ticks AND closes) -- MarketDataPipeline applies the
            bar-close guard itself, so this does not need to pre-filter."

This repo's own `exchanges.py` (`resolve_source()`, `fetch_klines`,
`fetch_funding`) already implements the REST half of this against
BloFin/Binance -- a real adapter should wrap those for
`historical_klines`/`funding_rate` and add a websocket client for
`stream_klines`, which does not exist elsewhere in this repo yet (the live
bot currently polls REST on an interval; see bot.py). Writing and proving
that websocket client against a real, rate-limited exchange endpoint is
out of scope for this module -- `MockExchangeStream` (below) implements
the same interface for fully offline testing, and everything in this file
is tested against it. Wiring a real websocket adapter is a follow-up, not
a change to this module's own logic.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

import pandas as pd

import indicators as ind
from risk_manager import PositionState, TradeLifecycleManager

log = logging.getLogger(__name__)

_REQUIRED_CANDLE_KEYS = ("ts", "open", "high", "low", "close", "volume")

_INTERVAL_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200, "1d": 86400,
}


def interval_to_seconds(interval: str) -> int:
    """Parse this repo's interval strings ('15m', '4h', ...) to seconds."""
    if interval in _INTERVAL_SECONDS:
        return _INTERVAL_SECONDS[interval]
    raise ValueError(f"unrecognized interval {interval!r}; known: {sorted(_INTERVAL_SECONDS)}")


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

class CandleBuffer:
    """Fixed-capacity, strictly-ordered ring buffer of closed candles.

    Timestamps are stored as UTC epoch milliseconds (int) -- this repo's
    existing convention (exchanges.py, smc_lite.py) and, being an absolute
    instant with no timezone attached, inherently unambiguous the way a
    naive datetime isn't. `to_dataframe()` exposes them as a "ts" column
    exactly as `directional_bias.py`'s `_extract_utc_timestamps` expects,
    so daily-anchored VWAP resets cleanly at 00:00 UTC.

    Args:
        capacity: Maximum candles retained (default 1000); oldest candles
            are silently dropped once full -- this is what keeps memory
            bounded over an unattended multi-day run.
    """

    def __init__(self, capacity: int = 1000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._data: deque[dict] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._data)

    def append(self, candle: dict) -> None:
        """Append one closed candle. Raises on a missing key or a
        non-increasing timestamp (out-of-order or duplicate) -- silently
        accepting either would corrupt every downstream rolling
        calculation, so this fails loudly instead."""
        missing = [k for k in _REQUIRED_CANDLE_KEYS if k not in candle]
        if missing:
            raise ValueError(f"candle missing required key(s): {missing}")
        ts = int(candle["ts"])
        if self._data and ts <= self._data[-1]["ts"]:
            raise ValueError(f"out-of-order or duplicate candle: ts={ts} <= last buffered ts={self._data[-1]['ts']}")
        self._data.append({
            "ts": ts,
            "open": float(candle["open"]), "high": float(candle["high"]),
            "low": float(candle["low"]), "close": float(candle["close"]),
            "volume": float(candle["volume"]),
        })

    def last(self) -> dict | None:
        return self._data[-1] if self._data else None

    def to_dataframe(self) -> pd.DataFrame:
        """Snapshot as a DataFrame (ts, open, high, low, close, volume),
        oldest first -- the exact shape indicators.py/directional_bias.py/
        entry_ladder.py/risk_manager.py expect."""
        if not self._data:
            return pd.DataFrame(columns=list(_REQUIRED_CANDLE_KEYS))
        return pd.DataFrame(list(self._data))


# ---------------------------------------------------------------------------
# HTF aggregation
# ---------------------------------------------------------------------------

class HTFAggregator:
    """Groups closed LTF candles into closed HTF candles, aligned to real
    UTC calendar boundaries (floor-division bucketing on timestamp), not
    "every N bars" -- so a gap in the LTF stream shortens one HTF bar
    instead of permanently drifting every later HTF boundary.

    A completed HTF bar is only ever returned once the FIRST bar of the
    *next* bucket has arrived -- proof the previous bucket is actually
    finished, never a guess. This is the same not-until-it's-truly-known
    discipline `indicators.detect_swing_pivots` uses for its own
    confirmation delay.
    """

    def __init__(self, htf_seconds: int) -> None:
        if htf_seconds <= 0:
            raise ValueError("htf_seconds must be > 0")
        self.htf_ms = htf_seconds * 1000
        self._current_bucket: int | None = None
        self._current: dict | None = None

    def ingest(self, ltf_candle: dict) -> dict | None:
        """Feed one closed LTF candle. Returns the just-completed HTF
        candle if this LTF candle started a new bucket, else None."""
        ts = int(ltf_candle["ts"])
        bucket = ts // self.htf_ms
        if self._current_bucket is None:
            self._current_bucket = bucket
            self._current = self._start(ltf_candle, bucket)
            return None
        if bucket == self._current_bucket:
            self._merge(ltf_candle)
            return None
        completed = self._current
        self._current_bucket = bucket
        self._current = self._start(ltf_candle, bucket)
        return completed

    def _start(self, c: dict, bucket: int) -> dict:
        return {
            "ts": bucket * self.htf_ms,
            "open": float(c["open"]), "high": float(c["high"]),
            "low": float(c["low"]), "close": float(c["close"]), "volume": float(c["volume"]),
        }

    def _merge(self, c: dict) -> None:
        cur = self._current
        cur["high"] = max(cur["high"], float(c["high"]))
        cur["low"] = min(cur["low"], float(c["low"]))
        cur["close"] = float(c["close"])
        cur["volume"] += float(c["volume"])


# ---------------------------------------------------------------------------
# Exchange interface + mock
# ---------------------------------------------------------------------------

class ExchangeStream(Protocol):
    async def historical_klines(self, symbol: str, interval: str, limit: int) -> list[dict]: ...
    async def funding_rate(self, symbol: str) -> float | None: ...
    async def stream_klines(self, symbol: str, interval: str, on_message: Callable[[dict], None]) -> Awaitable[None]: ...


class MockExchangeStream:
    """Replays fixed OHLCV history through the same interface a real
    exchange adapter would implement, so `MarketDataPipeline` is fully
    testable offline -- it never needs to know this isn't a real exchange.

    Args:
        historical: Candles returned by `historical_klines` for the traded
            symbol (and for BTCUSDT too, unless `btc_historical` is given).
        live: Messages replayed one at a time by `stream_klines`, each a
            dict with the traded symbol's OHLCV plus an "is_closed" bool
            (matching a typical kline-stream message shape) -- include
            `is_closed: False` entries to exercise the bar-close guard.
        funding_rate: Value `funding_rate()` returns every call (default
            0.0). Pass a 0-arg callable instead of a float to simulate a
            feed that changes over time or fails (raise inside the
            callable to exercise graceful degradation).
        btc_historical: Optional separate BTCUSDT history; defaults to the
            same series as `historical` if omitted.
    """

    def __init__(
        self,
        historical: list[dict],
        live: list[dict],
        funding_rate: float | Callable[[], float | None] | None = 0.0,
        btc_historical: list[dict] | None = None,
    ) -> None:
        self._historical = historical
        self._btc_historical = btc_historical if btc_historical is not None else historical
        self._live = live
        self._funding_rate = funding_rate
        self.sent_count = 0

    async def historical_klines(self, symbol: str, interval: str, limit: int) -> list[dict]:
        src = self._btc_historical if symbol.upper().startswith("BTC") else self._historical
        return list(src[-limit:])

    async def funding_rate(self, symbol: str) -> float | None:
        if callable(self._funding_rate):
            return self._funding_rate()
        return self._funding_rate

    async def stream_klines(self, symbol: str, interval: str, on_message: Callable[[dict], None]) -> None:
        for msg in self._live:
            on_message(msg)
            self.sent_count += 1
            await asyncio.sleep(0)  # yield control -- keeps this a real coroutine under a live event loop


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """What happened on one `on_candle_close` call, for tests/logging."""

    bar: dict
    bar_delta: float
    trade_opened: bool = False
    trade_diagnostics: dict | None = None
    position_events: list[dict] = field(default_factory=list)
    position_closed_this_bar: bool = False


class MarketDataPipeline:
    """Hybrid REST-warmup + live-stream pipeline driving a TradeLifecycleManager.

    Args:
        exchange: Anything implementing the `ExchangeStream` duck-typed
            interface (see module docstring); `MockExchangeStream` for tests.
        symbol: Traded pair, e.g. "ETHUSDT".
        trade_manager: The TradeLifecycleManager to drive.
        ltf_interval: Execution timeframe (default "15m").
        btc_htf_interval: BTC benchmark timeframe for RegimeFilter's macro
            beta gate (default "4h").
        symbol_htf_interval: The TRADED symbol's own higher timeframe, used
            by DirectionEngine's structure gate and risk_manager's TP3
            opposing-FVG check (default "1h") -- built locally via
            `HTFAggregator` from the same LTF stream, NOT fetched
            separately (this is a different series from `btc_htf_interval`,
            which is always BTCUSDT regardless of the traded symbol).
        warmup_bars: Candles requested on boot (default 300, per spec) --
            the exchange adapter may return fewer (BloFin's own REST wrapper
            in this repo caps a single call at 200); a shortfall is logged
            as a warning, not an error, and the strategy stack's own
            fail-safe gates (RegimeFilter/DirectionEngine) simply report
            "insufficient history" until enough live candles accumulate.
        buffer_capacity: CandleBuffer capacity for every buffer this
            pipeline owns (default 1000).
        heartbeat_grace_seconds: Extra seconds beyond one candle interval
            before a missing closed candle is flagged as a timeout (default
            10.0, per spec's "interval_seconds + 10s").
        funding_poll_seconds: How often to refresh funding rate + the BTC
            benchmark (default 300.0 = 5 minutes; spec calls for 5-15 min).
    """

    def __init__(
        self,
        *,
        exchange: ExchangeStream,
        symbol: str,
        trade_manager: TradeLifecycleManager,
        ltf_interval: str = "15m",
        btc_htf_interval: str = "4h",
        symbol_htf_interval: str = "1h",
        warmup_bars: int = 300,
        buffer_capacity: int = 1000,
        heartbeat_grace_seconds: float = 10.0,
        funding_poll_seconds: float = 300.0,
        ema_length: int = 50,
        atr_length: int = 14,
        supertrend_length: int = 10,
        supertrend_multiplier: float = 2.0,
    ) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.trade_manager = trade_manager
        self.ltf_interval = ltf_interval
        self.btc_htf_interval = btc_htf_interval
        self.symbol_htf_interval = symbol_htf_interval
        self.warmup_bars = warmup_bars
        self.interval_seconds = interval_to_seconds(ltf_interval)
        self.heartbeat_grace_seconds = heartbeat_grace_seconds
        self.funding_poll_seconds = funding_poll_seconds
        self.ema_length = ema_length
        self.atr_length = atr_length
        self.supertrend_length = supertrend_length
        self.supertrend_multiplier = supertrend_multiplier

        self.ltf_buffer = CandleBuffer(capacity=buffer_capacity)
        self.btc_buffer = CandleBuffer(capacity=buffer_capacity)
        self.symbol_htf_buffer = CandleBuffer(capacity=buffer_capacity)
        self._htf_aggregator = HTFAggregator(interval_to_seconds(symbol_htf_interval))

        self.position: PositionState | None = None
        self.closed_positions: list[PositionState] = []
        self.funding_rate: float | None = None
        self.funding_rate_updated_at: float | None = None

        self._last_candle_close_monotonic: float | None = None
        self._last_candle_ts: int | None = None
        self.reconnect_count = 0
        self.warnings: list[str] = []

    def _warn(self, message: str) -> None:
        log.warning(message)
        self.warnings.append(message)

    # -- boot / warmup --------------------------------------------------------

    async def warmup(self) -> None:
        """REST warmup: historical candles for the traded symbol and
        BTCUSDT, plus the current funding rate. Never raises -- a failed
        piece is logged as a warning and left at its safe default (empty
        buffer / funding_rate=None), matching this pipeline's graceful-
        degradation requirement from the very first call."""
        try:
            candles = await self.exchange.historical_klines(self.symbol, self.ltf_interval, self.warmup_bars)
        except Exception as e:
            self._warn(f"historical klines fetch failed for {self.symbol}: {e}")
            candles = []
        for c in candles:
            self.ltf_buffer.append(c)
            htf_bar = self._htf_aggregator.ingest(c)
            if htf_bar is not None:
                self.symbol_htf_buffer.append(htf_bar)
        if len(self.ltf_buffer) < self.warmup_bars:
            self._warn(f"only {len(self.ltf_buffer)}/{self.warmup_bars} {self.symbol} candles available on warmup")
        last = self.ltf_buffer.last()
        if last is not None:
            # Anchor the heartbeat clock and the gap-fill dedup cursor to the
            # newest warmup candle -- otherwise the first heartbeat check has
            # no baseline, and a later gap-fill would re-dispatch every
            # candle warmup already loaded (CandleBuffer would then correctly
            # reject them as out-of-order, but that's a startup crash we can
            # avoid entirely by just setting these here).
            self._last_candle_ts = last["ts"]
            self._last_candle_close_monotonic = time.monotonic()

        try:
            btc_candles = await self.exchange.historical_klines("BTCUSDT", self.btc_htf_interval, self.warmup_bars)
        except Exception as e:
            self._warn(f"historical klines fetch failed for BTCUSDT: {e}")
            btc_candles = []
        for c in btc_candles:
            self.btc_buffer.append(c)
        if len(self.btc_buffer) < self.warmup_bars:
            self._warn(f"only {len(self.btc_buffer)}/{self.warmup_bars} BTCUSDT candles available on warmup")

        await self._refresh_funding()

    async def _refresh_funding(self) -> None:
        try:
            self.funding_rate = await self.exchange.funding_rate(self.symbol)
            self.funding_rate_updated_at = time.monotonic()
        except Exception as e:
            self._warn(f"funding rate fetch failed: {e}")
            self.funding_rate = None  # graceful degradation: RegimeFilter treats None as pass-through

    # -- periodic REST poller -----------------------------------------------

    async def poll_funding_and_btc(self) -> None:
        """Run forever (until cancelled): refresh funding + the BTC
        benchmark every `funding_poll_seconds`."""
        while True:
            await asyncio.sleep(self.funding_poll_seconds)
            await self._refresh_funding()
            try:
                btc_candles = await self.exchange.historical_klines("BTCUSDT", self.btc_htf_interval, 2)
            except Exception as e:
                self._warn(f"BTC benchmark refresh failed: {e}")
                continue
            for c in btc_candles:
                if self.btc_buffer.last() is None or c["ts"] > self.btc_buffer.last()["ts"]:
                    self.btc_buffer.append(c)

    # -- heartbeat watchdog ---------------------------------------------------

    def is_heartbeat_timed_out(self) -> bool:
        """True if it's been longer than `interval_seconds + heartbeat_grace_seconds`
        since the last closed candle was dispatched. False before the first
        candle ever arrives -- that's a startup condition, not a timeout."""
        if self._last_candle_close_monotonic is None:
            return False
        elapsed = time.monotonic() - self._last_candle_close_monotonic
        return elapsed > (self.interval_seconds + self.heartbeat_grace_seconds)

    async def heartbeat_loop(self) -> None:
        """Run forever (until cancelled): poll `is_heartbeat_timed_out` and
        gap-fill via REST whenever it fires."""
        check_every = max(1.0, self.interval_seconds / 4)
        while True:
            await asyncio.sleep(check_every)
            if self.is_heartbeat_timed_out():
                await self._reconnect_and_gapfill()

    async def _reconnect_and_gapfill(self) -> None:
        """Best-effort resync: pull the most recent closed candles via REST
        and dispatch any this pipeline hasn't seen yet. Actual websocket
        reconnection is the exchange adapter's own responsibility (a real
        `stream_klines` should already retry internally); this method
        exists to backfill whatever candles were missed while it does."""
        self.reconnect_count += 1
        self._warn(f"heartbeat timeout -- gap-filling via REST (reconnect #{self.reconnect_count})")
        try:
            candles = await self.exchange.historical_klines(self.symbol, self.ltf_interval, 20)
        except Exception as e:
            self._warn(f"gap-fill REST fetch failed: {e}")
            return
        for c in sorted(candles, key=lambda c: c["ts"]):
            if self._last_candle_ts is None or c["ts"] > self._last_candle_ts:
                self.on_candle_close(c)

    # -- live stream ------------------------------------------------------------

    def _on_raw_message(self, msg: dict) -> None:
        """Bar-close guard: only a message explicitly marked closed reaches
        the strategy stack. Accepts either 'is_closed' or 'closed' (exchange
        wording varies); anything else is treated as an intermediate tick
        and dropped."""
        is_closed = msg.get("is_closed", msg.get("closed", False))
        if not is_closed:
            return
        candle = {k: msg[k] for k in _REQUIRED_CANDLE_KEYS}
        self.on_candle_close(candle)

    async def run(self) -> None:
        """Warm up, then run the live stream plus the heartbeat and funding
        pollers concurrently until the stream ends or is cancelled."""
        await self.warmup()
        heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        funding_task = asyncio.create_task(self.poll_funding_and_btc())
        try:
            await self.exchange.stream_klines(self.symbol, self.ltf_interval, self._on_raw_message)
        finally:
            heartbeat_task.cancel()
            funding_task.cancel()

    # -- strategy driver dispatcher --------------------------------------------

    def on_candle_close(self, bar: dict) -> DispatchResult:
        """The strategy driver: buffer the bar, advance HTF aggregation,
        compute this bar's indicator values, and either update the active
        position or evaluate a fresh trade.

        Args:
            bar: {"ts","open","high","low","close","volume"} for one closed candle.

        Returns:
            A DispatchResult describing what happened.
        """
        self._last_candle_close_monotonic = time.monotonic()
        self._last_candle_ts = int(bar["ts"])
        self.ltf_buffer.append(bar)
        symbol_htf_bar = self._htf_aggregator.ingest(bar)
        if symbol_htf_bar is not None:
            self.symbol_htf_buffer.append(symbol_htf_bar)

        bar_delta = float(
            ind.bar_delta(
                pd.Series([bar["high"]]), pd.Series([bar["low"]]),
                pd.Series([bar["close"]]), pd.Series([bar["volume"]]),
            ).iloc[0]
        )
        result = DispatchResult(bar=bar, bar_delta=bar_delta)

        df = self.ltf_buffer.to_dataframe()

        if self.position is not None and not self.position.closed:
            enriched = dict(bar)
            if len(df) >= self.supertrend_length:
                st = ind.supertrend(df["high"], df["low"], df["close"], length=self.supertrend_length, multiplier=self.supertrend_multiplier)
                st_dir = int(st.direction.iloc[-1])
                if st_dir != 0:
                    enriched["supertrend_direction"] = st_dir
            if len(df) >= self.ema_length:
                enriched["ema50"] = float(ind.ema(df["close"], self.ema_length).iloc[-1])
            if len(df) >= 20:
                # Step 10.10: PositionState's time-decay invalidation momentum
                # check -- must match backtest_engine.py's VectorizedSignalCache.
                # enrich() exactly (same fixed length 20), or the live dispatcher
                # and the backtest would trail/invalidate positions differently.
                enriched["ema20"] = float(ind.ema(df["close"], 20).iloc[-1])
            atr_series = ind.average_true_range(df["high"], df["low"], df["close"], length=self.atr_length)
            if not pd.isna(atr_series.iloc[-1]):
                enriched["atr"] = float(atr_series.iloc[-1])

            htf_bar_dict = None
            if symbol_htf_bar is not None:
                htf_bar_dict = {"high": symbol_htf_bar["high"], "low": symbol_htf_bar["low"]}

            events = self.position.update(enriched, htf_bar=htf_bar_dict)
            result.position_events = events
            if self.position.closed:
                self.closed_positions.append(self.position)
                self.position = None
                result.position_closed_this_bar = True
        else:
            htf_df = self.symbol_htf_buffer.to_dataframe()
            btc_df = self.btc_buffer.to_dataframe()
            position, diagnostics = self.trade_manager.open_trade(
                df, btc_df, self.funding_rate,
                htf_df=htf_df if len(htf_df) else None,
            )
            result.trade_diagnostics = diagnostics
            if position is not None:
                self.position = position
                result.trade_opened = True

        return result
