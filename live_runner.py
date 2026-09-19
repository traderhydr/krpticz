"""live_runner.py -- concrete exchange adapter, dynamic universe scanner &
paper trading runner (KRYPTIC Step 10).

Four pieces, usable independently or wired together via `LiveRunner`:

    LiveExchangeStream  -- concrete `ExchangeStream` (data_collector.py's
                            duck-typed interface) against BloFin or Binance
                            USDT-M: REST historical_klines/funding_rate
                            wrapped in `with_retry`, plus a real WebSocket
                            stream_klines with auto-reconnect and
                            ping/pong keep-alive.
    UniverseScanner     -- periodic (default 4h) top-N high-volume universe
                            scan with a stablecoin/leveraged-token filter
                            and qualify/disqualify hooks.
    DryRunHarness       -- paper-trading fee/slippage overlay on top of
                            risk_manager.py's own (idealized) fill/PnL
                            state machine, persisted atomically via
                            resilience_manager.StateStore.
    LiveRunner          -- orchestrates all of the above plus
                            multi_strategy_manager.MultiStrategyManager and
                            a console dashboard into one runnable loop.

Honest scope note (same convention this KRYPTIC track has used since
Step 6's data_collector.py docstring): this sandboxed environment's
outbound network is restricted to a small allowlist -- it has no route to
any exchange's REST/WebSocket endpoints or documentation site (verified:
both openapi.blofin.com and docs.blofin.com are blocked here). That means:

  - `LiveExchangeStream.historical_klines`/`funding_rate` reuse this
    repo's own `exchanges.py` helpers (`_rows_from_binance`,
    `_to_blofin_symbol`, `BLOFIN_BASE`, ...), which the live ZENITH/GEM bot
    already runs against real BloFin/Binance endpoints in production --
    those code paths are proven, this file only adds `with_retry` and a
    duck-typed wrapper around them.
  - `LiveExchangeStream.stream_klines` is new code with NO real-network
    verification possible in this environment. It's written against:
      * Binance: the public USDT-M futures combined-stream kline endpoint
        (wss://fstream.binance.com/stream), a long-stable, extensively
        documented public API independent of any doc-site access.
      * BloFin: the OKX-compatible convention (`{"op":"subscribe","args":
        [{"channel":"candleX","instId":...}]}`, a `{code,msg,data}`
        envelope, and a text "ping"/"pong" keep-alive) that this repo's
        OWN `exchanges.py` already documents BloFin's REST API as
        following -- but the exact WS channel/field names below are
        inferred from that family of API, NOT confirmed against BloFin's
        live WS docs (blocked here). Smoke-test this against a real
        connection before trusting it in production -- the same caveat
        `exchanges.py`'s own BloFin section already carries for its REST
        endpoints.
  - `test_live_runner.py` accordingly tests every pure/deterministic piece
    (universe filtering/ranking, fee/slippage math, DryRunHarness fills +
    crash-recovery, and LiveExchangeStream's REST retry/parsing logic
    against an injected fake HTTP client) without any real network, plus a
    full paper-trading run driven by `data_collector.MockExchangeStream` --
    it does not and cannot exercise the real `stream_klines` WebSocket
    code path.

DRY_RUN also isn't a live/paper TOGGLE in the sense of "flip it and real
orders start flowing" -- this repo has never wired up an authenticated
order-placement endpoint anywhere (ZENITH/GEM post signals to Cornix over
Telegram; they don't call an exchange trading API directly either). Paper
trading via `DryRunHarness` is therefore the entire order-execution path
today, regardless of this flag; the flag exists so that wiring, once it's
built, has an obvious place to gate on.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import httpx

import exchanges as ex
from engine import KrypticEngine
from multi_strategy_manager import ManagedStrategyEngine, MultiStrategyManager, RiskArbiter
from resilience_manager import StateStore, with_retry
from risk_manager import PositionState, TradeLifecycleManager

log = logging.getLogger(__name__)

DRY_RUN = os.getenv("KRYPTIC_DRY_RUN", "true").strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# 1. Concrete exchange adapter
# ---------------------------------------------------------------------------

_BINANCE_FAPI = "https://fapi.binance.com"
_BINANCE_WS_BASE = "wss://fstream.binance.com/stream"
_BLOFIN_WS_PUBLIC = "wss://openapi.blofin.com/ws/public"

_BLOFIN_CANDLE_BAR = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H", "1d": "1D",
}
_BINANCE_KLINE_INTERVAL = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1d",
}

_REQUIRED_CANDLE_KEYS = ("ts", "open", "high", "low", "close", "volume")


def _is_retryable_adapter_error(exc: BaseException) -> bool:
    """`with_retry`'s default predicate only recognizes HTTP-status/network
    exceptions; BloFin's `{code,msg,data}` envelope reports API-level
    errors (including rate limiting) via a JSON `code` field on an HTTP 200
    response (see exchanges.py's BLOFIN_BASE section), which surfaces here
    as a plain RuntimeError -- also worth retrying."""
    from resilience_manager import is_retryable_http_error
    return is_retryable_http_error(exc) or isinstance(exc, RuntimeError)


class LiveExchangeStream:
    """Concrete `ExchangeStream` (data_collector.py) against a real
    perpetual exchange. See this module's docstring for the honest scope
    note on `stream_klines` specifically.

    Args:
        venue: "blofin" (default) or "binance" -- defaults to the
            `DATA_SOURCE` env var, matching exchanges.py's own
            `resolve_source()` convention, so this adapter and the live
            ZENITH/GEM bot always point at the same venue.
        client: Injectable httpx.AsyncClient (tests use a fake one; a real
            deployment can leave this None and let one be created lazily).
        max_retries / base_delay: Forwarded to `with_retry` for every REST call.
        ws_ping_interval: Seconds between BloFin's application-level text
            "ping" frames (default 20.0); also used as Binance's WS-protocol
            ping_interval.
        ws_reconnect_base_delay / ws_reconnect_max_delay: Exponential
            backoff bounds for `stream_klines`'s auto-reconnect loop.
    """

    def __init__(
        self,
        *,
        venue: str | None = None,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        ws_ping_interval: float = 20.0,
        ws_reconnect_base_delay: float = 2.0,
        ws_reconnect_max_delay: float = 60.0,
    ) -> None:
        self.venue = (venue or os.getenv("DATA_SOURCE", "blofin")).strip().lower()
        if self.venue not in ("blofin", "binance"):
            log.warning("LiveExchangeStream: unknown venue=%r, defaulting to blofin", self.venue)
            self.venue = "blofin"
        self._client = client
        self._owns_client = client is None
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.ws_ping_interval = ws_ping_interval
        self.ws_reconnect_base_delay = ws_reconnect_base_delay
        self.ws_reconnect_max_delay = ws_reconnect_max_delay
        self.reconnect_count = 0
        self._subscribed: set[str] = set()
        self._pending_subscribe: set[str] = set()
        self._pending_unsubscribe: set[str] = set()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20.0)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- REST: historical klines ---------------------------------------------

    async def historical_klines(self, symbol: str, interval: str, limit: int) -> list[dict]:
        wrapped = with_retry(
            max_retries=self.max_retries, base_delay=self.base_delay, is_retryable=_is_retryable_adapter_error,
        )(self._historical_klines_once)
        return await wrapped(symbol, interval, limit)

    async def _historical_klines_once(self, symbol: str, interval: str, limit: int) -> list[dict]:
        client = await self._get_client()
        if self.venue == "binance":
            r = await client.get(
                f"{_BINANCE_FAPI}/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit}, headers=ex.UA, timeout=15,
            )
            r.raise_for_status()
            return ex._rows_from_binance(r.json())

        inst_id = ex._to_blofin_symbol(symbol)
        bar = _BLOFIN_CANDLE_BAR.get(interval, interval)
        r = await client.get(
            f"{ex.BLOFIN_BASE}/api/v1/market/candles",
            params={"instId": inst_id, "bar": bar, "limit": str(min(limit, 200))}, headers=ex.UA, timeout=15,
        )
        r.raise_for_status()
        body = r.json() or {}
        if str(body.get("code")) not in ("0", "00000"):
            raise RuntimeError(f"blofin candles error: {body.get('msg') or body.get('code')}")
        rows = [
            {"ts": int(k[0]), "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}
            for k in body.get("data") or []
        ]
        rows.sort(key=lambda c: c["ts"])
        return rows

    # -- REST: funding rate -----------------------------------------------------

    async def funding_rate(self, symbol: str) -> float | None:
        """Never raises (per the ExchangeStream protocol contract) --
        BloFin has no documented public funding-rate field (see
        exchanges.py's `fetch_funding_blofin`) so this always returns None
        for that venue without a network call; Binance failures, even ones
        `with_retry` can't recover, degrade to None rather than propagating."""
        if self.venue == "blofin":
            return None
        try:
            wrapped = with_retry(max_retries=self.max_retries, base_delay=self.base_delay, on_exhausted=lambda e: None)(self._funding_rate_once)
            return await wrapped(symbol)
        except Exception as e:
            log.warning("LiveExchangeStream: funding_rate(%s) failed: %s", symbol, e)
            return None

    async def _funding_rate_once(self, symbol: str) -> float | None:
        client = await self._get_client()
        r = await client.get(f"{_BINANCE_FAPI}/fapi/v1/premiumIndex", params={"symbol": symbol}, headers=ex.UA, timeout=10)
        r.raise_for_status()
        return float(r.json().get("lastFundingRate") or 0) * 100.0

    # -- WS: live stream ----------------------------------------------------------

    async def stream_klines(self, symbol: str | Sequence[str], interval: str, on_message: Callable[[dict], None]) -> None:
        """Multi-symbol-capable over ONE connection: `symbol` may be a
        single string (duck-type compatible with a standalone
        MarketDataPipeline/KrypticEngine) or a list (the universe
        scanner's shared-connection use case). Runs until cancelled,
        auto-reconnecting with exponential backoff on any drop. Every
        dispatched message includes `is_closed` (the bar-close guard) plus
        `symbol`, so a caller driving multiple symbols off one stream (e.g.
        `MultiStrategyManager.on_candle_close`) can route it correctly.

        Call `subscribe_symbols`/`unsubscribe_symbols` at any time while
        this is running to change the live symbol set without tearing the
        connection down.
        """
        self._subscribed = {s.upper() for s in ([symbol] if isinstance(symbol, str) else symbol)}
        attempt = 0
        while True:
            try:
                if self.venue == "binance":
                    await self._stream_binance(interval, on_message)
                else:
                    await self._stream_blofin(interval, on_message)
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.reconnect_count += 1
                delay = min(self.ws_reconnect_max_delay, self.ws_reconnect_base_delay * (2 ** attempt))
                log.warning(
                    "LiveExchangeStream(%s): stream dropped (%s) -- reconnecting in %.1fs (attempt %d, total reconnects %d)",
                    self.venue, e, delay, attempt + 1, self.reconnect_count,
                )
                await asyncio.sleep(delay)
                attempt += 1

    def subscribe_symbols(self, symbols: Sequence[str]) -> None:
        new = {s.upper() for s in symbols} - self._subscribed
        self._pending_subscribe |= new
        self._subscribed |= new

    def unsubscribe_symbols(self, symbols: Sequence[str]) -> None:
        gone = {s.upper() for s in symbols} & self._subscribed
        self._pending_unsubscribe |= gone
        self._subscribed -= gone

    async def _stream_binance(self, interval: str, on_message: Callable[[dict], None]) -> None:
        import websockets  # imported lazily: only needed by the live WS path, never by REST-only/offline tests

        biv = _BINANCE_KLINE_INTERVAL.get(interval, interval)
        streams = "/".join(f"{s.lower()}@kline_{biv}" for s in sorted(self._subscribed))
        url = f"{_BINANCE_WS_BASE}?streams={streams}"
        async with websockets.connect(url, ping_interval=self.ws_ping_interval, ping_timeout=self.ws_ping_interval * 2) as ws:
            # Binance's WS-protocol ping/pong is already handled automatically
            # by the `websockets` library (ping_interval/ping_timeout above)
            # -- this satisfies the spec's "automated ping/pong keep-alive"
            # for this venue without any extra application-level loop.
            while True:
                await self._flush_pending_binance(ws, biv)
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=self.ws_ping_interval * 3)
                except asyncio.TimeoutError:
                    continue
                msg = json.loads(raw)
                data = msg.get("data") or msg
                k = data.get("k")
                if not k:
                    continue
                on_message({
                    "symbol": str(k["s"]).upper(), "ts": int(k["t"]), "open": float(k["o"]), "high": float(k["h"]),
                    "low": float(k["l"]), "close": float(k["c"]), "volume": float(k["v"]), "is_closed": bool(k["x"]),
                })

    async def _flush_pending_binance(self, ws: Any, biv: str) -> None:
        # SUBSCRIBE/UNSUBSCRIBE control frames on the SAME connection --
        # Binance's combined-stream endpoint supports adding/dropping
        # streams live, no reconnect needed for a universe-scanner change.
        if self._pending_subscribe:
            params = [f"{s.lower()}@kline_{biv}" for s in self._pending_subscribe]
            await ws.send(json.dumps({"method": "SUBSCRIBE", "params": params, "id": int(time.time() * 1000)}))
            self._pending_subscribe.clear()
        if self._pending_unsubscribe:
            params = [f"{s.lower()}@kline_{biv}" for s in self._pending_unsubscribe]
            await ws.send(json.dumps({"method": "UNSUBSCRIBE", "params": params, "id": int(time.time() * 1000)}))
            self._pending_unsubscribe.clear()

    async def _stream_blofin(self, interval: str, on_message: Callable[[dict], None]) -> None:
        import websockets  # see _stream_binance's note

        bar = _BLOFIN_CANDLE_BAR.get(interval, interval)
        async with websockets.connect(_BLOFIN_WS_PUBLIC) as ws:
            if self._subscribed:
                await ws.send(json.dumps({
                    "op": "subscribe",
                    "args": [{"channel": f"candle{bar}", "instId": ex._to_blofin_symbol(s)} for s in sorted(self._subscribed)],
                }))
            ping_task = asyncio.create_task(self._blofin_ping_loop(ws))
            try:
                while True:
                    await self._flush_pending_blofin(ws, bar)
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=self.ws_ping_interval * 3)
                    except asyncio.TimeoutError:
                        continue
                    if raw == "pong":
                        continue
                    msg = json.loads(raw)
                    arg = msg.get("arg") or {}
                    channel = str(arg.get("channel", ""))
                    if not channel.startswith("candle"):
                        continue
                    symbol = ex._from_blofin_symbol(str(arg.get("instId", "")))
                    for row in msg.get("data") or []:
                        # OKX-family v5 candle push appends a "confirm" flag
                        # ("0"/"1") as the LAST element once the channel
                        # includes it; treat an absent flag as closed (older
                        # message shape / different bar width) rather than
                        # silently dropping every candle from an endpoint
                        # that doesn't send it -- see this module's
                        # docstring on why this is unverified against a
                        # live connection.
                        is_closed = str(row[-1]) == "1" if len(row) > 6 else True
                        on_message({
                            "symbol": symbol, "ts": int(row[0]), "open": float(row[1]), "high": float(row[2]),
                            "low": float(row[3]), "close": float(row[4]), "volume": float(row[5]), "is_closed": is_closed,
                        })
            finally:
                ping_task.cancel()

    async def _blofin_ping_loop(self, ws: Any) -> None:
        """OKX-family public WS convention: an application-level text
        "ping" every `ws_ping_interval` seconds, expecting a literal "pong"
        back -- NOT the WS-protocol ping/pong frame `websockets` already
        handles for Binance. This is the "automated ping/pong keep-alive"
        the spec calls out, for this venue specifically."""
        while True:
            await asyncio.sleep(self.ws_ping_interval)
            await ws.send("ping")

    async def _flush_pending_blofin(self, ws: Any, bar: str) -> None:
        if self._pending_subscribe:
            await ws.send(json.dumps({
                "op": "subscribe", "args": [{"channel": f"candle{bar}", "instId": ex._to_blofin_symbol(s)} for s in self._pending_subscribe],
            }))
            self._pending_subscribe.clear()
        if self._pending_unsubscribe:
            await ws.send(json.dumps({
                "op": "unsubscribe", "args": [{"channel": f"candle{bar}", "instId": ex._to_blofin_symbol(s)} for s in self._pending_unsubscribe],
            }))
            self._pending_unsubscribe.clear()


# ---------------------------------------------------------------------------
# 2. Dynamic universe scanner
# ---------------------------------------------------------------------------

_LEVERAGED_TOKEN_MARKERS = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "4L", "4S", "5L", "5S")
_STABLECOIN_BASES = frozenset({"USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDE", "PYUSD", "USDP", "GUSD", "EURC", "USD1"})


@dataclass
class UniverseFilterConfig:
    """Spec point 2's filter/rank thresholds.

    Args:
        min_quote_volume_usd: 24h quote volume floor (default $50,000,000).
        max_spread_pct: Max bid-ask spread, as a percent of mid (default
            0.05). A ticker with no bid/ask data (this repo's own
            exchanges.py `load_universe`/`load_universe_blofin` don't
            surface it today -- see `filter_and_rank_universe`) passes this
            check automatically rather than being silently excluded.
        top_n: Max qualifying symbols kept after ranking by volume (default
            20, spec's "Top 15-20" upper bound).
        quote_suffix: Quote currency suffix identifying a perpetual's base
            asset (default "USDT").
    """

    min_quote_volume_usd: float = 50_000_000.0
    max_spread_pct: float = 0.05
    top_n: int = 20
    quote_suffix: str = "USDT"


def _base_asset(symbol: str, quote_suffix: str) -> str:
    symbol_u = symbol.upper()
    return symbol_u[: -len(quote_suffix)] if symbol_u.endswith(quote_suffix) else symbol_u


def is_excluded_symbol(symbol: str, *, quote_suffix: str = "USDT") -> bool:
    """True for a stablecoin-vs-stablecoin pair or a leveraged token
    (3L/3S/UP/DOWN/BULL/BEAR-style base asset) -- spec point 2's exclusion rule."""
    base = _base_asset(symbol, quote_suffix)
    if base in _STABLECOIN_BASES:
        return True
    return any(base.endswith(marker) for marker in _LEVERAGED_TOKEN_MARKERS)


def spread_pct(bid: float | None, ask: float | None) -> float | None:
    """Bid-ask spread as a percent of the mid price, or None if bid/ask
    aren't both usable (missing, non-positive, or crossed)."""
    if bid is None or ask is None:
        return None
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 100.0


def filter_and_rank_universe(tickers: list[dict], config: UniverseFilterConfig | None = None) -> list[dict]:
    """Pure filter/rank step (spec point 2). Each input ticker dict needs
    at least {"symbol", "quote_vol"}; "bid"/"ask" are optional (see
    `UniverseFilterConfig.max_spread_pct`'s docstring for why their absence
    doesn't exclude a symbol). Returns the top `config.top_n` survivors,
    ranked by quote_vol descending.
    """
    cfg = config or UniverseFilterConfig()
    out = []
    for t in tickers:
        symbol = str(t.get("symbol") or "").upper()
        if not symbol.endswith(cfg.quote_suffix):
            continue
        if is_excluded_symbol(symbol, quote_suffix=cfg.quote_suffix):
            continue
        qv = float(t.get("quote_vol") or 0.0)
        if qv < cfg.min_quote_volume_usd:
            continue
        sp = spread_pct(t.get("bid"), t.get("ask"))
        if sp is not None and sp > cfg.max_spread_pct:
            continue
        out.append({**t, "symbol": symbol, "quote_vol": qv, "spread_pct": sp})
    out.sort(key=lambda t: t["quote_vol"], reverse=True)
    return out[: cfg.top_n]


class UniverseScanner:
    """Background periodic universe scan (spec point 2). Fetches tickers,
    filters/ranks them, and diffs the result against the previously
    qualifying set, firing `on_symbol_qualified`/`on_symbol_disqualified`
    for exactly the symbols whose membership changed -- what "subscribe"/
    "unsubscribe" actually DOES (registering a KrypticEngine, tearing one
    down once flat, ...) is entirely up to the caller; this class only
    tracks set membership and the scan cadence.

    Args:
        fetch_tickers: `async () -> list[dict]`, each dict at least
            {"symbol", "quote_vol"} (see `filter_and_rank_universe`).
        config: `UniverseFilterConfig` (defaults applied if omitted).
        scan_interval_seconds: Cadence for `run_forever` (default 4 hours,
            per spec).
        on_symbol_qualified / on_symbol_disqualified: `async (symbol) ->
            None` hooks, called once per newly-added/newly-dropped symbol.
    """

    def __init__(
        self,
        *,
        fetch_tickers: Callable[[], Awaitable[list[dict]]],
        config: UniverseFilterConfig | None = None,
        scan_interval_seconds: float = 4 * 3600.0,
        on_symbol_qualified: Callable[[str], Awaitable[None]] | None = None,
        on_symbol_disqualified: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.fetch_tickers = fetch_tickers
        self.config = config or UniverseFilterConfig()
        self.scan_interval_seconds = scan_interval_seconds
        self.on_symbol_qualified = on_symbol_qualified
        self.on_symbol_disqualified = on_symbol_disqualified
        self.current_universe: list[dict] = []
        self.qualifying_symbols: set[str] = set()
        self.last_scan_monotonic: float | None = None
        self.scan_count = 0

    async def scan_once(self) -> tuple[set[str], set[str]]:
        """Run one scan cycle. Returns (added, removed). A `fetch_tickers`
        failure is logged and treated as "no change this cycle" (keeps the
        previous universe rather than tearing every symbol down on a
        transient REST hiccup)."""
        try:
            tickers = await self.fetch_tickers()
        except Exception as e:
            log.warning("UniverseScanner: fetch_tickers failed, keeping previous universe: %s", e)
            return set(), set()

        ranked = filter_and_rank_universe(tickers, self.config)
        new_symbols = {t["symbol"] for t in ranked}
        added = new_symbols - self.qualifying_symbols
        removed = self.qualifying_symbols - new_symbols
        self.current_universe = ranked
        self.qualifying_symbols = new_symbols
        self.last_scan_monotonic = time.monotonic()
        self.scan_count += 1
        log.info(
            "UniverseScanner: scan #%d -- %d qualifying symbol(s) (+%d/-%d): %s",
            self.scan_count, len(new_symbols), len(added), len(removed), sorted(new_symbols),
        )
        for symbol in sorted(added):
            if self.on_symbol_qualified is not None:
                await self.on_symbol_qualified(symbol)
        for symbol in sorted(removed):
            if self.on_symbol_disqualified is not None:
                await self.on_symbol_disqualified(symbol)
        return added, removed

    async def run_forever(self) -> None:
        """Run until cancelled: scan, sleep `scan_interval_seconds`, repeat."""
        while True:
            await self.scan_once()
            await asyncio.sleep(self.scan_interval_seconds)


# ---------------------------------------------------------------------------
# 3. Paper trading (dry-run) engine
# ---------------------------------------------------------------------------

# Entry 1 fill types that are immediate MARKET fills (see entry_ladder.py's
# `entry1_prefilled`); every other ENTRY_FILL is a resting LIMIT order.
_MARKET_ENTRY_TYPES = frozenset({"market_breakout", "market_fallback_no_fvg"})
# Exit execution_log kinds that are market-triggered (stop-market / a
# dynamic close-on-signal exit); TP_HIT is a resting take-profit limit order.
# RUNNER_EXIT: an ATR chandelier-trail stop firing (Step 11.0's runner
# tier) -- a stop-market exit, same as SL_HIT.
_MARKET_EXIT_KINDS = frozenset({"SL_HIT", "RUNNER_EXIT", "TIME_DECAY_EXIT"})


@dataclass
class FeeSlippageModel:
    """Spec point 3's realistic cost model.

    Args:
        maker_fee_pct: Fee on a resting limit fill (default 0.02%) --
            every entry tier fill and every TP hit.
        taker_fee_pct: Fee on a market-triggered fill (default 0.05%) --
            the immediate breakout/fallback Entry 1 fill, every SL hit, and
            the dynamic TP5 runner exit.
        market_slippage_pct: Adverse price move applied to a market-
            triggered fill only (default 0.02%) -- resting limit fills
            (every other entry tier, every TP hit) execute at their exact
            level; nothing here models slippage on those.

    Fees/slippage are expressed as a fraction of `price * weight`, the same
    price-weighted-fraction unit family risk_manager.py's own
    `realized_pnl` already uses (this codebase has no real notional/
    leverage model to convert to a dollar amount -- see
    multi_strategy_manager.py's `_default_margin_estimator` for the same
    caveat).
    """

    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.05
    market_slippage_pct: float = 0.02

    def fee_amount(self, price: float, weight: float, *, is_market: bool) -> float:
        rate = self.taker_fee_pct if is_market else self.maker_fee_pct
        return price * weight * rate / 100.0

    def slipped_price(self, direction: str, price: float, *, is_exit: bool) -> float:
        """The market-fill price after slippage, always moved AGAINST the
        position: a LONG's market exit (or a SHORT's market entry) fills
        lower; a SHORT's market exit (or a LONG's market entry) fills higher."""
        slip = price * self.market_slippage_pct / 100.0
        worse_is_lower = (direction == "LONG") if is_exit else (direction == "SHORT")
        return price - slip if worse_is_lower else price + slip


@dataclass
class PaperFill:
    kind: str            # "ENTRY" | "EXIT"
    event_type: str       # the underlying execution_log "type", or "ENTRY_FILL_AT_OPEN" for the immediate tier-0 fill
    tier_index: int | None
    raw_price: float
    fill_price: float
    weight: float
    is_market: bool
    fee_amount: float
    slippage_cost: float
    bar_index: int


@dataclass
class PaperTradeRecord:
    """A fully-closed paper trade, fee/slippage-adjusted (spec point 4's
    "realized paper PnL (R-multiple and %), and duration")."""

    strategy_id: str
    symbol: str
    direction: str
    opened_at_ts: int | None
    closed_at_ts: int | None
    duration_seconds: float | None
    filled_weight: float
    avg_entry: float | None
    idealized_pnl: float          # PositionState.realized_pnl -- no fees/slippage, for comparison
    net_pnl: float                 # fee/slippage-adjusted
    total_fees: float
    total_slippage_cost: float
    r_multiple: float | None       # net_pnl / (|avg_entry - initial_sl| * filled_weight)
    pct_return: float | None       # net_pnl / (avg_entry * filled_weight) * 100
    breakeven_moved: bool = False  # True if PositionState.breakeven_moved had already fired before this trade closed
    fills: list[dict] = field(default_factory=list)
    # Institutional Excel/CSV report fields (setup diagnostics, planned
    # risk/targets, and the migrated stop) -- None/() wherever the trade
    # was never tracked through on_trade_opened's diagnostics param.
    initial_sl: float | None = None
    expected_vwap: float | None = None
    tp_levels: tuple = ()
    be_price: float | None = None      # migrated breakeven stop level, or None if it never moved
    ker_ratio: float | None = None
    rvol: float | None = None
    htf_aligned: bool | None = None
    risk_atr_multiple: float | None = None
    bars_held: int | None = None       # count of PositionState.update() calls from entry to close


@dataclass
class _OpenPaperTrade:
    strategy_id: str
    symbol: str
    direction: str
    initial_sl: float
    opened_at_ts: int | None
    fills: list[PaperFill] = field(default_factory=list)
    filled_weight: float = 0.0
    entry_notional: float = 0.0
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage_cost: float = 0.0
    # Setup-time diagnostics (Institutional Excel/CSV report, spec point 1/2)
    # -- captured once, at on_trade_opened(), from TradeLifecycleManager.
    # open_trade()'s own diagnostics dict; None/() for any field that
    # diagnostics didn't carry (e.g. an untracked position injected
    # directly, bypassing on_trade_opened).
    expected_vwap: float | None = None       # ladder's full theoretical VWAP ("entry_avg_intended")
    tp_levels: tuple = ()                    # (tp1, tp2, tp3, tp4) as priced at construction
    ker_ratio: float | None = None           # RegimeFilter chop gate's Kaufman Efficiency Ratio reading
    rvol: float | None = None                # RegimeFilter liquidity gate's Relative Volume reading
    htf_aligned: bool | None = None          # DirectionEngine's HTF structure condition (always True for
                                              # any trade that opened -- confluence requires all 4 to pass)
    risk_atr_multiple: float | None = None   # planned |expected_vwap - initial_sl| / ATR14 at construction

    @property
    def avg_entry(self) -> float | None:
        return self.entry_notional / self.filled_weight if self.filled_weight > 0 else None

    def add_entry(self, fill: PaperFill) -> None:
        self.fills.append(fill)
        self.filled_weight += fill.weight
        self.entry_notional += fill.fill_price * fill.weight
        self.total_fees += fill.fee_amount
        self.total_slippage_cost += fill.slippage_cost

    def add_exit(self, fill: PaperFill) -> None:
        self.fills.append(fill)
        avg = self.avg_entry
        if avg is not None:
            sign = 1.0 if self.direction == "LONG" else -1.0
            self.realized_pnl += (fill.fill_price - avg) * fill.weight * sign
        self.total_fees += fill.fee_amount
        self.total_slippage_cost += fill.slippage_cost


def _extract_setup_diagnostics(diagnostics: dict | None, direction: str) -> dict:
    """Pull the institutional report's setup-diagnostics fields out of
    `TradeLifecycleManager.open_trade()`'s own diagnostics dict (see
    risk_manager.py/entry_ladder.py's docstrings for its exact shape) --
    {} if diagnostics wasn't supplied, so `_OpenPaperTrade(**{})` still
    just falls back to every field's None/() default.

    `htf_aligned` reads DirectionEngine's htf_structure condition for
    whichever side actually opened -- always True for any trade that got
    this far, since a non-NEUTRAL bias requires all 4 confluence
    conditions (including htf_structure) to have passed. Still worth
    reporting explicitly rather than assuming a reader knows that."""
    if not diagnostics:
        return {}
    ladder_diag = diagnostics.get("ladder_diagnostics") or {}
    regime = ladder_diag.get("regime") or {}
    gates = regime.get("gates") or {}
    conditions = (ladder_diag.get("bias") or {}).get("conditions") or {}
    htf = conditions.get("htf_structure") or {}
    htf_aligned = htf.get("passed_long") if direction == "LONG" else htf.get("passed_short")
    tp_levels = diagnostics.get("tp_levels") or []
    return {
        "expected_vwap": diagnostics.get("expected_vwap"),
        "tp_levels": tuple(tp_levels[:4]),
        "ker_ratio": (gates.get("chop") or {}).get("value"),
        "rvol": (gates.get("liquidity_volume") or {}).get("value"),
        "htf_aligned": htf_aligned,
        "risk_atr_multiple": diagnostics.get("risk_atr_multiple"),
    }


def _dict_to_open_trade(d: dict) -> _OpenPaperTrade:
    d = dict(d)
    fills = [PaperFill(**f) for f in d.pop("fills", [])]
    trade = _OpenPaperTrade(**d)
    trade.fills = fills
    return trade


class DryRunHarness:
    """Paper-trading execution overlay (spec point 3). Reads the SAME
    `execution_log` events risk_manager.py's `PositionState` already
    produces -- whether/when a tier or TP/SL triggers is decided entirely
    by that existing, already-tested state machine; this class never
    touches it, only reads its events and recomputes a PARALLEL,
    fee/slippage-adjusted paper ledger from them (see `FeeSlippageModel`
    for which events are priced as maker/limit vs. taker/market+slippage).

    Persists atomically via `resilience_manager.StateStore` (spec point
    3's "Persist paper positions atomically ... so a crash during dry-run
    tests recovery identically") -- both completed trades AND any
    still-open paper position's accumulated fills/fees, so a crash
    mid-trade recovers the exact same partial ledger, not just the
    completed-trade history.
    """

    def __init__(
        self,
        *,
        state_dir: str | Path = ".",
        fees: FeeSlippageModel | None = None,
        ledger_filename: str = "paper_ledger.json",
    ) -> None:
        self.fees = fees or FeeSlippageModel()
        self.ledger_store = StateStore(Path(state_dir) / ledger_filename)
        self.closed_trades: list[dict] = []
        self._open: dict[tuple[str, str, str], _OpenPaperTrade] = {}
        # In-memory funnel counters for the institutional report's "Signals
        # generated vs. Filled trades vs. Expired/Cancelled setups" summary
        # -- not persisted (informational for the life of this process only,
        # same convention as everything else run-scoped in BacktestEngine).
        self.trades_opened_count = 0
        self.trades_expired_unfilled_count = 0
        self._load_ledger()

    def _load_ledger(self) -> None:
        payload = self.ledger_store.load_raw()
        if payload is None:
            return
        extra = payload.get("extra") or {}
        self.closed_trades = list(extra.get("closed_trades") or [])
        for key_str, trade_dict in (extra.get("open_trades") or {}).items():
            try:
                strategy_id, symbol, direction = key_str.split("|")
                self._open[(strategy_id, symbol, direction)] = _dict_to_open_trade(trade_dict)
            except (TypeError, ValueError) as e:
                log.error("paper ledger: failed to reconstruct open trade %r, dropping: %s", key_str, e)

    def _persist_ledger(self) -> None:
        open_payload = {"|".join(key): dataclasses.asdict(trade) for key, trade in self._open.items()}
        self.ledger_store.save(position=None, symbol="PAPER_LEDGER", extra={"closed_trades": self.closed_trades, "open_trades": open_payload})

    @staticmethod
    def _key(strategy_id: str, symbol: str, direction: str) -> tuple[str, str, str]:
        return (strategy_id.upper(), symbol.upper(), direction.upper())

    def on_trade_opened(self, *, strategy_id: str, symbol: str, position: PositionState, bar: dict, diagnostics: dict | None = None) -> None:
        """Call right after a new position opens. Handles the immediate
        market-style Entry 1 fill (`ladder.fills[0] is True` at
        construction), which never produces its own ENTRY_FILL
        execution_log event -- see entry_ladder.py's `entry1_prefilled`.

        `diagnostics`: optional `TradeLifecycleManager.open_trade()` return
        value (backtest_engine.py's `simulate_symbol` passes this through;
        the live path leaves it None, same as ManagedStrategyEngine never
        having captured it either) -- setup-time context for the
        institutional Excel/CSV report's diagnostics/risk/target columns.
        """
        key = self._key(strategy_id, symbol, position.direction)
        setup = _extract_setup_diagnostics(diagnostics, position.direction)
        trade = _OpenPaperTrade(
            strategy_id=strategy_id.upper(), symbol=symbol.upper(), direction=position.direction,
            initial_sl=position.initial_sl, opened_at_ts=int(bar["ts"]),
            **setup,
        )
        self._open[key] = trade
        self.trades_opened_count += 1
        if position.ladder.fills[0]:
            entry_type = position.ladder.entry_types[0] if position.ladder.entry_types else ""
            is_market = entry_type in _MARKET_ENTRY_TYPES
            raw_price = position.ladder.levels[0]
            weight = position.ladder.weights[0]
            fill_price = self.fees.slipped_price(position.direction, raw_price, is_exit=False) if is_market else raw_price
            fee = self.fees.fee_amount(fill_price, weight, is_market=is_market)
            trade.add_entry(PaperFill(
                kind="ENTRY", event_type="ENTRY_FILL_AT_OPEN", tier_index=0, raw_price=raw_price,
                fill_price=fill_price, weight=weight, is_market=is_market, fee_amount=fee,
                slippage_cost=abs(fill_price - raw_price) * weight, bar_index=0,
            ))
        self._persist_ledger()

    def on_bar_events(self, *, strategy_id: str, symbol: str, position: PositionState, events: list[dict]) -> None:
        """Call with one bar's newly-appended `execution_log` entries
        (`DispatchResult.position_events`)."""
        key = self._key(strategy_id, symbol, position.direction)
        trade = self._open.get(key)
        if trade is None:
            return
        for event in events:
            kind = event["type"]
            if kind == "ENTRY_FILL":
                tier_i = event["tier_index"]
                entry_type = position.ladder.entry_types[tier_i] if tier_i < len(position.ladder.entry_types) else ""
                is_market = entry_type in _MARKET_ENTRY_TYPES
                raw_price, weight = event["price"], event["weight"]
                fill_price = self.fees.slipped_price(position.direction, raw_price, is_exit=False) if is_market else raw_price
                fee = self.fees.fee_amount(fill_price, weight, is_market=is_market)
                trade.add_entry(PaperFill(
                    kind="ENTRY", event_type=kind, tier_index=tier_i, raw_price=raw_price, fill_price=fill_price,
                    weight=weight, is_market=is_market, fee_amount=fee, slippage_cost=abs(fill_price - raw_price) * weight,
                    bar_index=event["bar_index"],
                ))
            elif kind in ("TP_HIT", "SL_HIT", "RUNNER_EXIT", "TIME_DECAY_EXIT"):
                is_market = kind in _MARKET_EXIT_KINDS
                raw_price, weight = event["price"], event["closed_weight"]
                fill_price = self.fees.slipped_price(position.direction, raw_price, is_exit=True) if is_market else raw_price
                fee = self.fees.fee_amount(fill_price, weight, is_market=is_market)
                trade.add_exit(PaperFill(
                    kind="EXIT", event_type=kind, tier_index=event.get("tier_index"), raw_price=raw_price, fill_price=fill_price,
                    weight=weight, is_market=is_market, fee_amount=fee, slippage_cost=abs(fill_price - raw_price) * weight,
                    bar_index=event["bar_index"],
                ))
        self._persist_ledger()

    def on_position_closed(self, *, strategy_id: str, symbol: str, position: PositionState, closed_bar: dict) -> PaperTradeRecord | None:
        """Call once the position has fully closed. Returns the finished
        `PaperTradeRecord` (also appended to `self.closed_trades` and
        persisted), or None if this position was never tracked (e.g. it
        was injected directly, bypassing `on_trade_opened` -- same
        graceful-skip convention `ManagedStrategyEngine` already uses for
        an untracked position's `_entry_bar_ts`), or if it closed having
        never filled a single entry tier (see the `filled_weight <= 0`
        check below)."""
        key = self._key(strategy_id, symbol, position.direction)
        trade = self._open.pop(key, None)
        if trade is None:
            return None

        if trade.filled_weight <= 0:
            # TP2's cancel_unfilled_orders() fires whenever price reaches
            # the TP2 level, whether or not any entry tier has ever
            # filled -- so a setup that never pulls back into the ladder
            # before price runs straight to (or past) that level ends up
            # with every entry cancelled and `closed=True`, despite zero
            # size ever having opened. That's an expired/invalidated
            # setup, not a trade: no risk was ever taken and no PnL was
            # ever possible. Recording it would count a zero-risk,
            # zero-PnL no-op as a "breakeven" in win/loss/breakeven stats
            # and inflate total_trades -- so drop it here, same as the
            # untracked-position skip above.
            self.trades_expired_unfilled_count += 1
            self._persist_ledger()
            return None

        avg_entry = trade.avg_entry
        net_pnl = trade.realized_pnl - trade.total_fees
        opened_ts, closed_ts = trade.opened_at_ts, int(closed_bar["ts"])
        duration_seconds = (closed_ts - opened_ts) / 1000.0 if opened_ts is not None else None
        risk_per_unit = abs(avg_entry - trade.initial_sl) if avg_entry is not None else None
        risk_amount = risk_per_unit * trade.filled_weight if risk_per_unit else None
        r_multiple = net_pnl / risk_amount if risk_amount else None
        pct_return = (net_pnl / (avg_entry * trade.filled_weight) * 100.0) if avg_entry and trade.filled_weight else None

        exit_bar_indices = [f.bar_index for f in trade.fills if f.kind == "EXIT"]
        bars_held = (max(exit_bar_indices) + 1) if exit_bar_indices else None
        be_price = position.current_sl if position.breakeven_moved else None

        record = PaperTradeRecord(
            strategy_id=trade.strategy_id, symbol=trade.symbol, direction=trade.direction,
            opened_at_ts=opened_ts, closed_at_ts=closed_ts, duration_seconds=duration_seconds,
            filled_weight=trade.filled_weight, avg_entry=avg_entry, idealized_pnl=position.realized_pnl,
            net_pnl=net_pnl, total_fees=trade.total_fees, total_slippage_cost=trade.total_slippage_cost,
            r_multiple=r_multiple, pct_return=pct_return, breakeven_moved=position.breakeven_moved,
            fills=[dataclasses.asdict(f) for f in trade.fills],
            initial_sl=trade.initial_sl, expected_vwap=trade.expected_vwap, tp_levels=trade.tp_levels,
            be_price=be_price, ker_ratio=trade.ker_ratio, rvol=trade.rvol, htf_aligned=trade.htf_aligned,
            risk_atr_multiple=trade.risk_atr_multiple, bars_held=bars_held,
        )
        self.closed_trades.append(dataclasses.asdict(record))
        self._persist_ledger()
        return record


class PaperTradingObserver:
    """Wires a `DryRunHarness` to a `ManagedStrategyEngine`'s per-bar
    output: call `observe(managed, bar, result)` right after
    `managed.on_bar_close(bar)` (exactly what `LiveRunner.handle_bar` does)."""

    def __init__(self, harness: DryRunHarness) -> None:
        self.harness = harness

    def observe(self, managed: ManagedStrategyEngine, bar: dict, result: Any) -> PaperTradeRecord | None:
        if result.trade_opened:
            position = managed.active_position
            if position is not None:
                self.harness.on_trade_opened(strategy_id=managed.strategy_id, symbol=managed.symbol, position=position, bar=bar)

        if result.position_events:
            # A position that closed THIS bar is already off `active_position`
            # by the time on_bar_close returns -- read it back from the
            # engine's own closed-positions history instead so these events
            # are still attributed correctly.
            position = managed.active_position
            if position is None and managed.engine.closed_positions:
                position = managed.engine.closed_positions[-1]
            if position is not None:
                self.harness.on_bar_events(strategy_id=managed.strategy_id, symbol=managed.symbol, position=position, events=result.position_events)

        record = None
        if result.position_closed_this_bar:
            position = managed.engine.closed_positions[-1] if managed.engine.closed_positions else None
            if position is not None:
                record = self.harness.on_position_closed(strategy_id=managed.strategy_id, symbol=managed.symbol, position=position, closed_bar=bar)
        return record


# ---------------------------------------------------------------------------
# 4. Telemetry & console dashboard
# ---------------------------------------------------------------------------

def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = abs(seconds)
    hours, rem = divmod(int(seconds), 3600)
    minutes = rem // 60
    return f"{hours}h{minutes:02d}m"


def _short_detail(detail: dict) -> str:
    parts = []
    for k, v in detail.items():
        if isinstance(v, float):
            v = f"{v:.6g}"
        parts.append(f"{k}={v}")
    return ", ".join(parts)


class ConsoleDashboard:
    """Clean console/log summary (spec point 4). Doesn't own any state --
    call the `render_*` methods as events occur (`LiveRunner` does this
    automatically); `stream` is injectable (tests pass a list-appending
    callable instead of `print`)."""

    _KIND_LABELS = {
        "SETUP_APPROVED": "SETUP", "SETUP_REJECTED": "REJECTED", "TIER_FILL": "FILL",
        "VWAP_UPDATE": "VWAP", "BREAKEVEN_MOVE": "BREAKEVEN", "GHOST_ORDER_CANCEL": "PURGE",
        "RUNNER_EXIT": "RUNNER", "TIME_DECAY_EXIT": "TIME_DECAY", "POSITION_CLOSED": "CLOSED",
    }

    def __init__(self, *, stream: Callable[[str], None] | None = None) -> None:
        self._stream = stream

    def _print(self, line: str) -> None:
        if self._stream is not None:
            self._stream(line)
        else:
            print(line)

    def render_universe(self, symbols: list[str]) -> None:
        self._print(f"[UNIVERSE] {len(symbols)} active symbol(s): {', '.join(symbols) or '(none)'}")

    def render_setup_diagnostics(self, strategy_id: str, symbol: str, diagnostics: dict) -> None:
        reason = diagnostics.get("reason")
        if reason:
            self._print(f"[{strategy_id} {symbol}] setup check -- {reason}")

    def render_event(self, event: Any) -> None:
        label = self._KIND_LABELS.get(event.kind, event.kind)
        side = f" {event.pos_side}" if event.pos_side else ""
        self._print(f"[{event.strategy_id} {event.symbol}{side}] {label}: {_short_detail(event.detail)}")

    def render_closed_trade(self, record: PaperTradeRecord) -> None:
        r_str = f"{record.r_multiple:+.2f}R" if record.r_multiple is not None else "n/a"
        pct_str = f"{record.pct_return:+.2f}%" if record.pct_return is not None else "n/a"
        self._print(
            f"[{record.strategy_id} {record.symbol}] CLOSED {record.direction} -- "
            f"net_pnl={record.net_pnl:+.6g} ({pct_str}, {r_str}) fees={record.total_fees:.6g} "
            f"slippage={record.total_slippage_cost:.6g} duration={_format_duration(record.duration_seconds)}"
        )


# ---------------------------------------------------------------------------
# 5. Top-level runner
# ---------------------------------------------------------------------------

class LiveRunner:
    """Orchestrates the universe scanner, MultiStrategyManager, and (when
    `dry_run`) the DryRunHarness + console dashboard into one runnable
    paper-trading loop.

    Args:
        exchange: An `ExchangeStream` (a real `LiveExchangeStream` or, for
            tests, `data_collector.MockExchangeStream`).
        arbiter: Shared `RiskArbiter`.
        fetch_tickers: `async () -> list[dict]` for `UniverseScanner`.
        trade_manager_factory: `() -> TradeLifecycleManager`, called once
            per newly-qualified symbol (default: `TradeLifecycleManager`
            with its own defaults). Each symbol gets its OWN instance --
            sharing one across symbols would be safe today (it's stateless
            per call) but a distinct instance keeps that true even if it
            ever grows per-symbol state.
        strategy_name: `KrypticEngine.strategy_name` for every registered symbol.
        state_dir: Directory for every engine's scoped state file, plus the
            dry-run ledger.
        dry_run: Enables `DryRunHarness` + console trade-close reporting
            (default `DRY_RUN`, this module's env-driven flag).
        universe_config / scan_interval_seconds: Forwarded to `UniverseScanner`.
        ltf_interval: Execution timeframe (default "15m", per spec).
        engine_kwargs: Extra kwargs forwarded to every `KrypticEngine`.
        console: Injectable `ConsoleDashboard`.
    """

    def __init__(
        self,
        *,
        exchange: Any,
        arbiter: RiskArbiter,
        fetch_tickers: Callable[[], Awaitable[list[dict]]],
        trade_manager_factory: Callable[[], TradeLifecycleManager] | None = None,
        strategy_name: str = "KRYPTIC",
        state_dir: str | Path = ".",
        dry_run: bool = DRY_RUN,
        universe_config: UniverseFilterConfig | None = None,
        scan_interval_seconds: float = 4 * 3600.0,
        ltf_interval: str = "15m",
        engine_kwargs: dict | None = None,
        console: ConsoleDashboard | None = None,
    ) -> None:
        self.exchange = exchange
        self.strategy_name = strategy_name
        self.state_dir = Path(state_dir)
        self.ltf_interval = ltf_interval
        self.engine_kwargs = engine_kwargs or {}
        self.trade_manager_factory = trade_manager_factory or TradeLifecycleManager
        self.dry_run = dry_run
        self.console = console or ConsoleDashboard()

        self.manager = MultiStrategyManager(exchange=exchange, arbiter=arbiter, state_dir=state_dir, ltf_interval=ltf_interval)
        self.harness = DryRunHarness(state_dir=state_dir) if dry_run else None
        self.observer = PaperTradingObserver(self.harness) if self.harness is not None else None
        self.scanner = UniverseScanner(
            fetch_tickers=fetch_tickers, config=universe_config, scan_interval_seconds=scan_interval_seconds,
            on_symbol_qualified=self._on_symbol_qualified, on_symbol_disqualified=self._on_symbol_disqualified,
        )
        self._managed: dict[str, ManagedStrategyEngine] = {}
        self._pending_removal: set[str] = set()

    async def _on_symbol_qualified(self, symbol: str) -> None:
        symbol = symbol.upper()
        if symbol in self._managed:
            self._pending_removal.discard(symbol)  # re-qualified before its removal was swept -- keep it
            return
        engine = KrypticEngine(
            exchange=self.exchange, symbol=symbol, trade_manager=self.trade_manager_factory(),
            strategy_name=self.strategy_name, state_dir=self.state_dir, ltf_interval=self.ltf_interval,
            **self.engine_kwargs,
        )
        managed = self.manager.register_engine(engine)
        await self.manager.bus.warmup(symbol)
        await managed.warmup()
        self._managed[symbol] = managed
        if hasattr(self.exchange, "subscribe_symbols"):
            self.exchange.subscribe_symbols([symbol])
        self.console.render_universe(sorted(self._managed))
        log.info("[LiveRunner] registered %s for %s", self.strategy_name, symbol)

    async def _on_symbol_disqualified(self, symbol: str) -> None:
        self._pending_removal.add(symbol.upper())

    def sweep_pending_removals(self) -> list[str]:
        """Tear down any disqualified symbol whose position has reached a
        terminal state (spec point 2: "gracefully unsubscribe inactive
        symbols once their open PositionState reaches a terminal state") --
        never mid-trade. Safe to call as often as convenient (a no-op when
        nothing's pending or every pending symbol is still in a trade)."""
        removed = []
        for symbol in list(self._pending_removal):
            managed = self._managed.get(symbol)
            if managed is None:
                self._pending_removal.discard(symbol)
                continue
            if managed.active_position is not None:
                continue
            self.manager.bus.unregister(symbol, managed)
            del self._managed[symbol]
            self._pending_removal.discard(symbol)
            if hasattr(self.exchange, "unsubscribe_symbols"):
                self.exchange.unsubscribe_symbols([symbol])
            removed.append(symbol)
            self.console.render_universe(sorted(self._managed))
            log.info("[LiveRunner] unregistered %s for %s (disqualified, position flat)", self.strategy_name, symbol)
        return removed

    def handle_bar(self, symbol: str, bar: dict) -> Any | None:
        """Drive one closed candle through the harness for `symbol` -- the
        single ingestion point a real stream's callback (or a test) calls
        per closed candle. Returns the engine's DispatchResult, or None if
        `symbol` isn't currently registered (e.g. it was scanned out and
        already unsubscribed)."""
        managed = self._managed.get(symbol.upper())
        if managed is None:
            return None
        result = managed.on_bar_close(bar)
        if result.trade_diagnostics is not None:
            self.console.render_setup_diagnostics(managed.strategy_id, managed.symbol, result.trade_diagnostics)
        if self.observer is not None:
            record = self.observer.observe(managed, bar, result)
            if record is not None:
                self.console.render_closed_trade(record)
        self.sweep_pending_removals()
        return result

    def _on_raw_message(self, msg: dict) -> None:
        is_closed = msg.get("is_closed", msg.get("closed", False))
        if not is_closed:
            return
        symbol = msg.get("symbol")
        if not symbol:
            return
        bar = {k: msg[k] for k in _REQUIRED_CANDLE_KEYS}
        self.handle_bar(symbol, bar)

    async def start(self) -> tuple[asyncio.Task, asyncio.Task | None]:
        """Initial scan (populates the starting universe + engines), then
        launches the periodic rescan loop and the shared live stream as
        background tasks and returns them for the caller to await/cancel.
        Deliberately does not block forever itself, so shutdown stays
        entirely the caller's decision."""
        await self.scanner.scan_once()
        scan_task = asyncio.create_task(self.scanner.run_forever())
        stream_task = None
        if hasattr(self.exchange, "stream_klines"):
            stream_task = asyncio.create_task(
                self.exchange.stream_klines(list(self._managed), self.ltf_interval, self._on_raw_message)
            )
        return scan_task, stream_task


# ---------------------------------------------------------------------------
# Default ticker source + runnable entrypoint
# ---------------------------------------------------------------------------

async def default_fetch_tickers(venue: str | None = None) -> list[dict]:
    """Default `UniverseScanner` ticker source: reuses exchanges.py's own
    `load_universe`/`load_universe_blofin` (already used by the live bot),
    with no volume floor of its own -- `filter_and_rank_universe` applies
    the real one. Neither of exchanges.py's loaders surfaces bid/ask today,
    so the spread filter is a pass-through against this default source
    until that's added upstream in exchanges.py."""
    load_universe, _fetch_klines, _fetch_funding = ex.resolve_source(venue)
    async with httpx.AsyncClient(timeout=20) as client:
        universe = await load_universe(client, min_quote_vol=0.0)
    return universe["tickers"]


async def main() -> None:  # pragma: no cover -- real-network entrypoint, not exercised by tests (see module docstring)
    """Minimal runnable entrypoint, mirroring bot.py's own DATA_SOURCE/.env
    convention. NOT exercised against a real exchange anywhere in this
    repo's test suite -- this sandboxed environment has no outbound
    network route to any exchange. Smoke-test manually before trusting it live."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exchange = LiveExchangeStream()
    arbiter = RiskArbiter(account_equity=float(os.getenv("ACCOUNT_EQUITY", "10000")))
    runner = LiveRunner(
        exchange=exchange, arbiter=arbiter,
        fetch_tickers=lambda: default_fetch_tickers(exchange.venue),
        dry_run=DRY_RUN, state_dir=os.getenv("KRYPTIC_STATE_DIR", "."),
    )
    scan_task, stream_task = await runner.start()
    tasks = [t for t in (scan_task, stream_task) if t is not None]
    if tasks:
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
