"""backtest_engine.py -- vectorized/event-driven historical backtesting
engine for KRYPTIC (Step 10.5).

Simulates KRYPTIC across months of historical 15m candle data using the
EXACT, unmodified production decision classes (`RegimeFilter`,
`DirectionEngine`, `EntryLadderEngine`, `TradeLifecycleManager`,
`PositionState`) and the exact fee/slippage model `live_runner.py` already
uses live -- never a re-implementation of any of that logic. What this
module adds is purely in HOW OFTEN those (expensive) production functions
get called, and how the resulting trades get scored:

Why "vectorized" matters here (read this before changing the loop):
Driving `KrypticEngine`/`MarketDataPipeline` bar-by-bar exactly as
production does -- calling `RegimeFilter`/`DirectionEngine` fresh on every
single closed candle while flat -- was measured at ~140 seconds for one
symbol over just 90 days of 15m bars (8,640 bars), because
`DirectionEngine`/`RegimeFilter` recompute EMA/ADX/SuperTrend/KER/RVOL from
scratch, over whatever candle window they're given, on EVERY call. That's
fine live (bars arrive one every 15 real minutes); it does not scale to
replaying a year of history. `VectorizedSignalCache` below computes every
one of those indicators ONCE per symbol, vectorized over the FULL loaded
history (indicators.py's functions are all backward-looking/causal by
construction -- Step 1's zero-lookahead mandate -- so this is
mathematically what an ever-growing incremental computation would have
produced at each bar; there is no lookahead in reading `.iloc[i]` from a
series computed over the whole array). The event loop then uses that cache
to CHEAPLY decide, per bar, whether a setup is even numerically possible
(same thresholds as the 5 real RegimeFilter gates and the 4 real
DirectionEngine conditions, evaluated as boolean arrays) -- and calls the
REAL, unmodified `TradeLifecycleManager.open_trade()` (which internally
calls the real `EntryLadderEngine`/`RegimeFilter`/`DirectionEngine`, on an
UNBOUNDED "all history so far" slice, not a small window) ONLY on bars
that pass. Every actual trade decision is still made by production code,
verbatim; the cache only skips asking on bars that are guaranteed to fail.
`test_backtest_engine.py`'s cross-check test proves this produces IDENTICAL
trades to driving the real per-bar production path on the same data.

Two things worth knowing about how history is handled, one per stage:

- `VectorizedSignalCache`'s own indicator series (the CHEAP pre-filter) are
  computed once over the FULL available history, not a capped window. For
  a backward-looking, exponential-decay indicator (EMA/Wilder ADX/
  SuperTrend, all length <= 50 bars here), that's indistinguishable from a
  1000-candle rolling window once past the indicator's own warmup -- the
  cap exists purely for live 24/7 memory management, not as a considered
  part of the signal design, and computing over whatever history is
  actually available is, if anything, the more principled reference here.
- The ACTUAL, authoritative call to the real `open_trade()` on a candidate
  bar, by contrast, is deliberately handed ONLY the trailing
  `buffer_capacity` (default 1000) bars -- exactly matching production's
  own `CandleBuffer` capacity, not an approximation of it. This isn't
  optional for correctness so much as for speed: without it, a candidate
  late in a long backtest would hand `open_trade()` the ENTIRE history so
  far, an ever-growing cost a live deployment (which never buffers more
  than 1000 candles either) would never actually pay.

Reuses (never re-implements):
  - indicators.py for every underlying series.
  - regime_filter.RegimeFilter / directional_bias.DirectionEngine /
    entry_ladder.EntryLadderEngine / risk_manager.{PositionState,
    TradeLifecycleManager} for every actual trading decision.
  - live_runner.FeeSlippageModel / DryRunHarness for fee/slippage-adjusted
    fills and R-multiple/duration accounting -- the exact 0.05% taker /
    0.02% maker / 0.02% slippage model live paper-trading already uses.

Honest scope note (same convention as live_runner.py and backtest.py):
this sandboxed environment has no outbound route to any exchange, so
`HistoricalDataLoader`'s default paginated Binance fetcher (mirroring
backtest.py's own already-proven `fetch_klines_range`) is unverified
against a live connection here. Every test in test_backtest_engine.py
drives the loader against an injected fake page-fetcher and the engine
against synthetic OHLCV -- no real network anywhere in this file's tests.

Universe scaling (`--all-coins`/`--top-n`, main() only): reuses
exchanges.py's own `load_universe` (the same source live_runner.py's
UniverseScanner already scans on a schedule for live trading) and
live_runner.py's `filter_and_rank_universe`/`UniverseFilterConfig` for the
volume floor/top-N/stablecoin-and-leveraged-token exclusion, rather than
introducing a second universe-discovery implementation or a new exchange
client dependency. This is a ONE-SHOT snapshot for a single backtest run,
not a recurring scan. `main()` is a standalone, manually-invoked batch
script -- it shares no process, thread, or exchange-client object with
`bot.py` (ZENITH/GEM) or `live_runner.py`'s live path, so a large-universe
run cannot starve either of CPU/threads by construction. The one real
shared resource if you run one of these while a live bot trades on the
SAME exchange API key/IP is REST rate-limit weight -- `--concurrency`
(default 4) keeps this file's own parallel fetch conservative for exactly
that reason; it cannot coordinate with a live process's own request
accounting, which lives entirely outside this file.

Portfolio-level concurrency (`--max-open-trades`, main() only): a POST-HOC
diagnostic (see portfolio_overlap.py's own module docstring for the full
model and its stated approximations), not a constraint enforced during
simulation -- `BacktestEngine`/`simulate_symbol` still simulate every
symbol independently and unconstrained, exactly as before. Re-architecting
this into a single time-synchronized, capacity-constrained event loop
across the whole universe is a materially larger, separate piece of work.
"""
from __future__ import annotations

import asyncio
import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

import httpx
import numpy as np
import pandas as pd

import exchanges as ex
import indicators as ind
from data_collector import interval_to_seconds
from directional_bias import DirectionEngine
from entry_ladder import LADDER_WEIGHTS
from live_runner import DryRunHarness, FeeSlippageModel, UniverseFilterConfig, filter_and_rank_universe
from portfolio_overlap import overlap_report
from regime_filter import RegimeFilter
from risk_manager import PositionState, TradeLifecycleManager

log = logging.getLogger(__name__)

_REQUIRED_CANDLE_KEYS = ("ts", "open", "high", "low", "close", "volume")


# ---------------------------------------------------------------------------
# 1. Historical data ingestion & caching
# ---------------------------------------------------------------------------

_BINANCE_FAPI = "https://fapi.binance.com"
_BINANCE_KLINE_INTERVAL = {"15m": "15m", "1h": "1h", "4h": "4h"}
_BINANCE_FUNDING_INTERVAL_MS = 8 * 3600 * 1000  # Binance perp funding prints every 8h


def _interval_to_ms(interval: str) -> int:
    return interval_to_seconds(interval) * 1000


async def default_fetch_page_binance(client: httpx.AsyncClient, symbol: str, interval: str, start_time_ms: int, limit: int) -> list[dict]:
    """One page of Binance USDT-M futures klines starting at/after
    `start_time_ms`, oldest-first. Same public, well-documented endpoint
    and pagination shape as this repo's own proven `backtest.py`
    (`fetch_klines_range`, used by the live ZENITH/GEM backtester) --
    duplicated here in miniature (one page, no retry loop -- retry is
    `HistoricalDataLoader`'s job) rather than imported, so importing this
    module never drags in bot.py's Telegram/.env side effects the way
    `import backtest` would."""
    r = await client.get(
        f"{_BINANCE_FAPI}/fapi/v1/klines",
        params={"symbol": symbol, "interval": _BINANCE_KLINE_INTERVAL.get(interval, interval), "startTime": start_time_ms, "limit": limit},
        headers=ex.UA, timeout=20,
    )
    r.raise_for_status()
    body = r.json()
    if not isinstance(body, list):
        raise RuntimeError(f"Binance klines request rejected: {body}")
    return ex._rows_from_binance(body)


_BINANCE_FUNDING_MAX_LIMIT = 1000  # unlike klines (max 1500), the funding-rate-history endpoint caps `limit` at 1000


async def default_fetch_funding_page_binance(client: httpx.AsyncClient, symbol: str, start_time_ms: int, limit: int) -> list[dict]:
    """One page of Binance's historical funding-rate endpoint, oldest-first.
    Each row: {"ts": <funding time, ms>, "rate": <percent, matching
    exchanges.fetch_funding's existing *100 convention>}."""
    r = await client.get(
        f"{_BINANCE_FAPI}/fapi/v1/fundingRate",
        params={"symbol": symbol, "startTime": start_time_ms, "limit": min(limit, _BINANCE_FUNDING_MAX_LIMIT)},
        headers=ex.UA, timeout=20,
    )
    r.raise_for_status()
    body = r.json()
    if not isinstance(body, list):
        # A non-list body (Binance's {"code":...,"msg":...} error envelope)
        # means the request itself was rejected -- surface that clearly
        # instead of crashing on `row["fundingTime"]` for every "row" being
        # one character of an error message's string keys.
        raise RuntimeError(f"Binance fundingRate request rejected: {body}")
    return [{"ts": int(row["fundingTime"]), "rate": float(row["fundingRate"]) * 100.0} for row in body]


class HistoricalDataLoader:
    """Downloads (paginated) or loads cached historical OHLCV -- and,
    optionally, funding rate history -- for a symbol, aligned to the
    exact `{"ts","open","high","low","close","volume"}` shape every other
    KRYPTIC module already expects.

    Only the traded symbol's LTF (15m) candles and BTCUSDT's own HTF (4h)
    benchmark candles are ever downloaded -- the symbol's OWN 1h structure
    timeframe is deliberately NOT fetched separately. Production
    (`data_collector.HTFAggregator`) derives it causally from the same 15m
    stream it's already trading on; downloading a separately-sourced 1h
    series risks its bar boundaries silently disagreeing with what the
    live aggregator would have produced from gaps/timing in the SAME 15m
    data, which is exactly the "align timestamps ... without lookahead
    bias" failure mode this loader exists to avoid. `VectorizedSignalCache`
    (below) derives the 1h structure the same causal way production does.

    Args:
        cache_dir: Where cached CSVs live (default "./.backtest_cache").
        fetch_page: `async (client, symbol, interval, start_time_ms, limit)
            -> list[dict]`, returning up to `limit` candles at/after
            `start_time_ms`, oldest-first. Default: paginated Binance
            (`default_fetch_page_binance`) -- unverified live in this
            sandbox, see module docstring.
        fetch_funding_page: Same shape for funding history (default
            `default_fetch_funding_page_binance`); pass None to disable
            funding entirely (RegimeFilter treats missing funding as
            pass-through, so this is a safe, supported default too).
        page_limit: Candles requested per page (default 1500, Binance's max).
        max_pages: Hard cap on pages per `load()` call (default 3000 --
            4.5M candles, far beyond any realistic multi-year 15m window;
            purely a runaway-loop safety net).
        request_delay_seconds: Sleep between pages (default 0.2, a
            conservative rate-limit courtesy delay).
        client: Injectable httpx.AsyncClient (tests inject a fake one;
            production can leave this None and let one be created lazily).
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path = "./.backtest_cache",
        fetch_page: Callable[[httpx.AsyncClient, str, str, int, int], Awaitable[list[dict]]] = default_fetch_page_binance,
        fetch_funding_page: Callable[[httpx.AsyncClient, str, int, int], Awaitable[list[dict]]] | None = default_fetch_funding_page_binance,
        page_limit: int = 1500,
        max_pages: int = 3000,
        request_delay_seconds: float = 0.2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.fetch_page = fetch_page
        self.fetch_funding_page = fetch_funding_page
        self.page_limit = page_limit
        self.max_pages = max_pages
        self.request_delay_seconds = request_delay_seconds
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20.0)
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- caching --------------------------------------------------------------

    def _cache_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}_{interval}.csv"

    def _funding_cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}_funding.csv"

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["ts"])
        try:
            return pd.read_csv(path)
        except Exception as e:
            log.error("cache file %s is corrupt or unreadable, ignoring: %s", path, e)
            return pd.DataFrame(columns=["ts"])

    @staticmethod
    def _save_csv_atomic(path: Path, df: pd.DataFrame) -> None:
        """Same tmp-file + os.replace atomic-write pattern
        `resilience_manager.StateStore` uses for position state -- a
        crash mid-write can only ever leave the .tmp file, never a
        half-written cache the next run would silently corrupt-load."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)

    # -- OHLCV ------------------------------------------------------------------

    async def load(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Ensure the cache covers `[start_ms, end_ms]` (fetching only the
        missing edges, never a full re-download), then return that slice,
        oldest-first, deduplicated and sorted."""
        cached = self._load_csv(self._cache_path(symbol, interval))
        covered_start = int(cached["ts"].min()) if len(cached) else None
        covered_end = int(cached["ts"].max()) if len(cached) else None

        new_rows: list[dict] = []
        if covered_start is None or start_ms < covered_start:
            new_rows += await self._fetch_range(symbol, interval, start_ms, (covered_start - 1) if covered_start is not None else end_ms)
        if covered_end is None or end_ms > covered_end:
            fetch_from = max(start_ms, covered_end + 1) if covered_end is not None else start_ms
            new_rows += await self._fetch_range(symbol, interval, fetch_from, end_ms)

        if new_rows:
            combined = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True) if len(cached) else pd.DataFrame(new_rows)
            combined = combined.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
            self._save_csv_atomic(self._cache_path(symbol, interval), combined)
        else:
            combined = cached

        if not len(combined):
            return combined
        mask = (combined["ts"] >= start_ms) & (combined["ts"] <= end_ms)
        return combined.loc[mask].reset_index(drop=True)

    async def _fetch_range(self, symbol: str, interval: str, range_start_ms: int, range_end_ms: int) -> list[dict]:
        if range_start_ms > range_end_ms:
            return []
        client = await self._get_client()
        step_ms = _interval_to_ms(interval)
        out: list[dict] = []
        cursor = range_start_ms
        for _ in range(self.max_pages):
            if cursor > range_end_ms:
                break
            page = await self.fetch_page(client, symbol, interval, cursor, self.page_limit)
            if not page:
                break
            out.extend(page)
            new_cursor = int(page[-1]["ts"]) + step_ms
            if new_cursor <= cursor:
                break  # a non-advancing page would loop forever -- treat as exhausted
            cursor = new_cursor
            if len(page) < self.page_limit:
                break  # fewer than a full page means we've reached the exchange's latest data
            await asyncio.sleep(self.request_delay_seconds)
        return [c for c in out if range_start_ms <= c["ts"] <= range_end_ms]

    # -- funding ------------------------------------------------------------------

    async def load_funding(self, symbol: str, start_ms: int, end_ms: int) -> pd.Series:
        """Sparse funding-rate history as a `pd.Series` (percent, indexed
        by funding-print timestamp, sorted ascending) -- pass to
        `VectorizedSignalCache`/`simulate_symbol`'s `funding_series`, which
        looks up the most recent print AT OR BEFORE each bar (never a
        future one). Returns an empty Series if `fetch_funding_page` is
        None or nothing is cached/fetchable -- RegimeFilter's crowd-
        sentiment gate treats a missing funding rate as pass-through, so
        an empty result degrades gracefully rather than blocking every trade.
        """
        if self.fetch_funding_page is None:
            return pd.Series(dtype=float)
        cached = self._load_csv(self._funding_cache_path(symbol))
        # Unlike load()'s OHLCV caching, this only ever extends the cache
        # FORWARD from whatever's already covered -- funding history is a
        # supplementary signal (RegimeFilter treats it as pass-through when
        # absent), so backfilling an older edge isn't worth the extra
        # pagination path this method would otherwise need.
        covered_end = int(cached["ts"].max()) if len(cached) else None

        new_rows: list[dict] = []
        client = await self._get_client()
        if covered_end is None or end_ms > covered_end:
            fetch_from = max(start_ms, covered_end + 1) if covered_end is not None else start_ms
            cursor = fetch_from
            for _ in range(self.max_pages):
                if cursor > end_ms:
                    break
                try:
                    page = await self.fetch_funding_page(client, symbol, cursor, self.page_limit)
                except Exception as e:
                    # Funding is a supplementary signal everywhere else in
                    # this codebase (RegimeFilter treats it as pass-through
                    # when missing) -- a fetch failure here should degrade
                    # to "no funding data for this stretch", not take down
                    # an otherwise-successful OHLCV backtest run.
                    log.warning("funding fetch failed for %s, continuing without further funding data: %s", symbol, e)
                    break
                if not page:
                    break
                new_rows.extend(page)
                new_cursor = int(page[-1]["ts"]) + _BINANCE_FUNDING_INTERVAL_MS
                if new_cursor <= cursor:
                    break
                cursor = new_cursor
                if len(page) < self.page_limit:
                    break
                await asyncio.sleep(self.request_delay_seconds)

        if new_rows:
            combined = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True) if len(cached) else pd.DataFrame(new_rows)
            combined = combined.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
            self._save_csv_atomic(self._funding_cache_path(symbol), combined)
        else:
            combined = cached

        if not len(combined) or "rate" not in combined.columns:
            return pd.Series(dtype=float)
        mask = (combined["ts"] >= start_ms) & (combined["ts"] <= end_ms)
        sliced = combined.loc[mask].sort_values("ts")
        return pd.Series(sliced["rate"].to_numpy(), index=sliced["ts"].to_numpy())


def funding_asof(funding_series: pd.Series | None, ts: int) -> float | None:
    """The most recent funding print AT OR BEFORE `ts` -- never a future
    one. None if `funding_series` is empty/None or `ts` precedes every
    print (matches RegimeFilter's own "no funding data -> pass-through"
    convention)."""
    if funding_series is None or not len(funding_series):
        return None
    idx = np.searchsorted(funding_series.index.to_numpy(), ts, side="right") - 1
    if idx < 0:
        return None
    return float(funding_series.to_numpy()[idx])


# ---------------------------------------------------------------------------
# 2. Vectorized indicator / signal cache
# ---------------------------------------------------------------------------

def _htf_bucket_frame(df: pd.DataFrame, htf_ms: int) -> pd.DataFrame:
    """Groups `df` into HTF buckets by UTC-calendar floor-division on
    timestamp -- the same bucketing rule `data_collector.HTFAggregator`
    uses -- but for the WHOLE series at once. Returned frame has one row
    per bucket (sorted ascending), columns {"_bucket","high","low","close","volume"}.
    A bucket may be incomplete (still "in progress" at the end of `df`);
    `_asof_htf_row_index` never lets a bar see its own current bucket, only
    ones strictly before it, exactly matching `HTFAggregator.ingest`'s
    "only return a bucket once the FIRST bar of the NEXT one arrives" rule.
    """
    bucket = df["ts"].to_numpy() // htf_ms
    tmp = df.assign(_bucket=bucket)
    agg = tmp.groupby("_bucket", sort=True).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
    return agg.reset_index()


def _asof_htf_row_index(ts: np.ndarray, htf_ms: int, htf_frame: pd.DataFrame) -> np.ndarray:
    """For each LTF timestamp, the row index into `htf_frame` of the last
    FULLY CLOSED HTF bucket as of that bar (the bucket containing `ts`
    itself is, by definition, still open) -- -1 if none exists yet."""
    visible_bucket_id = (ts // htf_ms) - 1
    bucket_arr = htf_frame["_bucket"].to_numpy()
    idx = np.searchsorted(bucket_arr, visible_bucket_id, side="right") - 1
    return idx


class VectorizedSignalCache:
    """Precomputes, ONCE per symbol, every series `DirectionEngine`'s 4
    conditions and `RegimeFilter`'s 7 gates depend on -- see this module's
    docstring for why that's safe and why it's necessary for speed.
    Sourced from the ACTUAL `direction_engine`/`regime_filter` instances
    passed in, so the cache automatically matches whatever lengths/
    thresholds that configuration uses (defaults or custom-calibrated).

    Args:
        ltf_df: The traded symbol's closed 15m OHLCV, oldest-first, with a
            "ts" column (epoch ms).
        btc_df: BTCUSDT's closed HTF (e.g. 4h) OHLCV, oldest-first, "ts" column.
        direction_engine / regime_filter: The exact instances the
            production `TradeLifecycleManager` being backtested uses.
        symbol_htf_seconds: The traded symbol's own HTF interval in
            seconds (e.g. 3600 for "1h", matching production's
            `symbol_htf_interval` default) -- derived from `ltf_df` itself,
            never fetched separately (see `HistoricalDataLoader`'s docstring).
        funding_series: Optional funding-rate history (see `load_funding`).
    """

    def __init__(
        self,
        ltf_df: pd.DataFrame,
        btc_df: pd.DataFrame,
        *,
        direction_engine: DirectionEngine,
        regime_filter: RegimeFilter,
        symbol_htf_seconds: int = 3600,
        btc_htf_seconds: int = 4 * 3600,
        funding_series: pd.Series | None = None,
    ) -> None:
        self.ltf_df = ltf_df.reset_index(drop=True)
        self.btc_df = btc_df.reset_index(drop=True)
        self.de = direction_engine
        self.rf = regime_filter
        self.funding_series = funding_series
        n = len(self.ltf_df)
        ts = self.ltf_df["ts"].to_numpy()
        high, low, close, volume = self.ltf_df["high"], self.ltf_df["low"], self.ltf_df["close"], self.ltf_df["volume"]

        # -- DirectionEngine's four conditions, vectorized --
        self.ema_fast = ind.ema(close, direction_engine.ema_fast_length)
        self.ema_slow = ind.ema(close, direction_engine.ema_slow_length)  # == production's ema50 enrich series too (same default length)
        self.vwap = ind.daily_anchored_vwap(high, low, close, volume, pd.to_datetime(self.ltf_df["ts"], unit="ms", utc=True))
        st = ind.supertrend(high, low, close, length=direction_engine.supertrend_length, multiplier=direction_engine.supertrend_multiplier)
        self.supertrend_direction = st.direction
        self.atr14 = ind.average_true_range(high, low, close, length=14)  # shared by enrich, TradeLifecycleManager, EntryLadderEngine (all default length 14)
        self.ema20 = ind.ema(close, 20)  # Step 10.10: PositionState's time-decay invalidation momentum check

        self.symbol_htf_ms = symbol_htf_seconds * 1000
        self.htf_frame = _htf_bucket_frame(self.ltf_df, self.symbol_htf_ms)
        self.htf_row_idx = _asof_htf_row_index(ts, self.symbol_htf_ms, self.htf_frame)
        if len(self.htf_frame) >= direction_engine.htf_swing_left + 2 * direction_engine.htf_swing_right + 2:
            pivots = ind.detect_swing_pivots(self.htf_frame["high"], self.htf_frame["low"], left_bars=direction_engine.htf_swing_left, right_bars=direction_engine.htf_swing_right)
            htf_atr = None
            if direction_engine.htf_displacement_atr_mult > 0:
                htf_atr = ind.average_true_range(self.htf_frame["high"], self.htf_frame["low"], self.htf_frame["close"], length=direction_engine.htf_atr_length)
            events = ind.detect_structure_breaks(self.htf_frame["close"], pivots.pivot_high, pivots.pivot_low, atr=htf_atr, displacement_atr_mult=direction_engine.htf_displacement_atr_mult)
            self.htf_trend_by_bucket = events.trend.to_numpy()
        else:
            self.htf_trend_by_bucket = np.zeros(len(self.htf_frame), dtype=np.int64)

        # -- RegimeFilter's seven gates, vectorized --
        self.ker = ind.kaufman_efficiency_ratio(close, length=regime_filter.ker_length)
        adx_result = ind.average_directional_index(high, low, close, length=regime_filter.adx_length)
        self.adx = adx_result.adx
        self.rvol = ind.relative_volume(volume, length=regime_filter.rvol_length)
        self.stretch_atr = ind.average_true_range(high, low, close, length=regime_filter.stretch_atr_length)
        self.stretch_ema = ind.ema(close, regime_filter.stretch_ema_length)
        # Step 11.0 squeeze gate: `.rolling().rank(pct=True)` is pandas'
        # vectorized percentile-rank-of-the-window's-last-value -- the same
        # thing RegimeFilter._squeeze_gate computes per-bar as
        # `(window <= last_width).mean()`, modulo tie-breaking convention
        # (average rank vs. count-inclusive), which real, continuous BB
        # width values essentially never actually hit.
        self.bb_width = ind.bollinger_band_width(close, length=regime_filter.squeeze_bb_length, num_std=regime_filter.squeeze_bb_std)
        self.squeeze_percentile = self.bb_width.rolling(regime_filter.squeeze_lookback).rank(pct=True)

        self.btc_htf_ms = btc_htf_seconds * 1000
        self.btc_ema = ind.ema(self.btc_df["close"], regime_filter.btc_ema_length) if len(self.btc_df) else pd.Series(dtype=float)
        btc_ts = self.btc_df["ts"].to_numpy() if len(self.btc_df) else np.array([], dtype=np.int64)
        # A BTC HTF candle with open-time t covers [t, t+btc_htf_ms) and is
        # only fully closed once an LTF bar at or after t+btc_htf_ms arrives.
        cutoff = ts - self.btc_htf_ms
        self.btc_row_idx = np.searchsorted(btc_ts, cutoff, side="right") - 1 if len(btc_ts) else np.full(n, -1)

        self._n = n

    # -- per-bar reads --------------------------------------------------------

    def enrich(self, i: int) -> dict:
        """`{"atr","ema20"}` for `PositionState.update`, matching
        `MarketDataPipeline._enrich_bar`'s exact optional-field contract (a
        value is simply omitted if not yet computable). `supertrend_direction`/
        `ema50` are no longer part of this contract as of Step 10.10 (the
        dynamic runner tier that read them is gone) -- `self.supertrend_direction`
        the SERIES is still used by `bias()` above, just not surfaced here."""
        out: dict = {}
        atr = self.atr14.iat[i]
        if not pd.isna(atr):
            out["atr"] = float(atr)
        ema20 = self.ema20.iat[i]
        if not pd.isna(ema20):
            out["ema20"] = float(ema20)
        return out

    def htf_bar(self, i: int) -> dict | None:
        row = self.htf_row_idx[i]
        if row < 0:
            return None
        return {"high": float(self.htf_frame["high"].iat[row]), "low": float(self.htf_frame["low"].iat[row])}

    def htf_df_upto(self, i: int, *, buffer_capacity: int | None = None) -> pd.DataFrame | None:
        row = self.htf_row_idx[i]
        if row < 0:
            return None
        start = max(0, row + 1 - buffer_capacity) if buffer_capacity else 0
        return self.htf_frame.iloc[start: row + 1][["high", "low", "close"]].reset_index(drop=True)

    def btc_df_upto(self, i: int, *, buffer_capacity: int | None = None) -> pd.DataFrame:
        row = self.btc_row_idx[i]
        if row < 0:
            return self.btc_df.iloc[0:0]
        start = max(0, row + 1 - buffer_capacity) if buffer_capacity else 0
        return self.btc_df.iloc[start: row + 1].reset_index(drop=True)

    def bias(self, i: int) -> str:
        """DirectionEngine's exact 4-way confluence rule (see
        directional_bias.py's `get_directional_bias`), evaluated from the
        precomputed series instead of recomputing them."""
        close = float(self.ltf_df["close"].iat[i])
        ema_fast, ema_slow = self.ema_fast.iat[i], self.ema_slow.iat[i]
        vwap = self.vwap.iat[i]
        st_dir = self.supertrend_direction.iat[i]
        htf_row = self.htf_row_idx[i]
        htf_trend = self.htf_trend_by_bucket[htf_row] if htf_row >= 0 else 0

        if pd.isna(ema_fast) or pd.isna(ema_slow):
            ema_long = ema_short = False
        else:
            ema_long, ema_short = close > ema_fast > ema_slow, close < ema_fast < ema_slow
        vwap_long = vwap_short = False
        if not pd.isna(vwap):
            vwap_long, vwap_short = close > float(vwap), close < float(vwap)
        st_long, st_short = st_dir == 1, st_dir == -1
        htf_long, htf_short = htf_trend == 1, htf_trend == -1

        if ema_long and vwap_long and st_long and htf_long:
            return "LONG"
        if ema_short and vwap_short and st_short and htf_short:
            return "SHORT"
        return "NEUTRAL"

    def regime_ok(self, i: int, direction: str) -> bool:
        """The 7 RegimeFilter gates' exact thresholds, vectorized-then-read
        (see regime_filter.py's `_chop_gate`/`_trend_strength_gate`/
        `_liquidity_volume_gate`/`_macro_beta_gate`/`_stretch_gate`/
        `_squeeze_gate`/`_crowd_sentiment_gate` -- this mirrors each one's
        formula against the same config)."""
        rf = self.rf
        ker, adx, rvol = self.ker.iat[i], self.adx.iat[i], self.rvol.iat[i]
        if pd.isna(ker) or ker < rf.ker_min:
            return False
        if pd.isna(adx):
            return False
        adx_ok = adx >= rf.adx_min
        if not adx_ok and i > 0:
            prior_adx = self.adx.iat[i - 1]
            if not pd.isna(prior_adx) and adx >= rf.adx_slope_min_threshold and (adx - prior_adx) > rf.adx_slope_min_delta:
                adx_ok = True
        if not adx_ok:
            return False
        if pd.isna(rvol) or rvol < rf.rvol_min:
            return False

        stretch_atr, stretch_ema = self.stretch_atr.iat[i], self.stretch_ema.iat[i]
        if pd.isna(stretch_atr) or stretch_atr <= 0 or pd.isna(stretch_ema):
            return False
        close = float(self.ltf_df["close"].iat[i])
        if abs(close - float(stretch_ema)) / float(stretch_atr) > rf.stretch_atr_max:
            return False

        squeeze_pct = self.squeeze_percentile.iat[i]
        if pd.isna(squeeze_pct) or squeeze_pct > rf.squeeze_max_percentile:
            return False

        btc_row = self.btc_row_idx[i]
        if btc_row < 0 or btc_row >= len(self.btc_ema):
            return False
        btc_ema = self.btc_ema.iat[btc_row]
        if pd.isna(btc_ema):
            return False
        btc_close = float(self.btc_df["close"].iat[btc_row])
        macro_ok = (btc_close > btc_ema) if direction == "LONG" else (btc_close < btc_ema)
        if not macro_ok:
            return False

        funding = funding_asof(self.funding_series, int(self.ltf_df["ts"].iat[i]))
        if funding is not None:
            crowd_ok = (funding <= rf.funding_veto_pct) if direction == "LONG" else (funding >= -rf.funding_veto_pct)
            if not crowd_ok:
                return False
        return True

    def is_candidate(self, i: int) -> str | None:
        """LONG/SHORT if this bar's cheap-to-vectorize signals make a trade
        even numerically possible, else None. A superset check only --
        skipping a call to the real `open_trade()` on a bar this returns
        None for never changes what trades get taken, since the real
        RegimeFilter/DirectionEngine would fail this same bar too (same
        thresholds, same underlying series)."""
        direction = self.bias(i)
        if direction == "NEUTRAL":
            return None
        return direction if self.regime_ok(i, direction) else None


# ---------------------------------------------------------------------------
# 3. Event-driven simulation loop
# ---------------------------------------------------------------------------

def simulate_symbol(
    symbol: str,
    ltf_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    *,
    trade_manager: TradeLifecycleManager,
    harness: DryRunHarness,
    funding_series: pd.Series | None = None,
    strategy_id: str = "KRYPTIC",
    buffer_capacity: int = 1000,
) -> tuple[list[dict], int]:
    """Replay `ltf_df` chronologically through the exact production
    decision stack, recording fee/slippage-adjusted paper trades into
    `harness`. Returns `(trades, signals_generated)`: `trades` is
    `harness.closed_trades` (only the ones this call added, in case
    `harness` is shared/reused across symbols); `signals_generated` is the
    count of bars where `VectorizedSignalCache.is_candidate()` found a
    numerically-possible setup and `open_trade()` was actually called on it
    -- the top of the institutional report's signal funnel (see
    `harness.trades_opened_count`/`trades_expired_unfilled_count` for the
    next two stages of that same funnel).

    Args:
        ltf_df: The traded symbol's closed 15m OHLCV, oldest-first
            (include a leading warmup buffer before your intended analysis
            start date -- every indicator here needs its own lookback, and
            an insufficient one just means no trades in that stretch, not
            an error).
        btc_df: BTCUSDT's closed HTF OHLCV over the same window.
        trade_manager: The exact `TradeLifecycleManager` to backtest --
            pass one built the same way production configures it (its own
            `entry_ladder_engine.regime_filter`/`.direction_engine` supply
            every threshold `VectorizedSignalCache` uses).
        harness: A `DryRunHarness` to record trades into.
        funding_series: Optional (see `HistoricalDataLoader.load_funding`).
        buffer_capacity: Caps how much trailing history is handed to the
            real `open_trade()` call on a candidate bar (default 1000,
            matching `data_collector.MarketDataPipeline`'s own
            `CandleBuffer` default capacity) -- NOT an approximation
            introduced for speed, this is what makes a candidate bar's
            evaluation identical to what production would actually see: a
            live deployment never has more than `buffer_capacity` candles
            buffered either. Only the cheap `VectorizedSignalCache` pre-
            filter (deciding WHETHER a bar is even worth calling
            `open_trade()` on) uses the full available history -- see the
            module docstring for why that's safe. Without this cap, a
            candidate late in a long backtest would hand `open_trade()`
            (and the real RegimeFilter/DirectionEngine it calls) the ENTIRE
            history so far, an ever-growing, production-unrealistic cost
              that dominated a 90-day/3-symbol run's wall clock in testing.
    """
    entry_engine = trade_manager.entry_ladder_engine
    cache = VectorizedSignalCache(
        ltf_df, btc_df,
        direction_engine=entry_engine.direction_engine, regime_filter=entry_engine.regime_filter,
        funding_series=funding_series,
    )
    n = len(cache.ltf_df)
    before = len(harness.closed_trades)
    signals_generated = 0

    position: PositionState | None = None
    for i in range(n):
        bar = {k: cache.ltf_df[k].iat[i] if k in cache.ltf_df.columns else None for k in _REQUIRED_CANDLE_KEYS}
        bar["ts"] = int(bar["ts"])

        if position is not None and not position.closed:
            enriched = {**bar, **cache.enrich(i)}
            events = position.update(enriched, htf_bar=cache.htf_bar(i))
            if events:
                harness.on_bar_events(strategy_id=strategy_id, symbol=symbol, position=position, events=events)
            if position.closed:
                harness.on_position_closed(strategy_id=strategy_id, symbol=symbol, position=position, closed_bar=bar)
                position = None
            continue

        direction = cache.is_candidate(i)
        if direction is None:
            continue
        signals_generated += 1

        sub_ltf = cache.ltf_df.iloc[max(0, i + 1 - buffer_capacity): i + 1].reset_index(drop=True)
        sub_btc = cache.btc_df_upto(i, buffer_capacity=buffer_capacity)
        sub_htf = cache.htf_df_upto(i, buffer_capacity=buffer_capacity)
        funding = funding_asof(funding_series, bar["ts"])
        new_position, diagnostics = trade_manager.open_trade(sub_ltf, sub_btc, funding, htf_df=sub_htf)
        if new_position is not None and new_position.direction == direction:
            position = new_position
            harness.on_trade_opened(strategy_id=strategy_id, symbol=symbol, position=position, bar=bar, diagnostics=diagnostics)

    return harness.closed_trades[before:], signals_generated


# ---------------------------------------------------------------------------
# 4. Performance analytics & reporting
# ---------------------------------------------------------------------------

_BREAKEVEN_R_EPSILON = 0.05


@dataclass
class BacktestReport:
    """Institutional-style performance summary (spec point 3; extended for
    the institutional multi-tab Excel/CSV report)."""

    label: str
    start_ts: int | None
    end_ts: int | None
    trades: list[dict]
    equity_curve: list[dict]  # [{"ts": int, "equity": float}]
    risk_per_trade_pct: float
    starting_equity: float
    sample_size_target: int = 300

    # Scope/funnel (institutional report's "Window & Scope" and "Performance
    # Breakdown" sections) -- populated by BacktestEngine, not from_trades'
    # own trade list, since a dropped phantom trade (see live_runner.py's
    # `filled_weight <= 0` guard) never reaches `trades` at all.
    symbols: list[str] = field(default_factory=list)
    strategy_config: dict = field(default_factory=dict)
    signals_generated: int = 0
    setups_opened: int = 0
    expired_unfilled: int = 0

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float | None = None       # gross_win / gross_loss in net_pnl (price-unit-weighted) terms
    expectancy_r: float | None = None
    max_drawdown_pct: float = 0.0
    max_drawdown_r: float = 0.0
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    avg_duration_seconds: float | None = None
    entry_tier_hit_rate_pct: dict = field(default_factory=dict)

    # Institutional report additions: R-space P&L, payoff, streaks, side mix.
    gross_win_r: float = 0.0
    gross_loss_r: float = 0.0
    net_realized_r: float = 0.0
    avg_win_r: float | None = None
    avg_loss_r: float | None = None
    payoff_ratio: float | None = None        # avg_win_r / abs(avg_loss_r)
    total_return_pct: float = 0.0
    annualized_return_pct: float | None = None
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    total_fees_paid: float = 0.0
    total_slippage_paid: float = 0.0
    long_count: int = 0
    short_count: int = 0
    avg_fill_pct: float = 0.0
    trade_equity_rows: list[dict] = field(default_factory=list)  # per-trade equity/R walk, aligned with `trades`

    @property
    def meets_sample_size(self) -> bool:
        return self.total_trades >= self.sample_size_target

    @classmethod
    def from_trades(
        cls, label: str, trades: list[dict], *,
        starting_equity: float = 10_000.0, risk_per_trade_pct: float = 1.0, sample_size_target: int = 300,
        symbols: list[str] | None = None, strategy_config: dict | None = None,
        signals_generated: int = 0, setups_opened: int = 0, expired_unfilled: int = 0,
    ) -> "BacktestReport":
        trades = sorted(trades, key=lambda t: t["closed_at_ts"] or 0)
        report = cls(
            label=label, start_ts=trades[0]["opened_at_ts"] if trades else None, end_ts=trades[-1]["closed_at_ts"] if trades else None,
            trades=trades, equity_curve=[], risk_per_trade_pct=risk_per_trade_pct, starting_equity=starting_equity, sample_size_target=sample_size_target,
            symbols=symbols if symbols is not None else [label], strategy_config=strategy_config or {},
            signals_generated=signals_generated, setups_opened=setups_opened, expired_unfilled=expired_unfilled,
        )
        report._compute(trades)
        return report

    @classmethod
    def combine(cls, reports: list["BacktestReport"], *, label: str | None = None) -> "BacktestReport":
        all_trades = [t for r in reports for t in r.trades]
        combined_label = label or "COMBINED(" + ", ".join(r.label for r in reports) + ")"
        starting_equity = reports[0].starting_equity if reports else 10_000.0
        risk_per_trade_pct = reports[0].risk_per_trade_pct if reports else 1.0
        symbols = sorted({s for r in reports for s in r.symbols})
        strategy_config = next((r.strategy_config for r in reports if r.strategy_config), {})
        return cls.from_trades(
            combined_label, all_trades, starting_equity=starting_equity, risk_per_trade_pct=risk_per_trade_pct,
            symbols=symbols, strategy_config=strategy_config,
            signals_generated=sum(r.signals_generated for r in reports),
            setups_opened=sum(r.setups_opened for r in reports),
            expired_unfilled=sum(r.expired_unfilled for r in reports),
        )

    def _compute(self, trades: list[dict]) -> None:
        self.total_trades = len(trades)
        self.long_count = sum(1 for t in trades if t["direction"] == "LONG")
        self.short_count = sum(1 for t in trades if t["direction"] == "SHORT")
        self.avg_fill_pct = (100.0 * sum(t["filled_weight"] for t in trades) / self.total_trades) if trades else 0.0
        if not trades:
            return

        r_multiples = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
        outcomes: list[str] = []
        for t in trades:
            r = t["r_multiple"]
            if r is None:
                is_win = t["net_pnl"] > 0
                is_be = t["net_pnl"] == 0
            else:
                is_win = r > _BREAKEVEN_R_EPSILON
                is_be = abs(r) <= _BREAKEVEN_R_EPSILON
            if is_be:
                self.breakevens += 1
                outcomes.append("be")
            elif is_win:
                self.wins += 1
                outcomes.append("win")
            else:
                self.losses += 1
                outcomes.append("loss")
        decided = self.wins + self.losses
        self.win_rate_pct = (100.0 * self.wins / decided) if decided else 0.0

        gross_win = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
        gross_loss = -sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0)
        if gross_loss > 0:
            self.profit_factor = gross_win / gross_loss
        elif gross_win > 0:
            self.profit_factor = float("inf")
        else:
            self.profit_factor = 0.0

        self.expectancy_r = (sum(r_multiples) / len(r_multiples)) if r_multiples else None
        self.gross_win_r = sum(r for r in r_multiples if r > 0)
        self.gross_loss_r = -sum(r for r in r_multiples if r < 0)
        self.net_realized_r = sum(r_multiples) if r_multiples else 0.0
        winning_r = [r for r in r_multiples if r > _BREAKEVEN_R_EPSILON]
        losing_r = [r for r in r_multiples if r < -_BREAKEVEN_R_EPSILON]
        self.avg_win_r = (sum(winning_r) / len(winning_r)) if winning_r else None
        self.avg_loss_r = (sum(losing_r) / len(losing_r)) if losing_r else None
        self.payoff_ratio = (self.avg_win_r / abs(self.avg_loss_r)) if (self.avg_win_r is not None and self.avg_loss_r) else None

        # Longest consecutive win/loss streaks -- a breakeven trade resets both.
        cur_win = cur_loss = 0
        for outcome in outcomes:
            if outcome == "win":
                cur_win += 1
                cur_loss = 0
            elif outcome == "loss":
                cur_loss += 1
                cur_win = 0
            else:
                cur_win = cur_loss = 0
            self.longest_win_streak = max(self.longest_win_streak, cur_win)
            self.longest_loss_streak = max(self.longest_loss_streak, cur_loss)

        durations = [t["duration_seconds"] for t in trades if t["duration_seconds"] is not None]
        self.avg_duration_seconds = (sum(durations) / len(durations)) if durations else None

        for tier in range(4):
            hit = sum(1 for t in trades if any(f["kind"] == "ENTRY" and f.get("tier_index") == tier for f in t["fills"]))
            self.entry_tier_hit_rate_pct[f"entry_{tier + 1}"] = 100.0 * hit / self.total_trades

        equity = self.starting_equity
        curve = [{"ts": self.start_ts, "equity": equity}]
        peak_equity = equity
        max_dd = 0.0
        cum_r = 0.0
        peak_r = 0.0
        max_dd_r = 0.0
        total_fees_usd = 0.0
        total_slippage_usd = 0.0
        trade_equity_rows: list[dict] = []
        for t in trades:
            r = t["r_multiple"] or 0.0
            equity_before = equity
            equity *= (1.0 + r * self.risk_per_trade_pct / 100.0)
            curve.append({"ts": t["closed_at_ts"], "equity": equity})
            peak_equity = max(peak_equity, equity)
            dd_pct = (peak_equity - equity) / peak_equity * 100.0 if peak_equity > 0 else 0.0
            max_dd = max(max_dd, dd_pct)
            cum_r += r
            peak_r = max(peak_r, cum_r)
            dd_r = peak_r - cum_r
            max_dd_r = max(max_dd_r, dd_r)

            # Fees/slippage are tracked in the same "price units x relative
            # ladder weight" basis as realized_pnl (risk_manager.py's
            # PositionState deliberately models no equity/leverage -- see
            # its own docstring) -- NOT dollars at this report's assumed
            # position sizing. Rescale by the same factor that turns this
            # trade's r_multiple into its actual equity_before -> equity_after
            # move, so "$" figures here are internally consistent with
            # pnl_net_usd/equity_after everywhere else in the report.
            avg_entry, initial_sl, filled_weight = t.get("avg_entry"), t.get("initial_sl"), t.get("filled_weight") or 0.0
            risk_amount = abs(avg_entry - initial_sl) * filled_weight if (avg_entry is not None and initial_sl is not None and filled_weight) else None
            dollar_per_unit = (self.risk_per_trade_pct / 100.0 * equity_before / risk_amount) if risk_amount else None
            fee_usd = t["total_fees"] * dollar_per_unit if dollar_per_unit is not None else None
            slippage_usd = t["total_slippage_cost"] * dollar_per_unit if dollar_per_unit is not None else None
            total_fees_usd += fee_usd or 0.0
            total_slippage_usd += slippage_usd or 0.0

            trade_equity_rows.append({
                "equity_before": equity_before, "equity_after": equity,
                "peak_equity": peak_equity, "drawdown_pct": dd_pct,
                "cumulative_r": cum_r, "peak_r": peak_r, "drawdown_r": dd_r,
                "fee_usd": fee_usd, "slippage_usd": slippage_usd,
            })
        self.equity_curve = curve
        self.trade_equity_rows = trade_equity_rows
        self.max_drawdown_pct = max_dd
        self.max_drawdown_r = max_dd_r
        self.total_fees_paid = total_fees_usd
        self.total_slippage_paid = total_slippage_usd

        self.total_return_pct = (equity / self.starting_equity - 1.0) * 100.0 if self.starting_equity else 0.0
        if self.start_ts is not None and self.end_ts is not None and self.end_ts > self.start_ts and self.starting_equity > 0 and equity > 0:
            months = (self.end_ts - self.start_ts) / (30.4375 * 86400 * 1000)
            # A window under ~1 day makes 12/months a huge exponent --
            # extrapolating a couple of trades' return to "annualized" over
            # that short a span is meaningless anyway (and can literally
            # OverflowError on a synthetic/unit-test-scale timestamp gap).
            if months >= (1.0 / 30.4375):
                try:
                    self.annualized_return_pct = ((equity / self.starting_equity) ** (12.0 / months) - 1.0) * 100.0
                except OverflowError:
                    self.annualized_return_pct = None

        self.sharpe_ratio, self.sortino_ratio = self._sharpe_sortino(curve)

    @staticmethod
    def _sharpe_sortino(curve: list[dict]) -> tuple[float | None, float | None]:
        """Annualized Sharpe/Sortino from the equity curve's DAILY returns
        (crypto trades 24/7, so annualized via sqrt(365), not 252).
        Equity only changes when a trade closes, so the curve is
        forward-filled to a daily index first -- the standard way to turn
        an irregularly-spaced trade-level equity curve into a return series."""
        valid = [c for c in curve if c["ts"] is not None]
        if len(valid) < 3:
            return None, None
        s = pd.Series(
            [c["equity"] for c in valid],
            index=pd.to_datetime([c["ts"] for c in valid], unit="ms", utc=True),
        )
        s = s[~s.index.duplicated(keep="last")].sort_index()
        daily = s.resample("1D").last().ffill()
        returns = daily.pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0:
            return None, None
        sharpe = float(returns.mean() / returns.std() * np.sqrt(365))
        downside = returns[returns < 0]
        sortino = float(returns.mean() / downside.std() * np.sqrt(365)) if len(downside) and downside.std() > 0 else None
        return sharpe, sortino

    def format_summary(self) -> str:
        dur = self.avg_duration_seconds
        dur_str = f"{dur / 3600.0:.2f}h" if dur is not None else "n/a"
        pf_str = "inf" if self.profit_factor == float("inf") else (f"{self.profit_factor:.2f}" if self.profit_factor is not None else "n/a")
        sharpe_str = f"{self.sharpe_ratio:.2f}" if self.sharpe_ratio is not None else "n/a"
        sortino_str = f"{self.sortino_ratio:.2f}" if self.sortino_ratio is not None else "n/a"
        ev_str = f"{self.expectancy_r:+.3f}R" if self.expectancy_r is not None else "n/a"
        sample_flag = "MET" if self.meets_sample_size else "NOT MET"
        tiers = "  ".join(f"{k}={v:.1f}%" for k, v in self.entry_tier_hit_rate_pct.items())

        lines = [
            f"===== KRYPTIC Backtest Report: {self.label} =====",
            f"Total Trades:        {self.total_trades}  (target: {self.sample_size_target}+, {sample_flag})",
            f"Win / Loss / BE:     {self.wins} / {self.losses} / {self.breakevens}",
            f"Win Rate:            {self.win_rate_pct:.1f}%",
            f"Profit Factor:       {pf_str}",
            f"Expectancy (EV):     {ev_str} per trade",
            f"Max Drawdown:        {self.max_drawdown_pct:.2f}%",
            f"Sharpe Ratio:        {sharpe_str}",
            f"Sortino Ratio:       {sortino_str}",
            f"Avg Trade Duration:  {dur_str}",
            f"Entry Tier Hit Rate: {tiers or 'n/a'}",
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.format_summary())

    # -- institutional multi-tab report (Excel) / Trades-tab CSV -----------

    def _build_summary_rows(self) -> list[tuple]:
        """Tab 1 "Summary": Window & Scope, Core Strategy Parameters, and
        the Performance Breakdown (executed trades only -- expired/
        cancelled 0-fill setups are reported as their own funnel stage,
        never folded into win/loss/breakeven). Regime gate/ladder-weight
        values are read from `self.strategy_config` (populated by
        `BacktestEngine` from the actual `TradeLifecycleManager` that ran
        this backtest), never hardcoded -- so this always reflects
        whatever calibration actually produced this report."""
        window_str = f"{_fmt_ts(self.start_ts)} -> {_fmt_ts(self.end_ts)}" if self.start_ts and self.end_ts else "n/a"
        duration_str = "n/a"
        if self.start_ts is not None and self.end_ts is not None and self.end_ts > self.start_ts:
            days = (self.end_ts - self.start_ts) / 86_400_000.0
            duration_str = f"{days / 30.4375:.2f} months ({days:.1f} days)"

        cfg = self.strategy_config or {}
        ladder_weights = cfg.get("entry_ladder_weights")
        ladder_str = ", ".join(f"{w:.2f}" for w in ladder_weights) if ladder_weights else "n/a"
        ker_min, adx_min, rvol_min = cfg.get("ker_min"), cfg.get("adx_min"), cfg.get("rvol_min")
        gates_str = f"KER >= {ker_min}, ADX >= {adx_min}, RVOL >= {rvol_min}" if None not in (ker_min, adx_min, rvol_min) else "n/a"

        pf_str = "inf" if self.profit_factor == float("inf") else (round(self.profit_factor, 3) if self.profit_factor is not None else "n/a")
        ev_str = round(self.expectancy_r, 4) if self.expectancy_r is not None else "n/a"
        avg_win_str = round(self.avg_win_r, 3) if self.avg_win_r is not None else "n/a"
        avg_loss_str = round(self.avg_loss_r, 3) if self.avg_loss_r is not None else "n/a"
        payoff_str = round(self.payoff_ratio, 3) if self.payoff_ratio is not None else "n/a"
        annualized_str = round(self.annualized_return_pct, 2) if self.annualized_return_pct is not None else "n/a"
        dur_str = round(self.avg_duration_seconds / 3600.0, 2) if self.avg_duration_seconds is not None else "n/a"

        return [
            ("Backtest window (UTC)", window_str),
            ("Window duration", duration_str),
            ("Strategy profile", "kryptic_trend_ignition"),
            ("Universe symbols & size", f"{', '.join(self.symbols)} ({len(self.symbols)})" if self.symbols else "n/a"),
            ("", ""),
            ("Entry Ladder Weights", ladder_str),
            ("Regime Gates", gates_str),
            ("Risk per trade", f"{self.risk_per_trade_pct:.2f}%"),
            ("", ""),
            ("Signals generated", self.signals_generated),
            ("Setups opened", self.setups_opened),
            ("Expired/cancelled setups (0-fill)", self.expired_unfilled),
            ("Avg fill % of intended ladder", round(self.avg_fill_pct, 2)),
            ("Total Executed Trades", self.total_trades),
            ("Wins", self.wins),
            ("Losses", self.losses),
            (f"Breakeven (|R| <= {_BREAKEVEN_R_EPSILON})", self.breakevens),
            ("Win Rate %", round(self.win_rate_pct, 2)),
            ("Profit Factor", pf_str),
            ("Gross Profit R", round(self.gross_win_r, 3)),
            ("Gross Loss R", round(self.gross_loss_r, 3)),
            ("Net Realized R", round(self.net_realized_r, 3)),
            ("Expectancy (EV, R per trade)", ev_str),
            ("Avg Winning R", avg_win_str),
            ("Avg Losing R", avg_loss_str),
            ("Payoff Ratio (avg win R / |avg loss R|)", payoff_str),
            ("Total Return %", round(self.total_return_pct, 2)),
            ("Approx Annualized Return %", annualized_str),
            ("Max Drawdown %", round(self.max_drawdown_pct, 2)),
            ("Max Drawdown (R)", round(self.max_drawdown_r, 3)),
            ("Avg Hold Duration (hours)", dur_str),
            ("Longest Win Streak", self.longest_win_streak),
            ("Longest Loss Streak", self.longest_loss_streak),
            ("Total Fees Paid ($, equity-scaled)", round(self.total_fees_paid, 4)),
            ("Total Slippage Paid ($, equity-scaled)", round(self.total_slippage_paid, 4)),
            ("LONG trades", self.long_count),
            ("SHORT trades", self.short_count),
        ]

    def _build_trade_rows(self) -> list[dict]:
        """Tab 2 "Trades": one row per executed (filled) trade -- expired/
        cancelled 0-fill setups never reach `self.trades` at all (dropped
        upstream by live_runner.DryRunHarness.on_position_closed)."""
        return [
            _trade_to_report_row(t, equity_before=eq["equity_before"], equity_after=eq["equity_after"],
                                  fee_usd=eq["fee_usd"], slippage_usd=eq["slippage_usd"])
            for t, eq in zip(self.trades, self.trade_equity_rows)
        ]

    def _build_equity_curve_rows(self) -> list[dict]:
        """Tab 3 "Equity Curve": per-trade (not per-candle -- this engine
        doesn't retain a per-candle equity series) chronological R and $
        tracking, both cumulative and off-peak."""
        rows = []
        for i, (t, eq) in enumerate(zip(self.trades, self.trade_equity_rows), start=1):
            rows.append({
                "timestamp": _fmt_ts(t["closed_at_ts"]), "trade_num": i, "symbol": t["symbol"],
                "r_realized": t["r_multiple"], "cumulative_r": round(eq["cumulative_r"], 4),
                "peak_r": round(eq["peak_r"], 4), "drawdown_r": round(eq["drawdown_r"], 4),
                "account_equity": round(eq["equity_after"], 2), "peak_equity": round(eq["peak_equity"], 2),
                "drawdown_pct": round(eq["drawdown_pct"], 4),
            })
        return rows

    def _build_monthly_rows(self) -> list[dict]:
        """Tab 4 "Monthly": calendar-month performance, each month's Return
        %/Max DD % compounded fresh from 1.0 at that month's first trade
        (not a slice of the whole-window equity curve) -- Profit Factor
        here is R-based (gross win R / gross loss R), unlike the Summary
        tab's net_pnl-based one, since R is this report's one truly
        risk-normalized unit."""
        buckets: dict[str, list[int]] = {}
        for i, t in enumerate(self.trades):
            if t["closed_at_ts"] is None:
                continue
            month = datetime.fromtimestamp(t["closed_at_ts"] / 1000.0, tz=timezone.utc).strftime("%Y-%m")
            buckets.setdefault(month, []).append(i)

        rows = []
        for month in sorted(buckets):
            idxs = buckets[month]
            r_vals = [self.trades[i]["r_multiple"] for i in idxs if self.trades[i]["r_multiple"] is not None]
            wins = sum(1 for r in r_vals if r > _BREAKEVEN_R_EPSILON)
            losses = sum(1 for r in r_vals if r < -_BREAKEVEN_R_EPSILON)
            decided = wins + losses
            gross_win = sum(r for r in r_vals if r > 0)
            gross_loss = -sum(r for r in r_vals if r < 0)
            pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

            month_equity = 1.0
            peak = 1.0
            max_dd = 0.0
            for i in idxs:
                r = self.trades[i]["r_multiple"] or 0.0
                month_equity *= (1.0 + r * self.risk_per_trade_pct / 100.0)
                peak = max(peak, month_equity)
                if peak > 0:
                    max_dd = max(max_dd, (peak - month_equity) / peak * 100.0)

            rows.append({
                "month": month, "trades": len(idxs),
                "win_rate_pct": round(100.0 * wins / decided, 2) if decided else 0.0,
                "realized_r": round(sum(r_vals), 3) if r_vals else 0.0,
                "return_pct": round((month_equity - 1.0) * 100.0, 3),
                "max_dd_pct": round(max_dd, 3),
                "profit_factor": "inf" if pf == float("inf") else round(pf, 3),
            })
        return rows

    def _build_notes_rows(self) -> list[str]:
        """Tab 5 "Notes": engine invariants and execution assumptions a
        reader needs to correctly interpret the other four tabs."""
        fees = self.strategy_config.get("fees") if self.strategy_config else None
        fee_str = (
            f"{fees['taker_fee_pct']}% taker / {fees['maker_fee_pct']}% maker, "
            f"{fees['market_slippage_pct']}% market-order slippage"
            if fees else "see FeeSlippageModel defaults (0.05% taker / 0.02% maker / 0.02% slippage)"
        )
        return [
            "KRYPTIC backtest_engine.py reuses the exact, unmodified production decision classes",
            "(RegimeFilter, DirectionEngine, EntryLadderEngine, TradeLifecycleManager, PositionState) --",
            "see that module's own docstring for how the vectorized signal cache stays lookahead-free.",
            "",
            f"Fee/slippage model: {fee_str}.",
            "",
            "PositionState deliberately models NO account equity or leverage (pure relative ladder-weight",
            "sizing, weights summing to 1.0 per trade) -- see risk_manager.py's own docstring. Every '$'",
            "figure in this report (pnl_net_usd, equity_after, equity_pct_change, Total Fees/Slippage Paid)",
            "is therefore DERIVED, not measured: each trade's raw price-unit P&L is rescaled by the same",
            "factor that turns its r_multiple into this report's own risk_per_trade_pct-of-equity",
            "compounding model. Treat R-multiple figures (Expectancy, Gross Profit/Loss R, Payoff Ratio,",
            "Max Drawdown (R)) as the primary signal; treat compounded '$'/'%' figures as illustrative",
            "of ONE particular sizing convention, not a guaranteed real-money outcome.",
            "",
            "'score' and 'leverage' (Trades tab) are ZENITH/GEM concepts (see backtest.py) that KRYPTIC's",
            "TradeLifecycleManager does not compute -- left blank here rather than fabricated. 'stretch_atr'",
            "is KRYPTIC's own analog: the planned risk_atr_multiple at setup time (|expected_vwap -",
            "initial_sl| / ATR14), not an EMA-stretch reading (that's a TP5 EXIT-time signal, not available",
            "at entry). 'htf_aligned' is DirectionEngine's HTF structure confluence condition -- always True",
            "for every row here, since a non-NEUTRAL bias requires all 4 confluence conditions to pass.",
            "",
            "'Expired/cancelled setups' (Summary) are positions TradeLifecycleManager.open_trade() opened",
            "that never filled a single entry tier before every resting order was cancelled (typically:",
            "price ran straight to the TP2 invalidation level without ever pulling back into the ladder).",
            "Zero risk was ever taken and zero PnL was ever possible, so they are excluded from every",
            "win/loss/breakeven/R statistic in this report, not folded in as a breakeven no-op.",
            "",
            "Within one bar, a stop-loss touch is checked before take-profit/entry-fill checks (the",
            "conservative assumption when intrabar order is unknown from OHLC alone) -- see",
            "PositionState.update()'s own docstring for the full per-bar execution order.",
            "",
            f"Generated {_fmt_ts(int(time.time() * 1000))} UTC.",
        ]

    def export_csv(self, filepath: str | Path) -> None:
        """Dump the institutional report's "Trades" tab (see `export_excel`)
        as a standalone CSV -- one row per executed trade with setup
        diagnostics, planned risk/targets, fill quality, exit outcome, and
        the per-trade equity walk. Supersedes the 13-column Step 10.6
        schema (symbol/direction/entry_time/... are still here, just under
        this richer column set -- see `_TRADE_ROW_FIELDS`)."""
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TRADE_ROW_FIELDS)
            writer.writeheader()
            for row in self._build_trade_rows():
                writer.writerow(row)

    def export_excel(self, filepath: str | Path) -> None:
        """Write the institutional 5-tab workbook (Summary, Trades, Equity
        Curve, Monthly, Notes) via openpyxl. See each `_build_*_rows`
        method's docstring for that tab's exact contents/conventions."""
        df_summary = pd.DataFrame(self._build_summary_rows(), columns=["Metric", "Value"])
        df_trades = pd.DataFrame(self._build_trade_rows(), columns=_TRADE_ROW_FIELDS)
        df_equity = pd.DataFrame(self._build_equity_curve_rows(), columns=_EQUITY_ROW_FIELDS)
        df_monthly = pd.DataFrame(self._build_monthly_rows(), columns=_MONTHLY_ROW_FIELDS)
        df_notes = pd.DataFrame({"Notes": self._build_notes_rows()})

        with pd.ExcelWriter(filepath, engine="openpyxl") as xw:
            df_summary.to_excel(xw, sheet_name="Summary", index=False)
            df_trades.to_excel(xw, sheet_name="Trades", index=False)
            df_equity.to_excel(xw, sheet_name="Equity Curve", index=False)
            df_monthly.to_excel(xw, sheet_name="Monthly", index=False)
            df_notes.to_excel(xw, sheet_name="Notes", index=False)
            for ws in xw.book.worksheets:
                for col_cells in ws.columns:
                    length = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = min(60, max(10, length + 2))


def _fmt_ts(ts: int | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat()


def _trade_exit_reason(trade: dict) -> tuple[str, float | None]:
    """(exit_reason, exit_price) from a PaperTradeRecord-shaped dict's
    `fills` -- the LAST exit fill chronologically is what actually closed
    the position. "BE" (rather than "SL") means the Step 10.10 active
    scratch-win stop had already migrated by the time the stop fired --
    see risk_manager.PositionState.breakeven_moved -- a small guaranteed
    win, not a loss against the trade's original risk. RUNNER_EXIT was
    dormant from Step 5/10.8's original dynamic runner tier (removed by
    Step 10.10) until Step 11.0 reintroduced a (differently-built) ATR
    chandelier-trail runner tier that emits the same label."""
    exit_fills = [f for f in trade["fills"] if f["kind"] == "EXIT"]
    if not exit_fills:
        return "", None
    last_exit = max(exit_fills, key=lambda f: f["bar_index"])
    kind = last_exit["event_type"]
    if kind == "SL_HIT":
        reason = "BE" if trade.get("breakeven_moved") else "SL"
    elif kind == "RUNNER_EXIT":
        reason = "RUNNER"
    elif kind == "TIME_DECAY_EXIT":
        reason = "TIME_DECAY"
    elif kind == "TP_HIT":
        tier_index = last_exit.get("tier_index")
        reason = f"TP{tier_index + 1}" if tier_index is not None else "TP"
    else:
        reason = kind
    return reason, last_exit["fill_price"]


_TRADE_ROW_FIELDS = [
    "open_time", "symbol", "side", "score", "leverage", "htf_aligned", "vol_ratio", "stretch_atr", "ker_ratio",
    "entry_avg_intended", "entry_avg_filled", "fill_pct", "entry_tiers_hit",
    "sl", "be_price", "tp1", "tp2", "tp3", "risk_pct",
    "close_time", "close_reason", "bars_held", "hold_hours",
    "r_realized", "pnl_net_usd", "fees_paid", "slippage_paid", "equity_pct_change", "equity_after",
]
_EQUITY_ROW_FIELDS = [
    "timestamp", "trade_num", "symbol", "r_realized", "cumulative_r", "peak_r",
    "drawdown_r", "account_equity", "peak_equity", "drawdown_pct",
]
_MONTHLY_ROW_FIELDS = ["month", "trades", "win_rate_pct", "realized_r", "return_pct", "max_dd_pct", "profit_factor"]


def _trade_to_report_row(trade: dict, *, equity_before: float, equity_after: float, fee_usd: float | None, slippage_usd: float | None) -> dict:
    """Institutional "Trades" tab row (`_TRADE_ROW_FIELDS`) -- see
    `BacktestReport._build_notes_rows` for why `score`/`leverage` are
    always blank and what `stretch_atr`/`htf_aligned` actually mean here."""
    entry_tiers = sorted({f["tier_index"] + 1 for f in trade["fills"] if f["kind"] == "ENTRY" and f.get("tier_index") is not None})
    exit_reason, _exit_price = _trade_exit_reason(trade)
    duration_hours = trade["duration_seconds"] / 3600.0 if trade["duration_seconds"] is not None else None
    tp_levels = trade.get("tp_levels") or ()
    expected_vwap, initial_sl = trade.get("expected_vwap"), trade.get("initial_sl")
    risk_pct = abs(expected_vwap - initial_sl) / abs(expected_vwap) * 100.0 if (expected_vwap and initial_sl is not None) else None
    equity_pct_change = ((equity_after / equity_before) - 1.0) * 100.0 if equity_before else None
    return {
        "open_time": _fmt_ts(trade["opened_at_ts"]), "symbol": trade["symbol"], "side": trade["direction"],
        "score": None, "leverage": None,
        "htf_aligned": trade.get("htf_aligned"), "vol_ratio": trade.get("rvol"),
        "stretch_atr": trade.get("risk_atr_multiple"), "ker_ratio": trade.get("ker_ratio"),
        "entry_avg_intended": expected_vwap, "entry_avg_filled": trade["avg_entry"],
        "fill_pct": round(trade["filled_weight"] * 100.0, 2),
        "entry_tiers_hit": ",".join(str(t) for t in entry_tiers),
        "sl": initial_sl, "be_price": trade.get("be_price"),
        "tp1": tp_levels[0] if len(tp_levels) > 0 else None,
        "tp2": tp_levels[1] if len(tp_levels) > 1 else None,
        "tp3": tp_levels[2] if len(tp_levels) > 2 else None,
        "risk_pct": risk_pct,
        "close_time": _fmt_ts(trade["closed_at_ts"]), "close_reason": exit_reason,
        "bars_held": trade.get("bars_held"), "hold_hours": duration_hours,
        "r_realized": trade["r_multiple"], "pnl_net_usd": round(equity_after - equity_before, 4),
        "fees_paid": round(fee_usd, 4) if fee_usd is not None else None,
        "slippage_paid": round(slippage_usd, 4) if slippage_usd is not None else None,
        "equity_pct_change": equity_pct_change, "equity_after": round(equity_after, 4),
    }


# ---------------------------------------------------------------------------
# Top-level engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Convenience wrapper: one shared `DryRunHarness` + fee model across
    every symbol, producing a `BacktestReport` per symbol plus a combined
    one across all of them.

    Args:
        trade_manager_factory: `() -> TradeLifecycleManager`, called once
            per symbol (own instance per symbol, matching live_runner.py's
            `LiveRunner` convention -- default: production defaults).
        fees: `FeeSlippageModel` (default: its own production defaults --
            0.05% taker / 0.02% maker / 0.02% slippage, per spec).
        state_dir: Where the (small, informational) paper ledger lives.
        starting_equity / risk_per_trade_pct: `BacktestReport`'s equity-curve params.
    """

    def __init__(
        self,
        *,
        trade_manager_factory: Callable[[], TradeLifecycleManager] | None = None,
        fees: FeeSlippageModel | None = None,
        state_dir: str | Path = "./.backtest_scratch",
        starting_equity: float = 10_000.0,
        risk_per_trade_pct: float = 1.0,
        strategy_id: str = "KRYPTIC",
        buffer_capacity: int = 1000,
    ) -> None:
        self.trade_manager_factory = trade_manager_factory or TradeLifecycleManager
        self.harness = DryRunHarness(state_dir=state_dir, fees=fees)
        self.starting_equity = starting_equity
        self.risk_per_trade_pct = risk_per_trade_pct
        self.strategy_id = strategy_id
        self.buffer_capacity = buffer_capacity

    def _strategy_config(self, trade_manager: TradeLifecycleManager) -> dict:
        """Snapshot of the ACTUAL calibration this run used -- for the
        institutional report's Summary tab, never hardcoded, so it always
        reflects whatever thresholds this specific `TradeLifecycleManager`
        was built with (defaults or custom-calibrated)."""
        rf = trade_manager.entry_ladder_engine.regime_filter
        return {
            "entry_ladder_weights": list(LADDER_WEIGHTS),
            "ker_min": rf.ker_min, "adx_min": rf.adx_min, "rvol_min": rf.rvol_min,
            "fees": {
                "taker_fee_pct": self.harness.fees.taker_fee_pct,
                "maker_fee_pct": self.harness.fees.maker_fee_pct,
                "market_slippage_pct": self.harness.fees.market_slippage_pct,
            },
        }

    def run_symbol(self, symbol: str, ltf_df: pd.DataFrame, btc_df: pd.DataFrame, *, funding_series: pd.Series | None = None) -> BacktestReport:
        trade_manager = self.trade_manager_factory()
        opened_before = self.harness.trades_opened_count
        expired_before = self.harness.trades_expired_unfilled_count
        trades, signals_generated = simulate_symbol(
            symbol, ltf_df, btc_df, trade_manager=trade_manager, harness=self.harness,
            funding_series=funding_series, strategy_id=self.strategy_id, buffer_capacity=self.buffer_capacity,
        )
        setups_opened = self.harness.trades_opened_count - opened_before
        expired_unfilled = self.harness.trades_expired_unfilled_count - expired_before
        return BacktestReport.from_trades(
            symbol, trades, starting_equity=self.starting_equity, risk_per_trade_pct=self.risk_per_trade_pct,
            symbols=[symbol], strategy_config=self._strategy_config(trade_manager),
            signals_generated=signals_generated, setups_opened=setups_opened, expired_unfilled=expired_unfilled,
        )

    def run_many(self, data: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]]) -> dict[str, BacktestReport]:
        """`data`: {symbol: (ltf_df, btc_df, funding_series_or_None)}.
        Returns one report per symbol, plus one under the key "COMBINED"."""
        reports: dict[str, BacktestReport] = {}
        for symbol, (ltf_df, btc_df, funding_series) in data.items():
            reports[symbol] = self.run_symbol(symbol, ltf_df, btc_df, funding_series=funding_series)
        if len(reports) > 1:
            reports["COMBINED"] = BacktestReport.combine(list(reports.values()))
        return reports


# ---------------------------------------------------------------------------
# Universe discovery + concurrent fetch (main() only -- see module docstring)
# ---------------------------------------------------------------------------

async def discover_universe(
    *, min_quote_volume_usd: float, top_n: int, venue: str | None = None,
    load_universe_fn: Callable[[httpx.AsyncClient, float], Awaitable[dict]] | None = None,
) -> list[str]:
    """One-shot snapshot of qualifying active USDT linear perpetuals, for a
    single backtest run -- NOT a recurring scan (see live_runner.py's
    `UniverseScanner` for that). Reuses exchanges.py's own `load_universe`
    (the same source the live universe scanner already uses, selected via
    `venue`/`DATA_SOURCE` exactly like `resolve_source` does elsewhere) and
    `filter_and_rank_universe`'s volume-floor/top-N/stablecoin-and-
    leveraged-token exclusion, rather than a second implementation or a new
    exchange-client dependency. Returns symbols ranked by 24h quote volume
    descending.

    `load_universe_fn` overrides the venue-resolved default (tests inject a
    fake one; production leaves this None)."""
    if load_universe_fn is None:
        load_universe_fn, _fetch_klines, _fetch_funding = ex.resolve_source(venue)
    async with httpx.AsyncClient(timeout=20) as client:
        universe = await load_universe_fn(client, min_quote_vol=0.0)
    cfg = UniverseFilterConfig(min_quote_volume_usd=min_quote_volume_usd, top_n=top_n)
    ranked = filter_and_rank_universe(universe["tickers"], cfg)
    return [t["symbol"] for t in ranked]


async def _load_symbol_data(
    loader: HistoricalDataLoader, semaphore: asyncio.Semaphore, symbol: str,
    btc_df: pd.DataFrame, start_ms: int, end_ms: int,
) -> tuple[str, pd.DataFrame | None, pd.Series | None]:
    """One symbol's LTF+funding fetch, gated by `semaphore` -- bounds how
    many symbols this file fetches from the exchange in parallel. Kept
    conservative by the caller's default specifically because this process
    shares no rate-limit accounting with any live-trading process that
    might be running on the same exchange API key/IP (see module docstring).

    `HistoricalDataLoader`'s default fetcher is Binance-only with no
    per-symbol fallback (unlike exchanges.py's own `fetch_klines`, which
    the live bot uses and which silently falls through to MEXC/Bitget) --
    a hand-picked 2-3 symbol run never hit this, but a dynamically
    discovered universe of 100+ symbols WILL eventually include one
    Binance's klines endpoint 400s on (a listing/naming quirk, delisting,
    insufficient history, ...). Isolated here so one bad symbol can't take
    down an otherwise-successful multi-hundred-symbol run: returns
    `(symbol, None, None)` on any fetch failure, logged as a warning, for
    the caller to skip rather than propagate."""
    try:
        async with semaphore:
            ltf_df = await loader.load(symbol, "15m", start_ms, end_ms)
            funding = await loader.load_funding(symbol, start_ms, end_ms)
        return symbol, ltf_df, funding
    except Exception as e:
        log.warning("skipping %s -- historical fetch failed: %s", symbol, e)
        return symbol, None, None


# ---------------------------------------------------------------------------
# CLI entrypoint (real data -- unverified network, see module docstring)
# ---------------------------------------------------------------------------

async def main() -> None:  # pragma: no cover -- real-network entrypoint, not exercised by tests
    """Example 6-month, 3-symbol run against real downloaded data. NOT
    exercised anywhere in this repo's test suite -- this sandboxed
    environment has no outbound network route to any exchange. Smoke-test
    manually before relying on the downloaded data."""
    import argparse

    parser = argparse.ArgumentParser(description="KRYPTIC historical backtest")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT", help="Comma-separated symbol list; ignored if --all-coins or --top-n is given")
    parser.add_argument("--all-coins", action="store_true", help="Discover the tradeable universe dynamically (active USDT linear perpetuals meeting --min-quote-volume-24h) instead of --symbols")
    parser.add_argument("--top-n", type=int, default=None, help="Cap the discovered universe to the top N symbols by 24h quote volume. Implies --all-coins-style discovery even without that flag; omit for a plain --symbols run")
    parser.add_argument("--min-quote-volume-24h", type=float, default=20_000_000.0, help="24h quote volume floor (USD) for universe discovery (default $20M -- below this, wide spreads/slippage tend to invalidate the exit geometry's scratch buffer)")
    parser.add_argument("--concurrency", type=int, default=4, help="Max symbols fetched from the exchange in parallel (default 4 -- keep conservative, see module docstring on REST rate-limit sharing with any live trading process)")
    parser.add_argument("--max-open-trades", type=str, default=None, metavar="N[,N...]", help="Comma-separated portfolio concurrency caps (e.g. '3,5') for the post-hoc overlap diagnostic (portfolio_overlap.py) -- reports how many of the independently-simulated trades a real cap would have blocked and the resulting EV impact. Diagnostic only; does not change which trades the simulation itself records")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--days", type=int, default=None, help="Overrides --months with an exact day count (e.g. --days 180)")
    parser.add_argument("--warmup-days", type=int, default=10)
    parser.add_argument("--export-trades", type=str, default=None, metavar="PATH.csv", help="Deprecated alias for --export-csv")
    parser.add_argument("--export-csv", type=str, default=None, metavar="PATH.csv", help="Export the institutional Trades-tab report (all symbols, COMBINED) to this CSV path")
    parser.add_argument("--export-xlsx", type=str, default=None, metavar="PATH.xlsx", help="Export the institutional 5-tab workbook (Summary/Trades/Equity Curve/Monthly/Notes) to this .xlsx path -- requires openpyxl")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end_ms = int(time.time() * 1000)
    days = args.days if args.days is not None else args.months * 30
    start_ms = end_ms - days * 86400 * 1000
    warmup_ms = args.warmup_days * 86400 * 1000

    if args.all_coins or args.top_n is not None:
        top_n = args.top_n if args.top_n is not None else 1000  # effectively "every qualifying symbol"
        symbols = await discover_universe(min_quote_volume_usd=args.min_quote_volume_24h, top_n=top_n)
        print(f"Discovered {len(symbols)} qualifying symbol(s) (>= ${args.min_quote_volume_24h:,.0f} 24h quote volume, top {top_n}): {symbols}")
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    loader = HistoricalDataLoader()
    engine = BacktestEngine()
    btc_df = await loader.load("BTCUSDT", "4h", start_ms - warmup_ms, end_ms)

    semaphore = asyncio.Semaphore(args.concurrency)
    fetch_tasks = [_load_symbol_data(loader, semaphore, symbol, btc_df, start_ms - warmup_ms, end_ms) for symbol in symbols]
    data = {}
    skipped: list[str] = []
    for coro in asyncio.as_completed(fetch_tasks):
        symbol, ltf_df, funding = await coro
        if ltf_df is None or ltf_df.empty:
            skipped.append(symbol)
            continue
        data[symbol] = (ltf_df, btc_df, funding)
        print(f"{symbol}: {len(ltf_df)} 15m candle(s) loaded")
    await loader.aclose()
    if skipped:
        print(f"Skipped {len(skipped)} symbol(s) with no usable historical data: {sorted(skipped)}")

    # Inlined (not engine.run_many()) purely for progress visibility: a
    # 2-3 hand-picked-symbol run's silence during this step was a few
    # seconds and unnoticeable, but a 19+/100+ symbol --all-coins run can
    # take several minutes with ZERO output otherwise -- indistinguishable
    # from a real hang. run_many() itself stays print-free (library code,
    # also used by tests); this loop does the same work with progress logged.
    reports: dict[str, BacktestReport] = {}
    for i, (symbol, (ltf_df, symbol_btc_df, funding)) in enumerate(data.items(), start=1):
        t0 = time.time()
        print(f"[{i}/{len(data)}] Running {symbol} ({len(ltf_df)} bars)...", flush=True)
        reports[symbol] = engine.run_symbol(symbol, ltf_df, symbol_btc_df, funding_series=funding)
        print(f"[{i}/{len(data)}] {symbol} done in {time.time() - t0:.1f}s -- {reports[symbol].total_trades} trade(s)", flush=True)
    if len(reports) > 1:
        reports["COMBINED"] = BacktestReport.combine(list(reports.values()))

    for report in reports.values():
        report.print_summary()
        print()

    csv_path = args.export_csv or args.export_trades
    if csv_path:
        combined = reports.get("COMBINED") or next(iter(reports.values()))
        combined.export_csv(csv_path)
        print(f"Exported {combined.total_trades} trade(s) to {csv_path}")

    if args.export_xlsx:
        combined = reports.get("COMBINED") or next(iter(reports.values()))
        combined.export_excel(args.export_xlsx)
        print(f"Exported institutional report ({combined.total_trades} trade(s)) to {args.export_xlsx}")

    if args.max_open_trades:
        caps = [int(x) for x in args.max_open_trades.split(",")]
        combined = reports.get("COMBINED") or next(iter(reports.values()))
        print("\n=== Portfolio overlap diagnostic (post-hoc -- does not alter the simulated trades above) ===")
        for result in overlap_report(combined.trades, caps):
            print(
                f"  max_open_trades={result.max_open_trades:3d}  admitted={result.n_admitted:4d}/{result.n_total:<4d}  "
                f"rejected_sum_R={result.rejected_sum_R:+8.3f}  admitted_ev_R={result.admitted_ev_R:+.4f}  "
                f"baseline_ev_R={result.baseline_ev_R:+.4f}  peak_concurrent_observed={result.max_concurrent_observed}"
            )


if __name__ == "__main__":
    asyncio.run(main())
