"""engine.py -- multi-strategy-safe orchestrator (KRYPTIC Step 8).

`KrypticEngine` is the top-level object a KRYPTIC deployment actually runs.
It composes everything from Steps 1-7 -- indicators, RegimeFilter,
DirectionEngine, EntryLadderEngine, TradeLifecycleManager/PositionState,
MarketDataPipeline's buffering/aggregation/heartbeat machinery, and
resilience_manager's persistence/replay/clock-sync/retry tools -- into one
class, and adds exactly what's needed to run KRYPTIC safely ALONGSIDE this
repo's existing ZENITH and GEM strategies in the same process:

    1. Strategy-scoped state: its own state file
       (state_<strategy>_<symbol>.json), so a KRYPTIC crash/reboot cycle
       can never read, overwrite, or clear another strategy's saved state.
    2. Every persisted payload and log line is tagged with `strategy_id`,
       so a shared log stream or a human reading state_*.json files on disk
       can always tell which strategy produced which line.

`KrypticEngine` subclasses `MarketDataPipeline` (Step 6) rather than
re-implementing candle buffering, HTF aggregation, warmup, or the
heartbeat loop from scratch -- those are strategy-agnostic plumbing this
class only needs to extend, not replace. What it overrides:

    - `warmup()`: adds the exchange clock-drift check, crash recovery from
      this strategy's OWN scoped state file, and wraps the historical/
      funding REST calls in `with_retry` (per spec) -- reimplemented rather
      than monkeypatching `self.exchange`'s methods, since that exchange
      object may be SHARED with other strategies' engines running in the
      same process, and mutating it out from under them would be exactly
      the kind of cross-strategy collision this class exists to prevent.
    - `on_candle_close()` / `on_bar_close()`: the spec names the dispatcher
      `on_bar_close`; `on_candle_close` (inherited plumbing calls this name)
      is kept as a one-line alias so MarketDataPipeline's `_on_raw_message`/
      `_reconnect_and_gapfill` keep working unmodified. Adds strategy-scoped
      persistence after every fill/TP/breakeven/close and immediately on
      opening a new trade.
    - `_reconnect_and_gapfill()`: replaced with `GapFillReplayer`
      (per spec point 4) instead of the parent's simpler re-dispatch loop --
      replays missed candles through the active position specifically (not
      re-evaluating new entries for each missed bar), backfills the candle
      buffers with the same missed candles so they stay contiguous, and
      persists once if anything changed.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

import indicators as ind
from data_collector import DispatchResult, ExchangeStream, MarketDataPipeline
from resilience_manager import ClockSyncResult, GapFillReplayer, StateStore, check_clock_drift, with_retry
from risk_manager import PositionState, TradeLifecycleManager

log = logging.getLogger(__name__)


class KrypticEngine(MarketDataPipeline):
    """The strategy-isolated, crash-safe, gap-fill-aware KRYPTIC engine.

    Args:
        exchange: Duck-typed ExchangeStream (see data_collector.py); may be
            SHARED with other strategies' engines -- this class never
            mutates it.
        symbol: Traded pair, e.g. "BTCUSDT".
        trade_manager: The TradeLifecycleManager to drive.
        strategy_name: Used to build the scoped state filename and the
            `strategy_id` tag on every persisted payload/log line (default
            "KRYPTIC"). Two engines with the same strategy_name AND symbol
            would collide on the same state file -- don't run two of those
            in the same state_dir at once.
        state_dir: Directory for the scoped state file (default ".").
        clock_drift_max_ms: Passed to `check_clock_drift` (default 1000.0).
        **pipeline_kwargs: Forwarded to `MarketDataPipeline.__init__`
            (ltf_interval, btc_htf_interval, symbol_htf_interval,
            warmup_bars, buffer_capacity, heartbeat_grace_seconds,
            funding_poll_seconds, ema_length, atr_length,
            supertrend_length, supertrend_multiplier).
    """

    def __init__(
        self,
        *,
        exchange: ExchangeStream,
        symbol: str,
        trade_manager: TradeLifecycleManager,
        strategy_name: str = "KRYPTIC",
        state_dir: str | Path = ".",
        clock_drift_max_ms: float = 1000.0,
        **pipeline_kwargs: Any,
    ) -> None:
        super().__init__(exchange=exchange, symbol=symbol, trade_manager=trade_manager, **pipeline_kwargs)
        self.strategy_name = strategy_name
        self.strategy_id = strategy_name.upper()
        self.state_path = Path(state_dir) / f"state_{strategy_name.lower()}_{symbol.lower()}.json"
        self.state_store = StateStore(self.state_path)
        self.clock_drift_max_ms = clock_drift_max_ms
        self.gap_replayer = GapFillReplayer(exchange, symbol, self.ltf_interval, fetch_limit=min(self.warmup_bars, 200))
        self.last_clock_sync: ClockSyncResult | None = None
        self.closed_position_reports: list[dict] = []

    # `self.position` is MarketDataPipeline's own attribute; `active_position`
    # is the name the Step 8 spec uses. Same underlying value, two names --
    # not a second, independently-mutable copy of the state.
    @property
    def active_position(self) -> PositionState | None:
        return self.position

    @active_position.setter
    def active_position(self, value: PositionState | None) -> None:
        self.position = value

    # -- boot: clock sync + crash recovery + retried REST warmup ------------

    async def warmup(self) -> None:
        """Clock-drift check, crash recovery, then REST warmup (each
        historical/funding call wrapped in `with_retry`, per spec).

        This reimplements the buffer-population body of
        `MarketDataPipeline.warmup()` rather than calling `super().warmup()`
        and monkeypatching retry behavior onto `self.exchange`'s methods --
        that exchange instance may be shared with other strategies' engines
        running in the same process, and temporarily replacing its methods
        out from under them would itself be a cross-strategy collision.
        """
        if hasattr(self.exchange, "server_time"):
            try:
                self.last_clock_sync = await check_clock_drift(self.exchange, max_drift_ms=self.clock_drift_max_ms)
            except Exception as e:
                self._warn(f"[{self.strategy_id} {self.symbol}] clock drift check failed: {e}")
        else:
            self._warn(f"[{self.strategy_id} {self.symbol}] exchange has no server_time() -- skipping clock drift check")

        recovered = self.state_store.load_position()
        if recovered is not None:
            self.active_position = recovered
            self._last_candle_ts = self.state_store.load_last_candle_ts()
            log.warning(
                "[%s %s] crash recovery: restored an active %s position from %s "
                "(open_size=%.4f, realized_pnl=%.6g, %d prior execution event(s))",
                self.strategy_id, self.symbol, recovered.direction, self.state_path,
                recovered.open_size, recovered.realized_pnl, len(recovered.execution_log),
            )

        fetch_historical = with_retry(max_retries=5, base_delay=1.0)(self.exchange.historical_klines)
        fetch_funding = with_retry(max_retries=5, base_delay=1.0)(self.exchange.funding_rate)

        try:
            candles = await fetch_historical(self.symbol, self.ltf_interval, self.warmup_bars)
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
            # Don't let a fresh warmup candle clobber a MORE RECENT cursor
            # recovered from a crash -- only advance forward, never backward.
            if self._last_candle_ts is None or last["ts"] > self._last_candle_ts:
                self._last_candle_ts = last["ts"]
            self._last_candle_close_monotonic = time.monotonic()

        try:
            btc_candles = await fetch_historical("BTCUSDT", self.btc_htf_interval, self.warmup_bars)
        except Exception as e:
            self._warn(f"historical klines fetch failed for BTCUSDT: {e}")
            btc_candles = []
        for c in btc_candles:
            self.btc_buffer.append(c)
        if len(self.btc_buffer) < self.warmup_bars:
            self._warn(f"only {len(self.btc_buffer)}/{self.warmup_bars} BTCUSDT candles available on warmup")

        try:
            self.funding_rate = await fetch_funding(self.symbol)
            self.funding_rate_updated_at = time.monotonic()
        except Exception as e:
            self._warn(f"funding rate fetch failed: {e}")
            self.funding_rate = None

    # -- bar-close indicator enrichment --------------------------------------

    def _enrich_bar(self, bar: dict, df: pd.DataFrame) -> dict:
        enriched = dict(bar)
        if len(df) >= self.supertrend_length:
            st = ind.supertrend(df["high"], df["low"], df["close"], length=self.supertrend_length, multiplier=self.supertrend_multiplier)
            st_dir = int(st.direction.iloc[-1])
            if st_dir != 0:
                enriched["supertrend_direction"] = st_dir
        if len(df) >= self.ema_length:
            enriched["ema50"] = float(ind.ema(df["close"], self.ema_length).iloc[-1])
        atr_series = ind.average_true_range(df["high"], df["low"], df["close"], length=self.atr_length)
        if not pd.isna(atr_series.iloc[-1]):
            enriched["atr"] = float(atr_series.iloc[-1])
        return enriched

    # -- persistence helpers --------------------------------------------------

    def _persist(self, *, reason: str) -> None:
        self.state_store.save(
            position=self.active_position, symbol=self.symbol, last_candle_ts=self._last_candle_ts,
            extra={"strategy_id": self.strategy_id, "reason": reason},
        )

    def _finalize_closed_position(self, position: PositionState) -> dict:
        report = {
            "strategy_id": self.strategy_id, "symbol": self.symbol, "direction": position.direction,
            "realized_pnl": position.realized_pnl, "realized_weight": position.realized_weight,
            "bars_processed": position.bars_processed, "execution_log": list(position.execution_log),
        }
        log.info(
            "[%s %s] position CLOSED -- realized_pnl=%.6g over %d bar(s), %d execution event(s)",
            self.strategy_id, self.symbol, position.realized_pnl, position.bars_processed, len(position.execution_log),
        )
        self.closed_positions.append(position)
        self.closed_position_reports.append(report)
        self.state_store.clear()
        return report

    # -- unified live bar-close dispatcher ------------------------------------

    def on_candle_close(self, bar: dict) -> DispatchResult:
        """Kept so MarketDataPipeline's inherited `_on_raw_message`/
        `_reconnect_and_gapfill` (which call `self.on_candle_close(...)`)
        keep working unmodified; the actual logic lives in `on_bar_close`,
        the name the Step 8 spec uses."""
        return self.on_bar_close(bar)

    def on_bar_close(self, bar: dict) -> DispatchResult:
        """Strategy-tagged, persistence-backed version of
        MarketDataPipeline.on_candle_close: same buffer/HTF/enrichment/
        update-or-open flow, plus a StateStore.save() at every point the
        spec calls out (new trade opened; any fill/TP/breakeven event;
        position closed -> post-mortem + scoped state file cleared)."""
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

        if self.active_position is not None and not self.active_position.closed:
            enriched = self._enrich_bar(bar, df)
            htf_bar_dict = {"high": symbol_htf_bar["high"], "low": symbol_htf_bar["low"]} if symbol_htf_bar is not None else None
            events = self.active_position.update(enriched, htf_bar=htf_bar_dict)
            result.position_events = events
            if events:
                self._persist(reason="position_event")
            if self.active_position.closed:
                self._finalize_closed_position(self.active_position)
                self.active_position = None
                result.position_closed_this_bar = True
        else:
            htf_df = self.symbol_htf_buffer.to_dataframe()
            btc_df = self.btc_buffer.to_dataframe()
            position, diagnostics = self.trade_manager.open_trade(
                df, btc_df, self.funding_rate, htf_df=htf_df if len(htf_df) else None,
            )
            result.trade_diagnostics = diagnostics
            if position is not None:
                self.active_position = position
                result.trade_opened = True
                log.info(
                    "[%s %s] opened %s position -- levels=%s weights=%s initial_sl=%.6g",
                    self.strategy_id, self.symbol, position.direction,
                    position.ladder.levels, position.ladder.weights, position.initial_sl,
                )
                self._persist(reason="trade_opened")

        return result

    # -- reconnection replay ---------------------------------------------------

    async def _reconnect_and_gapfill(self) -> None:
        """Overrides MarketDataPipeline's simpler REST-refetch-and-redispatch
        with GapFillReplayer (per spec point 4): replay missed candles
        through the active position specifically, backfill the candle
        buffers with the same missed candles so they stay contiguous, and
        persist once if anything changed -- all before this method returns
        and the parent's heartbeat_loop resumes waiting on live streaming.
        """
        self.reconnect_count += 1
        self._warn(f"[{self.strategy_id} {self.symbol}] heartbeat timeout -- gap-fill replay (reconnect #{self.reconnect_count})")
        try:
            report = await self.gap_replayer.replay_missed_bars(self.active_position, self._last_candle_ts)
        except Exception as e:
            self._warn(f"[{self.strategy_id} {self.symbol}] gap-fill replay failed: {e}")
            return

        for c in report.candles:
            if self.ltf_buffer.last() is None or c["ts"] > self.ltf_buffer.last()["ts"]:
                self.ltf_buffer.append(c)
                htf_bar = self._htf_aggregator.ingest(c)
                if htf_bar is not None:
                    self.symbol_htf_buffer.append(htf_bar)

        if report.bars_replayed:
            self._last_candle_ts = report.gap_end_ts
            self._last_candle_close_monotonic = time.monotonic()

        if self.active_position is not None and self.active_position.closed:
            self._finalize_closed_position(self.active_position)
            self.active_position = None
            self._persist(reason="gap_replay_close")
        elif report.events:
            self._persist(reason="gap_replay")

        log.warning("[%s %s] %s", self.strategy_id, self.symbol, report.summary())
