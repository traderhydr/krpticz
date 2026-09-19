"""resilience_manager.py -- production armor for 24/7 execution (KRYPTIC Step 7).

Four independent pieces of protection, usable on their own or together:

    StateStore          -- atomic crash-safe persistence + cold recovery of
                            an active PositionState.
    GapFillReplayer      -- replays missed closed candles through an
                            existing position after a reconnect, producing
                            a post-mortem of anything that happened while
                            disconnected.
    check_clock_drift    -- one-shot exchange clock sync check on boot.
    with_retry           -- exponential-backoff-with-jitter decorator for
                            REST calls, retrying HTTP 429/5xx and common
                            transient network errors.

This module deliberately doesn't reach into data_collector.py's
MarketDataPipeline and wire itself in automatically -- StateStore.save()
and GapFillReplayer.replay_gap() are meant to be called from the exact
integration points the spec names (after every fill/TP/breakeven/close,
and right after a reconnect, respectively), and different deployments may
want different cadences (e.g. save on every event vs. every N seconds).
Leaving that call explicit in the driving code keeps this module testable
in isolation and keeps "when do we persist" a visible decision rather than
a hidden side effect.
"""
from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import httpx

from entry_ladder import EntryLadder
from risk_manager import PositionState

log = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# 1. State persistence & crash recovery
# ---------------------------------------------------------------------------

def _position_to_dict(position: PositionState) -> dict:
    """dataclasses.asdict() already recurses into the nested EntryLadder
    dataclass, so this is the entire serialization step -- every field on
    both dataclasses is already a JSON-safe type (str/float/int/bool/list/
    dict/None) except `opposing_fvg_bounds`, a tuple, which JSON round-trips
    as a list (restored explicitly in `_dict_to_position`)."""
    return dataclasses.asdict(position)


def _dict_to_position(d: dict) -> PositionState:
    d = dict(d)
    ladder = EntryLadder(**d.pop("ladder"))
    if d.get("opposing_fvg_bounds") is not None:
        d["opposing_fvg_bounds"] = tuple(d["opposing_fvg_bounds"])
    return PositionState(ladder=ladder, **d)


class StateStore:
    """Crash-safe JSON persistence for the currently active PositionState.

    Args:
        path: Where to persist state (e.g. "state.json"). A sibling
            "<path>.tmp" is used as the write staging file.

    Call `save()` after every state transition the spec calls out: opening
    a trade, any entry/TP fill, a breakeven migration, or a close. Each
    call fully overwrites the file with the CURRENT state (this is a
    snapshot store, not an append-only log) -- `execution_log` on
    PositionState already accumulates history, so the snapshot includes
    everything needed to reconstruct "what happened" as well as "what's
    open now".
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")

    def save(self, *, position: PositionState | None, symbol: str, last_candle_ts: int | None = None, extra: dict | None = None) -> None:
        """Atomically persist the current state: write to a temp file, fsync
        it, then os.replace() it over the real path. os.replace (like POSIX
        rename(2)) is atomic on a given filesystem -- a reader (including
        this same process reloading after a crash) only ever sees the
        fully-old or fully-new file, never a half-written one, even if the
        process is killed mid-write (the kill can only ever land on the
        .tmp file, which the next boot ignores)."""
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "saved_at_unix_ms": int(time.time() * 1000),
            "symbol": symbol,
            "last_candle_ts": last_candle_ts,
            "position": _position_to_dict(position) if position is not None else None,
            "extra": extra or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(self.tmp_path, self.path)

    def load_raw(self) -> dict | None:
        """The full persisted payload dict, or None if there's no state
        file yet or it's unreadable/corrupt (logged, never raised -- a
        corrupt state file on boot should mean 'start flat', not crash
        the bot before it can even begin trading again)."""
        if not self.path.exists():
            return None
        try:
            with open(self.path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.error("state file at %s is corrupt or unreadable, ignoring: %s", self.path, e)
            return None

    def load_position(self) -> PositionState | None:
        """Cold-recovery entry point: reconstruct the exact PositionState
        that was active when this process last saved, or None if there
        wasn't one (clean shutdown while flat, or no state file at all)."""
        payload = self.load_raw()
        if payload is None or payload.get("position") is None:
            return None
        try:
            return _dict_to_position(payload["position"])
        except (TypeError, ValueError) as e:
            log.error("state file's position payload failed to reconstruct, ignoring: %s", e)
            return None

    def load_last_candle_ts(self) -> int | None:
        payload = self.load_raw()
        if payload is None:
            return None
        return payload.get("last_candle_ts")

    def clear(self) -> None:
        """Persist an explicit flat state (position=None) -- call this once
        a position fully closes, so a crash right after doesn't cold-recover
        a position that's actually already done."""
        payload = self.load_raw() or {}
        self.save(position=None, symbol=payload.get("symbol", ""), last_candle_ts=payload.get("last_candle_ts"))


# ---------------------------------------------------------------------------
# 2. Disconnection gap-fill replay
# ---------------------------------------------------------------------------

@dataclass
class GapReplayReport:
    gap_start_ts: int | None
    gap_end_ts: int | None
    bars_replayed: int = 0
    events: list[dict] = field(default_factory=list)
    position_closed_during_gap: bool = False
    # Raw missed candles themselves (oldest first), for a caller that also
    # wants to backfill its own candle buffer/HTF aggregator with what was
    # missed -- e.g. engine.py's KrypticEngine, which needs its CandleBuffer
    # to stay contiguous across a reconnect, not just its position state.
    candles: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        if self.bars_replayed == 0:
            return "no missed candles to replay"
        kinds = [e["type"] for e in self.events]
        return (
            f"replayed {self.bars_replayed} missed candle(s) "
            f"[{self.gap_start_ts} -> {self.gap_end_ts}]: {len(self.events)} event(s) ({', '.join(kinds) or 'none'})"
            + (" -- POSITION CLOSED DURING OUTAGE" if self.position_closed_during_gap else "")
        )


class GapFillReplayer:
    """Fetches and replays closed candles missed during a disconnect,
    through an existing position, BEFORE live streaming resumes -- so an
    SL/TP that would have fired mid-outage is applied in the correct
    historical order rather than being skipped or only checked against
    whatever the first live candle happens to look like.

    Args:
        exchange: Anything with `async def historical_klines(symbol,
            interval, limit) -> list[dict]` (the same duck-typed interface
            data_collector.py's ExchangeStream uses).
        symbol: Traded pair.
        interval: Execution timeframe.
        fetch_limit: Max candles requested per gap-fill call (default 200,
            BloFin's own per-call cap in this repo's exchanges.py -- a gap
            longer than this needs more than one call, which is the
            caller's concern, not this class's).
    """

    def __init__(self, exchange: Any, symbol: str, interval: str, *, fetch_limit: int = 200) -> None:
        self.exchange = exchange
        self.symbol = symbol
        self.interval = interval
        self.fetch_limit = fetch_limit

    async def replay_gap(self, position: PositionState | None, last_known_ts: int | None) -> GapReplayReport:
        """Fetch every closed candle newer than `last_known_ts` and replay
        them through `position.update()` in chronological order.

        Args:
            position: The position active before the disconnect (None if
                flat -- in that case this only reports what was missed, it
                doesn't try to evaluate new entries; that's the live
                pipeline's job once replay hands control back).
            last_known_ts: The last candle timestamp processed before the
                disconnect (epoch ms). None means "no prior state" -- this
                fetches whatever the exchange returns and replays all of it.

        Returns:
            A GapReplayReport: every bar replayed and every resulting
            execution_log event (SL_HIT, TP_HIT, BREAKEVEN_MOVE,
            RUNNER_EXIT, ...), suitable for a detailed post-mortem log line.
        """
        candles = await self.exchange.historical_klines(self.symbol, self.interval, self.fetch_limit)
        missed = sorted(
            (c for c in candles if last_known_ts is None or int(c["ts"]) > last_known_ts),
            key=lambda c: c["ts"],
        )
        report = GapReplayReport(
            gap_start_ts=missed[0]["ts"] if missed else None,
            gap_end_ts=missed[-1]["ts"] if missed else None,
            candles=missed,
        )
        if not missed:
            return report

        for bar in missed:
            report.bars_replayed += 1
            if position is None or position.closed:
                continue
            events = position.update(bar)
            for e in events:
                report.events.append({**e, "ts": bar["ts"]})
            if position.closed:
                report.position_closed_during_gap = True

        if report.events:
            log.warning("gap-fill replay post-mortem: %s", report.summary())
        else:
            log.info("gap-fill replay: %s", report.summary())
        return report

    # Alias matching engine.py's KrypticEngine (Step 8) naming; identical behavior.
    replay_missed_bars = replay_gap


# ---------------------------------------------------------------------------
# 3. Exchange server clock sync
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClockSyncResult:
    local_time_ms: float
    server_time_ms: float
    round_trip_ms: float
    drift_ms: float
    critical: bool


async def check_clock_drift(exchange: Any, *, max_drift_ms: float = 1000.0) -> ClockSyncResult:
    """Compare local clock to the exchange's server time.

    Args:
        exchange: Anything with `async def server_time(self) -> int`
            (server's current time, epoch ms).
        max_drift_ms: Absolute drift beyond which this logs a critical
            warning (default 1000ms, per spec -- large drift causes
            signature rejection / recvWindow errors on authenticated
            exchange endpoints).

    Returns:
        ClockSyncResult. `drift_ms` is `local_time - server_time` at the
        midpoint of the request (a simple round-trip-compensated estimate:
        the server timestamp is assumed to have been generated halfway
        between when the request was sent and its response received --
        exact NTP-style sync isn't needed here, just enough precision to
        catch a badly-wrong system clock, which is normally off by seconds
        to hours, not milliseconds).
    """
    t0 = time.time() * 1000
    server_ms = float(await exchange.server_time())
    t1 = time.time() * 1000
    round_trip = t1 - t0
    local_mid = (t0 + t1) / 2
    drift = local_mid - server_ms
    critical = abs(drift) > max_drift_ms
    result = ClockSyncResult(local_time_ms=local_mid, server_time_ms=server_ms, round_trip_ms=round_trip, drift_ms=drift, critical=critical)
    if critical:
        log.critical(
            "clock drift %.1fms exceeds %.1fms threshold (local=%.0f server=%.0f) -- "
            "authenticated requests risk signature rejection / recvWindow errors",
            drift, max_drift_ms, local_mid, server_ms,
        )
    else:
        log.info("clock drift check OK: %.1fms (threshold %.1fms)", drift, max_drift_ms)
    return result


# ---------------------------------------------------------------------------
# 4. REST exponential backoff & rate-limit guard
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


def is_retryable_http_error(exc: BaseException) -> bool:
    """Default retry predicate: HTTP 429 or any 5xx, plus the common
    transient network failures httpx raises for a dropped connection or a
    timed-out request -- all conditions worth retrying, as opposed to e.g.
    a 4xx (other than 429), which means "the request itself is wrong" and
    retrying it identically will just fail identically."""
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.RemoteProtocolError))


def with_retry(
    max_retries: int = 5,
    base_delay: float = 1.0,
    *,
    is_retryable: Callable[[BaseException], bool] = is_retryable_http_error,
    on_exhausted: Callable[[BaseException], Any] | None = None,
    _sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    _jitter: Callable[[float, float], float] = random.uniform,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    """Decorator: exponential backoff with jitter for a REST call.

    delay = base_delay * (2 ** attempt) + uniform(0, 0.5), per spec.

    Args:
        max_retries: Retries AFTER the first attempt (default 5 -- so up to
            6 total attempts).
        base_delay: Base backoff seconds (default 1.0).
        is_retryable: Predicate deciding whether a raised exception should
            trigger a retry (default `is_retryable_http_error`: HTTP
            429/5xx + common transient network errors). Anything else
            propagates immediately, unretried.
        on_exhausted: If given, called with the final exception once
            retries are exhausted, and ITS return value is returned instead
            of raising -- an explicit opt-in to "swallow and fall back to
            something" for callers that want it. Default (None) re-raises
            the final exception: "aborts gracefully" here means the retry
            loop itself never crashes the event loop or hangs forever, not
            that a still-failing endpoint is silently treated as success --
            that's the same fail-loud-but-don't-crash convention
            data_collector.py already uses for its own REST calls.
        _sleep / _jitter: Injectable for deterministic tests; leave at their
            defaults in production.

    Usage:
        @with_retry(max_retries=5, base_delay=1.0)
        async def fetch_funding(client, symbol):
            r = await client.get(...)
            r.raise_for_status()
            return r.json()
    """

    def decorator(func: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> _T:
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if not is_retryable(e) or attempt >= max_retries:
                        if attempt >= max_retries and is_retryable(e):
                            log.error("with_retry: %s exhausted %d retries, giving up: %s", func.__name__, max_retries, e)
                            if on_exhausted is not None:
                                return on_exhausted(e)
                        raise
                    delay = base_delay * (2 ** attempt) + _jitter(0, 0.5)
                    log.warning(
                        "with_retry: %s failed (attempt %d/%d): %s -- retrying in %.2fs",
                        func.__name__, attempt + 1, max_retries + 1, e, delay,
                    )
                    await _sleep(delay)
                    attempt += 1

        return wrapper

    return decorator
