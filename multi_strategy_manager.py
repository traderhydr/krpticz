"""multi_strategy_manager.py -- multi-strategy orchestration & hedge mode
harness (KRYPTIC Step 9).

Coordinates multiple strategy engines (KRYPTIC, and -- via the same
duck-typed interface -- ZENITH/GEM once they're wrapped in an equivalent
engine) trading on one exchange account in hedge mode (dual-side position
mode, where a LONG and a SHORT on the same symbol are two independent
position buckets, not mutually exclusive).

Honest scope note: this repo's existing ZENITH/GEM strategies (strategy.py/
gem_strategy.py/bot.py) predate this KRYPTIC track entirely and use a
different architecture (Signal objects posted to Cornix over Telegram, not
a MarketDataPipeline/PositionState engine). Building a real ZenithEngine/
GemEngine wrapper around that architecture is a separate, much larger
undertaking outside this step's scope. Everything below is written against
a minimal duck-typed "strategy engine" interface --
`on_bar_close(bar) -> DispatchResult`-shaped, plus `active_position`,
`state_store`, `ltf_buffer`/`btc_buffer`/`symbol_htf_buffer`, `strategy_id`,
`symbol` -- that `engine.py`'s `KrypticEngine` already satisfies exactly.
`test_multi_strategy.py` uses THREE KrypticEngine instances (under
different `strategy_name`s) to stand in for KRYPTIC/ZENITH/GEM, which is
enough to prove the harness's own coordination logic (isolation, no
collisions, targeted cancellation, margin arbitration) -- it does not, and
cannot, prove anything about ZENITH/GEM's actual trading logic, which this
module never touches.

This module also doesn't touch any previously-shipped file. Where the
spec's hedge-mode state-file naming (`state_<strategy>_<symbol>_<direction>
.json`, note the added direction) differs from `KrypticEngine`'s own
single-direction convention (Step 8), `ManagedStrategyEngine` below
re-points `engine.state_store` to the correctly-scoped path itself, rather
than changing engine.py's constructor.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from resilience_manager import StateStore, check_clock_drift

log = logging.getLogger(__name__)

Direction = str  # "LONG" | "SHORT"

# ---------------------------------------------------------------------------
# 3. clOrdId namespacing & order tagging
# ---------------------------------------------------------------------------

STRATEGY_PREFIXES: dict[str, str] = {"KRYPTIC": "KRP", "ZENITH": "ZEN", "GEM": "GEM"}
POS_SIDE: dict[str, str] = {"LONG": "long", "SHORT": "short"}

_QUOTE_SUFFIXES = ("USDT", "USDC", "BUSD", "USD")


def strategy_prefix(strategy_id: str) -> str:
    """The clOrdId prefix for a strategy_id -- a small known table (matching
    the spec's own examples, KRP/ZEN) with a safe fallback (first 3 letters,
    uppercased) for any strategy this table doesn't already know about."""
    return STRATEGY_PREFIXES.get(strategy_id.upper(), strategy_id.upper()[:3])


def pos_side_for(direction: Direction) -> str:
    """Hedge-mode dual-side parameter for a trade direction: LONG -> "long", SHORT -> "short"."""
    try:
        return POS_SIDE[direction.upper()]
    except KeyError:
        raise ValueError('direction must be "LONG" or "SHORT"') from None


def order_side_for(direction: Direction, *, is_exit: bool) -> str:
    """Exchange order side (buy/sell) for one leg of a hedge-mode order.

    Opening a LONG or closing a SHORT is a buy; opening a SHORT or closing
    a LONG is a sell -- `posSide` (see `pos_side_for`) is what tells the
    exchange WHICH position bucket this order affects; `side` is the
    direction of that specific order, and the two are independent axes in
    dual-side mode (this is exactly why hedge mode needs both fields: a
    "sell" order with posSide=long REDUCES the long book, it does not open
    a short one).
    """
    direction = direction.upper()
    is_long_side = direction == "LONG"
    return "sell" if (is_long_side == is_exit) else "buy"


def short_symbol(symbol: str) -> str:
    """Strip a common quote-currency suffix for a compact clOrdId token
    (BTCUSDT -> BTC), per the spec's own examples."""
    symbol_u = symbol.upper()
    for suffix in _QUOTE_SUFFIXES:
        if symbol_u.endswith(suffix) and len(symbol_u) > len(suffix):
            return symbol_u[: -len(suffix)]
    return symbol_u


def make_cl_ord_id(strategy_id: str, direction: Direction, symbol: str, tier_id: str, timestamp: int) -> str:
    """clOrdId = f"{STRATEGY_PREFIX}_{POS_SIDE[:1]}_{SYMBOL}_{TIER}_{TIMESTAMP}"
    e.g. KRP_L_BTC_E1_1726510000 / ZEN_S_BTC_E2_1726510000 (per spec).

    Deterministic and reconstructible from (strategy_id, direction, symbol,
    tier_id, timestamp) alone -- no side-table needed to look an order back
    up later, which is what lets fills/cancellations be matched without a
    separately-persisted id map.
    """
    side_letter = pos_side_for(direction)[:1].upper()  # "L" | "S"
    return f"{strategy_prefix(strategy_id)}_{side_letter}_{short_symbol(symbol)}_{tier_id}_{int(timestamp)}"


@dataclass(frozen=True)
class OrderPayload:
    """One tagged order, exactly as it would be sent to a hedge-mode exchange endpoint."""

    cl_ord_id: str
    strategy_id: str
    symbol: str
    pos_side: str  # "long" | "short"
    side: str      # "buy" | "sell"
    tier_id: str
    price: float
    size_weight: float  # fraction of intended position size (this codebase's existing convention)
    order_type: str = "limit"

    def to_dict(self) -> dict:
        return {
            "clOrdId": self.cl_ord_id, "strategy_id": self.strategy_id, "symbol": self.symbol,
            "posSide": self.pos_side, "side": self.side, "tier_id": self.tier_id,
            "price": self.price, "size_weight": self.size_weight, "order_type": self.order_type,
        }


# ---------------------------------------------------------------------------
# Shared order book -- the "exchange open-orders list" this account-wide
# harness needs to prove targeted cancellation against.
# ---------------------------------------------------------------------------

class SharedOrderBook:
    """Tracks every currently-OPEN (unfilled) tagged order across every
    strategy/symbol/side, keyed by clOrdId. Real exchanges expose an
    analogous "open orders" list; this is this harness's in-memory
    equivalent, and the thing `cancel_by_strategy_and_side` proves is
    correctly scoped against.
    """

    def __init__(self) -> None:
        self._open: dict[str, OrderPayload] = {}

    def place(self, order: OrderPayload) -> None:
        self._open[order.cl_ord_id] = order

    def fill(self, cl_ord_id: str) -> OrderPayload | None:
        return self._open.pop(cl_ord_id, None)

    def cancel(self, cl_ord_id: str) -> OrderPayload | None:
        return self._open.pop(cl_ord_id, None)

    def cancel_by_strategy_and_side(self, strategy_id: str, direction: Direction, symbol: str | None = None) -> list[OrderPayload]:
        """The targeted-cancellation primitive (per spec point 3): only
        orders whose clOrdId starts with this strategy's own prefix AND
        whose posSide matches `direction` (and, if given, whose symbol
        matches) are removed -- every other strategy's orders, and this
        same strategy's orders on the OPPOSITE side, are left resting
        untouched.
        """
        prefix = strategy_prefix(strategy_id) + "_"
        pos_side = pos_side_for(direction)
        symbol_u = symbol.upper() if symbol else None
        to_cancel = [
            cid for cid, o in self._open.items()
            if cid.startswith(prefix) and o.pos_side == pos_side and (symbol_u is None or o.symbol.upper() == symbol_u)
        ]
        return [self._open.pop(cid) for cid in to_cancel]

    def open_orders(self, *, strategy_id: str | None = None, direction: Direction | None = None, symbol: str | None = None) -> list[OrderPayload]:
        orders = list(self._open.values())
        if strategy_id is not None:
            prefix = strategy_prefix(strategy_id) + "_"
            orders = [o for o in orders if o.cl_ord_id.startswith(prefix)]
        if direction is not None:
            pos_side = pos_side_for(direction)
            orders = [o for o in orders if o.pos_side == pos_side]
        if symbol is not None:
            symbol_u = symbol.upper()
            orders = [o for o in orders if o.symbol.upper() == symbol_u]
        return orders

    def __len__(self) -> int:
        return len(self._open)


# ---------------------------------------------------------------------------
# 4. Shared margin & risk arbiter
# ---------------------------------------------------------------------------

class RiskArbiter:
    """Shared margin/risk gatekeeper across every strategy on one account.

    Args:
        account_equity: Current account equity (the base every percentage
            below is computed against).
        global_risk_ceiling_pct: Max aggregate margin allocation across ALL
            strategies combined, as a percent of equity (default 6.0).
        strategy_budgets_pct: Per-strategy cap, as a percent of equity
            (default {"ZENITH": 35.0, "GEM": 35.0, "KRYPTIC": 30.0}, per spec).
            An unlisted strategy_id has a 0% budget (every request rejected)
            until given one.

    Every check fails safe: a request that would breach EITHER the
    strategy's own budget OR the global ceiling is rejected outright (never
    partially granted), with the reason logged and returned.
    """

    def __init__(
        self,
        *,
        account_equity: float,
        global_risk_ceiling_pct: float = 6.0,
        strategy_budgets_pct: dict[str, float] | None = None,
    ) -> None:
        if account_equity <= 0:
            raise ValueError("account_equity must be positive")
        self.account_equity = account_equity
        self.global_risk_ceiling_pct = global_risk_ceiling_pct
        self.strategy_budgets_pct = {k.upper(): v for k, v in (strategy_budgets_pct or {"ZENITH": 35.0, "GEM": 35.0, "KRYPTIC": 30.0}).items()}
        self._allocated: dict[str, float] = defaultdict(float)

    @property
    def global_ceiling_amount(self) -> float:
        return self.account_equity * self.global_risk_ceiling_pct / 100.0

    def strategy_budget_amount(self, strategy_id: str) -> float:
        return self.account_equity * self.strategy_budgets_pct.get(strategy_id.upper(), 0.0) / 100.0

    def allocated_for(self, strategy_id: str) -> float:
        return self._allocated[strategy_id.upper()]

    def total_allocated(self) -> float:
        return sum(self._allocated.values())

    def request_margin(self, strategy_id: str, required_margin: float) -> tuple[bool, str | None]:
        """Request `required_margin` (in the same currency as account_equity)
        for `strategy_id`. Returns (approved, reason) -- reason is None on
        approval, an explicit human-readable rejection reason otherwise.
        Approval atomically records the allocation; a caller that later
        abandons the trade (e.g. the trade was rejected for an unrelated
        reason after margin was already requested) should call
        `release_margin` to give it back.
        """
        strategy_id = strategy_id.upper()
        if required_margin <= 0:
            return False, "required_margin must be positive"

        budget = self.strategy_budget_amount(strategy_id)
        prospective_strategy_total = self._allocated[strategy_id] + required_margin
        if prospective_strategy_total > budget + 1e-9:
            reason = (
                f"strategy budget exceeded: {strategy_id} would allocate {prospective_strategy_total:.4f}, "
                f"budget is {budget:.4f} ({self.strategy_budgets_pct.get(strategy_id, 0.0)}% of equity)"
            )
            log.warning("[RiskArbiter] margin request REJECTED for %s: %s", strategy_id, reason)
            return False, reason

        prospective_global_total = self.total_allocated() + required_margin
        if prospective_global_total > self.global_ceiling_amount + 1e-9:
            reason = (
                f"global portfolio risk ceiling exceeded: total would be {prospective_global_total:.4f}, "
                f"ceiling is {self.global_ceiling_amount:.4f} ({self.global_risk_ceiling_pct}% of equity)"
            )
            log.warning("[RiskArbiter] margin request REJECTED for %s: %s", strategy_id, reason)
            return False, reason

        self._allocated[strategy_id] += required_margin
        log.info("[RiskArbiter] margin request APPROVED for %s: %.4f (strategy total now %.4f/%.4f, global %.4f/%.4f)",
                  strategy_id, required_margin, self._allocated[strategy_id], budget, self.total_allocated(), self.global_ceiling_amount)
        return True, None

    def release_margin(self, strategy_id: str, amount: float) -> None:
        strategy_id = strategy_id.upper()
        self._allocated[strategy_id] = max(0.0, self._allocated[strategy_id] - amount)


# ---------------------------------------------------------------------------
# 5. Structured telemetry
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    ts: float
    strategy_id: str
    symbol: str
    kind: str
    detail: dict
    pos_side: str | None = None


class Telemetry:
    """Structured, queryable event log spanning every registered strategy --
    setup approval/rejection, tier fills, VWAP updates, breakeven
    migrations, ghost-order cancellations, and dynamic runner exits, all
    tagged with strategy_id/symbol/pos_side (per spec point 5)."""

    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit(self, strategy_id: str, symbol: str, kind: str, detail: dict, *, pos_side: str | None = None) -> TelemetryEvent:
        event = TelemetryEvent(ts=time.time(), strategy_id=strategy_id.upper(), symbol=symbol.upper(), kind=kind, detail=detail, pos_side=pos_side)
        self.events.append(event)
        log.info("[TELEMETRY] %s %s %s %s: %s", event.strategy_id, event.symbol, pos_side or "-", kind, detail)
        return event

    def for_strategy(self, strategy_id: str) -> list[TelemetryEvent]:
        return [e for e in self.events if e.strategy_id == strategy_id.upper()]

    def for_symbol(self, symbol: str) -> list[TelemetryEvent]:
        return [e for e in self.events if e.symbol == symbol.upper()]

    def of_kind(self, kind: str) -> list[TelemetryEvent]:
        return [e for e in self.events if e.kind == kind]


# ---------------------------------------------------------------------------
# 1. Shared data bus
# ---------------------------------------------------------------------------

class SharedDataBus:
    """One exchange connection (REST warmup + live stream) per symbol,
    broadcasting every closed candle to every strategy engine registered
    for that symbol -- eliminating N-per-strategy duplicate WebSocket
    connections and redundant REST warmup calls for the same market data
    (per spec point 1).

    Each registered engine still keeps its OWN CandleBuffer/HTFAggregator
    (different strategies may want different EMA/ATR/SuperTrend lengths
    computed from that data) -- only the network fetch itself is shared;
    `warmup()` seeds every engine's own buffers directly from one shared
    fetch instead of letting each engine make its own REST call.
    """

    def __init__(
        self,
        *,
        exchange: Any,
        ltf_interval: str = "15m",
        btc_htf_interval: str = "4h",
        warmup_bars: int = 300,
    ) -> None:
        self.exchange = exchange
        self.ltf_interval = ltf_interval
        self.btc_htf_interval = btc_htf_interval
        self.warmup_bars = warmup_bars
        self._engines: dict[str, list[Any]] = defaultdict(list)
        self._ltf_cache: dict[str, list[dict]] = {}
        self._btc_cache: dict[str, list[dict]] = {}
        self._funding_cache: dict[str, float | None] = {}
        self.warnings: list[str] = []

    def register(self, symbol: str, engine: Any) -> None:
        """Register a strategy engine (anything with `on_bar_close(bar)`,
        `ltf_buffer`, `btc_buffer`) to receive every closed candle for `symbol`."""
        self._engines[symbol.upper()].append(engine)

    def unregister(self, symbol: str, engine: Any) -> None:
        engines = self._engines.get(symbol.upper())
        if engines and engine in engines:
            engines.remove(engine)

    def _warn(self, msg: str) -> None:
        log.warning(msg)
        self.warnings.append(msg)

    async def warmup(self, symbol: str) -> None:
        """ONE shared REST fetch (LTF + BTC) for `symbol`, then copies the
        result into every currently-registered engine's own buffers --
        each engine still runs whatever engine-specific setup it needs
        (e.g. KrypticEngine's own crash recovery / clock-drift check;
        `ManagedStrategyEngine.warmup` below drives that), it just never
        hits the exchange again to get data this call already fetched."""
        symbol_u = symbol.upper()
        if symbol_u not in self._ltf_cache:
            try:
                self._ltf_cache[symbol_u] = list(await self.exchange.historical_klines(symbol_u, self.ltf_interval, self.warmup_bars))
            except Exception as e:
                self._warn(f"shared historical klines fetch failed for {symbol_u}: {e}")
                self._ltf_cache[symbol_u] = []
        if "BTCUSDT" not in self._btc_cache:
            try:
                self._btc_cache["BTCUSDT"] = list(await self.exchange.historical_klines("BTCUSDT", self.btc_htf_interval, self.warmup_bars))
            except Exception as e:
                self._warn(f"shared historical klines fetch failed for BTCUSDT: {e}")
                self._btc_cache["BTCUSDT"] = []
        if symbol_u not in self._funding_cache:
            try:
                self._funding_cache[symbol_u] = await self.exchange.funding_rate(symbol_u)
            except Exception as e:
                self._warn(f"shared funding rate fetch failed for {symbol_u}: {e}")
                self._funding_cache[symbol_u] = None

        for engine in self._engines[symbol_u]:
            self.seed_engine(symbol_u, engine)

    def seed_engine(self, symbol: str, engine: Any) -> None:
        """Copy this bus's already-fetched shared candles/funding into one
        engine's own buffers. Safe to call more than once (e.g. re-seeding
        a newly-registered engine after the bus already warmed up) --
        CandleBuffer's own duplicate/out-of-order guard is used to skip
        bars the engine already has."""
        symbol_u = symbol.upper()
        for c in self._ltf_cache.get(symbol_u, []):
            self._safe_append(engine.ltf_buffer, c)
        for c in self._btc_cache.get("BTCUSDT", []):
            self._safe_append(engine.btc_buffer, c)
        if symbol_u in self._funding_cache:
            engine.funding_rate = self._funding_cache[symbol_u]
        last = engine.ltf_buffer.last()
        if last is not None:
            engine._last_candle_ts = last["ts"]
            engine._last_candle_close_monotonic = time.monotonic()

    @staticmethod
    def _safe_append(buffer: Any, candle: dict) -> None:
        try:
            buffer.append(candle)
        except ValueError:
            pass  # engine already has this bar

    def on_candle_close(self, symbol: str, bar: dict) -> None:
        """Fan out one closed candle to every engine registered for `symbol`.
        This is the single ingestion point a real websocket handler (or a
        MockExchangeStream-driven test loop) calls -- see engine.py's own
        MarketDataPipeline for why this repo has no real websocket client
        yet."""
        symbol_u = symbol.upper()
        for engine in list(self._engines.get(symbol_u, [])):
            try:
                engine.on_bar_close(bar)
            except Exception:
                log.exception("strategy engine %r raised while handling a shared bar for %s", engine, symbol_u)


# ---------------------------------------------------------------------------
# Per-engine coordination wrapper
# ---------------------------------------------------------------------------

class ManagedStrategyEngine:
    """Wraps one strategy engine (e.g. a KrypticEngine) with hedge-mode
    state-file scoping, margin arbitration, order tagging/cancellation, and
    telemetry -- without modifying the wrapped engine's own class.

    Args:
        engine: The underlying engine (duck-typed: needs `on_bar_close`,
            `active_position`, `state_store`/`state_path`, `strategy_id`,
            `symbol`, `ltf_buffer`, `btc_buffer`).
        arbiter: Shared RiskArbiter this engine's new trades must clear.
        order_book: Shared SharedOrderBook this engine's tiered orders are
            registered into / cancelled from.
        telemetry: Shared Telemetry sink.
        state_dir: Directory for this engine's hedge-mode-scoped state files.
        margin_estimator: `(position, diagnostics) -> required_margin`.
            Defaults to a flat `account_equity * risk_per_trade_pct`
            proxy (this codebase has no real position-sizing/leverage model
            to derive an exact margin figure from) -- pass a real one if
            you have a better estimate.
        risk_per_trade_pct: Used by the default margin_estimator (percent
            of the arbiter's account_equity, default 1.0).
    """

    def __init__(
        self,
        *,
        engine: Any,
        arbiter: RiskArbiter,
        order_book: SharedOrderBook,
        telemetry: Telemetry,
        state_dir: str | Path = ".",
        margin_estimator: Callable[[Any, dict], float] | None = None,
        risk_per_trade_pct: float = 1.0,
    ) -> None:
        self.engine = engine
        self.arbiter = arbiter
        self.order_book = order_book
        self.telemetry = telemetry
        self.state_dir = Path(state_dir)
        self.strategy_id = engine.strategy_id
        self.symbol = engine.symbol
        self.risk_per_trade_pct = risk_per_trade_pct
        self._margin_estimator = margin_estimator or self._default_margin_estimator
        self._allocated_margin: float = 0.0
        # Set by _handle_trade_opened when the harness itself drives a
        # position open; stays None if a position was attached directly
        # (e.g. injected in a test, or recovered mid-way through a prior
        # run without going through this harness) -- order-book fill/cancel
        # translation is skipped gracefully in that case, since there's no
        # way to reconstruct a clOrdId for orders this harness never placed.
        self._entry_bar_ts: int | None = None
        # Placeholder path used only until a position's direction is known
        # (see `_rescope_for_direction`) -- nothing meaningful ever persists
        # here for more than one call.
        self._rescope_for_direction("pending")

    # -- forwarding properties: let SharedDataBus treat this wrapper exactly
    # like the raw engine it wraps for warmup/seeding purposes (buffers
    # live on `self.engine`, not on the wrapper itself). --
    @property
    def ltf_buffer(self) -> Any:
        return self.engine.ltf_buffer

    @property
    def btc_buffer(self) -> Any:
        return self.engine.btc_buffer

    @property
    def funding_rate(self) -> float | None:
        return self.engine.funding_rate

    @funding_rate.setter
    def funding_rate(self, value: float | None) -> None:
        self.engine.funding_rate = value

    @property
    def active_position(self) -> Any:
        return self.engine.active_position

    @property
    def _last_candle_ts(self) -> int | None:
        return self.engine._last_candle_ts

    @_last_candle_ts.setter
    def _last_candle_ts(self, value: int | None) -> None:
        self.engine._last_candle_ts = value

    @property
    def _last_candle_close_monotonic(self) -> float | None:
        return self.engine._last_candle_close_monotonic

    @_last_candle_close_monotonic.setter
    def _last_candle_close_monotonic(self, value: float | None) -> None:
        self.engine._last_candle_close_monotonic = value

    def _default_margin_estimator(self, position: Any, diagnostics: dict) -> float:
        return self.arbiter.account_equity * self.risk_per_trade_pct / 100.0

    def _rescope_for_direction(self, direction: str) -> Path:
        """Point `self.engine.state_store` at
        state_<strategy>_<symbol>_<direction>.json (per spec point 5's
        hedge-mode-aware naming, which adds a direction segment
        KrypticEngine's own single-direction Step 8 convention doesn't
        have) without modifying engine.py."""
        path = self.state_dir / f"state_{self.strategy_id.lower()}_{self.symbol.lower()}_{direction.lower()}.json"
        self.engine.state_path = path
        self.engine.state_store = StateStore(path)
        return path

    def _clear_stale_pending_state(self) -> None:
        """`KrypticEngine.on_bar_close` (engine.py, Step 8, unmodified here)
        persists its OWN 'trade_opened' snapshot to whatever `state_store`
        it currently holds the instant a trade opens -- which, for a
        freshly-opened trade, is still this wrapper's placeholder "pending"
        file (see `__init__`), since `_handle_trade_opened` below only
        rescopes to the correctly-direction-scoped file AFTER that engine
        call already returned. Left alone, that "pending" file keeps
        showing a real (non-None) position forever, even once the actual
        position is correctly persisted at the newly-rescoped path -- a
        false "there's an active position here" read for anything that
        later inspects it between trades (crash recovery never reads it by
        mistake, since a subsequent trade rescopes state_store before any
        read happens, but the stale file is still on-disk state corruption
        by any other definition). Call this right after rescoping to the
        real direction so the placeholder file is put back to flat."""
        pending_path = self.state_dir / f"state_{self.strategy_id.lower()}_{self.symbol.lower()}_pending.json"
        if pending_path.exists():
            StateStore(pending_path).clear()

    async def warmup(self) -> None:
        """Deliberately does NOT call `self.engine.warmup()` -- that would
        make its own REST fetch (redundant: `SharedDataBus.warmup` already
        seeded this engine's buffers/funding from ONE shared fetch) and then
        crash trying to re-append candles its buffer already has. This
        reimplements only the two per-engine pieces a shared fetch can't
        cover: the clock-drift check and this engine's OWN crash recovery
        (its own scoped state file -- see `_rescope_for_direction`)."""
        if hasattr(self.engine.exchange, "server_time"):
            try:
                self.engine.last_clock_sync = await check_clock_drift(self.engine.exchange, max_drift_ms=self.engine.clock_drift_max_ms)
            except Exception as e:
                self.engine._warn(f"[{self.strategy_id} {self.symbol}] clock drift check failed: {e}")
        else:
            self.engine._warn(f"[{self.strategy_id} {self.symbol}] exchange has no server_time() -- skipping clock drift check")

        recovered = self.engine.state_store.load_position()
        if recovered is not None:
            self.engine.active_position = recovered
            saved_ts = self.engine.state_store.load_last_candle_ts()
            if saved_ts is not None and (self.engine._last_candle_ts is None or saved_ts > self.engine._last_candle_ts):
                self.engine._last_candle_ts = saved_ts
            self._rescope_for_direction(recovered.direction)
            log.warning(
                "[%s %s] crash recovery: restored an active %s position (open_size=%.4f, realized_pnl=%.6g, %d prior execution event(s))",
                self.strategy_id, self.symbol, recovered.direction, recovered.open_size, recovered.realized_pnl, len(recovered.execution_log),
            )

    def on_bar_close(self, bar: dict) -> Any:
        had_position_before = self.engine.active_position is not None
        result = self.engine.on_bar_close(bar)

        if result.trade_diagnostics is not None and not result.trade_opened and not had_position_before:
            # A setup was evaluated and didn't result in a trade -- only
            # worth a telemetry line when it's a genuine rejection (not
            # "still flat, nothing to evaluate yet" noise); NEUTRAL bias is
            # the overwhelmingly common case and would drown the log.
            reason = result.trade_diagnostics.get("reason")
            if reason and reason != "directional bias is NEUTRAL":
                self.telemetry.emit(self.strategy_id, self.symbol, "SETUP_REJECTED", {"reason": reason})

        if result.trade_opened:
            self._handle_trade_opened(result)
        elif had_position_before and self.engine.active_position is not None:
            self._handle_position_events(result)
        elif had_position_before and self.engine.active_position is None:
            self._handle_position_closed(result)

        return result

    def _handle_trade_opened(self, result: Any) -> None:
        position = self.engine.active_position
        direction = position.direction
        diagnostics = result.trade_diagnostics or {}

        required_margin = self._margin_estimator(position, diagnostics)
        approved, reason = self.arbiter.request_margin(self.strategy_id, required_margin)
        if not approved:
            log.warning("[%s %s] trade opened by the engine but REJECTED by RiskArbiter (%s) -- discarding", self.strategy_id, self.symbol, reason)
            self.telemetry.emit(self.strategy_id, self.symbol, "SETUP_REJECTED", {"reason": f"margin: {reason}"}, pos_side=pos_side_for(direction))
            self.engine.active_position = None
            self.engine.state_store.clear()
            return

        self._allocated_margin = required_margin
        self._rescope_for_direction(direction)
        self.engine._persist(reason="trade_opened")  # re-persist to the now-correctly-scoped file
        self._clear_stale_pending_state()

        self.telemetry.emit(self.strategy_id, self.symbol, "SETUP_APPROVED", {
            "direction": direction, "levels": list(position.ladder.levels), "required_margin": required_margin,
        }, pos_side=pos_side_for(direction))

        bar_ts = int(result.bar["ts"])
        for tier_i, (level, weight, filled) in enumerate(zip(position.ladder.levels, position.ladder.weights, position.ladder.fills)):
            if filled:
                continue  # already executed at open time (e.g. a market order) -- nothing resting to track
            if weight <= 0:
                continue  # a 0-weight tier (e.g. the calibrated ladder's Entry 4) has nothing to place a resting order for
            order = OrderPayload(
                cl_ord_id=make_cl_ord_id(self.strategy_id, direction, self.symbol, f"E{tier_i + 1}", bar_ts),
                strategy_id=self.strategy_id, symbol=self.symbol, pos_side=pos_side_for(direction),
                side=order_side_for(direction, is_exit=False), tier_id=f"E{tier_i + 1}",
                price=level, size_weight=weight,
            )
            self.order_book.place(order)
        self._entry_bar_ts = bar_ts

    def _handle_position_events(self, result: Any) -> None:
        position = self.engine.active_position
        direction = position.direction
        pos_side = pos_side_for(direction)

        cancel_events: list = []
        for event in result.position_events:
            kind = event["type"]
            if kind == "ENTRY_FILL":
                if self._entry_bar_ts is not None:
                    tier_id = f"E{event['tier_index'] + 1}"
                    cl_ord_id = make_cl_ord_id(self.strategy_id, direction, self.symbol, tier_id, self._entry_bar_ts)
                    self.order_book.fill(cl_ord_id)
                self.telemetry.emit(self.strategy_id, self.symbol, "TIER_FILL", event, pos_side=pos_side)
                self.telemetry.emit(self.strategy_id, self.symbol, "VWAP_UPDATE", {"filled_vwap": position.weighted_avg_entry}, pos_side=pos_side)
            elif kind == "BREAKEVEN_MOVE":
                self.telemetry.emit(self.strategy_id, self.symbol, "BREAKEVEN_MOVE", event, pos_side=pos_side)
            elif kind == "CANCEL":
                # A single bar can cancel several ladder tiers at once (e.g. TP2
                # firing cancels every unfilled tier below it), producing one
                # "CANCEL" execution_log entry per tier. Collect them and issue a
                # single order-book cancellation + telemetry event per bar rather
                # than one per tier, since the order book has no per-tier
                # granularity to cancel against and a second call would find
                # nothing left to remove.
                cancel_events.append(event)
            elif kind == "RUNNER_EXIT":
                # Step 11.0's ATR chandelier-trail runner tier firing.
                self.telemetry.emit(self.strategy_id, self.symbol, "RUNNER_EXIT", event, pos_side=pos_side)
            elif kind == "TIME_DECAY_EXIT":
                self.telemetry.emit(self.strategy_id, self.symbol, "TIME_DECAY_EXIT", event, pos_side=pos_side)

        if cancel_events:
            cancelled = self.order_book.cancel_by_strategy_and_side(self.strategy_id, direction, self.symbol)
            self.telemetry.emit(self.strategy_id, self.symbol, "GHOST_ORDER_CANCEL", {
                "engine_reported": cancel_events, "order_book_cancelled": [o.cl_ord_id for o in cancelled],
            }, pos_side=pos_side)

    def _handle_position_closed(self, result: Any) -> None:
        # The just-closed position is no longer on self.engine.active_position;
        # its final report is on engine.closed_position_reports[-1].
        report = self.engine.closed_position_reports[-1] if self.engine.closed_position_reports else {}
        direction = report.get("direction")
        if self._allocated_margin:
            self.arbiter.release_margin(self.strategy_id, self._allocated_margin)
            self._allocated_margin = 0.0
        if direction:
            leftover = self.order_book.cancel_by_strategy_and_side(self.strategy_id, direction, self.symbol)
            if leftover:
                self.telemetry.emit(self.strategy_id, self.symbol, "GHOST_ORDER_CANCEL", {
                    "reason": "position closed", "order_book_cancelled": [o.cl_ord_id for o in leftover],
                }, pos_side=pos_side_for(direction))
        self.telemetry.emit(self.strategy_id, self.symbol, "POSITION_CLOSED", report, pos_side=pos_side_for(direction) if direction else None)
        self._rescope_for_direction("pending")


# ---------------------------------------------------------------------------
# Top-level harness
# ---------------------------------------------------------------------------

class MultiStrategyManager:
    """Top-level multi-strategy, hedge-mode-safe harness: owns one
    SharedDataBus, one RiskArbiter, one SharedOrderBook, and one Telemetry
    sink, and wraps every registered strategy engine in a
    ManagedStrategyEngine bound to all four.
    """

    def __init__(
        self,
        *,
        exchange: Any,
        arbiter: RiskArbiter,
        state_dir: str | Path = ".",
        ltf_interval: str = "15m",
        btc_htf_interval: str = "4h",
        warmup_bars: int = 300,
    ) -> None:
        self.bus = SharedDataBus(exchange=exchange, ltf_interval=ltf_interval, btc_htf_interval=btc_htf_interval, warmup_bars=warmup_bars)
        self.arbiter = arbiter
        self.order_book = SharedOrderBook()
        self.telemetry = Telemetry()
        self.state_dir = Path(state_dir)
        self.managed_engines: list[ManagedStrategyEngine] = []

    def register_engine(self, engine: Any, *, margin_estimator: Callable[[Any, dict], float] | None = None, risk_per_trade_pct: float = 1.0) -> ManagedStrategyEngine:
        """Register one strategy engine for `engine.symbol`. Returns the
        ManagedStrategyEngine wrapper (also kept in `self.managed_engines`)."""
        managed = ManagedStrategyEngine(
            engine=engine, arbiter=self.arbiter, order_book=self.order_book, telemetry=self.telemetry,
            state_dir=self.state_dir, margin_estimator=margin_estimator, risk_per_trade_pct=risk_per_trade_pct,
        )
        self.managed_engines.append(managed)
        self.bus.register(engine.symbol, managed)
        return managed

    async def warmup_all(self) -> None:
        symbols = {m.symbol for m in self.managed_engines}
        for symbol in symbols:
            await self.bus.warmup(symbol)
        for managed in self.managed_engines:
            await managed.warmup()

    def on_candle_close(self, symbol: str, bar: dict) -> None:
        self.bus.on_candle_close(symbol, bar)
