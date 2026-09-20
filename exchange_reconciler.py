"""exchange_reconciler.py -- shadow-mode reconciliation of RiskGuard's
simulated trades against REAL Binance USD-M futures fills.

Why this exists: risk_guard.py's own module docstring is explicit that it
"never touches an exchange account" -- it replays each posted signal
forward against candles and simulates a fill/close outcome purely to
self-throttle new signals. That simulation assumes perfect fills at the
quoted price, no slippage, no fees, and has zero visibility into what
Cornix (which reads the same posted signal and executes for real) actually
did on the account. This module closes that visibility gap, in shadow mode
only: it reads real fills with a READ-ONLY Binance API key, compares them
against each open RiskGuard trade, and logs the discrepancy -- WITHOUT
changing risk.equity, risk.peak, or risk.paused. Nothing here can affect
whether bot.py posts a signal; it is purely an audit trail so a human (or
a later, explicitly-requested change) can see how far the simulation
actually diverges from reality before trusting it to override anything.

Attribution is the hard part, not the HTTP calls: Binance has no concept
of "which Cornix signal" a fill belongs to, and if two of OUR OWN signals
are ever open on the same (symbol, side) at once, their real fills are
genuinely indistinguishable on the exchange (hedge mode: same symbol +
positionSide bucket; one-way mode: the same net position outright). This
module refuses to guess in that case -- see `attribution_status()`. As of
the bot.py change that shipped alongside this module, `RiskGuard.can_open()`
already blocks a second same-(symbol, side) signal from any engine while
one is open specifically so this almost never happens; the ambiguity check
here is a defensive backstop, not the primary mechanism.

Price-based R, not dollar PnL: rather than depend on Binance's realized-PnL
income endpoint (which needs the real position size to convert into an
R-multiple, and this bot has no idea what quantity Cornix's own risk%
sizing chose), real R is computed the same scale-invariant way
risk_guard.py's own simulation already does it: `(exit_price - eavg) /
risk_px`, using only real fill PRICES. That makes it directly comparable
to the simulated R already stored on the trade, and never needs to know
the real position's dollar size at all.

Known limitations (shadow mode only, so none of these can affect live
signal posting -- they only limit how much the discrepancy log can tell
you):
  - Written from Binance's public API docs, not tested against a live
    account -- this sandbox's network policy blocks outbound access to
    Binance, the same limitation backtest.py's own docstring documents.
    Endpoint paths/params/response shapes may need adjustment; run
    `verify_read_only_permissions()` and a manual smoke test against your
    own account before trusting this in shadow mode.
  - Assumes the real position was flat immediately before this trade's
    open timestamp (guaranteed going forward by the can_open() guard, but
    NOT guaranteed for whatever's already open on your account the first
    time you turn this on, or for manual trading on the same symbol
    outside the bot). No attempt is made to reconstruct pre-existing
    position history; a mismatched starting state just produces a
    confusing discrepancy entry, not a crash.
  - Only reconciles entry fills and a final close; doesn't attempt to
    track partial-TP-by-partial-TP granularity the way the simulation's
    own tp_idx bookkeeping does.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import httpx

log = logging.getLogger("zenith.reconcile")

FAPI_BASE = "https://fapi.binance.com"
SAPI_BASE = "https://api.binance.com"


class PermissionError_(Exception):
    """Raised when the configured API key is not safely read-only."""


@dataclass
class RealFill:
    side: str  # "BUY" | "SELL"
    price: float
    qty: float
    time_ms: int
    position_side: str  # "LONG" | "SHORT" | "BOTH"


@dataclass
class ReconcileResult:
    key: str
    symbol: str
    side: str
    engine: str
    status: str  # "AMBIGUOUS" | "NO_REAL_FILLS_YET" | "OPEN" | "CLOSED" | "FETCH_FAILED"
    note: str = ""
    sim_eavg: float | None = None
    sim_fill_pct: float | None = None
    sim_r: float | None = None
    real_eavg: float | None = None
    real_exit: float | None = None
    real_r: float | None = None

    def discrepancy(self) -> dict | None:
        """None if there's nothing comparable yet (still open, or no real
        fills matched); otherwise the deltas worth a human's attention."""
        if self.real_eavg is None:
            return None
        out: dict = {}
        if self.sim_eavg is not None:
            out["eavg_delta_pct"] = round((self.real_eavg - self.sim_eavg) / self.sim_eavg * 100.0, 4)
        if self.real_r is not None and self.sim_r is not None:
            out["r_delta"] = round(self.real_r - self.sim_r, 4)
        return out or None


def _sign(secret: str, query: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


class BinanceSignedClient:
    """Minimal signed-request client for Binance's READ-ONLY account/trade
    endpoints. Deliberately separate from exchanges.py's public, unsigned
    market-data client -- a signed request against real account data is a
    different trust boundary than an unauthenticated klines fetch every
    other engine depends on, and keeping them in different modules means a
    bug here can't reach that shared path.

    `api_key`/`api_secret` should be a Binance key with ONLY "Enable
    Reading" checked -- never the trade-enabled key you gave Cornix. Call
    `verify_read_only()` once at startup and refuse to proceed if it
    raises; this class does not enforce that itself so callers can choose
    how strict to be (see bot.py's wiring).
    """

    def __init__(self, api_key: str, api_secret: str, *, recv_window: int = 10_000) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = recv_window
        self._server_time_offset_ms = 0

    async def _sync_clock(self, client: httpx.AsyncClient) -> None:
        try:
            r = await client.get(f"{FAPI_BASE}/fapi/v1/time", timeout=10)
            r.raise_for_status()
            server_ms = int(r.json()["serverTime"])
            self._server_time_offset_ms = server_ms - int(time.time() * 1000)
        except Exception as e:
            log.warning("reconciler: clock sync failed, using local time (%s)", e)

    async def _signed_get(self, client: httpx.AsyncClient, base: str, path: str, params: dict) -> dict | list:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000) + self._server_time_offset_ms
        params["recvWindow"] = self.recv_window
        query = urlencode(params)
        query += f"&signature={_sign(self.api_secret, query)}"
        r = await client.get(f"{base}{path}?{query}", headers={"X-MBX-APIKEY": self.api_key}, timeout=20)
        r.raise_for_status()
        return r.json()

    async def verify_read_only(self, client: httpx.AsyncClient) -> dict:
        """Fetch /sapi/v1/account/apiRestrictions and raise PermissionError_
        if this key can withdraw or trade. `enableReading` must be the only
        thing meaningfully on. Note: written from public docs, not verified
        against a live key -- see module docstring's known-limitations."""
        await self._sync_clock(client)
        data = await self._signed_get(client, SAPI_BASE, "/sapi/v1/account/apiRestrictions", {})
        if not isinstance(data, dict):
            raise PermissionError_(f"unexpected apiRestrictions response shape: {data!r}")
        if data.get("enableWithdrawals"):
            raise PermissionError_("this API key has withdrawals enabled -- use a read-only key instead")
        if data.get("enableSpotAndMarginTrading") or data.get("enableMargin"):
            raise PermissionError_("this API key has spot/margin trading enabled -- use a read-only key instead")
        if not data.get("enableReading", True):
            raise PermissionError_("this API key does not have reading enabled")
        return data

    async def user_trades(self, client: httpx.AsyncClient, symbol: str, start_ms: int) -> list[RealFill]:
        raw = await self._signed_get(
            client, FAPI_BASE, "/fapi/v1/userTrades",
            {"symbol": symbol, "startTime": start_ms, "limit": 1000},
        )
        out = []
        for t in raw or []:
            out.append(RealFill(
                side=t["side"], price=float(t["price"]), qty=float(t["qty"]),
                time_ms=int(t["time"]), position_side=t.get("positionSide", "BOTH"),
            ))
        out.sort(key=lambda f: f.time_ms)
        return out

    async def position_risk(self, client: httpx.AsyncClient, symbol: str) -> list[dict]:
        raw = await self._signed_get(client, FAPI_BASE, "/fapi/v2/positionRisk", {"symbol": symbol})
        return raw if isinstance(raw, list) else []


def attribution_status(trades: dict, symbol: str, side: str) -> str:
    """"NONE" | "CONFIDENT" | "AMBIGUOUS" -- how many open RiskGuard trades
    (from ANY engine) currently claim this exact (symbol, side). More than
    one means a real fill on this symbol+side can't be attributed to
    either specific trade -- see module docstring."""
    count = sum(1 for t in trades.values() if t.get("symbol") == symbol and t.get("side") == side)
    if count == 0:
        return "NONE"
    if count == 1:
        return "CONFIDENT"
    return "AMBIGUOUS"


def _weighted_avg(fills: list[RealFill]) -> float | None:
    total_qty = sum(f.qty for f in fills)
    if total_qty <= 0:
        return None
    return sum(f.price * f.qty for f in fills) / total_qty


def _split_increasing_reducing(fills: list[RealFill], side: str, position_side: str) -> tuple[list[RealFill], list[RealFill]]:
    """Hedge mode: a LONG positionSide position is increased by BUYs and
    reduced by SELLs; a SHORT positionSide position is increased by SELLs
    and reduced by BUYs. Filters to this trade's own positionSide first
    (hedge mode keeps LONG/SHORT as separate account-level buckets)."""
    relevant = [f for f in fills if f.position_side == position_side]
    increasing_side = "BUY" if side == "LONG" else "SELL"
    increasing = [f for f in relevant if f.side == increasing_side]
    reducing = [f for f in relevant if f.side != increasing_side]
    return increasing, reducing


class ExchangeReconciler:
    """Shadow-mode reconciliation loop. `run_cycle()` is meant to be
    called on the same cadence as RiskGuard.refresh() -- it never mutates
    `risk`, only reads `risk.trades` and returns/logs what it found."""

    def __init__(
        self,
        client_factory: BinanceSignedClient,
        *,
        position_mode: str = "hedge",
        log_path: Path | None = None,
    ) -> None:
        if position_mode != "hedge":
            # One-way mode collapses same-symbol positions across BOTH
            # directions into one bucket -- attribution_status()'s
            # (symbol, side) grouping isn't sufficient there. Not
            # implemented; fail loud rather than silently mismatch fills.
            raise NotImplementedError("ExchangeReconciler currently only supports position_mode='hedge'")
        self.signer = client_factory
        self.position_mode = position_mode
        self.log_path = log_path or Path(__file__).with_name("reconcile_log.jsonl")

    async def _reconcile_one(self, client: httpx.AsyncClient, key: str, tr: dict) -> ReconcileResult:
        symbol, side, engine = tr["symbol"], tr["side"], tr.get("engine", "?")
        base = ReconcileResult(key=key, symbol=symbol, side=side, engine=engine, status="FETCH_FAILED")
        try:
            open_ts = int(tr.get("last_ts") or 0) or int(time.time() * 1000) - 3_600_000
            fills = await self.signer.user_trades(client, symbol, open_ts)
            increasing, reducing = _split_increasing_reducing(fills, side, side)
        except Exception as e:
            base.note = f"fetch failed: {e}"
            return base

        real_eavg = _weighted_avg(increasing)
        if real_eavg is None:
            base.status = "NO_REAL_FILLS_YET"
            base.sim_eavg = tr.get("eavg")
            base.sim_fill_pct = tr.get("fill_pct")
            return base

        sign = 1.0 if side == "LONG" else -1.0
        risk_px = float(tr.get("risk_px") or 0.0)
        real_exit = _weighted_avg(reducing)

        result = ReconcileResult(
            key=key, symbol=symbol, side=side, engine=engine,
            status="OPEN", real_eavg=real_eavg,
            sim_eavg=tr.get("eavg"), sim_fill_pct=tr.get("fill_pct"), sim_r=tr.get("r_realized"),
        )
        if real_exit is not None and risk_px > 0:
            try:
                positions = await self.signer.position_risk(client, symbol)
                flat = all(abs(float(p.get("positionAmt", 0))) < 1e-9 for p in positions if p.get("positionSide") == side) if positions else False
            except Exception as e:
                flat = False
                result.note = f"position_risk check failed, treating as still open: {e}"
            if flat:
                result.status = "CLOSED"
                result.real_exit = real_exit
                result.real_r = (real_exit - real_eavg) / risk_px * sign
        return result

    async def run_cycle(self, client: httpx.AsyncClient, risk) -> list[ReconcileResult]:
        results: list[ReconcileResult] = []
        symbol_sides: dict[tuple[str, str], list[str]] = {}
        for key, tr in risk.trades.items():
            symbol_sides.setdefault((tr["symbol"], tr["side"]), []).append(key)

        for (symbol, side), keys in symbol_sides.items():
            if len(keys) > 1:
                for key in keys:
                    tr = risk.trades[key]
                    results.append(ReconcileResult(
                        key=key, symbol=symbol, side=side, engine=tr.get("engine", "?"),
                        status="AMBIGUOUS",
                        note=f"{len(keys)} open trades on {symbol}:{side} across engines -- cannot attribute real fills",
                    ))
                continue
            key = keys[0]
            try:
                results.append(await self._reconcile_one(client, key, risk.trades[key]))
            except Exception as e:
                log.warning("reconcile %s failed: %s", key, e)

        self._log_results(results)
        return results

    def _log_results(self, results: list[ReconcileResult]) -> None:
        if not results:
            return
        now = time.time()
        lines = []
        for r in results:
            row = {
                "ts": now, "key": r.key, "symbol": r.symbol, "side": r.side, "engine": r.engine,
                "status": r.status, "note": r.note,
                "sim_eavg": r.sim_eavg, "sim_fill_pct": r.sim_fill_pct, "sim_r": r.sim_r,
                "real_eavg": r.real_eavg, "real_exit": r.real_exit, "real_r": r.real_r,
                "discrepancy": r.discrepancy(),
            }
            if r.status in ("AMBIGUOUS", "FETCH_FAILED") or row["discrepancy"]:
                log.warning("reconcile %s %s:%s [%s] %s", r.engine, r.symbol, r.side, r.status, row["discrepancy"] or r.note)
            lines.append(json.dumps(row))
        try:
            with self.log_path.open("a") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            log.warning("reconcile_log write failed", exc_info=True)
