"""Portfolio-level risk guard for ZENITH.

ZENITH only posts signals — it never touches an exchange account, so it has
no first-hand view of realized drawdown. Without that feedback, raising
leverage (or anything else) to chase return is a guess. RiskGuard closes
that loop cheaply: it replays each posted signal forward against the same
candles the bot already fetches, keeps an approximate compounding equity
curve, and uses it to gate new signals so realized drawdown stays near the
configured ceiling regardless of how confident any single setup looks.

This is a simulation, not a broker connection:
- Entries are NOT assumed to fill instantly or completely. Each signal posts
  a 4-level limit ladder; a trade starts in the FILLING phase and only the
  levels price actually touches within CANCEL_VELAS bars count as filled
  (matching the cancel-unfilled-limits behavior CANCEL_VELAS already
  describes in the Telegram message). If price never reaches any level,
  the signal is a NO_FILL and never becomes a position. If only some levels
  fill before the cancel window (or the stop) hits, the trade proceeds at
  that smaller size — its eventual R-multiple outcome is scaled down by
  fill_pct (the fraction of the intended ladder that actually filled)
  before it's applied to equity, since only that fraction of capital was
  ever really at risk.
- Within one bar, a stop-loss touch is checked before take-profits (the
  conservative assumption when intrabar order is unknown from OHLC alone).
  If price reaches the stop in the same bar that would also have filled
  deeper entries, those entries are counted as filled first (price
  necessarily passed through them on the way to the stop) before the
  stop-out is applied — otherwise a fast, straight-down move would look
  like a costless miss instead of a fill-then-immediate-loss.
- Position sizing follows fixed-fractional risk: each trade changes equity
  by (risk_equity_pct / 100) * (fill_pct / 100) * R_realized, compounding.
  Leverage itself does not appear in this formula — under risk-based
  sizing it only determines margin/liquidation headroom (see
  leverage_from_quality's LIQ_SAFETY_FRACTION clamp in strategy.py), not
  the equity swing from a stop-out. So RISK_EQUITY_PCT (and now fill_pct)
  are the real DD levers here, not leverage.

It is intentionally approximate — good enough to self-throttle the bot,
not a substitute for reconciling against real fills. In particular it
still assumes each filled level fills at exactly its quoted price (no
partial-level fills, no slippage within a touched level) and ignores
market-impact from position size.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import exchanges
from strategy import closed_candles

log = logging.getLogger("zenith.risk")


class RiskGuard:
    def __init__(self, path: Path | None = None, start_equity: float = 100.0):
        self.path = path or Path(__file__).with_name("risk_state.json")
        self.equity = start_equity
        self.peak = start_equity
        self.paused = False
        self.paused_at_ms: float | None = None
        self.closed_count = 0
        self.trades: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.equity = float(data.get("equity", self.equity))
            self.peak = float(data.get("peak", self.equity))
            self.paused = bool(data.get("paused", False))
            paused_at = data.get("paused_at_ms")
            self.paused_at_ms = float(paused_at) if paused_at is not None else None
            self.closed_count = int(data.get("closed_count", 0))
            self.trades = data.get("trades") or {}
        except Exception:
            log.warning("risk_state load failed; starting fresh", exc_info=True)

    def _save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(
                    {
                        "equity": self.equity,
                        "peak": self.peak,
                        "paused": self.paused,
                        "paused_at_ms": self.paused_at_ms,
                        "closed_count": self.closed_count,
                        "trades": self.trades,
                    }
                )
            )
        except Exception:
            log.warning("risk_state save failed", exc_info=True)

    def drawdown_pct(self) -> float:
        if self.peak <= 0:
            return 0.0
        return max(0.0, (self.peak - self.equity) / self.peak * 100.0)

    def concurrent_count(self, side: str | None = None) -> int:
        if side is None:
            return len(self.trades)
        return sum(1 for t in self.trades.values() if t.get("side") == side)

    def update_pause_state(
        self,
        ceiling_pct: float,
        resume_pct: float,
        now_ms: float,
        cooldown_ms: float | None = None,
    ) -> bool:
        """Hysteresis: pause at the ceiling, only resume once DD recovers
        below resume_pct.

        Recovery normally comes from *existing open trades* still resolving
        while paused (can_open blocks new entries, not the ongoing
        monitoring of ones already open). But if the ceiling is breached
        with no open trades left -- or they all close without recovering
        DD -- equity stops moving entirely, so DD can never fall below
        resume_pct on its own: a permanent lockup, since escaping requires
        wins, and wins require trading, which is exactly what's blocked.

        `cooldown_ms` is the escape hatch: after that much time paused with
        no organic recovery, force a resume and reset the high-water mark
        to current equity (accepting the drawdown as the new baseline)
        rather than requiring an unreachable recovery. Pass None to disable
        (matches the old behavior, including its lockup risk)."""
        dd = self.drawdown_pct()
        if not self.paused and dd > ceiling_pct:
            self.paused = True
            self.paused_at_ms = now_ms
            log.warning("RiskGuard: drawdown %.2f%% > ceiling %.2f%% — pausing new signals", dd, ceiling_pct)
        elif self.paused and dd <= resume_pct:
            self.paused = False
            self.paused_at_ms = None
            log.info("RiskGuard: drawdown recovered to %.2f%% <= resume %.2f%% — resuming", dd, resume_pct)
        elif self.paused and cooldown_ms and (now_ms - (self.paused_at_ms or now_ms)) >= cooldown_ms:
            self.paused = False
            self.paused_at_ms = None
            self.peak = self.equity
            log.warning(
                "RiskGuard: pause cooldown elapsed without DD recovery (still %.2f%%) — "
                "forcing resume, resetting high-water mark to current equity=%.2f",
                dd, self.equity,
            )
        self._save()
        return self.paused

    def can_open(self, side: str, cfg: dict) -> bool:
        if self.paused:
            return False
        if self.concurrent_count() >= int(cfg["max_concurrent_trades"]):
            return False
        if self.concurrent_count(side) >= int(cfg["max_concurrent_same_side"]):
            return False
        return True

    def open_trade(
        self,
        *,
        symbol: str,
        side: str,
        timeframe: str,
        entries: list[float],
        entry_weights: list[float],
        sl: float,
        risk_px: float,
        be_buffer_r: float,
        tps: list[float],
        tp_weights: list[float],
        be_after_tp1: bool,
        cancel_velas: int,
        close_velas: int,
        risk_equity_pct: float,
        last_ts: float | int | None,
    ) -> str | None:
        """risk_px must be abs(E1 - SL) -- the 1R price distance the TP
        ladder was actually spaced with (Signal.extras["r_unit"]) -- NOT
        abs(eavg - sl). Eavg sits between E1 and the deeper entries and is
        always closer to SL than E1 is, so deriving risk_px from eavg here
        would silently understate 1R and inflate every R-multiple this
        trade ever realizes.

        Starts in the FILLING phase (see module docstring) -- eavg is not
        known yet since it depends on which of `entries` actually fill."""
        risk_px = abs(float(risk_px))
        if risk_px <= 0:
            return None
        key = f"{symbol}:{side}:{last_ts or int(time.time())}"
        self.trades[key] = {
            "symbol": symbol,
            "side": side,
            "timeframe": timeframe,
            "phase": "FILLING",
            "pending_entries": [[float(p), float(w)] for p, w in zip(entries, entry_weights)],
            "filled_weight": 0.0,
            "fill_price_sum": 0.0,
            "fill_pct": 0.0,
            "eavg": None,
            "sl_orig": float(sl),
            "sl_cur": float(sl),
            "be_price": None,
            "be_buffer_r": float(be_buffer_r),
            "be_after_tp1": bool(be_after_tp1),
            "tps": [float(x) for x in tps],
            "tp_weights": [float(x) for x in tp_weights],
            "tp_idx": 0,
            "remaining_weight": 0.0,
            "r_realized": 0.0,
            "risk_px": risk_px,
            "risk_equity_pct": float(risk_equity_pct),
            "cancel_velas": int(cancel_velas),
            "close_velas": int(close_velas),
            "bars_elapsed": 0,
            "last_ts": float(last_ts or 0),
            "close_reason": None,
        }
        self._save()
        return key

    def _advance_fill(self, tr: dict, bar: dict) -> bool | None:
        """Process one bar of the FILLING phase. Returns True/False if the
        trade closed/continues, or None if it transitioned to OPEN (caller
        should then run the bar through the OPEN-phase logic too, since a
        bar that completes the fill can also move price further)."""
        bull = tr["side"] == "LONG"
        hi, lo = float(bar["high"]), float(bar["low"])
        sl_cur = float(tr["sl_cur"])

        swept_to_sl = (lo <= sl_cur) if bull else (hi >= sl_cur)
        still_pending = []
        for price, weight in tr["pending_entries"]:
            touched = (lo <= price) if bull else (hi >= price)
            # If the bar swept down to SL, any entry between the prior price
            # and SL was necessarily crossed on the way there even if we
            # only check high/low, not intrabar path.
            on_the_way = swept_to_sl and ((price >= sl_cur) if bull else (price <= sl_cur))
            if touched or on_the_way:
                tr["filled_weight"] += weight
                tr["fill_price_sum"] += price * weight
            else:
                still_pending.append([price, weight])
        tr["pending_entries"] = still_pending

        if swept_to_sl:
            if tr["filled_weight"] <= 0:
                tr["close_reason"] = "NO_FILL"
                tr["fill_pct"] = 0.0
                return True
            tr["eavg"] = tr["fill_price_sum"] / tr["filled_weight"]
            tr["fill_pct"] = tr["filled_weight"]
            risk_px = float(tr["risk_px"])
            r_mult = ((sl_cur - tr["eavg"]) / risk_px) if bull else ((tr["eavg"] - sl_cur) / risk_px)
            tr["r_realized"] = r_mult
            tr["close_reason"] = "SL_DURING_FILL"
            return True

        if not tr["pending_entries"]:
            self._resolve_fill(tr, bull)
            return None

        if tr["bars_elapsed"] >= int(tr["cancel_velas"]) or tr["bars_elapsed"] >= int(tr["close_velas"]):
            if tr["filled_weight"] <= 0:
                tr["close_reason"] = "NO_FILL"
                tr["fill_pct"] = 0.0
                return True
            # Cancel window expired with a partial fill: proceed at the
            # smaller size actually achieved.
            self._resolve_fill(tr, bull)
            return None

        return False

    def _resolve_fill(self, tr: dict, bull: bool) -> None:
        """Finalize eavg/be_price from what actually filled and move to OPEN.

        be_price MUST be derived from the real filled eavg here, not from a
        value pre-computed at signal time against the *intended* full-ladder
        blend. When only a shallow level fills (partial fill), the real
        entry price can differ from that intended blend by enough to turn
        the "lock in a small profit" breakeven+buffer level into a level
        that is actually a small real loss relative to where the position
        truly entered -- exactly the bug this fixes."""
        tr["eavg"] = tr["fill_price_sum"] / tr["filled_weight"]
        tr["fill_pct"] = tr["filled_weight"]
        risk_px = float(tr["risk_px"])
        be_buffer_r = float(tr.get("be_buffer_r") or 0.0)
        tr["be_price"] = tr["eavg"] + be_buffer_r * risk_px if bull else tr["eavg"] - be_buffer_r * risk_px
        tr["remaining_weight"] = 100.0
        tr["phase"] = "OPEN"

    def _advance(self, tr: dict, bar: dict) -> bool:
        """Process one closed bar for one open trade. Returns True if closed."""
        tr["bars_elapsed"] = int(tr.get("bars_elapsed", 0)) + 1

        if tr.get("phase", "OPEN") == "FILLING":
            result = self._advance_fill(tr, bar)
            if result is not None:
                return result
            # fell through to OPEN this same bar -- keep going below with
            # the same bar, since completing the fill doesn't consume it

        bull = tr["side"] == "LONG"
        hi, lo, cl = float(bar["high"]), float(bar["low"]), float(bar["close"])
        risk_px = float(tr["risk_px"])
        sl_cur = float(tr["sl_cur"])

        stopped = (lo <= sl_cur) if bull else (hi >= sl_cur)
        if stopped:
            w = tr["remaining_weight"]
            r_mult = ((sl_cur - tr["eavg"]) / risk_px) if bull else ((tr["eavg"] - sl_cur) / risk_px)
            tr["r_realized"] += (w / 100.0) * r_mult
            tr["remaining_weight"] = 0.0
            tr["close_reason"] = "SL" if sl_cur == float(tr.get("sl_orig", sl_cur)) else "BE"
            return True

        tps = tr["tps"]
        weights = tr["tp_weights"]
        idx = int(tr["tp_idx"])
        while idx < len(tps):
            tp_price = float(tps[idx])
            hit = (hi >= tp_price) if bull else (lo <= tp_price)
            if not hit:
                break
            w = min(float(weights[idx]) if idx < len(weights) else 0.0, tr["remaining_weight"])
            r_mult = ((tp_price - tr["eavg"]) / risk_px) if bull else ((tr["eavg"] - tp_price) / risk_px)
            tr["r_realized"] += (w / 100.0) * r_mult
            tr["remaining_weight"] -= w
            idx += 1
            if idx == 1 and tr.get("be_after_tp1"):
                tr["sl_cur"] = float(tr["be_price"])
        tr["tp_idx"] = idx
        if tr["remaining_weight"] <= 0.01:
            tr["close_reason"] = "TP_ALL"
            return True

        tr["bars_elapsed"] = int(tr.get("bars_elapsed", 0)) + 1
        if tr["bars_elapsed"] >= int(tr["close_velas"]):
            w = tr["remaining_weight"]
            r_mult = ((cl - tr["eavg"]) / risk_px) if bull else ((tr["eavg"] - cl) / risk_px)
            tr["r_realized"] += (w / 100.0) * r_mult
            tr["remaining_weight"] = 0.0
            tr["close_reason"] = "TIME"
            return True
        return False

    def _close_trade(self, tr: dict) -> None:
        risk_pct = float(tr.get("risk_equity_pct") or 0.0)
        fill_pct = float(tr.get("fill_pct") or 0.0)
        r = float(tr.get("r_realized") or 0.0)
        # fill_pct scales the equity impact: only that fraction of the
        # intended ladder ever actually filled, so only that fraction of
        # RISK_EQUITY_PCT was ever really at risk. A NO_FILL trade has
        # fill_pct=0 (and r=0) and is correctly a no-op on equity.
        self.equity *= 1.0 + (risk_pct / 100.0) * (fill_pct / 100.0) * r
        self.peak = max(self.peak, self.equity)
        self.closed_count += 1
        log.info(
            "RiskGuard closed %s %s [%s] fill=%.0f%% R=%.2f equity=%.2f dd=%.2f%%",
            tr.get("symbol"),
            tr.get("side"),
            tr.get("close_reason"),
            fill_pct,
            r,
            self.equity,
            self.drawdown_pct(),
        )

    async def refresh(self, client) -> None:
        """Replay any new closed candles for each open (simulated) trade."""
        if not self.trades:
            return
        # Resolved per call (cheap: an os.getenv + a couple of name lookups),
        # not at import time -- this module is imported by bot.py before
        # bot.py's own load_dotenv() runs, so resolving DATA_SOURCE here
        # instead of at module scope ensures .env (not just the shell
        # environment) is already loaded by the time it matters. Must use
        # the same source live signals were built from, or replaying a
        # BloFin-priced trade against Binance candles would reintroduce the
        # exact price mismatch RiskGuard's DD simulation should reflect.
        _, fetch_klines, _ = exchanges.resolve_source()
        closed_keys = []
        for key, tr in list(self.trades.items()):
            try:
                raw, _ = await fetch_klines(
                    client, tr["symbol"], tr["timeframe"], limit=max(60, int(tr["close_velas"]) + 20)
                )
            except Exception as e:
                log.debug("risk_guard refresh %s failed: %s", tr.get("symbol"), e)
                continue
            candles = closed_candles(raw)
            last_ts = float(tr.get("last_ts") or 0)
            new_bars = [c for c in candles if float(c.get("ts") or 0) > last_ts]
            done = False
            for bar in new_bars:
                done = self._advance(tr, bar)
                tr["last_ts"] = float(bar.get("ts") or last_ts)
                if done:
                    break
            if done:
                self._close_trade(tr)
                closed_keys.append(key)
        for key in closed_keys:
            self.trades.pop(key, None)
        self._save()
