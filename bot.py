#!/usr/bin/env python3
"""ZENITH — SMC-LITE entries + MATRIX-ORIENT exits Telegram signal bot."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path):
        p = Path(path)
        if not p.exists():
            return
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import gem_strategy as gem
from exchanges import resolve_source
from formatter import format_signal
from risk_guard import RiskGuard
from strategy import (
    FIB_CFG,
    apply_strategy_profile,
    build_signal,
    btc_regime_from_candles,
    closed_candles,
    leverage_from_quality,
    score_symbol,
)

load_dotenv(Path(__file__).with_name(".env"))

# Resolved after load_dotenv() so DATA_SOURCE from .env (not just the shell
# environment) takes effect -- see exchanges.resolve_source(). Both ZENITH
# and GEM use these same three functions, so switching DATA_SOURCE moves
# both strategies' price data together.
load_universe, fetch_klines, fetch_funding = resolve_source()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("zenith")

DEFAULT_DENYLIST = "DOODUSDT,AKEUSDT,WLDUSDT,RAYSOLUSDT"


def _csv_floats(raw: str, default: list[float]) -> list[float]:
    if not raw or not raw.strip():
        return list(default)
    out = []
    for p in raw.split(","):
        p = p.strip()
        if p:
            out.append(float(p))
    return out or list(default)


def _csv_syms(raw: str) -> set[str]:
    if not raw or not raw.strip():
        return set()
    return {s.strip().upper() for s in raw.split(",") if s.strip()}


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _cfg() -> dict:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHANNEL_ID", "").strip()
    dry_run = _env_bool("DRY_RUN", False)
    if not dry_run and (not token or not chat):
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID in .env (or DRY_RUN=1)")
    raw_symbols = os.getenv("SYMBOLS", "ALL").strip()
    symbols = []
    if raw_symbols and raw_symbols.upper() != "ALL":
        symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
    deny_raw = os.getenv("SYMBOL_DENYLIST", DEFAULT_DENYLIST)
    return {
        "token": token,
        "chat": chat,
        "symbols": symbols,
        "denylist": _csv_syms(deny_raw),
        "strategy_profile": os.getenv("STRATEGY_PROFILE", "smc_lite").strip().lower() or "smc_lite",
        "leverage_min": int(os.getenv("LEVERAGE_MIN", "10")),
        "leverage_max": int(os.getenv("LEVERAGE_MAX", "15")),
        "timeframe": os.getenv("TIMEFRAME", "15m"),
        "scan_seconds": int(os.getenv("SCAN_SECONDS", "90")),
        "cooldown_min": int(os.getenv("COOLDOWN_MINUTES", "720")),
        "min_score": float(os.getenv("MIN_SCORE", "70")),
        "min_quote_vol": float(os.getenv("MIN_QUOTE_VOLUME_USD", "20000000")),
        "max_per_scan": int(os.getenv("MAX_PAIRS_PER_SCAN", "80")),
        "concurrency": int(os.getenv("SCAN_CONCURRENCY", "6")),
        "entry_weights": _csv_floats(os.getenv("ENTRY_WEIGHTS", ""), [45, 30, 15, 10]),
        "tp_weights": _csv_floats(os.getenv("TP_WEIGHTS", ""), [80, 10, 5, 3, 2]),
        "tp_multiples": _csv_floats(os.getenv("TP_MULTIPLES", ""), [0.2, 0.5, 0.95, 1.5, 2.3]),
        "max_posts_per_scan": int(os.getenv("MAX_POSTS_PER_SCAN", "2")),
        "side_balance": _env_bool("SIDE_BALANCE", True),
        "dry_run": dry_run,
        "log_rejects": _env_bool("LOG_REJECTS", False),
        "allow_shorts": _env_bool("ALLOW_SHORTS", True),
        "long_only": _env_bool("LONG_ONLY", False),
        "btc_regime": _env_bool("BTC_REGIME", True),
        "stop_aprieta": float(os.getenv("STOP_APRIETA", "1.0")),
        "be_after_tp1": _env_bool("BE_AFTER_TP1", True),
        "be_buffer_r": float(os.getenv("BE_BUFFER_R", "0.12")),
        "adaptive_tp": _env_bool("ADAPTIVE_TP", True),
        "wick_reject_max": float(os.getenv("WICK_REJECT_MAX", "0.45")),
        "cancel_velas": int(os.getenv("CANCEL_VELAS", "8")),
        "close_velas": int(os.getenv("CLOSE_VELAS", "20")),
        "r_min_pct": float(os.getenv("R_MIN_PCT", "1.00")),
        "r_max_pct": float(os.getenv("R_MAX_PCT", "1.85")),
        "risk_equity_pct": float(os.getenv("RISK_EQUITY_PCT", "0.65")),
        "liq_safety_fraction": float(os.getenv("LIQ_SAFETY_FRACTION", "0.50")),
        # Portfolio-level guardrails so realized drawdown stays near target
        # even though per-trade leverage went up (see risk_guard.py).
        "max_concurrent_trades": int(os.getenv("MAX_CONCURRENT_TRADES", "4")),
        "max_concurrent_same_side": int(os.getenv("MAX_CONCURRENT_SAME_SIDE", "3")),
        "dd_ceiling_pct": float(os.getenv("DD_CEILING_PCT", "3.0")),
        "dd_resume_pct": float(os.getenv("DD_RESUME_PCT", "1.5")),
        # Escape hatch for update_pause_state(): if the ceiling trips with
        # no open trades left to organically recover it, equity can never
        # move and DD can never fall below dd_resume_pct on its own -- a
        # permanent lockup. After this many hours paused, force a resume
        # and reset the high-water mark to current equity instead.
        "dd_pause_cooldown_hours": float(os.getenv("DD_PAUSE_COOLDOWN_HOURS", "48")),
        # --- GEM: second strategy engine (gem_strategy.py), same bot/chat/
        # RiskGuard as ZENITH above. GEM's own internals (score threshold,
        # SMC filters, R band, ...) are tunable generically as GEM_<KEY> --
        # see configure_gem_strategy() -- since gem_strategy.GEM_CFG mirrors
        # motor-gem.js's CFG object key-for-key.
        "gem_enabled": _env_bool("GEM_ENABLED", True),
        "gem_timeframe": os.getenv("GEM_TIMEFRAME", "15m"),
        "gem_top_n": int(os.getenv("GEM_TOP_N", "30")),
        "gem_min_quote_vol": float(os.getenv("GEM_MIN_QUOTE_VOLUME_USD", "2000000")),
        "gem_cooldown_min": int(os.getenv("GEM_COOLDOWN_MINUTES", "120")),
        "gem_max_posts_per_scan": int(os.getenv("GEM_MAX_SIGNALS_PER_SCAN", "3")),
        "gem_leverage_min": int(os.getenv("GEM_LEVERAGE_MIN", "10")),
        "gem_leverage_max": int(os.getenv("GEM_LEVERAGE_MAX", "10")),
    }


def configure_gem_strategy() -> None:
    """Env overrides onto gem_strategy.GEM_CFG, generic over its keys/types
    so every GEM engine parameter is tunable as GEM_<KEY> (e.g. GEM_MIN_SCORE,
    GEM_REQUIRE_BOS, GEM_R_MAX_PCT) without hand-listing each one here."""
    for key, current in list(gem.GEM_CFG.items()):
        raw = os.getenv(f"GEM_{key}", "").strip()
        if not raw:
            continue
        if isinstance(current, bool):
            gem.GEM_CFG[key] = _env_bool(f"GEM_{key}", current)
        elif isinstance(current, list):
            gem.GEM_CFG[key] = [float(x) for x in raw.split(",") if x.strip()]
        elif isinstance(current, int):
            gem.GEM_CFG[key] = int(float(raw))
        elif isinstance(current, float):
            gem.GEM_CFG[key] = float(raw)
        else:
            gem.GEM_CFG[key] = raw


class Cooldown:
    def __init__(self, minutes: int, path: Path | None = None):
        self.seconds = max(1, minutes) * 60
        self.path = path or Path(__file__).with_name("cooldown.json")
        self._m: dict[str, float] = {}
        self._lock = asyncio.Lock()
        if self.path.exists():
            try:
                self._m = {k: float(v) for k, v in json.loads(self.path.read_text()).items()}
            except Exception:
                self._m = {}

    def ready(self, key: str) -> bool:
        return time.time() >= float(self._m.get(key) or 0)

    async def hit(self, key: str) -> None:
        async with self._lock:
            self._m[key] = time.time() + self.seconds
            try:
                self.path.write_text(json.dumps(self._m))
            except Exception:
                pass


async def evaluate_one(
    client: httpx.AsyncClient,
    symbol: str,
    ticker: dict,
    cfg: dict,
    btc_regime: str = "neutral",
) -> dict | None:
    """Funding + LTF first; only fetch HTF if LTF prefilter passes (rate-limit friendly)."""
    try:
        fund = await fetch_funding(client, symbol)
        if fund is not None:
            ticker["funding"] = fund
    except Exception:
        pass

    candles, _src = await fetch_klines(client, symbol, cfg["timeframe"], limit=240)
    candles = closed_candles(candles)
    if len(candles) < 50:
        if cfg["log_rejects"]:
            log.info("reject %s: insufficient LTF candles", symbol)
        return None

    # Prefilter without HTF (saves 1h calls on most pairs)
    pre = score_symbol(
        candles,
        ticker,
        None,
        require_htf=False,
        entry_weights=cfg["entry_weights"],
        btc_regime=btc_regime,
        symbol=symbol,
        allow_shorts=cfg["allow_shorts"],
        long_only=cfg["long_only"],
    )
    if not pre or pre["score"] < cfg["min_score"]:
        if cfg["log_rejects"]:
            # score=None with no other detail usually means an internal
            # hard gate tripped before scoring even started (SMC-lite needs
            # >=168 closed candles for its EMA_SLOW=160 -- easy to fall short
            # of on an exchange that returns fewer bars per request than
            # Binance does) rather than "the setup scored low" -- candles/
            # change_pct/quote_vol below narrow that down.
            log.info(
                "reject %s: LTF prefilter score=%s (candles=%d change_pct=%.2f%% quote_vol=%s)",
                symbol,
                None if not pre else pre["score"],
                len(candles),
                float(ticker.get("change_pct") or 0),
                ticker.get("quote_vol"),
            )
        return None

    htf = None
    if cfg["timeframe"] != "1h":
        try:
            htf_raw, _ = await fetch_klines(client, symbol, "1h", limit=240)
            htf = closed_candles(htf_raw)
        except Exception:
            htf = None

    scored = score_symbol(
        candles,
        ticker,
        htf,
        require_htf=bool(htf),
        entry_weights=cfg["entry_weights"],
        btc_regime=btc_regime,
        symbol=symbol,
        allow_shorts=cfg["allow_shorts"],
        long_only=cfg["long_only"],
    )
    if not scored or scored["score"] < cfg["min_score"]:
        if cfg["log_rejects"]:
            log.info("reject %s: HTF/final score=%s", symbol, None if not scored else scored["score"])
        return None
    scored["last_ts"] = candles[-1].get("ts") if candles else None
    return scored


async def post_telegram(client: httpx.AsyncClient, cfg: dict, text: str) -> None:
    url = f"https://api.telegram.org/bot{cfg['token']}/sendMessage"
    r = await client.post(url, json={"chat_id": cfg["chat"], "text": text}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"telegram {r.status_code} {r.text[:200]}")


async def scan_once(client: httpx.AsyncClient, cfg: dict, cool: Cooldown, universe: dict, risk: RiskGuard) -> None:
    """Score the universe, then post both sides when possible (long + short)."""
    batch = universe.get("tickers") or []
    if cfg["symbols"]:
        want = set(cfg["symbols"])
        batch = [t for t in batch if t["symbol"] in want]
    deny = cfg.get("denylist") or set()
    if deny:
        batch = [t for t in batch if t["symbol"] not in deny]
    batch = batch[: cfg["max_per_scan"]]

    btc_regime = "neutral"
    if cfg.get("btc_regime"):
        try:
            btc_raw, _ = await fetch_klines(client, "BTCUSDT", "1h", limit=240)
            btc_regime = btc_regime_from_candles(closed_candles(btc_raw))
            log.info("BTC regime=%s", btc_regime)
        except Exception as e:
            log.warning("BTC regime fetch failed: %s", e)

    sem = asyncio.Semaphore(cfg["concurrency"])
    candidates: list[tuple[dict, object, int, str]] = []

    async def score_one(t: dict):
        symbol = t["symbol"]
        if not cool.ready(f"sym:{symbol}"):
            return
        async with sem:
            try:
                scored = await evaluate_one(client, symbol, dict(t), cfg, btc_regime=btc_regime)
            except Exception as e:
                log.warning("scan %s failed: %s", symbol, e)
                return
        if not scored:
            return
        if cfg.get("long_only") and scored["side"] != "LONG":
            return
        if not cfg.get("allow_shorts", True) and scored["side"] != "LONG":
            return
        if not cool.ready(f"side:{symbol}:{scored['side']}"):
            return
        fib = scored.get("fib") or {}
        lev = leverage_from_quality(
            scored["score"],
            side=scored["side"],
            lev_min=cfg["leverage_min"],
            lev_max=cfg["leverage_max"],
            rpct=fib.get("Rpct"),
            ratr=fib.get("Ratr"),
            vol_ratio=scored.get("vol_ratio"),
        )
        exch_max = int((universe.get("max_lev") or {}).get(symbol) or cfg["leverage_max"])
        if exch_max < cfg["leverage_min"]:
            if cfg["log_rejects"]:
                log.info("reject %s: exch max lev %s < min %s", symbol, exch_max, cfg["leverage_min"])
            return
        # Don't force back up to leverage_min: leverage_from_quality() may
        # have already pulled leverage below it for liquidation safety on a
        # wide-stop trade, and re-raising it here would undo that guard.
        lev = max(1, min(lev, exch_max, cfg["leverage_max"]))
        sig = build_signal(
            symbol,
            scored,
            leverage=lev,
            timeframe=cfg["timeframe"],
            entry_weights=cfg["entry_weights"],
            tp_weights=cfg["tp_weights"],
        )
        candidates.append((scored, sig, lev, symbol))

    await asyncio.gather(*(score_one(t) for t in batch))
    if not candidates:
        log.info("scan: no candidates")
        return

    candidates.sort(key=lambda x: float(x[0]["score"]), reverse=True)
    picks: list[tuple] = []
    if cfg.get("side_balance", True) and not cfg.get("long_only"):
        best_long = next((c for c in candidates if c[0]["side"] == "LONG"), None)
        best_short = next((c for c in candidates if c[0]["side"] == "SHORT"), None)
        if best_long:
            picks.append(best_long)
        if best_short:
            picks.append(best_short)
        # fill remaining slots by score if max_posts > 2
        for c in candidates:
            if len(picks) >= cfg["max_posts_per_scan"]:
                break
            if c in picks:
                continue
            picks.append(c)
    else:
        picks = candidates[: cfg["max_posts_per_scan"]]

    picks = picks[: cfg["max_posts_per_scan"]]
    for scored, sig, lev, symbol in picks:
        if not risk.can_open(sig.side, cfg):
            log.info(
                "risk_guard: skip %s %s (paused=%s open=%s dd=%.2f%%)",
                symbol, sig.side, risk.paused, risk.concurrent_count(), risk.drawdown_pct(),
            )
            continue
        text = format_signal(sig)
        if cfg["dry_run"]:
            log.info("DRY_RUN would post %s %s score=%s lev=%sx\n%s", symbol, sig.side, sig.score, lev, text)
        else:
            try:
                await post_telegram(client, cfg, text)
            except Exception as e:
                log.error("telegram %s: %s", symbol, e)
                continue
        await cool.hit(f"sym:{symbol}")
        await cool.hit(f"side:{symbol}:{sig.side}")
        risk.open_trade(
            symbol=symbol,
            side=sig.side,
            timeframe=cfg["timeframe"],
            entries=list(sig.entries),
            entry_weights=list(sig.entry_weights),
            sl=float(sig.sl),
            risk_px=float(sig.extras.get("r_unit") or abs(sig.entries[0] - sig.sl)),
            be_buffer_r=float(sig.extras.get("be_buffer_r") or 0.0),
            tps=list(sig.tps),
            tp_weights=list(sig.tp_weights),
            be_after_tp1=bool(sig.extras.get("be_after_tp1", True)),
            cancel_velas=cfg["cancel_velas"],
            close_velas=cfg["close_velas"],
            risk_equity_pct=cfg["risk_equity_pct"],
            last_ts=scored.get("last_ts"),
        )
        log.info("posted %s %s score=%s lev=%sx", symbol, sig.side, sig.score, lev)


async def evaluate_one_gem(client: httpx.AsyncClient, symbol: str, cfg: dict) -> dict | None:
    try:
        fund = await fetch_funding(client, symbol)
    except Exception:
        fund = None
    candles, _src = await fetch_klines(client, symbol, cfg["gem_timeframe"], limit=200)
    candles = closed_candles(candles)
    if len(candles) < 50:
        if cfg["log_rejects"]:
            log.info("gem reject %s: insufficient candles", symbol)
        return None
    result = gem.evaluate(gem.from_repo_candles(candles), fund)
    if not result.get("ok"):
        if cfg["log_rejects"]:
            log.info("gem reject %s: %s (%s)", symbol, result.get("reason"), result.get("code"))
        return None
    result["last_ts"] = candles[-1].get("ts") if candles else None
    return result


async def gem_scan_once(client: httpx.AsyncClient, cfg: dict, cool: Cooldown, universe: dict, risk: RiskGuard) -> None:
    """Same shape as scan_once() but for the GEM engine: its own (lower)
    volume floor/top-N cut, its own cooldown namespace, but the same
    RiskGuard so both strategies share one portfolio drawdown ceiling."""
    tickers = universe.get("tickers") or []
    if cfg["symbols"]:
        want = set(cfg["symbols"])
        tickers = [t for t in tickers if t["symbol"] in want]
    deny = cfg.get("denylist") or set()
    if deny:
        tickers = [t for t in tickers if t["symbol"] not in deny]
    batch = [t for t in tickers if t.get("quote_vol", 0) >= cfg["gem_min_quote_vol"]]
    batch.sort(key=lambda t: t.get("quote_vol", 0), reverse=True)
    batch = batch[: cfg["gem_top_n"]]

    sem = asyncio.Semaphore(cfg["concurrency"])
    candidates: list[tuple[dict, object, int, str]] = []

    async def score_one(t: dict):
        symbol = t["symbol"]
        if not cool.ready(f"gem:{symbol}"):
            return
        async with sem:
            try:
                result = await evaluate_one_gem(client, symbol, cfg)
            except Exception as e:
                log.warning("gem scan %s failed: %s", symbol, e)
                return
        if not result:
            return
        lev = leverage_from_quality(
            result["nota"],
            side=result["dir"],
            lev_min=cfg["gem_leverage_min"],
            lev_max=cfg["gem_leverage_max"],
            rpct=result["Rpct"],
            ratr=result["Ratr"],
            vol_ratio=result["volRel"],
        )
        exch_max = int((universe.get("max_lev") or {}).get(symbol) or cfg["gem_leverage_max"])
        if exch_max < cfg["gem_leverage_min"]:
            if cfg["log_rejects"]:
                log.info("gem reject %s: exch max lev %s < min %s", symbol, exch_max, cfg["gem_leverage_min"])
            return
        lev = max(1, min(lev, exch_max, cfg["gem_leverage_max"]))
        sig = gem.build_signal(symbol, result, leverage=lev, timeframe=cfg["gem_timeframe"])
        candidates.append((result, sig, lev, symbol))

    await asyncio.gather(*(score_one(t) for t in batch))
    if not candidates:
        log.info("gem scan: no candidates")
        return

    candidates.sort(key=lambda x: float(x[0]["nota"]), reverse=True)
    picks = candidates[: cfg["gem_max_posts_per_scan"]]
    for result, sig, lev, symbol in picks:
        if not risk.can_open(sig.side, cfg):
            log.info(
                "risk_guard: skip GEM %s %s (paused=%s open=%s dd=%.2f%%)",
                symbol, sig.side, risk.paused, risk.concurrent_count(), risk.drawdown_pct(),
            )
            continue
        text = format_signal(sig)
        if cfg["dry_run"]:
            log.info("DRY_RUN would post GEM %s %s score=%s lev=%sx\n%s", symbol, sig.side, sig.score, lev, text)
        else:
            try:
                await post_telegram(client, cfg, text)
            except Exception as e:
                log.error("telegram %s: %s", symbol, e)
                continue
        await cool.hit(f"gem:{symbol}")
        risk.open_trade(
            symbol=symbol,
            side=sig.side,
            timeframe=cfg["gem_timeframe"],
            entries=list(sig.entries),
            entry_weights=list(sig.entry_weights),
            sl=float(sig.sl),
            risk_px=float(sig.extras.get("r_unit") or abs(sig.entries[0] - sig.sl)),
            be_buffer_r=0.0,
            tps=list(sig.tps),
            tp_weights=list(sig.tp_weights),
            be_after_tp1=bool(sig.extras.get("be_after_tp1", True)),
            cancel_velas=int(sig.extras.get("cancel_velas") or cfg["cancel_velas"]),
            close_velas=int(sig.extras.get("close_velas") or cfg["close_velas"]),
            risk_equity_pct=cfg["risk_equity_pct"],
            last_ts=result.get("last_ts"),
        )
        log.info("posted GEM %s %s score=%s lev=%sx", symbol, sig.side, sig.score, lev)


def configure_strategy(cfg: dict) -> str:
    """Apply cfg + env overrides onto FIB_CFG. Shared by bot.py and
    backtest.py so a backtest run reflects the exact same strategy
    configuration the live bot would use — no risk of the two drifting
    apart. Returns the resolved profile name."""
    profile = apply_strategy_profile(cfg["strategy_profile"])
    # SMC_LITE=1 forces smc_lite even if STRATEGY_PROFILE was matrix_orient
    if _env_bool("SMC_LITE", profile == "smc_lite"):
        profile = apply_strategy_profile("smc_lite")
        FIB_CFG["SMC_LITE"] = True
    else:
        FIB_CFG["SMC_LITE"] = profile == "smc_lite"
    for _k, _cast in (
        ("EXIGIR_BOS", _env_bool),
        ("EXIGIR_FVG", _env_bool),
        ("EXIGIR_CHOCH_LTF", _env_bool),
        ("EXIGIR_DESCUENTO", _env_bool),
        ("EXIGIR_POI_OR_CHOCH", _env_bool),
        ("EXIGIR_FVG_INSIDE", _env_bool),
        ("SMC_POI_SOFT", _env_bool),
        ("SMC_PD_SOFT", _env_bool),
    ):
        if os.getenv(_k, "").strip() != "":
            FIB_CFG[_k] = _cast(_k, bool(FIB_CFG.get(_k, False)))
    if os.getenv("SMC_ENTRY_MODE", "").strip():
        FIB_CFG["SMC_ENTRY_MODE"] = os.getenv("SMC_ENTRY_MODE").strip().lower()
    for _k in ("SMC_FVG_NEAR_ATR", "SMC_BUFFER_POI", "SMC_EXPANSION", "VOL_MIN"):
        if os.getenv(_k, "").strip():
            FIB_CFG[_k] = float(os.getenv(_k))
    for _k in ("SMC_SWING_HTF", "SMC_SWING_LTF", "SMC_FVG_VELAS", "SMC_OB_VELAS"):
        if os.getenv(_k, "").strip():
            FIB_CFG[_k] = int(float(os.getenv(_k)))
    FIB_CFG["BTC_REGIME"] = bool(cfg["btc_regime"])
    FIB_CFG["ALLOW_SHORTS"] = bool(cfg["allow_shorts"]) and not bool(cfg["long_only"])
    FIB_CFG["LONG_ONLY"] = bool(cfg["long_only"])
    FIB_CFG["LEV_MIN"] = int(cfg["leverage_min"])
    FIB_CFG["LEV_MAX"] = int(cfg["leverage_max"])
    FIB_CFG["STOP_APRIETA"] = float(cfg["stop_aprieta"])
    FIB_CFG["BE_AFTER_TP1"] = bool(cfg["be_after_tp1"])
    FIB_CFG["BE_BUFFER_R"] = float(cfg["be_buffer_r"])
    FIB_CFG["ADAPTIVE_TP"] = bool(cfg["adaptive_tp"])
    FIB_CFG["WICK_REJECT_MAX"] = float(cfg["wick_reject_max"])
    FIB_CFG["RISK_EQUITY_PCT"] = float(cfg["risk_equity_pct"])
    FIB_CFG["LIQ_SAFETY_FRACTION"] = float(cfg["liq_safety_fraction"])
    FIB_CFG["CANCEL_VELAS"] = int(cfg["cancel_velas"])
    FIB_CFG["CLOSE_VELAS"] = int(cfg["close_velas"])
    FIB_CFG["R_MIN_PCT"] = float(cfg["r_min_pct"])
    FIB_CFG["R_MAX_PCT"] = float(cfg["r_max_pct"])
    # SL_CAP stays profile default unless SL_CAP_PCT env set
    if os.getenv("SL_CAP_PCT", "").strip():
        FIB_CFG["SL_CAP_PCT"] = float(os.getenv("SL_CAP_PCT"))
    FIB_CFG["MIN_SCORE"] = float(cfg["min_score"])
    FIB_CFG["ENTRY_WEIGHTS"] = list(cfg["entry_weights"])
    FIB_CFG["CIERRES"] = list(cfg["tp_weights"])
    if len(cfg["tp_multiples"]) == 5:
        FIB_CFG["TP"] = list(cfg["tp_multiples"])
    if len(cfg["entry_weights"]) != 4 or len(cfg["tp_weights"]) != 5:
        raise SystemExit("Need 4 ENTRY_WEIGHTS and 5 TP_WEIGHTS")
    return profile


async def main() -> None:
    cfg = _cfg()
    profile = configure_strategy(cfg)
    configure_gem_strategy()
    cool = Cooldown(cfg["cooldown_min"])
    cool_gem = Cooldown(cfg["gem_cooldown_min"], path=Path(__file__).with_name("cooldown_gem.json"))
    risk = RiskGuard()
    async with httpx.AsyncClient(headers={"User-Agent": "zenith-bot/1.0"}) as client:
        log.info(
            "data source=%s (both strategies + RiskGuard price/candle data)",
            os.getenv("DATA_SOURCE", "blofin").strip().lower() or "blofin",
        )
        log.info(
            "ZENITH started — profile=%s scan every %ss, tf=%s, lev %s-%sx, min_score=%s STOP_APRIETA=%s CLOSE=%s dry_run=%s, allow_shorts=%s long_only=%s side_balance=%s max_posts=%s deny=%s risk_pct=%s dd_ceiling=%s%% max_concurrent=%s/%s",
            profile,
            cfg["scan_seconds"],
            cfg["timeframe"],
            cfg["leverage_min"],
            cfg["leverage_max"],
            cfg["min_score"],
            cfg["stop_aprieta"],
            cfg["close_velas"],
            cfg["dry_run"],
            cfg["allow_shorts"],
            cfg["long_only"],
            cfg["side_balance"],
            cfg["max_posts_per_scan"],
            ",".join(sorted(cfg["denylist"])) or "(none)",
            cfg["risk_equity_pct"],
            cfg["dd_ceiling_pct"],
            cfg["max_concurrent_trades"],
            cfg["max_concurrent_same_side"],
        )
        if cfg["gem_enabled"]:
            log.info(
                "GEM enabled — tf=%s top_n=%s min_vol=%s lev %s-%sx cooldown=%smin max_posts=%s",
                cfg["gem_timeframe"], cfg["gem_top_n"], cfg["gem_min_quote_vol"],
                cfg["gem_leverage_min"], cfg["gem_leverage_max"], cfg["gem_cooldown_min"],
                cfg["gem_max_posts_per_scan"],
            )
        while True:
            try:
                await risk.refresh(client)
                risk.update_pause_state(
                    cfg["dd_ceiling_pct"],
                    cfg["dd_resume_pct"],
                    now_ms=time.time() * 1000,
                    cooldown_ms=cfg["dd_pause_cooldown_hours"] * 3_600_000,
                )
                log.info(
                    "risk_guard: equity=%.2f dd=%.2f%% paused=%s open=%s (long=%s short=%s) closed=%s",
                    risk.equity,
                    risk.drawdown_pct(),
                    risk.paused,
                    risk.concurrent_count(),
                    risk.concurrent_count("LONG"),
                    risk.concurrent_count("SHORT"),
                    risk.closed_count,
                )
                universe = await load_universe(client, cfg["min_quote_vol"])
                log.info("universe %s names", len(universe.get("tickers") or []))
                await scan_once(client, cfg, cool, universe, risk)
                if cfg["gem_enabled"]:
                    # Separate fetch: GEM's volume floor is typically much
                    # lower than ZENITH's, so it needs its own universe
                    # rather than reusing ZENITH's (already filtered) one.
                    gem_universe = await load_universe(client, cfg["gem_min_quote_vol"])
                    log.info("gem universe %s names", len(gem_universe.get("tickers") or []))
                    await gem_scan_once(client, cfg, cool_gem, gem_universe, risk)
            except Exception as e:
                log.exception("scan loop: %s", e)
            await asyncio.sleep(cfg["scan_seconds"])


if __name__ == "__main__":
    asyncio.run(main())
