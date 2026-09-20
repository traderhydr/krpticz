#!/usr/bin/env python3
"""ZENITH + GEM walk-forward backtest -> Excel report.

Standalone script. Run it on a machine with real internet access to Binance
Futures — the sandbox this bot was developed in has outbound access to
exchange APIs blocked by policy, so this script could not be executed or
timed there. It reuses the exact same strategy code the live bot runs
(strategy.py / gem_strategy.py, risk_guard.py) and the same config-
application path (bot.configure_strategy / bot.configure_gem_strategy), so
results reflect what bot.py would actually do live — this is not a separate
reimplementation of either strategy.

Usage:
    pip install -r requirements-backtest.txt
    python backtest.py --smoke                       # ~3 day, 5-symbol sanity check first
    python backtest.py --months 6 --top-n 30 --out backtest_report.xlsx
    python backtest.py --engine gem --months 6 --out backtest_gem.xlsx
    python backtest.py --engine both --months 6 --out backtest_both.xlsx      # ZENITH + GEM
    python backtest.py --engine all --months 6 --out backtest_all.xlsx        # ZENITH + GEM + KRYPTIC

KRYPTIC (--engine kryptic / all) runs kryptic_strategy.py's
TradeLifecycleManager.open_trade() against a pandas DataFrame window per
symbol per bar -- much heavier per-evaluation than ZENITH/GEM's pure-Python
evaluate() functions, so a KRYPTIC-inclusive run is noticeably slower.
Narrow it with --kryptic-top-n or a coarser --every for a faster preview.

A full 6-month x 30-symbol run evaluates on the order of ~500k symbol-bars
in pure Python and can take a long time (the exact figure depends on your
machine — this was never benchmarked in the dev sandbox). Start with
--smoke to confirm everything works, then scale up; use --every 2-4 or a
smaller --top-n for a faster, coarser preview run.

What it does, in order:
  1. Builds cfg / FIB_CFG / GEM_CFG exactly like bot.py (env vars / .env /
     CLI overrides via bot.configure_strategy + bot.configure_gem_strategy)
     — so this reflects the live leverage bands, risk settings, and each
     engine's own filter thresholds.
  2. Picks a symbol universe per engine that's actually running: ZENITH uses
     top --top-n USDT perpetuals by *current* 24h quote volume >=
     MIN_QUOTE_VOLUME_USD; GEM uses its own (lower) GEM_MIN_QUOTE_VOLUME_USD
     floor and GEM_TOP_N cut — same as bot.py's live universes. Historical
     candles are fetched once for the union of both engines' symbols.
  3. Downloads --months of historical 15m + 1h candles per symbol (public
     Binance Futures klines, paginated), plus a warmup buffer before the
     start date so every evaluated bar has full lookback.
  4. Walks the same 15m timeline bot.py's scan loop would have walked: one
     "scan" per newly-closed 15m bar. Each running engine scores every
     symbol in its own universe on the engine's own rolling window
     (strategy.score_symbol for ZENITH, gem_strategy.evaluate for GEM),
     applying the same per-engine cooldown / max-posts-per-scan bot.py
     applies live (ZENITH also side-balances long/short; GEM just takes the
     top-N by score, matching bot.gem_scan_once). Both engines share one
     RiskGuard, so MAX_CONCURRENT_TRADES / MAX_CONCURRENT_SAME_SIDE and the
     drawdown-ceiling pause apply across them jointly, exactly like live.
  5. Simulates each posted signal forward bar-by-bar with the same
     risk_guard.RiskGuard._advance() logic live trades use (conservative:
     SL checked before TP within a bar), compounding one shared equity
     curve via RISK_EQUITY_PCT.
  6. Writes an .xlsx with Summary / Trades / Equity Curve / Monthly / Notes
     sheets. Trades carry an "engine" column; Summary breaks out win rate /
     avg R / profit factor per engine when more than one ran.

Known simplifications (also written into the output's Notes sheet):
  - Funding-rate veto is not modeled for either engine (no historical
    funding data fetched) — a minor soft filter, not the core edge.
  - CVD/OI absorption was never modeled, live or here (stub only).
  - Each engine's symbol universe is TODAY's qualifying set, held fixed
    across the whole lookback window (mild survivorship bias vs. the exact
    universe that would have been live-scanned that far back).
  - Fills assumed exact at the published average entry; no slippage or
    partial-fill modeling beyond the ladder-fill simulation RiskGuard
    already does live.
  - This always fetches from Binance, regardless of the live bot's
    DATA_SOURCE setting (e.g. blofin) — see README's "Price data source"
    section. Results approximate the strategies' edge, not an exact replay
    of what would have happened on a different exchange's price action.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

# bot._cfg() raises SystemExit without Telegram creds unless DRY_RUN is set.
# Set the default *before* importing bot so its own .env load (which uses
# setdefault-style, non-overriding semantics) can't clobber it.
os.environ.setdefault("DRY_RUN", "1")

import bot  # noqa: E402
import gem_strategy as gem  # noqa: E402
import kryptic_strategy as kryptic  # noqa: E402
from exchanges import load_universe  # noqa: E402
from risk_guard import RiskGuard  # noqa: E402
from risk_manager import TradeLifecycleManager  # noqa: E402
from strategy import (  # noqa: E402
    build_signal,
    btc_regime_from_candles,
    leverage_from_quality,
    score_symbol,
)

log = logging.getLogger("zenith.backtest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

UA = {"User-Agent": "zenith-backtest/1.0"}
INTERVAL_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}


async def fetch_klines_range(
    client: httpx.AsyncClient, symbol: str, interval: str, start_ms: int, end_ms: int, *, limit: int = 1500
) -> list[dict]:
    """Paginate Binance Futures klines across [start_ms, end_ms)."""
    out: list[dict] = []
    cur = start_ms
    step_ms = INTERVAL_MS[interval]
    while cur < end_ms:
        batch = None
        for attempt in range(5):
            try:
                r = await client.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": cur,
                        "endTime": end_ms,
                        "limit": limit,
                    },
                    headers=UA,
                    timeout=20,
                )
                if r.status_code in (429, 418):
                    wait = float(r.headers.get("Retry-After") or 5.0)
                    log.warning("%s %s rate-limited (%s) — sleeping %.1fs", symbol, interval, r.status_code, wait)
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                if attempt == 4:
                    log.warning("klines fetch failed for %s %s @ %s: %s", symbol, interval, cur, e)
                    return out
                await asyncio.sleep(1.5 * (attempt + 1))
        if batch is None:
            return out
        if not batch:
            break
        for k in batch:
            out.append(
                {
                    "ts": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        last_open = int(batch[-1][0])
        if len(batch) < limit:
            break
        cur = last_open + step_ms
        await asyncio.sleep(0.08)

    out.sort(key=lambda c: c["ts"])
    seen: set[int] = set()
    dedup = []
    for c in out:
        if c["ts"] in seen:
            continue
        seen.add(c["ts"])
        dedup.append(c)
    return [c for c in dedup if start_ms <= c["ts"] < end_ms]


def evaluate_one_backtest(
    ltf_window: list[dict], htf_window: list[dict] | None, ticker: dict, cfg: dict, btc_regime: str
) -> dict | None:
    """Historical-data twin of bot.evaluate_one(): same two-pass
    (prefilter then HTF-required) scoring, no network, no funding."""
    if len(ltf_window) < 50:
        return None
    sym = ticker["symbol"]
    pre = score_symbol(
        ltf_window,
        ticker,
        None,
        require_htf=False,
        entry_weights=cfg["entry_weights"],
        btc_regime=btc_regime,
        symbol=sym,
        allow_shorts=cfg["allow_shorts"],
        long_only=cfg["long_only"],
    )
    if not pre or pre["score"] < cfg["min_score"]:
        return None
    scored = score_symbol(
        ltf_window,
        ticker,
        htf_window,
        require_htf=bool(htf_window),
        entry_weights=cfg["entry_weights"],
        btc_regime=btc_regime,
        symbol=sym,
        allow_shorts=cfg["allow_shorts"],
        long_only=cfg["long_only"],
    )
    if not scored or scored["score"] < cfg["min_score"]:
        return None
    scored["last_ts"] = ltf_window[-1].get("ts")
    return scored


def evaluate_one_backtest_gem(ltf_window: list[dict]) -> dict | None:
    """Historical-data twin of bot.evaluate_one_gem(): no network, no
    funding data (same simplification ZENITH's backtest path already has)."""
    if len(ltf_window) < 50:
        return None
    result = gem.evaluate(gem.from_repo_candles(ltf_window), None)
    if not result.get("ok"):
        return None
    result["last_ts"] = ltf_window[-1].get("ts")
    return result


def evaluate_one_backtest_kryptic(
    ltf_window: list[dict], btc_window: list[dict], trade_manager: TradeLifecycleManager,
):
    """Historical-data twin of bot.evaluate_one_kryptic(): no network, no
    funding data. Builds the pandas DataFrames TradeLifecycleManager.open_trade()
    needs directly from the same rolling windows ZENITH/GEM already slice."""
    if len(ltf_window) < 80 or len(btc_window) < 50:
        return None, None
    df = kryptic.candles_to_df(ltf_window)
    btc_df = kryptic.candles_to_df(btc_window)
    return trade_manager.open_trade(df, btc_df, None, htf_df=None)


async def build_universe(client: httpx.AsyncClient, cfg: dict, args: argparse.Namespace) -> tuple[list[str], dict]:
    """ZENITH's universe: top --top-n by 24h quote volume >= MIN_QUOTE_VOLUME_USD."""
    universe = await load_universe(client, cfg["min_quote_vol"])
    max_lev = universe.get("max_lev") or {}
    if args.symbols.strip():
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        return syms, max_lev
    deny = cfg["denylist"]
    tickers = [t for t in (universe.get("tickers") or []) if t["symbol"] not in deny]
    top_n = 5 if args.smoke else args.top_n
    # --top-n <= 0 means "all symbols meeting MIN_QUOTE_VOLUME_USD", not just
    # a fixed top slice by volume.
    syms = [t["symbol"] for t in tickers] if top_n <= 0 else [t["symbol"] for t in tickers[:top_n]]
    return syms, max_lev


async def build_universe_gem(client: httpx.AsyncClient, cfg: dict, args: argparse.Namespace) -> tuple[list[str], dict]:
    """GEM's own (typically lower-volume-floor, wider) universe, mirroring
    bot.gem_scan_once's live universe selection."""
    universe = await load_universe(client, cfg["gem_min_quote_vol"])
    max_lev = universe.get("max_lev") or {}
    if args.symbols.strip():
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        return syms, max_lev
    deny = cfg["denylist"]
    tickers = [t for t in (universe.get("tickers") or []) if t["symbol"] not in deny]
    tickers.sort(key=lambda t: t.get("quote_vol", 0), reverse=True)
    top_n = 5 if args.smoke else cfg["gem_top_n"]
    syms = [t["symbol"] for t in tickers] if top_n <= 0 else [t["symbol"] for t in tickers[:top_n]]
    return syms, max_lev


async def build_universe_kryptic(client: httpx.AsyncClient, cfg: dict, args: argparse.Namespace) -> tuple[list[str], dict]:
    """KRYPTIC's own universe, mirroring bot.kryptic_scan_once's live universe selection."""
    universe = await load_universe(client, cfg["kryptic_min_quote_vol"])
    max_lev = universe.get("max_lev") or {}
    if args.symbols.strip():
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        return syms, max_lev
    deny = cfg["denylist"]
    tickers = [t for t in (universe.get("tickers") or []) if t["symbol"] not in deny]
    tickers.sort(key=lambda t: t.get("quote_vol", 0), reverse=True)
    top_n = 5 if args.smoke else cfg["kryptic_top_n"]
    syms = [t["symbol"] for t in tickers] if top_n <= 0 else [t["symbol"] for t in tickers[:top_n]]
    return syms, max_lev


def _fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _progress(step: int, total: int, t0: float, risk: RiskGuard) -> None:
    elapsed = time.time() - t0
    pct = 100.0 * step / max(1, total)
    rate = step / elapsed if elapsed > 0 else 0
    eta = (total - step) / rate if rate > 0 else 0
    log.info(
        "progress %d/%d (%.1f%%) equity=%.2f dd=%.2f%% open=%d elapsed=%.0fs eta=%.0fs",
        step,
        total,
        pct,
        risk.equity,
        risk.drawdown_pct(),
        risk.concurrent_count(),
        elapsed,
        eta,
    )


async def run(args: argparse.Namespace) -> None:
    run_zenith = args.engine in ("zenith", "both", "all")
    run_gem = args.engine in ("gem", "both", "all")
    run_kryptic = args.engine in ("kryptic", "all")

    cfg = bot._cfg()
    # Supply-side overrides: widen how many of the *already-qualifying*
    # signals get captured (universe breadth, per-scan/concurrent caps,
    # cooldown) without touching MIN_SCORE or any entry filter. These are
    # CLI-only so a backtest run can experiment without editing .env.
    if args.max_posts_per_scan is not None:
        cfg["max_posts_per_scan"] = args.max_posts_per_scan
    if args.max_concurrent_trades is not None:
        cfg["max_concurrent_trades"] = args.max_concurrent_trades
    if args.max_concurrent_same_side is not None:
        cfg["max_concurrent_same_side"] = args.max_concurrent_same_side
    if args.cooldown_minutes is not None:
        cfg["cooldown_min"] = args.cooldown_minutes
    if args.min_quote_vol is not None:
        cfg["min_quote_vol"] = args.min_quote_vol
    if args.dd_pause_cooldown_hours is not None:
        cfg["dd_pause_cooldown_hours"] = args.dd_pause_cooldown_hours
    if args.gem_max_posts_per_scan is not None:
        cfg["gem_max_posts_per_scan"] = args.gem_max_posts_per_scan
    if args.gem_cooldown_minutes is not None:
        cfg["gem_cooldown_min"] = args.gem_cooldown_minutes
    if args.gem_min_quote_vol is not None:
        cfg["gem_min_quote_vol"] = args.gem_min_quote_vol
    if args.gem_top_n is not None:
        cfg["gem_top_n"] = args.gem_top_n
    if args.kryptic_max_posts_per_scan is not None:
        cfg["kryptic_max_posts_per_scan"] = args.kryptic_max_posts_per_scan
    if args.kryptic_cooldown_minutes is not None:
        cfg["kryptic_cooldown_min"] = args.kryptic_cooldown_minutes
    if args.kryptic_min_quote_vol is not None:
        cfg["kryptic_min_quote_vol"] = args.kryptic_min_quote_vol
    if args.kryptic_top_n is not None:
        cfg["kryptic_top_n"] = args.kryptic_top_n

    profile = bot.configure_strategy(cfg) if run_zenith else cfg.get("strategy_profile", "smc_lite")
    if run_gem:
        bot.configure_gem_strategy()
    if run_kryptic:
        bot.configure_kryptic_strategy()
    kryptic_trade_manager = TradeLifecycleManager(
        initial_sl_atr_mult=cfg["kryptic_initial_sl_atr_mult"],
        tp1_atr_mult=cfg["kryptic_tp1_atr_mult"],
        tp2_atr_mult=cfg["kryptic_tp2_atr_mult"],
        tp3_atr_mult=cfg["kryptic_tp3_atr_mult"],
        tp4_atr_mult=cfg["kryptic_tp4_atr_mult"],
        tp5_atr_mult=cfg["kryptic_tp5_atr_mult"],
    ) if run_kryptic else None

    if run_zenith:
        log.info(
            "ZENITH: profile=%s lev=%s-%sx min_score=%s risk_equity_pct=%s%% dd_ceiling=%s%% "
            "max_posts_per_scan=%s max_concurrent=%s/%s cooldown_min=%s min_quote_vol=%s",
            profile,
            cfg["leverage_min"],
            cfg["leverage_max"],
            cfg["min_score"],
            cfg["risk_equity_pct"],
            cfg["dd_ceiling_pct"],
            cfg["max_posts_per_scan"],
            cfg["max_concurrent_trades"],
            cfg["max_concurrent_same_side"],
            cfg["cooldown_min"],
            cfg["min_quote_vol"],
        )
    if run_gem:
        log.info(
            "GEM: lev=%s-%sx min_score=%s max_posts_per_scan=%s cooldown_min=%s min_quote_vol=%s top_n=%s",
            cfg["gem_leverage_min"],
            cfg["gem_leverage_max"],
            gem.GEM_CFG["MIN_SCORE"],
            cfg["gem_max_posts_per_scan"],
            cfg["gem_cooldown_min"],
            cfg["gem_min_quote_vol"],
            cfg["gem_top_n"],
        )
    if run_kryptic:
        log.info(
            "KRYPTIC: lev=%s-%sx score(fixed)=%s max_posts_per_scan=%s cooldown_min=%s min_quote_vol=%s top_n=%s",
            cfg["kryptic_leverage_min"],
            cfg["kryptic_leverage_max"],
            kryptic.KRYPTIC_CFG["SCORE"],
            cfg["kryptic_max_posts_per_scan"],
            cfg["kryptic_cooldown_min"],
            cfg["kryptic_min_quote_vol"],
            cfg["kryptic_top_n"],
        )

    months = 0.1 if args.smoke else args.months
    end_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = end_ms - int(months * 30.4375 * 24 * 3600 * 1000)
    warmup_ms = int(args.warmup_days * 24 * 3600 * 1000)
    fetch_start_ms = start_ms - warmup_ms

    ltf_by_sym: dict[str, list[dict]] = {}
    htf_by_sym: dict[str, list[dict]] = {}
    zenith_max_lev: dict[str, int] = {}
    gem_max_lev: dict[str, int] = {}
    kryptic_max_lev: dict[str, int] = {}
    zenith_symbols: list[str] = []
    gem_symbols: list[str] = []
    kryptic_symbols: list[str] = []

    async with httpx.AsyncClient(headers=UA) as client:
        if run_zenith:
            zenith_symbols, zenith_max_lev = await build_universe(client, cfg, args)
            log.info("ZENITH universe: %d symbols: %s", len(zenith_symbols), ", ".join(zenith_symbols))
        if run_gem:
            gem_symbols, gem_max_lev = await build_universe_gem(client, cfg, args)
            log.info("GEM universe: %d symbols: %s", len(gem_symbols), ", ".join(gem_symbols))
        if run_kryptic:
            kryptic_symbols, kryptic_max_lev = await build_universe_kryptic(client, cfg, args)
            log.info("KRYPTIC universe: %d symbols: %s", len(kryptic_symbols), ", ".join(kryptic_symbols))

        all_symbols = sorted(set(zenith_symbols) | set(gem_symbols) | set(kryptic_symbols))
        if not all_symbols:
            raise SystemExit("No symbols in universe — check network access / volume floors")

        fetch_symbols = list(all_symbols)
        # KRYPTIC's RegimeFilter always needs BTC context (not gated by
        # cfg["btc_regime"], which is a ZENITH-only knob).
        if (run_zenith and cfg.get("btc_regime") or run_kryptic) and "BTCUSDT" not in fetch_symbols:
            fetch_symbols.append("BTCUSDT")

        sem = asyncio.Semaphore(args.fetch_concurrency)

        async def fetch_pair(sym: str) -> tuple[str, list[dict], list[dict]]:
            async with sem:
                ltf = await fetch_klines_range(client, sym, "15m", fetch_start_ms, end_ms)
                htf = await fetch_klines_range(client, sym, "1h", fetch_start_ms - 10 * 24 * 3600 * 1000, end_ms)
            return sym, ltf, htf

        log.info("Downloading historical klines for %d symbols (this can take a while)...", len(fetch_symbols))
        results = await asyncio.gather(*(fetch_pair(s) for s in fetch_symbols))

        for sym, ltf, htf in results:
            if len(ltf) < 250:
                log.warning("Skipping %s: only %d LTF candles fetched (need warmup+window)", sym, len(ltf))
                continue
            ltf_by_sym[sym] = ltf
            htf_by_sym[sym] = htf

    all_symbols = [s for s in all_symbols if s in ltf_by_sym]
    zenith_symbols = [s for s in zenith_symbols if s in ltf_by_sym]
    gem_symbols = [s for s in gem_symbols if s in ltf_by_sym]
    kryptic_symbols = [s for s in kryptic_symbols if s in ltf_by_sym]
    if not all_symbols:
        raise SystemExit("No symbol had enough historical data to backtest")
    btc_htf = htf_by_sym.get("BTCUSDT")

    anchor = max(all_symbols, key=lambda s: len(ltf_by_sym[s]))
    master_ts_all = [c["ts"] for c in ltf_by_sym[anchor] if c["ts"] >= start_ms]
    master_ts_all = master_ts_all[:: max(1, args.every)]
    if not master_ts_all:
        raise SystemExit("Empty master timeline — widen --months or check fetched data")
    log.info(
        "Master timeline: %d bars, %s -> %s",
        len(master_ts_all),
        _fmt_ts(master_ts_all[0]),
        _fmt_ts(master_ts_all[-1]),
    )

    idx = {s: 0 for s in all_symbols}
    htf_idx = {s: 0 for s in all_symbols}
    btc_htf_idx = 0
    # Cooldown is per (symbol, direction) for every engine -- a repeat
    # signal in the SAME direction on a symbol is blocked until it expires,
    # but the OPPOSITE direction is never blocked by it (matches
    # bot.py's live scan_once/gem_scan_once/kryptic_scan_once).
    side_cooldown: dict[str, int] = {}
    gem_side_cooldown: dict[str, int] = {}
    kryptic_side_cooldown: dict[str, int] = {}
    cooldown_ms = cfg["cooldown_min"] * 60 * 1000
    gem_cooldown_ms = cfg["gem_cooldown_min"] * 60 * 1000
    kryptic_cooldown_ms = cfg["kryptic_cooldown_min"] * 60 * 1000

    out_path = Path(args.out)
    risk_state_path = out_path.with_name(out_path.stem + "_risk_state.json")
    risk = RiskGuard(path=risk_state_path, start_equity=args.start_equity)
    # Always start this run from a clean slate, even if a stale state file
    # from a previous invocation (same --out) happens to exist.
    risk.trades = {}
    risk.equity = args.start_equity
    risk.peak = args.start_equity
    risk.paused = False
    risk.closed_count = 0

    trade_records: dict[str, dict] = {}
    equity_curve: list[dict] = []
    pause_events = 0
    was_paused = False

    t0 = time.time()
    for step, master_ts in enumerate(master_ts_all):
        # 1) advance every open (simulated) trade with each symbol's newly closed bar
        for sym in all_symbols:
            candles = ltf_by_sym[sym]
            n = len(candles)
            i = idx[sym]
            while i < n and candles[i]["ts"] <= master_ts:
                i += 1
            new_bar = candles[i - 1] if i > idx[sym] else None
            idx[sym] = i

            hcandles = htf_by_sym[sym]
            m = len(hcandles)
            j = htf_idx[sym]
            while j < m and hcandles[j]["ts"] <= master_ts:
                j += 1
            htf_idx[sym] = j

            if new_bar is None:
                continue
            for key in [k for k, tr in risk.trades.items() if tr["symbol"] == sym]:
                tr = risk.trades[key]
                done = risk._advance(tr, new_bar)
                tr["last_ts"] = new_bar["ts"]
                if done:
                    risk._close_trade(tr)
                    rec = trade_records.get(key)
                    if rec is not None:
                        rec["close_ts"] = new_bar["ts"]
                        rec["close_reason"] = tr.get("close_reason")
                        rec["r_realized"] = tr.get("r_realized")
                        rec["bars_held"] = tr.get("bars_elapsed")
                        rec["equity_after"] = risk.equity
                        rec["fill_pct"] = tr.get("fill_pct")
                        rec["entry_avg_filled"] = tr.get("eavg")
                    equity_curve.append(
                        {"ts": new_bar["ts"], "equity": risk.equity, "drawdown_pct": risk.drawdown_pct()}
                    )
                    del risk.trades[key]

        btc_window: list[dict] = []
        if (run_zenith and cfg.get("btc_regime") or run_kryptic) and btc_htf:
            while btc_htf_idx < len(btc_htf) and btc_htf[btc_htf_idx]["ts"] <= master_ts:
                btc_htf_idx += 1
            btc_window = btc_htf[max(0, btc_htf_idx - 240) : btc_htf_idx]
        btc_regime = btc_regime_from_candles(btc_window) if (run_zenith and cfg.get("btc_regime") and btc_window) else "neutral"

        risk.update_pause_state(
            cfg["dd_ceiling_pct"],
            cfg["dd_resume_pct"],
            now_ms=master_ts,
            cooldown_ms=cfg["dd_pause_cooldown_hours"] * 3_600_000,
        )
        if risk.paused and not was_paused:
            pause_events += 1
        was_paused = risk.paused

        # 2a) ZENITH scan: every symbol in its universe, on its own rolling
        # 240-bar window, same as bot.evaluate_one — side-balanced picks.
        if run_zenith:
            candidates = []
            for sym in zenith_symbols:
                i = idx[sym]
                window = ltf_by_sym[sym][max(0, i - 240) : i]
                if len(window) < 50:
                    continue
                j = htf_idx[sym]
                htf_window = htf_by_sym[sym][max(0, j - 240) : j] or None
                ticker = {"symbol": sym}
                scored = evaluate_one_backtest(window, htf_window, ticker, cfg, btc_regime)
                if not scored:
                    continue
                if cfg.get("long_only") and scored["side"] != "LONG":
                    continue
                if not cfg.get("allow_shorts", True) and scored["side"] != "LONG":
                    continue
                side_key = f"{sym}:{scored['side']}"
                if side_cooldown.get(side_key, 0) > master_ts:
                    continue
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
                exch_max = int(zenith_max_lev.get(sym) or cfg["leverage_max"])
                if exch_max < cfg["leverage_min"]:
                    continue
                lev = max(1, min(lev, exch_max, cfg["leverage_max"]))
                sig = build_signal(
                    sym,
                    scored,
                    leverage=lev,
                    timeframe=cfg["timeframe"],
                    entry_weights=cfg["entry_weights"],
                    tp_weights=cfg["tp_weights"],
                )
                candidates.append((scored, sig, lev, sym))

            if candidates:
                candidates.sort(key=lambda x: float(x[0]["score"]), reverse=True)
                picks = []
                if cfg.get("side_balance", True) and not cfg.get("long_only"):
                    best_long = next((c for c in candidates if c[0]["side"] == "LONG"), None)
                    best_short = next((c for c in candidates if c[0]["side"] == "SHORT"), None)
                    if best_long:
                        picks.append(best_long)
                    if best_short:
                        picks.append(best_short)
                    for c in candidates:
                        if len(picks) >= cfg["max_posts_per_scan"]:
                            break
                        if c in picks:
                            continue
                        picks.append(c)
                else:
                    picks = candidates[: cfg["max_posts_per_scan"]]
                picks = picks[: cfg["max_posts_per_scan"]]

                for scored, sig, lev, sym in picks:
                    if not risk.can_open(sig.side, cfg):
                        continue
                    key = risk.open_trade(
                        symbol=sym,
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
                        last_ts=master_ts,
                    )
                    if key is None:
                        continue
                    side_cooldown[f"{sym}:{sig.side}"] = master_ts + cooldown_ms
                    trade_records[key] = {
                        "key": key,
                        "engine": "ZENITH",
                        "symbol": sym,
                        "side": sig.side,
                        "score": sig.score,
                        "leverage": lev,
                        "open_ts": master_ts,
                        "entry_avg_intended": sig.extras.get("avg_entry"),
                        "entry_avg_filled": None,
                        "fill_pct": None,
                        "sl": sig.sl,
                        "be_price": sig.extras.get("lock_sl_after_tp1"),
                        "tp1": sig.tps[0] if sig.tps else None,
                        "tp5": sig.tps[-1] if sig.tps else None,
                        "risk_pct": sig.extras.get("risk_pct"),
                        "htf_aligned": sig.extras.get("htf_aligned"),
                        "vol_ratio": sig.extras.get("vol_ratio"),
                        "stretch_atr": sig.extras.get("stretch_atr"),
                        "impulse_vol_ratio": sig.extras.get("impulse_vol_ratio"),
                        "smc_bias": sig.extras.get("smc_bias"),
                        "smc_bos": sig.extras.get("smc_bos"),
                        "smc_poi_inside": sig.extras.get("smc_poi_inside"),
                        "smc_poi_near": sig.extras.get("smc_poi_near"),
                        "smc_pd": sig.extras.get("smc_pd"),
                        "smc_choch_ltf": sig.extras.get("smc_choch_ltf"),
                        "close_ts": None,
                        "close_reason": None,
                        "r_realized": None,
                        "bars_held": None,
                        "equity_after": None,
                    }

        # 2b) GEM scan: its own universe/window, no side-balancing or HTF —
        # top-N by score, same as bot.gem_scan_once.
        if run_gem:
            gem_candidates = []
            for sym in gem_symbols:
                i = idx[sym]
                window = ltf_by_sym[sym][max(0, i - 240) : i]
                if len(window) < 50:
                    continue
                result = evaluate_one_backtest_gem(window)
                if not result:
                    continue
                if gem_side_cooldown.get(f"{sym}:{result['dir']}", 0) > master_ts:
                    continue
                lev = leverage_from_quality(
                    result["nota"],
                    side=result["dir"],
                    lev_min=cfg["gem_leverage_min"],
                    lev_max=cfg["gem_leverage_max"],
                    rpct=result["Rpct"],
                    ratr=result["Ratr"],
                    vol_ratio=result["volRel"],
                )
                exch_max = int(gem_max_lev.get(sym) or cfg["gem_leverage_max"])
                if exch_max < cfg["gem_leverage_min"]:
                    continue
                lev = max(1, min(lev, exch_max, cfg["gem_leverage_max"]))
                sig = gem.build_signal(sym, result, leverage=lev, timeframe=cfg["timeframe"])
                gem_candidates.append((result, sig, lev, sym))

            if gem_candidates:
                gem_candidates.sort(key=lambda x: float(x[0]["nota"]), reverse=True)
                gem_picks = gem_candidates[: cfg["gem_max_posts_per_scan"]]

                for result, sig, lev, sym in gem_picks:
                    if not risk.can_open(sig.side, cfg):
                        continue
                    key = risk.open_trade(
                        symbol=sym,
                        side=sig.side,
                        timeframe=cfg["timeframe"],
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
                        last_ts=master_ts,
                    )
                    if key is None:
                        continue
                    gem_side_cooldown[f"{sym}:{sig.side}"] = master_ts + gem_cooldown_ms
                    trade_records[key] = {
                        "key": key,
                        "engine": "GEM",
                        "symbol": sym,
                        "side": sig.side,
                        "score": sig.score,
                        "leverage": lev,
                        "open_ts": master_ts,
                        "entry_avg_intended": sig.extras.get("avg_entry"),
                        "entry_avg_filled": None,
                        "fill_pct": None,
                        "sl": sig.sl,
                        "be_price": None,
                        "tp1": sig.tps[0] if sig.tps else None,
                        "tp5": sig.tps[-1] if sig.tps else None,
                        "risk_pct": sig.extras.get("risk_pct"),
                        "htf_aligned": None,
                        "vol_ratio": sig.extras.get("vol_ratio"),
                        "stretch_atr": None,
                        "impulse_vol_ratio": None,
                        "smc_bias": sig.extras.get("smc_trend"),
                        "smc_bos": sig.extras.get("smc_bos"),
                        "smc_poi_inside": None,
                        "smc_poi_near": None,
                        "smc_pd": None,
                        "smc_choch_ltf": sig.extras.get("smc_choch"),
                        "close_ts": None,
                        "close_reason": None,
                        "r_realized": None,
                        "bars_held": None,
                        "equity_after": None,
                    }

        # 2c) KRYPTIC scan: its own universe/window, TradeLifecycleManager's
        # RegimeFilter+DirectionEngine gate (pass/fail, no score) -- picks
        # ranked by tightest risk_atr_multiple, same as bot.kryptic_scan_once.
        if run_kryptic:
            kryptic_candidates = []
            for sym in kryptic_symbols:
                i = idx[sym]
                window = ltf_by_sym[sym][max(0, i - 240) : i]
                position, diagnostics = evaluate_one_backtest_kryptic(window, btc_window, kryptic_trade_manager)
                if position is None:
                    continue
                if kryptic_side_cooldown.get(f"{sym}:{position.direction}", 0) > master_ts:
                    continue
                lev = leverage_from_quality(
                    kryptic.KRYPTIC_CFG["SCORE"],
                    side=position.direction,
                    lev_min=cfg["kryptic_leverage_min"],
                    lev_max=cfg["kryptic_leverage_max"],
                    ratr=diagnostics.get("risk_atr_multiple"),
                )
                exch_max = int(kryptic_max_lev.get(sym) or cfg["kryptic_leverage_max"])
                if exch_max < cfg["kryptic_leverage_min"]:
                    continue
                lev = max(1, min(lev, exch_max, cfg["kryptic_leverage_max"]))
                sig = kryptic.build_signal(
                    sym, position, diagnostics,
                    leverage=lev, timeframe=cfg["timeframe"], reference=float(window[-1]["close"]),
                )
                kryptic_candidates.append((diagnostics, sig, lev, sym))

            if kryptic_candidates:
                kryptic_candidates.sort(key=lambda x: float(x[0].get("risk_atr_multiple") or 999.0))
                kryptic_picks = kryptic_candidates[: cfg["kryptic_max_posts_per_scan"]]

                for diagnostics, sig, lev, sym in kryptic_picks:
                    if not risk.can_open(sig.side, cfg):
                        continue
                    key = risk.open_trade(
                        symbol=sym,
                        side=sig.side,
                        timeframe=cfg["timeframe"],
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
                        last_ts=master_ts,
                    )
                    if key is None:
                        continue
                    kryptic_side_cooldown[f"{sym}:{sig.side}"] = master_ts + kryptic_cooldown_ms
                    trade_records[key] = {
                        "key": key,
                        "engine": "KRYPTIC",
                        "symbol": sym,
                        "side": sig.side,
                        "score": sig.score,
                        "leverage": lev,
                        "open_ts": master_ts,
                        "entry_avg_intended": None,
                        "entry_avg_filled": None,
                        "fill_pct": None,
                        "sl": sig.sl,
                        "be_price": None,
                        "tp1": sig.tps[0] if sig.tps else None,
                        "tp5": sig.tps[-1] if sig.tps else None,
                        "risk_pct": None,
                        "htf_aligned": None,
                        "vol_ratio": None,
                        "stretch_atr": None,
                        "impulse_vol_ratio": None,
                        "smc_bias": None,
                        "smc_bos": None,
                        "smc_poi_inside": None,
                        "smc_poi_near": None,
                        "smc_pd": None,
                        "smc_choch_ltf": None,
                        "close_ts": None,
                        "close_reason": None,
                        "r_realized": None,
                        "bars_held": None,
                        "equity_after": None,
                    }

        if step % 500 == 0:
            _progress(step, len(master_ts_all), t0, risk)

    # time-exit anything still open (or still filling) at the end of the window
    for key, tr in list(risk.trades.items()):
        sym = tr["symbol"]
        candles = ltf_by_sym.get(sym) or []
        if not candles:
            continue
        last_bar = candles[-1]
        bull = tr["side"] == "LONG"
        if tr.get("phase") == "FILLING":
            # Ran out of data before we could resolve whether/how much of
            # the ladder would ever fill -- inconclusive, not a real outcome.
            tr["close_reason"] = "END_OF_BACKTEST_UNFILLED"
            tr["r_realized"] = 0.0
            tr["fill_pct"] = 0.0
        else:
            w = tr["remaining_weight"]
            r_mult = (
                ((last_bar["close"] - tr["eavg"]) / tr["risk_px"])
                if bull
                else ((tr["eavg"] - last_bar["close"]) / tr["risk_px"])
            )
            tr["r_realized"] += (w / 100.0) * r_mult
            tr["close_reason"] = "END_OF_BACKTEST"
        risk._close_trade(tr)
        rec = trade_records.get(key)
        if rec is not None:
            rec["close_ts"] = last_bar["ts"]
            rec["close_reason"] = tr["close_reason"]
            rec["r_realized"] = tr["r_realized"]
            rec["bars_held"] = tr.get("bars_elapsed")
            rec["equity_after"] = risk.equity
            rec["fill_pct"] = tr.get("fill_pct")
            rec["entry_avg_filled"] = tr.get("eavg")
        equity_curve.append({"ts": last_bar["ts"], "equity": risk.equity, "drawdown_pct": risk.drawdown_pct()})
        del risk.trades[key]

    log.info("Backtest done in %.1fs — %d trades closed, final equity=%.2f", time.time() - t0, len(trade_records), risk.equity)

    # Checkpoint the raw results *before* touching Excel/pandas: the walk-
    # forward simulation above is the expensive part (network + CPU, can
    # run tens of minutes at full scale). If the Excel write fails for any
    # reason (missing openpyxl, disk full, a locked file on Windows), this
    # checkpoint means that work doesn't have to be redone -- rerun with
    # --from-raw <this file> to regenerate the report from it directly.
    raw_path = Path(args.out).with_suffix("").with_name(Path(args.out).stem + "_raw.json")
    cfg_safe = {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool)) or v is None}
    try:
        raw_path.write_text(
            json.dumps(
                {
                    "cfg": cfg_safe,
                    "profile": profile,
                    "engine_mode": args.engine,
                    "zenith_symbols": zenith_symbols,
                    "gem_symbols": gem_symbols,
                    "kryptic_symbols": kryptic_symbols,
                    "trade_records": trade_records,
                    "equity_curve": equity_curve,
                    "pause_events": pause_events,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "start_equity": args.start_equity,
                }
            )
        )
        log.info("Checkpointed raw results to %s", raw_path)
    except Exception:
        log.warning("Failed to write raw results checkpoint", exc_info=True)

    try:
        write_report(
            args, cfg, profile, args.engine, zenith_symbols, gem_symbols, kryptic_symbols,
            trade_records, equity_curve, pause_events, start_ms, end_ms,
        )
    except Exception:
        log.error(
            "Writing the Excel report failed, but results are safe in %s -- "
            "fix the issue (e.g. `pip install openpyxl`) then rerun with "
            "`--from-raw %s` to regenerate the report without redoing the backtest.",
            raw_path, raw_path,
        )
        raise

    try:
        risk_state_path.unlink(missing_ok=True)
    except Exception:
        pass


def write_report(
    args: argparse.Namespace,
    cfg: dict,
    profile: str,
    engine_mode: str,
    zenith_symbols: list[str],
    gem_symbols: list[str],
    kryptic_symbols: list[str],
    trade_records: dict[str, dict],
    equity_curve: list[dict],
    pause_events: int,
    start_ms: int,
    end_ms: int,
) -> None:

    trades = [dict(r) for r in trade_records.values() if r["close_ts"] is not None]
    trades.sort(key=lambda r: r["open_ts"])
    for r in trades:
        r.setdefault("engine", "ZENITH")  # backward compat with pre-GEM raw.json checkpoints
        r["open_time"] = _fmt_ts(r["open_ts"])
        r["close_time"] = _fmt_ts(r["close_ts"])
        r["hold_hours"] = round((r["close_ts"] - r["open_ts"]) / 3_600_000.0, 2)
        r["r_realized"] = round(float(r["r_realized"] or 0.0), 3)
        fill_pct = float(r.get("fill_pct") or 0.0)
        r["fill_pct"] = round(fill_pct, 1)
        # Matches RiskGuard._close_trade's actual equity formula: only the
        # fraction of the ladder that filled was ever really at risk.
        r["equity_pct_change"] = round(
            (cfg["risk_equity_pct"] / 100.0) * (fill_pct / 100.0) * r["r_realized"] * 100.0, 4
        )

    df_trades = pd.DataFrame(
        trades,
        columns=[
            "engine", "open_time", "symbol", "side", "score", "leverage", "entry_avg_intended", "entry_avg_filled",
            "fill_pct", "sl", "be_price", "tp1", "tp5", "risk_pct", "htf_aligned", "vol_ratio",
            "stretch_atr", "impulse_vol_ratio", "smc_bias", "smc_bos", "smc_poi_inside", "smc_poi_near",
            "smc_pd", "smc_choch_ltf", "close_time", "close_reason", "bars_held", "hold_hours",
            "r_realized", "equity_pct_change", "equity_after",
        ],
    )

    # NO_FILL / END_OF_BACKTEST_UNFILLED signals never became a real
    # position (fill_pct == 0) -- exclude them from win-rate/R/profit-factor
    # stats (they're neither a win nor a loss) but keep them visible in the
    # Trades sheet and report their count separately for transparency.
    filled = [r for r in trades if (r.get("fill_pct") or 0) > 0]
    no_fill = [r for r in trades if not (r.get("fill_pct") or 0) > 0]

    def perf_stats(rows: list[dict]) -> dict:
        n = len(rows)
        wins = [r for r in rows if r["r_realized"] > 0]
        losses = [r for r in rows if r["r_realized"] <= 0]
        gross_win = sum(r["r_realized"] for r in wins)
        gross_loss = abs(sum(r["r_realized"] for r in losses))
        return {
            "trades": n,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(100.0 * len(wins) / n, 2) if n else 0.0,
            "avg_R": round(sum(r["r_realized"] for r in rows) / n, 3) if n else 0.0,
            "avg_win_R": round(sum(r["r_realized"] for r in wins) / len(wins), 3) if wins else 0.0,
            "avg_loss_R": round(sum(r["r_realized"] for r in losses) / len(losses), 3) if losses else 0.0,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else ("inf" if gross_win > 0 else 0.0),
            "avg_leverage": round(sum(r["leverage"] or 0 for r in rows) / n, 1) if n else 0.0,
            "avg_hold_hours": round(sum(r["hold_hours"] for r in rows) / n, 2) if n else 0.0,
            "avg_fill_pct": round(sum(r["fill_pct"] for r in rows) / n, 1) if n else 0.0,
        }

    overall = perf_stats(filled)
    n = overall["trades"]

    start_equity = args.start_equity
    final_equity = trades[-1]["equity_after"] if trades else start_equity
    total_return_pct = (final_equity / start_equity - 1.0) * 100.0
    max_dd = max((e["drawdown_pct"] for e in equity_curve), default=0.0)
    months_actual = (end_ms - start_ms) / (30.4375 * 24 * 3600 * 1000)
    annualized_return_pct = (
        ((final_equity / start_equity) ** (12.0 / months_actual) - 1.0) * 100.0
        if months_actual > 0 and final_equity > 0
        else None
    )

    def side_stats(side: str) -> dict:
        rows = [r for r in filled if r["side"] == side]
        w = [r for r in rows if r["r_realized"] > 0]
        return {
            "side": side,
            "trades": len(rows),
            "win_rate_pct": round(100.0 * len(w) / len(rows), 2) if rows else 0.0,
            "avg_R": round(sum(r["r_realized"] for r in rows) / len(rows), 3) if rows else 0.0,
        }

    engines_present = sorted({r["engine"] for r in filled}) or sorted({r["engine"] for r in trades})

    summary_rows = [
        ("Backtest window (UTC)", f"{_fmt_ts(start_ms)} -> {_fmt_ts(end_ms)}"),
        ("Months", round(months_actual, 2)),
        ("Engine(s) run", engine_mode),
        ("Strategy profile (ZENITH)", profile),
    ]
    if zenith_symbols:
        summary_rows += [
            ("ZENITH universe size", len(zenith_symbols)),
            ("ZENITH universe symbols", ", ".join(zenith_symbols)),
            ("ZENITH leverage band", f"{cfg['leverage_min']}-{cfg['leverage_max']}x"),
            ("ZENITH MIN_SCORE", cfg["min_score"]),
        ]
    if gem_symbols:
        summary_rows += [
            ("GEM universe size", len(gem_symbols)),
            ("GEM universe symbols", ", ".join(gem_symbols)),
            ("GEM leverage band", f"{cfg['gem_leverage_min']}-{cfg['gem_leverage_max']}x"),
            ("GEM MIN_SCORE", gem.GEM_CFG["MIN_SCORE"]),
        ]
    if kryptic_symbols:
        summary_rows += [
            ("KRYPTIC universe size", len(kryptic_symbols)),
            ("KRYPTIC universe symbols", ", ".join(kryptic_symbols)),
            ("KRYPTIC leverage band", f"{cfg['kryptic_leverage_min']}-{cfg['kryptic_leverage_max']}x"),
            ("KRYPTIC SCORE (fixed -- feeds leverage sizing only, entry gate is pass/fail)", kryptic.KRYPTIC_CFG["SCORE"]),
        ]
    summary_rows += [
        ("RISK_EQUITY_PCT per trade (shared)", cfg["risk_equity_pct"]),
        ("DD_CEILING_PCT / DD_RESUME_PCT (shared)", f"{cfg['dd_ceiling_pct']} / {cfg['dd_resume_pct']}"),
        ("MAX_CONCURRENT_TRADES / SAME_SIDE (shared)", f"{cfg['max_concurrent_trades']} / {cfg['max_concurrent_same_side']}"),
        ("", ""),
        ("Signals generated", len(trades)),
        ("Signals never filled (cancelled/no data)", len(no_fill)),
        ("Trades (filled, at least partially)", n),
        ("Avg fill %% of intended ladder", overall["avg_fill_pct"]),
        ("Wins", overall["wins"]),
        ("Losses", overall["losses"]),
        ("Win rate %", overall["win_rate_pct"]),
        ("Avg R per trade", overall["avg_R"]),
        ("Avg winning R", overall["avg_win_R"]),
        ("Avg losing R", overall["avg_loss_R"]),
        ("Profit factor", overall["profit_factor"]),
        ("Avg leverage used", overall["avg_leverage"]),
        ("Avg hold (hours)", overall["avg_hold_hours"]),
        ("", ""),
        ("Start equity", start_equity),
        ("Final equity", round(final_equity, 2)),
        ("Total return % (window)", round(total_return_pct, 2)),
        ("Approx annualized return %", round(annualized_return_pct, 2) if annualized_return_pct is not None else "n/a"),
        ("Max drawdown % (simulated)", round(max_dd, 2)),
        ("Drawdown-ceiling pause events", pause_events),
    ]
    df_summary = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    df_side = pd.DataFrame([side_stats("LONG"), side_stats("SHORT")])

    df_engine = None
    if len(engines_present) > 1:
        engine_rows = []
        for eng in engines_present:
            rows = [r for r in filled if r["engine"] == eng]
            stats = perf_stats(rows)
            stats["engine"] = eng
            engine_rows.append(stats)
        df_engine = pd.DataFrame(
            engine_rows,
            columns=["engine", "trades", "wins", "losses", "win_rate_pct", "avg_R", "avg_win_R",
                     "avg_loss_R", "profit_factor", "avg_leverage", "avg_hold_hours", "avg_fill_pct"],
        )

    df_equity = pd.DataFrame(equity_curve)
    if not df_equity.empty:
        df_equity["time"] = df_equity["ts"].apply(_fmt_ts)
        df_equity = df_equity[["time", "equity", "drawdown_pct"]]

    monthly: dict[str, list[dict]] = {}
    for r in filled:
        mk = datetime.fromtimestamp(r["open_ts"] / 1000, tz=timezone.utc).strftime("%Y-%m")
        monthly.setdefault(mk, []).append(r)
    monthly_rows = []
    for mk in sorted(monthly.keys()):
        rows = monthly[mk]
        w = [r for r in rows if r["r_realized"] > 0]
        monthly_rows.append(
            {
                "month": mk,
                "trades": len(rows),
                "win_rate_pct": round(100.0 * len(w) / len(rows), 2) if rows else 0.0,
                "avg_R": round(sum(r["r_realized"] for r in rows) / len(rows), 3) if rows else 0.0,
                "sum_R": round(sum(r["r_realized"] for r in rows), 3),
            }
        )
    df_monthly = pd.DataFrame(monthly_rows)

    notes = [
        "This backtest reuses the live bot's exact strategy code (strategy.py / gem_strategy.py /",
        "kryptic_strategy.py, risk_guard.py) and config-application path (bot.configure_strategy /",
        "bot.configure_gem_strategy / bot.configure_kryptic_strategy), so results reflect what",
        "bot.py would do live.",
        "",
        "When more than one engine ran, they shared ONE RiskGuard instance (one equity curve, one",
        "drawdown ceiling, one MAX_CONCURRENT_TRADES/SAME_SIDE budget) exactly like the live bot --",
        "see the 'By Engine' block in Summary for each engine's own win rate / avg R / profit factor.",
        "",
        "Cooldown is per (symbol, direction) for every engine: a same-direction repeat on a symbol",
        "is blocked until its cooldown elapses, but a reversal (opposite direction) is never blocked",
        "by it -- matches bot.py's live scan_once/gem_scan_once/kryptic_scan_once.",
        "",
        "Simplifications vs. a full live run:",
        "- Funding-rate veto is not modeled for any engine (no historical funding data fetched);",
        "  a minor soft filter, not the core edge.",
        "- CVD/OI absorption was never modeled, live or here (stub only) -- see README.",
        "- Each engine's symbol universe is TODAY's qualifying set by 24h volume, held fixed across",
        "  the whole lookback window (mild survivorship bias vs. the exact universe that would have",
        "  been live-scanned back then).",
        "- The entry ladder IS simulated: only levels price actually touches within CANCEL_VELAS bars",
        "  count as filled, and a signal that never fills contributes zero equity impact ('Signals",
        "  never filled' above). But each touched level still fills at exactly its quoted price --",
        "  no slippage within a fill, no partial fill of a single level, no market-impact modeling.",
        "- Within one bar, a stop-loss touch is checked before take-profits (conservative assumption",
        "  when intrabar order is unknown from OHLC alone) -- same assumption risk_guard.py uses live.",
        "- This always fetches from Binance regardless of the live bot's DATA_SOURCE (e.g. blofin) --",
        "  approximates each strategy's edge, not an exact replay of a different exchange's price action.",
        "- KRYPTIC's own entry gate (RegimeFilter + DirectionEngine) is pass/fail, not a continuous",
        "  score -- its 'score' column is a fixed value that only feeds leverage sizing, never a filter.",
        "- Compounding caution: with hundreds of trades, small unmodeled optimism in the per-trade edge",
        "  (see above) compounds exponentially. Treat the win rate / avg R / drawdown as the signal to",
        "  trust; treat the compounded total-return figure as illustrative, not a real-world forecast.",
        "",
        f"Generated {_fmt_ts(int(time.time() * 1000))} UTC.",
    ]
    df_notes = pd.DataFrame({"Notes": notes})

    with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
        df_summary.to_excel(xw, sheet_name="Summary", index=False)
        next_row = len(df_summary) + 2
        df_side.to_excel(xw, sheet_name="Summary", index=False, startrow=next_row)
        next_row += len(df_side) + 2
        if df_engine is not None:
            df_engine.to_excel(xw, sheet_name="Summary", index=False, startrow=next_row)
        df_trades.to_excel(xw, sheet_name="Trades", index=False)
        df_equity.to_excel(xw, sheet_name="Equity Curve", index=False)
        df_monthly.to_excel(xw, sheet_name="Monthly", index=False)
        df_notes.to_excel(xw, sheet_name="Notes", index=False)
        for ws in xw.book.worksheets:
            for col_cells in ws.columns:
                length = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(60, max(10, length + 2))

    log.info("Wrote %s (%d trades)", args.out, n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZENITH + GEM walk-forward backtest -> Excel report")
    p.add_argument(
        "--engine", choices=["zenith", "gem", "kryptic", "both", "all"], default="zenith",
        help="Which engine(s) to backtest -- 'both' is ZENITH+GEM (legacy), 'all' is ZENITH+GEM+KRYPTIC",
    )
    p.add_argument("--months", type=float, default=6.0)
    p.add_argument("--top-n", type=int, default=30, help="ZENITH: symbols to scan by 24h volume; <= 0 means all qualifying symbols")
    p.add_argument("--symbols", type=str, default="", help="Comma list to override auto universe selection for BOTH engines")
    p.add_argument("--out", type=str, default="backtest_report.xlsx")
    p.add_argument("--warmup-days", type=float, default=12.0)
    p.add_argument("--fetch-concurrency", type=int, default=6)
    p.add_argument("--start-equity", type=float, default=100.0)
    p.add_argument("--every", type=int, default=1, help="Evaluate every Nth 15m bar (coarser/faster preview)")
    p.add_argument("--smoke", action="store_true", help="Quick ~3-day, 5-symbol sanity run")
    p.add_argument(
        "--max-posts-per-scan", type=int, default=None,
        help="Override ZENITH's MAX_POSTS_PER_SCAN (default from .env/bot.py, normally 2) -- caps how many "
             "already-qualifying signals get taken per scan; raise to capture more of them",
    )
    p.add_argument(
        "--max-concurrent-trades", type=int, default=None,
        help="Override MAX_CONCURRENT_TRADES (default from .env/bot.py, normally 4) -- shared across engines",
    )
    p.add_argument(
        "--max-concurrent-same-side", type=int, default=None,
        help="Override MAX_CONCURRENT_SAME_SIDE (default from .env/bot.py, normally 3) -- shared across engines",
    )
    p.add_argument(
        "--cooldown-minutes", type=int, default=None,
        help="Override ZENITH's COOLDOWN_MINUTES (default from .env/bot.py, normally 720 = 12h) -- how long "
             "before the same symbol+side can signal again",
    )
    p.add_argument(
        "--min-quote-vol", type=float, default=None,
        help="Override ZENITH's MIN_QUOTE_VOLUME_USD (default from .env/bot.py, normally 20000000) -- lower "
             "this to widen the universe to less-liquid symbols; combine with --top-n 0",
    )
    p.add_argument(
        "--dd-pause-cooldown-hours", type=float, default=None,
        help="Override DD_PAUSE_COOLDOWN_HOURS (default from .env/bot.py, normally 48) -- if the "
             "drawdown ceiling trips and DD hasn't organically recovered after this many hours "
             "(e.g. because no trades are open to recover it), force a resume and reset the "
             "high-water mark to current equity instead of pausing forever",
    )
    p.add_argument("--gem-max-posts-per-scan", type=int, default=None, help="Override GEM_MAX_SIGNALS_PER_SCAN")
    p.add_argument("--gem-cooldown-minutes", type=int, default=None, help="Override GEM_COOLDOWN_MINUTES")
    p.add_argument("--gem-min-quote-vol", type=float, default=None, help="Override GEM_MIN_QUOTE_VOLUME_USD")
    p.add_argument("--gem-top-n", type=int, default=None, help="Override GEM_TOP_N; <= 0 means all qualifying symbols")
    p.add_argument("--kryptic-max-posts-per-scan", type=int, default=None, help="Override KRYPTIC_MAX_SIGNALS_PER_SCAN")
    p.add_argument("--kryptic-cooldown-minutes", type=int, default=None, help="Override KRYPTIC_COOLDOWN_MINUTES")
    p.add_argument("--kryptic-min-quote-vol", type=float, default=None, help="Override KRYPTIC_MIN_QUOTE_VOLUME_USD")
    p.add_argument("--kryptic-top-n", type=int, default=None, help="Override KRYPTIC_TOP_N; <= 0 means all qualifying symbols")
    p.add_argument(
        "--from-raw", type=str, default=None,
        help="Regenerate the Excel report from a previously-saved <out>_raw.json checkpoint instead "
             "of re-running the (potentially very long) network fetch + walk-forward simulation. "
             "Use this after fixing a failure that happened during the Excel write step.",
    )
    return p.parse_args()


def regenerate_from_raw(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.from_raw).read_text())
    args.start_equity = float(data.get("start_equity", args.start_equity))
    zenith_symbols = data.get("zenith_symbols")
    if zenith_symbols is None:
        zenith_symbols = data.get("symbols") or []  # pre-GEM checkpoint format
    gem_symbols = data.get("gem_symbols") or []
    kryptic_symbols = data.get("kryptic_symbols") or []
    write_report(
        args,
        data["cfg"],
        data["profile"],
        data.get("engine_mode", "zenith"),
        zenith_symbols,
        gem_symbols,
        kryptic_symbols,
        data["trade_records"],
        data["equity_curve"],
        data["pause_events"],
        data["start_ms"],
        data["end_ms"],
    )


def _check_report_deps() -> None:
    """Fail fast on a missing pandas/openpyxl instead of discovering it only
    after a run that can take minutes to hours. This has bitten real users
    running backtest.py outside the venv they installed dependencies into
    (`python -m pip install ...` before running avoids the mismatch)."""
    missing = []
    for mod in ("pandas", "openpyxl"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            f"Missing required package(s) for the Excel report: {', '.join(missing)}.\n"
            f"Install them into the SAME Python that's about to run this script:\n"
            f"  python -m pip install {' '.join(missing)}\n"
            f"(python -m pip, not bare pip, guarantees it installs where `python` will look -- "
            f"a common cause of 'it's installed but not found' is a venv/PATH mismatch.)"
        )


def main() -> None:
    args = parse_args()
    _check_report_deps()
    if args.from_raw:
        regenerate_from_raw(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
