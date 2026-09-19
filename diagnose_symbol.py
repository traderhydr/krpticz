#!/usr/bin/env python3
"""Read-only diagnostic: walks one live symbol through ZENITH SMC-lite's
early gate sequence and prints PASS/FAIL with the actual numbers, stopping
at the first failure.

evaluate_smc_lite() in strategy.py returns a bare None on any internal gate
failing, with no detail on which one -- fine for the live bot, useless for
figuring out *why* nothing is signaling. This script doesn't touch
strategy.py or change any bot behavior: it calls the same helper functions
strategy.py itself uses (_ohlcv, _atr, smc_lite.htf_bias, ...) in the same
order evaluate_smc_lite checks them, so the numbers can't drift from what
the live bot actually computes -- only the up-front early-gate subset
covered here (through EXIGIR_BOS) is duplicated, not the full ~30-gate
function.

Usage:
    python diagnose_symbol.py BTCUSDT
    python diagnose_symbol.py BTCUSDT ETHUSDT SOLUSDT
"""
from __future__ import annotations

import asyncio
import sys

import httpx

import bot as botmod
import smc_lite
from strategy import FIB_CFG, _atr, _efficiency_ratio, _ema, _ohlcv, _vol_sma, closed_candles, var24h


async def diagnose(client: httpx.AsyncClient, cfg: dict, symbol: str) -> None:
    print(f"\n=== {symbol} ===")
    candles_raw, src = await botmod.fetch_klines(client, symbol, cfg["timeframe"], limit=240)
    candles = closed_candles(candles_raw)
    print(f"data source: {src}, closed candles: {len(candles)}")
    if len(candles) < 50:
        print("FAIL [0] fewer than 50 closed candles -- can't evaluate at all")
        return

    C = FIB_CFG
    n = len(candles)
    need = max(int(C["EMA_SLOW"]) + 8, int(C["IMPULSO_VELAS"]) + 16, 97)
    ok = n >= need
    print(f"[1] history: need>={need}, have={n} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        return

    opens, highs, lows, closes, vols = _ohlcv(candles)
    last = closes[-1]
    atr = _atr(highs, lows, closes)
    if atr <= 0 or last <= 0:
        print(f"FAIL [1b] atr={atr} last={last} (degenerate)")
        return

    atr_pct = atr / last * 100.0
    lo, hi = float(C["ATR_PCT_MIN"]), float(C["ATR_PCT_MAX"])
    ok = lo <= atr_pct <= hi
    print(f"[2] ATR%: {atr_pct:.3f} in [{lo}, {hi}] -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        return

    vol_ratio = vols[-1] / _vol_sma(vols)
    vmin = float(C["VOL_MIN"])
    ok = vol_ratio >= vmin
    print(f"[3] vol_ratio: {vol_ratio:.3f} >= {vmin} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        return

    er = _efficiency_ratio(closes, 20)
    er_min = float(C["ER_MIN"])
    ok = er >= er_min
    print(f"[4] efficiency ratio: {er:.3f} >= {er_min} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        return

    ema_f = _ema(closes, int(C["EMA_FAST"]))
    ema_s = _ema(closes, int(C["EMA_SLOW"]))
    stretch = abs(last - ema_f) / atr
    smax = float(C["STRETCH_ATR_MAX"])
    ok = stretch <= smax
    print(f"[5] stretch from EMA{int(C['EMA_FAST'])}: {stretch:.3f} ATR <= {smax} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        return

    htf = smc_lite.aggregate_htf_from_ltf(candles, 4)
    min_htf = int(C.get("SMC_MIN_HTF_BARS", 30))
    ok = len(htf) >= min_htf
    print(f"[6] aggregated HTF bars: {len(htf)} >= {min_htf} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        return

    swing_htf = int(C.get("SMC_SWING_HTF", 2))
    bias = smc_lite.htf_bias(htf, swing_htf)
    estruct = smc_lite.estructura_fuerte(htf, swing_htf)
    exigir_bos = bool(C.get("EXIGIR_BOS", True))
    ok = not (exigir_bos and not bias)
    print(f"[7] HTF bias={bias!r} BOS={estruct.bos!r} choch={estruct.choch} "
          f"(EXIGIR_BOS={exigir_bos}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("    >>> No defined HTF swing-structure trend right now (ranging/choppy).")
        print("    >>> EXIGIR_BOS=1 hard-requires one. If this shows up across most symbols")
        print("    >>> at once, that's a genuinely quiet/rangy market for this gate, not a bug.")
        return

    v24 = var24h(closes)
    floor = float(C["GAINER_24H_MIN"])
    print(f"[8] 24h change: {v24:.2f}% (gainer floor +-{floor}%, direction-dependent) -- HTF bias={bias}")
    print("Cleared every gate checked here (history/ATR%/volume/efficiency/stretch/HTF bias).")
    print("Remaining gates in evaluate_smc_lite (impulse window, EMA alignment, BTC regime,")
    print("gainer floor, funding veto, POI/FVG proximity, LTF CHoCH, R-band, wick rejection)")
    print("aren't duplicated here -- reaching this far means the setup is a live candidate.")


async def main(symbols: list[str]) -> None:
    cfg = botmod._cfg()
    profile = botmod.configure_strategy(cfg)
    print(
        f"profile={profile} timeframe={cfg['timeframe']} MIN_SCORE={cfg['min_score']} "
        f"EXIGIR_BOS={FIB_CFG.get('EXIGIR_BOS')} EXIGIR_POI_OR_CHOCH={FIB_CFG.get('EXIGIR_POI_OR_CHOCH')} "
        f"GAINER_24H_MIN={FIB_CFG.get('GAINER_24H_MIN')} BTC_REGIME={FIB_CFG.get('BTC_REGIME')}"
    )
    async with httpx.AsyncClient(headers={"User-Agent": "zenith-bot/1.0"}) as client:
        for symbol in symbols:
            try:
                await diagnose(client, cfg, symbol)
            except Exception as e:
                print(f"ERROR diagnosing {symbol}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_symbol.py SYMBOL [SYMBOL...]   (e.g. BTCUSDT ETHUSDT)")
        raise SystemExit(1)
    asyncio.run(main([s.upper() for s in sys.argv[1:]]))
