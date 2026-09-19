"""GEM engine — Fibonacci-retracement entries, ATR-based stop, R-based
filters, SMC (structure/FVG/order-block/sweep) confluence measurements.

This is a line-for-line port of the live-signal path of the uploaded
motor-gem.js ("GEM ENGINE v4"): the `evaluar()` function and everything it
calls. Ported to Python so GEM runs as a second strategy inside this same
bot process, sharing ZENITH's exchange layer (exchanges.py), Signal type,
Telegram feed and RiskGuard instead of a separate Node service.

Not ported (also unused by the original bot.js live bot, so no live
behavior is lost): `evaluarSMC` (an alternate full-SMC entry mode gated by
a MODO_SMC flag that motor-gem.js itself never wires into `evaluar()`),
and `simular`/`revisarTesis` (its offline backtest helpers).

Candle shape is kept as the original {t,o,h,l,c,v} (oldest first) rather
than translated to English field names, so this file stays a direct,
checkable translation of motor-gem.js. Use `from_repo_candles()` to adapt
this repo's shared {ts,open,high,low,close,volume} kline shape into it.
"""
from __future__ import annotations

from strategy import Signal

# ---------------------------------------------------------------------------
# PARAMETERS — ported from motor-gem.js CFG. Only keys actually read by the
# functions below are kept (motor-gem.js carried several dead/backtest-only
# keys — SILENCIO_*, TOPE_*, MODO_SMC, CORTE_SIN_RECHAZO, etc. — that
# bot.js's live loop never read either).
# ---------------------------------------------------------------------------
GEM_CFG: dict = {
    "ATR_PERIOD": 14,
    "IMPULSE_BARS": 30,       # leg window; shorter = closer stop
    "FIB_E1": 0.79,           # deep retracement level for E1

    "BUFFER_ATR": 1.0,        # SL = leg extreme -+ BUFFER_ATR * ATR
    "STOP_APRIETA": 1.0,      # <1 moves the *working* stop to X*R from Eavg
    "MARKET_2ENTRIES": False,  # with MARKET_WEIGHT>0: market + only E1 (2 legs)
    "E2_FRACTION": 0.35,
    "E3_FRACTION": 0.75,
    "E2_SEP_MIN_ATR": 0.5,
    "E2_SEP_MAX_ATR": 1.2,
    "EQUAL_ENTRIES": True,
    "WEIGHT_E1": 0.40,
    "WEIGHT_E2": 0.60,

    "REACH_MAX_ATR": 1.2,     # E1 never further than this from current price
    "ENTRY_MAX_ATR": 99.0,
    "MARKET_WEIGHT": 0.0,     # 0 = all on retracement; 0.30 = 30% enters now

    "FEE_PCT": 0.13,          # round-trip fee at 10x, % of notional
    "MAX_FEE_R": 0.15,        # discard if fees eat more than this fraction of R

    "VOL_JUMP": 2.0,
    "REQUIRE_JUMP": False,
    "DELTA_MIN": 0.05,
    "REQUIRE_DELTA": False,
    "MIN_SCORE": 0,           # 0 = accept everything; 70 = sniper mode

    "STOCH_N": 14,
    "STOCH_D": 3,
    "REQUIRE_STOCH": False,
    "STOCH_MAX_LONG": 45,
    "STOCH_MIN_SHORT": 55,

    "SMC_SWING_BARS": 3,      # bars each side to call a fractal swing
    "SMC_FVG_BARS": 40,       # how far back to look for unmitigated gaps
    "SMC_OB_BARS": 20,        # how far back to look for the order block
    "REQUIRE_BOS": False,
    "REQUIRE_HTF": False,
    "REQUIRE_FVG": False,
    "REQUIRE_SWEEP": False,
    "AVOID_CHOCH": False,

    "R_MIN_PCT": 1.5,
    "R_MAX_PCT": 4.5,
    "R_MIN_ATR": 1.2,
    "R_MAX_ATR": 2.4,
    "VOL_MIN": 0.5,
    "FUNDING_VETO": 0.10,     # % per 8h

    "TP": [0.80, 1.40, 2.40],   # in multiples of R, off Eavg
    "TP1_FIXED": False,
    "TP1_FIXED_PCT": 1.00,
    "CLOSES": [0.70, 0.25, 0.05],   # size closed at TP1/TP2/TP3

    "CANCEL_BARS": 6,   # bars unfilled before the ladder is cancelled
    "CLOSE_BARS": 8,    # bars without reaching UMBRAL before a time-exit
    "BE_ON_TP1": True,  # move stop to breakeven once TP1 is hit
}


def _reject(code: str, reason: str) -> dict:
    return {"ok": False, "code": code, "reason": reason}


def from_repo_candles(candles: list[dict]) -> list[dict]:
    """Adapt this repo's shared kline shape into motor-gem.js's {t,o,h,l,c,v}."""
    return [
        {"t": c["ts"], "o": c["open"], "h": c["high"], "l": c["low"], "c": c["close"], "v": c["volume"]}
        for c in candles
    ]


# ===================== INDICATORS =====================


def atr_series(candles: list[dict], period: int | None = None) -> list[float]:
    period = GEM_CFG["ATR_PERIOD"] if period is None else period
    tr = []
    for i in range(1, len(candles)):
        h, l = candles[i]["h"], candles[i]["l"]
        prev_c = candles[i - 1]["c"]
        tr.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    if len(tr) < period:
        return []
    out = []
    a = sum(tr[:period]) / period
    out.append(a)
    for i in range(period, len(tr)):
        a = (a * (period - 1) + tr[i]) / period
        out.append(a)
    return out


def swings(candles: list[dict], n: int | None = None) -> tuple[list[dict], list[dict]]:
    n = GEM_CFG["SMC_SWING_BARS"] if n is None else n
    highs, lows = [], []
    length = len(candles)
    for i in range(n, length - n):
        is_high, is_low = True, True
        j = i - n
        while j <= i + n and (is_high or is_low):
            if j != i:
                if candles[j]["h"] >= candles[i]["h"]:
                    is_high = False
                if candles[j]["l"] <= candles[i]["l"]:
                    is_low = False
            j += 1
        if is_high:
            highs.append({"i": i, "p": candles[i]["h"]})
        if is_low:
            lows.append({"i": i, "p": candles[i]["l"]})
    return highs, lows


def structure(candles: list[dict]) -> dict:
    highs, lows = swings(candles)
    empty = {"trend": None, "bos": None, "choch": False, "last_high": None, "last_low": None}
    if len(highs) < 2 or len(lows) < 2:
        return empty
    ua, pa = highs[-1], highs[-2]
    ub, pb = lows[-1], lows[-2]
    trend = None
    if ua["p"] > pa["p"] and ub["p"] > pb["p"]:
        trend = "UP"
    elif ua["p"] < pa["p"] and ub["p"] < pb["p"]:
        trend = "DOWN"
    close = candles[-1]["c"]
    bos, choch = None, False
    if close > ua["p"]:
        bos = "UP"
        if trend == "DOWN":
            choch = True
    elif close < ub["p"]:
        bos = "DOWN"
        if trend == "UP":
            choch = True
    return {"trend": trend, "bos": bos, "choch": choch, "last_high": ua["p"], "last_low": ub["p"]}


def fvg_unmitigated(candles: list[dict], direction: str) -> dict:
    last = candles[-1]["c"]
    start = max(0, len(candles) - GEM_CFG["SMC_FVG_BARS"])
    for i in range(len(candles) - 3, start - 1, -1):
        a, c = candles[i], candles[i + 2]
        if direction == "LONG" and a["h"] < c["l"]:
            if not any(v["l"] <= a["h"] for v in candles[i + 3:]):
                return {"has": True, "low": a["h"], "high": c["l"], "inside": a["h"] <= last <= c["l"]}
        if direction == "SHORT" and a["l"] > c["h"]:
            if not any(v["h"] >= a["l"] for v in candles[i + 3:]):
                return {"has": True, "low": c["h"], "high": a["l"], "inside": c["h"] <= last <= a["l"]}
    return {"has": False, "inside": False}


def liquidity_sweep(candles: list[dict], direction: str) -> bool:
    highs, lows = swings(candles)
    v = candles[-1]
    if direction == "SHORT" and highs:
        ref = highs[-1]["p"]
        return v["h"] > ref and v["c"] < ref
    if direction == "LONG" and lows:
        ref = lows[-1]["p"]
        return v["l"] < ref and v["c"] > ref
    return False


def order_block(candles: list[dict], direction: str) -> dict | None:
    start = max(0, len(candles) - GEM_CFG["SMC_OB_BARS"])
    for i in range(len(candles) - 2, start - 1, -1):
        c = candles[i]
        matches = (c["c"] < c["o"]) if direction == "LONG" else (c["c"] > c["o"])
        if matches:
            return {"high": c["h"], "low": c["l"], "bars": len(candles) - 1 - i}
    return None


def to_higher_tf(candles: list[dict], factor: int = 4) -> list[dict]:
    out = []
    i = len(candles) - factor
    while i >= 0:
        g = candles[i:i + factor]
        out.insert(0, {
            "t": g[0]["t"], "o": g[0]["o"], "c": g[-1]["c"],
            "h": max(x["h"] for x in g),
            "l": min(x["l"] for x in g),
            "v": sum(x["v"] for x in g),
        })
        i -= factor
    return out


def analyze_smc(candles: list[dict], direction: str) -> dict:
    e = structure(candles)
    h1 = to_higher_tf(candles)
    e_h1 = structure(h1) if len(h1) >= 12 else {"trend": None}
    target = "UP" if direction == "LONG" else "DOWN"
    f = fvg_unmitigated(candles, direction)
    return {
        "trend": e["trend"], "bos": e["bos"], "choch": e["choch"],
        "trend_h1": e_h1.get("trend"),
        "bos_in_favor": e["bos"] == target,
        "h1_in_favor": e_h1.get("trend") == target,
        "fvg": f["has"], "fvg_inside": bool(f.get("inside")),
        "sweep": liquidity_sweep(candles, direction),
        "ob": order_block(candles, direction),
    }


def impulse_leg(candles: list[dict], n: int | None = None) -> dict:
    n = GEM_CFG["IMPULSE_BARS"] if n is None else n
    s = candles[-n:]
    hi = lo = 0
    for i, c in enumerate(s):
        if c["h"] > s[hi]["h"]:
            hi = i
        if c["l"] < s[lo]["l"]:
            lo = i
    H, L = s[hi]["h"], s[lo]["l"]
    rng = H - L
    bullish = hi > lo
    f62 = (H - rng * GEM_CFG["FIB_E1"]) if bullish else (L + rng * GEM_CFG["FIB_E1"])
    return {"bullish": bullish, "H": H, "L": L, "range": rng, "f62": f62}


def bar_delta(c: dict) -> float:
    rng = c["h"] - c["l"]
    if not (rng > 0):
        return 0.0
    return c["v"] * (2 * (c["c"] - c["l"]) / rng - 1)


def leg_flow(candles: list[dict], n: int | None = None) -> dict:
    n = GEM_CFG["IMPULSE_BARS"] if n is None else n
    s = candles[-n:]
    hi = lo = 0
    for i, c in enumerate(s):
        if c["h"] > s[hi]["h"]:
            hi = i
        if c["l"] < s[lo]["l"]:
            lo = i
    leg = s[min(hi, lo):max(hi, lo) + 1]
    if len(leg) < 2:
        return {"flow": 0.0, "divergence": False}
    delta = sum(bar_delta(c) for c in leg)
    vol = sum(c["v"] for c in leg)
    if not (vol > 0):
        return {"flow": 0.0, "divergence": False}
    flow = (delta / vol) * (1 if hi > lo else -1)
    return {"flow": flow, "divergence": flow < GEM_CFG["DELTA_MIN"]}


_WEIGHTS = {"centrality": 25, "reach": 20, "flow": 20, "volume": 15, "fees": 10, "funding": 10}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _centered(v: float, lo: float, hi: float) -> float:
    if not (hi > lo):
        return 0.5
    return _clamp(1 - abs((v - (lo + hi) / 2) / ((hi - lo) / 2)))


def score_setup(s: dict, funding_pct8h: float | None) -> dict:
    c = {
        "centrality": (_centered(s["Ratr"], GEM_CFG["R_MIN_ATR"], GEM_CFG["R_MAX_ATR"]) +
                       _centered(s["Rpct"], GEM_CFG["R_MIN_PCT"], GEM_CFG["R_MAX_PCT"])) / 2,
        "reach": _clamp(1 - s["alcanceATR"] / GEM_CFG["REACH_MAX_ATR"]),
        "flow": _clamp((s["flujo"] + 0.15) / 0.60),
        "volume": _clamp((s["volRel"] - GEM_CFG["VOL_MIN"]) / 2.0),
        "fees": _clamp(1 - s["dragComision"] / GEM_CFG["MAX_FEE_R"]),
        "funding": 0.5 if funding_pct8h is None else _clamp(
            0.5 - (funding_pct8h if s["dir"] == "LONG" else -funding_pct8h) / (2 * GEM_CFG["FUNDING_VETO"])
        ),
    }
    total = sum(c[k] * _WEIGHTS[k] for k in _WEIGHTS)
    return {"nota": round(total), "componentes": c}


def net_zero_factor() -> float:
    closes = GEM_CFG["CLOSES"] or [0.4, 0.35, 0.25]
    remainder = 1 - closes[0]
    if remainder <= 0.001:
        return 0.0
    x = (closes[0] * GEM_CFG["TP"][0]) / remainder
    return 0.0 if x >= GEM_CFG["TP"][0] else x


def stochastic(candles: list[dict], n: int | None = None, d: int | None = None) -> dict | None:
    n = GEM_CFG["STOCH_N"] if n is None else n
    d = GEM_CFG["STOCH_D"] if d is None else d
    if len(candles) < n + d:
        return None
    ks = []
    for i in range(len(candles) - d, len(candles)):
        window = candles[i - n + 1:i + 1]
        hi = max(x["h"] for x in window)
        lo = min(x["l"] for x in window)
        ks.append(((candles[i]["c"] - lo) / (hi - lo)) * 100 if hi > lo else 50)
    K = ks[-1]
    D = sum(ks) / len(ks)
    return {"K": K, "D": D, "rising": ks[-1] > ks[0]}


def relative_volume(candles: list[dict]) -> float:
    if len(candles) < 21:
        return 1.0
    window = candles[-21:-1]
    avg = sum(c["v"] for c in window) / 20
    return candles[-1]["v"] / avg if avg > 0 else 1.0


# ===================== THE SYSTEM =====================


def evaluate(
    candles: list[dict],
    funding_pct8h: float | None = None,
    *,
    min_gainer_pct: float | None = None,
    change_24h: float | None = None,
) -> dict:
    if len(candles) < max(GEM_CFG["IMPULSE_BARS"], GEM_CFG["ATR_PERIOD"] + 2):
        return _reject("HISTORY", "insufficient history")

    series = atr_series(candles)
    if not series:
        return _reject("ATR", "ATR not computable")
    atr = series[-1]
    price = candles[-1]["c"]
    if not (atr > 0) or not (price > 0):
        return _reject("DATA", "invalid data")

    imp = impulse_leg(candles)
    if not (imp["range"] > 0):
        return _reject("LEG", "leg has no range")
    direction = "LONG" if imp["bullish"] else "SHORT"

    funding_applied = False
    if funding_pct8h is not None:
        funding_applied = True
        if direction == "LONG" and funding_pct8h > GEM_CFG["FUNDING_VETO"]:
            return _reject("FUNDING", f"funding veto {funding_pct8h:.3f}%/8h: crowd already long")
        if direction == "SHORT" and funding_pct8h < -GEM_CFG["FUNDING_VETO"]:
            return _reject("FUNDING", f"funding veto {funding_pct8h:.3f}%/8h: crowd already short")

    E1 = imp["f62"]
    cap = GEM_CFG["ENTRY_MAX_ATR"] * atr
    if direction == "LONG" and price - E1 > cap:
        E1 = price - cap
    if direction == "SHORT" and E1 - price > cap:
        E1 = price + cap
    reach_atr = abs(E1 - price) / atr
    if reach_atr > GEM_CFG["REACH_MAX_ATR"]:
        return _reject(
            "REACH",
            f"E1 is {reach_atr:.2f} ATR from price (max {GEM_CFG['REACH_MAX_ATR']}): that's noise, not an impulse",
        )

    SL = (imp["L"] - GEM_CFG["BUFFER_ATR"] * atr) if direction == "LONG" else (imp["H"] + GEM_CFG["BUFFER_ATR"] * atr)
    dist_sl = abs(E1 - SL)
    if not (dist_sl > 0):
        return _reject("SL", "degenerate stop")

    sep = GEM_CFG["E2_FRACTION"] * dist_sl
    sep = min(max(sep, GEM_CFG["E2_SEP_MIN_ATR"] * atr), GEM_CFG["E2_SEP_MAX_ATR"] * atr)
    E2 = (E1 - sep) if direction == "LONG" else (E1 + sep)

    if direction == "LONG" and E2 <= SL:
        return _reject("E2_SL", "E2 would fall past the stop")
    if direction == "SHORT" and E2 >= SL:
        return _reject("E2_SL", "E2 would fall past the stop")

    if GEM_CFG["MARKET_WEIGHT"] > 0 and GEM_CFG["MARKET_2ENTRIES"]:
        entries = [price, E1]
        weights = [GEM_CFG["MARKET_WEIGHT"], 1 - GEM_CFG["MARKET_WEIGHT"]]
    elif GEM_CFG["MARKET_WEIGHT"] > 0:
        entries = [price, E1, E2]
        if GEM_CFG["EQUAL_ENTRIES"]:
            weights = [1 / 3, 1 / 3, 1 / 3]
        else:
            weights = [
                GEM_CFG["MARKET_WEIGHT"],
                (1 - GEM_CFG["MARKET_WEIGHT"]) * GEM_CFG["WEIGHT_E1"],
                (1 - GEM_CFG["MARKET_WEIGHT"]) * GEM_CFG["WEIGHT_E2"],
            ]
    elif GEM_CFG["E3_FRACTION"] > 0:
        sep3 = GEM_CFG["E3_FRACTION"] * dist_sl
        sep3 = min(max(sep3, sep * 1.5), 0.92 * dist_sl)
        E3 = (E1 - sep3) if direction == "LONG" else (E1 + sep3)
        if direction == "LONG" and E3 <= SL:
            return _reject("E3_SL", "E3 would fall past the stop")
        if direction == "SHORT" and E3 >= SL:
            return _reject("E3_SL", "E3 would fall past the stop")
        entries = [E1, E2, E3]
        weights = [1 / 3, 1 / 3, 1 / 3]
    else:
        entries = [E1, E2]
        weights = [GEM_CFG["WEIGHT_E1"], GEM_CFG["WEIGHT_E2"]]

    Eavg = sum(e * w for e, w in zip(entries, weights))
    riesgo = abs(Eavg - SL)
    Rpct = riesgo / Eavg * 100
    Ratr = riesgo / atr

    if Rpct < GEM_CFG["R_MIN_PCT"]:
        return _reject("R_TIGHT", f"R = {Rpct:.2f}% < {GEM_CFG['R_MIN_PCT']}%: stop sits inside the noise")
    if Rpct > GEM_CFG["R_MAX_PCT"]:
        return _reject("R_WIDE", f"R = {Rpct:.2f}% > {GEM_CFG['R_MAX_PCT']}%: not enough cushion at 10x")
    if Ratr < GEM_CFG["R_MIN_ATR"]:
        return _reject("R_ATR_LOW", f"R = {Ratr:.2f} ATR < {GEM_CFG['R_MIN_ATR']}: fits inside a single bar")
    if Ratr > GEM_CFG["R_MAX_ATR"]:
        return _reject("R_ATR_HIGH", f"R = {Ratr:.2f} ATR > {GEM_CFG['R_MAX_ATR']}: paying for volatility you don't need")

    fee_drag = GEM_CFG["FEE_PCT"] / Rpct
    if fee_drag > GEM_CFG["MAX_FEE_R"]:
        return _reject("FEES", f"fees eat {fee_drag:.2f}R (R = {Rpct:.2f}%): no margin left")

    st = stochastic(candles)
    if GEM_CFG["REQUIRE_STOCH"] and st:
        if direction == "LONG" and st["K"] > GEM_CFG["STOCH_MAX_LONG"]:
            return _reject("STOCH", f"stoch %K at {st['K']:.0f}: not enough pullback (max {GEM_CFG['STOCH_MAX_LONG']} on LONG)")
        if direction == "SHORT" and st["K"] < GEM_CFG["STOCH_MIN_SHORT"]:
            return _reject("STOCH", f"stoch %K at {st['K']:.0f}: not enough bounce (min {GEM_CFG['STOCH_MIN_SHORT']} on SHORT)")

    smc = analyze_smc(candles, direction)
    if GEM_CFG["REQUIRE_BOS"] and not smc["bos_in_favor"]:
        return _reject("SMC_BOS", f"structure hasn't broken in favor of {direction} (BOS: {smc['bos'] or 'none'})")
    if GEM_CFG["REQUIRE_HTF"] and not smc["h1_in_favor"]:
        return _reject("SMC_HTF", f"1h trend is {smc['trend_h1'] or 'undefined'}: doesn't support {direction}")
    if GEM_CFG["REQUIRE_FVG"] and not smc["fvg"]:
        return _reject("SMC_FVG", "no unmitigated fair-value gap in this direction")
    if GEM_CFG["REQUIRE_SWEEP"] and not smc["sweep"]:
        return _reject("SMC_SWEEP", "last bar hasn't swept liquidity of any prior swing")
    if GEM_CFG["AVOID_CHOCH"] and smc["choch"]:
        return _reject("SMC_CHOCH", "recent change of character: structure just turned against")

    fl = leg_flow(candles)
    if GEM_CFG["REQUIRE_DELTA"] and fl["divergence"]:
        moved = "rose" if direction == "LONG" else "fell"
        return _reject("DIVERGENCE", f"price {moved} but net flow was {fl['flow'] * 100:.1f}%: move without backing")

    vr = relative_volume(candles)
    if GEM_CFG["REQUIRE_JUMP"] and vr < GEM_CFG["VOL_JUMP"]:
        return _reject("NO_JUMP", f"volume {vr:.1f}x average: no notable jump (needs {GEM_CFG['VOL_JUMP']}x)")
    if vr < GEM_CFG["VOL_MIN"]:
        return _reject("VOLUME", f"volume at {vr * 100:.0f}% of average: unreliable levels")

    if min_gainer_pct is not None and (change_24h is None or abs(change_24h) < min_gainer_pct):
        shown = 0.0 if change_24h is None else change_24h
        return _reject("UNIVERSE", f"24h change {shown:.1f}%: outside the gainer universe")

    sign = 1 if direction == "LONG" else -1
    TPs = [Eavg + sign * m * riesgo for m in GEM_CFG["TP"]]
    if GEM_CFG["TP1_FIXED"]:
        base = entries[0] if entries else E1
        fixed = base * (1 + sign * GEM_CFG["TP1_FIXED_PCT"] / 100)
        ordered = (fixed < TPs[1]) if sign > 0 else (fixed > TPs[1])
        if len(TPs) < 2 or ordered:
            TPs[0] = fixed

    SL_op = SL
    if 0 < GEM_CFG["STOP_APRIETA"] < 1:
        SL_op = (Eavg - GEM_CFG["STOP_APRIETA"] * riesgo) if direction == "LONG" else (Eavg + GEM_CFG["STOP_APRIETA"] * riesgo)
        deepest = min(entries) if direction == "LONG" else max(entries)
        if direction == "LONG" and SL_op >= deepest:
            SL_op = deepest - 0.05 * riesgo
        if direction == "SHORT" and SL_op <= deepest:
            SL_op = deepest + 0.05 * riesgo

    result = {
        "ok": True, "dir": direction, "precio": price, "atr": atr,
        "E1": E1, "E2": E2, "Eavg": Eavg, "SL": SL_op, "SL_original": SL,
        "TPs": TPs, "riesgo": riesgo, "Rpct": Rpct, "Ratr": Ratr,
        "entradas": entries, "pesos": weights,
        "sepATR": sep / atr, "alcanceATR": reach_atr, "volRel": vr,
        "fundingAplicado": funding_applied,
        "flujo": fl["flow"], "salto": vr >= GEM_CFG["VOL_JUMP"], "dragComision": fee_drag,
        "stochK": round(st["K"], 1) if st else None,
        "stochD": round(st["D"], 1) if st else None,
        "stochSube": st["rising"] if st else None,
        "smcTendencia": smc["trend"], "smcBOS": smc["bos"], "smcCHoCH": smc["choch"],
        "smcH1": smc["trend_h1"], "smcBosAFavor": smc["bos_in_favor"], "smcH1AFavor": smc["h1_in_favor"],
        "smcFVG": smc["fvg"], "smcFVGDentro": smc["fvg_inside"], "smcBarrido": smc["sweep"],
        "smcOB": smc["ob"]["bars"] if smc["ob"] else None,
        "riesgoNetoCero": Eavg - sign * net_zero_factor() * riesgo,
    }
    pt = score_setup(result, funding_pct8h)
    result["nota"] = pt["nota"]
    result["componentes"] = pt["componentes"]
    if result["nota"] < GEM_CFG["MIN_SCORE"]:
        out = _reject("LOW_SCORE", f"score {result['nota']}/100, below the minimum {GEM_CFG['MIN_SCORE']}")
        out["nota"] = result["nota"]
        return out
    return result


# ===================== SIGNAL BUILDING =====================


def _dec(x: float) -> int:
    x = abs(x)
    if x >= 1000:
        return 2
    if x >= 1:
        return 4
    if x >= 0.01:
        return 6
    if x >= 0.0001:
        return 8
    return 10


def _round_px(x: float) -> float:
    """GEM's own finer-grained rounding (vs. strategy.py's _round_px):
    the original engine sweeps very low-priced alts, where a coarser
    scheme would round straight to 0."""
    return round(x, _dec(x))


def _pct_weights(fracs: list[float]) -> list[float]:
    """RiskGuard expects entry/TP weights as percentage points summing to
    ~100 (see risk_guard.py); GEM's own pesos/CIERRES are fractions
    summing to ~1."""
    total = sum(fracs) or 1.0
    return [round(f / total * 100, 4) for f in fracs]


def build_signal(symbol: str, result: dict, *, leverage: int, timeframe: str) -> Signal:
    entries = [_round_px(x) for x in result["entradas"]]
    entry_weights = _pct_weights(result["pesos"])
    tps = [_round_px(x) for x in result["TPs"]]
    tp_weights = _pct_weights(list(GEM_CFG["CLOSES"])[:len(tps)])
    sl = _round_px(result["SL"])
    eavg = _round_px(result["Eavg"])
    # GEM spaces its TP ladder off Eavg (TPs = Eavg + m*riesgo), so riesgo
    # (abs(Eavg-SL)) *is* the 1R distance RiskGuard needs here — unlike
    # ZENITH, which spaces TPs off E1 (see strategy.build_signal's r_unit).
    r_unit = abs(result["Eavg"] - result["SL"])
    return Signal(
        symbol=symbol,
        side=result["dir"],
        leverage=leverage,
        timeframe=timeframe,
        reference=_round_px(result["precio"]),
        entries=entries,
        entry_weights=entry_weights,
        tps=tps,
        tp_weights=tp_weights,
        sl=sl,
        score=round(float(result["nota"]), 1),
        reasons=[],
        extras={
            "engine": "GEM",
            "setup": "GEM-FIB_SMC",
            "avg_entry": eavg,
            "risk_pct": round(float(result["Rpct"]), 2),
            "fee_drag_r": round(float(result["dragComision"]), 3),
            "r_unit": round(r_unit, 10),
            "be_after_tp1": bool(GEM_CFG["BE_ON_TP1"]),
            "cancel_velas": int(GEM_CFG["CANCEL_BARS"]),
            "close_velas": int(GEM_CFG["CLOSE_BARS"]),
            "stoch_k": result.get("stochK"),
            "stoch_d": result.get("stochD"),
            "smc_trend": result.get("smcTendencia"),
            "smc_bos": result.get("smcBOS"),
            "smc_choch": result.get("smcCHoCH"),
            "vol_ratio": result.get("volRel"),
        },
    )
