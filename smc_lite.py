"""SMC-lite: deterministic MATRIX-inspired structure / FVG / POI helpers for ZENITH.

Ported conceptually from motor-gem.js (swings, estructuraFuerte, poisActivos,
premiumDescuento, chochLTF) — clean Python, no JS paste. No CVD/OI absorption
in backtest (optional stub only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Default knobs (also mirrored into strategy.FIB_CFG when profile=smc_lite)
SMC_DEFAULTS = {
    "SMC_LITE": True,
    "SMC_SWING_HTF": 2,
    "SMC_SWING_LTF": 2,
    "SMC_FVG_VELAS": 40,
    "SMC_OB_VELAS": 20,
    "SMC_EXPANSION": 1.5,
    "SMC_BUFFER_POI": 0.25,
    "SMC_TOL_IGUALES": 0.15,
    "EXIGIR_BOS": True,          # require HTF bias (trend or BOS)
    "EXIGIR_FVG": False,         # hard: must have live FVG/OB POI nearby
    "EXIGIR_CHOCH_LTF": False,   # hard: 15m CHoCH/BOS confirm
    "EXIGIR_POI_OR_CHOCH": True,
    "EXIGIR_FVG_INSIDE": False,
    "EXIGIR_DESCUENTO": False,   # hard: longs discount / shorts premium
    "SMC_POI_SOFT": True,        # score-boost when FVG/OB near or inside
    "SMC_PD_SOFT": True,         # score-boost for premium/discount alignment
    "SMC_FVG_NEAR_ATR": 2.20,    # "near" POI if within this ATR of zone
    "SMC_ENTRY_MODE": "poi_fib", # poi | fib | poi_fib (POI when available else fib)
    "SMC_MIN_HTF_BARS": 30,
}


@dataclass
class Swing:
    i: int
    p: float


@dataclass
class Structure:
    tendencia: str | None = None  # ALCISTA | BAJISTA
    bos: str | None = None
    choch: bool = False
    fuerte_alto: float | None = None
    fuerte_bajo: float | None = None
    ultimo_alto: float | None = None
    ultimo_bajo: float | None = None


@dataclass
class Zone:
    tipo: str  # FVG | OB
    suelo: float
    techo: float
    i: int
    dentro: bool = False


def _ohlc(candles: list[dict]) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    o = [float(c["open"]) for c in candles]
    h = [float(c["high"]) for c in candles]
    l = [float(c["low"]) for c in candles]
    c_ = [float(c["close"]) for c in candles]
    v = [float(c.get("volume") or 0.0) for c in candles]
    return o, h, l, c_, v


def atr_last(candles: list[dict], period: int = 14) -> float:
    if len(candles) < period + 1:
        return max(float(candles[-1]["close"]) * 0.005, 1e-9)
    _, highs, lows, closes, _ = _ohlc(candles)
    trs = []
    for i in range(1, len(closes)):
        trs.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    return sum(trs[-period:]) / period


def swings(candles: list[dict], n: int = 2) -> tuple[list[Swing], list[Swing]]:
    """Fractal swings: leave last n bars unconfirmed."""
    _, highs, lows, _, _ = _ohlc(candles)
    altos: list[Swing] = []
    bajos: list[Swing] = []
    m = len(candles)
    if m < 2 * n + 1:
        return altos, bajos
    for i in range(n, m - n):
        es_alto = True
        es_bajo = True
        hi, lo = highs[i], lows[i]
        for j in range(i - n, i + n + 1):
            if j == i:
                continue
            if highs[j] >= hi:
                es_alto = False
            if lows[j] <= lo:
                es_bajo = False
            if not es_alto and not es_bajo:
                break
        if es_alto:
            altos.append(Swing(i, hi))
        if es_bajo:
            bajos.append(Swing(i, lo))
    return altos, bajos


def estructura_fuerte(candles: list[dict], n: int = 2) -> Structure:
    altos, bajos = swings(candles, n)
    empty = Structure()
    if len(altos) < 2 or len(bajos) < 2:
        return empty
    ua, pa = altos[-1], altos[-2]
    ub, pb = bajos[-1], bajos[-2]
    tendencia = None
    if ua.p > pa.p and ub.p > pb.p:
        tendencia = "ALCISTA"
    elif ua.p < pa.p and ub.p < pb.p:
        tendencia = "BAJISTA"

    mas_alto = max(altos, key=lambda s: s.p)
    mas_bajo = min(bajos, key=lambda s: s.p)
    antes_alto = [b for b in bajos if b.i < mas_alto.i]
    antes_bajo = [a for a in altos if a.i < mas_bajo.i]
    fuerte_bajo = min(antes_alto, key=lambda s: s.p).p if antes_alto else ub.p
    fuerte_alto = max(antes_bajo, key=lambda s: s.p).p if antes_bajo else ua.p

    cierre = float(candles[-1]["close"])
    bos = None
    choch = False
    if cierre > ua.p:
        bos = "ALCISTA"
    elif cierre < ub.p:
        bos = "BAJISTA"
    if tendencia == "ALCISTA" and cierre < fuerte_bajo:
        choch = True
        bos = "BAJISTA"
    if tendencia == "BAJISTA" and cierre > fuerte_alto:
        choch = True
        bos = "ALCISTA"
    return Structure(
        tendencia=tendencia,
        bos=bos,
        choch=choch,
        fuerte_alto=fuerte_alto,
        fuerte_bajo=fuerte_bajo,
        ultimo_alto=ua.p,
        ultimo_bajo=ub.p,
    )


def htf_bias(candles: list[dict], n: int = 2) -> str | None:
    """Directional bias from HTF structure: prefer BOS after CHoCH, else BOS or trend."""
    e = estructura_fuerte(candles, n)
    if e.choch and e.bos:
        return e.bos
    return e.bos or e.tendencia


def pois_activos(
    candles: list[dict],
    dir_: str,
    atr: float,
    *,
    fvg_lookback: int = 40,
    ob_lookback: int = 20,
    expansion: float = 1.5,
) -> list[Zone]:
    """Unmitigated FVG + order-block zones in trade direction."""
    _, highs, lows, closes, _ = _ohlc(candles)
    opens = [float(c["open"]) for c in candles]
    zonas: list[Zone] = []
    ult = closes[-1]
    m = len(candles)
    start_fvg = max(0, m - fvg_lookback)
    for i in range(m - 3, start_fvg - 1, -1):
        a_h, a_l = highs[i], lows[i]
        c_h, c_l = highs[i + 2], lows[i + 2]
        if dir_ == "LONG" and a_h < c_l:
            mitigated = any(lows[j] <= a_h for j in range(i + 3, m))
            if not mitigated:
                zonas.append(Zone("FVG", a_h, c_l, i, a_h <= ult <= c_l))
        if dir_ == "SHORT" and a_l > c_h:
            mitigated = any(highs[j] >= a_l for j in range(i + 3, m))
            if not mitigated:
                zonas.append(Zone("FVG", c_h, a_l, i, c_h <= ult <= a_l))

    start_ob = max(1, m - ob_lookback)
    for i in range(m - 2, start_ob - 1, -1):
        imp_o, imp_c = opens[i], closes[i]
        prev_o, prev_c = opens[i - 1], closes[i - 1]
        cuerpo = abs(imp_c - imp_o)
        if not (atr > 0 and cuerpo >= expansion * atr):
            continue
        if dir_ == "LONG" and imp_c > imp_o and prev_c < prev_o:
            if not any(lows[j] <= lows[i - 1] for j in range(i + 1, m)):
                zonas.append(
                    Zone("OB", lows[i - 1], highs[i - 1], i - 1, lows[i - 1] <= ult <= highs[i - 1])
                )
        if dir_ == "SHORT" and imp_c < imp_o and prev_c > prev_o:
            if not any(highs[j] >= highs[i - 1] for j in range(i + 1, m)):
                zonas.append(
                    Zone("OB", lows[i - 1], highs[i - 1], i - 1, lows[i - 1] <= ult <= highs[i - 1])
                )
    for z in zonas:
        z.dentro = z.suelo <= ult <= z.techo
    return zonas


def premium_descuento(candles: list[dict], n: int = 2) -> dict[str, Any]:
    altos, bajos = swings(candles, n)
    if not altos or not bajos:
        return {"zona": None, "pos": None}
    alto = max(s.p for s in altos)
    bajo = min(s.p for s in bajos)
    if not (alto > bajo):
        return {"zona": None, "pos": None}
    pos = (float(candles[-1]["close"]) - bajo) / (alto - bajo)
    return {
        "zona": "PREMIUM" if pos > 0.5 else "DESCUENTO",
        "pos": round(pos * 100.0, 1),
        "alto": alto,
        "bajo": bajo,
        "equilibrio": (alto + bajo) / 2.0,
    }


def choch_ltf(candles: list[dict], dir_: str, n: int = 2) -> bool:
    """LTF confirmation: bullish BOS/CHoCH for LONG, bearish for SHORT."""
    e = estructura_fuerte(candles, n)
    if dir_ == "LONG":
        return e.bos == "ALCISTA" or (e.tendencia == "BAJISTA" and e.choch)
    return e.bos == "BAJISTA" or (e.tendencia == "ALCISTA" and e.choch)


def nearest_poi(
    zonas: list[Zone],
    price: float,
    atr: float,
    near_atr: float = 1.2,
) -> Zone | None:
    """Prefer zone containing price; else closest zone within near_atr * ATR."""
    if not zonas:
        return None
    inside = [z for z in zonas if z.dentro]
    if inside:
        return max(inside, key=lambda z: z.i)
    best = None
    best_d = 1e18
    for z in zonas:
        mid = 0.5 * (z.suelo + z.techo)
        d = abs(price - mid)
        if d < best_d:
            best_d = d
            best = z
    if best is None:
        return None
    half = 0.5 * (best.techo - best.suelo)
    if best_d <= max(near_atr * atr, half + 0.15 * atr):
        return best
    return None


def absorb_stub(*_a, **_k) -> dict:
    """Optional CVD/OI absorption stub — always neutral in backtest."""
    return {"ok": False, "favor": None, "note": "absorption not modeled in backtest"}


def aggregate_htf_from_ltf(ltf: list[dict], factor: int = 4) -> list[dict]:
    """Aggregate 15m → 1h when real HTF missing (mirrors aMarcoMayor)."""
    out: list[dict] = []
    for i in range(len(ltf) - factor, -1, -factor):
        g = ltf[i : i + factor]
        if len(g) < factor:
            continue
        out.insert(
            0,
            {
                "ts": g[0].get("ts", 0),
                "open": float(g[0]["open"]),
                "close": float(g[-1]["close"]),
                "high": max(float(x["high"]) for x in g),
                "low": min(float(x["low"]) for x in g),
                "volume": sum(float(x.get("volume") or 0) for x in g),
            },
        )
    return out
