"""ZENITH SMC-LITE + MATRIX-ORIENT exits: HTF BOS/FVG/POI entries, MATRIX TP/BE/lev.

SMC-lite (default): swing structure bias, optional FVG/OB POI entries, premium/discount,
LTF CHoCH soft/hard gates — plus retained quality pack (ER, stretch, impulse, funding,
denylist / $20M floor in bot). Exits stay MATRIX-ORIENT (early TP, BE after TP1, 8–12x).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import smc_lite
from smc_lite import SMC_DEFAULTS

# ---------------------------------------------------------------------------
# MATRIX-ORIENT live defaults (STRATEGY_PROFILE=matrix_orient)
# Borrowed from Cornix MATRIX / MATRIX Fib BT: early TP mix, STOP_APRIETA,
# wider R, short holds. ZENITH keeps HTF / stretch / impulse / ER / vol floor.
# ---------------------------------------------------------------------------
FIB_CFG = {
    "STRATEGY_PROFILE": "smc_lite",
    "IMPULSO_VELAS": 24,
    "ALCANCE_MAX_ATR": 1.60,
    # Tightened risk band (was 0.95-2.2%/0.65-2.0 ATR) to offset higher
    # 10-15x leverage and cap tail-loser size — targets ~3% max drawdown.
    "R_MIN_PCT": 1.00,
    "R_MAX_PCT": 1.85,
    "R_MIN_ATR": 0.65,
    "R_MAX_ATR": 1.70,
    "SL_CAP_PCT": 1.40,
    # Raised from 0.45: backtest diagnostics (vol_ratio field) showed clean
    # wins (TP_ALL) average ~28% higher confirm-bar volume than fast
    # failures (SL_DURING_FILL, where price sweeps the whole entry ladder
    # and stops out within the cancel window) -- weak-volume confirm bars
    # correlate with immediate reversals. On fill-weighted equity impact
    # (not just raw win rate), 0.8 turned a -1.1% backtested year into a
    # consistently positive one across a range of nearby thresholds.
    "VOL_MIN": 0.8,
    "ER_MIN": 0.24,
    "ER_MIN_SHORT": 0.28,
    "ATR_PCT_MIN": 0.12,
    "ATR_PCT_MAX": 1.60,
    "EMA_FAST": 50,
    "EMA_SLOW": 160,
    "FUNDING_VETO": 0.12,
    "FUNDING_VETO_LONG": 0.12,
    "FUNDING_VETO_SHORT": 0.10,
    "COMISION_PCT": 0.13,
    "MAX_COMISION_R": 0.45,
    "BUFFER_ATR": 0.55,
    "FIB_ENTRIES": [0.382, 0.50, 0.618, 0.786],
    "FIB_MIN_DEPTH": 0.50,
    "HARD_FIB50": False,
    "HARD_E2_CHASE": False,
    # On by default: veto alt entries against the prevailing BTC regime.
    # Alts are BTC-beta; this cuts correlated counter-trend losers that
    # otherwise cluster and drive drawdown during a BTC impulse.
    "BTC_REGIME": True,
    "ENTRY_WEIGHTS": [45.0, 30.0, 15.0, 10.0],
    # 5 Cornix TPs — MATRIX-inspired distances with heavy early take-profit
    # (~70% at TP1 ~0.45R, then 0.8 / 1.4 / 2.0 / 2.4 style)
    "TP": [0.2, 0.5, 0.95, 1.5, 2.3],
    "CIERRES": [80.0, 10.0, 5.0, 3.0, 2.0],
    # ~3–4h hold on 15m: CLOSE 14 bars ≈ 3.5h; cancel unanswered limits sooner
    "CANCEL_VELAS": 8,
    "CLOSE_VELAS": 20,
    "GAINER_24H_MIN": 1.8,
    "STRETCH_ATR_MAX": 2.20,
    "IMPULSE_VOL_MIN": 1.15,
    "HTF_EMA_FAST": 50,
    "HTF_EMA_SLOW": 160,
    "HTF_ALIGN_TOL": 0.002,
    "HTF_ALIGN_TOL_SHORT": 0.0015,
    # Equity risked per trade, in %. Lowered from 1.00 alongside the higher
    # leverage band so per-trade equity impact (and simulated DD) stays put.
    "RISK_EQUITY_PCT": 0.65,
    # Leverage band raised to 10-15x; leverage_from_quality() clamps this
    # further per-trade via LIQ_SAFETY_FRACTION so SL-hit loss never gets
    # close to isolated-margin liquidation.
    "LEV_MIN": 10,
    "LEV_MAX": 15,
    # Max fraction of isolated margin that the stop-loss distance may burn
    # at the chosen leverage (rpct/100 * leverage <= this). Keeps 15x setups
    # away from liquidation even on the wider end of the risk band.
    "LIQ_SAFETY_FRACTION": 0.50,
    "ALLOW_SHORTS": True,
    "LONG_ONLY": False,
    # After TP1: move SL to BE; optionally tighten residual stop toward 0.85×R
    "STOP_APRIETA": 1.0,
    "BE_AFTER_TP1": True,
    # Lock a little profit rather than exact breakeven after TP1 (in R of
    # the *original* stop distance) — turns give-back scratch trades into
    # small winners, lifting both winrate and return together.
    "BE_BUFFER_R": 0.12,
    # Shift size from TP1 toward the further targets as setup score rises,
    # so the best-quality entries harvest more of the 1.5-2.3R distance
    # instead of capping 80% of size at ~0.2R. Same stop/risk, more upside.
    "ADAPTIVE_TP": True,
    # Reject the confirm bar if it shows a strong opposite-direction wick
    # (rejection/exhaustion) even though the close/body agreed with side.
    "WICK_REJECT_MAX": 0.45,
    "MIN_SCORE": 70.0,
    # SMC-lite entry knobs (STRATEGY_PROFILE=smc_lite)
    "SMC_LITE": True,
    "SMC_SWING_HTF": 2,
    "SMC_SWING_LTF": 2,
    "SMC_FVG_VELAS": 40,
    "SMC_OB_VELAS": 20,
    "SMC_EXPANSION": 1.5,
    "SMC_BUFFER_POI": 0.25,
    "EXIGIR_BOS": True,
    "EXIGIR_FVG": False,
    "EXIGIR_CHOCH_LTF": False,
    "EXIGIR_POI_OR_CHOCH": True,
    "EXIGIR_FVG_INSIDE": False,
    "EXIGIR_DESCUENTO": False,
    "SMC_POI_SOFT": True,
    "SMC_PD_SOFT": True,
    "SMC_FVG_NEAR_ATR": 2.20,
    "SMC_ENTRY_MODE": "poi_fib",
    "SMC_MIN_HTF_BARS": 30,
}

# Legacy riskcap snapshot (opt-in via STRATEGY_PROFILE=riskcap)
_RISKCAP_OVERRIDES = {
    "STRATEGY_PROFILE": "riskcap",
    "R_MIN_PCT": 0.95,
    "R_MAX_PCT": 1.70,
    "R_MIN_ATR": 0.65,
    "R_MAX_ATR": 2.00,
    "SL_CAP_PCT": 1.65,
    "TP": [0.20, 0.42, 0.75, 1.20, 1.90],
    "CIERRES": [88.0, 6.0, 3.0, 2.0, 1.0],
    "CANCEL_VELAS": 8,
    "CLOSE_VELAS": 32,
    "LEV_MIN": 5,
    "LEV_MAX": 25,
    "STOP_APRIETA": 1.0,
    "MIN_SCORE": 65.0,
}


def apply_strategy_profile(name: str | None = None) -> str:
    """Apply named FIB_CFG profile. Default → smc_lite (MATRIX exits + SMC entries)."""
    profile = (name or FIB_CFG.get("STRATEGY_PROFILE") or "smc_lite").strip().lower()
    if profile in ("riskcap", "matrix_riskcap", "legacy"):
        FIB_CFG.update(_RISKCAP_OVERRIDES)
        FIB_CFG["SMC_LITE"] = False
        return "riskcap"
    if profile in ("matrix_orient", "matrix", "orient"):
        FIB_CFG["STRATEGY_PROFILE"] = "matrix_orient"
        FIB_CFG["SMC_LITE"] = False
        return "matrix_orient"
    # smc_lite (default): MATRIX exits + SMC entry helpers
    FIB_CFG["STRATEGY_PROFILE"] = "smc_lite"
    FIB_CFG["SMC_LITE"] = True
    for k, v in SMC_DEFAULTS.items():
        FIB_CFG.setdefault(k, v)
    FIB_CFG["SMC_LITE"] = True
    return "smc_lite"


@dataclass
class Signal:
    symbol: str
    side: str
    leverage: int
    timeframe: str
    reference: float
    entries: list[float]
    entry_weights: list[float]
    tps: list[float]
    tp_weights: list[float]
    sl: float
    score: float
    reasons: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)


def closed_candles(candles: list[dict]) -> list[dict]:
    """Drop the still-forming last bar so signals don't flip mid-candle."""
    if len(candles) < 3:
        return list(candles)
    return candles[:-1]


def leverage_from_quality(
    score: float,
    *,
    side: str = "LONG",
    lev_min: int | None = None,
    lev_max: int | None = None,
    rpct: float | None = None,
    ratr: float | None = None,
    vol_ratio: float | None = None,
) -> int:
    """Map setup quality into [lev_min, lev_max].

    Primary driver is score (65→min, ~94→max). Wider stop % / ATR risk gently
    pulls leverage down; strong volume gently pushes it up. `rpct` is percent.
    Defaults come from FIB_CFG (10-15x band, quality-scaled).

    A hard liquidation-safety clamp caps leverage so that
    rpct/100 * leverage never exceeds LIQ_SAFETY_FRACTION (default 0.50) —
    i.e. a full stop-out burns at most half of isolated margin, keeping the
    15x end of the band well clear of liquidation even on the wider stops.
    """
    lo = int(FIB_CFG["LEV_MIN"] if lev_min is None else lev_min)
    hi = int(FIB_CFG["LEV_MAX"] if lev_max is None else lev_max)
    s = max(0.0, min(100.0, float(score)))
    t = max(0.0, min(1.0, (s - 65.0) / 29.0))
    if vol_ratio is not None:
        t *= max(0.85, min(1.08, 0.75 + 0.2 * float(vol_ratio)))
        t = max(0.0, min(1.0, t))
    ladder = float(lo) + t * (float(hi) - float(lo))
    if rpct and rpct > 0:
        # Wider stops → slightly less leverage (band ~1.5..2.8 for matrix_orient)
        ladder *= max(0.82, min(1.08, 2.10 / float(rpct)))
    if ratr is not None and ratr > 0:
        ladder *= max(0.88, min(1.06, 1.35 - 0.2 * float(ratr)))
    lev = max(float(lo), min(float(hi), ladder))
    if rpct and rpct > 0:
        # Hard ceiling — never re-raised back to lo, since lo could itself
        # be unsafe for an unusually wide stop (e.g. a misconfigured R band).
        liq_frac = float(FIB_CFG.get("LIQ_SAFETY_FRACTION", 0.50) or 0.50)
        liq_cap = (liq_frac * 100.0) / float(rpct)
        lev = min(lev, liq_cap)
    lev = max(1.0, min(float(hi), lev))
    return int(max(1, min(hi, round(lev))))


def _round_px(px: float) -> float:
    if px >= 1000:
        return round(px, 1)
    if px >= 100:
        return round(px, 2)
    if px >= 1:
        return round(px, 4)
    return round(px, 6)


def _floor_clamp_entries(entries: list[float], sl_floor: float, atr: float, bull: bool) -> list[float]:
    """Clamp every entry to at least `sl_floor` away from SL, like a plain
    max()/min() clamp would -- but when a tight SL_CAP_PCT pushes the floor
    above two or more of the deepest fib/POI entries, don't let them all
    collapse onto the exact same clamped price (that posts a redundant
    duplicate limit order to Cornix at the same level instead of a real
    second one). Walk from the entry closest to SL outward and nudge each
    earlier one at least one `step` farther out whenever the floor would
    otherwise tie it with the entry just inside it."""
    ordered = sorted(entries, reverse=bull)
    step = max(atr * 0.02, abs(sl_floor) * 0.0005, 1e-9)
    out = list(ordered)
    for i in range(len(out) - 1, -1, -1):
        v = max(out[i], sl_floor) if bull else min(out[i], sl_floor)
        if i < len(out) - 1:
            nxt = out[i + 1]
            if bull and v <= nxt:
                v = nxt + step
            elif not bull and v >= nxt:
                v = nxt - step
        out[i] = v
    return out


def _ohlcv(candles: list[dict]):
    opens = [float(c["open"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    closes = [float(c["close"]) for c in candles]
    vols = [float(c.get("volume") or 0.0) for c in candles]
    return opens, highs, lows, closes, vols


def _atr(highs, lows, closes, period: int = 14) -> float:
    if len(closes) < period + 1:
        return max(closes[-1] * 0.005, 1e-9)
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    return sum(trs[-period:]) / period


def _vol_sma(vols, period: int = 20) -> float:
    if not vols:
        return 1.0
    w = vols[-period:] if len(vols) >= period else vols
    return max(sum(w) / len(w), 1e-9)


def _ema(xs: list[float], period: int) -> float:
    if not xs:
        return 0.0
    k = 2.0 / (period + 1.0)
    e = xs[0]
    for x in xs[1:]:
        e = k * x + (1.0 - k) * e
    return e


def _efficiency_ratio(closes: list[float], period: int = 20) -> float:
    if len(closes) < period + 1:
        return 0.0
    change = abs(closes[-1] - closes[-1 - period])
    path = sum(abs(closes[i] - closes[i - 1]) for i in range(len(closes) - period, len(closes)))
    return (change / path) if path > 0 else 0.0


def _wick_rejected(bar_o: float, bar_hi: float, bar_lo: float, bar_c: float, bull: bool, max_frac: float) -> bool:
    """True if the confirm bar shows a strong opposite-direction wick.

    A long that closes green but stabs well above the body into resistance
    (or a short that closes red but wicks well below into support) often
    marks exhaustion/rejection rather than a clean breakout — even though
    the close/body direction alone passes the existing confirm-bar check.
    """
    if max_frac <= 0:
        return False
    rng = bar_hi - bar_lo
    if rng <= 0:
        return False
    if bull:
        wick = bar_hi - max(bar_o, bar_c)
    else:
        wick = min(bar_o, bar_c) - bar_lo
    return (wick / rng) > max_frac


def var24h(closes: list[float]) -> float:
    if len(closes) <= 96 or closes[-97] <= 0:
        return 0.0
    return (closes[-1] / closes[-97] - 1.0) * 100.0


def btc_regime_from_candles(htf_candles: list[dict] | None) -> str:
    """Return 'bull' | 'bear' | 'neutral' from BTC 1h closed candles."""
    C = FIB_CFG
    if not htf_candles:
        return "neutral"
    need = int(C["HTF_EMA_SLOW"]) + 8
    if len(htf_candles) < need:
        return "neutral"
    _, _, _, closes, _ = _ohlcv(htf_candles)
    ema_f = _ema(closes, int(C["HTF_EMA_FAST"]))
    ema_s = _ema(closes, int(C["HTF_EMA_SLOW"]))
    last = closes[-1]
    if ema_f >= ema_s and last >= ema_f:
        return "bull"
    if ema_f <= ema_s and last <= ema_f:
        return "bear"
    return "neutral"


def evaluate_fib(
    candles: list[dict],
    funding: float | None = None,
    *,
    require_gainer: bool = True,
    htf_candles: list[dict] | None = None,
    require_htf: bool = False,
    entry_weights: list[float] | None = None,
    btc_regime: str | None = None,
    symbol: str | None = None,
    allow_shorts: bool | None = None,
    long_only: bool | None = None,
) -> dict | None:
    C = FIB_CFG
    n = len(candles)
    need = max(int(C["EMA_SLOW"]) + 8, int(C["IMPULSO_VELAS"]) + 16, 97 if require_gainer else 50)
    if n < need:
        return None
    opens, highs, lows, closes, vols = _ohlcv(candles)
    last = closes[-1]
    if last <= 0:
        return None
    atr = _atr(highs, lows, closes)
    if atr <= 0:
        return None
    atr_pct = atr / last * 100.0
    if atr_pct < float(C["ATR_PCT_MIN"]) or atr_pct > float(C["ATR_PCT_MAX"]):
        return None
    vol_ratio = vols[-1] / _vol_sma(vols)
    if vol_ratio < float(C["VOL_MIN"]):
        return None
    er = _efficiency_ratio(closes, 20)
    if er < float(C["ER_MIN"]):
        return None
    ema_f = _ema(closes, int(C["EMA_FAST"]))
    ema_s = _ema(closes, int(C["EMA_SLOW"]))

    stretch = abs(last - ema_f) / atr
    if stretch > float(C["STRETCH_ATR_MAX"]):
        return None

    htf_aligned = False
    htf_bull_ok = htf_bear_ok = True
    htf_ready = False
    if htf_candles:
        need_htf = int(C["HTF_EMA_SLOW"]) + 8
        if len(htf_candles) >= need_htf:
            htf_ready = True
            _, _, _, htf_c, _ = _ohlcv(htf_candles)
            htf_f = _ema(htf_c, int(C["HTF_EMA_FAST"]))
            htf_s = _ema(htf_c, int(C["HTF_EMA_SLOW"]))
            htf_last = htf_c[-1]
            tol_l = float(C["HTF_ALIGN_TOL"])
            tol_s = float(C.get("HTF_ALIGN_TOL_SHORT", C["HTF_ALIGN_TOL"]))
            htf_bull_ok = htf_last >= htf_f * (1.0 - tol_l) and htf_f >= htf_s * (1.0 - tol_l)
            htf_bear_ok = htf_last <= htf_f * (1.0 + tol_s) and htf_f <= htf_s * (1.0 + tol_s)
        elif require_htf:
            return None
    elif require_htf:
        return None

    look = int(C["IMPULSO_VELAS"])
    w_h, w_l = highs[-look:], lows[-look:]
    hh, ll = max(w_h), min(w_l)
    hh_i = look - 1 - w_h[::-1].index(hh)
    ll_i = look - 1 - w_l[::-1].index(ll)
    rng = hh - ll
    if rng <= atr * 0.35 or hh_i == ll_i:
        return None
    base = n - look
    a0, a1 = base + min(ll_i, hh_i), base + max(ll_i, hh_i)
    a1 = min(n - 1, max(a0, a1))
    impulse_slice = vols[a0 : a1 + 1] or vols[-look:]
    impulse_vol = sum(impulse_slice) / len(impulse_slice)
    vol_sma = _vol_sma(vols)
    impulse_vol_ratio = impulse_vol / vol_sma
    if impulse_vol_ratio < float(C["IMPULSE_VOL_MIN"]):
        return None
    bull = hh_i > ll_i
    if bull and not (last >= ema_f * 0.997 and ema_f >= ema_s * 0.998):
        return None
    if not bull and not (last <= ema_f * 1.003 and ema_f <= ema_s * 1.002):
        return None
    side = "LONG" if bull else "SHORT"

    _long_only = bool(C["LONG_ONLY"] if long_only is None else long_only)
    _allow_shorts = bool(C["ALLOW_SHORTS"] if allow_shorts is None else allow_shorts)
    if _long_only and not bull:
        return None
    if not _allow_shorts and not bull:
        return None

    if not bull and er < float(C.get("ER_MIN_SHORT", C["ER_MIN"])):
        return None

    if bull and not htf_bull_ok:
        return None
    if not bull and not htf_bear_ok:
        return None
    if htf_ready:
        htf_aligned = (bull and htf_bull_ok) or ((not bull) and htf_bear_ok)

    sym_u = (symbol or "").upper()
    is_btc = sym_u in ("BTCUSDT", "BTCUSD", "BTC")
    regime = (btc_regime or "neutral").lower()
    if C.get("BTC_REGIME", False) and not is_btc:
        if regime == "bull" and not bull:
            return None
        if regime == "bear" and bull:
            return None

    v24 = var24h(closes)
    if require_gainer:
        floor = float(C["GAINER_24H_MIN"])
        if bull and v24 < floor:
            return None
        if not bull and v24 > -floor:
            return None

    last_rng = highs[-1] - lows[-1]
    if last_rng > 0:
        last_body = abs(closes[-1] - opens[-1]) / last_rng
        last_bull = closes[-1] >= opens[-1]
        if last_body >= 0.62 and last_bull != bull:
            return None

    if funding is not None:
        veto_l = float(C.get("FUNDING_VETO_LONG", C["FUNDING_VETO"]))
        veto_s = float(C.get("FUNDING_VETO_SHORT", C["FUNDING_VETO"]))
        if bull and funding > veto_l:
            return None
        if not bull and funding < -veto_s:
            return None

    struct_sl = (ll - float(C["BUFFER_ATR"]) * atr) if bull else (hh + float(C["BUFFER_ATR"]) * atr)
    cap = float(C["SL_CAP_PCT"]) / 100.0
    fibs = list(C["FIB_ENTRIES"])
    entries = [(hh - f * rng) if bull else (ll + f * rng) for f in fibs]
    entries = sorted(entries, reverse=bull)
    while len(entries) < 4:
        entries.append(entries[-1])
    entries = entries[:4]
    e1 = entries[0]
    cap_sl = e1 * (1.0 - cap) if bull else e1 * (1.0 + cap)
    sl = max(struct_sl, cap_sl) if bull else min(struct_sl, cap_sl)
    if bull:
        sl = min(sl, e1 * (1.0 - 0.0065))
        entries = _floor_clamp_entries(entries, sl + 0.12 * atr, atr, bull=True)
    else:
        sl = max(sl, e1 * (1.0 + 0.0065))
        entries = _floor_clamp_entries(entries, sl - 0.12 * atr, atr, bull=False)
    e1, e2, e3, e4 = entries[:4]
    ew = list(entry_weights) if entry_weights and len(entry_weights) >= 4 else list(C["ENTRY_WEIGHTS"])
    ew = [float(x) for x in ew[:4]]
    s = sum(ew) or 1.0
    eavg = sum(p * (w / s) for p, w in zip(entries, ew))
    risk = abs(e1 - sl)
    if risk <= 0:
        return None
    rpct = risk / e1 * 100.0
    ratr = risk / atr
    if not (float(C["R_MIN_PCT"]) <= rpct <= float(C["R_MAX_PCT"])):
        return None
    if not (float(C["R_MIN_ATR"]) <= ratr <= float(C["R_MAX_ATR"])):
        return None
    reach = abs(last - e1) / atr
    if reach > float(C["ALCANCE_MAX_ATR"]):
        return None
    if bull and last < e4:
        return None
    if not bull and last > e4:
        return None

    depth = float(C.get("FIB_MIN_DEPTH", 0.50))
    fib_depth_px = (hh - depth * rng) if bull else (ll + depth * rng)
    if bull:
        i0 = base + hh_i
        tagged_50 = i0 < n and min(lows[i0:]) <= fib_depth_px
    else:
        i0 = base + ll_i
        tagged_50 = i0 < n and max(highs[i0:]) >= fib_depth_px
    if C.get("HARD_FIB50", False) and not tagged_50:
        return None
    at_e2_or_deeper = (last <= e2) if bull else (last >= e2)
    if C.get("HARD_E2_CHASE", False) and not at_e2_or_deeper:
        return None

    fee = float(C["COMISION_PCT"])
    drag = (2.0 * fee) / rpct if rpct else 99.0
    if drag > float(C["MAX_COMISION_R"]):
        return None
    tps = [(e1 + risk * m) if bull else (e1 - risk * m) for m in C["TP"]]

    # Confirmation bar (last closed): no same-bar SL geometry; close in direction
    bar_hi, bar_lo = highs[-1], lows[-1]
    bar_o, bar_c = opens[-1], closes[-1]
    if bull:
        if bar_lo <= sl:
            return None
        if bar_c < bar_o:
            return None
    else:
        if bar_hi >= sl:
            return None
        if bar_c > bar_o:
            return None
    if _wick_rejected(bar_o, bar_hi, bar_lo, bar_c, bull, float(C.get("WICK_REJECT_MAX", 0.0) or 0.0)):
        return None

    # Operational SL: MATRIX STOP_APRIETA tightens managed stop toward Eavg
    stop_aprieta = float(C.get("STOP_APRIETA", 1.0) or 1.0)
    risk_eavg = abs(eavg - sl)
    if 0 < stop_aprieta < 1.0 and risk_eavg > 0:
        sl_op = (eavg - stop_aprieta * risk_eavg) if bull else (eavg + stop_aprieta * risk_eavg)
        # Never loosen past structural/filter SL; only tighten toward entry
        if bull:
            sl_op = max(sl_op, sl)
        else:
            sl_op = min(sl_op, sl)
    else:
        sl_op = sl

    score = 70.0
    score += min(8.0, vol_ratio * 3.5)
    score += min(8.0, er * 18.0)
    score += min(6.0, abs(v24) / 4.0)
    score += 6.0 if reach <= 0.55 else 2.0
    score += 4.0 if (bull and last > ema_f) or (not bull and last < ema_f) else 0.0
    score += min(5.0, max(0.0, (impulse_vol_ratio - 1.0) * 4.0))
    score += 3.0 if stretch <= 1.0 else (1.0 if stretch <= 1.6 else 0.0)
    if htf_aligned:
        score += 4.0
    if tagged_50:
        score += 2.0
    if C.get("BTC_REGIME", False) and regime in ("bull", "bear") and (
        (bull and regime == "bull") or ((not bull) and regime == "bear")
    ):
        score += 2.0
    score -= drag * 6.0
    score = max(0.0, min(99.0, score))
    return {
        "ok": True,
        "side": side,
        "E1": e1,
        "E2": e2,
        "E3": e3,
        "E4": e4,
        "entries": entries,
        "entry_weights": ew,
        "Eavg": eavg,
        "SL": sl,
        "SL_op": sl_op,
        "TPs": tps,
        "Rpct": rpct,
        "Ratr": ratr,
        "dragComision": drag,
        "score": score,
        "last": last,
        "atr": atr,
        "atr_pct": atr_pct,
        "vol_ratio": vol_ratio,
        "change_pct": v24,
        "impulse_high": hh,
        "impulse_low": ll,
        "stretch_atr": stretch,
        "impulse_vol_ratio": impulse_vol_ratio,
        "htf_aligned": htf_aligned,
        "btc_regime": regime,
        "stop_aprieta": stop_aprieta,
        "reasons": [
            f"EMA {int(C['EMA_FAST'])}/{int(C['EMA_SLOW'])} + ER {er:.2f}",
            f"fib R {rpct:.2f}% cap {C['SL_CAP_PCT']}%",
            f"fee-drag {drag:.2f}R",
            f"24h {v24:.1f}%",
            f"stretch {stretch:.2f}ATR · impulse vol {impulse_vol_ratio:.2f}x",
            ("HTF aligned" if htf_aligned else "HTF not required/off"),
            (f"BTC regime {regime}" if C.get("BTC_REGIME", False) else "BTC regime off"),
            ("fib0.50 tagged" if tagged_50 else "normal fib entries"),
            f"profile {C.get('STRATEGY_PROFILE', 'matrix_orient')} STOP_APRIETA={stop_aprieta}",
        ],
    }


def evaluate_smc_lite(
    candles: list[dict],
    funding: float | None = None,
    *,
    require_gainer: bool = True,
    htf_candles: list[dict] | None = None,
    require_htf: bool = False,
    entry_weights: list[float] | None = None,
    btc_regime: str | None = None,
    symbol: str | None = None,
    allow_shorts: bool | None = None,
    long_only: bool | None = None,
) -> dict | None:
    """SMC-lite entries + MATRIX-orient geometry/exits.

    Direction from HTF strong structure (BOS/trend). Entries prefer live FVG/OB
    POI proximity; fall back to fib pullback when SMC_ENTRY_MODE=poi_fib.
    Keeps ER / stretch / impulse / funding / confirm-bar hygiene.
    """
    C = FIB_CFG
    n = len(candles)
    need = max(int(C["EMA_SLOW"]) + 8, int(C["IMPULSO_VELAS"]) + 16, 97 if require_gainer else 50)
    if n < need:
        return None
    opens, highs, lows, closes, vols = _ohlcv(candles)
    last = closes[-1]
    if last <= 0:
        return None
    atr = _atr(highs, lows, closes)
    if atr <= 0:
        return None
    atr_pct = atr / last * 100.0
    if atr_pct < float(C["ATR_PCT_MIN"]) or atr_pct > float(C["ATR_PCT_MAX"]):
        return None
    vol_ratio = vols[-1] / _vol_sma(vols)
    if vol_ratio < float(C["VOL_MIN"]):
        return None
    er = _efficiency_ratio(closes, 20)
    if er < float(C["ER_MIN"]):
        return None
    ema_f = _ema(closes, int(C["EMA_FAST"]))
    ema_s = _ema(closes, int(C["EMA_SLOW"]))
    stretch = abs(last - ema_f) / atr
    if stretch > float(C["STRETCH_ATR_MAX"]):
        return None

    # --- HTF structure bias (real 1h or aggregated) ---
    htf = list(htf_candles) if htf_candles else None
    if (not htf or len(htf) < int(C.get("SMC_MIN_HTF_BARS", 30))) and not require_htf:
        htf = smc_lite.aggregate_htf_from_ltf(candles, 4)
    if require_htf and (not htf or len(htf) < int(C.get("SMC_MIN_HTF_BARS", 30))):
        return None
    if not htf or len(htf) < int(C.get("SMC_MIN_HTF_BARS", 30)):
        return None

    swing_htf = int(C.get("SMC_SWING_HTF", 2))
    bias = smc_lite.htf_bias(htf, swing_htf)
    estruct = smc_lite.estructura_fuerte(htf, swing_htf)
    if C.get("EXIGIR_BOS", True) and not bias:
        return None

    # Also keep EMA HTF soft alignment as secondary
    htf_aligned = False
    htf_bull_ok = htf_bear_ok = True
    need_htf_ema = int(C["HTF_EMA_SLOW"]) + 8
    if len(htf) >= need_htf_ema:
        _, _, _, htf_c, _ = _ohlcv(htf)
        htf_f = _ema(htf_c, int(C["HTF_EMA_FAST"]))
        htf_s = _ema(htf_c, int(C["HTF_EMA_SLOW"]))
        htf_last = htf_c[-1]
        tol_l = float(C["HTF_ALIGN_TOL"])
        tol_s = float(C.get("HTF_ALIGN_TOL_SHORT", C["HTF_ALIGN_TOL"]))
        htf_bull_ok = htf_last >= htf_f * (1.0 - tol_l) and htf_f >= htf_s * (1.0 - tol_l)
        htf_bear_ok = htf_last <= htf_f * (1.0 + tol_s) and htf_f <= htf_s * (1.0 + tol_s)

    # Impulse window (kept for fib fallback + vol quality)
    look = int(C["IMPULSO_VELAS"])
    w_h, w_l = highs[-look:], lows[-look:]
    hh, ll = max(w_h), min(w_l)
    hh_i = look - 1 - w_h[::-1].index(hh)
    ll_i = look - 1 - w_l[::-1].index(ll)
    rng = hh - ll
    if rng <= atr * 0.35 or hh_i == ll_i:
        return None
    base = n - look
    a0, a1 = base + min(ll_i, hh_i), base + max(ll_i, hh_i)
    a1 = min(n - 1, max(a0, a1))
    impulse_slice = vols[a0 : a1 + 1] or vols[-look:]
    impulse_vol = sum(impulse_slice) / len(impulse_slice)
    vol_sma = _vol_sma(vols)
    impulse_vol_ratio = impulse_vol / vol_sma
    if impulse_vol_ratio < float(C["IMPULSE_VOL_MIN"]):
        return None

    # Side from impulse (fib-compatible); HTF bias gates when EXIGIR_BOS
    impulse_bull = hh_i > ll_i
    bull = impulse_bull
    if C.get("EXIGIR_BOS", True):
        if bias == "ALCISTA" and not bull:
            return None
        if bias == "BAJISTA" and bull:
            return None
        if not bias:
            return None

    if bull and not (last >= ema_f * 0.997 and ema_f >= ema_s * 0.998):
        return None
    if not bull and not (last <= ema_f * 1.003 and ema_f <= ema_s * 1.002):
        return None

    side = "LONG" if bull else "SHORT"
    _long_only = bool(C["LONG_ONLY"] if long_only is None else long_only)
    _allow_shorts = bool(C["ALLOW_SHORTS"] if allow_shorts is None else allow_shorts)
    if _long_only and not bull:
        return None
    if not _allow_shorts and not bull:
        return None
    if not bull and er < float(C.get("ER_MIN_SHORT", C["ER_MIN"])):
        return None

    if bull and not htf_bull_ok:
        return None
    if not bull and not htf_bear_ok:
        return None
    htf_aligned = (bull and htf_bull_ok) or ((not bull) and htf_bear_ok)

    sym_u = (symbol or "").upper()
    is_btc = sym_u in ("BTCUSDT", "BTCUSD", "BTC")
    regime = (btc_regime or "neutral").lower()
    if C.get("BTC_REGIME", False) and not is_btc:
        if regime == "bull" and not bull:
            return None
        if regime == "bear" and bull:
            return None

    v24 = var24h(closes)
    if require_gainer:
        floor = float(C["GAINER_24H_MIN"])
        if bull and v24 < floor:
            return None
        if not bull and v24 > -floor:
            return None

    last_rng = highs[-1] - lows[-1]
    if last_rng > 0:
        last_body = abs(closes[-1] - opens[-1]) / last_rng
        last_bull = closes[-1] >= opens[-1]
        if last_body >= 0.62 and last_bull != bull:
            return None

    if funding is not None:
        veto_l = float(C.get("FUNDING_VETO_LONG", C["FUNDING_VETO"]))
        veto_s = float(C.get("FUNDING_VETO_SHORT", C["FUNDING_VETO"]))
        if bull and funding > veto_l:
            return None
        if not bull and funding < -veto_s:
            return None

    # --- POI / FVG on HTF (and soft LTF FVG) ---
    atr_htf = smc_lite.atr_last(htf)
    zonas = smc_lite.pois_activos(
        htf,
        side,
        atr_htf,
        fvg_lookback=int(C.get("SMC_FVG_VELAS", 40)),
        ob_lookback=int(C.get("SMC_OB_VELAS", 20)),
        expansion=float(C.get("SMC_EXPANSION", 1.5)),
    )
    # Also scan LTF FVGs for proximity boost
    zonas_ltf = smc_lite.pois_activos(
        candles,
        side,
        atr,
        fvg_lookback=min(40, int(C.get("SMC_FVG_VELAS", 40))),
        ob_lookback=min(20, int(C.get("SMC_OB_VELAS", 20))),
        expansion=float(C.get("SMC_EXPANSION", 1.5)),
    )
    near_atr = float(C.get("SMC_FVG_NEAR_ATR", 1.20))
    poi = smc_lite.nearest_poi(zonas, last, atr, near_atr)
    if poi is None:
        poi = smc_lite.nearest_poi(zonas_ltf, last, atr, near_atr)
    poi_inside = bool(poi and poi.dentro)
    poi_near = poi is not None

    if C.get("EXIGIR_FVG", False) and not poi_near:
        return None

    # LTF CHoCH confirm (15m stands in for MATRIX 5m)
    choch_ok = smc_lite.choch_ltf(candles, side, int(C.get("SMC_SWING_LTF", 2)))
    if C.get("EXIGIR_CHOCH_LTF", False) and not choch_ok:
        return None
    if C.get("EXIGIR_POI_OR_CHOCH", False) and not (poi_near or choch_ok):
        return None
    if C.get("EXIGIR_FVG_INSIDE", False) and not poi_inside:
        return None

    pd = smc_lite.premium_descuento(htf, swing_htf)
    if C.get("EXIGIR_DESCUENTO", False) and pd.get("zona"):
        if bull and pd["zona"] == "PREMIUM":
            return None
        if (not bull) and pd["zona"] == "DESCUENTO":
            return None
    pd_aligned = False
    if pd.get("zona"):
        pd_aligned = (bull and pd["zona"] == "DESCUENTO") or ((not bull) and pd["zona"] == "PREMIUM")

    # --- Entries: POI-based or fib fallback ---
    mode = str(C.get("SMC_ENTRY_MODE", "poi_fib")).lower()
    used_poi = False
    fibs = list(C["FIB_ENTRIES"])
    if mode in ("poi", "poi_fib") and poi is not None:
        suelo, techo = float(poi.suelo), float(poi.techo)
        mid = 0.5 * (suelo + techo)
        if bull:
            entries = [techo, mid, suelo, suelo - 0.15 * atr]
            entries = sorted(entries, reverse=True)
        else:
            entries = [suelo, mid, techo, techo + 0.15 * atr]
            entries = sorted(entries)
        buf = float(C.get("SMC_BUFFER_POI", C.get("BUFFER_ATR", 0.55)))
        struct_sl = (suelo - buf * atr) if bull else (techo + buf * atr)
        used_poi = True
    else:
        if mode == "poi":
            return None
        entries = [(hh - f * rng) if bull else (ll + f * rng) for f in fibs]
        entries = sorted(entries, reverse=bull)
        struct_sl = (ll - float(C["BUFFER_ATR"]) * atr) if bull else (hh + float(C["BUFFER_ATR"]) * atr)

    while len(entries) < 4:
        entries.append(entries[-1])
    entries = entries[:4]

    cap = float(C["SL_CAP_PCT"]) / 100.0
    e1 = entries[0]
    cap_sl = e1 * (1.0 - cap) if bull else e1 * (1.0 + cap)
    sl = max(struct_sl, cap_sl) if bull else min(struct_sl, cap_sl)
    if bull:
        sl = min(sl, e1 * (1.0 - 0.0065))
        entries = _floor_clamp_entries(entries, sl + 0.12 * atr, atr, bull=True)
    else:
        sl = max(sl, e1 * (1.0 + 0.0065))
        entries = _floor_clamp_entries(entries, sl - 0.12 * atr, atr, bull=False)
    e1, e2, e3, e4 = entries[:4]
    ew = list(entry_weights) if entry_weights and len(entry_weights) >= 4 else list(C["ENTRY_WEIGHTS"])
    ew = [float(x) for x in ew[:4]]
    s = sum(ew) or 1.0
    eavg = sum(p * (w / s) for p, w in zip(entries, ew))
    risk = abs(e1 - sl)
    if risk <= 0:
        return None
    rpct = risk / e1 * 100.0
    ratr = risk / atr
    if not (float(C["R_MIN_PCT"]) <= rpct <= float(C["R_MAX_PCT"])):
        return None
    if not (float(C["R_MIN_ATR"]) <= ratr <= float(C["R_MAX_ATR"])):
        return None
    reach = abs(last - e1) / atr
    if reach > float(C["ALCANCE_MAX_ATR"]):
        return None
    if bull and last < e4:
        return None
    if not bull and last > e4:
        return None

    depth = float(C.get("FIB_MIN_DEPTH", 0.50))
    fib_depth_px = (hh - depth * rng) if bull else (ll + depth * rng)
    if bull:
        i0 = base + hh_i
        tagged_50 = i0 < n and min(lows[i0:]) <= fib_depth_px
    else:
        i0 = base + ll_i
        tagged_50 = i0 < n and max(highs[i0:]) >= fib_depth_px
    if C.get("HARD_FIB50", False) and not tagged_50 and not used_poi:
        return None
    at_e2_or_deeper = (last <= e2) if bull else (last >= e2)
    if C.get("HARD_E2_CHASE", False) and not at_e2_or_deeper:
        return None

    fee = float(C["COMISION_PCT"])
    drag = (2.0 * fee) / rpct if rpct else 99.0
    if drag > float(C["MAX_COMISION_R"]):
        return None
    tps = [(e1 + risk * m) if bull else (e1 - risk * m) for m in C["TP"]]

    bar_hi, bar_lo = highs[-1], lows[-1]
    bar_o, bar_c = opens[-1], closes[-1]
    if bull:
        if bar_lo <= sl:
            return None
        if bar_c < bar_o:
            return None
    else:
        if bar_hi >= sl:
            return None
        if bar_c > bar_o:
            return None
    if _wick_rejected(bar_o, bar_hi, bar_lo, bar_c, bull, float(C.get("WICK_REJECT_MAX", 0.0) or 0.0)):
        return None

    stop_aprieta = float(C.get("STOP_APRIETA", 1.0) or 1.0)
    risk_eavg = abs(eavg - sl)
    if 0 < stop_aprieta < 1.0 and risk_eavg > 0:
        sl_op = (eavg - stop_aprieta * risk_eavg) if bull else (eavg + stop_aprieta * risk_eavg)
        if bull:
            sl_op = max(sl_op, sl)
        else:
            sl_op = min(sl_op, sl)
    else:
        sl_op = sl

    score = 72.0
    score += min(8.0, vol_ratio * 3.5)
    score += min(8.0, er * 18.0)
    score += min(6.0, abs(v24) / 4.0)
    score += 6.0 if reach <= 0.55 else 2.0
    score += 4.0 if (bull and last > ema_f) or (not bull and last < ema_f) else 0.0
    score += min(5.0, max(0.0, (impulse_vol_ratio - 1.0) * 4.0))
    score += 3.0 if stretch <= 1.0 else (1.0 if stretch <= 1.6 else 0.0)
    if htf_aligned:
        score += 3.0
    if bias:
        score += 4.0
    if estruct.bos and ((bull and estruct.bos == "ALCISTA") or ((not bull) and estruct.bos == "BAJISTA")):
        score += 3.0
    if C.get("SMC_POI_SOFT", True):
        if poi_inside:
            score += 6.0
        elif poi_near:
            score += 3.5
    if C.get("SMC_PD_SOFT", True) and pd_aligned:
        score += 3.0
    if choch_ok:
        score += 3.0
    if used_poi:
        score += 2.0
    if tagged_50:
        score += 1.5
    if C.get("BTC_REGIME", False) and regime in ("bull", "bear") and (
        (bull and regime == "bull") or ((not bull) and regime == "bear")
    ):
        score += 2.0
    score -= drag * 6.0
    score = max(0.0, min(99.0, score))

    return {
        "ok": True,
        "side": side,
        "E1": e1,
        "E2": e2,
        "E3": e3,
        "E4": e4,
        "entries": entries,
        "entry_weights": ew,
        "Eavg": eavg,
        "SL": sl,
        "SL_op": sl_op,
        "TPs": tps,
        "Rpct": rpct,
        "Ratr": ratr,
        "dragComision": drag,
        "score": score,
        "last": last,
        "atr": atr,
        "atr_pct": atr_pct,
        "vol_ratio": vol_ratio,
        "change_pct": v24,
        "impulse_high": hh,
        "impulse_low": ll,
        "stretch_atr": stretch,
        "impulse_vol_ratio": impulse_vol_ratio,
        "htf_aligned": htf_aligned,
        "btc_regime": regime,
        "stop_aprieta": stop_aprieta,
        "smc_bias": bias,
        "smc_bos": estruct.bos,
        "smc_choch": estruct.choch,
        "smc_poi": (poi.tipo if poi else None),
        "smc_poi_inside": poi_inside,
        "smc_poi_near": poi_near,
        "smc_pd": pd.get("zona"),
        "smc_choch_ltf": choch_ok,
        "smc_used_poi": used_poi,
        "reasons": [
            f"SMC-lite HTF bias {bias or 'n/a'} BOS={estruct.bos}",
            f"EMA {int(C['EMA_FAST'])}/{int(C['EMA_SLOW'])} + ER {er:.2f}",
            f"{'POI '+poi.tipo if used_poi and poi else 'fib'} R {rpct:.2f}%",
            f"FVG/POI {'inside' if poi_inside else ('near' if poi_near else 'none')}",
            f"PD {pd.get('zona') or 'n/a'} · LTF CHoCH {'yes' if choch_ok else 'no'}",
            f"fee-drag {drag:.2f}R · 24h {v24:.1f}%",
            f"stretch {stretch:.2f}ATR · impulse vol {impulse_vol_ratio:.2f}x",
            f"profile {C.get('STRATEGY_PROFILE', 'smc_lite')} STOP_APRIETA={stop_aprieta}",
        ],
    }



def score_symbol(
    candles: list[dict],
    ticker: dict,
    htf_candles: list[dict] | None = None,
    *,
    require_htf: bool = False,
    entry_weights: list[float] | None = None,
    btc_regime: str | None = None,
    symbol: str | None = None,
    allow_shorts: bool | None = None,
    long_only: bool | None = None,
) -> dict | None:
    funding = ticker.get("funding") if ticker else None
    try:
        funding = None if funding is None else float(funding)
    except (TypeError, ValueError):
        funding = None
    sym = symbol or (ticker.get("symbol") if ticker else None)
    use_smc = bool(FIB_CFG.get("SMC_LITE", False)) or str(
        FIB_CFG.get("STRATEGY_PROFILE", "")
    ).lower() in ("smc_lite", "smc")
    eval_fn = evaluate_smc_lite if use_smc else evaluate_fib
    fib = eval_fn(
        candles,
        funding,
        require_gainer=True,
        htf_candles=htf_candles,
        require_htf=require_htf,
        entry_weights=entry_weights,
        btc_regime=btc_regime,
        symbol=sym,
        allow_shorts=allow_shorts,
        long_only=long_only,
    )
    if not fib:
        return None
    return {
        "side": fib["side"],
        "score": float(fib["score"]),
        "reasons": list(fib["reasons"]),
        "last": float(fib["last"]),
        "atr": float(fib["atr"]),
        "atr_pct": float(fib["atr_pct"]),
        "vol_ratio": float(fib["vol_ratio"]),
        "change_pct": float(fib["change_pct"]),
        "swing_high": float(fib["impulse_high"]),
        "swing_low": float(fib["impulse_low"]),
        "fib": fib,
        "engine": "ZENITH-SMC-LITE" if use_smc else "ZENITH",
    }


def _adaptive_tp_weights(
    weights: list[float],
    score: float,
    min_score: float = 65.0,
    max_shift_frac: float = 0.35,
) -> list[float]:
    """Shift size from TP1 toward TP2-5 as score climbs above min_score.

    At score<=min_score, weights pass through unchanged (same front-loaded
    ~80% TP1 as before). At score~95+, up to max_shift_frac of TP1's weight
    moves to the remaining targets pro-rata, so the best-quality setups
    (higher score = more confirmation) hold more size into the 0.5-2.3R
    zone instead of capping most size at ~0.2R. Total weight is unchanged,
    so this does not add risk — it only changes how winners are harvested.
    """
    if len(weights) < 2:
        return list(weights)
    w = [max(0.0, float(x)) for x in weights]
    t = max(0.0, min(1.0, (float(score) - min_score) / 25.0))
    shift = t * max_shift_frac * w[0]
    rest_sum = sum(w[1:]) or 1.0
    out = [w[0] - shift]
    for x in w[1:]:
        out.append(x + shift * (x / rest_sum))
    return out


def build_signal(
    symbol: str,
    scored: dict,
    *,
    leverage: int,
    timeframe: str,
    entry_spread_pct: float = 0.35,
    sl_pct: float = 1.8,
    tp_ladder: list | None = None,
    entry_weights: list | None = None,
    tp_weights: list | None = None,
) -> Signal:
    last = float(scored["last"])
    side = scored["side"]
    fib = scored.get("fib") or {}
    raw_e = fib.get("entries") or [fib.get("E1"), fib.get("E2"), fib.get("E3"), fib.get("E4")]
    entries = [_round_px(float(x)) for x in raw_e if x is not None][:4]
    while len(entries) < 4:
        entries.append(entries[-1] if entries else _round_px(last))
    # Publish filter SL; managed SL_op (STOP_APRIETA) rides in extras for bots/sims
    sl = _round_px(float(fib["SL"]))
    sl_op = _round_px(float(fib.get("SL_op", sl)))
    tps = [_round_px(float(x)) for x in (fib.get("TPs") or [])]
    while len(tps) < 5:
        step = abs(entries[0] - sl) * 0.6
        last_tp = tps[-1] if tps else entries[0]
        tps.append(_round_px(last_tp + step if side == "LONG" else last_tp - step))
    tps = tps[:5]
    ew = [float(x) for x in (fib.get("entry_weights") or entry_weights or FIB_CFG["ENTRY_WEIGHTS"])][:4]
    tw = [float(x) for x in (tp_weights or FIB_CFG["CIERRES"])][:5]
    if bool(FIB_CFG.get("ADAPTIVE_TP", False)) and len(tw) == 5:
        tw = _adaptive_tp_weights(tw, float(scored["score"]), float(FIB_CFG.get("MIN_SCORE", 65.0)))
    eavg = _round_px(float(fib["Eavg"]))
    be_on = bool(FIB_CFG.get("BE_AFTER_TP1", True))
    stop_aprieta = float(fib.get("stop_aprieta", FIB_CFG.get("STOP_APRIETA", 1.0)) or 1.0)
    profile = str(FIB_CFG.get("STRATEGY_PROFILE", "matrix_orient"))
    # BE-after-TP1 lock: a bit past exact breakeven (in R of the entry->SL
    # distance) so a give-back to the old entry level still nets a small
    # win instead of a scratch trade.
    #
    # "R" here must be abs(E1 - SL) -- the same unit evaluate_fib/
    # evaluate_smc_lite used to space the TP ladder (tps = e1 + risk*m) --
    # NOT abs(Eavg - SL). Eavg sits between E1 and the deeper entries and
    # is therefore always closer to SL than E1 is, so using it as the R
    # denominator understates the true 1R distance and silently inflates
    # every downstream R-multiple (RiskGuard equity/drawdown accounting
    # included) every single time a TP is reached.
    r_unit = abs(float(fib.get("E1", entries[0])) - sl)
    be_buffer_r = float(FIB_CFG.get("BE_BUFFER_R", 0.0) or 0.0)
    if side == "LONG":
        be_price = _round_px(eavg + be_buffer_r * r_unit)
    else:
        be_price = _round_px(eavg - be_buffer_r * r_unit)
    return Signal(
        symbol=symbol,
        side=side,
        leverage=leverage,
        timeframe=timeframe,
        reference=_round_px(last),
        entries=entries,
        entry_weights=ew,
        tps=tps,
        tp_weights=tw,
        sl=sl,
        score=round(float(scored["score"]), 1),
        reasons=list(scored.get("reasons") or []),
        extras={
            "setup": f"ZENITH-{profile.upper()}",
            "strategy": f"ZENITH-{profile.upper()}",
            "avg_entry": eavg,
            "risk_pct": round(float(fib["Rpct"]), 2),
            "fee_drag_r": round(float(fib["dragComision"]), 3),
            "lock_sl_after_tp1": be_price,
            "be_buffer_r": be_buffer_r,
            # True 1R price distance (abs(E1 - SL)) that the TP ladder was
            # spaced with. RiskGuard must use this, not abs(avg_entry - sl),
            # to compute R-multiples consistently with how TPs were built.
            "r_unit": round(r_unit, 10),
            "be_after_tp1": be_on,
            "stop_aprieta": stop_aprieta,
            "sl_op": sl_op,
            "leverage_mode": "confidence",
            "cancel_velas": int(FIB_CFG["CANCEL_VELAS"]),
            "close_velas": int(FIB_CFG["CLOSE_VELAS"]),
            "htf_aligned": bool(fib.get("htf_aligned")),
            "btc_regime": fib.get("btc_regime", "neutral"),
            # Diagnostic fields (already computed for scoring, not otherwise
            # surfaced) -- exposed so a backtest run can correlate them
            # against which trades hit SL_DURING_FILL vs. TP_ALL instead of
            # guessing which confirm-bar filter to tighten.
            "vol_ratio": scored.get("vol_ratio"),
            "stretch_atr": fib.get("stretch_atr"),
            "impulse_vol_ratio": fib.get("impulse_vol_ratio"),
            "smc_bias": fib.get("smc_bias"),
            "smc_bos": fib.get("smc_bos"),
            "smc_poi_inside": fib.get("smc_poi_inside"),
            "smc_poi_near": fib.get("smc_poi_near"),
            "smc_pd": fib.get("smc_pd"),
            "smc_choch_ltf": fib.get("smc_choch_ltf"),
        },
    )
