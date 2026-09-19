"""Public perp tickers + klines. Binance first, then MEXC, then Bitget --
plus a BloFin-only source (see resolve_source()) for when signal prices
need to match a Cornix account that executes on BloFin."""
from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger("zenith.ex")
UA = {"User-Agent": "zenith-bot/1.0"}


async def _binance_max_leverage(client: httpx.AsyncClient) -> dict[str, int]:
    """Prefer leverageBracket; fall back empty if unavailable."""
    out: dict[str, int] = {}
    try:
        r = await client.get("https://fapi.binance.com/fapi/v1/leverageBracket", headers=UA, timeout=20)
        if r.status_code != 200:
            return out
        data = r.json()
        if not isinstance(data, list):
            return out
        for row in data:
            sym = row.get("symbol")
            brackets = row.get("brackets") or []
            if not sym or not brackets:
                continue
            # Highest allowed initial leverage is usually bracket 0
            lev = int(brackets[0].get("initialLeverage") or 0)
            if lev > 0:
                out[sym] = lev
    except Exception as e:
        log.debug("leverageBracket failed: %s", e)
    return out


async def load_universe(client: httpx.AsyncClient, min_quote_vol: float) -> dict:
    venues: dict[str, list[str]] = {}
    tickers: list[dict] = []
    max_lev: dict[str, int] = {}
    try:
        r = await client.get("https://fapi.binance.com/fapi/v1/exchangeInfo", headers=UA, timeout=20)
        r.raise_for_status()
        for s in r.json().get("symbols") or []:
            if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING":
                sym = s["symbol"]
                venues.setdefault(sym, []).append("Binance")
                lf = (s.get("leverageFilter") or {}).get("maxLeverage")
                max_lev[sym] = int(lf) if lf else 25
        brackets = await _binance_max_leverage(client)
        for sym, lev in brackets.items():
            if sym in max_lev:
                max_lev[sym] = max(int(lev), 1)
        r = await client.get("https://fapi.binance.com/fapi/v1/ticker/24hr", headers=UA, timeout=20)
        r.raise_for_status()
        for t in r.json():
            sym = t.get("symbol")
            if sym not in venues:
                continue
            qv = float(t.get("quoteVolume") or 0)
            if qv < min_quote_vol:
                continue
            tickers.append(
                {
                    "symbol": sym,
                    "last": float(t.get("lastPrice") or 0),
                    "change_pct": float(t.get("priceChangePercent") or 0),
                    "quote_vol": qv,
                }
            )
    except Exception as e:
        log.warning("binance universe failed: %s — using MEXC list", e)
        try:
            r = await client.get("https://contract.mexc.com/api/v1/contract/ticker", headers=UA, timeout=20)
            data = (r.json() or {}).get("data") or []
            if isinstance(data, dict):
                data = [data]
            for t in data:
                name = str(t.get("symbol") or "")
                if not name.endswith("_USDT"):
                    continue
                sym = name.replace("_", "")
                qv = float(t.get("amount24") or t.get("volume24") or 0)
                if qv < min_quote_vol / 10:
                    continue
                venues.setdefault(sym, []).append("MEXC")
                max_lev[sym] = 25
                tickers.append(
                    {
                        "symbol": sym,
                        "last": float(t.get("lastPrice") or 0),
                        "change_pct": float(t.get("riseFallRate") or 0) * 100,
                        "quote_vol": qv,
                    }
                )
        except Exception as e2:
            log.error("mexc universe failed: %s", e2)
    tickers.sort(key=lambda x: x.get("quote_vol") or 0, reverse=True)
    return {"tickers": tickers, "venues": venues, "max_lev": max_lev}


async def fetch_funding(client: httpx.AsyncClient, symbol: str) -> float | None:
    try:
        r = await client.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            headers=UA,
            timeout=10,
        )
        if r.status_code == 200:
            return float(r.json().get("lastFundingRate") or 0) * 100.0
    except Exception:
        pass
    return None


def _mexc_sym(symbol: str) -> str:
    return symbol[:-4] + "_USDT" if symbol.endswith("USDT") else symbol


def _rows_from_binance(raw: list) -> list[dict]:
    out = []
    for k in raw:
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
    return out


async def fetch_klines(client: httpx.AsyncClient, symbol: str, interval: str, limit: int = 240) -> tuple[list[dict], str]:
    try:
        r = await client.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            headers=UA,
            timeout=15,
        )
        if r.status_code == 200:
            rows = _rows_from_binance(r.json())
            if len(rows) >= 50:
                return rows, "Binance"
    except Exception as e:
        log.debug("binance klines %s: %s", symbol, e)

    iv = {"15m": "Min15", "1h": "Min60", "5m": "Min5", "4h": "Hour4"}.get(interval, "Min15")
    try:
        r = await client.get(
            f"https://contract.mexc.com/api/v1/contract/kline/{_mexc_sym(symbol)}",
            params={"interval": iv},
            headers=UA,
            timeout=15,
        )
        d = (r.json() or {}).get("data") or {}
        ts = d.get("time") or []
        o, h, l, c, v = d.get("open") or [], d.get("high") or [], d.get("low") or [], d.get("close") or [], d.get("vol") or []
        n = min(len(ts), len(o), len(h), len(l), len(c), len(v), limit)
        rows = []
        start = max(0, len(ts) - n)
        for i in range(start, start + n):
            t = int(ts[i])
            if t < 10_000_000_000:
                t *= 1000
            rows.append({"ts": t, "open": float(o[i]), "high": float(h[i]), "low": float(l[i]), "close": float(c[i]), "volume": float(v[i])})
        if len(rows) >= 50:
            return rows, "MEXC"
    except Exception as e:
        log.debug("mexc klines %s: %s", symbol, e)

    bg = {"15m": "15m", "1h": "1H", "5m": "5m", "4h": "4H"}.get(interval, "15m")
    try:
        r = await client.get(
            "https://api.bitget.com/api/v2/mix/market/candles",
            params={"symbol": symbol, "granularity": bg, "productType": "USDT-FUTURES", "limit": str(min(limit, 200))},
            headers=UA,
            timeout=15,
        )
        data = (r.json() or {}).get("data") or []
        rows = []
        for k in data:
            rows.append(
                {
                    "ts": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        rows.sort(key=lambda x: x["ts"])
        if len(rows) >= 50:
            return rows, "Bitget"
    except Exception as e:
        log.debug("bitget klines %s: %s", symbol, e)

    return [], "none"


# ---------------------------------------------------------------------------
# BloFin -- a single-exchange source (no cross-venue fallback) so that when
# a Cornix account executes on BloFin, every scanned symbol is confirmed
# actually listed there, and its price data matches what Cornix will fill
# at. Endpoint shapes below (response envelope {code,msg,data}, OKX-style
# instId/bar params, symbol format BTC-USDT) come from BloFin's public API
# docs via an existing hand-written adapter in this codebase's history --
# only the tickers/candles endpoints were exercised there. BloFin's API
# doesn't publish a documented funding-rate ticker field or a per-symbol
# max-leverage endpoint, so fetch_funding_blofin() always returns None
# (funding veto is a soft filter, not the core edge -- see README) and
# load_universe_blofin() leaves max_lev empty, which callers already treat
# as "don't restrict below your own configured leverage_max". Verify with
# a real request before trusting this in production, the same caveat this
# repo's other non-Binance/Bitget adapters carry.
# ---------------------------------------------------------------------------
BLOFIN_BASE = "https://openapi.blofin.com"
_BLOFIN_BAR = {"15m": "15m", "1h": "1H", "5m": "5m", "4h": "4H"}


def _to_blofin_symbol(symbol: str) -> str:
    return f"{symbol[:-4]}-USDT" if symbol.endswith("USDT") else symbol


def _from_blofin_symbol(inst_id: str) -> str:
    return inst_id.replace("-", "")


async def load_universe_blofin(client: httpx.AsyncClient, min_quote_vol: float) -> dict:
    tickers: list[dict] = []
    venues: dict[str, list[str]] = {}
    try:
        r = await client.get(f"{BLOFIN_BASE}/api/v1/market/tickers", headers=UA, timeout=20)
        r.raise_for_status()
        body = r.json() or {}
        if str(body.get("code")) not in ("0", "00000"):
            raise RuntimeError(body.get("msg") or f"code {body.get('code')}")
        for t in body.get("data") or []:
            inst_id = str(t.get("instId") or "")
            if not inst_id.endswith("-USDT"):
                continue
            last = float(t.get("last") or 0)
            open24h = float(t.get("open24h") or 0)
            qv = float(t.get("volCurrency24h") or t.get("vol24h") or 0)
            if last <= 0 or qv < min_quote_vol:
                continue
            sym = _from_blofin_symbol(inst_id)
            venues[sym] = ["BloFin"]
            tickers.append(
                {
                    "symbol": sym,
                    "last": last,
                    "change_pct": (last / open24h - 1) * 100 if open24h > 0 else 0.0,
                    "quote_vol": qv,
                }
            )
    except Exception as e:
        log.error("blofin universe failed: %s", e)
    tickers.sort(key=lambda x: x.get("quote_vol") or 0, reverse=True)
    return {"tickers": tickers, "venues": venues, "max_lev": {}}


async def fetch_funding_blofin(client: httpx.AsyncClient, symbol: str) -> float | None:
    return None


async def fetch_klines_blofin(client: httpx.AsyncClient, symbol: str, interval: str, limit: int = 240) -> tuple[list[dict], str]:
    inst_id = _to_blofin_symbol(symbol)
    bar = _BLOFIN_BAR.get(interval, "15m")
    try:
        r = await client.get(
            f"{BLOFIN_BASE}/api/v1/market/candles",
            params={"instId": inst_id, "bar": bar, "limit": str(min(limit, 200))},
            headers=UA,
            timeout=15,
        )
        body = r.json() or {}
        if str(body.get("code")) not in ("0", "00000"):
            raise RuntimeError(body.get("msg") or f"code {body.get('code')}")
        rows = []
        for k in body.get("data") or []:
            rows.append(
                {
                    "ts": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        rows.sort(key=lambda x: x["ts"])
        if len(rows) >= 50:
            return rows, "BloFin"
    except Exception as e:
        log.debug("blofin klines %s: %s", symbol, e)

    return [], "none"


def resolve_source(name: str | None = None):
    """Pick the (load_universe, fetch_klines, fetch_funding) triple the bot
    and RiskGuard use. DATA_SOURCE=blofin (default): BloFin only, so every
    scanned symbol and price is confirmed tradeable there -- matches a
    Cornix account executing on BloFin. DATA_SOURCE=binance: the original
    Binance -> MEXC -> Bitget fallback chain, for a Cornix account executing
    somewhere else (or wider universe coverage when exact-venue matching
    doesn't matter)."""
    name = (name if name is not None else os.getenv("DATA_SOURCE", "blofin")).strip().lower()
    if name == "binance":
        return load_universe, fetch_klines, fetch_funding
    if name != "blofin":
        log.warning("unknown DATA_SOURCE=%r; using blofin", name)
    return load_universe_blofin, fetch_klines_blofin, fetch_funding_blofin
