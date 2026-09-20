"""kryptic_strategy.py -- Telegram-signal adapter for the KRYPTIC engine.

engine.py's `KrypticEngine` is a persistent, websocket-driven, per-symbol
position-tracking engine (`MarketDataPipeline`/`PositionState`, with its
own crash recovery and gap-fill replay) -- a different shape from ZENITH's
(`strategy.py`) and GEM's (`gem_strategy.py`) one-shot-per-scan
`evaluate()`/`build_signal()` functions that `bot.py` already polls on a
REST cadence. Standing up the full streaming engine just to post Telegram
signals would be a large, separate undertaking -- see
`multi_strategy_manager.py`'s own "honest scope note" about ZENITH/GEM not
being wrapped into that engine interface.

This module instead reuses the same synchronous, pandas-based pieces
`KrypticEngine` itself is built from -- `EntryLadderEngine`/`RegimeFilter`/
`DirectionEngine` via `TradeLifecycleManager.open_trade()` -- directly
against the closed-candle DataFrame `bot.py` already fetches each scan,
exactly the way `strategy.score_symbol()`/`gem_strategy.evaluate()` do. So,
like ZENITH and GEM, KRYPTIC here only ever POSTS a signal to Telegram/
Cornix and hands it to the shared `RiskGuard` for its own independent
replay -- it never opens or tracks a persistent position of its own the
way `KrypticEngine` does.
"""
from __future__ import annotations

import pandas as pd

from risk_manager import PositionState, TradeLifecycleManager
from strategy import Signal

KRYPTIC_CFG: dict = {
    # KRYPTIC's own entry gate (RegimeFilter.evaluate_market_conditions +
    # DirectionEngine.get_directional_bias, both pass/fail) is binary --
    # unlike ZENITH/GEM's evaluate(), it has no continuous 0-100 quality
    # score. This fixed value only feeds leverage_from_quality's
    # score-to-leverage mapping and bot.py's cross-candidate ranking; it
    # never gates whether a setup is taken.
    "SCORE": 75.0,
    "BE_AFTER_TP1": True,
    "CANCEL_BARS": 8,
    "CLOSE_BARS": 20,
}


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
    return round(float(x), _dec(x))


def candles_to_df(candles: list[dict]) -> pd.DataFrame:
    """Adapt this repo's shared {ts,open,high,low,close,volume} kline
    shape into the OHLCV DataFrame indicators.py/regime_filter.py/
    directional_bias.py/entry_ladder.py/risk_manager.py all expect."""
    return pd.DataFrame(candles)


def evaluate(
    df: pd.DataFrame,
    btc_df: pd.DataFrame,
    funding_rate: float | None = None,
    htf_df: pd.DataFrame | None = None,
    *,
    trade_manager: TradeLifecycleManager | None = None,
) -> tuple[PositionState | None, dict]:
    """Thin pass-through to `TradeLifecycleManager.open_trade()`, kept as
    its own function so `bot.py`'s `evaluate_one_kryptic()` reads the same
    shape as `evaluate_one_gem()`/`strategy.score_symbol()`."""
    mgr = trade_manager or TradeLifecycleManager()
    return mgr.open_trade(df, btc_df, funding_rate, htf_df=htf_df)


def build_signal(
    symbol: str,
    position: PositionState,
    diagnostics: dict,
    *,
    leverage: int,
    timeframe: str,
    reference: float,
) -> Signal:
    """Convert an opened `PositionState` (4-tier `EntryLadder` + 5 fixed
    TP targets, both real weights now -- see entry_ladder.py/risk_manager.py)
    into the same `Signal` shape ZENITH/GEM post."""
    ladder = position.ladder
    entries = [_round_px(x) for x in ladder.levels]
    entry_weights = [round(w * 100, 4) for w in ladder.weights]
    tps = [_round_px(x) for x in position.tp_levels]
    tp_weights = [round(w * 100, 4) for w in position.tp_weights]
    sl = _round_px(position.initial_sl)
    r_unit = abs(ladder.levels[0] - position.initial_sl)
    return Signal(
        symbol=symbol,
        side=position.direction,
        leverage=leverage,
        timeframe=timeframe,
        reference=_round_px(reference),
        entries=entries,
        entry_weights=entry_weights,
        tps=tps,
        tp_weights=tp_weights,
        sl=sl,
        score=float(KRYPTIC_CFG["SCORE"]),
        reasons=[],
        extras={
            "engine": "KRYPTIC",
            "setup": "KRYPTIC-LADDER",
            "risk_atr_multiple": diagnostics.get("risk_atr_multiple"),
            "r_unit": round(r_unit, 10),
            "be_after_tp1": bool(KRYPTIC_CFG["BE_AFTER_TP1"]),
            "cancel_velas": int(KRYPTIC_CFG["CANCEL_BARS"]),
            "close_velas": int(KRYPTIC_CFG["CLOSE_BARS"]),
        },
    )
