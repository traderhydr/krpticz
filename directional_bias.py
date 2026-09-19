"""directional_bias.py -- directional bias engine for KRYPTIC (Step 3).

`DirectionEngine` answers "which way, if any, does the confluence of trend
indicators point right now?" It composes four independently-computed
signals -- EMA stack, session VWAP, SuperTrend, and HTF structure -- and
only calls a direction when ALL FOUR agree; any disagreement (or any signal
that can't yet be computed) reports NEUTRAL rather than guessing.

Like indicators.py and regime_filter.py, every input DataFrame is assumed
to contain only fully CLOSED candles, and every comparison uses the last
row (i.e. "Price" means "the last closed candle's close").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

import indicators as ind

Bias = Literal["LONG", "SHORT", "NEUTRAL"]


@dataclass(frozen=True)
class ConditionResult:
    passed_long: bool
    passed_short: bool
    detail: str
    values: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_utc_timestamps(df: pd.DataFrame) -> pd.Series:
    """Get a datetime64[UTC] Series aligned to df's index, from either a
    'ts' column (epoch milliseconds, this repo's existing candle
    convention -- see exchanges.py/smc_lite.py) or a DatetimeIndex."""
    if "ts" in df.columns:
        return pd.to_datetime(df["ts"], unit="ms", utc=True)
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        return pd.Series(idx, index=df.index)
    raise ValueError("df must have a 'ts' column (epoch ms, UTC) or a DatetimeIndex to anchor VWAP/HTF aggregation")


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}")


class DirectionEngine:
    """Directional bias from EMA stack + session VWAP + SuperTrend + HTF structure.

    Args:
        ema_fast_length: Fast EMA length for the stack check (default 20).
        ema_slow_length: Slow EMA length for the stack check (default 50).
        supertrend_length: SuperTrend ATR length (default 10).
        supertrend_multiplier: SuperTrend ATR multiplier (default 2.0).
        htf_aggregate_factor: When `htf_df` is not supplied, how many `df`
            bars to aggregate into one HTF bar (default 4, matching this
            repo's existing smc_lite.aggregate_htf_from_ltf convention).
        htf_swing_left / htf_swing_right: Swing pivot window for HTF
            structure detection (default 2/2, matching this repo's existing
            SMC_SWING_HTF convention in smc_lite.py).
        htf_displacement_atr_mult: Optional ATR-multiple displacement filter
            for HTF BOS/CHoCH classification (default 0.0 = any break counts).
        htf_atr_length: ATR length used only for the displacement filter
            above (default 14; irrelevant if htf_displacement_atr_mult=0).
    """

    def __init__(
        self,
        *,
        ema_fast_length: int = 20,
        ema_slow_length: int = 50,
        supertrend_length: int = 10,
        supertrend_multiplier: float = 2.0,
        htf_aggregate_factor: int = 4,
        htf_swing_left: int = 2,
        htf_swing_right: int = 2,
        htf_displacement_atr_mult: float = 0.0,
        htf_atr_length: int = 14,
    ) -> None:
        self.ema_fast_length = ema_fast_length
        self.ema_slow_length = ema_slow_length
        self.supertrend_length = supertrend_length
        self.supertrend_multiplier = supertrend_multiplier
        self.htf_aggregate_factor = htf_aggregate_factor
        self.htf_swing_left = htf_swing_left
        self.htf_swing_right = htf_swing_right
        self.htf_displacement_atr_mult = htf_displacement_atr_mult
        self.htf_atr_length = htf_atr_length

    # -- internal helpers --------------------------------------------------

    def _aggregate_htf(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Aggregate every `htf_aggregate_factor` rows of `df` into one HTF
        bar (oldest-first grouping, matching smc_lite.aggregate_htf_from_ltf).
        Returns None if there isn't at least one full group."""
        n = len(df)
        factor = self.htf_aggregate_factor
        n_groups = n // factor
        if n_groups < 1:
            return None
        # Keep only the trailing n_groups*factor rows so every group is full
        # size, grouping from the END so the most recent HTF bar is complete.
        trimmed = df.iloc[n - n_groups * factor :]
        group_id = np.arange(len(trimmed)) // factor
        agg = trimmed.groupby(group_id).agg(
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        return agg.reset_index(drop=True)

    def _htf_structure_condition(self, df: pd.DataFrame, htf_df: pd.DataFrame | None) -> ConditionResult:
        if htf_df is None:
            htf_df = self._aggregate_htf(df)
            source = f"aggregated from df (factor={self.htf_aggregate_factor})"
        else:
            source = "caller-supplied htf_df"

        min_bars = self.htf_swing_left + 2 * self.htf_swing_right + 2
        if htf_df is None or len(htf_df) < min_bars:
            have = 0 if htf_df is None else len(htf_df)
            return ConditionResult(
                passed_long=False, passed_short=False,
                detail=f"insufficient HTF history ({source}): need >= {min_bars} bars, have {have}",
                values={"htf_trend": None},
            )
        _require_columns(htf_df, ["high", "low", "close"], "htf_df")

        htf_high, htf_low, htf_close = htf_df["high"], htf_df["low"], htf_df["close"]
        pivots = ind.detect_swing_pivots(htf_high, htf_low, left_bars=self.htf_swing_left, right_bars=self.htf_swing_right)
        atr = None
        if self.htf_displacement_atr_mult > 0:
            atr = ind.average_true_range(htf_high, htf_low, htf_close, length=self.htf_atr_length)
        events = ind.detect_structure_breaks(
            htf_close, pivots.pivot_high, pivots.pivot_low,
            atr=atr, displacement_atr_mult=self.htf_displacement_atr_mult,
        )
        htf_trend = int(events.trend.iloc[-1])
        detail = (
            f"HTF structure ({source}) trend = "
            f"{'bullish (confirmed BOS/CHoCH up)' if htf_trend == 1 else 'bearish (confirmed BOS/CHoCH down)' if htf_trend == -1 else 'undefined (no confirmed break yet)'}"
        )
        return ConditionResult(
            passed_long=htf_trend == 1, passed_short=htf_trend == -1,
            detail=detail, values={"htf_trend": htf_trend},
        )

    def _ema_stack_condition(self, close: pd.Series) -> ConditionResult:
        ema_fast = ind.ema(close, self.ema_fast_length)
        ema_slow = ind.ema(close, self.ema_slow_length)
        last_close = float(close.iloc[-1])
        last_fast, last_slow = ema_fast.iloc[-1], ema_slow.iloc[-1]
        if pd.isna(last_fast) or pd.isna(last_slow):
            return ConditionResult(
                passed_long=False, passed_short=False,
                detail=f"EMA stack not yet computable (need >= {self.ema_slow_length} bars)",
                values={"close": last_close, "ema_fast": None, "ema_slow": None},
            )
        last_fast, last_slow = float(last_fast), float(last_slow)
        golden = last_close > last_fast > last_slow
        death = last_close < last_fast < last_slow
        detail = (
            f"close={last_close:.4f} EMA{self.ema_fast_length}={last_fast:.4f} EMA{self.ema_slow_length}={last_slow:.4f} -- "
            f"{'golden stack (bullish)' if golden else 'death stack (bearish)' if death else 'no clean stack'}"
        )
        return ConditionResult(
            passed_long=golden, passed_short=death, detail=detail,
            values={"close": last_close, "ema_fast": last_fast, "ema_slow": last_slow},
        )

    def _vwap_condition(self, df: pd.DataFrame) -> ConditionResult:
        timestamp = _extract_utc_timestamps(df)
        vwap = ind.daily_anchored_vwap(df["high"], df["low"], df["close"], df["volume"], timestamp)
        last_close = float(df["close"].iloc[-1])
        last_vwap = vwap.iloc[-1]
        if pd.isna(last_vwap):
            return ConditionResult(
                passed_long=False, passed_short=False,
                detail="session VWAP not yet computable (zero cumulative volume today)",
                values={"close": last_close, "vwap": None},
            )
        last_vwap = float(last_vwap)
        above = last_close > last_vwap
        below = last_close < last_vwap
        detail = f"close={last_close:.4f} vs session VWAP={last_vwap:.4f} -- {'above' if above else 'below' if below else 'at'} VWAP"
        return ConditionResult(
            passed_long=above, passed_short=below, detail=detail,
            values={"close": last_close, "vwap": last_vwap},
        )

    def _supertrend_condition(self, df: pd.DataFrame) -> ConditionResult:
        st = ind.supertrend(df["high"], df["low"], df["close"], length=self.supertrend_length, multiplier=self.supertrend_multiplier)
        last_dir = int(st.direction.iloc[-1])
        last_line = st.trend_line.iloc[-1]
        if last_dir == 0:
            return ConditionResult(
                passed_long=False, passed_short=False,
                detail=f"SuperTrend({self.supertrend_length}, {self.supertrend_multiplier}) not yet computable (warmup)",
                values={"direction": 0, "trend_line": None},
            )
        detail = f"SuperTrend({self.supertrend_length}, {self.supertrend_multiplier}) direction = {'bullish (+1)' if last_dir == 1 else 'bearish (-1)'}, line={float(last_line):.4f}"
        return ConditionResult(
            passed_long=last_dir == 1, passed_short=last_dir == -1,
            detail=detail, values={"direction": last_dir, "trend_line": float(last_line)},
        )

    # -- public API ----------------------------------------------------------

    def get_directional_bias(self, df: pd.DataFrame, htf_df: pd.DataFrame | None = None) -> tuple[Bias, dict]:
        """Classify the current directional bias.

        Args:
            df: Working-timeframe closed-candle OHLCV DataFrame (needs
                "open"[optional], "high", "low", "close", "volume", and
                either a "ts" epoch-ms column or a DatetimeIndex -- required
                for the VWAP day-anchor and, if `htf_df` is not supplied,
                for inferring HTF aggregation order).
            htf_df: Optional pre-built higher-timeframe closed-candle OHLC
                DataFrame (needs "high", "low", "close"; any index). If
                omitted, `df` is aggregated internally every
                `htf_aggregate_factor` bars.

        Returns:
            (bias, diagnostics) where `bias` is "LONG"/"SHORT"/"NEUTRAL" and
            `diagnostics` is a plain dict:
                {
                    "bias": "LONG" | "SHORT" | "NEUTRAL",
                    "conditions": {
                        "ema_stack": {"passed_long":..., "passed_short":..., "detail":..., "values": {...}},
                        "vwap": {...},
                        "supertrend": {...},
                        "htf_structure": {...},
                    },
                    "long_conditions_met": int,   # 0-4, out of 4
                    "short_conditions_met": int,  # 0-4, out of 4
                }
            LONG requires all four conditions' `passed_long`; SHORT requires
            all four `passed_short`; anything else (including a condition
            that couldn't be computed at all, which reports both as False)
            is NEUTRAL.

        Raises:
            ValueError: if required columns/timestamp info are missing.
        """
        _require_columns(df, ["high", "low", "close", "volume"], "df")

        conditions = {
            "ema_stack": self._ema_stack_condition(df["close"]),
            "vwap": self._vwap_condition(df),
            "supertrend": self._supertrend_condition(df),
            "htf_structure": self._htf_structure_condition(df, htf_df),
        }

        long_met = sum(1 for c in conditions.values() if c.passed_long)
        short_met = sum(1 for c in conditions.values() if c.passed_short)

        if long_met == len(conditions):
            bias: Bias = "LONG"
        elif short_met == len(conditions):
            bias = "SHORT"
        else:
            bias = "NEUTRAL"

        diagnostics = {
            "bias": bias,
            "conditions": {name: c.to_dict() for name, c in conditions.items()},
            "long_conditions_met": long_met,
            "short_conditions_met": short_met,
        }
        return bias, diagnostics
