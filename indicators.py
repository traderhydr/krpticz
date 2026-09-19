"""indicators.py -- vectorized, leakage-safe indicator library for KRYPTIC.

This module is deliberately independent of strategy.py / gem_strategy.py /
smc_lite.py (ZENITH and GEM's existing list-based helpers, left untouched).
It's the first building block of KRYPTIC: a fresh, pandas-vectorized
implementation built around one non-negotiable contract:

    A value computed here must never depend on data that would not yet
    exist at the moment a live bot evaluates it.

Two constraints follow directly from that contract, and every function below
is built to satisfy them:

1. Zero lookahead in pivot detection. A swing high/low at bar t is only
   knowable once `right_bars` bars have printed *after* it (you need those
   future bars' prices to know t was in fact the local extreme). This module
   never exposes a pivot's price at index t -- it is only ever attached to
   the bar at index t + right_bars, the first bar where it is actually
   confirmed. `detect_swing_pivots()` keeps the pivot's *price* (known only
   in hindsight) and its *confirmation index* (when a live bot would learn
   about it) explicitly separate for exactly this reason.

2. Bar-close semantics. Every function here assumes every row of the input
   Series/DataFrame is a fully CLOSED candle. This module has no notion of
   "the currently-forming candle" -- that filtering happens one layer up, in
   the caller's data feed (this repo already does this via
   `strategy.closed_candles()`). If a live bot's feed still includes the
   in-progress candle, the caller must drop it (or shift by one bar) before
   calling anything here; nothing in this file will do that for you, and
   doing it twice would silently throw away a real closed bar.

All functions take/return `pandas.Series`/`pandas.DataFrame` sharing the
caller's index (so results line back up with the source candles), are
vectorized end-to-end except where the calculation is *inherently*
sequential (structure/BOS/CHoCH -- see `detect_structure_breaks` for why),
and leave warmup rows as `NaN` rather than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "kaufman_efficiency_ratio",
    "true_range",
    "average_true_range",
    "ema",
    "ADXResult",
    "average_directional_index",
    "SuperTrendResult",
    "supertrend",
    "daily_anchored_vwap",
    "SwingPivots",
    "detect_swing_pivots",
    "StructureEvents",
    "detect_structure_breaks",
    "relative_volume",
    "bar_delta",
    "atr_dynamic_bands",
    "bollinger_band_width",
    "FVG_RETRACEMENT_LEVELS",
    "FIB_RETRACEMENT_LEVELS",
    "FIB_EXTENSION_LEVELS",
    "fibonacci_levels",
    "fibonacci_levels_from_swing",
    "detect_fair_value_gaps",
    "active_fvg_zones_asof",
]


def _require_same_index(*series: pd.Series) -> None:
    ref = series[0].index
    for s in series[1:]:
        if not ref.equals(s.index):
            raise ValueError("all inputs must share the same index")


# ---------------------------------------------------------------------------
# 1. Kaufman Efficiency Ratio (KER)
# ---------------------------------------------------------------------------

def kaufman_efficiency_ratio(close: pd.Series, length: int = 14) -> pd.Series:
    """Kaufman Efficiency Ratio: net displacement / total path length.

    KER = |close[t] - close[t-length]| / sum(|close[i] - close[i-1]|) over the
    trailing `length` bars. Ranges 0 (pure chop -- price round-tripped without
    net progress) to 1 (a straight, efficient trend with no backtracking).

    Args:
        close: Closed-candle close prices.
        length: Lookback window in bars (default 14).

    Returns:
        Series aligned to `close`'s index, named "KER_{length}". The first
        `length` rows are NaN (insufficient history). Where the path length
        is exactly 0 (price never moved, so displacement is also 0/0), the
        result is defined as 0.0 rather than NaN or inf, per spec.

    Raises:
        ValueError: if length < 1.
    """
    if length < 1:
        raise ValueError("length must be >= 1")

    displacement = (close - close.shift(length)).abs()
    path_length = close.diff().abs().rolling(window=length, min_periods=length).sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        ker = displacement / path_length
    ker = ker.mask(path_length == 0, 0.0)
    return ker.rename(f"KER_{length}")


# ---------------------------------------------------------------------------
# True Range / ATR (prerequisite for structure displacement + dynamic bands)
# ---------------------------------------------------------------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range: max(high-low, |high-prev_close|, |low-prev_close|).

    The first bar has no previous close, so its True Range is simply
    high[0] - low[0] (there is nothing to gap from).
    """
    _require_same_index(high, low, close)
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = ranges.max(axis=1)
    tr.iloc[0] = (high.iloc[0] - low.iloc[0])
    return tr.rename("TR")


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder-smoothed ATR (equivalent to pandas_ta's default `ta.atr`).

    Uses an exponential moving average with alpha = 1/length (Wilder's
    original smoothing), not a simple moving average.

    Returns:
        Series named "ATR_{length}"; first `length` rows are NaN.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    tr = true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return atr.rename(f"ATR_{length}")


def ema(series: pd.Series, length: int) -> pd.Series:
    """Standard exponential moving average (alpha = 2/(length+1), i.e. `span=length`).

    Distinct from the Wilder smoothing (`alpha = 1/length`) used internally
    by `average_true_range`/`average_directional_index` -- this is the
    conventional "EMA50"/"EMA200" trend-reference indicator.

    Returns:
        Series named "EMA_{length}"; first `length - 1` rows are NaN.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    return series.ewm(span=length, adjust=False, min_periods=length).mean().rename(f"EMA_{length}")


@dataclass(frozen=True)
class ADXResult:
    """Output of `average_directional_index`, all Series aligned to the input index."""

    adx: pd.Series
    plus_di: pd.Series
    minus_di: pd.Series


def average_directional_index(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> ADXResult:
    """Wilder's Average Directional Index, plus its +DI/-DI components.

    Standard formula: directional movement (+DM/-DM) and true range are each
    Wilder-smoothed (alpha=1/length) into +DI/-DI, DX = 100 * |+DI - -DI| /
    (+DI + -DI) is computed per bar, then DX itself is Wilder-smoothed into
    ADX. ADX measures trend STRENGTH only (how directional price movement
    is), not direction -- use +DI vs -DI for direction if needed.

    Returns:
        ADXResult; first ~`2*length` rows are NaN (ADX needs a smoothed DX
        series, which itself needs smoothed +DM/-DM/TR to warm up first).
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    _require_same_index(high, low, close)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    # diff() leaves the first row NaN; there is no prior bar to move from.
    plus_dm.iloc[0] = 0.0
    minus_dm.iloc[0] = 0.0

    atr = average_true_range(high, low, close, length=length)
    smoothed_plus_dm = plus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * smoothed_plus_dm / atr
        minus_di = 100.0 * smoothed_minus_dm / atr
        di_sum = plus_di + minus_di
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    dx = dx.mask(di_sum == 0, 0.0)

    adx = dx.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    return ADXResult(
        adx=adx.rename(f"ADX_{length}"),
        plus_di=plus_di.rename(f"PLUS_DI_{length}"),
        minus_di=minus_di.rename(f"MINUS_DI_{length}"),
    )


@dataclass(frozen=True)
class SuperTrendResult:
    """Output of `supertrend`, both Series aligned to the input index.

    trend_line: the active trailing band (support while bullish, resistance
        while bearish) -- NaN during ATR warmup.
    direction: int8, +1 bullish / -1 bearish / 0 undefined (warmup).
    """

    trend_line: pd.Series
    direction: pd.Series


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 10, multiplier: float = 2.0) -> SuperTrendResult:
    """Causal SuperTrend (matches the standard/Pine-Script reference formula).

    basic_upper/basic_lower = hl2 +/- multiplier*ATR(length) for each bar,
    then a "sticky" band-ratchet rule keeps each band from loosening while
    price still respects it (ratchets in the trend's favor, resets on the
    opposite side once broken), and direction flips only when the close
    actually crosses the OPPOSITE band's PRIOR value:
        - bearish -> bullish: close[i] crosses above upper_band[i-1]
        - bullish -> bearish: close[i] crosses below lower_band[i-1]

    This is a genuine recurrence relation (each bar's bands and direction
    depend on the prior bar's bands and direction, same as
    `detect_structure_breaks`), so -- deliberately, and for the same reason
    documented there -- this makes one explicit O(n) sequential pass over
    numpy arrays rather than a vectorized pandas expression.

    Returns:
        SuperTrendResult. Rows before ATR's own warmup are NaN/0
        (undefined); the very first valid row picks an arbitrary but
        deterministic initial direction (close vs. hl2) that self-corrects
        via the normal flip logic within a few bars regardless of that
        choice.

    Raises:
        ValueError: if length < 1, or inputs have mismatched indices.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    _require_same_index(high, low, close)

    atr = average_true_range(high, low, close, length=length)
    hl2 = (high + low) / 2.0
    basic_upper = (hl2 + multiplier * atr).to_numpy(dtype=float)
    basic_lower = (hl2 - multiplier * atr).to_numpy(dtype=float)
    hl2_arr = hl2.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(c)

    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    direction = np.zeros(n, dtype=np.int8)
    trend_line = np.full(n, np.nan)

    atr_valid = ~np.isnan(atr.to_numpy(dtype=float))
    if not atr_valid.any():
        idx = close.index
        return SuperTrendResult(
            trend_line=pd.Series(trend_line, index=idx, name=f"SUPERTREND_{length}_{multiplier}"),
            direction=pd.Series(direction, index=idx, name=f"SUPERTREND_DIR_{length}_{multiplier}"),
        )

    first_valid = int(np.argmax(atr_valid))
    upper_band[first_valid] = basic_upper[first_valid]
    lower_band[first_valid] = basic_lower[first_valid]
    direction[first_valid] = 1 if c[first_valid] >= hl2_arr[first_valid] else -1
    trend_line[first_valid] = lower_band[first_valid] if direction[first_valid] == 1 else upper_band[first_valid]

    for i in range(first_valid + 1, n):
        upper_band[i] = min(basic_upper[i], upper_band[i - 1]) if c[i - 1] < upper_band[i - 1] else basic_upper[i]
        lower_band[i] = max(basic_lower[i], lower_band[i - 1]) if c[i - 1] > lower_band[i - 1] else basic_lower[i]

        if direction[i - 1] == -1 and c[i] > upper_band[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and c[i] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        trend_line[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    idx = close.index
    return SuperTrendResult(
        trend_line=pd.Series(trend_line, index=idx, name=f"SUPERTREND_{length}_{multiplier}"),
        direction=pd.Series(direction, index=idx, name=f"SUPERTREND_DIR_{length}_{multiplier}"),
    )


# ---------------------------------------------------------------------------
# Daily UTC-anchored VWAP
# ---------------------------------------------------------------------------

def daily_anchored_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, timestamp: pd.Series) -> pd.Series:
    """Volume-weighted average (typical) price, resetting at each 00:00 UTC boundary.

    Uses typical price (H+L+C)/3 as the per-bar price (the standard proxy
    when only OHLCV bars, not individual trades/ticks, are available).

    Args:
        high, low, close, volume: Closed-candle price/volume Series sharing
            one index.
        timestamp: A datetime64 Series (tz-aware in any zone, or naive --
            naive is assumed to already be UTC) sharing the same index,
            used only to find each row's UTC calendar day for the reset
            grouping. Not required to be sorted, but every function in this
            module otherwise assumes chronological order, so it should be.

    Returns:
        Series named "VWAP". Each row's cumulative (price*volume)/volume
        since the most recent UTC midnight at or before that row -- a
        same-bar-safe, backward-only aggregation (row i only ever sums rows
        <= i within its own UTC day). NaN wherever the day's cumulative
        volume is exactly 0 (undefined, not "zero VWAP").

    Raises:
        TypeError: if `timestamp` is not datetime64-typed.
        ValueError: if inputs have mismatched indices.
    """
    _require_same_index(high, low, close, volume, timestamp)
    if not pd.api.types.is_datetime64_any_dtype(timestamp):
        raise TypeError("timestamp must be a datetime64 Series (tz-aware or naive-UTC)")

    ts = pd.DatetimeIndex(timestamp.to_numpy())
    ts = ts.tz_convert("UTC") if ts.tz is not None else ts.tz_localize("UTC")
    utc_day = pd.Series(ts.normalize(), index=high.index)

    typical_price = (high + low + close) / 3.0
    price_volume = typical_price * volume

    cum_pv = price_volume.groupby(utc_day).cumsum()
    cum_vol = volume.groupby(utc_day).cumsum()

    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = cum_pv / cum_vol
    vwap = vwap.mask(cum_vol == 0, np.nan)
    return vwap.rename("VWAP")


# ---------------------------------------------------------------------------
# 2a. Causal swing pivot detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwingPivots:
    """Output of `detect_swing_pivots`.

    Every Series here is aligned to the CONFIRMATION bar (index t +
    right_bars), never to the pivot bar itself (index t). A row is NaN
    unless a pivot was just confirmed on that exact bar.
    """

    pivot_high: pd.Series
    pivot_low: pd.Series
    pivot_high_source_index: pd.Series
    pivot_low_source_index: pd.Series


def _neighbor_extremes(series: pd.Series, left: int, right: int, mode: str) -> tuple[pd.Series, pd.Series]:
    """(left_extreme, right_extreme) over [t-left, t-1] and [t+1, t+right], excluding t itself.

    Excluding the center bar (rather than including it, as a single
    (left+right+1)-wide rolling max/min would) is what makes pivot detection
    strict-unique: a tie anywhere in the neighborhood correctly disqualifies
    t as a pivot instead of both bars being flagged. On a dead-flat run of
    prices, for example, every bar "ties" its neighbors, and none of them
    should be reported as a swing extreme.

    Implementation note (right side): a trailing rolling window of length
    `right` evaluated AT ROW (t+right) covers exactly [t+1, t+right].
    Shifting that trailing-rolling result backward by `right` re-aligns it
    to row t -- the same technique `detect_swing_pivots` then uses again,
    on the boolean result, to delay exposure until row t+right.
    """
    agg = "max" if mode == "max" else "min"
    left_extreme = getattr(series.rolling(window=left, min_periods=left), agg)().shift(1)
    right_trailing = getattr(series.rolling(window=right, min_periods=right), agg)()
    right_extreme = right_trailing.shift(-right)
    return left_extreme, right_extreme


def detect_swing_pivots(high: pd.Series, low: pd.Series, left_bars: int = 5, right_bars: int = 5) -> SwingPivots:
    """Causal fractal swing high/low detection.

    A bar at index t is a swing high if high[t] is the maximum high over the
    window [t-left_bars, t+right_bars] (swing low: symmetric on lows). That
    can only be known once the right-hand `right_bars` bars have closed, so
    the pivot is never attached to row t -- it's attached to row
    t + right_bars (`pivot_high`/`pivot_low`), alongside the original bar's
    index label (`pivot_high_source_index`/`pivot_low_source_index`) for
    callers that need to know when the extreme actually printed vs. when it
    became tradeable knowledge.

    Args:
        high: Closed-candle highs.
        low: Closed-candle lows.
        left_bars: Bars required on the left of the pivot (default 5).
        right_bars: Bars required on the right before confirmation (default 5).

    Returns:
        SwingPivots. All four Series share `high`'s index; the first
        `left_bars + 2*right_bars` rows are always NaN (not enough history
        to confirm anything yet).

    Raises:
        ValueError: if left_bars or right_bars < 1, or high/low indices differ.
    """
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must both be >= 1")
    _require_same_index(high, low)

    left_high_max, right_high_max = _neighbor_extremes(high, left_bars, right_bars, "max")
    left_low_min, right_low_min = _neighbor_extremes(low, left_bars, right_bars, "min")

    # Strict inequality on both sides: a tie anywhere in the neighborhood
    # disqualifies t as a pivot (see _neighbor_extremes' docstring).
    is_pivot_high_at_t = (high > left_high_max) & (high > right_high_max)
    is_pivot_low_at_t = (low < left_low_min) & (low < right_low_min)

    # Delay everything to the confirmation bar (t + right_bars) -- this is
    # the one line that actually enforces "not active until t + R".
    confirmed_high = is_pivot_high_at_t.shift(right_bars, fill_value=False)
    confirmed_low = is_pivot_low_at_t.shift(right_bars, fill_value=False)

    source_index = pd.Series(high.index, index=high.index)

    pivot_high = high.shift(right_bars).where(confirmed_high)
    pivot_low = low.shift(right_bars).where(confirmed_low)
    pivot_high_source_index = source_index.shift(right_bars).where(confirmed_high)
    pivot_low_source_index = source_index.shift(right_bars).where(confirmed_low)

    return SwingPivots(
        pivot_high=pivot_high.rename("pivot_high"),
        pivot_low=pivot_low.rename("pivot_low"),
        pivot_high_source_index=pivot_high_source_index.rename("pivot_high_source_index"),
        pivot_low_source_index=pivot_low_source_index.rename("pivot_low_source_index"),
    )


# ---------------------------------------------------------------------------
# 2b. Break of Structure / Change of Character
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructureEvents:
    """Output of `detect_structure_breaks`, all Series aligned to `close`'s index."""

    bos_up: pd.Series          # bool: bullish continuation break
    bos_down: pd.Series        # bool: bearish continuation break
    choch_up: pd.Series        # bool: bullish reversal break
    choch_down: pd.Series      # bool: bearish reversal break
    trend: pd.Series           # int8: running structural trend after this bar's close (-1/0/1)
    broken_high_level: pd.Series  # float: the confirmed swing high that was violated (NaN otherwise)
    broken_low_level: pd.Series   # float: the confirmed swing low that was violated (NaN otherwise)


def detect_structure_breaks(
    close: pd.Series,
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    *,
    atr: pd.Series | None = None,
    displacement_atr_mult: float = 0.0,
) -> StructureEvents:
    """Classify closes breaking confirmed swing levels as BOS or CHoCH.

    `pivot_high`/`pivot_low` must already be confirmation-aligned (i.e. the
    direct output of `detect_swing_pivots` -- non-NaN only on the bar where
    that level became known). Passing raw, un-delayed pivot prices here
    would silently reintroduce the lookahead bias `detect_swing_pivots` was
    built to remove.

    A close beyond the last confirmed level continues the prevailing trend
    (BOS) or reverses it (CHoCH):
      - trend == UP and close breaks above the last confirmed high -> BOS up
      - trend == DOWN and close breaks above the last confirmed high -> CHoCH up (flips trend to UP)
      - symmetric for breaks below the last confirmed low
      - the very first break of the series only establishes the initial
        trend; it is not flagged as either (there is no prior "character" to
        change, and no prior "trend" to continue).
    Once a level is broken it is consumed (cleared) so that price
    continuing to trade beyond it doesn't refire the same event every bar --
    the next event at that side waits for a fresh confirmed pivot.

    `displacement_atr_mult` (with `atr` supplied) requires the close to clear
    the level by that many ATR, filtering marginal/noise breaks; 0.0 (default)
    means any close beyond the level counts.

    Note on statefulness: classifying a break as BOS vs. CHoCH depends on the
    trend *as of the previous break*, which is itself an output of this same
    function -- a genuine recurrence relation, not something a rolling/
    cumulative pandas op can express. This function makes one explicit
    O(n) sequential pass over plain numpy arrays (not `.iloc` in a pandas
    loop) to keep that pass fast; every other computation in this module is
    fully vectorized.

    Raises:
        ValueError: if displacement_atr_mult > 0 and atr is None, or if any
            two inputs have mismatched indices.
    """
    _require_same_index(close, pivot_high, pivot_low)
    if atr is not None:
        _require_same_index(close, atr)
    if displacement_atr_mult > 0 and atr is None:
        raise ValueError("atr must be provided when displacement_atr_mult > 0")

    n = len(close)
    c = close.to_numpy(dtype=float)
    ph = pivot_high.to_numpy(dtype=float)
    pl = pivot_low.to_numpy(dtype=float)
    margin = (atr.to_numpy(dtype=float) * displacement_atr_mult) if atr is not None else np.zeros(n)

    bos_up = np.zeros(n, dtype=bool)
    bos_down = np.zeros(n, dtype=bool)
    choch_up = np.zeros(n, dtype=bool)
    choch_down = np.zeros(n, dtype=bool)
    trend_out = np.zeros(n, dtype=np.int8)
    broken_high = np.full(n, np.nan)
    broken_low = np.full(n, np.nan)

    last_high = np.nan
    last_low = np.nan
    trend = 0  # 0 = undefined, 1 = up, -1 = down

    for i in range(n):
        if not np.isnan(ph[i]):
            last_high = ph[i]
        if not np.isnan(pl[i]):
            last_low = pl[i]

        m = margin[i]

        if not np.isnan(last_high) and c[i] > last_high + m:
            if trend == -1:
                choch_up[i] = True
            elif trend == 1:
                bos_up[i] = True
            trend = 1
            broken_high[i] = last_high
            last_high = np.nan  # consumed -- needs a fresh confirmed pivot to break again

        if not np.isnan(last_low) and c[i] < last_low - m:
            if trend == 1:
                choch_down[i] = True
            elif trend == -1:
                bos_down[i] = True
            trend = -1
            broken_low[i] = last_low
            last_low = np.nan

        trend_out[i] = trend

    idx = close.index
    return StructureEvents(
        bos_up=pd.Series(bos_up, index=idx, name="bos_up"),
        bos_down=pd.Series(bos_down, index=idx, name="bos_down"),
        choch_up=pd.Series(choch_up, index=idx, name="choch_up"),
        choch_down=pd.Series(choch_down, index=idx, name="choch_down"),
        trend=pd.Series(trend_out, index=idx, name="trend"),
        broken_high_level=pd.Series(broken_high, index=idx, name="broken_high_level"),
        broken_low_level=pd.Series(broken_low, index=idx, name="broken_low_level"),
    )


# ---------------------------------------------------------------------------
# 3. Relative Volume
# ---------------------------------------------------------------------------

def relative_volume(volume: pd.Series, length: int = 20) -> pd.Series:
    """RVOL = volume / SMA(volume, length).

    Returns:
        Series named "RVOL_{length}". NaN for warmup rows and for any row
        where the trailing average volume is exactly 0 (division undefined,
        not just "low volume").
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    avg_volume = volume.rolling(window=length, min_periods=length).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rvol = volume / avg_volume
    rvol = rvol.mask(avg_volume == 0, np.nan)
    return rvol.rename(f"RVOL_{length}")


def bar_delta(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Close-Location-Value order-flow proxy: volume * (2*(close-low)/(high-low) - 1).

    Ranges from -volume (close printed at the low, proxy for all-selling) to
    +volume (close at the high, all-buying). This is a same-bar, OHLCV-only
    approximation of order flow -- there's no real buy/sell trade split
    available from candle data alone. Vectorized port of the identical
    per-bar formula already used elsewhere in this repo
    (gem_strategy.py's `bar_delta`), kept numerically identical so both
    strategies agree on what "order flow" means.

    Returns:
        Series named "BAR_DELTA". 0.0 (not NaN) on a zero-range bar (division
        undefined, but "no information" is a reasonable default for a
        doji/flat bar rather than propagating NaN).
    """
    _require_same_index(high, low, close, volume)
    rng = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        clv = 2.0 * (close - low) / rng - 1.0
    delta = (volume * clv).mask(rng <= 0, 0.0)
    return delta.rename("BAR_DELTA")


# ---------------------------------------------------------------------------
# 4a. ATR dynamic bands
# ---------------------------------------------------------------------------

def atr_dynamic_bands(close: pd.Series, atr: pd.Series, multiplier: float = 2.0) -> pd.DataFrame:
    """Symmetric ATR bands: close +/- (multiplier * atr).

    Both inputs must already reflect the same (closed) bar -- this is a
    same-bar calculation, not a forward-looking one.

    Returns:
        DataFrame with columns "upper_band", "lower_band", indexed like `close`.
    """
    _require_same_index(close, atr)
    return pd.DataFrame(
        {
            "upper_band": close + multiplier * atr,
            "lower_band": close - multiplier * atr,
        },
        index=close.index,
    )


def bollinger_band_width(close: pd.Series, length: int = 20, num_std: float = 2.0) -> pd.Series:
    """Normalized Bollinger Band width: `(upper - lower) / mid`, where
    `mid` is the `length`-bar simple moving average and `upper`/`lower`
    are `mid +/- num_std` sample standard deviations of `close` over the
    same window.

    A pure volatility-magnitude measure, not a squeeze/compression
    percentile -- callers (see regime_filter.py's `_squeeze_gate`) rank
    this series' own recent history to decide whether volatility is
    currently CONTRACTED relative to itself, which this function alone
    doesn't know.

    Returns:
        Series named "BB_WIDTH_{length}"; first `length - 1` rows are NaN.
    """
    if length < 2:
        raise ValueError("length must be >= 2")
    mid = close.rolling(length).mean()
    std = close.rolling(length).std(ddof=0)
    width = (2.0 * num_std * std) / mid
    return width.rename(f"BB_WIDTH_{length}")


# ---------------------------------------------------------------------------
# 4b. Fibonacci retracement / extension calculator
# ---------------------------------------------------------------------------

FIB_RETRACEMENT_LEVELS: tuple[float, ...] = (0.50, 0.705, 0.790, 0.886)
FIB_EXTENSION_LEVELS: tuple[float, ...] = (1.272, 1.618)
FVG_RETRACEMENT_LEVELS = FIB_RETRACEMENT_LEVELS  # backward-compat alias, same values


def fibonacci_levels(
    leg_start: float,
    leg_end: float,
    *,
    retracements: tuple[float, ...] = FIB_RETRACEMENT_LEVELS,
    extensions: tuple[float, ...] = FIB_EXTENSION_LEVELS,
) -> dict[str, dict[float, float]]:
    """Fibonacci retracement/extension price levels for one impulse leg.

    Direction-agnostic: `leg_start` is where the impulse began, `leg_end` is
    where it ended, in either direction (bullish leg: start=low, end=high;
    bearish leg: start=high, end=low). Retracement levels sit BETWEEN start
    and end (pullback zone); extension levels sit BEYOND end, continuing the
    leg's original direction.

    Args:
        leg_start: Price where the impulse leg originated.
        leg_end: Price where the impulse leg terminated.
        retracements: Fib ratios for pullback levels (default 0.50, 0.705,
            0.790, 0.886).
        extensions: Fib ratios for continuation levels beyond leg_end
            (default 1.272, 1.618).

    Returns:
        {"retracements": {ratio: price, ...}, "extensions": {ratio: price, ...}}

    Raises:
        ValueError: if leg_start == leg_end (zero-range leg).

    Example:
        >>> fibonacci_levels(100.0, 110.0)["retracements"][0.5]
        105.0
        >>> fibonacci_levels(100.0, 110.0)["extensions"][1.272]
        112.72
    """
    if leg_start == leg_end:
        raise ValueError("leg_start and leg_end must differ (zero-range leg)")
    leg_range = leg_end - leg_start
    retracement_prices = {level: leg_end - level * leg_range for level in retracements}
    extension_prices = {level: leg_start + level * leg_range for level in extensions}
    return {"retracements": retracement_prices, "extensions": extension_prices}


def fibonacci_levels_from_swing(
    swing_low: float,
    swing_high: float,
    *,
    bias: str,
    retracements: tuple[float, ...] = FIB_RETRACEMENT_LEVELS,
    extensions: tuple[float, ...] = FIB_EXTENSION_LEVELS,
) -> dict[str, dict[float, float]]:
    """Convenience wrapper over `fibonacci_levels` for the common swing-low/
    swing-high framing, picking leg direction from `bias`.

    Args:
        swing_low: The lower of the two swing prices.
        swing_high: The higher of the two swing prices.
        bias: "LONG" treats the leg as low->high (retracement levels sit
            below swing_high, for pullback buys; extensions project above
            swing_high). "SHORT" treats the leg as high->low (retracement
            levels sit above swing_low, for pullback sells; extensions
            project below swing_low).

    Raises:
        ValueError: if bias is not "LONG"/"SHORT", or swing_low >= swing_high.
    """
    if swing_low >= swing_high:
        raise ValueError("swing_low must be strictly less than swing_high")
    if bias == "LONG":
        return fibonacci_levels(swing_low, swing_high, retracements=retracements, extensions=extensions)
    if bias == "SHORT":
        return fibonacci_levels(swing_high, swing_low, retracements=retracements, extensions=extensions)
    raise ValueError('bias must be "LONG" or "SHORT"')


# ---------------------------------------------------------------------------
# 2c. Fair Value Gaps
# ---------------------------------------------------------------------------

def detect_fair_value_gaps(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Classic 3-bar Fair Value Gap detection with full mitigation history.

    Bullish FVG at bar t: low[t] > high[t-2]  -> gap = (high[t-2], low[t])
    Bearish FVG at bar t: high[t] < low[t-2]  -> gap = (low[t-2], high[t])

    Unlike the pivot/structure functions above, a gap's existence and its
    boundaries are knowable the instant bar t closes (no future bars needed)
    -- there is no lookahead issue in *detecting* one. Its **mitigation**
    status is a different matter: whether/when it later got pierced is only
    knowable as time passes. This function reports `mitigated_at` as a
    single fully-resolved fact per gap using the ENTIRE Series passed in,
    which is exactly right for building a historical catalog of gaps but
    WRONG to feed directly into a bar-by-bar backtest loop as if it were
    known at `formed_at` -- see `active_fvg_zones_asof` for the correct,
    point-in-time-safe way to query "which gaps are open as of bar i".

    Args:
        high: Closed-candle highs.
        low: Closed-candle lows.

    Returns:
        DataFrame, one row per gap, columns:
          formed_at            -- index label of bar t (the gap's third candle)
          direction             -- "bullish" | "bearish"
          upper                 -- upper boundary of the gap
          lower                 -- lower boundary of the gap
          consequent_encroachment -- exact 0.50 midpoint of the gap
          mitigated_at          -- index label of the first later bar whose
                                    wick pierced the gap, or pd.NaT if it
                                    never gets pierced within this data
          is_mitigated          -- bool, `mitigated_at` is not null
        Sorted by `formed_at`. Empty (correctly typed, zero rows) if no gaps
        are found.
    """
    _require_same_index(high, low)
    idx = high.index
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    n = len(idx)

    bullish_cond = np.zeros(n, dtype=bool)
    bearish_cond = np.zeros(n, dtype=bool)
    if n > 2:
        bullish_cond[2:] = l[2:] > h[:-2]
        bearish_cond[2:] = h[2:] < l[:-2]

    records: list[dict] = []

    for t in np.flatnonzero(bullish_cond):
        upper, lower = l[t], h[t - 2]
        pierce = np.flatnonzero(l[t + 1:] <= upper)
        mitigated_pos = t + 1 + pierce[0] if pierce.size else None
        records.append(
            {
                "formed_at": idx[t],
                "direction": "bullish",
                "upper": upper,
                "lower": lower,
                "consequent_encroachment": 0.5 * (upper + lower),
                "mitigated_at": idx[mitigated_pos] if mitigated_pos is not None else pd.NaT,
            }
        )

    for t in np.flatnonzero(bearish_cond):
        upper, lower = l[t - 2], h[t]
        pierce = np.flatnonzero(h[t + 1:] >= lower)
        mitigated_pos = t + 1 + pierce[0] if pierce.size else None
        records.append(
            {
                "formed_at": idx[t],
                "direction": "bearish",
                "upper": upper,
                "lower": lower,
                "consequent_encroachment": 0.5 * (upper + lower),
                "mitigated_at": idx[mitigated_pos] if mitigated_pos is not None else pd.NaT,
            }
        )

    columns = ["formed_at", "direction", "upper", "lower", "consequent_encroachment", "mitigated_at"]
    if not records:
        df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame.from_records(records, columns=columns)
        df = df.sort_values("formed_at").reset_index(drop=True)
    df["is_mitigated"] = df["mitigated_at"].notna()
    return df


def active_fvg_zones_asof(fvg_events: pd.DataFrame, as_of_index) -> pd.DataFrame:
    """Point-in-time-safe filter: which gaps are open exactly as of `as_of_index`.

    This is the only correct way to use `detect_fair_value_gaps`'s output
    inside a bar-by-bar loop: it returns gaps that had already formed by
    `as_of_index` AND had not yet been mitigated by `as_of_index` --
    ignoring the fact that `fvg_events` may "know" about a later mitigation
    that, from the perspective of `as_of_index`, hasn't happened yet.

    Args:
        fvg_events: Output of `detect_fair_value_gaps`.
        as_of_index: An index label present in the original candle series
            (comparisons use `<=`/`>`, so it must be orderable against
            `formed_at`/`mitigated_at`, e.g. a timestamp or integer position).

    Returns:
        Filtered copy of `fvg_events` (same columns).
    """
    formed = fvg_events["formed_at"] <= as_of_index
    not_yet_mitigated = fvg_events["mitigated_at"].isna() | (fvg_events["mitigated_at"] > as_of_index)
    return fvg_events.loc[formed & not_yet_mitigated].copy()
