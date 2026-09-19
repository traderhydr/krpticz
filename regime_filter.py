"""regime_filter.py -- pre-flight market regime & condition gate for KRYPTIC.

`RegimeFilter` answers one question: "does the system have permission to
trade right now?" It makes no entry/exit decisions of its own and knows
nothing about breakouts, retracements, or structure -- it's a permission
check meant to run BEFORE a strategy-specific entry trigger is even
evaluated. All seven gates below are hard: if any one fails, trading is not
allowed.

Like indicators.py, this module assumes every row of every input DataFrame
is a fully CLOSED candle (see indicators.py's module docstring for why) --
it always reads the LAST row of each Series it computes, on the assumption
that "last row" means "the most recently closed bar", not an in-progress one.

A gate that cannot be computed (insufficient history, no funding data,
degenerate zero-range series) fails SAFE: it is reported as not-passed with
a clear reason, rather than raising or silently skipping. The one exception
is funding: real markets frequently have no funding data available (e.g. a
newly-listed perp), and GEM's own convention (gem_strategy.py) is that a
missing funding rate simply means the veto isn't evaluated, not that the
trade is blocked -- this module follows the same convention for consistency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

import indicators as ind

Direction = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class GateResult:
    passed: bool
    value: float | None
    threshold: float | None
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required column(s): {missing}")


class RegimeFilter:
    """Pre-flight permission gate: macro trend, chop, trend strength, liquidity, exhaustion/stretch, squeeze/compression, crowding.

    Args:
        btc_ema_length: EMA length for the macro beta gate (default 50,
            evaluated on `btc_df`'s own timeframe -- pass in 4h candles if
            that's the intended granularity; this class does not resample).
        ker_length: Kaufman Efficiency Ratio lookback (default 14).
        ker_min: Minimum KER to consider the market non-choppy (default 0.36
            -- Step 10.6 re-tightening; was 0.32 (itself down from 0.38) --
            the looser 0.32 was letting too much consolidation chop through
            as "trending", per the negative-expectancy backtest it produced).
        adx_length: ADX lookback (default 14).
        adx_min: Minimum ADX to consider the trend strong enough on its own
            (default 20.0 -- Step 10.6, was 18.0, itself down from 22.0).
            The gate also passes on a weaker but RISING trend: ADX >=
            `adx_slope_min_threshold` (default 16.0) AND current ADX >
            prior ADX + `adx_slope_min_delta` (default 1.5) -- catching a
            trend early, right as it's building strength, rather than
            waiting for it to already clear the higher bar. See
            `_trend_strength_gate`.
        adx_slope_min_threshold: Floor for the slope-based fallback path
            above (default 16.0, Step 10.6; was 15.0).
        adx_slope_min_delta: Minimum ADX increase (current - prior bar) the
            slope-based fallback path requires (default 1.5, Step 10.6;
            was 1.0 -- the shallower delta was letting a trend "count" as
            rising on noise, not a real acceleration).
        rvol_length: Relative volume SMA lookback (default 20).
        rvol_min: Minimum RVOL to consider liquidity/participation adequate
            (default 1.10 -- was 1.20).
        funding_veto_pct: Funding rate magnitude (in %, same units as
            `funding_rate`) beyond which the crowd is considered
            over-leveraged in that direction (default 0.10, matching
            gem_strategy.py's FUNDING_VETO convention of %/8h -- pass
            whatever period your funding_rate is already expressed in, this
            class does not convert between periods).
        stretch_atr_length: ATR lookback for the exhaustion/stretch gate
            below (default 14, matching every other ATR14 use in this
            codebase).
        stretch_ema_length: Baseline EMA length the stretch gate measures
            distance from (default 50 -- the same "ema50" DirectionEngine's
            EMA stack condition and PositionState's TP5 runner check both
            already use, so "baseline EMA" means one consistent line
            everywhere in KRYPTIC, not a fourth EMA convention).
        stretch_atr_max: Maximum allowed |close - baseline EMA| / ATR14
            (default 3.5, Step 10.10 -- was 2.0, Step 10.8). The Step 10.8
            180-day BTC/ETH/SOL backtest showed entries past a 2.0 ATR
            stretch winning only 45.5% of the time vs. the overall 54.8%,
            but the SAME gate, re-run on real data, also collapsed trade
            volume from 704 to 35 (95%) -- a 2.0 ATR breakout is common,
            not exceptional, on 15m crypto candles, so the gate was
            rejecting healthy momentum along with genuine exhaustion.
            3.5 is a looser compromise restoring candidate volume toward
            the original baseline while still catching the most extreme
            already-blown-out entries.
        squeeze_bb_length / squeeze_bb_std: Bollinger Band parameters
            (defaults 20 / 2.0, the conventional settings) for the Step
            11.0 volatility squeeze/compression gate below.
        squeeze_lookback: How many trailing bars of BB-width history the
            squeeze gate ranks the current bar's width against (default
            100).
        squeeze_max_percentile: The current BB width must sit at or below
            this percentile of its own trailing `squeeze_lookback`-bar
            history to pass (default 0.85 -- deliberately LOOSE: reject
            only the most-expanded ~15% of readings, not "require a real
            compression"). This default is a cautious starting point, not
            a validated one -- checking this repo's own synthetic
            cross-check fixture (test_backtest_engine.py's `make_candles`)
            found that EVERY bar clearing the trend-strength/chop gates
            (ADX/KER) already had a squeeze percentile of 0.59+, most
            0.85+: by the time a trend is confirmed enough to pass those
            gates, the 20-bar BB width has typically already caught up
            with the move, so a tight compression requirement checked on
            the SAME bar as trend confirmation is structurally in tension
            with the gates around it. This is the exact same failure mode
            `stretch_atr_max` hit in Step 10.8 (2.0 ATR collapsed real
            candidate volume 704 -> 35 over a real 180-day backtest,
            corrected to a looser 3.5 in Step 10.10) -- treat this default
            as equally unvalidated until it's been checked the same way:
            a real backtest, watching total_trades before and after.
    """

    def __init__(
        self,
        *,
        btc_ema_length: int = 50,
        ker_length: int = 14,
        ker_min: float = 0.36,
        adx_length: int = 14,
        adx_min: float = 20.0,
        adx_slope_min_threshold: float = 16.0,
        adx_slope_min_delta: float = 1.5,
        rvol_length: int = 20,
        rvol_min: float = 1.10,
        funding_veto_pct: float = 0.10,
        stretch_atr_length: int = 14,
        stretch_ema_length: int = 50,
        stretch_atr_max: float = 3.5,
        squeeze_bb_length: int = 20,
        squeeze_bb_std: float = 2.0,
        squeeze_lookback: int = 100,
        squeeze_max_percentile: float = 0.85,
    ) -> None:
        self.btc_ema_length = btc_ema_length
        self.ker_length = ker_length
        self.ker_min = ker_min
        self.adx_length = adx_length
        self.adx_min = adx_min
        self.adx_slope_min_threshold = adx_slope_min_threshold
        self.adx_slope_min_delta = adx_slope_min_delta
        self.rvol_length = rvol_length
        self.rvol_min = rvol_min
        self.funding_veto_pct = funding_veto_pct
        self.stretch_atr_length = stretch_atr_length
        self.stretch_ema_length = stretch_ema_length
        self.stretch_atr_max = stretch_atr_max
        self.squeeze_bb_length = squeeze_bb_length
        self.squeeze_bb_std = squeeze_bb_std
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_max_percentile = squeeze_max_percentile

    # -- gates -----------------------------------------------------------

    def _macro_beta_gate(self, btc_df: pd.DataFrame, direction: Direction) -> GateResult:
        _require_columns(btc_df, ["close"], "btc_df")
        if len(btc_df) < self.btc_ema_length:
            return GateResult(
                passed=False, value=None, threshold=None,
                detail=f"insufficient BTC history: need >= {self.btc_ema_length} bars, have {len(btc_df)}",
            )
        btc_ema = ind.ema(btc_df["close"], self.btc_ema_length)
        last_close = float(btc_df["close"].iloc[-1])
        last_ema = btc_ema.iloc[-1]
        if pd.isna(last_ema):
            return GateResult(
                passed=False, value=last_close, threshold=None,
                detail="BTC EMA not yet computable (warmup)",
            )
        last_ema = float(last_ema)
        if direction == "LONG":
            passed = last_close > last_ema
            detail = (
                f"BTC close {last_close:.4f} is {'above' if passed else 'not above'} "
                f"its {self.btc_ema_length}-bar EMA {last_ema:.4f} (required: above, for LONG)"
            )
        else:
            passed = last_close < last_ema
            detail = (
                f"BTC close {last_close:.4f} is {'below' if passed else 'not below'} "
                f"its {self.btc_ema_length}-bar EMA {last_ema:.4f} (required: below, for SHORT)"
            )
        return GateResult(passed=passed, value=last_close, threshold=last_ema, detail=detail)

    def _chop_gate(self, df: pd.DataFrame) -> GateResult:
        _require_columns(df, ["close"], "df")
        ker = ind.kaufman_efficiency_ratio(df["close"], length=self.ker_length)
        last_ker = ker.iloc[-1] if len(ker) else float("nan")
        if pd.isna(last_ker):
            return GateResult(
                passed=False, value=None, threshold=self.ker_min,
                detail=f"KER({self.ker_length}) not yet computable (need >= {self.ker_length + 1} bars)",
            )
        last_ker = float(last_ker)
        passed = last_ker >= self.ker_min
        detail = (
            f"KER({self.ker_length}) = {last_ker:.3f} "
            f"{'>=' if passed else '<'} {self.ker_min} -- market is {'trending' if passed else 'choppy'}"
        )
        return GateResult(passed=passed, value=last_ker, threshold=self.ker_min, detail=detail)

    def _trend_strength_gate(self, df: pd.DataFrame) -> GateResult:
        """Passes if ADX already clears `adx_min` outright, OR -- catching a
        trend early, while it's still building rather than after it's
        already strong -- if ADX clears the lower `adx_slope_min_threshold`
        AND is rising fast enough (current > prior + `adx_slope_min_delta`).
        The slope check needs the prior bar's ADX too, so it additionally
        fails safe (not just on ADX itself being unavailable) when there
        isn't yet a bar before the last one to compare against.
        """
        _require_columns(df, ["high", "low", "close"], "df")
        adx_result = ind.average_directional_index(df["high"], df["low"], df["close"], length=self.adx_length)
        last_adx = adx_result.adx.iloc[-1] if len(adx_result.adx) else float("nan")
        if pd.isna(last_adx):
            return GateResult(
                passed=False, value=None, threshold=self.adx_min,
                detail=f"ADX({self.adx_length}) not yet computable (need >= ~{2 * self.adx_length} bars)",
            )
        last_adx = float(last_adx)

        if last_adx >= self.adx_min:
            return GateResult(
                passed=True, value=last_adx, threshold=self.adx_min,
                detail=f"ADX({self.adx_length}) = {last_adx:.2f} >= {self.adx_min} -- trend is strong enough",
            )

        prior_adx = adx_result.adx.iloc[-2] if len(adx_result.adx) >= 2 else float("nan")
        if last_adx >= self.adx_slope_min_threshold and not pd.isna(prior_adx):
            slope = last_adx - float(prior_adx)
            if slope > self.adx_slope_min_delta:
                return GateResult(
                    passed=True, value=last_adx, threshold=self.adx_min,
                    detail=(
                        f"ADX({self.adx_length}) = {last_adx:.2f} < {self.adx_min}, but >= {self.adx_slope_min_threshold} "
                        f"and rising fast ({float(prior_adx):.2f} -> {last_adx:.2f}, +{slope:.2f} > {self.adx_slope_min_delta}) "
                        f"-- trend strength gate passes on slope"
                    ),
                )

        detail = (
            f"ADX({self.adx_length}) = {last_adx:.2f} < {self.adx_min}, and either below the "
            f"{self.adx_slope_min_threshold} slope-fallback floor or not rising fast enough -- trend is too weak"
        )
        return GateResult(passed=False, value=last_adx, threshold=self.adx_min, detail=detail)

    def _liquidity_volume_gate(self, df: pd.DataFrame) -> GateResult:
        _require_columns(df, ["volume"], "df")
        rvol = ind.relative_volume(df["volume"], length=self.rvol_length)
        last_rvol = rvol.iloc[-1] if len(rvol) else float("nan")
        if pd.isna(last_rvol):
            return GateResult(
                passed=False, value=None, threshold=self.rvol_min,
                detail=f"RVOL({self.rvol_length}) not yet computable (insufficient history or zero average volume)",
            )
        last_rvol = float(last_rvol)
        passed = last_rvol >= self.rvol_min
        detail = (
            f"RVOL({self.rvol_length}) = {last_rvol:.2f} "
            f"{'>=' if passed else '<'} {self.rvol_min} -- participation is {'adequate' if passed else 'too thin'}"
        )
        return GateResult(passed=passed, value=last_rvol, threshold=self.rvol_min, detail=detail)

    def _stretch_gate(self, df: pd.DataFrame) -> GateResult:
        """Exhaustion gate (Step 10.8): reject a candidate whose last close
        already sits more than `stretch_atr_max` ATRs away from its own
        baseline EMA -- a breakout that has already expanded that far is
        exhaustion, not ignition, and the 180-day backtest showed exactly
        that: entries past this threshold won only 45.5% of the time vs.
        the overall 54.8%. Direction-agnostic (stretch is a magnitude, not
        a sign) since an overextended move is a bad entry either way."""
        _require_columns(df, ["high", "low", "close"], "df")
        atr = ind.average_true_range(df["high"], df["low"], df["close"], length=self.stretch_atr_length)
        ema = ind.ema(df["close"], self.stretch_ema_length)
        last_atr = atr.iloc[-1] if len(atr) else float("nan")
        last_ema = ema.iloc[-1] if len(ema) else float("nan")
        if pd.isna(last_atr) or last_atr <= 0 or pd.isna(last_ema):
            return GateResult(
                passed=False, value=None, threshold=self.stretch_atr_max,
                detail=f"ATR({self.stretch_atr_length})/EMA({self.stretch_ema_length}) not yet computable (warmup) -- stretch gate fails safe",
            )
        last_close = float(df["close"].iloc[-1])
        stretch = abs(last_close - float(last_ema)) / float(last_atr)
        passed = stretch <= self.stretch_atr_max
        detail = (
            f"stretch = |close - EMA{self.stretch_ema_length}| / ATR({self.stretch_atr_length}) = {stretch:.2f} "
            f"{'<=' if passed else '>'} {self.stretch_atr_max} -- entry is {'not' if passed else 'already'} overextended"
        )
        return GateResult(passed=passed, value=stretch, threshold=self.stretch_atr_max, detail=detail)

    def _squeeze_gate(self, df: pd.DataFrame) -> GateResult:
        """Volatility squeeze/compression gate (Step 11.0): requires the
        current Bollinger Band width to sit at or below
        `squeeze_max_percentile` of its own trailing `squeeze_lookback`-bar
        history -- i.e. volatility has CONTRACTED relative to itself
        recently, the classic pre-expansion "squeeze," before a breakout
        is allowed to qualify. This is a magnitude-of-volatility check
        relative to the market's OWN recent regime, not an absolute
        threshold -- a quiet altcoin and a busy major each get judged
        against their own trailing history, not each other's."""
        _require_columns(df, ["close"], "df")
        width = ind.bollinger_band_width(df["close"], length=self.squeeze_bb_length, num_std=self.squeeze_bb_std)
        min_bars = self.squeeze_bb_length + self.squeeze_lookback
        if len(width) < min_bars:
            return GateResult(
                passed=False, value=None, threshold=self.squeeze_max_percentile,
                detail=f"squeeze lookback not yet computable (need >= {min_bars} bars, have {len(width)})",
            )
        last_width = width.iloc[-1]
        window = width.iloc[-self.squeeze_lookback:]
        if pd.isna(last_width) or window.isna().any():
            return GateResult(
                passed=False, value=None, threshold=self.squeeze_max_percentile,
                detail=f"BB width({self.squeeze_bb_length}) not yet computable over the full lookback window (warmup)",
            )
        last_width = float(last_width)
        percentile = float((window <= last_width).mean())
        passed = percentile <= self.squeeze_max_percentile
        detail = (
            f"BB width percentile over trailing {self.squeeze_lookback} bars = {percentile:.2f} "
            f"{'<=' if passed else '>'} {self.squeeze_max_percentile} -- volatility is "
            f"{'compressed enough for a pre-ignition squeeze' if passed else 'not compressed -- already expanding/expanded'}"
        )
        return GateResult(passed=passed, value=percentile, threshold=self.squeeze_max_percentile, detail=detail)

    def _crowd_sentiment_gate(self, funding_rate: float | None, direction: Direction) -> GateResult:
        if funding_rate is None:
            return GateResult(
                passed=True, value=None, threshold=self.funding_veto_pct,
                detail="no funding rate supplied -- gate not evaluated (pass-through, matches GEM's convention)",
            )
        funding_rate = float(funding_rate)
        if direction == "LONG":
            passed = funding_rate <= self.funding_veto_pct
            detail = (
                f"funding {funding_rate:.4f}% {'<=' if passed else '>'} +{self.funding_veto_pct}% -- "
                f"crowd is {'not' if passed else 'already'} over-leveraged long"
            )
        else:
            passed = funding_rate >= -self.funding_veto_pct
            detail = (
                f"funding {funding_rate:.4f}% {'>=' if passed else '<'} -{self.funding_veto_pct}% -- "
                f"crowd is {'not' if passed else 'already'} over-leveraged short"
            )
        return GateResult(passed=passed, value=funding_rate, threshold=self.funding_veto_pct, detail=detail)

    # -- public API --------------------------------------------------------

    def evaluate_market_conditions(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame,
        funding_rate: float | None,
        *,
        direction: Direction,
    ) -> tuple[bool, dict]:
        """Run all five gates and return (allowed, diagnostics).

        Args:
            df: The candidate symbol's own closed-candle OHLCV DataFrame
                (needs "close", "high", "low", "volume" columns), on
                whichever timeframe the chop/trend-strength/liquidity gates
                should evaluate.
            btc_df: BTCUSDT's closed-candle OHLCV DataFrame (needs "close"),
                already at the timeframe the macro beta gate should use
                (e.g. 4h) -- this method does not resample.
            funding_rate: Current funding rate for the candidate symbol, in
                the same period convention you intend `funding_veto_pct` to
                mean (e.g. %/8h). None if unavailable -- see class docstring
                for why that passes rather than blocks.
            direction: "LONG" or "SHORT" -- the trade direction being
                considered, since the macro beta and crowd sentiment gates
                are direction-dependent.

        Returns:
            (allowed, diagnostics) where `allowed` is True only if every
            gate passed, and `diagnostics` is a plain, JSON-serializable
            dict:
                {
                    "allowed": bool,
                    "direction": "LONG" | "SHORT",
                    "failed_gates": [names...],
                    "gates": {
                        "macro_beta": {"passed": ..., "value": ..., "threshold": ..., "detail": "..."},
                        "chop": {...},
                        "trend_strength": {...},
                        "liquidity_volume": {...},
                        "stretch": {...},
                        "squeeze": {...},
                        "crowd_sentiment": {...},
                    },
                }

        Raises:
            ValueError: if direction is invalid, or df/btc_df are missing
                required columns. Never raises for insufficient history or
                missing funding data -- those are fail-safe gate results.
        """
        if direction not in ("LONG", "SHORT"):
            raise ValueError('direction must be "LONG" or "SHORT"')

        gates = {
            "macro_beta": self._macro_beta_gate(btc_df, direction),
            "chop": self._chop_gate(df),
            "trend_strength": self._trend_strength_gate(df),
            "liquidity_volume": self._liquidity_volume_gate(df),
            "stretch": self._stretch_gate(df),
            "squeeze": self._squeeze_gate(df),
            "crowd_sentiment": self._crowd_sentiment_gate(funding_rate, direction),
        }

        allowed = all(g.passed for g in gates.values())
        failed_gates = [name for name, g in gates.items() if not g.passed]

        diagnostics = {
            "allowed": allowed,
            "direction": direction,
            "failed_gates": failed_gates,
            "gates": {name: g.to_dict() for name, g in gates.items()},
        }
        return allowed, diagnostics
