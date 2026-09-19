"""entry_diagnostics.py -- profiles entry-side signal attributes (candle
conviction, volume ratio, entry-ladder fill depth) against realized trade
outcomes under the Step 11.3 settled exit geometry, and reports simple
threshold what-if analysis for three candidate entry filters raised in
that step's chat: a candle-body-conviction gate, a vol_ratio gate, and
truncating resting entry-ladder fills to the first N bars after signal.

Motivation: the Step 11.2 production verification backtest's TIME_DECAY
bucket (24 trades, -$1,435, avg -0.596R) was shown by grid_resimulation's
Round 6 time_decay_bars sweep to be untouchable by exit-side timing --
avg loss stayed flat (~-0.66 to -0.69R) across every cutoff from 3 to 6
bars, meaning the adverse move is already baked in within the first ~45
minutes, not something a later exit trigger can catch sooner. This
script asks the entry-side question instead: were there OBSERVABLE
attributes of the signal bar itself (how convincingly it closed, how
much volume confirmed it) or of the ladder's own fill behavior (how many
tiers got dragged into worse and worse prices before the position ever
established) that would have flagged these trades in advance.

Two-phase design, same architecture as grid_resimulation.py:
  1. Reuse grid_resimulation.generate_frozen_trades() to get the exact
     same real, unmodified production entries this whole KRYPTIC track
     has used since Step 10.11 -- identical signals, identical fills.
  2. For each frozen trade, capture the signal bar's own OHLC + rvol
     (BEFORE any fill activity -- this is what a pre-entry gate would
     see) and the trade's ladder-fill depth, then replay ONE FIXED exit
     geometry (the Step 11.3 settled values) via grid_resimulation's own
     replay_exit() to get the realized outcome. Join the two into one
     per-trade profile row.

This is a DIAGNOSTIC, not a gate -- it doesn't reject any signal itself
or change any production behavior. It measures whether the proposed
gates WOULD separate good and bad outcomes on the real, frozen baseline
population, and reports the false-positive cost (real winners a given
threshold would also have rejected) alongside the true-positive benefit
(decay/SL trades it would have prevented), so a decision to actually
implement a gate is made on that trade-off, not on a plausible-sounding
formula alone.

NOT implemented here: the separate "0.30 ATR favorable-excursion within
3 bars" early-stagnation exit idea -- that's an exit-side condition
(already ruled unhelpful by Round 6's flat avg_R finding), not an entry
filter, and is out of this script's scope.

This repo's own dev sandbox has no outbound route to any exchange and no
cached OHLCV -- run this on a machine with real market data access, same
as grid_resimulation.py.

Usage:
    python3 -m entry_diagnostics --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 180
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest_engine import HistoricalDataLoader, VectorizedSignalCache
from grid_resimulation import FrozenTrade, GeometryParams, generate_frozen_trades, replay_exit
from risk_manager import TradeLifecycleManager

log = logging.getLogger(__name__)

# Step 11.3 settled exit geometry (see grid_resimulation.py's own grid
# history) -- held FIXED here, not swept; this script profiles entries,
# not exits.
SETTLED_GEOMETRY = GeometryParams(
    sl_atr=2.60, tp1_atr=1.10, tp1_w=0.20, tp2_atr=2.80, tp2_w=0.50, runner_w=0.30,
    scratch_trigger_atr_mult=0.60, scratch_offset_atr_mult=0.35, time_decay_bars=6,
)

VOL_RATIO_THRESHOLDS = [1.5, 1.8, 2.0, 2.2]
# (close-low)/(high-low) for LONG, (high-close)/(high-low) for SHORT --
# "how close to the favorable extreme did the signal bar close," ALREADY
# direction-normalized in _signal_features() below (there is no separate
# "raw" vs "directional" variant of this metric in this file -- every
# close_location value this script has ever emitted is the mirrored one).
# >=0.75 is the "top/bottom 25%" framing from the chat; the lower values
# are included, and the grid is fine enough (0.05 steps), so a threshold
# that guts trade volume -- or a plateau vs. a lucky bin edge -- shows up
# as such rather than being silently skipped or aliased away.
CLOSE_LOCATION_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
BODY_TO_RANGE_THRESHOLDS = [0.50, 0.60, 0.70]  # |close-open|/(high-low)
# Bars-after-signal cutoff for resting ladder fills; None = no truncation
# (the real, unmodified baseline).
MAX_FILL_OFFSET_GRID: list[int | None] = [1, 2, 3, None]

_BAD_EXIT_REASONS = {"TIME_DECAY", "SL"}
# NOTE: replay_exit() (grid_resimulation.py) emits "RUNNER_TRAIL" for the
# trailing-runner tier, never bare "RUNNER" -- an earlier version of this
# set had "RUNNER" instead, which silently never matched anything, so
# every RUNNER_TRAIL trade a threshold rejected was missing from
# good_rejected in every threshold_whatif() table printed before this fix.
_GOOD_EXIT_REASONS = {"TP1", "TP2", "RUNNER_TRAIL"}

_FrozenBundle = tuple[list[FrozenTrade], VectorizedSignalCache]


def _signal_features(cache: VectorizedSignalCache, signal_index: int, direction: str) -> dict:
    """Candle-shape + volume features of the SIGNAL bar itself -- what a
    pre-entry gate would see, before any ladder fill has happened.

    `close_location` is ALREADY direction-normalized: (close-low)/range
    for LONG (close near the HIGH is favorable), (high-close)/range for
    SHORT (close near the LOW is favorable) -- there has never been an
    unmirrored/raw variant of this metric in this file. A real per-trade
    profile once showed SHORT TP2 trades averaging LOWER close_location
    (0.473) than SHORT TIME_DECAY trades (0.563) -- that is NOT this
    formula silently reverting to raw (close-low)/range for shorts (it
    doesn't); it is the metric genuinely not separating good from bad
    SHORT outcomes the same way it separates LONG ones (which DOES show
    the expected monotonic ordering). See threshold_whatif_by_direction()."""
    bar = cache.ltf_df.iloc[signal_index]
    o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    rng = h - l
    body_to_range = abs(c - o) / rng if rng > 0 else float("nan")
    if direction == "LONG":
        close_location = (c - l) / rng if rng > 0 else float("nan")
    else:
        close_location = (h - c) / rng if rng > 0 else float("nan")
    vol_ratio = cache.rvol.iat[signal_index]
    return {
        "body_to_range": body_to_range,
        "close_location": close_location,
        "vol_ratio": float(vol_ratio) if not pd.isna(vol_ratio) else float("nan"),
    }


def _htf_features(cache: VectorizedSignalCache, signal_index: int) -> dict:
    """HTF-structure staleness + working-timeframe EMA bias strength at
    signal time -- probing the "a stale/lagging HTF confirmation is
    misclassifying a counter-trend pullback as a fresh breakout" idea
    raised for SHORT's per-symbol asymmetry (BTC SHORT profitable, ETH/SOL
    SHORT not). `htf_trend` itself is NOT diagnostic on its own --
    DirectionEngine's HTF condition requires it to already match
    direction as a precondition for a ladder to build at all (every row
    in this profile already cleared it -- see backtest_engine.py's own
    Notes sheet: "'htf_aligned' ... always True for every row here").
    What CAN vary: how long ago that confirmation happened (a stale one
    is weaker evidence of a live trend than a fresh one), and the
    working-timeframe EMA20/EMA50 stack's own spread (bias strength,
    independent of the HTF gate).

    `htf_break_age_buckets`: HTF buckets (0 = the current, most-recent
    bucket) since `htf_trend_by_bucket` last changed value -- None if the
    HTF bucket lookup itself isn't available yet (early warmup).
    `ema_spread_atr`: (ema_fast - ema_slow) / atr14 at the signal bar,
    UNSIGNED (not flipped for SHORT) -- more negative means a stronger
    bearish EMA stack in raw terms, so compare LONG vs SHORT populations
    separately, not pooled."""
    bucket_idx = int(cache.htf_row_idx[signal_index]) if signal_index < len(cache.htf_row_idx) else -1
    if bucket_idx < 0 or bucket_idx >= len(cache.htf_trend_by_bucket):
        age = None
    else:
        trend = cache.htf_trend_by_bucket
        current = trend[bucket_idx]
        age = 0
        j = bucket_idx
        while j > 0 and trend[j - 1] == current:
            j -= 1
            age += 1
    ema_fast, ema_slow, atr = cache.ema_fast.iat[signal_index], cache.ema_slow.iat[signal_index], cache.atr14.iat[signal_index]
    if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(atr) or atr <= 0:
        ema_spread_atr = float("nan")
    else:
        ema_spread_atr = float((ema_fast - ema_slow) / atr)
    return {"htf_break_age_buckets": age, "ema_spread_atr": ema_spread_atr}


def _fill_depth(trade: FrozenTrade) -> dict:
    """How many entry-ladder tiers filled, and what fraction of the
    intended position size that represents -- the "adverse fill" signal
    raised in Step 11.3's chat (deep multi-tier fills implying price
    dragged well past the breakout level before the ladder even finished
    building)."""
    tiers_hit = sorted({tier_i for _, tier_i, _ in trade.fill_events})
    total_w = sum(trade.ladder_weights)
    filled_w = sum(trade.ladder_weights[t] for t in tiers_hit)
    return {
        "n_tiers_hit": len(tiers_hit),
        "fill_pct": round(100.0 * filled_w / total_w, 2) if total_w > 0 else float("nan"),
    }


def _truncate_fills(trade: FrozenTrade, max_offset: int | None) -> FrozenTrade:
    """Counterfactual: drop any resting ladder tier that (in reality)
    filled LATER than `max_offset` bars after the signal bar -- simulates
    "stop waiting for a deeper retracement fill after bar N." A trade
    left with zero fill_events after truncation is a trade that never
    established a position at all under this rule (mirrored by
    replay_exit returning None for it, same as a real full cancel)."""
    if max_offset is None:
        return trade
    kept = tuple(e for e in trade.fill_events if e[0] <= max_offset)
    return replace(trade, fill_events=kept)


def profile_trades(frozen_by_symbol: dict[str, _FrozenBundle], geometry: GeometryParams = SETTLED_GEOMETRY) -> pd.DataFrame:
    """One row per frozen trade that established a position: signal-bar
    features + ladder fill depth + realized outcome under `geometry`."""
    rows = []
    for symbol, (trades, cache) in frozen_by_symbol.items():
        atr_arr = cache.atr14.to_numpy()
        ema20_arr = cache.ema20.to_numpy()
        high_arr = cache.ltf_df["high"].to_numpy()
        low_arr = cache.ltf_df["low"].to_numpy()
        close_arr = cache.ltf_df["close"].to_numpy()
        for t in trades:
            result = replay_exit(t, atr_arr=atr_arr, ema20_arr=ema20_arr, high_arr=high_arr, low_arr=low_arr, close_arr=close_arr, params=geometry)
            if result is None:
                continue
            row = {"symbol": symbol, "signal_ts": t.signal_ts, "direction": t.direction}
            row.update(_signal_features(cache, t.signal_index, t.direction))
            row.update(_htf_features(cache, t.signal_index))
            row.update(_fill_depth(t))
            row.update(result)
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("signal_ts").reset_index(drop=True)


def bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean signal-bar/fill-depth attributes per exit_reason bucket --
    answers "profile these entry attributes against trade outcomes"
    directly: are TIME_DECAY/SL trades observably different at entry
    from TP1/TP2/RUNNER trades, on average?"""
    if df.empty:
        return df
    agg = df.groupby("exit_reason").agg(
        n=("r_multiple", "size"),
        avg_R=("r_multiple", "mean"),
        sum_R=("r_multiple", "sum"),
        avg_body_to_range=("body_to_range", "mean"),
        avg_close_location=("close_location", "mean"),
        avg_vol_ratio=("vol_ratio", "mean"),
        avg_fill_pct=("fill_pct", "mean"),
        avg_n_tiers_hit=("n_tiers_hit", "mean"),
    ).round(4)
    return agg.sort_values("sum_R")


def symbol_direction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean R and HTF/EMA-bias features per (symbol, direction) --
    probes the per-symbol SHORT asymmetry (BTC SHORT profitable, ETH/SOL
    SHORT not): is a stale HTF confirmation or a weaker EMA stack
    disproportionately present on the underperforming symbols' SHORTs?
    `avg_ema_spread_atr` is intentionally UNSIGNED (see `_htf_features`'s
    own docstring) -- compare within a direction, not across LONG/SHORT."""
    if df.empty:
        return df
    agg = df.groupby(["symbol", "direction"]).agg(
        n=("r_multiple", "size"),
        avg_R=("r_multiple", "mean"),
        sum_R=("r_multiple", "sum"),
        avg_htf_break_age_buckets=("htf_break_age_buckets", "mean"),
        avg_ema_spread_atr=("ema_spread_atr", "mean"),
    ).round(4)
    return agg.sort_values("sum_R")


def symbol_exclusion_whatif(df: pd.DataFrame, direction: str, exclude_symbols: list[str]) -> dict:
    """Direct answer to "does disabling SHORTs on symbol X alone flip the
    book net positive": EV/sum_R for `direction`'s population with and
    without `exclude_symbols` removed, one symbol at a time and combined
    -- so a "flip to positive" claim also shows the resulting MAGNITUDE,
    not just the sign, and each remaining symbol's own standalone number
    stays visible instead of being averaged away again."""
    sub = df[df.direction == direction]
    out = {"baseline": {"n": len(sub), "sum_R": round(float(sub.r_multiple.sum()), 4), "ev_R": round(float(sub.r_multiple.mean()), 4) if len(sub) else float("nan")}}
    for sym in exclude_symbols:
        remaining = sub[sub.symbol != sym]
        out[f"excl_{sym}"] = {
            "n": len(remaining),
            "sum_R": round(float(remaining.r_multiple.sum()), 4) if len(remaining) else 0.0,
            "ev_R": round(float(remaining.r_multiple.mean()), 4) if len(remaining) else float("nan"),
        }
    remaining_all = sub[~sub.symbol.isin(exclude_symbols)]
    out["excl_all_listed"] = {
        "n": len(remaining_all),
        "sum_R": round(float(remaining_all.r_multiple.sum()), 4) if len(remaining_all) else 0.0,
        "ev_R": round(float(remaining_all.r_multiple.mean()), 4) if len(remaining_all) else float("nan"),
    }
    return out


def threshold_whatif(
    df: pd.DataFrame, column: str, thresholds: list[float], *,
    higher_is_better: bool = True, track_reasons: list[str] | None = None,
) -> pd.DataFrame:
    """For each threshold, simulates REJECTING every signal that fails
    it (keep if column >= threshold when higher_is_better, else <=), and
    reports: how many TIME_DECAY/SL trades that would have prevented
    (the benefit), how many real TP1/TP2/RUNNER_TRAIL winners it would
    ALSO have rejected (the cost), the surviving population's own EV and
    profit factor -- so the trade-off is visible, not just the benefit.

    `track_reasons` (e.g. ["TIME_DECAY", "TP2", "RUNNER_TRAIL"]) adds one
    `<reason>_kept` column per named exit_reason -- finer-grained than
    the bad_rejected/good_rejected split, for when the aggregate hides an
    asymmetry (e.g. this metric behaving differently for LONG vs. SHORT --
    see threshold_whatif_by_direction)."""
    if df.empty:
        return df
    valid = df.dropna(subset=[column])
    ev_all = round(float(valid["r_multiple"].mean()), 4) if len(valid) else float("nan")
    rows = []
    for thr in thresholds:
        keep = (valid[column] >= thr) if higher_is_better else (valid[column] <= thr)
        kept, rejected = valid[keep], valid[~keep]
        if len(kept):
            wins, losses = kept["r_multiple"][kept["r_multiple"] > 0], kept["r_multiple"][kept["r_multiple"] <= 0]
            gross_loss = abs(float(losses.sum()))
            profit_factor_kept = round(float(wins.sum()) / gross_loss, 4) if gross_loss > 0 else float("nan")
        else:
            profit_factor_kept = float("nan")
        row = dict(
            threshold=thr,
            n_kept=len(kept), n_rejected=len(rejected),
            bad_rejected=int(rejected["exit_reason"].isin(_BAD_EXIT_REASONS).sum()),
            good_rejected=int(rejected["exit_reason"].isin(_GOOD_EXIT_REASONS).sum()),
            ev_R_kept=round(float(kept["r_multiple"].mean()), 4) if len(kept) else float("nan"),
            ev_R_all=ev_all,
            profit_factor_kept=profit_factor_kept,
        )
        for reason in track_reasons or []:
            row[f"{reason}_kept"] = int((kept["exit_reason"] == reason).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_whatif_by_direction(
    df: pd.DataFrame, column: str, thresholds: list[float], *,
    higher_is_better: bool = True, track_reasons: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Same as threshold_whatif(), split by direction -- a metric that
    looks discriminating in the blended population can be driven entirely
    by one direction while doing nothing (or the opposite) for the other;
    this makes that visible instead of averaging it away."""
    return {
        direction: threshold_whatif(sub, column, thresholds, higher_is_better=higher_is_better, track_reasons=track_reasons)
        for direction, sub in df.groupby("direction")
    }


def fill_truncation_whatif(
    frozen_by_symbol: dict[str, _FrozenBundle],
    max_offsets: list[int | None] = MAX_FILL_OFFSET_GRID,
    geometry: GeometryParams = SETTLED_GEOMETRY,
) -> pd.DataFrame:
    """Item 3: what if resting entry-ladder tiers were cancelled after
    `max_offset` bars post-signal instead of being left open indefinitely?
    Re-derives each trade's filled weight/vwap/outcome from the
    TRUNCATED fill sequence -- a trade that never fills any tier under
    truncation drops out entirely, same as a real cancel-to-zero-fill."""
    rows = []
    for max_offset in max_offsets:
        results: list[dict] = []
        for symbol, (trades, cache) in frozen_by_symbol.items():
            atr_arr = cache.atr14.to_numpy()
            ema20_arr = cache.ema20.to_numpy()
            high_arr = cache.ltf_df["high"].to_numpy()
            low_arr = cache.ltf_df["low"].to_numpy()
            close_arr = cache.ltf_df["close"].to_numpy()
            for t in trades:
                truncated = _truncate_fills(t, max_offset)
                r = replay_exit(truncated, atr_arr=atr_arr, ema20_arr=ema20_arr, high_arr=high_arr, low_arr=low_arr, close_arr=close_arr, params=geometry)
                if r is not None:
                    results.append(r)
        if not results:
            continue
        r_values = np.array([r["r_multiple"] for r in results])
        decay = [r["r_multiple"] for r in results if r["exit_reason"] == "TIME_DECAY"]
        rows.append(dict(
            max_fill_offset=("none" if max_offset is None else max_offset),
            n_trades=len(results), ev_R=round(float(r_values.mean()), 4),
            win_rate_pct=round(100.0 * float((r_values > 0).mean()), 2),
            n_time_decay=len(decay),
            time_decay_avg_R=round(float(np.mean(decay)), 4) if decay else float("nan"),
        ))
    return pd.DataFrame(rows)


async def main() -> None:  # pragma: no cover -- real-network entrypoint, not exercised by tests
    parser = argparse.ArgumentParser(description="KRYPTIC entry-signal diagnostics")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--warmup-days", type=int, default=10)
    parser.add_argument("--cache-dir", default="./.backtest_cache")
    parser.add_argument("--out", default="entry_diagnostics_per_trade.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * 86400 * 1000
    warmup_ms = args.warmup_days * 86400 * 1000

    loader = HistoricalDataLoader(cache_dir=args.cache_dir)
    baseline_mgr = TradeLifecycleManager()  # current production defaults -- defines the baseline entry gating
    btc_df = await loader.load("BTCUSDT", "4h", start_ms - warmup_ms, end_ms)

    frozen_by_symbol: dict[str, _FrozenBundle] = {}
    total_trades = 0
    for symbol in args.symbols.split(","):
        symbol = symbol.strip().upper()
        ltf_df = await loader.load(symbol, "15m", start_ms - warmup_ms, end_ms)
        funding = await loader.load_funding(symbol, start_ms - warmup_ms, end_ms)
        trades, cache = generate_frozen_trades(symbol, ltf_df, btc_df, funding, baseline_mgr)
        trades = [t for t in trades if t.signal_ts >= start_ms]
        frozen_by_symbol[symbol] = (trades, cache)
        total_trades += len(trades)
        print(f"{symbol}: {len(trades)} baseline trade(s) generated for profiling")
    await loader.aclose()

    print(f"\nProfiling {total_trades} frozen trade(s) under the Step 11.3 settled exit geometry...")
    t0 = time.time()
    df = profile_trades(frozen_by_symbol)
    df.to_csv(args.out, index=False)
    print(f"Done in {time.time() - t0:.1f}s -- wrote {len(df)} row(s) to {args.out}\n")

    print("=== Bucket summary (entry attributes by exit_reason) ===")
    print(bucket_summary(df).to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== vol_ratio threshold what-if (keep if vol_ratio >= threshold) ===")
    print(threshold_whatif(df, "vol_ratio", VOL_RATIO_THRESHOLDS).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    _CL_REASONS = ["TIME_DECAY", "TP2", "RUNNER_TRAIL"]
    print("\n=== close_location threshold what-if, ALL directions (keep if close_location >= threshold) ===")
    print(threshold_whatif(df, "close_location", CLOSE_LOCATION_THRESHOLDS, track_reasons=_CL_REASONS)
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n--- same, split by direction (a blended signal can hide one direction doing nothing/the opposite) ---")
    for direction, table in threshold_whatif_by_direction(df, "close_location", CLOSE_LOCATION_THRESHOLDS, track_reasons=_CL_REASONS).items():
        print(f"\n{direction}:")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== body_to_range threshold what-if (keep if body_to_range >= threshold) ===")
    print(threshold_whatif(df, "body_to_range", BODY_TO_RANGE_THRESHOLDS).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Fill-truncation what-if (cancel resting ladder tiers after N bars) ===")
    print(fill_truncation_whatif(frozen_by_symbol).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Per-symbol/direction summary (HTF staleness + EMA-stack bias strength) ===")
    print(symbol_direction_summary(df).to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== SHORT symbol-exclusion what-if ===")
    for label, stats in symbol_exclusion_whatif(df, "SHORT", ["SOLUSDT", "ETHUSDT"]).items():
        print(f"  {label:20s} n={stats['n']:4d}  sum_R={stats['sum_R']:+8.3f}  ev_R={stats['ev_R']:+.4f}")


if __name__ == "__main__":
    asyncio.run(main())
