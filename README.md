# ZENITH + GEM + KRYPTIC (three-strategy Telegram signal bot)

Telegram / Cornix signal bot running **three independent strategy engines**
in one process, one chat, one Telegram bot token, and one shared `RiskGuard`
portfolio drawdown ceiling:

- **ZENITH** (`strategy.py`) — default live profile **`smc_lite`**: HTF BOS /
  swing bias + FVG/OB POI proximity entries (MATRIX-inspired), with
  MATRIX-ORIENT exits (early TPs, BE+buffer after TP1, leverage **10–15x**,
  `CLOSE_VELAS=20`).
- **GEM** (`gem_strategy.py`) — Fibonacci-retracement entries (79% pullback
  of a recent impulse leg), ATR-based stop, R-based quality filters, SMC
  (structure/FVG/order-block/liquidity-sweep) confluence measurements. A
  Python port of an uploaded `motor-gem.js` engine, cross-checked
  numerically against the original Node implementation for its core E1/E2/E3
  and TP1/TP2/TP3 math (see "Porting notes" below) — a 4th entry tier and 2
  more take-profits were added on top of that verified port so GEM posts the
  same 4-entry/5-TP shape as ZENITH and KRYPTIC. Signals post as `GEM SIGNAL`
  instead of `ZENITH SIGNAL` so the two are easy to tell apart in the
  channel; toggle with `GEM_ENABLED` in `.env`.
- **KRYPTIC** (`kryptic_strategy.py`, over `entry_ladder.py`/
  `regime_filter.py`/`directional_bias.py`/`risk_manager.py`) — a 4-tier
  scale-in entry ladder (breakout/FVG-proximal, two Fibonacci retracement
  tiers, a liquidity-sweep/ATR-band tier), gated by `RegimeFilter` +
  `DirectionEngine`, with 5 fixed ATR-multiple take-profits re-anchored to
  the real filled VWAP every bar. Unlike ZENITH/GEM, its own entry gate is
  pass/fail rather than a continuous quality score. This module is a
  Telegram-signal adapter around the same synchronous pieces
  `engine.py`'s `KrypticEngine` (the persistent, websocket-driven,
  crash-recoverable engine used by `multi_strategy_manager.py`'s hedge-mode
  harness) is built from — it posts signals the same way ZENITH/GEM do
  rather than running that full streaming engine. Toggle with
  `KRYPTIC_ENABLED` in `.env`.

**All three engines now post the same signal shape: 4 scale-in entries and
5 take-profit targets.**

All three engines scan on the same `SCAN_SECONDS` cadence, post to the same
`TELEGRAM_CHANNEL_ID`, and share cooldown bookkeeping only *within* each
engine (a symbol can signal once per engine independently). Each engine's
cooldown is keyed per **(symbol, direction)**, not just per symbol: a
same-direction repeat on the same coin is blocked until `COOLDOWN_MINUTES`/
`GEM_COOLDOWN_MINUTES`/`KRYPTIC_COOLDOWN_MINUTES` elapses, but a fresh
signal in the *opposite* direction on that same coin is never blocked by
it — it posts as soon as the engine finds one. Engines still share one
`RiskGuard` instance, so combined simulated drawdown across all three
strategies gates new signals from any of them. See `.env.example`'s GEM and
KRYPTIC sections for every tunable knob.

## Price data source (Cornix mismatch fix)

Both strategies and `RiskGuard` fetch tickers/candles/funding through
`exchanges.resolve_source()`, which reads `DATA_SOURCE` from `.env`:

- **`blofin`** (default) — BloFin only, no cross-venue fallback. Every
  symbol the bot scans is confirmed actually listed on BloFin, and every
  price a signal is built from is BloFin's own — so if your Cornix account
  executes on BloFin, there's no cross-exchange price/listing mismatch
  between what gets posted and what Cornix fills.
- **`binance`** — the original Binance → MEXC → Bitget fallback chain
  (broader universe, more historical depth), for a Cornix account that
  executes somewhere else, or when exact-venue matching doesn't matter.

Switching `DATA_SOURCE` moves ZENITH, GEM, *and* `RiskGuard`'s drawdown
replay together — RiskGuard replaying a BloFin-priced trade against Binance
candles would just reintroduce the same mismatch one level down.

BloFin's public API doesn't publish a documented funding-rate ticker field
or a per-symbol max-leverage endpoint, so under `DATA_SOURCE=blofin`,
funding veto is always inactive (matches this repo's own prior BloFin
adapter for GEM, which made the same call) and leverage is never capped
below your configured `LEVERAGE_MAX`/`GEM_LEVERAGE_MAX` by an exchange
limit. The BloFin tickers/candles endpoints (`exchanges.py`'s
`load_universe_blofin`/`fetch_klines_blofin`) were verified against a mock
response shape, not a live request — this sandbox's network policy blocks
outbound access to every exchange host, BloFin included, so **verify
against BloFin's real API before trusting this in production**, the same
caveat this repo's other non-Binance/Bitget adapters already carry.

## ZENITH details

Default live profile **`smc_lite`**: HTF BOS / swing bias + FVG/OB POI proximity entries (MATRIX-inspired), with MATRIX-ORIENT exits (early TPs, BE+buffer after TP1, leverage **10–15x**, `CLOSE_VELAS=20`).

Fallback profile: `STRATEGY_PROFILE=matrix_orient` (fib pullback entries only).

## Cornix MATRIX bar

Cornix “Rapid Scalpers Premium” (~20d live): **WR 77%**, **max DD 9.8%**, avg hold **~3h15m**, leverage **10x**, Sharpe ~2.38.

Prior ZENITH matrix_orient (exit-only port): ~**68% WR / 6m**, ~**65% WR / 20d**, DD ≈5.7%. This pack adds **SMC-lite entries**.

**Ship backtest (p10):** 6m **WR 76.6%**, DD **4.2%**, ret **+22.8%**, hold ~1.8h, lev ~12x — near Cornix 77% on 6m; ~20d still thin (WR 66.7%, n=6). Bar not fully cleared on 20d without ABSORCION.

## Winrate/return optimization pass (leverage 10–15x, target ≤3% DD)

Leverage moved up from 8–12x to **10–15x**. Since this bot only posts signals (no
account/broker connection), leverage under risk-based sizing mainly governs
margin/liquidation headroom, not the equity swing from a stop-out — so pushing
leverage up without also tightening the levers that *do* drive drawdown would
just add liquidation risk for no return benefit. This pass changes both sides:

**Entry quality (raises winrate):**
- `MIN_SCORE` 65 → 70; `BTC_REGIME=1` by default (alts are BTC-beta — veto
  counter-trend alt entries against the prevailing BTC regime instead of only
  score-rewarding alignment).
- New `WICK_REJECT_MAX` (default 0.45): rejects the confirm bar if it shows a
  strong opposite-direction wick (rejection/exhaustion) even when close/body
  direction passed.

**Risk band (offsets the higher leverage, targets ~3% DD):**
- `R_MIN_PCT` 0.95→1.00, `R_MAX_PCT` 2.20→1.85, `R_MAX_ATR` 2.0→1.7,
  `SL_CAP_PCT` 1.65→1.40 — cuts tail-loser size.
- `RISK_EQUITY_PCT` 1.00→0.65 — the actual per-trade drawdown lever under
  risk-based sizing; set your executor's (Cornix/etc.) per-trade risk to
  match this.
- New `LIQ_SAFETY_FRACTION` (default 0.50): hard leverage ceiling so a
  stop-out can never burn more than this fraction of isolated margin — keeps
  the 15x end of the band clear of liquidation even on the widest stops in
  the R band.

**Exits (raises return without adding risk):**
- New `BE_BUFFER_R` (default 0.12): after TP1, SL locks to BE **+ 0.12R**
  instead of exact breakeven, so a give-back to entry still nets a small win
  instead of a scratch trade.
- New `ADAPTIVE_TP` (default on): shifts size from TP1 toward TP2–5 as setup
  score rises above `MIN_SCORE` (up to 35% of TP1's weight at score ~95),
  so the best-quality setups hold more size into the 0.5–2.3R zone instead of
  capping ~80% of size at ~0.2R. Same stop, same risk — just harvests winners
  differently.

**Portfolio guardrail (`risk_guard.py`, new):** the bot has no broker
connection, so it can't see realized drawdown directly. `RiskGuard` replays
every posted signal forward against the same candles the bot already fetches
(conservative bar-order assumption: SL checked before TP within a bar),
keeps an approximate compounding equity curve from `RISK_EQUITY_PCT`, and:
- pauses new signals once simulated drawdown exceeds `DD_CEILING_PCT`
  (default 3.0), resuming only once it recovers under `DD_RESUME_PCT`
  (default 1.5) — hysteresis to avoid flapping at the ceiling;
- caps simultaneously open (simulated) trades via `MAX_CONCURRENT_TRADES`
  (default 4) and same-direction exposure via `MAX_CONCURRENT_SAME_SIDE`
  (default 3), so correlated alts can't stack drawdown risk unnoticed.

This is a simulation for self-throttling, not a substitute for reconciling
against real fills — it assumes full fills at the published average entry.
State persists to `risk_state.json` (gitignored) next to the bot.

## Backtest -> Excel report

`backtest.py` runs a walk-forward backtest against real historical Binance
Futures candles and writes an `.xlsx` report (Summary / Trades / Equity
Curve / Monthly / Notes sheets). It reuses the live strategy code directly
(`strategy.py` / `gem_strategy.py` / `kryptic_strategy.py`, `risk_guard.py`)
and the same config path (`bot.configure_strategy` /
`bot.configure_gem_strategy` / `bot.configure_kryptic_strategy`), so it
reflects what `bot.py` would actually do live, not a separate
reimplementation. `--engine` selects which engine(s) run:
`zenith` (default) / `gem` / `kryptic` / `both` (ZENITH+GEM) / `all`
(ZENITH+GEM+KRYPTIC).

**Must be run somewhere with real internet access to `fapi.binance.com`**
— it could not be run or timed in the sandbox this bot was developed in
(outbound access there is policy-blocked), so treat any runtime figure as
indicative, not guaranteed on your machine.

```bash
pip install -r requirements-backtest.txt
python backtest.py --smoke                                   # ~3 day / 5-symbol sanity check first
python backtest.py --engine all --months 6 --top-n 30 --out backtest_report.xlsx
```

Key flags: `--engine` (see above), `--months` (default 6), `--top-n`
(default 30, ranked by current 24h quote volume like the live universe),
`--symbols` (comma list to override auto-selection for every running
engine), `--every` (evaluate every Nth 15m bar for a faster/coarser
preview), `--out` (xlsx path). `--kryptic-max-posts-per-scan` /
`--kryptic-cooldown-minutes` / `--kryptic-min-quote-vol` /
`--kryptic-top-n` mirror the existing `--gem-*` overrides for KRYPTIC.

**KRYPTIC is much slower per bar than ZENITH/GEM**: it runs
`TradeLifecycleManager.open_trade()` against a pandas DataFrame window per
symbol per bar (regime filter + direction bias + entry ladder, all
pandas-based), vs. ZENITH/GEM's pure-Python `evaluate()` functions. Including
it (`--engine kryptic` or `all`) noticeably increases total runtime — start
with `--smoke` or a narrower `--kryptic-top-n`/coarser `--every` before a
full 6-month `--engine all` run.

Cooldown in the backtest is per (symbol, direction) for every engine, same
as the live bot: a same-direction repeat on a symbol is blocked until its
cooldown window elapses, but a reversal is never blocked by it.

In dev-sandbox timing with synthetic data of the same shape (real network
access wasn't available to time it against actual Binance data), ZENITH+GEM
at 15 symbols x 6 months took ~30s single-threaded — a default 30-symbol x
6-month `--engine both` run should land in the low minutes, scaling roughly
linearly with symbols x months. `--engine all`/`kryptic` will run
meaningfully slower per the KRYPTIC note above. Start with `--smoke`, then
scale up.

Known simplifications (also written into the report's Notes sheet):
funding-rate veto isn't modeled (no historical funding pulled — a minor
soft filter, not the core edge); CVD/OI absorption was never modeled live
or here (stub only); the symbol universe is today's top-N by volume held
fixed across the lookback window (mild survivorship bias); fills are
assumed exact at the published average entry (no slippage modeled).

## SMC-lite knobs (`.env`)

| Knob | Default | Meaning |
|---|---|---|
| `STRATEGY_PROFILE` | `smc_lite` | `smc_lite` / `matrix_orient` / `riskcap` |
| `SMC_LITE` | `1` | Force SMC-lite entry path |
| `EXIGIR_BOS` | `1` | Require HTF structure bias |
| `EXIGIR_FVG` | `0` | Hard-require live FVG/OB near price |
| `EXIGIR_CHOCH_LTF` | `0` | Hard-require 15m CHoCH/BOS confirm |
| `EXIGIR_DESCUENTO` | `0` | Longs only discount / shorts only premium |
| `EXIGIR_POI_OR_CHOCH` | `1` | Require FVG/OB near **or** 15m CHoCH |
| `EXIGIR_FVG_INSIDE` | `0` | Hard-require price inside FVG/OB |
| `SMC_POI_SOFT` / `SMC_PD_SOFT` | `1` | Score-boost POI / PD alignment |
| `SMC_ENTRY_MODE` | `poi_fib` | `poi` \| `fib` \| `poi_fib` |
| `SMC_FVG_NEAR_ATR` | `2.20` | POI proximity band (ship) |
| `SMC_BUFFER_POI` | `0.25` | SL cushion beyond POI (ATR) |

**Exits (MATRIX-ORIENT):** `TP` / `CIERRES` (adaptive via `ADAPTIVE_TP`) / `BE_AFTER_TP1` + `BE_BUFFER_R` / `STOP_APRIETA` / `CLOSE_VELAS=20` / `LEV 10–15` (liq-safety clamped via `LIQ_SAFETY_FRACTION`).

**Kept hygiene:** HTF EMA align, stretch, impulse vol, ER, confirm bar + wick-rejection (`WICK_REJECT_MAX`), denylist, `MIN_QUOTE_VOLUME_USD=20000000`, `SIDE_BALANCE=1`.

**Portfolio-level (new):** `risk_guard.py` drawdown ceiling (`DD_CEILING_PCT`/`DD_RESUME_PCT`) and concurrency caps (`MAX_CONCURRENT_TRADES`/`MAX_CONCURRENT_SAME_SIDE`) — see optimization section above.

**Not modeled in backtest:** CVD/OI absorption (stub only).

## GEM (`gem_strategy.py`)

Ported from an uploaded `motor-gem.js` ("GEM ENGINE v4", Node) reference
implementation. Only the live-signal path (`evaluar()` and everything it
calls) was ported — the same scope the original `bot.js` live bot itself
used; an alternate full-SMC entry mode (`evaluarSMC`, gated by a `MODO_SMC`
flag the original bot.js never wired in) and offline backtest helpers
(`simular`/`revisarTesis`) were left out since nothing live exercised them.

**How it fits in:** GEM runs as a second engine inside this same bot
process instead of a separate Node service — it reuses ZENITH's exchange
layer (`exchanges.py`, Binance → MEXC → Bitget fallback), `Signal` type,
Telegram feed and `RiskGuard`, rather than its own bespoke 9-exchange
adapter layer (only one of which — Bitget — the original bundle verified
against a live API).

**Porting notes:**
- Candle shape stays the original `{t,o,h,l,c,v}` (oldest-first) inside
  `gem_strategy.py` rather than being translated to this repo's
  `{ts,open,high,low,close,volume}` shape, so the module reads as a direct,
  checkable line-for-line port of `motor-gem.js`; `from_repo_candles()`
  adapts between the two shapes at the call site.
- `GEM_CFG` mirrors `motor-gem.js`'s `CFG` object key-for-key (translated to
  English, and with keys the original engine's live path never read — the
  time-of-day silence window, position-count caps, backtest-only knobs —
  dropped rather than carried over unused). Every remaining key is tunable
  from `.env` as `GEM_<KEY>` (e.g. `GEM_MIN_SCORE`, `GEM_REQUIRE_BOS`,
  `GEM_R_MAX_PCT`) via a generic, type-aware override loader in
  `bot.py:configure_gem_strategy()` — no per-field boilerplate to keep in
  sync.
- **Deviation from the port (KRPTICZ 4-entry/5-TP standardization):** the
  original `evaluar()`'s default entry path built 3 entries (E1/E2/E3) and
  `CFG.TP`/`CFG.CLOSES` had 3 take-profit tiers. A 4th entry (`E4`, gated by
  `GEM_CFG["E4_FRACTION"]`/`GEM_CFG["ENTRY_WEIGHTS"]`) and 2 more
  take-profits (`TP4`/`TP5`, via `TP`/`CLOSES` now holding 5 values each)
  were added so GEM posts the same 4-entry/5-TP signal shape as ZENITH and
  KRYPTIC. This is the one place `gem_strategy.py` is no longer numerically
  identical to `motor-gem.js` — every other rejection code, gate, and the
  E1/E2/E3/TP1/TP2/TP3 math below remain the original, verified port.
- Numerically verified against the original: a comparison harness ran both
  `motor-gem.js`'s `evaluar()` (Node) and `gem_strategy.evaluate()` (Python)
  over identical synthetic candle series — random-walk scenarios (uptrend,
  downtrend, choppy, low-priced alt, low-volume, volatile) exercising the
  early reject codes, plus hand-built "clean impulse + 79% pullback" setups
  tuned to clear every filter and reach the full accept path — across
  several funding-rate values and with each `REQUIRE_*`/`EXIGIR_*` gate
  toggled on individually. Every numeric field (entries, stop, take-profits,
  R%, R/ATR, score and its components, stochastic, SMC structure/FVG/sweep)
  and every rejection code matched exactly (rejection codes and SMC
  trend/bias strings were themselves anglicized during the port — e.g.
  `ALCANCE`→`REACH`, `ALCISTA`→`UP` — and mapped for comparison, not
  compared as literal strings).
- Entry/TP position-size weights are normalized to percentages summing to
  100 when building a `Signal` (`gem_strategy.build_signal`'s
  `_pct_weights`) — `RiskGuard` expects percentage points (see
  `risk_guard.py`), while GEM's own `pesos`/`CIERRES` are fractions summing
  to ~1.
- GEM spaces its take-profit ladder off the weighted-average entry (`Eavg`),
  not off `E1` the way ZENITH does — so unlike ZENITH's `build_signal`
  (which must use `abs(E1-SL)` as its R unit for `RiskGuard`), GEM's R unit
  is exactly `abs(Eavg-SL)`, matching how its own TPs were built.

## KRYPTIC (`kryptic_strategy.py`)

Third strategy engine, standing on top of the KRYPTIC track's own modules
(`entry_ladder.py`, `regime_filter.py`, `directional_bias.py`,
`risk_manager.py`) that predate this repo's Telegram-signal bot and were
originally built for `engine.py`'s `KrypticEngine` — a persistent,
websocket-driven, crash-recoverable position-tracking engine used by
`multi_strategy_manager.py`'s hedge-mode harness, a different architecture
from ZENITH/GEM's one-shot-per-scan `evaluate()`/`build_signal()` functions.
Building a real order-executing bridge between that engine and this bot is
a separate, much larger undertaking (see `multi_strategy_manager.py`'s own
"honest scope note").

`kryptic_strategy.py` instead reuses the same synchronous, pandas-based
pieces `KrypticEngine` itself is built from — `EntryLadderEngine`/
`RegimeFilter`/`DirectionEngine` via `TradeLifecycleManager.open_trade()` —
directly against the closed-candle DataFrame `bot.py` already fetches each
scan, exactly the way ZENITH/GEM's own `evaluate()` functions do. So, like
ZENITH and GEM, KRYPTIC here only ever posts a signal to Telegram/Cornix and
hands it to the shared `RiskGuard` for its own independent replay — it never
opens or tracks a persistent position the way `KrypticEngine` does.

**4-entry/5-TP standardization:**
- `entry_ladder.py`'s `LADDER_WEIGHTS` changed from `(0.40, 0.35, 0.25,
  0.0)` (3 active tiers — the origin-sweep/ATR-band Entry 4 was
  0-weighted, "effectively disabled") to `(0.40, 0.30, 0.20, 0.10)`, all 4
  tiers now carrying real size.
- `risk_manager.py`'s `TP_WEIGHTS` changed from `(0.20, 0.50, 0.0, 0.0,
  0.0)` (2 fixed targets, with the remaining 0.30 going to an ATR
  chandelier-trail runner tier instead of a 3rd fixed target) to `(0.20,
  0.30, 0.20, 0.15, 0.15)` — 5 real, fixed-price targets, all re-anchored
  to the real filled VWAP every bar the same way TP1/TP2 always were. The
  trailing-runner mechanism (`PositionState.runner_weight`, default now
  `0.0`) still exists and works exactly as before for a caller who
  explicitly reconfigures `tp_weights` toward the legacy 2-target-plus-
  runner geometry.
- Since KRYPTIC has no continuous quality score (its `RegimeFilter`/
  `DirectionEngine` gate is pass/fail), `kryptic_strategy.KRYPTIC_CFG
  ["SCORE"]` (default 75) is a fixed value that only feeds
  `leverage_from_quality`'s score-to-leverage mapping and `bot.py`'s
  cross-candidate ranking within one scan — it never gates whether a setup
  is taken.

## Run (Google Cloud VM)

```bash
cd zenithupdated
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python bot.py
```

`.env` holds Telegram secrets — keep private. Copy knobs from `.env.example`.

```bash
sudo cp zenith.service /etc/systemd/system/zenith.service
# edit User= and paths, then:
sudo systemctl daemon-reload
sudo systemctl enable --now zenith
sudo journalctl -u zenith -f
```

Do **not** `pip install telegram`. This bot uses **httpx**.
