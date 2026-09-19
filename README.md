# KRYPTIC

A standalone trading engine split out of [`traderhydr/cloude-bot`](https://github.com/traderhydr/cloude-bot)
(the ZENITH + GEM signal bot repo), where it was originally built to run
alongside those two strategies in the same process. This repo contains
only the KRYPTIC-specific modules and their tests; it has no dependency on
`bot.py`, `strategy.py`, `gem_strategy.py`, `smc_lite.py`, `risk_guard.py`,
or `formatter.py` from that repo.

## Modules

- `indicators.py` — shared TA primitives (ATR, EMA, efficiency ratio, etc.)
- `directional_bias.py` — HTF trend/bias engine
- `regime_filter.py` — squeeze/stretch market-regime gates
- `entry_ladder.py` — impulse-leg discovery, FVG/retracement entry pricing
- `risk_manager.py` — `TradeLifecycleManager`/`PositionState`: fill/TP/BE/exit state machine
- `resilience_manager.py` — clock-sync, gap-fill replay, retry, atomic state persistence
- `data_collector.py` — `MarketDataPipeline`: candle buffering/aggregation/heartbeat
- `engine.py` — `KrypticEngine`, the top-level per-symbol orchestrator
- `multi_strategy_manager.py` — `MultiStrategyManager`/`RiskArbiter`/`SharedOrderBook` for running several strategies against one account safely
- `live_runner.py` — exchange adapter (BloFin/Binance), universe scanner, paper-trading dry-run harness, live entrypoint
- `backtest_engine.py` — historical backtest engine with an institutional-style Excel/CSV report, backtest entrypoint
- `grid_resimulation.py` — parameter-grid resimulation off frozen trade fixtures
- `entry_diagnostics.py` — HTF staleness/EMA-bias/symbol-exclusion diagnostic tooling
- `portfolio_overlap.py` — post-hoc portfolio-concurrency-cap diagnostic
- `exchanges.py` — public perp market data (Binance/MEXC/Bitget fallback, or BloFin-only), copied over from `cloude-bot` since it's the one module both repos share

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m unittest discover -p "test_*.py" -v
```

Live (paper-trading dry run by default):

```bash
KRYPTIC_DRY_RUN=true DATA_SOURCE=blofin ACCOUNT_EQUITY=10000 python3 live_runner.py
```

Backtest:

```bash
python3 backtest_engine.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --months 6 --export-xlsx report.xlsx
```

Both entrypoints need real outbound network access to the exchange REST/WS
endpoints — see each file's module docstring for what has and hasn't been
smoke-tested against a live connection.

## Provenance

Split from `cloude-bot` at the point where its own test suite (`test_engine.py`,
`test_live_runner.py`, `test_multi_strategy.py`, `test_backtest_engine.py`,
`test_entry_ladder.py`, `test_grid_resimulation.py`, `test_regime_filter.py`,
`test_risk_manager.py`, `test_entry_diagnostics.py`, `test_portfolio_overlap.py`)
all passed. History was not carried over — this repo starts from a single
initial commit of the current KRYPTIC file set.
