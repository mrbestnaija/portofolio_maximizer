# Stochastic PnL Optimization Strategy (Non‑Hardcoded Guardrails)

This document defines how PnL optimization must work in this project without hardcoding any specific strategy or parameter set. All optimization is configuration‑driven, stochastic, and regime‑aware.

## Principles

- **No hardcoded strategies**  
  - All knobs (e.g., `min_return`, confidence thresholds, risk caps, model weights, sizing coefficients) are defined in configuration or learned from data, not baked into code as “magic numbers”.  
  - Code only defines *how* to search and evaluate, not *what the final strategy is*.

- **Stochastic, non‑convex optimization**  
  - Treat each strategy configuration as a point on a non‑convex outcome surface (P&L, drawdown, hit rate, etc.).  
  - Use stochastic, gradient‑free methods (random search, evolutionary / bandit style selection, simulated annealing) over a configurable search space, never assuming convexity or a single optimum.

- **Per‑regime and per‑strategy evaluation**  
  - Regimes (bull/bear/sideways, high/low volatility, high/low drift) are detected from existing metrics (GARCH volatility, split drift PSI, volume/volatility drift, etc.).  
  - Each strategy family (TS‑only, TS+LLM fallback, different ensembles, sizing rules) is re‑evaluated per regime; no global “one size fits all” setting.

- **Cached optima, but never frozen**  
  - The best‑performing configuration per strategy+regime is cached for fast use.  
  - Any significant regime change or performance degradation automatically triggers re‑testing and re‑optimization; caches are hints, not permanent truths.

- **Guardrails remain external and stable**  
  - Global safety constraints such as `min_expected_return`, `max_risk_score`, and capital at risk per trade come from configuration and/or `QUANTIFIABLE_SUCCESS_CRITERIA.md`, not from optimization code.  
  - The optimizer is *constrained* by these guardrails; it cannot lower them to chase PnL.

## High‑Level Architecture

### 1. Strategy configuration space (config‑driven)

- New config file (example): `config/strategy_optimization_config.yml`  
  - Defines parameter families and distributions, e.g.:
    - Ensemble weights ranges (SARIMAX / SAMOSSA / MSSA_RL / GARCH).  
    - Sizing scheme variants (Kelly fraction caps, per‑ticker/asset‑class caps, diversification penalties).  
    - Execution options (limit vs market bias, slippage model families).  
  - No single “best” value is embedded in code; only ranges/prior distributions are.

### 2. Optimizer engine (non‑convex search)

- Implemented as a reusable module (e.g., `etl/strategy_optimizer.py`) that:
  - Samples candidate configurations stochastically from the config space.  
  - Runs backtests or reads realized performance from the DB (via `DatabaseManager`) for each candidate.  
  - Scores candidates on a multi‑objective function:  
    - PnL, profit factor, drawdown, hit rate, Sharpe, calibration error (expected vs realized), etc.  
  - Uses simple but robust search strategies:
    - Random search / Latin hypercube for exploration.  
    - Bandit‑style selection or evolutionary updates for exploitation.  
  - Produces a Pareto set or a “current best under constraints” per regime.

### 3. Regime detection and triggers

- Regimes are inferred from existing metrics, for example:
  - Volatility from GARCH forecasts / realized volatility.  
  - Drift metrics from `split_drift_metrics` (PSI, vol_psi, volatility_ratio).  
  - Volume / liquidity shifts from `data_quality_snapshots`.
- Simple, configuration‑driven rules decide regime buckets (e.g., “high vol”, “low vol”); thresholds live in config.
- Triggers to re‑optimize:
  - Regime classification changes.  
  - Performance drops below thresholds defined in `QUANTIFIABLE_SUCCESS_CRITERIA.md` (e.g., win rate, profit factor, max drawdown).  
  - Drift metrics exceed configured tolerances.

### 4. Caching and retrieval

- Results stored in a lightweight cache (DB table or JSON), keyed by:
  - Strategy family / version.  
  - Regime descriptor.  
  - Optimization run timestamp and evaluation dataset.
- The live router / trader reads “best known config for current regime” from this cache.
- If no cache entry exists or the entry is stale/failing metrics, the system falls back to a safe baseline config and optionally schedules a fresh optimization run.

## Interaction With Suggested PnL Enhancements

The earlier PnL‑improvement checklist is implemented *via* this stochastic optimization layer, not by hardcoding:

- **Forecast calibration & model mix**  
  - The optimizer varies ensemble weights and model inclusion (SARIMAX vs MSSA vs GARCH) and scores each mix on calibration error + PnL, subject to the same `min_expected_return` and `max_risk_score`.

- **Position sizing**  
  - Different sizing rules (Kelly caps, risk parity, per‑asset caps) are parameterized; the optimizer searches over these while respecting fixed risk floors and not changing `min_return`/`max_risk`.

- **Diversification / uncorrelated markets**  
  - Basket choice and diversification penalties are part of the search space (e.g., “penalty for correlated exposure”), not hardcoded; the optimizer discovers which allocations improve PnL in a given regime.

- **Execution & trade management**  
  - Slippage models, “don't trade inside noise” filters, and trailing‑stop parameters become tunable knobs in the config space; stochastic search finds combinations that reduce leakage while preserving existing guardrails.

## Evaluation & Governance

- Every optimization run is logged with:
  - Parameter ranges, sampled candidates, evaluation datasets, and outcome metrics.  
  - Regime definition used, and constraints applied.
- Before adopting a new configuration in live or paper trading:
  - It must meet or exceed the success criteria defined in `QUANTIFIABLE_SUCCESS_CRITERIA.md` over a sufficiently long backtest.  
  - It must be validated on data not used for parameter search (hold‑out or forward period).
- The optimization layer itself is tested:
  - Unit tests for sampling, scoring, and caching logic.  
  - Regression tests to ensure guardrail configs cannot be lowered by the optimizer.

## Higher‑Order Hyper‑Parameter Driver (Project‑Wide Default Mode)

Beyond the per‑candidate search space defined in `config/strategy_optimization_config.yml`,
the system treats a small set of *meta‑parameters* as higher‑order hyper‑parameters:

- Evaluation window (`START` / `END` dates used for ETL + backtest),
- Quant success criteria (currently `min_expected_profit`),
- Time Series signal guardrails (`signal_routing.time_series.min_expected_return`).

These are not hardcoded; instead, they are tuned stochastically via a dedicated Bash driver:

- `bash/run_post_eval.sh` orchestrates:
  - `scripts/run_etl_pipeline.py` (regime‑aware ETL + CV drift),
  - `scripts/run_auto_trader.py` (paper‑trading backtest loop),
  - `scripts/run_strategy_optimization.py` (candidate‑level optimizer).
- It samples combinations of:
  - Time windows (e.g., 30–120‑day lookbacks),
  - `min_expected_profit` thresholds (e.g., 500 → 50 USD),
  - `min_expected_return` thresholds (e.g., 2% → 0.3%),
  while holding one dimension constant per trial (partial differencing / hold‑one‑out).

The driver:

- Implements a non‑convex, stochastic bandit‑style search with:
  - Default policy: 30% exploration, 70% exploitation (`HYPEROPT_EXPLORE_PCT=30`),
  - Dynamic adjustment (successful trials shrink explore %, failures expand it).
- Scores each trial using realized `total_profit` over a short evaluation window read from the
  temporary trial database.
- Logs all trials to `logs/hyperopt/hyperopt_<RUN_ID>.log` and re‑runs the best configuration
  as `<RUN_ID>_best` so downstream dashboards treat it as the current regime optimum.

### Integration with Other Entry Points

To make the higher‑order driver the *default orchestration mode* across the project:

- `bash/run_end_to_end.sh`  
  - Reads `HYPEROPT_ROUNDS` from the environment.
  - When `HYPEROPT_ROUNDS > 0`, delegates to `bash/run_post_eval.sh` (passing `TICKERS`,
    `START_DATE`/`END_DATE`) instead of running a single ETL + auto‑trader pass.

- `bash/run_auto_trader.sh`  
  - Also reads `HYPEROPT_ROUNDS`.
  - When `HYPEROPT_ROUNDS > 0`, delegates to `bash/run_post_eval.sh` so ad‑hoc auto‑trader
    runs automatically get higher‑order optimization.

This means that in production or research setups, exporting a single environment variable:

```bash
export HYPEROPT_ROUNDS=5
```

promotes the higher‑order hyper‑parameter search to the default mode for all orchestrators that
run the strategy optimizer stack.

Guardrails remain external and immutable:

- The driver only *tightens* or *loosens* thresholds within ranges that are consistent with
  `QUANTIFIABLE_SUCCESS_CRITERIA.md`.
- It never bypasses max drawdown, risk score, or capital limits enforced by
  `quant_success_config.yml`, `signal_routing_config.yml`, and `DatabaseManager`.

## Current State (2025-11-23)
- Optimizer, CLI, cache, dashboard surface are live and config-driven.
- Regime-aware candidate evaluation now runs a lightweight simulation (`backtesting/candidate_simulator.py`):
  - Loads OHLCV from DB for the regime window.
  - Uses candidate parameters to adjust signal confidence and execution costs.
  - Executes via `PaperTradingEngine` on an isolated in-memory DB to avoid polluting live data.
  - Returns PnL metrics (total_return, profit_factor, win_rate, max_drawdown, total_trades) per candidate; constraints are applied on these metrics.
- Remaining ambition: swap the lightweight simulator for a full replay of historical OHLCV + forecasts -> `SignalRouter`/`TimeSeriesSignalGenerator`/`PaperTradingEngine` with candidate.params mapped into the real stack. This will provide true candidate-specific PnL for non-convex optimization under the existing guardrails.

- 2025-11-23: Wired StrategyOptimizer into bash/run_post_eval.sh and run_strategy_optimization.py with regime-aware evaluation windows and persistent strategy_configs cache; dashboard meta now surfaces best-known strategy params/metrics per regime without hardcoding strategies.

## Regime-Aware Exploration vs Exploitation (NEW)

To keep optimisation PnL-first without overfitting, a lightweight regime / exploration layer now sits on top of the existing paper trading engine:

- `scripts/update_regime_state.py`:
  - Reads realised trades from `trade_executions` via `DatabaseManager`.
  - For each ticker, collects the last *N* trades with non-null `realized_pnl`.
  - If `n_trades < N_min`:
    - Marks the ticker as `mode: exploration`, `state: neutral`.
  - Otherwise computes a simple Sharpe-like score `sharpe_N = mean / (std + ε)` and classifies the regime as:
    - `green` (Sharpe above a positive threshold),
    - `red` (Sharpe below a negative threshold),
    - `neutral` (in between).
  - Writes the result to `config/regime_state.yml` under:
    ```yaml
    regime_state:
      AAPL:
        n_trades: 12
        sharpe_N: 0.85
        mode: exploitation
        state: green
    ```

- `execution/paper_trading_engine.PaperTradingEngine`:
  - Uses `_get_regime_risk_multiplier(ticker)` inside `_calculate_position_size` to scale the standard 2% per-trade risk cap:
    - **Exploration** (`mode == "exploration"`): multiplies the cap by `0.25`, so early trades in low-sample regimes use micro-sizing.
    - **Red regime** (`state == "red"`): multiplies the cap by `0.3`, shrinking risk for sleeves with poor realised Sharpe.
    - **Green regime** (`state == "green"`): multiplies the cap by `1.2`, allowing a modest scale-up in statistically strong sleeves.
    - **Neutral / missing state**: uses `1.0` (no change).

This pattern implements a simple exploration/exploitation policy:
- New or weakly-tested tickers get **more observations** at tiny size instead of being hard-blocked.
- Strong regimes are allowed slightly more capital, consistent with the project’s quantitative success criteria and risk caps.
- The mechanism is fully config-driven; if `config/regime_state.yml` is absent, behaviour falls back to the original fixed 2% cap.
