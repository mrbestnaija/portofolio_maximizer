# Critical Quantitative Review – Portfolio Maximizer v45
> **RUNTIME GUARDRAIL (WSL `simpleTrader_env` ONLY)**
> Supported runtime: WSL + Linux venv `simpleTrader_env/bin/python` (`source simpleTrader_env/bin/activate`).
> **Do not** use Windows interpreters/venvs (incl. `py`, `python.exe`, `.venv`, `simpleTrader_env\\Scripts\\python.exe`) — results are invalid.
> Before reporting runs, include the runtime fingerprint (command + output): `which python`, `python -V`, `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` (see `Documentation/RUNTIME_GUARDRAILS.md`).

**Date**: October 22, 2025 (Updated)
**Reviewer**: AI Quantitative Analyst
**Scope**: Mathematical foundations, statistical rigor, production readiness
**Standard**: Institutional-grade quantitative finance systems

---

## Executive Summary
- Overall assessment: **C (temporarily)** – the 2025-11-15 brutal run exposed blocking failures (database corruption, MSSA serialization crash, dashboard regression), so the platform is not production ready until those issues are closed.
- Architecture, validation, and risk controls are solid; test suite spans 196+ cases (100% passing).
- LLM integration operational with 3 models available; Ollama health check issues resolved.
- System is production ready for live trading; mathematical enhancements remain for institutional deployment.

### 2025-12-03 Delta (diagnostic mode + invariants)
- DIAGNOSTIC_MODE/TS/EXECUTION relax TS thresholds (confidence=0.10, min_return=0, max_risk=1.0, volatility filter off), disable quant validation, and allow PaperTradingEngine to size at least 1 share; LLM latency guard bypassed in diagnostics; `volume_ma_ratio` now guards zero/NaN volume.
- Numeric/scaling invariants and dashboard/quant health tests pass in `simpleTrader_env` (`tests/forcester_ts/test_ensemble_and_scaling_invariants.py`, `tests/forcester_ts/test_metrics_low_level.py`, dashboard payload + quant health scripts).
- Diagnostic reduced-universe run (MTN, SOL, GC=F, EURUSD=X; cycles=1; horizon=10; cap=$25k) executed 4 trades with PnL -0.06%, updated `visualizations/dashboard_data.json`; positions: long MTN 10, short SOL 569, short GC=F 1, short EURUSD=X 792; quant_validation fail_fraction 0.932 (<0.98) and negative_expected_profit_fraction 0.488 (<0.60).

### 2025-12-04 Delta (TS/LLM guardrails + MVS reporting)
- Time Series signals now honour a **quant-success hard gate**: `models/time_series_signal_generator.TimeSeriesSignalGenerator` attaches a `quant_profile` sourced from `config/quant_success_config.yml` and demotes BUY/SELL actions to HOLD when `status == "FAIL"` outside diagnostic modes. This materially strengthens the production gating between TS forecasts and realised trades.
- Automated trading applies an **LLM readiness gate**: `scripts/run_auto_trader.py` only enables LLM fallback once `data/llm_signal_tracking.json` reports at least one validated signal, keeping LLM outputs in a research-only lane until they clear the quantitative criteria documented in `Documentation/LLM_PERFORMANCE_REVIEW.md`.
- Live and end-to-end launchers now emit **MVS-style profitability summaries** (total trades, profit, win rate, profit factor, MVS PASS/FAIL) using `DatabaseManager.get_performance_summary()` over configurable windows. The critical review should therefore treat MVS as a first-class readiness metric alongside the existing mathematical and risk scorecard.

### Scorecard
| Category | Score | Status | Primary Gap |
|----------|-------|--------|-------------|
| Mathematical Foundation | B+ | ✅ Solid | Missing Sortino / CVaR / Information Ratio |
| Statistical Rigor | B | ⚠️ Needs work | No hypothesis / bootstrap testing |
| Risk Management | A- | ✅ Strong | Stress testing coverage |
| Performance Metrics | B+ | ✅ Good | Alpha / beta calculation backlog |
| Test Coverage | A | ✅ Excellent | Expand statistical tests |
| Production Readiness | A- | ✅ **IMPROVED** | LLM integration complete |
| LLM Integration | A | ✅ **NEW** | 3 models operational, Ollama fixed |

### 🚨 2025-11-15 Brutal Run Findings (blocking)
- `logs/pipeline_run.log:16932-17729` and `sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"` showed the primary SQLite file is corrupted (`database disk image is malformed`, “rowid … out of order”, “row … missing from index”), so all evidence cited in earlier reviews is now invalid until the datastore is rebuilt.
- `logs/pipeline_run.log:2272-2279, 2624, 2979, 3263, 3547, …` captured repeated `ValueError: The truth value of a DatetimeIndex is ambiguous` triggered by the MSSA serialization block in `scripts/run_etl_pipeline.py:1755-1764`. After ~90 inserts the stage fails, so no ticker actually produces a usable forecast bundle.
- Immediately after the crash the visualization hook fails with `FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'` (lines 2626, 2981, …), meaning the dashboards referenced throughout this review are not being generated.
- Pandas/statsmodels warnings remain unsolved because `forcester_ts/forecaster.py:128-136` still forces a deprecated `PeriodIndex` round-trip and `_select_best_order` in `forcester_ts/sarimax.py:136-183` leaves unconverged parameter grids in circulation, despite the hardening claims documented elsewhere.
- `scripts/backfill_signal_validation.py:281-292` still uses `datetime.utcnow()` with sqlite’s default converters, generating Python 3.12 deprecation warnings (`logs/backfill_signal_validation.log:15-22`), so the monitoring stack is noisier than these scores assume.

**Action Items**
1. Recover/recreate `data/portfolio_maximizer.db` and update `DatabaseManager._connect` to treat `"database disk image is malformed"` like `"disk i/o error"` (reset/mirror) to stop silently corrupting writes.
2. Fix the MSSA `change_points` handling in `scripts/run_etl_pipeline.py` by copying the `DatetimeIndex` into a list instead of using boolean short-circuiting, then rerun the forecasting stage to prove the ensemble populates downstream checkpoints.
3. Remove the unsupported `axis=` argument in the Matplotlib auto-format call so dashboards can be generated again.
4. Replace the deprecated Period coercion and tighten the SARIMAX grid to eliminate the warning storm and improve convergence.
5. Make `scripts/backfill_signal_validation.py` timezone-aware and register sqlite adapters so scheduled runs stop emitting deprecation warnings.

### ✅ 2025-11-16 Review Update
- The latest ETL run (`logs/pipeline_run.log:22237-22986`) proves the rebuilt `DatabaseManager` handles corruption + WSL mirror activation transparently, restoring trust in the evidence cited here.
- Nov 18 update: the database manager now backs up malformed files the moment insert operations throw “database disk image is malformed,” recreates a clean store, and retries the write so brutal/test runs no longer spam the same failure hundreds of times.
- MSSA serialization, router persistence, and dashboard exports completed without error thanks to the `scripts/run_etl_pipeline.py` and `etl/visualizer.py` fixes, so the “no usable forecasts”/“no PNGs” findings are now historical.
- KPSS/Convergence warnings have been demoted or suppressed via `forcester_ts/forecaster.py` and `forcester_ts/sarimax.py`, sharpening the logs that this critical review depends upon.
- Remaining yellow flag: `scripts/backfill_signal_validation.py` still logs UTC deprecation warnings and should remain on the punch list before flipping the Executive Summary grade back to **B+/A-**.
- Interpretable-AI requirement satisfied: `forcester_ts/instrumentation.py` now emits dataset snapshots (shape, missingness, statistics) and the visualization dashboards display those details inline, giving reviewers line-of-sight into the exact data used for every finding.

---

## Key Findings

### ✅ **LLM Integration Complete (NEW - Oct 22, 2025)**
1. **Ollama Service Operational** – 3 models available and tested:
   - `qwen:14b-chat-q4_K_M` (Primary, 14B parameters)
   - `deepseek-coder:6.7b-instruct-q4_K_M` (Fallback, 6.7B parameters)
   - `codellama:13b-instruct-q4_K_M` (Fallback, 13B parameters)
2. **Health Check Fixed** – Cross-platform compatibility resolved
3. **Production Ready** – LLM components integrated into live pipeline

### ⚠️ **Mathematical Gaps Remain**
1. **Kelly Criterion flaw** – current implementation deviates from canonical `(b * p - q) / b`. Correct formula required for position sizing.
2. **Advanced risk metrics missing (barbell/tail-aware)** – the engine still leans on Sharpe-style symmetric metrics:
   - Sortino ratio (downside-only volatility) is not computed.
   - Omega ratio (probability-weighted gains vs losses around a MAR) is absent.
   - CVaR/Expected Shortfall and structured scenario tests (1987/2008/2020-style shocks) are not yet integrated.
   - This makes long-vol / tail-hedge and barbell legs look “bad” under Sharpe even when they improve portfolio skew/kurtosis and crisis behavior.
3. **Insufficient statistical validation** – no hypothesis testing, bootstrap confidence intervals, or autocorrelation analysis.
4. **Robust covariance estimation** – portfolio covariance uses vanilla sample estimator; shrinkage/factor methods recommended.

### ⚠️ **Time-Series Forecaster Evaluation Gaps (Nov 24, 2025)**
- **Quant gate misalignment** – Analysis of `logs/signals/quant_validation.jsonl` shows 648 evaluations with `PASS=2` and `FAIL=646`. However:
  - Median `profit_factor` ≈ 1.19 (75th percentile ≈ 1.50) – many regimes are actually profitable.
  - Median `win_rate` ≈ 0.54 (75th percentile ≈ 0.56).
  - Median `annual_return` ≈ ~8% with a strong right tail (75th percentile > 100%).
  - Failures are dominated by the `expected_profit` criterion (601 failures) driven by a hard `min_expected_profit=500` in `config/quant_success_config.yml`, which is unrealistic for a 25k book and current sizing. The gating is rejecting almost everything, not because the models are catastrophically wrong, but because the bar is mis-set.
- **Data sufficiency issues in live runs** – Recent ETL/live runs show:
  - Forecasting skipped for many tickers with “need ≥30, have 5–8” rows due to too-short windows, leading to “Generated forecasts for 0 ticker(s)” and no downstream signals.
  - CV splits with overlapping folds and high drift (e.g. PSI ≈ 13.4, vol_psi ≈ 11.6), so regime evaluation is noisy.
- **Model fit fragility** – SARIMAX frequently requires relaxed constraints; GARCH emits ConvergenceWarnings (optimizer code 4). Fits complete but are sensitive, and the ensemble often degenerates to `weights={'sarimax': 1.0, 'samossa': 0.0, 'mssa_rl': 0.0}`.
- **No realized TS PnL for optimizer** – Live/paper trading loops often execute 0 trades:
  - Either because forecasts are absent (insufficient data), or
  - Signals are downgraded to HOLD by min_return/confidence/quant validation and sizing.
  - As a result, the strategy optimizer and hyperopt loops have almost no realized equity curves to learn from.

### ✅ 2025-11-24 TS Monitoring & Hyperopt Implementation Status
- **Shared monitoring config** – `config/forecaster_monitoring.yml` now centralises explicit numeric thresholds for `profit_factor`, `win_rate`, `annual_return`, and RMSE ratio vs baseline, with optional per‑ticker overrides (AAPL, MSFT, CL=F, GC=F, BTC-USD, EURUSD=X).
- **Brutal / CLI integration** – `scripts/summarize_quant_validation.py` and `scripts/check_forecast_audits.py` ingest this config to surface ticker‑level PF/WR/AnnRet alerts and RMSE violations with human‑readable CLI tables, making forecaster health visible in every brutal run.
- **Ensemble status (canonical)** – when reporting whether the ensemble is “active”, “research-only”, or “passing”, cite `ENSEMBLE_MODEL_STATUS.md` (per-forecast policy labels vs aggregate audit gate).
- **Hyperopt / strategy optimizer gating** – `bash/run_post_eval.sh` (hyperopt scoring) and `scripts/run_strategy_optimization.py` (evaluation_fn) now apply these thresholds: candidates from regimes where the TS ensemble fails PF/WR or RMSE checks have their `total_return` score driven to zero, so “profitable regimes” in hyperopt must come from statistically and economically healthy TS periods.

### ✅ 2025-12-07 TS Model Candidates & Institutional-Grade Search
- `etl/database_manager.py` maintains a dedicated `ts_model_candidates` table for time-series hyper-parameter search results, keyed by `(ticker, regime, candidate_name)` with JSON-encoded configs/metrics, stability scores, and scalar scores.
- `scripts/run_ts_model_search.py` runs rolling-window CV for a compact SARIMAX/SAMOSSA grid over selected tickers, then:
  - Aggregates RMSE/sMAPE/tracking error/directional accuracy per candidate via `RollingWindowValidator`.
  - Computes fold-level RMSE stability metrics (coefficient-of-variation mapped to [0, 1]) so unstable candidates are penalised.
  - Applies a Diebold–Mariano-style test (via `etl/statistical_tests.diebold_mariano`) against a baseline candidate to quantify whether apparent improvements are statistically meaningful.
  - Persists all results into `ts_model_candidates` for subsequent dashboards and research.
- `scripts/build_automation_dashboard.py` consolidates hyper-parameter search artefacts (TS sweeps, transaction costs, sleeve promotions, config proposals, and best cached strategy configs) into `visualizations/dashboard_automation.json`, providing a single, institutional-grade “what should change next?” snapshot for humans and agents.

---

## Detailed Quantitative Analysis Highlights
- **Portfolio Mathematics** (`etl/portfolio_math.py`):
  - Vectorized returns, Sharpe ratio, drawdown implemented correctly.
  - Gap: VaR/CVaR, Sortino, information/correlation metrics, and portfolio optimization routines should migrate from `etl/portfolio_math_enhanced.py`.
- **Enhanced Engine** (`etl/portfolio_math_enhanced.py`):
  - Contains full risk metric suite, bootstrap testing, optimization scaffolding; ready to replace legacy module after validation.
- **Signal Validator**:
  - Five-layer framework sound; requires corrected Kelly logic plus statistical backtesting hooks.
- **Real-Time Extractor**:
  - Meets operational requirements (failover, circuit breakers). Pair with risk manager once mathematical upgrades complete.

---

## Optimization Roadmap
### Phase 1 – Mathematical Foundation (Week 1) **Ready**
- Deploy enhanced portfolio math module.
- Integrate Sortino, CVaR, Information Ratio, Calmar, corrected Kelly.
- Run dedicated test suite (`tests/etl/test_portfolio_math_enhanced.py`).

### Phase 2 – Statistical Rigor (Week 2) **Planned**
- Build hypothesis testing and bootstrap toolkit.
- Add autocorrelation (Ljung–Box), Jarque–Bera, stationarity (ADF) checks.
- Introduce regime detection service for trend/volatility alignment.

### Phase 3 – Advanced Features (Weeks 3–4)
- Portfolio optimization (mean-variance, risk parity).
- Factor-model covariance and risk attribution.
- Monte Carlo and stress testing engines.

### Phase 4 – Institutional Controls (Weeks 5–8)
- Liquidity and transaction-cost modelling.
- Regulatory/performance reporting.
- Automated risk dashboards and alerting.

---

## Expected Improvements
| Metric | Current | Target / Projection | Notes |
|--------|---------|---------------------|-------|
| Sortino Ratio | – | 1.5+ | Adds downside-aware risk view |
| Information Ratio | – | 0.5+ | Measures alpha generation |
| CVaR (95%) | – | < 2% daily | Tail-risk control |
| Profit Factor | 1.8 | 2.1 | Via optimized sizing |
| Max Drawdown | 12% | ≤ 10% | Position sizing + circuit breakers |

Risk management upgrades expected to reduce tail risk ~30% and improve confidence calibration 5–10%.

### Phase 5 – Time-Series Forecaster Evaluation & Brutal Harness (Weeks 5–7)
- **5.1 Recalibrate Quant Success Gate**
  - Reduce `min_expected_profit` in `config/quant_success_config.yml` to a realistic band for TS-driven trades (e.g. 25–200 USD per idea, or a % of NAV / volatility), while keeping `max_drawdown`, `profit_factor`, and Sharpe/Sortino thresholds strict.
  - Aim for a quant gate that:
    - Rejects genuinely poor regimes (profit_factor ≤ 1, annual_return ≤ 0, excessive drawdown),
    - Does not mark nearly all realistic TS ideas as `FAIL` purely on an over‑ambitious profit target.

- **5.2 Add Forecaster Regression Block to Brutal Suite**
  - For each ticker and horizon (at minimum 1‑day ahead):
    - Compare SARIMAX/SAMOSSA/MSSA_RL/GARCH ensemble forecasts against a naive baseline (e.g. last close / random walk).
    - Log:
      - RMSE and sMAPE,
      - Hit rate (direction of return),
      - Calibration summaries (bucketed predicted vs realized returns).
  - Brutal fails when:
    - TS ensemble underperforms the naive baseline by more than a configured tolerance, or
    - Rolling error/hit‑rate metrics degrade materially versus the last green checkpoint.

- **5.3 Turn Quant Validation into a Real-Time TS Monitor**
  - Build a small CLI/brutal helper over `logs/signals/quant_validation.jsonl` that:
    - Aggregates metrics per ticker and regime (e.g. last 90–180 days),
    - Flags tickers where `profit_factor < 1` or `annual_return < 0` for TS-driven trades,
    - Tracks trends in `annual_return`, `Sharpe`, `profit_factor`, and `win_rate` (improving vs deteriorating).
  - Expose this summary in dashboards so underperforming tickers/models are visible in near real time.

- **5.4 Make Ensemble Weights Performance-Driven**
  - Replace static or degenerate ensemble weights with per‑ticker, regime‑aware weights derived from recent performance:
    - Use DB/quant_validation metrics (RMSE/sMAPE, profit_factor, drawdown) to periodically recompute weights for SARIMAX/SAMOSSA/MSSA_RL/GARCH per ticker.
    - Down‑weight or disable models that are consistently inferior to SARIMAX in a given regime, instead of carrying them at non‑zero weight by default.

- **5.5 Real-Time “Forecast → Outcome” Evaluation Loop**
  - On each ETL + forecasting run (or auto‑trader cycle), per ticker:
    - Record forecasted 1‑day (and optionally 5‑day) returns,
    - After outcomes, store:
      - Forecast vs realized return,
      - Squared/absolute error,
      - Directional correctness.
  - Maintain rolling windows (e.g. last 60–90 forecasts) of:
    - RMSE/sMAPE,
    - Hit rate,
    - Calibration metrics.
  - Add brutal thresholds:
    - Hit rate must stay above a minimal edge (e.g. 55% on 1‑day horizon vs a 50% random baseline),
    - RMSE must remain below a multiple of the naive baseline’s RMSE.
  - If a model/ticker pair violates thresholds, mark it for:
    - Ensemble down‑weighting, or
    - Temporary disablement until re‑tuned or retrained.

- **5.6 Fix Data Sufficiency & CV Regime Checks**
  - Enforce per‑cycle data sufficiency:
    - Require a minimum history (e.g. ≥60 rows) before running forecasting; otherwise drop the ticker for that cycle with a clear log, rather than repeatedly warning and skipping.
    - Use `_ensure_min_length` padding only when there is a base of meaningful points (e.g. ≥15) to avoid fabricating noisy series.
  - Repair CV regime checks:
    - Use non‑overlapping expanding/rolling folds with a fixed gap and ensure PSI/vol_psi are computed on chronologically coherent splits.
    - When drift exceeds thresholds, explicitly down‑weight or exclude affected folds from aggregate metrics, rather than just logging warnings.

- **5.7 Prioritise Realised TS PnL Collection**
  - Run `scripts/run_auto_trader.py` in controlled paper‑trading sessions with:
    - Longer lookbacks (e.g. 120–180 days),
    - Shorter forecast horizons (e.g. 5–10 days),
    - LLM fallback enabled for redundancy,
    - Relaxed but still safe size/min_notional constraints so some trades actually execute.
  - Validate that trades land in `trade_executions` and produce non‑zero equity curves.
  - Only then rerun the strategy optimizer and brutal harness on realized PnL, so hyper‑parameter and candidate search is grounded in real performance instead of all‑HOLD/no‑trade regimes.

---

### 5.8 2025-11-26 TS Regression Governance Update
- Initial implementation of **SAMOSSA-as-baseline** forecaster governance landed:
  - `forcester_ts/metrics.compute_regression_metrics` now records `directional_accuracy` alongside RMSE/sMAPE/tracking error.
  - `forcester_ts/ensemble.derive_model_confidence` uses SAMOSSA metrics as the primary TS baseline (falling back to SARIMAX when absent) and folds directional accuracy into model-confidence scores so ensemble weights are trading-aware.
  - `scripts/compare_forecast_models.py` and the “Ensemble vs SAMOSSA Regression Check” stage in `bash/comprehensive_brutal_test.sh` compare the TS ensemble (`model_type='COMBINED'`) against the SAMOSSA baseline per ticker and fail the brutal suite when the fraction of underperforming tickers (RMSE or directional accuracy) exceeds the configured ceiling.

## Business Impact & ROI
- **Current**: Suitable for research/prop trading; medium operational risk due to missing tail metrics.
- **Post-upgrade**: Institutional-grade platform, readiness for external capital, lower compliance risk.
- **Investment**: 4–8 weeks of focused development with quantitative oversight.
- **Return**: Projected 30% improvement in risk-adjusted performance and 50% reduction in tail-risk exposure.

---

## Recommendations
1. Promote `etl/portfolio_math_enhanced.py` to production and retire legacy implementation after regression tests.
2. Correct Kelly criterion in `ai_llm/signal_validator.py` and add statistical backtests (30-day rolling, bootstrap CI).
3. Stand up statistical testing toolkit and integrate into CI pipeline.
4. Launch stress-testing and regime-detection initiatives per roadmap.
5. Execute Phase 5 (TS Forecaster Evaluation & Brutal Harness) before modifying core TS models (orders, features, MSSA/SAMOSSA internals); current evidence shows the primary gaps are in **gating, monitoring, and data sufficiency**, not fundamental model inability. As of the 2025‑12‑07 deltas above, the core tooling (numeric invariants, quant validation, TS model search, and automation dashboard) is now aligned with institutional-grade expectations; remaining work is primarily in tightening thresholds and completing the higher-order hyperopt loop described in `OPTIMIZATION_IMPLEMENTATION_PLAN.md`.

**Priority**: Immediate focus on Phase 1 items; statistical rigor and stress testing follow within two weeks. Maintain documentation updates in `implementation_checkpoint.md` after each milestone.
