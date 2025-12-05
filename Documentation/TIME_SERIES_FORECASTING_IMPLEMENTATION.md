# Time Series Forecasting Implementation - Complete
**Refactoring to Time Series as Default Signal Generator**

**Date**: 2025-12-04 (Updated)  
**Status**: 🔴 **PARTIALLY BLOCKED – 2025-11-15 brutal run exposed regressions; structural issues fixed and brutal harness green, global quant validation still RED**  
**Refactoring Status**: See `Documentation/REFACTORING_STATUS.md` for detailed progress. For how TS-first signals feed NAV-centric risk buckets and the Taleb barbell shell (with LLM as capped fallback), see `Documentation/NAV_RISK_BUDGET_ARCH.md` and `Documentation/NAV_BAR_BELL_TODO.md`.
**Baseline**: SAMOSSA is the canonical Time Series baseline for regression metrics and ensemble comparisons; SARIMAX is retained as a secondary candidate/fallback when SAMOSSA metrics are unavailable.

### 2025-12-03 Delta (diagnostic mode + invariants)
- DIAGNOSTIC_MODE/TS/EXECUTION relax TS thresholds (confidence=0.10, min_return=0, max_risk=1.0, volatility filter off), disable quant validation, and allow PaperTradingEngine to size at least 1 share; LLM latency guard is bypassed in diagnostics and `volume_ma_ratio` now guards zero/NaN volume.
- Numeric/scaling invariants and dashboard/quant health tests pass in `simpleTrader_env` (`tests/forcester_ts/test_ensemble_and_scaling_invariants.py`, `tests/forcester_ts/test_metrics_low_level.py`, dashboard payload + quant health scripts).
- Diagnostic reduced-universe run (MTN, SOL, GC=F, EURUSD=X; cycles=1; horizon=10; cap=$25k) executed 4 trades with PnL -0.06%, updated `visualizations/dashboard_data.json`; positions: long MTN 10, short SOL 569, short GC=F 1, short EURUSD=X 792; quant_validation fail_fraction 0.932 (<0.98) and negative_expected_profit_fraction 0.488 (<0.60).

### 2025-12-04 Delta (TS quant gates + LLM guardrails)
- `models/time_series_signal_generator.TimeSeriesSignalGenerator` now treats quant validation as a **hard gate** for Time Series trades: when the per-signal `quant_profile.status` is `FAIL` and diagnostic toggles are off, BUY/SELL actions are demoted to HOLD while retaining full provenance (`quant_validation` attached to the signal) and JSONL logging via `logs/signals/quant_validation.jsonl`. Thresholds come from `config/quant_success_config.yml` (profit_factor, win_rate, min_expected_profit, etc.).
- `config/signal_routing_config.yml` and the generator’s defaults are aligned on a **0.3% minimum expected return** (`min_expected_return=0.003`), so TS signals must clear a realistic edge after friction before routing/execution; diagnostic modes still reduce this floor to zero for stress runs.
- `scripts/run_auto_trader.py` now enforces an **LLM readiness gate**: LLM fallback is only enabled when `data/llm_signal_tracking.json` reports at least one validated signal; otherwise the router runs in TS-only mode even if `--enable-llm` is passed. Diagnostic runs bypass this guard so LLM behaviour can still be exercised during experiments.

### 2025-11-12 Hardening Notes
- models/time_series_signal_generator.py now normalises pandas/NumPy payloads before evaluation, eliminating the "truth value of a Series is ambiguous" crash and stamping decision context (expected return, confidence, risk, volatility) into provenance. logs/ts_signal_demo.json captures a SELL signal produced directly from SQLite OHLCV data.
- Checkpoint metadata writes use Path.replace, so repeated Windows runs no longer fail with [WinError 183] when successive checkpoints are saved.
- scripts/backfill_signal_validation.py injects the repo root into sys.path, allowing Task Scheduler or the brutal suite to invoke the script from any working directory without ModuleNotFoundError.
- Stack reference: Time-series dependencies must stay within the Tier-1 baseline documented in `Documentation/QUANT_TIME_SERIES_STACK.md` (reuse the YAML/JSON AI-companion snippets there when provisioning new agents or CI containers).
- `bash/comprehensive_brutal_test.sh` now runs in **Time Series-first** mode by default; export `BRUTAL_ENABLE_LLM=1` only if you need to benchmark the legacy LLM fallback. This keeps the brutal gate aligned with the TS-first mandate captured in this document.

## Condensed “practical recipe” for SARIMAX/SAMOSSA stability

1. **Keep preprocessing enabled** – every price series is interpolated (`method="time"`), de-NaN’d, and optionally log-transformed after scaling. This maintains the correlation structure while improving numerical conditioning.
2. **Attach explicit frequency hints** – we preserve the inferred `freqstr`/`inferred_freq` on `series.attrs["_pm_freq_hint"]`. When no hint exists but the median gap is ~1 day, we explicitly use `"B"` (business day) and log the decision.
3. **Run with constraints first, relax only on demand** – the primary fit uses `enforce_stationarity=True` and `enforce_invertibility=True`. If convergence fails, we retry with relaxed constraints, emitting a warning event so the operator knows a fallback occurred.
4. **Keep the order grid data-aware** – `_exceeds_parameter_budget` plus conservative `max_p/max_q/max_P/max_Q` caps ensure each candidate has enough observations. High-order seasonal MA terms are never allowed to consume short series.
5. **Handle zeros explicitly** – if `log_transform=True` and we encounter non-positive values, we add the smallest possible positive shift Δ (recorded in attrs + model summary) before logging. Otherwise users can keep `log_transform=False`.
6. **Scale tiny/huge series, don’t shift arbitrarily** – `_scale_series` multiplies by a factor between 1/1000 and 1000 so statsmodels sees stable magnitudes without corrupting the data.
7. **Treat invertibility warnings as diagnostics** – they describe the solver’s path, not necessarily the final model. We forward them to `logs/warnings/warning_events.log` and continue when the solver converges.

### 🚨 2025-11-15 Brutal Run Regression
- `logs/pipeline_run.log:2272-2279, 2624, 2979, 3263, 3547, …` show the MSSA serialization block (`scripts/run_etl_pipeline.py:1755-1764`) still raises `ValueError: The truth value of a DatetimeIndex is ambiguous` after ~90 inserts per ticker, contradicting the hardening claim above. Every ticker finishes with “Generated forecasts for 0 ticker(s)” so Stage 8 has nothing to route.
- `logs/pipeline_run.log:16932-17729` together with `sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"` confirm the database is corrupted (`database disk image is malformed`, “rowid … out of order/missing from index”), so the persisted SARIMAX/SAMOSSA/MSSA rows referenced later in this document are now invalid. `DatabaseManager._connect` must treat this error like `"disk i/o error"` (reset/mirror) before re-running the stage.
- The visualization hook fails immediately afterwards with `FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'` (lines 2626, 2981, …), so forecast dashboards are not being generated.
- Pandas/statsmodels warnings still flood the log because `forcester_ts/forecaster.py:128-136` forces a deprecated `PeriodIndex` round-trip and `_select_best_order` in `forcester_ts/sarimax.py:136-183` keeps unconverged parameter grids, undermining the “ValueWarning-free” narrative above. *(Nov 16 update: these warnings are now captured automatically in `logs/warnings/warning_events.log`, so follow that file for regression analysis even when console output stays quiet.)*
- `scripts/backfill_signal_validation.py:281-292` continues to use `datetime.utcnow()` and sqlite’s default converters, causing the deprecation warnings documented in `logs/backfill_signal_validation.log:15-22`.

**Blocking actions**
1. Recover/rebuild `data/portfolio_maximizer.db` and update `DatabaseManager._connect` so `"database disk image is malformed"` reuses the disk-I/O recovery path.
2. Replace `change_points = mssa_result.get('change_points') or []` with logic that copies the `DatetimeIndex` into a list without boolean coercion, rerun `python scripts/run_etl_pipeline.py --stage time_series_forecasting`, and confirm Stage 8 consumes the ensemble outputs.
3. Remove the unsupported `axis=` argument when calling `FigureBase.autofmt_xdate()` so dashboards exist again.
4. Replace the deprecated Period coercion and narrow the SARIMAX search grid to eliminate the warning storm and improve convergence.
5. Modernize `scripts/backfill_signal_validation.py` with timezone-aware timestamps + sqlite adapters before re-enabling nightly validation/backfills.

### ✅ 2025-11-16 Interpretability & Telemetry Upgrade
- `forcester_ts/instrumentation.py` now captures per-model timing, configuration, diagnostic artifacts, and data snapshots (shape, window, missing %, statistical moments). `TimeSeriesForecaster.forecast()` embeds this report under `instrumentation_report` and (when `ensemble_kwargs.audit_log_dir` or the `TS_FORECAST_AUDIT_DIR` env var is set) writes JSON audits to disk.
- Each SARIMAX/SAMOSSA/MSSA/GARCH fit/forecast phase is wrapped in the instrumentation context manager so change-points, orders, and information-criteria become searchable logs aligned with `Documentation/QUANT_TIME_SERIES_STACK.md` guidance on interpretable AI. Regression benchmarking (`compute_regression_metrics` → RMSE, sMAPE, tracking error) is logged alongside the dataset diagnostics, and the comprehensive dashboard prints this metadata directly on the figure so visual evidence matches the dataset actually processed.

### ✅ 2025-11-18 SARIMAX Convergence Hardening
- Guided by the regression notes in `Documentation/BRUTAL_TEST_README.md`, `SYSTEM_ERROR_MONITORING_GUIDE.md`, and `CRITICAL_REVIEW.md`, the SARIMAX forecaster now rescales every series into the statsmodels “1–1000” range before fitting, suppressing the DataScale warnings that polluted the brutal logs and improving optimizer stability.
- `_select_best_order` enforces a data-per-parameter budget (mirroring the checkpointing/logging guardrails) and stops the grid search after repeated non-converged fits, so the warning recorder referenced in `Documentation/CHECKPOINTING_AND_LOGGING.md` no longer fills with redundant combinations.
- Frequency hints are stored—but no longer forced onto the pandas index—preventing the `PeriodIndex` coercion that previously triggered ValueWarnings and cascaded into the non-convergence pattern seen in `Documentation/implementation_checkpoint.md`.
- SAMoSSA now mirrors the `Documentation/SAMOSSA_IMPLEMENTATION_CHECKLIST.md`: time-series inputs are interpolated/normalised, Page matrices enforce \(1 < L \leq \sqrt{T}\), HSVT outputs are rescaled to the original units, and residuals are passed through an AutoReg fallback so deterministic + AR components are forecast independently as demanded by the checklist.
- Nov 18 update: SARIMAX/SAMoSSA both use native `Series.ffill()/bfill()` (no deprecated `fillna(method=...)`), default to Business-day frequencies when none can be inferred, and carry those indices through AutoReg + statsmodels fit/forecast paths—eliminating the `ValueWarning`/`UserWarning` spam recorded in the brutal run logs while staying aligned with `Documentation/QUANT_TIME_SERIES_STACK.md`.


---

## ✅ Implementation Summary

The time-series stack now comprises **SARIMAX**, **GARCH**, and the newly promoted **SAMOSSA (Seasonal Adaptive Moving-Window SSA)** modules, all coordinated by the unified forecaster and persisted through the ETL pipeline. The SAMOSSA work follows the mathematical and implementation blueprint documented in `SAMOSSA_algorithm_description.md`, `SAMOSSA_INTEGRATION.md`, and `mSSA_with_RL_changPoint_detection.json`, laying the groundwork for reinforcement-learning interventions and CUSUM change-point alarms while keeping production gating rules intact.

### 🔄 2025-11-09 Wiring Update (Phase 5.4b)
- `forcester_ts/` is now the canonical home for **SARIMAX**, **GARCH**, **SAMOSSA**, and **MSSA-RL**. `etl/time_series_forecaster.py` remains as a thin compatibility shim, so dashboards, notebooks, and ETL all import the same implementations.
- `TimeSeriesForecaster` now records frequency hints without forcing pandas to coerce the index into a fixed `PeriodIndex`, eliminating the `ValueWarning` spam called out in `Documentation/CHECKPOINTING_AND_LOGGING.md` and keeping SARIMAX aligned with the log-stream guardrails.
- `models/time_series_signal_generator.py` now consumes GARCH volatility series safely (scalar conversion prevents the `The truth value of a Series is ambiguous` crash) and stamps HOLD provenance with ISO timestamps. This fix restores Time Series signal generation inside the monitoring job so `llm_signal_backtests` stops reporting **NO_DATA**.
- Targeted regression tests executed under `simpleTrader_env`:
  - `pytest tests/models/test_time_series_signal_generator.py -q`
  - `pytest tests/integration/test_time_series_signal_integration.py::TestTimeSeriesForecastingToSignalIntegration::test_forecast_to_signal_flow -vv`
- Monitoring + backfill context:
  - `scripts/monitor_llm_system.py` ingests the same SQLite regression metrics and now surfaces latency plus `llm_signal_backtests` summaries (written to `logs/latency_benchmark.json`).
  - `schedule_backfill.bat` replays nightly validation so the Time Series ensemble always has fresh metrics before routing signals (Task Scheduler registration still pending—see `NEXT_TO_DO.md`).

### 🎯 **ARCHITECTURAL REFACTORING IN PROGRESS**

**Current State**: Time Series ensemble drives signal generation (routing + monitoring live); pipeline stage reordering + nightly validation still hardening.

**Target State**: Time Series ensemble remains the **DEFAULT signal generator**, with LLM as **fallback/redundancy**, after Stage 5/6 promotion inside `scripts/run_etl_pipeline.py`.

**Status**: 🟡 **HARDENING** – Core components created and routed, Stage 5 promotion + scheduler registration in progress.

See `Documentation/REFACTORING_STATUS.md` for complete status and critical issues.

### 📝 Operational Logging & Checkpoints (2025‑11‑11)
- `forcester_ts/forecaster.py` now records structured events for every model fit/forecast phase (start → success/failure) and emits them through the standard logging pipeline, satisfying the guardrails in `AGENT_INSTRUCTION.md`, `CHECKPOINTING_AND_LOGGING.md`, and `BRUTAL_TEST_README.md`.
- Each model module (SARIMAX, SAMOSSA, MSSA‑RL, GARCH) reports dataset size, selected hyperparameters, and diagnostics; failures are caught per component so the ensemble keeps running while the event stream captures the root cause.
- Ensemble metadata now includes `model_events` + `model_errors`, which the brutal test harness stores under `logs/brutal/results_*`, making it trivial to trace caching/forecast decisions back to the exact model invocation.

---

## 📦 Components Implemented

### 1. SARIMAX Forecasting Module ✅
- **File**: `etl/time_series_forecaster.py` (via `forcester_ts/sarimax.py`)
- **Highlights**:
  - Auto-order selection (AIC/BIC grid search with stationarity tests).
  - Seasonal period detection and exogenous support.
  - Ljung–Box and Jarque–Bera diagnostics for residual governance.
  - 95% confidence interval generation for mean forecasts.

### 2. GARCH Volatility Modelling ✅
- **File**: `etl/time_series_forecaster.py` (via `forcester_ts/garch.py`)
- **Highlights**:
  - Supports GARCH/EGARCH/GJR-GARCH with configurable distributions.
  - Returns-based fit with variance/volatility horizon forecasts.
  - AIC/BIC surfaced for monitoring dashboards.

### 3. SAMOSSA Forecasting Module ✅
- **File**: `etl/time_series_forecaster.py` (via `forcester_ts/samossa.py`)
- **Highlights**:
  - Builds Hankel/Page matrices for SSA decomposition (`Y = F + E`) and retains leading components until ≥90% energy is captured (TruncatedSVD heuristic).
  - Supports configurable window length, retained component count, residual ARIMA orders, and maximum forecast horizon (defaults in `config/forecasting_config.yml`).
  - Optional residual ARIMA `(p,d,q)` with seasonal `(P,D,Q,s)` structure to model stochastic components, matching the design in `mSSA_with_RL_changPoint_detection.json`.
  - Emits deterministic forecasts with diagonal-averaged reconstruction, residual forecasts, explained-variance diagnostics, and confidence intervals sized by residual variance.
  - Provides hooks for CUSUM-based change-point scoring and future Q-learning policy interventions (per `SAMOSSA_algorithm_description.md`) once Phase B RL gating criteria are satisfied.

### 4. Unified Forecasting Interface ✅
- **Package Layout**: `forcester_ts/` bundles the production forecasters (`sarimax.py`, `garch.py`, `samossa.py`, `mssa_rl.py`) plus the unified coordinator (`forecaster.py`).
- **Backwards Compatibility**: `etl/time_series_forecaster.py` re-exports the public API so existing ETL imports remain valid while sharing logic with dashboards and notebooks.
- **Highlights**:
  - `TimeSeriesForecaster` now orchestrates SARIMAX, GARCH, SAMOSSA, and the new MSSA-RL change-point forecaster.
  - Produces hybrid mean forecasts by blending deterministic models (SARIMAX + SAMOSSA/MSSA-RL) while retaining GARCH volatility for risk sizing.
  - Respects per-model guardrails (e.g., SAMOSSA `max_forecast_steps`, MSSA window sizing) to prevent over-extension.
  - Delivers per-model diagnostics (orders, EVR, change points, Q-learning table) for governance checks and statistical validation.
  - Exposes `TimeSeriesForecaster.evaluate()` so walk-forward tests and dashboards can compute RMSE, sMAPE, and tracking error without re-implementing metrics.

### 5. Time Series Signal Generator 🆕 **NEW - REFACTORING**
- **File**: `models/time_series_signal_generator.py` (350 lines)
- **Status**: ✅ Created, ⏳ Pipeline integration pending
- **Purpose**: Convert time series forecasts to trading signals (DEFAULT signal generator)
- **Highlights**:
  - Converts ensemble forecasts (SARIMAX, SAMOSSA, GARCH, MSSA-RL) to actionable trading signals
  - Calculates confidence scores based on model agreement, forecast strength, and diagnostics
  - Calculates risk scores based on volatility, confidence intervals, and expected returns
  - Determines actions (BUY/SELL/HOLD) with configurable thresholds
  - Calculates target prices and stop losses
  - Provides comprehensive reasoning and provenance metadata

### 6. Signal Router 🆕 **NEW - REFACTORING**
- **File**: `models/signal_router.py` (250 lines)
- **Status**: ✅ Created, ⏳ Pipeline integration pending
- **Purpose**: Route signals with Time Series as PRIMARY, LLM as FALLBACK
- **Highlights**:
  - Time Series ensemble is DEFAULT signal source
  - LLM serves as fallback when Time Series unavailable or fails
  - Supports redundancy mode (run both for validation)
  - Feature flags for gradual rollout
  - Maintains backward compatibility with existing signal consumers
  - Unified signal interface for downstream components

#### Regression Metrics & Backtesting ✅
- **File**: `forcester_ts/metrics.py`
- **Why**: Continuous forecasts need regression-grade validation (classification-style confusion matrices do not apply).
- **Metrics**:
  - **RMSE** – square-root of mean squared residuals.
  - **sMAPE** – symmetric MAPE, robust when prices hover near zero.
  - **Tracking Error** - standard deviation of residuals (proxy for portfolio tracking error).
  - **Directional Accuracy** - hit-rate on the sign of forecasted vs realised returns; trading-aligned metric that now feeds ensemble confidence and TS health checks.
- **Workflow**:
  1. Call `forecaster.forecast(...)` to cache the horizon.
  2. Once you have realised prices for the same index, call `forecaster.evaluate(actual_series)`; the returned dict includes the metrics above plus `n_observations`.
  3. Metrics flow into SQLite (`time_series_forecasts.regression_metrics`) and dashboards via `etl/dashboard_loader.py`, so ensemble weighting can blend AIC/EVR with realised performance and directional edge.
  4. One-sided variance-ratio tests (a pragmatic Diebold–Mariano proxy) screen models before the ensemble grid-search finalises weights.

#### Ensemble & GPU Enhancements ⚙️
- **Heuristics**: `derive_model_confidence` now mixes information criteria, realised metrics, and F-tests to prefer models whose residual volatility is statistically lower than the SARIMAX baseline.
- **Change-point boosts**: When MSSA-RL detects dense, recent structural breaks (≤7 trading days), its confidence score is boosted so regime-aware forecasts dominate during stress.
- **GPU assist**: `MSSARLConfig.use_gpu` can leverage local CuPy accelerators to parallelise the mSSA SVD step, mirroring the best practices adopted in recent SSA research pipelines.

### 7. Database Integration ✅
- **File**: `etl/database_manager.py`
- **Schema**:
  ```sql
  CREATE TABLE time_series_forecasts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ticker TEXT NOT NULL,
      forecast_date DATE NOT NULL,
      model_type TEXT NOT NULL CHECK(model_type IN ('SARIMAX', 'GARCH', 'SAMOSSA', 'MSSA_RL', 'COMBINED')),
      forecast_horizon INTEGER NOT NULL,
      forecast_value REAL NOT NULL,
      lower_ci REAL,
      upper_ci REAL,
      volatility REAL,
      model_order TEXT,
      aic REAL,
      bic REAL,
      diagnostics TEXT,
      regression_metrics TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(ticker, forecast_date, model_type, forecast_horizon)
  )
  ```
- **Migration Guard**: Automatic schema upgrade replays existing rows into the expanded CHECK constraint so legacy data remains intact.
- **API**: `DatabaseManager.save_forecast()` accepts SAMOSSA payloads (model metadata + explained variance) in addition to SARIMAX/GARCH.

**⚠️ REFACTORING NOTE**: New unified `trading_signals` table required for Time Series signals. See `Documentation/REFACTORING_STATUS.md` Issue 3.

### 8. Pipeline Integration 🟡 **REFACTORING IN PROGRESS**
- **File**: `scripts/run_etl_pipeline.py`
- **Current State**: Time Series forecasting runs as Stage 7, AFTER LLM signal generation
- **Target State**: Time Series forecasting moves to Stage 5, BEFORE signal generation
- **New Stages Required**:
  - Stage 5: Time Series Forecasting (moved up)
  - Stage 6: Time Series Signal Generation (NEW)
  - Stage 7: Signal Router (NEW)
  - Stage 8: LLM Signal Generation (fallback/redundancy)

**Current Stage 7 (`time_series_forecasting`)**:
  - Reads the SAMOSSA block from `config/forecasting_config.yml`.
  - Instantiates `TimeSeriesForecaster` with SARIMAX/GARCH/SAMOSSA/MSSA-RL configs.
  - Persists SARIMAX, GARCH, SAMOSSA, MSSA-RL, and ensemble (`COMBINED`) results using `DatabaseManager.save_forecast`.
  - Logs diagnostic metadata (model orders, AIC/BIC, explained variance) for observability.
  - Optional visualization hook (`pipeline.visualization.auto_dashboard`) renders forecast and signal dashboards via `etl/dashboard_loader.py` and `TimeSeriesVisualizer`.

**⚠️ REFACTORING REQUIRED**: See `Documentation/REFACTORING_STATUS.md` Issue 1 for detailed pipeline integration plan.

### 8.1 Dashboard Health Quick-Check (NEW)
- **File**: `scripts/check_dashboard_health.py`
- **Usage**:
  ```bash
  python scripts/check_dashboard_health.py \
    --dashboard-path visualizations/dashboard_data.json \
    --config-path config/forecaster_monitoring.yml
  ```
- Reads the latest `dashboard_data.json` emitted by `scripts/run_auto_trader.py` and prints:
  - Run metadata (run_id, timestamp, tickers, cycles).
  - Forecaster health vs thresholds from `config/forecaster_monitoring.yml`:
    - `profit_factor_ok`, `win_rate_ok`, `rmse_ok`, plus the underlying metrics and limits.
  - A small per‑ticker summary based on the last few executed signals in the dashboard:
    - Trade count, win rate, simple profit factor, and any PF/WR alerts based on the same monitoring config.
- This keeps **brutal CLIs**, **hyperopt**, and **dashboard inspection** aligned on a single set of TS health thresholds without requiring manual JSON inspection.

### 9. Configuration ✅
- **File**: `config/forecasting_config.yml`
- **Options**:
  - SARIMAX: `max_p`, `max_d`, `max_q`, seasonal orders, trend flags.
  - GARCH: `p`, `q`, volatility model, distribution choice.
  - SAMOSSA: `window_length`, `n_components`, `use_residual_arima`, `arima_order`, `seasonal_order`, `min_series_length`, `max_forecast_steps`, `reconstruction_method`.
  - Global: `default_forecast_horizon`, combined forecast toggles.

**🆕 NEW**: `config/signal_routing_config.yml` required for signal routing. See `Documentation/REFACTORING_STATUS.md` Issue 4.

### 10. Tests ✅ (Base Implementation)
- **File**: `tests/etl/test_time_series_forecaster.py`
- **Coverage**:
  - SARIMAX & GARCH initialisation, fit, and forecast regression checks.
  - SAMOSSA fit/forecast validation with synthetic 180-day series.
  - Unified forecaster with and without SAMOSSA enabled.
  - Guardrails for insufficient history, missing data handling, and residual modelling fallbacks.

**⚠️ REFACTORING NOTE**: New tests required for `TimeSeriesSignalGenerator` and `SignalRouter`. See `Documentation/REFACTORING_STATUS.md` Issue 5.

---

## 🚀 Pipeline Flow

### Production Flow (Time Series-First, Current)
```
1. Data Extraction
2. Data Validation
3. Data Preprocessing
4. Data Storage / Splitting
5. Time Series Forecasting      (PRIMARY SIGNAL SOURCE)
   - SARIMAX mean forecasts
   - GARCH volatility forecasts
   - SAMOSSA SSA-based forecasts
   - MSSA-RL change-point forecasts
   - Ensemble/combined bundles
6. Time Series Signal Generation (convert forecasts to actions + confidence)
7. Signal Router                (routes TS first, invokes LLM fallback only when required)
8. LLM Market Analysis          (optional fallback diagnostics)
9. LLM Signal Generation        (optional fallback signals)
10. LLM Risk Assessment         (optional fallback risk summary)
11. Signal Validation & Execution
12. Persistence to `trading_signals`
```

### Legacy Flow (LLM-First, Deprecated)
```
1. Data Extraction
2. Data Validation
3. Data Preprocessing
4. Data Storage / Splitting
5. LLM Market Analysis          (optional)
6. LLM Signal Generation        (optional) - PRIOR PRIMARY SIGNAL SOURCE (retired)
7. LLM Risk Assessment          (optional)
8. Time Series Forecasting      - Enhanced (separate, not used for signals)
   - SARIMAX mean forecasts
   - GARCH volatility forecasts
   - SAMOSSA SSA-based forecasts
   - MSSA-RL change-point forecasts
   - Hybrid mean + diagnostics (AIC/BIC/EVR/CUSUM hooks)
9. Persistence to `time_series_forecasts`
```

> ✅ **Refactoring Status**: Core components created. Pipeline integration complete for TS-first flow; LLM stages are fallback only. See Documentation/REFACTORING_STATUS.md for detailed progress and critical issues.
## ?? Usage Examples

### Current Usage (Forecasting Only)
```python
from etl.time_series_forecaster import TimeSeriesForecaster
import pandas as pd

prices = pd.read_csv("data/samples/aapl.csv", index_col="date", parse_dates=True)["Close"]

forecaster = TimeSeriesForecaster(
    samossa_config={
        "enabled": True,
        "window_length": 40,
        "n_components": 6,
        "min_series_length": 120,
        "max_forecast_steps": 21,
    }
)

forecaster.fit(data=prices)
forecast_bundle = forecaster.forecast(steps=14)

samossa = forecast_bundle["samossa_forecast"]["forecast"]
sarimax = forecast_bundle["mean_forecast"]["forecast"]
volatility = forecast_bundle["volatility_forecast"]["volatility"]
hybrid = forecast_bundle["combined"].get("hybrid_mean")

# Optional: backtest metrics once you have realised prices
holdout = prices.iloc[-14:]
metrics = forecaster.evaluate(holdout)
print("Ensemble RMSE:", metrics["ensemble"]["rmse"])
```

### New Usage (Signal Generation) 🆕
```python
from etl.time_series_forecaster import TimeSeriesForecaster
from models.time_series_signal_generator import TimeSeriesSignalGenerator
from models.signal_router import SignalRouter
import pandas as pd

# Step 1: Generate forecasts
prices = pd.read_csv("data/samples/aapl.csv", index_col="date", parse_dates=True)["Close"]
forecaster = TimeSeriesForecaster()
forecaster.fit(data=prices)
forecast_bundle = forecaster.forecast(steps=30)

# Step 2: Generate signal from forecast
current_price = prices.iloc[-1]
signal_generator = TimeSeriesSignalGenerator(
    confidence_threshold=0.55,
    min_expected_return=0.02,
    max_risk_score=0.7
)

signal = signal_generator.generate_signal(
    forecast_bundle=forecast_bundle,
    current_price=current_price,
    ticker="AAPL",
    market_data=prices
)

print(f"Signal: {signal.action}, Confidence: {signal.confidence:.2f}")
print(f"Expected Return: {signal.expected_return:.2%}")
print(f"Risk Score: {signal.risk_score:.2f}")

# Step 3: Route signal (with LLM fallback if needed)
router = SignalRouter(
    config={
        'time_series_primary': True,
        'llm_fallback': True
    }
)

bundle = router.route_signal(
    ticker="AAPL",
    forecast_bundle=forecast_bundle,
    current_price=current_price,
    market_data=prices
)

print(f"Primary Signal: {bundle.primary_signal['action']}")
if bundle.fallback_signal:
    print(f"Fallback Signal: {bundle.fallback_signal['action']}")
```

### CLI Usage (After Refactoring)
```bash
# Pipeline run with Time Series as default signal generator
python scripts/run_etl_pipeline.py \
    --tickers AAPL,MSFT \
    --include-frontier-tickers \
    --start 2024-01-01 --end 2024-06-30 \
    --config config/pipeline_config.yml

# Enable LLM redundancy mode (run both TS and LLM)
python scripts/run_etl_pipeline.py \
    --tickers AAPL,MSFT \
    --include-frontier-tickers \
    --enable-llm \
    --config config/pipeline_config.yml

# Force LLM-only mode (legacy behavior)
python scripts/run_etl_pipeline.py \
    --tickers AAPL,MSFT \
    --include-frontier-tickers \
    --enable-llm \
    --config config/pipeline_config.yml \
    --signal-source llm  # New flag (to be implemented)
```

`--include-frontier-tickers` taps `etl/frontier_markets.py` so every multi-ticker training
run carries the curated Nigeria → Bulgaria coverage list referenced in
`Documentation/arch_tree.md`. Keep synthetic mode enabled until data-source specific ticker
suffix mappings for NGX/NSE/BSE are configured.

---

## 📚 Dependencies

```bash
pip install statsmodels arch
```

Optional / recommended for SAMOSSA roadmap:
- `numpy`, `pandas`, `scipy`
- `cupy`, `numba`, `dask` (GPU + parallel SSA per roadmap)
- `loguru` or structured logging (runtime diagnostics)

---

## 🔄 REFACTORING PLAN: Time Series as Default Signal Generator

### Overview

This refactoring shifts the architecture from **LLM-first** to **Time Series-first** signal generation, with LLM serving as fallback/redundancy. This provides:

1. **Deterministic Signals**: Time Series models provide consistent, reproducible signals
2. **Lower Latency**: No LLM inference required for primary signals
3. **Better Performance**: Statistical models often outperform LLM for price prediction
4. **Redundancy**: LLM still available when Time Series fails or needs validation

### Implementation Phases

#### Phase 1: Foundation ✅ COMPLETE
- [x] Create `TimeSeriesSignalGenerator` class
- [x] Create `SignalRouter` class
- [x] Create `models/__init__.py`
- [x] Create status tracking document

#### Phase 2: Pipeline Integration ⏳ IN PROGRESS
- [ ] Refactor `scripts/run_etl_pipeline.py` stage order
- [ ] Move Time Series forecasting before signal generation
- [ ] Add Time Series signal generation stage
- [ ] Integrate Signal Router
- [ ] Update stage dependencies

**Estimated Effort**: 4-6 hours  
**Status**: See `Documentation/REFACTORING_STATUS.md` Issue 1

#### Phase 3: Database & Persistence ⏳ PENDING
- [ ] Design unified `trading_signals` table schema
- [ ] Create migration script
- [ ] Update `DatabaseManager.save_signal()` method
- [ ] Migrate existing `llm_signals` data
- [ ] Update all signal retrieval queries

**Estimated Effort**: 3-4 hours  
**Status**: See `Documentation/REFACTORING_STATUS.md` Issue 3

#### Phase 4: Backward Compatibility ⏳ PENDING
- [ ] Create signal schema adapter
- [ ] Update `execution/paper_trading_engine.py`
- [ ] Update `ai_llm/signal_validator.py`
- [ ] Update `scripts/track_llm_signals.py`
- [ ] Update monitoring dashboards

**Estimated Effort**: 6-8 hours  
**Status**: See `Documentation/REFACTORING_STATUS.md` Issue 6

#### Phase 5: Testing ⏳ PENDING
- [ ] Unit tests for `TimeSeriesSignalGenerator`
- [ ] Unit tests for `SignalRouter`
- [ ] Integration tests for pipeline
- [ ] Backward compatibility tests
- [ ] Performance benchmarks

**Estimated Effort**: 8-10 hours  
**Status**: See `Documentation/REFACTORING_STATUS.md` Issue 5

#### Phase 6: Configuration & Documentation ⏳ PENDING
- [ ] Create `config/signal_routing_config.yml`
- [ ] Update `config/pipeline_config.yml`
- [ ] Update all documentation files
- [ ] Create migration guide
- [ ] Update API documentation

**Estimated Effort**: 4-6 hours  
**Status**: See `Documentation/REFACTORING_STATUS.md` Issues 4 & 7

### Critical Issues

See `Documentation/REFACTORING_STATUS.md` for complete list of critical issues:

1. **Pipeline Integration Implemented – Validation Pending** 🟡  
   Time Series forecasting, signal generation, and routing now execute before any LLM stages in `scripts/run_etl_pipeline.py`, but the refactored flow still needs end-to-end validation.
2. **Signal Schema Mismatch** ⚠️ HIGH
3. **Database Schema Updates Required** ⚠️ HIGH
4. **Configuration System Not Updated** ⚠️ MEDIUM
5. **Testing Infrastructure Missing** ❌ HIGH
6. **Backward Compatibility Not Guaranteed** ⚠️ MEDIUM
7. **Documentation Not Updated** ⚠️ MEDIUM

### Success Criteria

- [ ] Time Series signals generated by default in pipeline
- [ ] LLM signals used only as fallback
- [ ] All existing tests pass
- [ ] New tests cover signal generation and routing
- [ ] Database schema supports both signal types
- [ ] Backward compatibility maintained
- [ ] Documentation updated
- [ ] Performance benchmarks met (<5s total signal generation)

---

## ✅ Verification Checklist

### Base Implementation
- [x] SARIMAX forecaster implemented
- [x] GARCH forecaster implemented
- [x] SAMOSSA forecaster implemented
- [x] Unified forecaster updated for hybrid aggregation
- [x] Database schema migrated to allow `'SAMOSSA'`
- [x] Pipeline stage saves SARIMAX/GARCH/SAMOSSA rows
- [x] Forecasting configuration exposes SAMOSSA controls
- [x] Regression tests cover all three models
- [x] Documentation refreshed
- [x] Security sanitisation preserved around persistence paths

### Refactoring (In Progress)
- [x] TimeSeriesSignalGenerator created
- [x] SignalRouter created
- [x] Status tracking document created
- [x] Pipeline integration order updated (validation pending)
- [ ] Database schema updated
- [ ] Signal schema unified
- [ ] Tests written
- [ ] Configuration updated
- [ ] Documentation synchronized

### Future Enhancements
- [ ] Q-learning intervention layer (scheduled for Phase B RL gate)
- [ ] CUSUM alerts promoted to monitoring dashboards
- [ ] GPU acceleration (CuPy/Numba) for SSA decomposition

---

## 🎯 Status & Next Enhancements

- **Implementation**: ✅ Core models complete  
- **Integration**: ✅ Pipeline + DB wired (forecasting only)  
- **Testing**: ✅ Regression suite updated (forecasting only)  
- **Latest Validation Attempt**: `bash/comprehensive_brutal_test.sh` now completes end-to-end in `simpleTrader_env` (see `logs/brutal/results_20251204_190220/`), with profit-critical, ETL, Time Series forecasting, signal routing, integration, and security suites all passing. The previous `Broken pipe` timeout in the Time Series block and the missing `tests/etl/test_data_validator.py` file have been remediated. The remaining gating item for this document is global quant validation health, which is still RED (FAIL_fraction above `max_fail_fraction=0.90`) per `scripts/check_quant_validation_health.py` and `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md`.
- **Documentation**: ✅ Current (forecasting only)  
- **Refactoring**: 🟡 **IN PROGRESS** - See `Documentation/REFACTORING_STATUS.md`

### Immediate Next Steps (Refactoring)
1. **Pipeline & Autonomous Loop Validation** (CRITICAL) - Run `scripts/run_etl_pipeline.py` (with/without `--enable-llm`) and `scripts/run_auto_trader.py` to prove the reordered stages route TS signals before LLM fallback.
2. **Database Schema** (HIGH) - Create unified signal storage
3. **Testing** (HIGH) - Execute the pending Time Series signal generator/router test suites and log results.
4. **Backward Compatibility** (MEDIUM) - Ensure existing code works
5. **ETL Error Remediation** (CRITICAL) - Resolve `DataStorage.train_validation_test_split()` signature mismatch, zero-fold CV `ZeroDivisionError`, SQLite `disk I/O` errors, and missing parquet engines surfaced in `logs/errors/errors.log` so forecasting can run on live data feeds.

### Phase B Focus (Per `SAMOSSA_INTEGRATION.md`)
1. Promote CUSUM-based change-point scores into monitoring.
2. Add RL policy loop (Q-learning) for regime-aware interventions after paper-trading validation meets MVS/PRS gates (`QUANTIFIABLE_SUCCESS_CRITERIA.md`).
3. Evaluate GPU acceleration (CuPy/Numba) once profiling identifies CPU bottlenecks (`SYSTEM_STATUS_2025-10-22.md` performance budgets).

---

## 📚 Related Documentation

- **Refactoring Status**: `Documentation/REFACTORING_STATUS.md` - Detailed progress and critical issues
- **Implementation Summary**: `Documentation/FORECASTING_IMPLEMENTATION_SUMMARY.md`
- **SAMOSSA Algorithm**: `Documentation/SAMOSSA_algorithm_description.md`
- **SAMOSSA Integration**: `Documentation/SAMOSSA_INTEGRATION.md`
- **Unified Roadmap**: `Documentation/UNIFIED_ROADMAP.md`
- **Stub Implementation**: `Documentation/STUB_IMPLEMENTATION_PLAN.md`

---

**Last Updated**: 2025-11-06  
**Status**: ✅ Forecasting Complete | 🟡 Refactoring In Progress  
**Next Review**: After pipeline integration complete
