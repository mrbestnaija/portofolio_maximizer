# Time Series Forecasting Implementation - Complete
**Refactoring to Time Series as Default Signal Generator**

**Date**: 2025-11-06 (Updated)  
**Status**: ✅ **FULLY IMPLEMENTED & WIRED** → 🟡 **REFACTORING IN PROGRESS**  
**Refactoring Status**: See `Documentation/REFACTORING_STATUS.md` for detailed progress

---

## ✅ Implementation Summary

The time-series stack now comprises **SARIMAX**, **GARCH**, and the newly promoted **SAMOSSA (Seasonal Adaptive Moving-Window SSA)** modules, all coordinated by the unified forecaster and persisted through the ETL pipeline. The SAMOSSA work follows the mathematical and implementation blueprint documented in `SAMOSSA_algorithm_description.md`, `SAMOSSA_INTEGRATION.md`, and `mSSA_with_RL_changPoint_detection.json`, laying the groundwork for reinforcement-learning interventions and CUSUM change-point alarms while keeping production gating rules intact.

### 🎯 **ARCHITECTURAL REFACTORING IN PROGRESS**

**Current State**: Time Series forecasting exists but is NOT used for signal generation. LLM is the primary signal source.

**Target State**: Time Series ensemble becomes **DEFAULT signal generator**, with LLM as **fallback/redundancy**.

**Status**: 🟡 **INITIALIZED** - Core components created, pipeline integration pending.

See `Documentation/REFACTORING_STATUS.md` for complete status and critical issues.

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
  - **Tracking Error** – standard deviation of residuals (proxy for portfolio tracking error).
- **Workflow**:
  1. Call `forecaster.forecast(...)` to cache the horizon.
  2. Once you have realised prices for the same index, call `forecaster.evaluate(actual_series)`; the returned dict includes the metrics above plus `n_observations`.
  3. Metrics flow into SQLite (`time_series_forecasts.regression_metrics`) and dashboards via `etl/dashboard_loader.py`, so ensemble weighting can blend AIC/EVR with realised performance.
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

## 🔄 Pipeline Flow

### Current Flow (Before Refactoring)
```
1. Data Extraction
2. Data Validation
3. Data Preprocessing
4. Data Storage / Splitting
5. LLM Market Analysis          (optional)
6. LLM Signal Generation        (optional) ← PRIMARY SIGNAL SOURCE
7. LLM Risk Assessment          (optional)
8. Time Series Forecasting      ★ Enhanced (separate, not used for signals)
   ├─ SARIMAX mean forecasts
   ├─ GARCH volatility forecasts
   ├─ SAMOSSA SSA-based forecasts
   ├─ MSSA-RL change-point forecasts
   └─ Hybrid mean + diagnostics (AIC/BIC/EVR/CUSUM hooks)
9. Persistence to `time_series_forecasts`
```

### Target Flow (After Refactoring)
```
1. Data Extraction
2. Data Validation
3. Data Preprocessing
4. Data Storage / Splitting
5. Time Series Forecasting      ★ MOVED UP, PRIMARY SIGNAL SOURCE
   ├─ SARIMAX mean forecasts
   ├─ GARCH volatility forecasts
   ├─ SAMOSSA SSA-based forecasts
   ├─ MSSA-RL change-point forecasts
   └─ Ensemble forecast bundle
6. Time Series Signal Generation ★ NEW
   ├─ Convert forecasts to signals
   ├─ Calculate confidence scores
   ├─ Calculate risk scores
   └─ Determine actions (BUY/SELL/HOLD)
7. Signal Router                 ★ NEW
   ├─ Route Time Series signals (PRIMARY)
   ├─ Route LLM signals (FALLBACK)
   └─ Combine/reconcile signals
8. LLM Market Analysis          (fallback/redundancy)
9. LLM Signal Generation        (fallback/redundancy)
10. LLM Risk Assessment         (fallback/redundancy)
11. Signal Validation & Execution
12. Persistence to `trading_signals` (unified table)
```

> ℹ️ **Refactoring Status**: Core components created. Pipeline integration in progress. See `Documentation/REFACTORING_STATUS.md` for detailed progress and critical issues.

---

## 📊 Usage Examples

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
    --tickers AAPL MSFT \
    --start 2024-01-01 --end 2024-06-30 \
    --config config/pipeline_config.yml

# Enable LLM redundancy mode (run both TS and LLM)
python scripts/run_etl_pipeline.py \
    --tickers AAPL MSFT \
    --enable-llm \
    --config config/pipeline_config.yml

# Force LLM-only mode (legacy behavior)
python scripts/run_etl_pipeline.py \
    --tickers AAPL MSFT \
    --enable-llm \
    --config config/pipeline_config.yml \
    --signal-source llm  # New flag (to be implemented)
```

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

1. **Pipeline Integration Not Started** ❌ CRITICAL
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
- [ ] Pipeline integration complete
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
- **Documentation**: ✅ Current (forecasting only)  
- **Refactoring**: 🟡 **IN PROGRESS** - See `Documentation/REFACTORING_STATUS.md`

### Immediate Next Steps (Refactoring)
1. **Pipeline Integration** (CRITICAL) - Integrate Time Series signal generation into pipeline
2. **Database Schema** (HIGH) - Create unified signal storage
3. **Testing** (HIGH) - Write comprehensive tests
4. **Backward Compatibility** (MEDIUM) - Ensure existing code works

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
