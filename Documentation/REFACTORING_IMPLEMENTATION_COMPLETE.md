# Time Series as Default Signal Generator - Implementation Status
**Date**: 2025-11-06  
**Status**: 🔴 **BLOCKED – 2025-11-15 brutal run regressions**

### Nov 12, 2025 Snapshot
- Signal generator hardened (pandas-safe payload handling + provenance decision context); see logs/ts_signal_demo.json for real BUY/SELL output.
- Checkpoint metadata persistence uses Path.replace, eliminating [WinError 183] when multiple checkpoints save on Windows.
- Validator backfill script now bootstraps sys.path, so nightly automation hits the same code paths exercised by the brutal suite.

### 🚨 2025-11-15 Brutal Run Findings (blocking)
- `logs/pipeline_run.log:16932-17729` and `sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"` confirmed the SQLite datastore is corrupted, so none of the TS-first routing claims in this document can be validated until `DatabaseManager._connect` treats `"database disk image is malformed"` like `"disk i/o error"` (reset/mirror) and the file is rebuilt.
- `logs/pipeline_run.log:2272-2279, 2624, 2979, 3263, 3547, …` show Stage 7 failing on every ticker with `ValueError: The truth value of a DatetimeIndex is ambiguous` because `scripts/run_etl_pipeline.py:1755-1764` evaluates `mssa_result.get('change_points') or []`. Despite the Nov‑09 regression update, the crash persists and the pipeline logs “Generated forecasts for 0 ticker(s)”.
- Immediately afterwards the visualization hook throws `FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'` (lines 2626, 2981, …), so there are no dashboard artefacts to accompany this refactor.
- Pandas/statsmodels warning spam remains unresolved (`forcester_ts/forecaster.py:128-136` Period round-trip; `_select_best_order` in `forcester_ts/sarimax.py:136-183` keeps unconverged grids), contradicting the hardening narrative above.
- `scripts/backfill_signal_validation.py:281-292` still calls `datetime.utcnow()` with sqlite’s default converters, generating Python 3.12 deprecation warnings (`logs/backfill_signal_validation.log:15-22`) each time the nightly validation job runs.

**Mandatory fix list**
1. Recover/rebuild `data/portfolio_maximizer.db` and extend `DatabaseManager._connect` so `"database disk image is malformed"` hits the reset/mirror path.
2. Patch `scripts/run_etl_pipeline.py:1755-1764` to copy the MSSA `DatetimeIndex` into a list before iterating, rerun the forecasting stage, and confirm Stage 8 receives TS forecasts.
3. Drop the unsupported `axis=` argument when calling `FigureBase.autofmt_xdate()` so visualization artefacts exist again.
4. Replace the deprecated Period coercion + tighten the SARIMAX grid to eliminate the warning storm.
5. Update `scripts/backfill_signal_validation.py` with timezone-aware timestamps and sqlite adapters before re-enabling nightly validation.

---

## 🎉 Executive Summary

The refactoring to make **Time Series ensemble the DEFAULT signal generator** with **LLM as fallback/redundancy** has been **IMPLEMENTED**. All critical components are in place and integrated into the pipeline. **⚠️ ROBUST TESTING REQUIRED** before considering this complete. The TS ensemble now feeds `forecaster.evaluate(...)` regression metrics (RMSE / sMAPE / tracking error) into SQLite so the router can enforce performance-based fallbacks.

> **2025-11-09 Regression Update**  
> - `models/time_series_signal_generator.py` now converts GARCH volatility series to scalars and records HOLD provenance timestamps, eliminating the “truth value of a Series is ambiguous” crash reported by monitoring/integration tests.  
> - Tests executed under `simpleTrader_env`:  
>   - `pytest tests/models/test_time_series_signal_generator.py -q`  
>   - `pytest tests/integration/test_time_series_signal_integration.py::TestTimeSeriesForecastingToSignalIntegration::test_forecast_to_signal_flow -vv`

---

## 🟡 Implemented Components (Testing Required)

### 1. Core Signal Generation 🟡
- **`models/time_series_signal_generator.py`** (350 lines)
  - Converts Time Series forecasts to trading signals
  - Calculates confidence scores, risk metrics, and actions
  - Supports all Time Series models (SARIMAX, SAMOSSA, GARCH, MSSA-RL)

### 2. Signal Routing 🟡
- **`models/signal_router.py`** (250 lines)
  - Routes signals with Time Series as PRIMARY
  - LLM as FALLBACK when Time Series unavailable
  - Supports redundancy mode for validation

### 3. Signal Adapter 🟡
- **`models/signal_adapter.py`** (200 lines)
  - Unified signal interface for backward compatibility
  - Converts between Time Series and LLM signal formats
  - Ensures downstream consumers work with both signal types

### 4. Database Schema 🟡
- **`etl/database_manager.py`** (Updated)
  - New `trading_signals` table for unified signal storage
  - Supports both Time Series and LLM signals
  - `save_trading_signal()` method for unified persistence

### 5. Pipeline Integration 🟡
- **`scripts/run_etl_pipeline.py`** (Updated)
  - New stage: `time_series_signal_generation` (Stage 8)
  - New stage: `signal_router` (Stage 9)
  - Time Series forecasting runs before signal generation
  - LLM signals serve as fallback/redundancy

### 6. Configuration 🟡
- **`config/signal_routing_config.yml`** (Created)
  - Feature flags for routing behavior
  - Thresholds for signal generation
  - Fallback trigger configuration

### 7. Package Structure 🟡
- **`models/__init__.py`** (Updated)
  - Exports all signal generation and routing components
  - Clean package interface

---

## 📊 Implementation Details

### Pipeline Flow (After Refactoring)

```
1. Data Extraction
2. Data Validation
3. Data Preprocessing
4. Data Storage
5. Time Series Forecasting ← PRIMARY SIGNAL SOURCE
   ├─ SARIMAX forecasts
   ├─ SAMOSSA forecasts
   ├─ GARCH volatility
   ├─ MSSA-RL change-points
   └─ Ensemble forecast bundle
6. Time Series Signal Generation ← NEW
   ├─ Convert forecasts to signals
   ├─ Calculate confidence scores
   ├─ Calculate risk scores
   └─ Determine actions (BUY/SELL/HOLD)
7. Signal Router ← NEW
   ├─ Route Time Series signals (PRIMARY)
   ├─ Route LLM signals (FALLBACK)
   └─ Combine/reconcile signals
8. LLM Market Analysis (fallback/redundancy)
9. LLM Signal Generation (fallback/redundancy)
10. LLM Risk Assessment (fallback/redundancy)
```

### Database Schema

**New Table: `trading_signals`**
```sql
CREATE TABLE trading_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_date DATE NOT NULL,
    signal_timestamp TIMESTAMP,
    action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL', 'HOLD')),
    source TEXT NOT NULL CHECK(source IN ('TIME_SERIES', 'LLM', 'HYBRID')),
    model_type TEXT,
    confidence REAL CHECK(confidence BETWEEN 0 AND 1),
    entry_price REAL NOT NULL,
    target_price REAL,
    stop_loss REAL,
    expected_return REAL,
    risk_score REAL,
    volatility REAL,
    reasoning TEXT,
    provenance TEXT,  -- JSON string
    validation_status TEXT DEFAULT 'pending',
    ...
    UNIQUE(ticker, signal_date, source, model_type)
)
```

### Signal Flow

1. **Time Series Forecasting** generates forecast bundles
2. **Time Series Signal Generator** converts forecasts to signals
3. **Signal Router** routes signals:
   - PRIMARY: Time Series signals
   - FALLBACK: LLM signals (if TS unavailable)
   - REDUNDANCY: Both (if enabled)
4. **Signal Adapter** normalizes signals for downstream consumers
5. **Database Manager** persists to unified `trading_signals` table

---

## 🔧 Configuration

### Signal Routing Config (`config/signal_routing_config.yml`)

```yaml
signal_routing:
  time_series_primary: true  # DEFAULT: true
  llm_fallback: true          # DEFAULT: true
  llm_redundancy: false       # Run both for validation
  
  time_series:
    confidence_threshold: 0.55
    min_expected_return: 0.02
    max_risk_score: 0.7
    use_volatility_filter: true
```

---

## ⚠️ Remaining Tasks

### High Priority
1. **Unit Tests** 🟡 WRITTEN - ROBUST TESTING REQUIRED
   - `tests/models/test_time_series_signal_generator.py` (300 lines, 15 tests) - **NEEDS EXECUTION & VALIDATION**
   - `tests/models/test_signal_router.py` (250 lines, 12 tests) - **NEEDS EXECUTION & VALIDATION**
   - `tests/models/test_signal_adapter.py` (150 lines, 11 tests) - **NEEDS EXECUTION & VALIDATION**

2. **Integration Tests** 🟡 WRITTEN - ROBUST TESTING REQUIRED
   - `tests/integration/test_time_series_signal_integration.py` (400 lines, 12 tests) - **NEEDS EXECUTION & VALIDATION**
   - End-to-end pipeline tests (forecast → signal → routing → database) - **NEEDS VALIDATION**
   - Signal routing validation - **NEEDS VALIDATION**
   - Database persistence tests (unified trading_signals table) - **NEEDS VALIDATION**

### Medium Priority
3. **Documentation Updates** ⏳ IN PROGRESS
   - Update all roadmap documents
   - Update architecture diagrams
   - Create migration guide

4. **Performance Benchmarks** ⏳ PENDING
   - Signal generation latency
   - Routing overhead
   - Database query performance

---

## 🎯 Success Criteria

- [🟡] Time Series signals generated by default in pipeline - **IMPLEMENTED, TESTING REQUIRED**
- [🟡] LLM signals used only as fallback - **IMPLEMENTED, TESTING REQUIRED**
- [🟡] Unified signal storage in database - **IMPLEMENTED, TESTING REQUIRED**
- [🟡] Backward compatibility maintained - **IMPLEMENTED, TESTING REQUIRED**
- [🟡] Configuration system in place - **IMPLEMENTED, TESTING REQUIRED**
- [🟡] Comprehensive test coverage (38 unit tests + 12 integration tests = 50 total) - **WRITTEN, NEEDS EXECUTION**
- [🟡] Documentation synchronized - **UPDATED**
- [🟡] Integration tests written - **NEEDS EXECUTION & VALIDATION**
- [ ] Performance benchmarks met - **PENDING**
- [ ] Robust end-to-end testing with real data - **REQUIRED**
- [ ] Production validation - **REQUIRED**

---

## 📝 Usage Example

```python
# Pipeline automatically generates Time Series signals
python scripts/run_etl_pipeline.py \
    --tickers AAPL MSFT \
    --start 2024-01-01 --end 2024-06-30 \
    --config config/pipeline_config.yml

# Signals are saved to trading_signals table
# Time Series signals are PRIMARY
# LLM signals are FALLBACK (if enabled)
```

---

## 🔄 Next Steps

1. **Testing**: Write comprehensive unit and integration tests
2. **Documentation**: Update all documentation files
3. **Performance**: Benchmark signal generation and routing
4. **Validation**: Run end-to-end tests with real data
5. **Monitoring**: Add metrics and logging for signal routing

---

**Last Updated**: 2025-11-06  
**Status**: 🟡 **IMPLEMENTED - ROBUST TESTING REQUIRED**  
**Next Review**: After robust testing and validation complete

