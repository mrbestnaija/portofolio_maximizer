# Time Series as Default Signal Generator - Refactoring Status
**Project Initialization & Critical Issues**

**Date**: 2025-11-06 (Updated)  
**Status**: 🟡 **IMPLEMENTED - ROBUST TESTING REQUIRED**  
**Priority**: CRITICAL - Architectural refactoring

### Nov 12, 2025 Snapshot
- Signal generator hardened (pandas-safe payload handling + provenance decision context); see logs/ts_signal_demo.json for real BUY/SELL output.
- Checkpoint metadata persistence uses Path.replace, eliminating [WinError 183] when multiple checkpoints save on Windows.
- Validator backfill script now bootstraps sys.path, so nightly automation hits the same code paths exercised by the brutal suite.


---

## 📋 Executive Summary

This document tracks the refactoring of Portfolio Maximizer to use **Time Series ensemble as the DEFAULT signal generator** with **LLM as fallback/redundancy**. This represents a fundamental architectural shift from LLM-first to Time Series-first signal generation.

> Update (Nov 2025): The TS ensemble now writes RMSE / sMAPE / tracking-error metrics to SQLite (`time_series_forecasts.regression_metrics`) via `forecaster.evaluate(...)`, and `signal_router.py` consumes those metrics when deciding whether to accept TS output or fall back to an LLM. See `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md` for the full persistence contract.

> Update (Nov 9, 2025): `scripts/monitor_llm_system.py` now reads those metrics, logs latency benchmarks, and surfaces `llm_signal_backtests`; `schedule_backfill.bat` replays nightly validation (Task Scheduler registration pending). `models/time_series_signal_generator.py` was hardened (volatility scalar conversion + HOLD provenance timestamps) and regression-tested with `pytest tests/models/test_time_series_signal_generator.py -q` plus the targeted integration smoke.

### Current Architecture (Before Refactoring)
```
Pipeline Flow:
1. Data Extraction
2. Data Validation  
3. Data Preprocessing
4. Data Storage
5. LLM Market Analysis (optional)
6. LLM Signal Generation (optional) ← PRIMARY SIGNAL SOURCE
7. LLM Risk Assessment (optional)
8. Time Series Forecasting ← SEPARATE, NOT USED FOR SIGNALS
```

### Target Architecture (After Refactoring)
```
Pipeline Flow:
1. Data Extraction
2. Data Validation
3. Data Preprocessing
4. Data Storage
5. Time Series Forecasting ← MOVED UP, PRIMARY SIGNAL SOURCE
   ├─ SARIMAX forecasts
   ├─ SAMOSSA forecasts
   ├─ GARCH volatility
   ├─ MSSA-RL change-points
   └─ Ensemble signal generation
6. Signal Router ← NEW: Routes TS (primary) + LLM (fallback)
7. LLM Market Analysis (fallback/redundancy)
8. LLM Signal Generation (fallback/redundancy)
9. LLM Risk Assessment (fallback/redundancy)
```

---

## ✅ Initialization Status

### Files Created

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `models/time_series_signal_generator.py` | 🟡 Implemented | ~350 | Converts TS forecasts to trading signals - **TESTING REQUIRED** |
| `models/signal_router.py` | 🟡 Implemented | ~250 | Routes TS (primary) + LLM (fallback) - **TESTING REQUIRED** |
| `Documentation/REFACTORING_STATUS.md` | ✅ Created | This file | Status tracking |

### Files to Create/Modify

| File | Status | Priority | Purpose |
|------|--------|----------|---------|
| `models/__init__.py` | 🟡 Implemented | HIGH | Package initialization - **TESTING REQUIRED** |
| `models/signal_adapter.py` | 🟡 Implemented | HIGH | Signal compatibility layer - **TESTING REQUIRED** |
| `tests/models/test_time_series_signal_generator.py` | 🟡 Written | HIGH | Unit tests written - **ROBUST TESTING REQUIRED** |
| `tests/models/test_signal_router.py` | 🟡 Written | HIGH | Unit tests written - **ROBUST TESTING REQUIRED** |
| `scripts/run_etl_pipeline.py` | 🟡 Updated | CRITICAL | Pipeline integration implemented - **ROBUST TESTING REQUIRED** |
| `etl/database_manager.py` | 🟡 Updated | HIGH | Unified trading_signals table added - **TESTING REQUIRED** |
| `config/signal_routing_config.yml` | 🟡 Created | MEDIUM | Signal routing config - **TESTING REQUIRED** |

---

## 🚨 CRITICAL ISSUES

### Issue 1: Pipeline Integration Not Started 🟡 IMPLEMENTED - TESTING REQUIRED
**Status**: 🟡 Implemented - **ROBUST TESTING REQUIRED**  
**Impact**: Time Series signals now generated in pipeline - **NEEDS VALIDATION**  
**Location**: `scripts/run_etl_pipeline.py`

**Required Changes**:
1. Move `time_series_forecasting` stage BEFORE signal generation
2. Add `time_series_signal_generation` stage after forecasting
3. Integrate `SignalRouter` to route signals
4. Update stage dependencies

**Code Changes Needed**:
```python
# In scripts/run_etl_pipeline.py

# BEFORE: Stage order
# 5. LLM Signal Generation
# 8. Time Series Forecasting

# AFTER: Stage order  
# 5. Time Series Forecasting (moved up)
# 6. Time Series Signal Generation (NEW)
# 7. Signal Router (NEW)
# 8. LLM Signal Generation (fallback/redundancy)
```

**Dependencies**:
- `models/time_series_signal_generator.py` ✅ Created
- `models/signal_router.py` ✅ Created
- Pipeline refactoring ⏳ Pending

**Estimated Effort**: 4-6 hours

---

### Issue 2: Signal Schema Mismatch 🟡 IMPLEMENTED - TESTING REQUIRED
**Status**: 🟡 Resolved with SignalAdapter - **ROBUST TESTING REQUIRED**  
**Impact**: Unified signal interface implemented - **NEEDS VALIDATION**

**Current LLM Signal Schema**:
```python
{
    'ticker': str,
    'action': 'BUY'|'SELL'|'HOLD',
    'confidence': float,
    'reasoning': str,
    'signal_timestamp': str,
    'signal_type': str,
    'llm_model': str,
    'fallback': bool
}
```

**New Time Series Signal Schema**:
```python
{
    'ticker': str,
    'action': 'BUY'|'SELL'|'HOLD',
    'confidence': float,
    'entry_price': float,
    'target_price': float,
    'stop_loss': float,
    'expected_return': float,
    'risk_score': float,
    'reasoning': str,
    'signal_timestamp': str,
    'model_type': str,
    'signal_type': 'TIME_SERIES',
    'volatility': float,
    'provenance': dict
}
```

**Required Actions**:
1. Create unified signal schema adapter
2. Update all downstream consumers:
   - `execution/paper_trading_engine.py`
   - `ai_llm/signal_validator.py`
   - `etl/database_manager.py` (signal persistence)
   - Monitoring dashboards

**Estimated Effort**: 6-8 hours

---

### Issue 3: Database Schema Updates Required 🟡 IMPLEMENTED - TESTING REQUIRED
**Status**: 🟡 Unified trading_signals table created - **ROBUST TESTING REQUIRED**  
**Impact**: Schema implemented - **NEEDS VALIDATION WITH REAL DATA**

**Current Schema** (`llm_signals` table):
- Designed for LLM signals only
- Missing Time Series-specific fields (target_price, stop_loss, expected_return, risk_score, volatility)

**Required Changes**:
```sql
-- Option 1: Extend existing table
ALTER TABLE llm_signals ADD COLUMN target_price REAL;
ALTER TABLE llm_signals ADD COLUMN stop_loss REAL;
ALTER TABLE llm_signals ADD COLUMN expected_return REAL;
ALTER TABLE llm_signals ADD COLUMN risk_score REAL;
ALTER TABLE llm_signals ADD COLUMN volatility REAL;
ALTER TABLE llm_signals ADD COLUMN model_type TEXT DEFAULT 'LLM';
ALTER TABLE llm_signals ADD COLUMN provenance TEXT;  -- JSON

-- Option 2: Create unified signals table (RECOMMENDED)
CREATE TABLE trading_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_date DATE NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL', 'HOLD')),
    source TEXT NOT NULL CHECK(source IN ('TIME_SERIES', 'LLM', 'HYBRID')),
    model_type TEXT,
    confidence REAL,
    entry_price REAL,
    target_price REAL,
    stop_loss REAL,
    expected_return REAL,
    risk_score REAL,
    volatility REAL,
    reasoning TEXT,
    provenance TEXT,  -- JSON
    signal_timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, signal_date, source, model_type)
);
```

**Migration Strategy**:
1. Create new `trading_signals` table
2. Migrate existing `llm_signals` data
3. Update `DatabaseManager.save_signal()` method
4. Update all signal retrieval queries

**Estimated Effort**: 3-4 hours

---

### Issue 4: Configuration System Not Updated 🟡 IMPLEMENTED - TESTING REQUIRED
**Status**: 🟡 signal_routing_config.yml created - **ROBUST TESTING REQUIRED**  
**Impact**: Configuration implemented - **NEEDS VALIDATION**

**Required Configuration** (`config/signal_routing_config.yml`):
```yaml
signal_routing:
  # Primary signal source
  time_series_primary: true  # DEFAULT: true
  
  # Fallback/redundancy
  llm_fallback: true  # DEFAULT: true
  llm_redundancy: false  # Run both for validation
  
  # Time Series model flags
  enable_samossa: true
  enable_sarimax: true
  enable_garch: true
  enable_mssa_rl: true
  
  # Signal generation thresholds
  confidence_threshold: 0.55  # Minimum confidence for signal
  min_expected_return: 0.02   # 2% minimum expected return
  max_risk_score: 0.7         # Maximum risk score
  
  # Fallback triggers
  ts_fallback_triggers:
    - forecast_unavailable
    - confidence_below_threshold
    - model_failure
    - insufficient_data
```

**Estimated Effort**: 1-2 hours

---

### Issue 5: Testing Infrastructure Missing ❌ HIGH
**Status**: No tests for new components  
**Impact**: Cannot verify correctness

**Required Tests**:
1. `tests/models/test_time_series_signal_generator.py`
   - Test signal generation from forecasts
   - Test confidence calculation
   - Test risk score calculation
   - Test action determination
   - Test edge cases (missing data, invalid forecasts)

2. `tests/models/test_signal_router.py`
   - Test Time Series primary routing
   - Test LLM fallback routing
   - Test redundancy mode
   - Test feature flag toggles
   - Test batch routing

3. Integration tests
   - Test pipeline integration
   - Test database persistence
   - Test signal validator compatibility

**Estimated Effort**: 8-10 hours

---

### Issue 6: Backward Compatibility Not Guaranteed ⚠️ MEDIUM
**Status**: Existing code expects LLM signals  
**Impact**: Breaking changes possible

**Affected Components**:
- `execution/paper_trading_engine.py` - Expects LLM signal format
- `ai_llm/signal_validator.py` - Validates LLM signals
- `scripts/track_llm_signals.py` - Tracks LLM signals only
- Monitoring dashboards - Display LLM-specific metrics

**Required Actions**:
1. Create signal adapter/wrapper for backward compatibility
2. Update all consumers to handle both signal types
3. Add feature flag to revert to LLM-only mode
4. Comprehensive integration testing

**Estimated Effort**: 6-8 hours

---

### Issue 7: Documentation Not Updated ⚠️ MEDIUM
**Status**: Documentation still shows LLM-first architecture  
**Impact**: Confusion for developers/users

**Files Requiring Updates**:
- `README.md` - Update architecture description
- `Documentation/UNIFIED_ROADMAP.md` - Update strategy
- `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md` - Add refactoring details
- `Documentation/arch_tree.md` - Update component descriptions
- All other roadmap/todo documents

**Estimated Effort**: 4-6 hours

---

## 📊 Implementation Progress

### Phase 1: Foundation (Current)
- [x] Create `TimeSeriesSignalGenerator` class
- [x] Create `SignalRouter` class
- [x] Create status tracking document
- [ ] Create `models/__init__.py`
- [ ] Write unit tests
- [ ] Update configuration system

**Progress**: 3/6 tasks (50%)

### Phase 2: Pipeline Integration
- [ ] Refactor pipeline stage order
- [ ] Integrate Time Series signal generation
- [ ] Integrate Signal Router
- [ ] Update stage dependencies
- [ ] Test end-to-end flow

**Progress**: 0/5 tasks (0%)

### Phase 3: Database & Persistence
- [ ] Design unified signal schema
- [ ] Create migration script
- [ ] Update `DatabaseManager.save_signal()`
- [ ] Migrate existing data
- [ ] Update retrieval queries

**Progress**: 0/5 tasks (0%)

### Phase 4: Backward Compatibility
- [ ] Create signal adapter
- [ ] Update paper trading engine
- [ ] Update signal validator
- [ ] Update monitoring tools
- [ ] Add feature flags

**Progress**: 0/5 tasks (0%)

### Phase 5: Testing & Validation
- [ ] Unit tests for signal generator
- [ ] Unit tests for signal router
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Backward compatibility tests

**Progress**: 0/5 tasks (0%)

### Phase 6: Documentation
- [ ] Update README.md
- [ ] Update all roadmap documents
- [ ] Update architecture diagrams
- [ ] Create migration guide
- [ ] Update API documentation

**Progress**: 0/5 tasks (0%)

---

## 🎯 Critical Path

### Must Complete Before Production

1. **Pipeline Integration** (Issue 1) - CRITICAL
   - Blocks all other work
   - Required for signals to be generated
   - Estimated: 4-6 hours

2. **Database Schema Updates** (Issue 3) - HIGH
   - Required for signal persistence
   - Blocks testing with real data
   - Estimated: 3-4 hours

3. **Signal Schema Unification** (Issue 2) - HIGH
   - Required for downstream compatibility
   - Blocks integration testing
   - Estimated: 6-8 hours

4. **Testing Infrastructure** (Issue 5) - HIGH
   - Required for validation
   - Blocks confidence in changes
   - Estimated: 8-10 hours

### Can Complete in Parallel

5. **Configuration System** (Issue 4) - MEDIUM
6. **Backward Compatibility** (Issue 6) - MEDIUM
7. **Documentation Updates** (Issue 7) - MEDIUM

---

## 📅 Estimated Timeline

### Week 1: Foundation & Critical Path
- **Days 1-2**: Complete pipeline integration (Issue 1)
- **Days 3-4**: Database schema updates (Issue 3)
- **Day 5**: Signal schema unification (Issue 2) - Start

### Week 2: Integration & Testing
- **Days 1-2**: Complete signal schema unification (Issue 2)
- **Days 3-4**: Testing infrastructure (Issue 5)
- **Day 5**: Integration testing

### Week 3: Compatibility & Documentation
- **Days 1-2**: Backward compatibility (Issue 6)
- **Days 3-4**: Configuration system (Issue 4)
- **Day 5**: Documentation updates (Issue 7)

**Total Estimated Time**: 3 weeks (15 working days)

---

## ⚠️ Risk Assessment

### High Risk Items
1. **Breaking Changes**: Downstream consumers may break
   - **Mitigation**: Comprehensive testing, feature flags, gradual rollout
   
2. **Performance Impact**: Time Series forecasting adds latency
   - **Mitigation**: Parallel execution, caching, performance benchmarks
   
3. **Data Migration**: Existing signal data may be incompatible
   - **Mitigation**: Careful migration script, data validation

### Medium Risk Items
1. **Configuration Complexity**: Multiple feature flags
   - **Mitigation**: Clear documentation, sensible defaults
   
2. **Testing Coverage**: Large surface area to test
   - **Mitigation**: Incremental testing, integration test suite

---

## 🔧 Next Immediate Actions

### ✅ Completed (Nov 6, 2025)
1. ✅ Create `TimeSeriesSignalGenerator` - DONE
2. ✅ Create `SignalRouter` - DONE
3. ✅ Create status document - DONE
4. ✅ Create `models/__init__.py` - DONE
5. ✅ Pipeline integration - DONE
6. ✅ Database schema updates - DONE
7. ✅ Config-driven orchestration - DONE
8. ✅ Test files written - DONE (50 tests)

### 🚀 Next Immediate Action: Test Execution & Validation

**Priority**: **CRITICAL** - All components implemented, testing required before production

**Action Plan**:
1. Execute unit tests (38 tests) - `tests/models/`
2. Execute integration tests (12 tests) - `tests/integration/`
3. Validate config-driven orchestration - Dry-run pipeline
4. Validate database integration - Test signal persistence
5. Document test results

**See**: `Documentation/NEXT_IMMEDIATE_ACTION.md` for detailed test execution plan

**Estimated Time**: ~2 hours

---

## 📝 Notes

1. **Feature Flags**: All changes behind feature flags for safe rollout
2. **Backward Compatibility**: Maintain LLM-only mode via config
3. **Gradual Migration**: Can run both systems in parallel initially
4. **Testing**: Comprehensive test coverage before production
5. **Documentation**: Keep all docs synchronized

---

**Last Updated**: 2025-11-06  
**Status**: 🟡 INITIALIZED - CRITICAL ISSUES IDENTIFIED  
**Next Review**: After pipeline integration complete

