# Next Immediate Action: Test Execution & Validation
**Last Updated**: 2025-12-27  
**Status**: 🟡 **GATED – Live/paper profitability evidence still required (paper-window MVS now PASS)**

## 2025-12-27 Update (Verified)
- Installed `arch==8.0.0` so GARCH runs via the `arch` backend in full environments (EWMA fallback remains when `arch` is absent).
- Pytest stability: `pytest.ini` pins temp dirs under `/tmp` for WSL permission consistency and sets `pythonpath=.` to prevent `ModuleNotFoundError` when running single test modules.

## 2025-12-26 Update (Verified)
- **MVS PASS (paper-window replay)** achieved with `scripts/run_mvs_paper_window.py` (≥30 realised trades, positive PnL, WR/PF thresholds). Report: `reports/mvs_paper_window_20251226_183023.md`.
- Exit safety: `SignalValidator` now treats exposure-reducing exits (SELL closing a long / BUY covering a short) as always executable, so liquidation/time exits are not blocked by trend/regime guardrails.
- WSL SQLite reliability: `DatabaseManager` now cleans stale `*-wal`/`*-shm` artifacts on `/mnt/*` paths and refreshes/reconciles mirrors to avoid stale/unsynced state.

**Immediate focus now**
1. Re-run `bash/run_end_to_end.sh` in real paper/live windows until ≥30 realised trades are achieved (not just replay) and confirm MVS stays PASS.
2. Keep quant-validation health GREEN/YELLOW while increasing trade count (avoid “all HOLD” regimes).

## 2025-12-25 Update (Verified)
- Former 2025-11-15 engineering blockers (SQLite corruption recovery, MSSA change-point ambiguity, Matplotlib dashboard crashes, SARIMAX warning floods, backfill timestamp hygiene) are resolved in-code.
- Integration + profit-critical verification now passes; see `Documentation/PROJECT_STATUS.md` and `Documentation/INTEGRATION_TESTING_COMPLETE.md`.
- Immediate focus shifts from “make it run” to “make it profitable”: drive MVS PASS and quant-validation health GREEN/YELLOW on paper/live windows.

> The remaining sections below retain the original 2025-11-15 blocker write-up for historical context.

---

## 🎯 Overview

With config-driven orchestration complete, the **next immediate step** is to incorporate the latest architectural updates and immediately validate them end-to-end. The new autonomous trading loop, README/roadmap repositioning, and pipeline stage reordering are all live; they must now be treated as production-critical work that requires verification plus remediation of the blocking errors surfaced in the logs.

### 🚨 2025-11-15 Brutal Run Findings (blocking)
- `logs/pipeline_run.log:16932-17729` and `sqlite3 data/portfolio_maximizer.db "PRAGMA integrity_check;"` confirmed the database is corrupted (`database disk image is malformed`, “rowid … out of order/missing from index”), so none of the validation runs below can proceed until the datastore is rebuilt and `DatabaseManager._connect` treats this error like the existing disk-I/O branch.
- `logs/pipeline_run.log:2272-2279, 2624, 2979, 3263, 3547, …` show Stage 7 failing on every ticker with `ValueError: The truth value of a DatetimeIndex is ambiguous` because `scripts/run_etl_pipeline.py:1755-1764` evaluates `mssa_result.get('change_points') or []`. Every “Saved forecast …” message is followed by “Generated forecasts for 0 ticker(s)”, so Stage 8 never executes.
- The visualization hook then fails with `FigureBase.autofmt_xdate() got an unexpected keyword argument 'axis'` (lines 2626, 2981, …), meaning required dashboard artefacts are missing.
- Pandas/statsmodels warning spam remains (`forcester_ts/forecaster.py:128-136` Period round-trip; `_select_best_order` in `forcester_ts/sarimax.py:136-183` keeps unconverged grids) despite earlier hardening notes.
- `scripts/backfill_signal_validation.py:281-292` still relies on `datetime.utcnow()` and sqlite’s default converters, producing the Python 3.12 deprecation warnings logged in `logs/backfill_signal_validation.log:15-22`.

**Immediate blocking tasks**
1. Rebuild/recover `data/portfolio_maximizer.db` and extend `DatabaseManager._connect` so `"database disk image is malformed"` follows the disk-I/O recovery path.
2. Patch the MSSA `change_points` block to cast the `DatetimeIndex` to a list, rerun `python scripts/run_etl_pipeline.py --stage time_series_forecasting`, and confirm forecasts exist before executing any test suites.
3. Remove the unsupported `axis=` argument from the Matplotlib auto-format call so dashboards referenced later can actually be generated.
4. Replace the deprecated Period coercion and tighten the SARIMAX order grid to silence warning spam that currently hides actionable log lines.
5. Update `scripts/backfill_signal_validation.py` to be timezone-aware and register sqlite adapters prior to the nightly job.

### 🔄 Immediate Updates To Acknowledge

1. ✅ `scripts/run_auto_trader.py` now chains extraction → validation → forecasting → signal routing → execution with optional LLM fallback, keeping cash/positions/trade history synchronized each cycle.  
2. ✅ `README.md` markets Portfolio Maximizer as an **Autonomous Profit Engine**, highlights the hands-free loop in Key Features, and provides a Quick Start recipe plus project-structure pointer so operators can launch `run_auto_trader.py` instantly.  
3. ✅ `Documentation/UNIFIED_ROADMAP.md` elevates the autonomous loop to first-class status, expands the infrastructure/analysis bullets, and pushes the immediate action plan toward broker integration + live routing readiness.  
4. ✅ `scripts/run_etl_pipeline.py` rebuilds the stage planner so Time Series forecasting/signal routing execute *before* any LLM work; LLM stages are appended only after the router to enforce the “TS primary, LLM fallback” contract.  
5. ⚠️ `logs/errors/errors.log` reveals unresolved failures that block ETL/forecasting on live data (DataStorage CV signature mismatch, zero-fold CV division by zero, SQLite disk I/O, missing pyarrow/fastparquet checkpoints) plus LLM latency/migration warnings. These are now part of the immediate queue.
6. 🧪 `bash/comprehensive_brutal_test.sh` (Nov 12) ran to validate the stack: profit-critical tests all passed; ETL unit suites passed except `tests/etl/test_data_validator.py` (file missing); the run timed out during Time Series forecasting tests with a `Broken pipe`, so no TS/LLM regression coverage was captured. The brutal test must be rerun after addressing the missing test file and the timeout root cause.

Testing already executed: `python3 -m compileall scripts/run_auto_trader.py`. Next validation steps are listed below.

---

## ✅ Completed Prerequisites

1. ✅ **Time Series Signal Generation** - Implemented
2. ✅ **Signal Router** - Implemented  
3. ✅ **Signal Adapter** - Implemented
4. ✅ **Database Schema** - Updated
5. ✅ **Pipeline Integration** - Complete
6. ✅ **Config-Driven Orchestration** - Complete
7. ✅ **Test Files Written** - 50 tests (38 unit + 12 integration)

---

## 🚀 Next Immediate Action: Validate Autonomous + ETL Stack

### Step 1: Execute Unit Tests

**Command**:
```bash
# Activate virtual environment
.\simpleTrader_env\Scripts\Activate.ps1

# Run Time Series signal generator tests
python -m pytest tests/models/test_time_series_signal_generator.py -v --tb=short

# Run signal router tests
python -m pytest tests/models/test_signal_router.py -v --tb=short

# Run signal adapter tests
python -m pytest tests/models/test_signal_adapter.py -v --tb=short

# Run all model tests
python -m pytest tests/models/ -v --tb=short
```

**Expected Results**:
- All 38 unit tests should pass
- No import errors
- All signal generation logic validated
- All routing logic validated
- All adapter conversions validated

### Step 2: Execute Integration Tests

**Command**:
```bash
# Run integration tests
python -m pytest tests/integration/test_time_series_signal_integration.py -v --tb=short
```

**Expected Results**:
- All 12 integration tests should pass
- End-to-end pipeline flow validated
- Database persistence validated
- Signal routing validated

### Step 3: Validate Config-Driven Orchestration

**Command**:
```bash
# Dry-run test with config-driven stages
python scripts/run_etl_pipeline.py \
    --tickers AAPL,MSFT \
    --include-frontier-tickers \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --dry-run \
    --verbose
```

**Validation Checklist**:
- [ ] Stages execute in correct order (from config)
- [ ] Time Series forecasting stage runs
- [ ] Time Series signal generation stage runs
- [ ] Signal router stage runs
- [ ] Dependencies are respected
- [ ] Enabled/disabled flags work
- [ ] No errors in stage execution

### Step 4: Validate Database Integration

### Step 5: Run Autonomous Loop Smoke Test

**Command**:
```bash
python scripts/run_auto_trader.py \
  --tickers AAPL,MSFT \
  --include-frontier-tickers \
  --lookback-days 365 \
  --forecast-horizon 30 \
  --initial-capital 25000 \
  --cycles 1 \
  --sleep-seconds 0 \
  --verbose
```

**Validation Checklist**:
- [ ] DataSourceManager pulls live/synthetic data without cache/SQLite/parquet errors  
- [ ] TimeSeriesForecaster produces a forecast bundle per ticker  
- [ ] SignalRouter emits a primary TS signal and optional LLM fallback  
- [ ] PaperTradingEngine executes trades, updates cash/positions, and logs PnL  
- [ ] Run terminates cleanly and summaries reflect the new autonomous flow

**Command**:
```bash
# Run database tests
python -m pytest tests/etl/test_database_manager.py::test_save_trading_signal -v
```

**Validation Checklist**:
- [ ] `trading_signals` table exists
- [ ] Time Series signals save correctly
- [ ] LLM signals save correctly
- [ ] Unified schema works
- [ ] No constraint violations

---

## 📋 Test Execution Plan

### Phase 1: Unit Test Validation (30 minutes)
1. Execute `test_time_series_signal_generator.py` (15 tests)
2. Execute `test_signal_router.py` (12 tests)
3. Execute `test_signal_adapter.py` (11 tests)
4. Fix any failures
5. Document results

### Phase 2: Integration Test Validation (45 minutes)
1. Execute `test_time_series_signal_integration.py` (12 tests)
2. Validate end-to-end flow
3. Fix any failures
4. Document results

### Phase 3: Pipeline Integration Validation (30 minutes)
1. Dry-run pipeline with new stages
2. Verify stage ordering
3. Verify dependency resolution
4. Verify config-driven execution
5. Document results

### Phase 4: Database Validation (15 minutes)

### Phase 5: Autonomous Loop Smoke (15 minutes)  
1. Run the command above (or `--dry-run` if live keys unavailable).  
2. Verify logs show the TS-first ordering (extraction → validation → preprocessing → storage → TS forecasting → TS signal gen → signal router → LLM).  
3. Confirm PaperTradingEngine records at least one execution result and cash/positions update.  
4. Capture console/log excerpts for documentation.
1. Test database schema
2. Test signal persistence
3. Test query operations
4. Document results

**Total Estimated Time**: ~2 hours

---

## 🎯 Success Criteria

### Test Execution
- [ ] All 38 unit tests pass
- [ ] All 12 integration tests pass
- [ ] No import errors
- [ ] No runtime errors

### Pipeline Validation
- [ ] Config-driven stages execute correctly
- [ ] Stage ordering is correct
- [ ] Dependencies are respected
- [ ] Enable/disable flags work

### Database Validation

### Autonomous Loop
- [ ] Pipeline logs show the TS-first order before LLM fallback  
- [ ] `execution.paper_trading_engine` reports executed/rejected trades  
- [ ] Cash/positions totals update between cycles  
- [ ] No blocking errors in `logs/pipeline_run.log` or `logs/errors/errors.log`
- [ ] `bash/comprehensive_brutal_test.sh` completes Time Series forecasting suites without timeout or Broken pipe errors; `tests/etl/test_data_validator.py` is restored so the ETL phase no longer flags a missing file.
- [ ] Schema is correct
- [ ] Signals persist correctly
- [ ] Queries work correctly
- [ ] No constraint violations

---

## ⚠️ If Tests Fail

### Common Issues & Fixes

1. **Import Errors**
   - Check `models/__init__.py` exports
   - Verify package structure
   - Check Python path

2. **Missing Dependencies**
   - Check `requirements.txt`
   - Install missing packages
   - Verify virtual environment

3. **Database Errors**
   - Check database schema
   - Verify migrations
   - Check constraints

4. **Config Errors**

5. **Autonomous Loop Failures**
   - Inspect `logs/pipeline_run.log` and `logs/errors/errors.log` for the DataStorage CV bug, zero-fold CV division, SQLite disk I/O, or parquet-missing errors.  
   - Install `pyarrow` or `fastparquet`, fix database permissions, and rerun migrations before re-attempting the loop.
   - Verify `pipeline_config.yml` syntax
   - Check `signal_routing_config.yml`
   - Validate YAML structure

---

## 📝 Documentation Updates Required

After successful test execution:

1. Update `REFACTORING_STATUS.md` - Mark tests as executed
2. Update `REFACTORING_IMPLEMENTATION_COMPLETE.md` - Update test status
3. Update `TESTING_IMPLEMENTATION_SUMMARY.md` - Add execution results
4. Update `INTEGRATION_TESTING_COMPLETE.md` - Add execution results
5. Update `implementation_checkpoint.md` - Mark testing complete

---

## 🔄 Next Steps After Testing

Once tests pass:

1. **Performance Benchmarks** - Measure signal generation latency
2. **Production Validation** - Test with real data
3. **Documentation Finalization** - Update all docs with test results
4. **Deployment Preparation** - Prepare for production rollout

---

**Last Updated**: 2025-11-06  
**Status**: 🟡 **READY FOR EXECUTION**  
**Priority**: **CRITICAL** - Blocking production deployment

