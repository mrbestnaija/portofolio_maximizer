# Testing Guide - Profit-Critical Functions

**Per AGENT_INSTRUCTION.md Guidelines**  
**Version**: 1.0  
**Date**: 2025-10-14

### Nov 12, 2025 Test Execution
- pytest tests/models/test_time_series_signal_generator.py -q and targeted integration cases now pass under simpleTrader_env, validating the pandas-safe signal generator changes.
- Pipeline dry-runs continue to hit the Ollama latency regression (brutal suite fails only in 	ests/ai_llm/test_ollama_client.py::test_generate_switches_model_when_token_rate_low); keep this test on every verification list until latency <5 s.
- logs/ts_signal_demo.json is the new smoke artifact proving BUY/SELL signals can be generated outside the test harness.


---

## 🎯 **Testing Philosophy**

### **From AGENT_INSTRUCTION.md**:
```
Maximum 500 lines of test code (Phase 4-6)
Test ONLY profit-critical functions
Don't spend more time testing than developing
Focus: Business logic that loses money if broken
```

### **What We Test**:
- ✅ Profit calculations (exact to the penny)
- ✅ Win rate calculations (primary success metric)
- ✅ Profit factor (system profitability indicator)
- ✅ Database persistence (prevents data loss)
- ✅ MVS criteria validation (prevents false positives)
- ✅ Report accuracy (correct profit display)

### **What We DON'T Test**:
- ❌ UI/presentation logic
- ❌ Logging output format
- ❌ Configuration file parsing
- ❌ Non-critical helper functions

---

## 🚀 **Quick Start - Run All Tests**

### **Option 1: Bash Script (Recommended)**
```bash
cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45

# Activate environment
source simpleTrader_env/bin/activate

# Run profit-critical tests
bash bash/test_profit_critical_functions.sh

# Run real-time pipeline tests
bash bash/test_real_time_pipeline.sh
```

### **Option 2: Direct pytest**
```bash
# Activate environment
source simpleTrader_env/bin/activate

# Run all profit-critical tests
pytest tests/integration/test_profit_critical_functions.py \
       tests/integration/test_llm_report_generation.py \
       -v --tb=short

# Run specific test class
pytest tests/integration/test_profit_critical_functions.py::TestProfitCriticalDatabaseFunctions -v

# Run single test
pytest tests/integration/test_profit_critical_functions.py::TestProfitCriticalDatabaseFunctions::test_profit_calculation_accuracy -v
```

---

## 📊 **Test Categories**

### **1. Profit-Critical Database Functions** (12 tests)
**File**: `tests/integration/test_profit_critical_functions.py`

#### **Tests**:
1. **test_profit_calculation_accuracy**: Total profit = exact sum
2. **test_profit_factor_calculation**: Gross profit / Gross loss
3. **test_negative_profit_tracking**: Losses tracked correctly
4. **test_llm_analysis_persistence**: LLM data not lost
5. **test_signal_validation_status_tracking**: Prevents auto-trading
6. **test_ohlcv_data_saves_correctly**: Price data accurate
7. **test_mvs_passing_system**: Correct MVS identification
8. **test_mvs_failing_system**: Correct MVS failure detection
9. **test_duplicate_signal_prevention**: No double-counting
10. **test_database_query_performance**: < 0.5s for 1000 trades

**Run**: `pytest tests/integration/test_profit_critical_functions.py -v`

---

### **2. Report Generation Accuracy** (7 tests)
**File**: `tests/integration/test_llm_report_generation.py`

#### **Tests**:
1. **test_profit_report_total_profit**: Report shows correct profit
2. **test_profit_report_win_rate**: Win rate calculation correct
3. **test_profit_report_profit_factor**: Profit factor accurate
4. **test_mvs_criteria_evaluation**: MVS status correct
5. **test_comprehensive_report_json_format**: Valid JSON output
6. **test_text_report_format**: Human-readable text
7. **test_mvs_passing_criteria**: MVS passing correctly identified

**Run**: `pytest tests/integration/test_llm_report_generation.py -v`

---

### **3. Time-Series Signal Stack & Routing** (14+ tests)
**File**: `tests/etl/test_time_series_forecaster.py`, `tests/models/test_time_series_signal_generator.py`, `tests/models/test_signal_router.py`

#### **Why it Matters**
- The refactor described in `Documentation/REFACTORING_IMPLEMENTATION_COMPLETE.md` makes the time-series ensemble the default signal generator; these tests ensure SARIMAX/SAMOSSA/GARCH/MSSA-RL outputs stay stable, ensemble regression metrics persist correctly, and `signal_router.py` always prefers TS signals before falling back to LLM.
- Regression metrics (RMSE / sMAPE / tracking error) are now part of the SQLite persistence contract (`time_series_forecasts.regression_metrics`) and feed dashboards/weighting logic—tests must fail fast if those fields disappear.

#### **Run**
```bash
pytest tests/etl/test_time_series_forecaster.py -q
pytest tests/models/test_time_series_signal_generator.py -q
pytest tests/models/test_signal_router.py -q
```

### **4. Monitoring & Nightly Backfill** (NEW – Nov 2025)
**Why it Matters**: The monitoring job now enforces the <5 s latency target, surfaces `llm_signal_backtests`, and validates that Time Series signals actually exist. Nightly validation backfills keep those metrics fresh so dashboards never regress to `NO_DATA`.

**Run**
```bash
# LLM monitoring smoke (writes logs/latency_benchmark.json + monitoring report)
.\simpleTrader_env\Scripts\python.exe scripts/monitor_llm_system.py --headless

# Register and dry-run the nightly backfill helper
schedule_backfill.bat  # (from PowerShell) – mirror Task Scheduler wiring before prod
```

**Checks**
- `logs/latency_benchmark.json` updated with latest latency/token-rate snapshot.
- `logs/llm_monitoring_report_*.json` contains `signal_backtests.summary`.
- SQLite tables `llm_signal_backtests` and `time_series_forecasts` show fresh timestamps after the helper executes.

---

## 🧪 **Real-Time Testing Commands**

### **Test 1: Single Ticker Pipeline**
```bash
source simpleTrader_env/bin/activate

# Run pipeline with AAPL
python scripts/run_etl_pipeline.py --tickers AAPL --enable-llm --verbose

# Expected output:
# ✓ Database manager initialized
# ✓ LLM initialized: qwen:14b-chat-q4_K_M
# ✓ AAPL: Trend=bullish, Strength=7/10 (24.3s)
# ✓ Saved to database
```

### **Test 2: Multi-Ticker Pipeline**
```bash
# Run with 3 tickers
python scripts/run_etl_pipeline.py --tickers AAPL,MSFT,GOOGL --enable-llm --verbose

# Verify all 3 tickers processed
# Check database for 3 sets of data
```

### **Test 3: Text Report Generation**
```bash
# Generate text report
python scripts/generate_llm_report.py --format text

# Should output:
# ===============================================================================
# LLM PORTFOLIO PERFORMANCE - COMPREHENSIVE REPORT
# ===============================================================================
# Total Profit: $X,XXX.XX
# Win Rate: XX.X%
# Profit Factor: X.XX
```

### **Test 4: JSON Report Generation**
```bash
# Generate JSON report
python scripts/generate_llm_report.py --format json > metrics.json

# Verify valid JSON
python -c "import json; print(json.load(open('metrics.json'))['reports']['profit_loss']['metrics']['total_profit_usd'])"
```

### **Test 5: HTML Dashboard Generation**
```bash
# Generate HTML dashboard
python scripts/generate_llm_report.py --format html --output dashboard.html

# Check file created
ls -lh dashboard.html

# Open in browser (optional)
# firefox dashboard.html  # or your browser
```

### **Test 6: Database Queries**
```bash
# Total profit
sqlite3 data/portfolio_maximizer.db \
  "SELECT SUM(realized_pnl) FROM trade_executions;"

# Win rate
sqlite3 data/portfolio_maximizer.db \
  "SELECT 
     COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) * 100.0 / COUNT(*) as win_rate
   FROM trade_executions;"

# Latest signals
sqlite3 data/portfolio_maximizer.db \
  "SELECT * FROM llm_signals ORDER BY signal_date DESC LIMIT 5;"

# LLM analyses count
sqlite3 data/portfolio_maximizer.db \
  "SELECT ticker, COUNT(*) as count FROM llm_analyses GROUP BY ticker;"
```

---

## ✅ **Test Success Criteria**

### **All Tests Must Pass Before Production**:
```
Per AGENT_INSTRUCTION.md:
- ✓ Profit calculations exact (< $0.01 error)
- ✓ Win rate calculations correct (< 0.1% error)
- ✓ Profit factor accurate (< 0.01 error)
- ✓ Database queries < 0.5 seconds
- ✓ MVS criteria correctly evaluated
- ✓ Reports show accurate metrics
- ✓ No data loss in persistence
```

---

## 🔍 **Troubleshooting**

### **Issue: Tests fail with "No module named 'etl'"**
**Solution**:
```bash
# Ensure you're in project root
cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45

# Activate virtual environment
source simpleTrader_env/bin/activate

# Run tests
pytest tests/integration/test_profit_critical_functions.py -v
```

### **Issue: "Ollama not found"**
**Solution**:
```bash
# Check if Ollama is running
ollama list

# If not found, start Ollama
ollama serve

# Or run tests without LLM
python scripts/run_etl_pipeline.py --tickers AAPL
# (without --enable-llm flag)
```

### **Issue: "Database locked"**
**Solution**:
```bash
# Close all connections
pkill -f portfolio_maximizer.db

# Or delete test database
rm -f data/portfolio_maximizer.db
rm -f data/test_*.db

# Re-run pipeline
python scripts/run_etl_pipeline.py --tickers AAPL --enable-llm
```

### **Issue: Profit calculations seem wrong**
**Solution**:
```bash
# Run profit-critical tests
pytest tests/integration/test_profit_critical_functions.py::TestProfitCriticalDatabaseFunctions::test_profit_calculation_accuracy -v

# Check database directly
sqlite3 data/portfolio_maximizer.db "SELECT * FROM trade_executions;"
```

---

## 📈 **Performance Benchmarks**

**Per AGENT_INSTRUCTION.md - Performance Requirements**:

| Operation | Target | Current | Status |
|-----------|--------|---------|--------|
| Database init | < 1s | ~0.1s | ✅ |
| Profit summary (1000 trades) | < 0.5s | ~0.2s | ✅ |
| LLM analysis (per ticker) | < 60s | 20-40s | ✅ |
| Report generation | < 5s | ~2s | ✅ |
| Pipeline (3 tickers + LLM) | < 5min | 2-3min | ✅ |

---

## 📋 **Test Checklist**

### **Before Every Development Session**:
- [ ] Run profit-critical tests: `bash bash/test_profit_critical_functions.sh`
- [ ] All tests passing (100%)
- [ ] No profit calculation errors
- [ ] Database persistence working

### **Before Any Deployment**:
- [ ] Full test suite passes
- [ ] Real-time pipeline test successful
- [ ] Report generation working (all 3 formats)
- [ ] Database queries functional
- [ ] MVS criteria correctly evaluated
- [ ] No performance regressions

### **Weekly Validation** (Per AGENT_INSTRUCTION.md):
- [ ] Strategy performance test (if trading)
- [ ] System integration test
- [ ] Cost accounting (must be $0/month for local LLM)
- [ ] Complexity audit (line count check)

---

## 📚 **References**

- **Test Files**:
  - `tests/integration/test_profit_critical_functions.py`
  - `tests/integration/test_llm_report_generation.py`

- **Test Scripts**:
  - `bash/test_profit_critical_functions.sh`
  - `bash/test_real_time_pipeline.sh`

- **Documentation**:
  - `Documentation/AGENT_INSTRUCTION.md` (Testing guidelines)
  - `Documentation/QUANTIFIABLE_SUCCESS_CRITERIA.md` (MVS/PRS criteria)
  - `Documentation/QUICK_REFERENCE_OPTIMIZED_SYSTEM.md` (Usage guide)

---

**Status**: ✅ **TEST SUITE COMPLETE**  
**Total Tests**: 19 (profit-critical only)  
**Test Lines**: < 500 (per AGENT_INSTRUCTION.md)  
**Focus**: Money-affecting business logic only  
**Philosophy**: Test what matters, skip what doesn't


