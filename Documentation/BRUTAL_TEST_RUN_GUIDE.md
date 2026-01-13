# How to Run Comprehensive Brutal Test

> **RUNTIME GUARDRAIL (WSL `simpleTrader_env` ONLY)**  
> Supported runtime: WSL + Linux venv `simpleTrader_env/bin/python` (`source simpleTrader_env/bin/activate`).  
> **Do not** use Windows interpreters/venvs (incl. `py`, `python.exe`, `.venv`, `simpleTrader_env\\Scripts\\python.exe`) — results are invalid.  
> Before reporting runs, include the runtime fingerprint (command + output): `which python`, `python -V`, `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` (see `Documentation/RUNTIME_GUARDRAILS.md`).

**Script**: `bash/comprehensive_brutal_test.sh`  
**Duration**: 3-6 hours (configurable)  
**Purpose**: Exhaustive testing of all project components

---

## Prerequisites

1. **Virtual Environment**: Must be activated
2. **Python Dependencies**: All packages installed
3. **Test Database**: Will be created automatically
4. **Ollama** (Optional): For LLM tests

---

## Quick Start

### Option 1: Windows PowerShell (Supported via WSL only)

```powershell
wsl -e bash -lc "cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45 && source simpleTrader_env/bin/activate && bash bash/comprehensive_brutal_test.sh"
```

### Option 2: Windows WSL (Recommended)

```bash
# Navigate to project root
cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45

# Activate virtual environment
source simpleTrader_env/bin/activate

# Make script executable (first time only)
chmod +x bash/comprehensive_brutal_test.sh

# Run the test
bash bash/comprehensive_brutal_test.sh
```

### Option 3: Git Bash (Windows) (Unsupported)

Do not run this repository under Git Bash/Windows Python. Use **Option 2 (Windows WSL)** instead.

### Option 4: Linux/macOS (Unsupported)

This repo is validated only under WSL with `simpleTrader_env`. If you are not in WSL, switch to **Option 2**.

---

## Configuration Options

### Environment Variables

Set these before running to customize the test:

```bash
# Test duration (hours) - default: 4
export TEST_DURATION_HOURS=6

# Iterations per test - default: 3
export ITERATIONS_PER_TEST=5

# Tickers to test - default: AAPL,MSFT,GOOGL (frontier list auto-appended via --include-frontier-tickers)
export TICKERS_LIST="AAPL,MSFT,GOOGL,TSLA,AMZN"

# Date range - default: 2020-01-01 to 2024-01-01
export START_DATE="2020-01-01"
export END_DATE="2024-01-01"

# Run the test
bash bash/comprehensive_brutal_test.sh
```

### Windows PowerShell Example

```powershell
# Set environment variables
$env:TEST_DURATION_HOURS = "6"
$env:ITERATIONS_PER_TEST = "5"
$env:TICKERS_LIST = "AAPL,MSFT,GOOGL"

# Run via WSL (supported runtime)
wsl -e bash -lc "cd /mnt/c/Users/Bestman/personal_projects/portfolio_maximizer_v45 && source simpleTrader_env/bin/activate && TEST_DURATION_HOURS=6 ITERATIONS_PER_TEST=5 TICKERS_LIST='AAPL,MSFT,GOOGL' bash bash/comprehensive_brutal_test.sh"
```

`bash/comprehensive_brutal_test.sh` now passes `--include-frontier-tickers`, so the Nigeria → Bulgaria atlas from `etl/frontier_markets.py` is exercised automatically alongside whatever base symbols you define in `TICKERS_LIST`.

---

## What the Test Does

The test runs in this order:

1. **Environment Setup** (2-5 minutes)
   - Activates virtual environment
   - Checks Python and dependencies
   - Verifies Ollama (optional)
   - Creates test database

2. **Profit-Critical Tests** (Priority 1 - 10-15 minutes)
   - Profit calculation accuracy
   - Profit factor calculation
   - Report generation accuracy

3. **ETL Unit Tests** (Priority 2 - 20-30 minutes)
   - Data storage tests
   - Preprocessor tests
   - Validator tests
   - Cross-validation tests
   - Data source manager tests
   - Checkpoint manager tests

4. **Time Series Tests** (Priority 2 - 15-20 minutes)
   - Forecasting model tests
   - Signal generator tests

5. **Signal Routing Tests** (Priority 2 - 10-15 minutes)
   - Signal router tests
   - Signal adapter tests

6. **Integration Tests** (Priority 2 - 20-30 minutes)
   - Time Series signal integration
   - LLM-ETL pipeline integration

7. **LLM Integration Tests** (Priority 3 - 15-20 minutes, optional)
   - Ollama client tests
   - Market analyzer tests
   - Signal validator tests
   - LLM enhancements tests

8. **Security Tests** (Priority 4 - 5-10 minutes)
   - Security validation tests

9. **Pipeline Execution Tests** (Priority 5 - 30-45 minutes)
   - Basic pipeline execution
   - Pipeline with cross-validation
   - Pipeline with Time Series forecasting

10. **Database Integrity Tests** (Priority 6 - 5-10 minutes)
    - Schema validation
    - Data consistency checks

11. **Performance Benchmarking** (Priority 7 - 30-60 minutes)
    - 5 performance runs
    - Statistics generation

12. **Report Generation** (1-2 minutes)
    - Final report with all results

**Total Estimated Time**: 3-6 hours (depending on system and configuration)

---

## Output Structure

After running, you'll find results in:

```
logs/brutal/results_YYYYMMDD_HHMMSS/
├── logs/                          # Individual test logs
│   ├── profit_critical.log
│   ├── etl_unit_tests.log
│   ├── time_series_tests.log
│   ├── signal_routing_tests.log
│   ├── integration_tests.log
│   ├── llm_tests.log
│   ├── security_tests.log
│   ├── pipeline_execution.log
│   ├── database_integrity.log
│   └── test.log                  # Main test log
├── reports/                       # Generated reports
│   └── final_report.md           # Comprehensive report
├── artifacts/                     # Test artifacts
│   └── test_database.db          # Test database
├── performance/                   # Performance data
│   ├── performance_results.csv
│   └── performance_benchmark.log
├── stage_summary.csv              # Stage pass/fail summary
└── test.log                       # Complete test log
```

---

## Interpreting Results

### Success Indicators

✅ **All tests passed**: Exit code 0, "All tests passed! ✅"

✅ **Final Report**: Check `reports/final_report.md` for:
- Total tests run
- Pass/fail counts
- Pass rate percentage
- Stage-by-stage breakdown

### Failure Indicators

❌ **Some tests failed**: Exit code 1, "Some tests failed"

**What to do**:
1. Check `logs/` directory for specific test failures
2. Review `reports/final_report.md` for summary
3. Check `test.log` for detailed error messages
4. Fix issues and re-run specific test categories

### Common Issues

**Issue**: "Virtual environment not found"
- **Solution**: Ensure you're in the project root and `simpleTrader_env` exists

**Issue**: "Required packages not available"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: "Ollama not found - LLM tests skipped"
- **Solution**: This is OK - LLM tests are optional. Install Ollama if you want to test LLM integration.

**Issue**: "Test file not found"
- **Solution**: Some tests may be optional. Check if the file exists in `tests/` directory.

---

## Running Specific Test Categories

If you want to run only specific tests, you can modify the script or run pytest directly:

### Run Only Profit-Critical Tests

```bash
python -m pytest tests/integration/test_profit_critical_functions.py -v
```

### Run Only ETL Unit Tests

```bash
python -m pytest tests/etl/ -v
```

### Run Only Time Series Tests

```bash
python -m pytest tests/etl/test_time_series_forecaster.py tests/models/test_time_series_signal_generator.py -v
```

### Run Only Security Tests

```bash
python -m pytest -m security -v
```

---

## Monitoring Progress

The test script provides real-time output:

```
=== Comprehensive Brutal Test Suite ===
[INFO] Per AGENT_INSTRUCTION.md, AGENT_DEV_CHECKLIST.md, and project guidelines
[INFO] Expected duration: 4 hours

=== Environment Setup ===
[PASS] Virtual environment activated
[INFO] Python: Python 3.12.0
[PASS] Required packages available
[PASS] Environment setup complete

=== Profit-Critical Function Tests ===
[INFO] Per AGENT_INSTRUCTION.md: Testing money-affecting logic only

--- Profit Calculation Accuracy ---
[PASS] Profit calculation accuracy tests

--- Profit Factor Calculation ---
[PASS] Profit factor calculation tests
...
```

---

## Tips

1. **Run Overnight**: Since tests take 3-6 hours, consider running overnight
2. **Check Disk Space**: Tests create logs and artifacts (~100-500 MB)
3. **Monitor System**: Tests are CPU/memory intensive
4. **Review Logs**: Check `logs/` directory if any tests fail
5. **Incremental Testing**: Run specific test categories first before full suite

---

## Troubleshooting

### Script Won't Run (Windows)

```powershell
# Use WSL instead
wsl bash bash/comprehensive_brutal_test.sh

# Or use Git Bash
"C:\Program Files\Git\bin\bash.exe" bash/comprehensive_brutal_test.sh
```

### Permission Denied (Linux/macOS)

```bash
chmod +x bash/comprehensive_brutal_test.sh
./bash/comprehensive_brutal_test.sh
```

### Tests Timeout

Reduce iterations:
```bash
export ITERATIONS_PER_TEST=1
bash bash/comprehensive_brutal_test.sh
```

### Out of Memory

Reduce test scope:
```bash
export TICKERS_LIST="AAPL"  # Test with fewer tickers
bash bash/comprehensive_brutal_test.sh
```

---

## Next Steps After Testing

1. **Review Report**: Check `reports/final_report.md`
2. **Fix Failures**: Address any failed tests
3. **Re-run**: Run specific test categories after fixes
4. **Document**: Update test results in project documentation

---

**Status**: Ready to run  
**Last Updated**: 2025-11-09  
**Compliance**: ✅ AGENT_INSTRUCTION.md, ✅ AGENT_DEV_CHECKLIST.md, ✅ API_KEYS_SECURITY.md

