# Phase 7.3 Ensemble Fix - Pipeline Test Plan

**Date:** 2026-01-20
**Test ID:** phase7.3_ensemble_garch_weights
**Status:** ⏳ RUNNING

---

## Test Objective

Verify that the ensemble weighting fixes successfully incorporate GARCH into the ensemble and improve RMSE ratio from 1.682x to target <1.1x.

---

## Changes Being Tested

### Code Changes (Completed)
1. **Config:** Added GARCH-dominant candidate weights (85%, 70%, 60%)
2. **Forecaster:** Included GARCH in ensemble forecast dictionaries
3. **Forecaster:** Added GARCH to holdout inverse-RMSE reweighting
4. **Ensemble:** Added GARCH confidence scoring
5. **Pipeline:** Changed ensemble model_type to 'ENSEMBLE'
6. **Diagnostics:** Flexible ensemble key matching

### Expected Behavior
- Ensemble will select GARCH-dominant weight combinations when GARCH outperforms
- Inverse-RMSE reweighting will favor GARCH (if within 5% of best, or use 100% GARCH if 5%+ better)
- Database will contain 'ENSEMBLE' model_type records with GARCH in weights

---

## Test Configuration

**Command:**
```bash
simpleTrader_env\Scripts\python.exe scripts\run_etl_pipeline.py \
    --tickers AAPL,MSFT,NVDA \
    --start 2024-07-01 \
    --end 2026-01-18 \
    --execution-mode live
```

**Tickers:** AAPL, MSFT, NVDA
**Date Range:** 2024-07-01 to 2026-01-18 (18 months)
**Execution Mode:** live
**Log File:** `logs/phase7.3_ensemble_test.log`

---

## Success Criteria

### Primary (MUST PASS)
1. ✅ **Ensemble forecasts saved to database** with model_type='ENSEMBLE'
2. ✅ **GARCH appears in ensemble weights** (weight > 0%)
3. ✅ **RMSE ratio < 1.5x** for at least 2/3 tickers (50% improvement from 1.682x)

### Secondary (SHOULD PASS)
4. 🎯 **RMSE ratio < 1.2x** for at least 2/3 tickers (ensemble within 20% of best)
5. 🎯 **GARCH weight >= 60%** for at least 2/3 tickers (dominant model)

### Stretch Goal (IDEAL)
6. 🌟 **RMSE ratio < 1.1x** for at least 2/3 tickers (TARGET MET)
7. 🌟 **GARCH weight >= 85%** or 100% for at least 1 ticker (optimal weighting)

---

## Verification Steps

### Step 1: Check Pipeline Logs (While Running)
```bash
# Monitor progress
tail -f logs/phase7.3_ensemble_test.log

# Look for:
# - "ENSEMBLE build_complete weights={'garch': 0.XX, ...}"
# - "ENSEMBLE reweighted_from_holdout weights={'garch': 0.XX, ...}"
# - No errors during ensemble blending
```

### Step 2: Verify Database Records (After Completion)
```bash
# Check ensemble forecasts exist
simpleTrader_env\Scripts\python.exe scripts\check_ensemble_weights.py --ticker AAPL
simpleTrader_env\Scripts\python.exe scripts\check_ensemble_weights.py --ticker MSFT
simpleTrader_env\Scripts\python.exe scripts\check_ensemble_weights.py --ticker NVDA
```

**Expected Output:**
```
ENSEMBLE WEIGHTS VERIFICATION - AAPL
=========================================
1. Ensemble Forecasts in Database: 30

2. Latest Ensemble Weights:
   garch       :  85.00%  # CRITICAL: GARCH should be dominant
   sarimax     :  10.00%
   samossa     :   5.00%

3. Model Confidence Scores:
   garch       : 0.8450
   sarimax     : 0.5230
   samossa     : 0.4120

6. Individual Model Performance (RMSE):
   GARCH       : RMSE=  30.6400
   SARIMAX     : RMSE= 229.4400
   SAMoSSA     : RMSE= 234.8800
   ENSEMBLE    : RMSE=  35.2000  # Should be close to GARCH

7. RMSE Analysis:
   Best Single Model: GARCH (RMSE=30.6400)
   Ensemble RMSE:     35.2000
   RMSE Ratio:        1.149x
   ✓ GOOD: Ensemble within 20% of best model
```

### Step 3: Run Full Diagnostics (After Completion)
```bash
# Generate diagnostic visualizations
simpleTrader_env\Scripts\python.exe scripts\run_ensemble_diagnostics.py --ticker AAPL --days 30
simpleTrader_env\Scripts\python.exe scripts\run_ensemble_diagnostics.py --ticker MSFT --days 30
simpleTrader_env\Scripts\python.exe scripts\run_ensemble_diagnostics.py --ticker NVDA --days 30
```

**Files Generated:**
- `visualizations/ensemble_diagnostics/AAPL/error_decomposition.png`
- `visualizations/ensemble_diagnostics/AAPL/confidence_calibration.png`
- `visualizations/ensemble_diagnostics/AAPL/weight_optimization.png`
- `visualizations/ensemble_diagnostics/AAPL/ensemble_diagnostics_report.txt`

**Check:**
- Error decomposition shows RMSE ratio <1.2x
- Weight optimization shows current weights are near-optimal (or optimal if 100% GARCH)
- Confidence calibration shows GARCH has high confidence score

---

## Failure Scenarios & Troubleshooting

### Scenario 1: No Ensemble Forecasts in Database
**Symptom:** `check_ensemble_weights.py` shows 0 ensemble forecasts

**Possible Causes:**
1. Ensemble not enabled in config
2. Ensemble building failed (check logs for errors)
3. Not enough models succeeded (need at least 2 for ensemble)

**Troubleshooting:**
```bash
# Check config
grep -A 5 "ensemble:" config/forecasting_config.yml

# Check logs for ensemble errors
grep -i "ensemble" logs/phase7.3_ensemble_test.log | grep -i "error\|fail"

# Check if individual models succeeded
sqlite3 data/portfolio_maximizer.db "SELECT model_type, COUNT(*) FROM time_series_forecasts WHERE ticker='AAPL' GROUP BY model_type;"
```

### Scenario 2: GARCH Weight is 0%
**Symptom:** `check_ensemble_weights.py` shows GARCH not in weights

**Possible Causes:**
1. GARCH forecast failed (check logs)
2. GARCH not included in forecast dictionaries (code regression)
3. Config candidate_weights don't include GARCH (check config)

**Troubleshooting:**
```bash
# Check if GARCH forecasts exist
sqlite3 data/portfolio_maximizer.db "SELECT COUNT(*) FROM time_series_forecasts WHERE ticker='AAPL' AND model_type='GARCH';"

# Check logs for GARCH errors
grep -i "garch" logs/phase7.3_ensemble_test.log | grep -i "error\|fail"

# Verify config has GARCH weights
grep -A 10 "candidate_weights:" config/forecasting_config.yml | grep "garch"
```

### Scenario 3: RMSE Ratio Still High (>1.5x)
**Symptom:** Ensemble RMSE is 1.5x+ best model even with GARCH included

**Possible Causes:**
1. GARCH weight too low (<40%) - other models contaminating ensemble
2. GARCH itself has high RMSE (not actually best model for this ticker)
3. Confidence scaling reducing GARCH weight inappropriately

**Troubleshooting:**
```bash
# Check individual model RMSE
simpleTrader_env\Scripts\python.exe scripts\check_ensemble_weights.py --ticker AAPL

# If GARCH weight is low, check why:
# - Look at confidence scores (is GARCH confidence low?)
# - Check if holdout reweighting happened (may need more data)

# Run diagnostics to see optimal weights
simpleTrader_env\Scripts\python.exe scripts\run_ensemble_diagnostics.py --ticker AAPL
# Check weight_optimization.png - are optimal weights very different from current?
```

### Scenario 4: Pipeline Fails During Forecasting
**Symptom:** Pipeline crashes or hangs

**Troubleshooting:**
```bash
# Check latest log entries
tail -100 logs/phase7.3_ensemble_test.log

# Common errors:
# - Memory issues: Reduce data range or tickers
# - Import errors: Check all dependencies installed
# - Config errors: Validate YAML syntax

# If hanging, may be waiting for LLM (if enabled)
# Kill and restart without LLM:
python scripts/run_etl_pipeline.py --tickers AAPL --start 2024-07-01 --end 2026-01-18 --execution-mode live
```

---

## Post-Test Analysis

### Metrics to Extract

For each ticker (AAPL, MSFT, NVDA):

| Metric | Value | Pass Criteria |
|--------|-------|---------------|
| GARCH Weight | XX.X% | >= 40% (acceptable), >= 60% (good) |
| Ensemble RMSE | XX.XX | N/A (absolute value) |
| Best Model RMSE | XX.XX | N/A |
| RMSE Ratio | X.XXx | < 1.5 (acceptable), < 1.2 (good), < 1.1 (target) |
| GARCH RMSE Rank | 1st/2nd/3rd/etc | 1st or 2nd preferred |

### Comparison to Baseline

**Before Fix (from diagnostics 2026-01-20):**
- AAPL GARCH RMSE: 30.64 (best)
- AAPL Ensemble RMSE: ~51.5 (estimated from 1.682x ratio)
- AAPL RMSE Ratio: 1.682x ❌

**After Fix (this test):**
- AAPL GARCH RMSE: ??? (should be similar ~30-35)
- AAPL Ensemble RMSE: ??? (target ~33-37)
- AAPL RMSE Ratio: ??? (target <1.2x)

**Expected Improvement:**
- RMSE Ratio: 1.682x → 1.1-1.2x (28-34% improvement)
- Or: 1.682x → 1.0x if 100% GARCH (optimal)

---

## Timeline

**Start:** 2026-01-20 (time when pipeline launched)
**Expected Duration:** 30-60 minutes (3 tickers, 18 months data each)
**Verification:** +10 minutes (check_ensemble_weights.py, diagnostics)
**Total:** ~70 minutes end-to-end

---

## Rollback Plan

If test fails catastrophically:

1. **Revert code changes:**
   ```bash
   git checkout config/forecasting_config.yml
   git checkout forcester_ts/forecaster.py
   git checkout forcester_ts/ensemble.py
   git checkout scripts/run_etl_pipeline.py
   ```

2. **Clear bad ensemble records:**
   ```sql
   DELETE FROM time_series_forecasts WHERE model_type = 'ENSEMBLE';
   ```

3. **Document failure mode** in this file and create issue

---

## Success Declaration

Test is SUCCESSFUL if:
1. ✅ All 3 tickers have ensemble forecasts in database
2. ✅ GARCH weight > 0% for all 3 tickers
3. ✅ RMSE ratio < 1.5x for at least 2/3 tickers

Test is HIGHLY SUCCESSFUL if:
1. 🎯 RMSE ratio < 1.2x for all 3 tickers
2. 🎯 GARCH weight >= 60% for at least 2/3 tickers

Test EXCEEDS EXPECTATIONS if:
1. 🌟 RMSE ratio < 1.1x for all 3 tickers (TARGET MET)
2. 🌟 System achieves pure GARCH ensemble (100% weight) for any ticker

---

## Next Steps After Test

### If Successful
1. Document results in this file
2. Update [PHASE_7.3_ENSEMBLE_FIX.md](PHASE_7.3_ENSEMBLE_FIX.md) with actual results
3. Run diagnostics on all tickers and generate reports
4. Proceed with production deployment planning

### If Partially Successful (RMSE 1.2-1.5x)
1. Analyze which tickers failed and why
2. Consider ticker-specific config overrides
3. Investigate confidence scoring (may be downweighting GARCH)
4. Re-run with adjusted thresholds

### If Failed (RMSE >1.5x)
1. Deep dive into logs - why is GARCH not dominant?
2. Check if GARCH forecasts are actually good (may be data quality issue)
3. Verify code changes were applied correctly
4. Consider reverting and debugging in isolation

---

## Appendix: Expected Log Patterns

### Good Patterns (What We Want to See)

```
INFO - forcester_ts.forecaster - GARCH config loaded: p=1, q=1
INFO - forcester_ts.forecaster - SARIMAX config loaded: kwargs keys=[...], max_p=3
INFO - forcester_ts.forecaster - SAMoSSA config loaded: window_length=60, n_components=8

INFO - forcester_ts.ensemble - derive_model_confidence: confidence={'sarimax': 0.52, 'garch': 0.85, 'samossa': 0.41, 'mssa_rl': 0.38}
INFO - forcester_ts.ensemble - EnsembleCoordinator.select_weights: selected {'garch': 0.85, 'sarimax': 0.10, 'samossa': 0.05} with score=0.724

INFO - forcester_ts.forecaster - ENSEMBLE build_complete: weights={'garch': 0.85, 'sarimax': 0.10, 'samossa': 0.05}
```

### Bad Patterns (What Indicates Failure)

```
ERROR - forcester_ts.forecaster - GARCH fit_failed: <some error>
ERROR - forcester_ts.ensemble - No valid models for ensemble (all failed)
WARNING - forcester_ts.ensemble - EnsembleCoordinator.select_weights: no candidate had positive score

INFO - forcester_ts.ensemble - EnsembleCoordinator.select_weights: selected {'sarimax': 0.6, 'samossa': 0.4} with score=0.532
# ^ BAD: GARCH not in weights at all!
```

---

**Test Status:** ⏳ RUNNING
**Last Updated:** 2026-01-20
**Test Result:** [TO BE FILLED AFTER COMPLETION]
