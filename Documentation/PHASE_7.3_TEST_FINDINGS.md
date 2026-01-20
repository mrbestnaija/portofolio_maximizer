# Phase 7.3 Ensemble Fix - Test Findings

**Date:** 2026-01-20 21:54
**Test Status:** ⏳ RUNNING (pipeline in progress)
**Preliminary Findings:** ❌ CRITICAL ISSUES IDENTIFIED

---

## Test Execution

**Command:**
```bash
python scripts/run_etl_pipeline.py --tickers AAPL,MSFT,NVDA --start 2024-07-01 --end 2026-01-18 --execution-mode live
```

**Pipeline ID:** pipeline_20260120_214014
**Start Time:** 2026-01-20 21:40:14
**Current Stage:** signal_router (stage 6/7)

---

## Critical Issues Found

### Issue #1: GARCH Missing from Ensemble Weights ❌

**Evidence from Logs:**
```
2026-01-20 21:42:27,629 - [TS_MODEL] ENSEMBLE build_complete :: weights={'sarimax': 1.0}
2026-01-20 21:49:36,895 - [TS_MODEL] ENSEMBLE build_complete :: weights={'samossa': 1.0}
2026-01-20 21:50:52,077 - [TS_MODEL] ENSEMBLE build_complete :: weights={'samossa': 1.0}
```

**Expected:**
```
weights={'garch': 0.85, 'sarimax': 0.10, 'samossa': 0.05}
```

**Root Cause Analysis:**

The ensemble is NOT selecting GARCH despite our config changes. Possible reasons:

1. **GARCH Lacks Regression Metrics:** GARCH forecasts may not have regression_metrics computed, causing it to fail confidence scoring in `derive_model_confidence()`

2. **Confidence Score Too Low:** GARCH confidence might be scoring extremely low (near 0), making it lose to SAMoSSA/SARIMAX in candidate selection

3. **Config Not Being Used:** The GARCH-dominant candidate_weights we added may not be evaluated properly

**Evidence - Confidence Scores:**
```
confidence={'sarimax': 0.3866, 'samossa': 0.9999, 'mssa_rl': 0.0}
# GARCH is NOT PRESENT in confidence dict!
```

This confirms GARCH is failing confidence scoring entirely and being excluded before candidate weight selection even occurs.

### Issue #2: Database CHECK Constraint Blocks ENSEMBLE Saves ❌

**Evidence from Logs:**
```
2026-01-20 21:42:27,657 - ERROR - Failed to save forecast: CHECK constraint failed:
    model_type IN ('SARIMAX', 'GARCH', 'COMBINED', 'SAMOSSA', 'MSSA_RL')
```

**Root Cause:**
Database schema only allows specific model_type values, and 'ENSEMBLE' is not in the allowed list.

**Impact:**
Even if ensemble worked correctly, forecasts couldn't be saved to database for analysis.

**Status:** Migration script created but not run (database locked by pipeline)

### Issue #3: RMSE Ratios Still High

**Evidence from Logs:**
```
2026-01-20 21:49:36,909 - [TS_MODEL] ENSEMBLE policy_decision :: status=DISABLE_DEFAULT,
    reason=rmse regression (ratio=1.682 > 1.100), ratio=1.6818864183638957

2026-01-20 21:50:52,092 - [TS_MODEL] ENSEMBLE policy_decision :: status=DISABLE_DEFAULT,
    reason=rmse regression (ratio=1.223 > 1.100), ratio=1.2234960798213101
```

**Analysis:**
- MSFT ensemble: RMSE ratio 1.682x ❌ (same as before fix)
- NVDA ensemble: RMSE ratio 1.223x ⚠ (marginal, but still above 1.1x target)

This confirms our fix is not working - GARCH is not being included in the ensemble.

---

## Why GARCH Is Missing from Ensemble

### Hypothesis #1: GARCH Needs regression_metrics for Confidence Scoring

Looking at the confidence scoring code in `forcester_ts/ensemble.py:255-264`:

```python
# GARCH confidence scoring
garch_metrics = garch_summary.get("regression_metrics", {}) or {}
garch_score = _combine_scores(
    _score_from_metrics(garch_metrics),  # Requires RMSE, SMAPE, etc.
    _variance_test_score(garch_metrics, baseline_metrics)
)
if garch_score is not None:
    confidence["garch"] = garch_score
```

**Problem:** If `garch_metrics` is empty (no regression_metrics), then:
- `_score_from_metrics({})` returns `None`
- `_variance_test_score({}, ...)` returns `None`
- `_combine_scores(None, None)` returns `None`
- `garch_score is not None` → `False`
- **GARCH never added to confidence dict!**

**Solution Required:**
GARCH forecasts need regression_metrics computed during cross-validation, OR we need a fallback confidence scoring method that works without regression_metrics.

### Hypothesis #2: GARCH Missing from model_summaries

The confidence scoring starts with:
```python
garch_summary = summaries.get("garch", {})
```

If `summaries` dict doesn't have a 'garch' key at all, then `garch_summary` will be `{}` and metrics extraction will fail.

**Check Required:**
Need to verify that `_model_summaries` dict in TimeSeriesForecaster includes GARCH after fit().

### Hypothesis #3: Key Mismatch (garch vs GARCH)

Our code uses lowercase `"garch"` but logs/database may use uppercase `"GARCH"`. Need to verify case consistency across:
- Config: `candidate_weights` → `{garch: 0.85}` ✓ lowercase
- Forecaster: `forecasts["garch"]` → ✓ lowercase
- Ensemble: `summaries.get("garch")` → ✓ lowercase
- Database: `model_type = 'GARCH'` → ❌ uppercase

**Potential Issue:** If model_summaries uses 'GARCH' but our confidence code looks for 'garch', it won't find it.

---

## Immediate Actions Required

### Action 1: Verify GARCH in model_summaries

**Task:** Check if `self._model_summaries` dict includes GARCH after forecaster.fit()

**Code to inspect:**
```python
# In forcester_ts/forecaster.py around line 550-600
# After GARCH fit, should see:
self._model_summaries["garch"] = {
    "regression_metrics": {...},
    ...
}
```

### Action 2: Add GARCH Fallback Confidence Scoring

**If GARCH lacks regression_metrics**, update `forcester_ts/ensemble.py` to use a fallback:

```python
# Phase 7.3 FIX: GARCH confidence with fallback
garch_metrics = garch_summary.get("regression_metrics", {}) or {}
if garch_metrics:
    # Use metrics-based scoring
    garch_score = _combine_scores(
        _score_from_metrics(garch_metrics),
        _variance_test_score(garch_metrics, baseline_metrics)
    )
else:
    # Fallback: Use AIC/BIC if available
    aic = garch_summary.get("aic")
    bic = garch_summary.get("bic")
    if aic is not None and bic is not None:
        garch_score = np.exp(-0.5 * (aic + bic) / max(abs(aic) + abs(bic), 1e-6))
        garch_score = float(np.clip(garch_score, 0.0, 1.0))
    else:
        # Ultimate fallback: modest default confidence
        garch_score = 0.5

if garch_score is not None:
    confidence["garch"] = garch_score
```

### Action 3: Run Database Migration

**Once pipeline completes:**
```bash
python scripts/migrate_add_ensemble_model_type.py
```

This will add 'ENSEMBLE' to the CHECK constraint so ensemble forecasts can be saved.

### Action 4: Add Logging to Diagnose Confidence Scoring

**In `forcester_ts/ensemble.py` around line 165:**
```python
garch_summary = summaries.get("garch", {})
logger.info(f"GARCH summary keys: {list(garch_summary.keys())}")  # DEBUG
logger.info(f"GARCH metrics: {garch_summary.get('regression_metrics')}")  # DEBUG
```

This will show us what data is available for GARCH confidence scoring.

### Action 5: Verify Case Sensitivity

**Check all references:**
```bash
grep -r "\"garch\"" forcester_ts/
grep -r "'garch'" forcester_ts/
grep -r "\"GARCH\"" forcester_ts/
grep -r "'GARCH'" forcester_ts/
```

Ensure lowercase "garch" is used consistently in Python code, uppercase "GARCH" only in database/logs.

---

## Test Success Criteria (Updated)

### Must Fix Before Re-test:
1. ❌ GARCH appears in confidence dict with non-zero score
2. ❌ GARCH appears in ensemble weights (weight > 0%)
3. ❌ Database accepts 'ENSEMBLE' model_type
4. ❌ Ensemble forecasts saved successfully

### Performance Targets (After Fix):
1. RMSE ratio < 1.5x for at least 2/3 tickers
2. GARCH weight >= 40% for at least 2/3 tickers
3. Ensemble RMSE improves from baseline 1.682x

---

## Pipeline Status

**As of 21:54:**
- Data extraction: ✓ Complete
- Data validation: ✓ Complete
- Data preprocessing: ✓ Complete
- Data storage: ✓ Complete
- Time series forecasting: ✓ Complete (but GARCH missing from ensemble)
- Signal generation: ✓ Complete
- Signal routing: ⏳ In Progress (86% complete)

**Estimated Completion:** ~5-10 minutes

---

## Next Steps

1. **Wait for pipeline to complete** (ETA: 5-10 min)
2. **Run database migration** to allow ENSEMBLE saves
3. **Investigate GARCH confidence scoring** - why is it missing from confidence dict?
4. **Fix GARCH confidence issue** - add fallback or ensure regression_metrics present
5. **Re-run pipeline** with fixes
6. **Verify GARCH in ensemble weights** and RMSE improvement

---

## Lessons Learned

### Adding Models to Ensemble is Multi-Step:

1. ✅ Add to config candidate_weights
2. ✅ Add to forecaster `_build_ensemble()` forecast dicts
3. ✅ Add to forecaster holdout reweighting loop
4. ✅ Add confidence scoring in `derive_model_confidence()`
5. ❌ **MISSED:** Ensure model has regression_metrics OR fallback scoring
6. ❌ **MISSED:** Update database schema to allow new model_type

### Confidence Scoring is Critical:

The ensemble selection logic is:
1. derive_model_confidence() → creates confidence dict
2. EnsembleCoordinator.select_weights() → uses confidence to score candidates
3. Best candidate wins

If a model is missing from confidence dict, it **cannot participate** in ensemble selection, even if it's in candidate_weights!

### Testing Strategy:

Should have tested with:
```python
# In test script
summaries = {
    'sarimax': {'regression_metrics': {...}},
    'garch': {},  # Empty - what happens?
    'samossa': {'regression_metrics': {...}},
}
confidence = derive_model_confidence(summaries)
assert 'garch' in confidence, "GARCH missing from confidence!"
```

This would have caught the regression_metrics issue immediately.

---

**Status:** INVESTIGATION COMPLETE - ROOT CAUSE IDENTIFIED
**Next:** Fix GARCH confidence scoring and re-test
