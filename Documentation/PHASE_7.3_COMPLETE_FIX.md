# Phase 7.3 Ensemble GARCH Integration - Complete Fix

**Date:** 2026-01-20
**Status:** ✅ CODE COMPLETE - Ready for Testing
**Critical Bug Fixed:** GARCH missing from regression_metrics evaluation

---

## Root Cause Analysis

### The Missing Line

In `forcester_ts/forecaster.py` at line ~906, the regression metrics evaluation was missing GARCH:

**BEFORE (Buggy Code):**
```python
_evaluate_model("sarimax", self._latest_results.get("sarimax_forecast"))
_evaluate_model("samossa", self._latest_results.get("samossa_forecast"))
_evaluate_model("mssa_rl", self._latest_results.get("mssa_rl_forecast"))
# GARCH missing!
```

**AFTER (Fixed):**
```python
_evaluate_model("sarimax", self._latest_results.get("sarimax_forecast"))
_evaluate_model("garch", self._latest_results.get("garch_forecast"))  # ← ADDED
_evaluate_model("samossa", self._latest_results.get("samossa_forecast"))
_evaluate_model("mssa_rl", self._latest_results.get("mssa_rl_forecast"))
```

### Impact Chain

This single missing line caused a cascade of failures:

1. **No regression_metrics for GARCH**
   - GARCH forecasts generated but never evaluated
   - No RMSE, SMAPE, tracking_error, directional_accuracy computed

2. **No confidence score for GARCH**
   - `derive_model_confidence()` requires regression_metrics
   - Without metrics → `_score_from_metrics({})` returns `None`
   - Without score → GARCH never added to confidence dict

3. **GARCH excluded from ensemble**
   - `EnsembleCoordinator.select_weights()` uses confidence dict
   - Models not in confidence dict cannot participate
   - Even though config had `{garch: 0.85}` weights, GARCH had 0% confidence → excluded

4. **Ensemble still uses poor models**
   - Falls back to SAMoSSA (confidence 0.9999) or SARIMAX
   - RMSE ratio remains high (1.682x, 1.223x)
   - Barbell policy blocks production use

---

## Complete Fix Summary

### All Changes Made (Chronological)

#### 1. Config: Add GARCH-Dominant Candidate Weights
**File:** `config/forecasting_config.yml`
**Lines:** 69-83

```yaml
candidate_weights:
  # Phase 7.3 FIX: Add GARCH-dominant weights
  - {garch: 0.85, sarimax: 0.10, samossa: 0.05}
  - {garch: 0.70, samossa: 0.20, mssa_rl: 0.10}
  - {garch: 0.60, sarimax: 0.25, samossa: 0.15}
  # Original candidates (fallback)
  - {sarimax: 0.6, samossa: 0.4}
  - {sarimax: 0.45, samossa: 0.35, mssa_rl: 0.2}
  - {sarimax: 0.5, mssa_rl: 0.5}
  # Pure model fallbacks
  - {garch: 1.0}
  - {samossa: 1.0}
  - {mssa_rl: 1.0}
```

**Why:** Allows ensemble to select GARCH-dominant blends when confidence supports it.

#### 2. Forecaster: Include GARCH in Ensemble Forecast Dicts
**File:** `forcester_ts/forecaster.py`
**Lines:** 708-725

```python
forecasts = {
    "sarimax": self._extract_series(results.get("sarimax_forecast")),
    "garch": self._extract_series(results.get("garch_forecast")),  # ← ADDED
    "samossa": self._extract_series(results.get("samossa_forecast")),
    "mssa_rl": self._extract_series(results.get("mssa_rl_forecast")),
}
# Same for lowers and uppers
```

**Why:** Makes GARCH forecasts available for ensemble blending.

#### 3. Forecaster: Include GARCH in Holdout Reweighting
**File:** `forcester_ts/forecaster.py`
**Line:** 939

```python
for model in ("sarimax", "garch", "samossa", "mssa_rl"):  # ← "garch" added
    rmse_val = (metrics_map.get(model) or {}).get("rmse")
    if isinstance(rmse_val, (int, float)) and float(rmse_val) >= 0:
        rmse_by_model[model] = float(rmse_val)
```

**Why:** Allows GARCH to participate in inverse-RMSE reweighting after holdout evaluation.

#### 4. Ensemble: Add GARCH Confidence Scoring
**File:** `forcester_ts/ensemble.py`
**Lines:** 165, 255-264

```python
garch_summary = summaries.get("garch", {})  # ← Line 165

# GARCH confidence scoring - Phase 7.3 addition
garch_metrics = garch_summary.get("regression_metrics", {}) or {}
garch_score = _combine_scores(
    _score_from_metrics(garch_metrics),
    _variance_test_score(garch_metrics, baseline_metrics)
    if baseline_metrics
    else None,
)
if garch_score is not None:
    confidence["garch"] = garch_score
```

**Why:** Computes confidence score for GARCH based on regression metrics (once they exist).

#### 5. **CRITICAL FIX:** Add GARCH to Regression Metrics Evaluation
**File:** `forcester_ts/forecaster.py`
**Line:** 907

```python
_evaluate_model("garch", self._latest_results.get("garch_forecast"))  # ← ADDED
```

**Why:** **This is the critical missing piece.** Without this, GARCH never gets regression_metrics computed, causing all downstream failures. This single line enables the entire ensemble integration.

#### 6. Pipeline: Change Ensemble Model Type
**File:** `scripts/run_etl_pipeline.py`
**Line:** 2133

```python
'model_type': 'ENSEMBLE',  # Was 'COMBINED'
```

**Why:** Clearer distinction between ensemble and individual COMBINED forecast.

#### 7. Diagnostics: Flexible Ensemble Key Matching
**File:** `scripts/run_ensemble_diagnostics.py`
**Lines:** 280-288

```python
ensemble_key = None
for key in ['ensemble', 'ENSEMBLE', 'combined', 'COMBINED']:
    if key.lower() in [k.lower() for k in model_forecasts.keys()]:
        ensemble_key = next(k for k in model_forecasts.keys() if k.lower() == key.lower())
        break
```

**Why:** Diagnostics work regardless of model_type naming convention.

---

## Database Migration (Pending)

**Issue:** Database CHECK constraint only allows: SARIMAX, GARCH, COMBINED, SAMOSSA, MSSA_RL

**Fix:** Run migration script to add 'ENSEMBLE':

```bash
python scripts/migrate_add_ensemble_model_type.py
```

**Status:** Script created, waiting for pipeline to complete before running (database currently locked).

---

## Expected Behavior After Fix

### During Forecasting:

1. **GARCH fit and forecast** (already working)
   ```
   [TS_MODEL] GARCH fit_start :: points=209
   [TS_MODEL] GARCH fit_complete :: order={'p': 3, 'q': 1}, aic=979.27
   [TS_MODEL] GARCH forecast_complete
   ```

2. **GARCH regression_metrics computed** (NEW - from fix)
   ```
   [DEBUG] Evaluating GARCH: metrics={'rmse': 30.64, 'smape': 0.12, ...}
   ```

3. **GARCH confidence score derived** (NEW - from fix)
   ```
   [ENSEMBLE] derive_model_confidence: confidence={'sarimax': 0.52, 'garch': 0.85, 'samossa': 0.41}
   ```

4. **GARCH-dominant weights selected** (NEW - from fix)
   ```
   [ENSEMBLE] build_complete :: weights={'garch': 0.85, 'sarimax': 0.10, 'samossa': 0.05}
   ```

5. **RMSE ratio improves** (NEW - expected outcome)
   ```
   [ENSEMBLE] policy_decision :: status=APPROVED, ratio=1.12 < 1.100
   ```

### Database Records:

```sql
SELECT model_type, COUNT(*), AVG(forecast_value)
FROM time_series_forecasts
WHERE ticker = 'AAPL'
GROUP BY model_type;

-- Expected output:
-- SARIMAX | 30 | 245.67
-- GARCH   | 30 | 248.32
-- SAMoSSA | 30 | 243.21
-- MSSA-RL | 30 | 241.89
-- ENSEMBLE| 30 | 248.01  ← Should now exist with value near GARCH
```

---

## Verification Steps

### 1. Check GARCH Has Regression Metrics

After pipeline runs:
```bash
# Check logs for GARCH metrics
grep "garch.*rmse\|GARCH.*metrics" logs/phase7.3_ensemble_test.log

# Expected to see:
# metrics_map={'sarimax': {...}, 'garch': {...}, 'samossa': {...}}
```

### 2. Verify GARCH in Confidence Dict

```bash
# Check ensemble confidence calculation
grep "confidence.*garch" logs/phase7.3_ensemble_test.log

# Expected to see:
# confidence={'sarimax': 0.XX, 'garch': 0.XX, 'samossa': 0.XX}
```

### 3. Verify GARCH in Ensemble Weights

```bash
# Check ensemble weights
grep "ENSEMBLE build_complete" logs/phase7.3_ensemble_test.log

# Expected to see:
# weights={'garch': 0.85, ...} or weights={'garch': 1.0}
# NOT: weights={'samossa': 1.0} (what we saw before fix)
```

### 4. Verify RMSE Ratio Improvement

```bash
# Check ensemble RMSE ratio
grep "ENSEMBLE policy_decision" logs/phase7.3_ensemble_test.log

# Before fix:
# ratio=1.682 > 1.100 ❌

# After fix (expected):
# ratio=1.15 < 1.200 ✓ or
# ratio=1.08 < 1.100 ✓✓
```

### 5. Run Comprehensive Diagnostics

```bash
python scripts/run_ensemble_diagnostics.py --ticker AAPL --days 30
python scripts/check_ensemble_weights.py --ticker AAPL
```

**Expected output:**
```
GARCH Weight: 85.00%
✓ GARCH-dominant ensemble (>= 60%)

RMSE Analysis:
   Best Single Model: GARCH (RMSE=30.64)
   Ensemble RMSE:     35.12
   RMSE Ratio:        1.146x
   ✓ GOOD: Ensemble within 20% of best model
```

---

## Success Criteria

### Code Complete ✅
- [x] GARCH in config candidate_weights
- [x] GARCH in forecaster ensemble blend dicts
- [x] GARCH in holdout reweighting loop
- [x] GARCH confidence scoring implemented
- [x] **GARCH in regression_metrics evaluation** ← CRITICAL FIX
- [x] Ensemble model_type updated
- [x] Diagnostics updated
- [x] Migration script created

### Testing Pending ⏳
- [ ] Pipeline runs without errors
- [ ] GARCH appears in confidence dict (weight > 0)
- [ ] GARCH appears in ensemble weights (weight > 40%)
- [ ] RMSE ratio < 1.5x (acceptable)
- [ ] RMSE ratio < 1.2x (good)
- [ ] RMSE ratio < 1.1x (target)
- [ ] Database accepts ENSEMBLE records
- [ ] Diagnostics run successfully

---

## Files Modified

| File | Lines | Change | Critical? |
|------|-------|--------|-----------|
| [config/forecasting_config.yml](../config/forecasting_config.yml) | 69-83 | Add GARCH candidate weights | Medium |
| [forcester_ts/forecaster.py](../forcester_ts/forecaster.py) | 708-725 | Include GARCH in blend dicts | Medium |
| [forcester_ts/forecaster.py](../forcester_ts/forecaster.py) | 907 | **Add GARCH metrics eval** | **🔴 CRITICAL** |
| [forcester_ts/forecaster.py](../forcester_ts/forecaster.py) | 939 | Include GARCH in reweighting | Medium |
| [forcester_ts/ensemble.py](../forcester_ts/ensemble.py) | 165, 255-264 | Add GARCH confidence scoring | High |
| [scripts/run_etl_pipeline.py](../scripts/run_etl_pipeline.py) | 2133 | Change to ENSEMBLE type | Low |
| [scripts/run_ensemble_diagnostics.py](../scripts/run_ensemble_diagnostics.py) | 280-288 | Flexible ensemble matching | Low |

**Most Critical Change:** `forcester_ts/forecaster.py:907` - Without this single line, all other changes are ineffective.

---

## Timeline

| Time | Event |
|------|-------|
| 21:40 | Pipeline started (first attempt) |
| 21:42 | Discovered GARCH missing from ensemble weights |
| 21:43 | Discovered database CHECK constraint issue |
| 21:54 | **Root cause identified:** GARCH missing from regression_metrics evaluation |
| 21:57 | **Critical fix applied:** Added GARCH to `_evaluate_model()` calls |
| 22:00 | Waiting for current pipeline to complete |
| TBD | Run database migration |
| TBD | Re-run pipeline with complete fix |
| TBD | Verify RMSE improvement and GARCH integration |

---

## Lessons Learned

### 1. Ensemble Integration Requires Multiple Touch Points

Adding a model to the ensemble isn't just config - it requires updates across:
- Config (candidate weights)
- Forecaster (forecast dicts, holdout eval, **regression_metrics**)
- Ensemble (confidence scoring)
- Database (schema constraints)
- Diagnostics (key matching)

**Missing any one piece breaks the entire chain.**

### 2. Regression Metrics Are the Gateway

The confidence scoring system requires regression_metrics as its primary input. Without them:
- No confidence score
- No ensemble participation
- Model effectively doesn't exist for ensemble purposes

**Always verify metrics computation when adding models.**

### 3. Silent Failures in Ensemble Selection

The ensemble coordinator silently ignores models with zero confidence. No error, no warning - just exclusion. This made debugging harder.

**Better logging:** "Model 'garch' excluded from ensemble (confidence=0 or missing)"

### 4. Test Coverage Gap

We had no test that verified:
```python
# After forecaster.fit() and forecast()
for model in ['sarimax', 'garch', 'samossa', 'mssa_rl']:
    assert model in forecaster._model_summaries
    assert 'regression_metrics' in forecaster._model_summaries[model]
```

This test would have caught the bug immediately.

---

## Next Steps

1. ⏳ **Wait for current pipeline** to complete (~5 min remaining)
2. 🔧 **Run database migration** to allow ENSEMBLE records
3. ▶️ **Re-run pipeline** with complete fix on AAPL, MSFT, NVDA
4. ✅ **Verify all success criteria** met
5. 📊 **Generate diagnostic reports** for all tickers
6. 📝 **Update production documentation** with results
7. 🚀 **Deploy to production** if RMSE ratio < 1.1x achieved

---

**Status:** READY FOR TESTING
**Confidence:** HIGH - Root cause identified and fixed
**Risk:** LOW - Single-line fix with clear impact chain
