# Phase 7.5 Implementation Plan: Regime Detection Integration

**Phase**: 7.5 - Regime Detection Integration
**Started**: 2026-01-24
**Status**: 🔄 In Progress
**Option**: A (Regime Detection - HIGH VALUE)

---

## Implementation Overview

### Objective
Integrate [forcester_ts/regime_detector.py](../forcester_ts/regime_detector.py) into the ensemble system to enable adaptive model weights based on detected market regimes.

### Key Design Decisions

1. **Feature Flag**: `regime_detection.enabled` in `forecasting_config.yml` (default: false initially)
2. **Fallback Strategy**: If regime detection fails or disabled → use Phase 7.4 static weights
3. **Integration Point**: `TimeSeriesForecaster.forecast()` before ensemble build
4. **Weight Selection**: Regime-specific candidate reordering (preserve quantile calibration)
5. **Logging**: All regime detections logged for analysis

---

## Regime Detector Analysis

### Implementation Status
✅ **COMPLETE** - [forcester_ts/regime_detector.py](../forcester_ts/regime_detector.py) (340 lines)

### Regime Classifications

| Regime | Conditions | Recommended Models | Use Case |
|--------|-----------|-------------------|----------|
| **LIQUID_RANGEBOUND** | Low vol (<15%), weak trend, mean-reverting (H<0.5), stationary | GARCH, SARIMAX | Stable markets, volatility forecasting optimal |
| **MODERATE_RANGEBOUND** | Low vol, stationary, some trend | GARCH, SARIMAX, SAMoSSA | Mixed conditions |
| **MODERATE_TRENDING** | Medium vol/trend | SAMoSSA, GARCH, PatchTST | Trend following optimal |
| **HIGH_VOL_TRENDING** | High vol (>30%), strong trend (>60%) | SAMoSSA, PatchTST, MSSA_RL | Volatile directional moves |
| **CRISIS** | Extreme vol (>50%) | GARCH, SARIMAX | Crisis mode, defensive |
| **MODERATE_MIXED** | Default fallback | GARCH, SAMoSSA, SARIMAX | No clear regime |

### Detection Features

**Volatility Metrics**:
- Realized volatility (annualized std)
- Vol-of-vol (volatility clustering)

**Trend Metrics**:
- Trend strength (linear regression R²)
- Hurst exponent (H<0.5: mean-reverting, H>0.5: trending)

**Stationarity**:
- ADF test p-value (<0.05 = stationary)

**Tail Risk**:
- Skewness and kurtosis

### Key Methods

```python
# Main detection interface:
regime_result = detector.detect_regime(price_series, returns_series)
# Returns:
{
    'regime': 'LIQUID_RANGEBOUND',  # Classification
    'features': {...},  # All computed features
    'recommendations': ['garch', 'sarimax'],  # Preferred models
    'confidence': 0.75  # Classification confidence (0.3-0.95)
}

# Candidate reordering (already implemented):
preferred_candidates = detector.get_preferred_candidates(
    regime_result,
    all_candidates  # From forecasting_config.yml
)
# Returns candidates sorted by regime alignment
```

---

## Integration Architecture

### Configuration Schema

**File**: [config/forecasting_config.yml](../config/forecasting_config.yml)

```yaml
# NEW: Regime detection configuration
regime_detection:
  enabled: false  # Feature flag (start disabled)
  lookback_window: 60  # Days to analyze
  vol_threshold_low: 0.15  # 15% annual vol
  vol_threshold_high: 0.30  # 30% annual vol
  trend_threshold_weak: 0.30  # Weak trend boundary
  trend_threshold_strong: 0.60  # Strong trend boundary

  # Regime-specific candidate preferences (optional override)
  # If not specified, uses detector's built-in recommendations
  regime_candidate_overrides:
    LIQUID_RANGEBOUND:
      # Reorder candidates to prefer GARCH-heavy weights
      preferred_model_order: ['garch', 'sarimax', 'samossa']
    HIGH_VOL_TRENDING:
      preferred_model_order: ['samossa', 'garch', 'mssa_rl']
    # ... other regimes

# EXISTING: Ensemble candidates (Phase 7.4)
ensemble:
  enabled: true
  confidence_scaling: quantile  # Preserve Phase 7.4 calibration
  candidate_weights:  # 9 candidates as before
    - garch: 0.85
      sarimax: 0.1
      samossa: 0.05
    # ... 8 more candidates
```

### Code Changes

#### 1. TimeSeriesForecasterConfig (forcester_ts/forecaster.py)

```python
@dataclass
class TimeSeriesForecasterConfig:
    # ... existing fields ...

    # NEW: Regime detection
    regime_detection_enabled: bool = False
    regime_detection_kwargs: Dict[str, Any] = field(default_factory=dict)
```

#### 2. TimeSeriesForecaster.__init__ (forcester_ts/forecaster.py)

```python
def __init__(self, ...):
    # ... existing initialization ...

    # NEW: Initialize regime detector if enabled
    self._regime_detector: Optional[RegimeDetector] = None
    if self.config.regime_detection_enabled:
        from forcester_ts.regime_detector import RegimeDetector, RegimeConfig
        regime_config = RegimeConfig(**self.config.regime_detection_kwargs)
        self._regime_detector = RegimeDetector(regime_config)
        logger.info("Regime detection ENABLED with config: %s", regime_config)
```

#### 3. TimeSeriesForecaster.forecast() - Main Integration Point

**Current Flow** (Phase 7.4):
```
1. Fit individual models (SARIMAX, GARCH, SAMoSSA, MSSA_RL)
2. Generate forecasts from each model
3. Build ensemble with static candidate weights
4. Return ensemble forecast
```

**NEW Flow** (Phase 7.5):
```
1. Detect market regime (if enabled)
2. Fit individual models (unchanged)
3. Generate forecasts from each model (unchanged)
4. Reorder candidates based on regime (NEW)
5. Build ensemble with regime-preferred candidates
6. Log regime detection result (NEW)
7. Return ensemble forecast with regime metadata
```

**Implementation**:
```python
def forecast(self, series: pd.Series, ...) -> Dict[str, Any]:
    logger.info("[TS_MODEL] FORECAST start :: series_length=%d", len(series))

    # NEW: Detect regime before modeling
    regime_result = None
    if self._regime_detector:
        try:
            regime_result = self._regime_detector.detect_regime(series)
            logger.info(
                "[TS_MODEL] REGIME detected :: regime=%s, confidence=%.2f, features=%s",
                regime_result['regime'],
                regime_result['confidence'],
                regime_result['features']
            )
        except Exception as e:
            logger.warning("[TS_MODEL] REGIME detection failed: %s (falling back to static)", e)
            regime_result = None

    # Fit models (unchanged from Phase 7.4)
    self._fit_models(series, ...)

    # Generate forecasts (unchanged)
    forecasts = self._generate_forecasts(...)

    # NEW: Reorder candidates if regime detected
    if regime_result and self._ensemble_config.enabled:
        original_candidates = self._ensemble_config.candidate_weights
        preferred_candidates = self._regime_detector.get_preferred_candidates(
            regime_result,
            original_candidates
        )
        logger.info(
            "[TS_MODEL] REGIME candidate_reorder :: original_top=%s, preferred_top=%s",
            original_candidates[0] if original_candidates else None,
            preferred_candidates[0] if preferred_candidates else None
        )
        # Temporarily use preferred candidates for this forecast
        self._ensemble_config.candidate_weights = preferred_candidates

    # Build ensemble (existing logic, now with reordered candidates)
    ensemble_forecast = self._build_ensemble(forecasts, ...)

    # Restore original candidates (if modified)
    if regime_result and self._ensemble_config.enabled:
        self._ensemble_config.candidate_weights = original_candidates

    # Add regime metadata to result
    result = {
        'ensemble_forecast': ensemble_forecast,
        'individual_forecasts': forecasts,
        'regime': regime_result['regime'] if regime_result else 'STATIC',
        'regime_confidence': regime_result['confidence'] if regime_result else None,
        'regime_features': regime_result['features'] if regime_result else None,
    }

    return result
```

#### 4. Database Schema Update (if needed)

**Option A**: Store regime in existing `diagnostics` JSON field
```python
# In database save:
diagnostics = {
    'regime': regime_result['regime'],
    'regime_confidence': regime_result['confidence'],
    'regime_features': regime_result['features'],
    # ... existing diagnostics
}
```

**Option B**: Add `regime` column to forecasts table (more complex)
- Not recommended for Phase 7.5 (avoid migration)
- Use diagnostics JSON for now

---

## Implementation Sequence

### Step 1: Configuration Setup ✅ NEXT

**File**: [config/forecasting_config.yml](../config/forecasting_config.yml)

**Changes**:
1. Add `regime_detection` section with all thresholds
2. Set `enabled: false` initially (feature flag)
3. Document each threshold's purpose

**Testing**: Load config and verify parsing

### Step 2: Forecaster Config Extension

**File**: [forcester_ts/forecaster.py](../forcester_ts/forecaster.py)

**Changes**:
1. Add `regime_detection_enabled` and `regime_detection_kwargs` to `TimeSeriesForecasterConfig`
2. Update `_build_config_from_kwargs()` to handle regime config
3. Initialize `RegimeDetector` in `__init__` if enabled

**Testing**: Instantiate forecaster with regime config

### Step 3: Regime Detection Integration

**File**: [forcester_ts/forecaster.py](../forcester_ts/forecaster.py)

**Changes**:
1. Add regime detection call at start of `forecast()`
2. Log regime result with all features
3. Handle detection failures gracefully (fallback to static)

**Testing**: Run single forecast, verify regime logged

### Step 4: Candidate Reordering

**File**: [forcester_ts/forecaster.py](../forcester_ts/forecaster.py)

**Changes**:
1. Call `get_preferred_candidates()` after regime detection
2. Temporarily swap candidates for ensemble build
3. Restore original candidates after build
4. Log candidate reordering

**Testing**: Verify different regimes produce different candidate orders

### Step 5: Result Metadata

**File**: [forcester_ts/forecaster.py](../forcester_ts/forecaster.py)

**Changes**:
1. Add regime fields to forecast result dict
2. Ensure regime data flows to database (via diagnostics)
3. Update logging to include regime in summary

**Testing**: Check database for regime in diagnostics field

### Step 6: Multi-Ticker Validation

**Script**: Run multi-ticker pipeline with regime detection enabled

**Validation**:
1. Regime detected in 95%+ of forecasts
2. Candidate reordering observed
3. RMSE maintains or improves vs Phase 7.4
4. No performance regression (time, errors)

**Testing**: Compare logs to Phase 7.4 baseline

### Step 7: Documentation

**Files**: Create comprehensive Phase 7.5 docs

1. `PHASE_7.5_REGIME_INTEGRATION.md` - Implementation details
2. `PHASE_7.5_VALIDATION.md` - Test results and analysis
3. `PHASE_7.5_FINAL_SUMMARY.md` - Completion report

---

## Testing Strategy

### Unit Tests

**File**: [tests/test_regime_integration.py](../tests/test_regime_integration.py) (NEW)

```python
def test_regime_detector_integration():
    """Test regime detector integrates with forecaster."""
    config = TimeSeriesForecasterConfig(
        regime_detection_enabled=True,
        regime_detection_kwargs={'lookback_window': 60}
    )
    forecaster = TimeSeriesForecaster(config=config)

    # Verify detector initialized
    assert forecaster._regime_detector is not None

def test_regime_detection_disabled_fallback():
    """Test fallback when regime detection disabled."""
    config = TimeSeriesForecasterConfig(regime_detection_enabled=False)
    forecaster = TimeSeriesForecaster(config=config)

    # Verify detector not initialized
    assert forecaster._regime_detector is None

def test_candidate_reordering():
    """Test candidates reordered based on regime."""
    # Mock regime result
    regime_result = {
        'regime': 'HIGH_VOL_TRENDING',
        'recommendations': ['samossa', 'mssa_rl']
    }

    # Original candidates (Phase 7.4)
    candidates = [
        {'garch': 0.85, 'sarimax': 0.1, 'samossa': 0.05},
        {'samossa': 0.6, 'garch': 0.3, 'mssa_rl': 0.1},
    ]

    # Reorder
    detector = RegimeDetector()
    preferred = detector.get_preferred_candidates(regime_result, candidates)

    # Verify SAMoSSA-heavy candidate first
    assert 'samossa' in preferred[0]
    assert preferred[0]['samossa'] > preferred[0].get('garch', 0)
```

### Integration Tests

**Test Cases**:
1. Low vol rangebound data → LIQUID_RANGEBOUND → GARCH dominant
2. High vol trending data → HIGH_VOL_TRENDING → SAMoSSA dominant
3. Crisis data (50%+ vol) → CRISIS → Defensive weights
4. Regime transition → Different candidates selected across periods

### Validation Tests

**Multi-Ticker Test** (AAPL, MSFT, NVDA):
- Run with regime detection enabled
- Verify regimes detected correctly
- Compare RMSE to Phase 7.4 static baseline
- Check no performance regression

**Metrics to Track**:
| Metric | Phase 7.4 Baseline | Phase 7.5 Target | Status |
|--------|-------------------|------------------|--------|
| Regime classification rate | N/A | >95% (not UNKNOWN) | TBD |
| RMSE ratio (AAPL) | 1.043 | ≤1.043 (no regression) | TBD |
| RMSE ratio (MSFT) | TBD | <1.1 | TBD |
| RMSE ratio (NVDA) | TBD | <1.1 | TBD |
| Forecast time | ~6 min | <7 min (+16% max) | TBD |
| Database errors | 0 | 0 | TBD |

---

## Risk Mitigation

### Risk 1: Regime Detection Too Slow

**Mitigation**:
- Measure detection time per forecast
- If >1 second → cache regime for N forecasts
- Feature flag allows instant disable

### Risk 2: Wrong Regime Classification

**Mitigation**:
- Log all features for post-analysis
- Conservative thresholds (avoid edge cases)
- MODERATE_MIXED fallback catches unclear cases
- Can tune thresholds in config without code changes

### Risk 3: Performance Regression

**Mitigation**:
- Baseline: Phase 7.4 RMSE = 1.043 (AAPL)
- Acceptance: Phase 7.5 RMSE ≤ 1.15 (max 10% regression)
- Rollback: Disable feature flag if regression observed

### Risk 4: Candidate Reordering Breaks Ensemble

**Mitigation**:
- Preserve quantile calibration (no changes to ensemble.py logic)
- Only reorder, don't modify weights
- Restore original candidates after each forecast
- Extensive logging of reordering actions

---

## Success Criteria

### Minimum Viable Success

- ✅ Regime detection integrated and working
- ✅ Feature flag `regime_detection.enabled` functional
- ✅ Regime logged in 95%+ of forecasts
- ✅ Candidate reordering observed
- ✅ No database errors
- ✅ RMSE maintains Phase 7.4 baseline (no >10% regression)

### Stretch Goals

- 🎯 RMSE improvement >5% in volatile markets vs Phase 7.4
- 🎯 Regime transitions detected with >90% accuracy
- 🎯 Forecast time increase <5% (vs Phase 7.4's ~6 min)

### Phase 7.5 Complete

- ✅ Code: All integration points implemented
- ✅ Config: regime_detection section in forecasting_config.yml
- ✅ Tests: Unit + integration + multi-ticker validation passed
- ✅ Docs: 3 comprehensive documents created
- ✅ Git: Committed and pushed to master
- ✅ Validation: Multi-ticker test shows no regression

---

## Timeline

**Estimated**: 2-3 days

| Day | Tasks | Hours |
|-----|-------|-------|
| **Day 1** | Config setup, forecaster extension, basic integration | 6h |
| **Day 2** | Candidate reordering, testing, debugging | 6h |
| **Day 3** | Multi-ticker validation, documentation, git commit | 4h |

**Total**: ~16 hours active development

---

## Next Immediate Action

**Start with Step 1**: Configuration Setup

1. Edit [config/forecasting_config.yml](../config/forecasting_config.yml)
2. Add `regime_detection` section
3. Set `enabled: false` (feature flag)
4. Document all thresholds

**Ready to proceed?**

---

**Document Created**: 2026-01-24
**Status**: 🔄 In Progress (Step 1 next)
**Phase**: 7.5 - Regime Detection Integration
