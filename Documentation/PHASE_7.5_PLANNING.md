# Phase 7.5 Planning: Next Steps After GARCH Ensemble Integration

**Planning Date**: 2026-01-24
**Previous Phase**: 7.4 - GARCH Ensemble Integration (✅ COMPLETE)
**Status**: Planning phase selection

---

## Phase 7.4 Completion Summary

**Achievements**:
- ✅ Fixed ensemble config preservation bug (100% GARCH selection)
- ✅ Integrated quantile-based confidence calibration (0.3-0.9 range)
- ✅ Migrated database for ENSEMBLE support (360 records preserved)
- ✅ Validated across single-ticker (AAPL) and multi-ticker (AAPL/MSFT/NVDA)
- ✅ Achieved 29% RMSE improvement (1.470 → 1.043)

**Current System State**:
- Ensemble consistently selects GARCH at 85% weight
- 9 candidate configurations evaluated in every build
- Zero database errors, production-ready on Windows
- Comprehensive documentation (11 files created)
- Git commits: b02b0ee (Phase 7.4), 050e065 (checklist update)

---

## Phase 7.5 Options Analysis

### Option A: Regime Detection Integration 🌟 RECOMMENDED

**Description**: Integrate [forcester_ts/regime_detector.py](../forcester_ts/regime_detector.py) to enable adaptive ensemble weights based on detected market regimes.

#### Technical Details

**File Ready**: [forcester_ts/regime_detector.py](../forcester_ts/regime_detector.py) (340 lines, complete implementation)

**Regime Types** (6 total):
1. LIQUID_RANGEBOUND - Low volatility, mean-reverting
2. HIGH_VOL_TRENDING - Directional movement with high volatility
3. LOW_VOL_TRENDING - Smooth directional movement
4. CHOPPY_HIGH_VOL - High volatility without clear direction
5. TRANSITIONAL - Regime change in progress
6. UNKNOWN - Unable to classify

**Detection Features**:
- Hurst exponent (persistence measure)
- ADF test (stationarity)
- Trend strength (directional consistency)
- Volatility regime (rolling std analysis)

**Integration Points**:
1. [forcester_ts/ensemble.py](../forcester_ts/ensemble.py) - Add regime-aware weight selection
2. [forcester_ts/forecaster.py](../forcester_ts/forecaster.py) - Detect regime before ensemble build
3. [config/forecasting_config.yml](../config/forecasting_config.yml) - Add regime-specific weight mappings

#### Expected Workflow

```python
# In TimeSeriesForecaster.forecast():
1. Detect current market regime
2. Load regime-specific ensemble weights from config
3. Build ensemble with adaptive weights
4. Log regime detection for analysis

# Example regime-specific weights:
LIQUID_RANGEBOUND:
  - SARIMAX: 0.6 (mean reversion favored)
  - GARCH: 0.3 (volatility less critical)
  - SAMoSSA: 0.1

HIGH_VOL_TRENDING:
  - GARCH: 0.7 (volatility critical)
  - SARIMAX: 0.2
  - SAMoSSA: 0.1

LOW_VOL_TRENDING:
  - SAMoSSA: 0.5 (trend patterns)
  - SARIMAX: 0.3
  - GARCH: 0.2
```

#### Implementation Tasks

**Estimated Effort**: 2-3 days

1. **Regime Detection Integration** (~4 hours)
   - Add regime detection call in forecaster.py
   - Store regime in forecast metadata
   - Log regime transitions

2. **Configuration Updates** (~2 hours)
   - Add regime-specific weights to forecasting_config.yml
   - Define fallback weights for UNKNOWN regime
   - Document regime weight rationale

3. **Ensemble Weight Selection** (~3 hours)
   - Modify ensemble.py to accept regime parameter
   - Implement regime-based weight lookup
   - Preserve quantile calibration logic

4. **Testing & Validation** (~4 hours)
   - Test regime detection on historical data
   - Validate weight switching across regimes
   - Compare performance vs static weights (Phase 7.4 baseline)

5. **Documentation** (~2 hours)
   - Create PHASE_7.5_REGIME_DETECTION.md
   - Document regime definitions and weight rationale
   - Add examples of regime-specific forecasts

#### Expected Benefits

**Performance**:
- Improved RMSE in volatile markets (GARCH dominance)
- Better trend capture in trending markets (SAMoSSA/SARIMAX)
- Adaptive behavior reduces over-fitting to single regime

**Operational**:
- Explainable model selection (regime context)
- Regime logs enable post-analysis
- Foundation for future regime-based strategies

**Risk Mitigation**:
- Reduced exposure to model weakness in specific conditions
- Fallback to current static weights if detection fails
- Feature flag for easy rollback

#### Success Criteria

- ✅ Regime detected in 95%+ of forecasts (not UNKNOWN)
- ✅ Weight switching observed across different regimes
- ✅ RMSE ratio maintains or improves vs Phase 7.4 baseline
- ✅ No increase in forecast generation time (>10%)
- ✅ Zero database or runtime errors

#### Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Regime detection too slow | High | Cache regime for N periods, only recompute periodically |
| Wrong regime classification | Medium | Conservative thresholds, UNKNOWN fallback to Phase 7.4 weights |
| Increased complexity | Medium | Feature flag `regime_detection.enabled`, default false initially |
| Documentation debt | Low | Create docs alongside implementation |

---

### Option B: Ensemble Weight Optimization

**Description**: Use [scripts/optimize_ensemble_weights.py](../scripts/optimize_ensemble_weights.py) to find optimal weights via scipy.optimize.

#### Technical Details

**File Ready**: [scripts/optimize_ensemble_weights.py](../scripts/optimize_ensemble_weights.py) (300+ lines)

**Optimization Method**:
- scipy.optimize.minimize with SLSQP
- Objective: Minimize RMSE on validation set
- Constraints: Weights sum to 1.0, all weights >= 0

**Current Weights** (Phase 7.4):
```python
# Static weights from config/forecasting_config.yml:
candidate_1: {'garch': 0.85, 'sarimax': 0.1, 'samossa': 0.05}
# ... 8 more candidates
```

#### Implementation Tasks

**Estimated Effort**: 1-2 days

1. **Historical Data Collection** (~2 hours)
   - Extract validation set forecasts from database
   - Compute actual vs predicted for each model
   - Store in optimization-ready format

2. **Weight Optimization** (~1 hour)
   - Run optimize_ensemble_weights.py
   - Validate optimized weights make sense
   - Compare to current 85% GARCH weights

3. **Configuration Update** (~1 hour)
   - Update forecasting_config.yml with optimized weights
   - Document optimization parameters used
   - Keep Phase 7.4 weights as fallback

4. **Validation** (~3 hours)
   - Run multi-ticker test with optimized weights
   - Compare RMSE to Phase 7.4 baseline
   - Check for overfitting (train vs validation performance)

5. **Documentation** (~1 hour)
   - Document optimization process
   - Justify weight changes
   - Provide rollback instructions

#### Expected Benefits

**Performance**:
- Potentially improved RMSE if current weights are suboptimal
- Data-driven rather than intuition-based weights

**Risks**:
- May overfit to validation set
- Current 85% GARCH already performs well (diminishing returns)
- Static weights don't adapt to changing market conditions

#### Success Criteria

- ✅ Optimized weights achieve better validation RMSE than Phase 7.4
- ✅ Test set RMSE doesn't degrade (no overfitting)
- ✅ Multi-ticker validation shows consistent improvement
- ✅ Weights are interpretable (not all on one model)

#### Recommendation

**Priority**: Medium (after regime detection)

**Rationale**: Current 85% GARCH is already performing well (1.043 RMSE). Optimization might provide 5-10% improvement, but regime detection offers more value through adaptability. Consider this AFTER Phase 7.5A if regime detection doesn't yield expected gains.

---

### Option C: Holdout Audit Accumulation

**Description**: Run the system 20+ times to accumulate holdout audits and transition ENSEMBLE from RESEARCH_ONLY to production status.

#### Technical Details

**Current Status**:
```python
# From logs:
[TS_MODEL] ENSEMBLE holdout_wait :: effective_audits=1, required_audits=20
[TS_MODEL] ENSEMBLE policy_decision :: status=RESEARCH_ONLY
```

**Requirement**: 20+ holdout audits with margin lift >= 0.020 (2%)

#### Implementation Tasks

**Estimated Effort**: 1 week (mostly wait time)

1. **Automated Audit Collection** (~2 hours)
   - Create script to run pipeline daily
   - Store audit results in database
   - Track progress toward 20 audits

2. **Audit Analysis** (~1 hour)
   - Monitor RMSE ratio trends
   - Check for margin lift >= 2%
   - Identify any degradation patterns

3. **Production Transition** (~1 hour)
   - Once 20 audits collected, system auto-transitions
   - Verify production status in logs
   - Document transition date

4. **Documentation** (~30 min)
   - Log audit collection progress
   - Document production criteria met

#### Expected Benefits

**Operational**:
- ENSEMBLE transitions from RESEARCH_ONLY to production
- Increased confidence in ensemble performance
- Historical audit trail for analysis

**Risks**:
- May not achieve 2% margin lift (ratio regression)
- 20 audits could take weeks to accumulate
- Low immediate value (system already works in RESEARCH_ONLY)

#### Success Criteria

- ✅ 20+ holdout audits collected
- ✅ Average RMSE ratio < 1.1 across audits
- ✅ Margin lift >= 2% achieved
- ✅ ENSEMBLE status = production (not RESEARCH_ONLY)

#### Recommendation

**Priority**: Low (background task)

**Rationale**: This is maintenance work that happens naturally as the system runs. Set up automated collection and let it accumulate in background while working on higher-value features (Options A or B).

---

## Recommended Phase 7.5 Sequence

### Primary Path: Option A (Regime Detection) 🌟

**Timeline**: 2-3 days active development

**Sequence**:
1. Day 1: Integrate regime detection into forecaster
2. Day 2: Configure regime-specific weights, test switching
3. Day 3: Validate on historical data, compare to Phase 7.4
4. Ongoing: Collect audit data (Option C) in background

**Why This Order**:
- Highest value: Adaptive weights > static optimization
- Foundation for future work: Regime awareness enables strategies
- Leverages existing code: regime_detector.py is complete
- Reversible: Feature flag allows easy rollback

### Secondary Path: Option B (Weight Optimization)

**When**: If regime detection doesn't achieve expected improvement, OR as enhancement after 7.5A

**Condition**: Only proceed if:
- Regime detection shows <5% RMSE improvement, OR
- Specific regime weights need fine-tuning

### Ongoing: Option C (Audit Accumulation)

**When**: Parallel to Options A/B (background task)

**Setup**:
```bash
# Create daily cron job or scheduled task:
# Windows Task Scheduler:
# - Trigger: Daily at 7:00 AM
# - Action: Run pipeline with auto mode
# - Log: Append to logs/daily_audits.log

./simpleTrader_env/Scripts/python.exe scripts/run_etl_pipeline.py \
    --tickers AAPL,MSFT,NVDA \
    --execution-mode auto \
    --enable-llm >> logs/daily_audits.log 2>&1
```

---

## Phase 7.5 Success Definition

### Minimum Viable Success (Phase 7.5A - Regime Detection)

- ✅ Regime detection integrated and working (95%+ classification rate)
- ✅ Ensemble weights switch based on regime
- ✅ RMSE ratio maintains Phase 7.4 baseline (<1.1 for 2/3 tickers)
- ✅ No performance regression (forecast time, errors)
- ✅ Documentation complete with examples

### Stretch Goals

- 🎯 RMSE improvement > 10% in volatile regimes (vs static weights)
- 🎯 Regime transition detection accuracy > 90%
- 🎯 10+ audits collected during development (toward Option C goal)

### Phase 7.5 Complete Criteria

**Code**:
- Regime detection integrated into TimeSeriesForecaster
- Regime-specific weights configurable in forecasting_config.yml
- Feature flag `regime_detection.enabled` working
- Tests added for regime detection logic

**Validation**:
- Multi-ticker test passed (AAPL/MSFT/NVDA)
- Regime switching observed across test period
- Performance metrics logged and analyzed

**Documentation**:
- PHASE_7.5_REGIME_DETECTION.md (implementation details)
- PHASE_7.5_VALIDATION.md (test results)
- PHASE_7.5_FINAL_SUMMARY.md (completion report)
- AGENT_DEV_CHECKLIST.md updated

**Git**:
- Changes committed to master
- Pushed to GitHub
- Commit message references Phase 7.5

---

## Alternative: Skip to Phase 8

If none of the Phase 7.5 options seem high-value, consider:

**Phase 8: Production Deployment & Monitoring**
- Set up automated daily runs
- Create alerting for anomalies
- Build performance dashboard
- Document deployment procedures

**Phase 9: Strategy Integration**
- Connect forecasts to trading signals
- Implement position sizing based on confidence
- Backtest complete strategy
- Paper trading validation

---

## Decision Matrix

| Criterion | Option A (Regime) | Option B (Optimize) | Option C (Audits) | Skip to Phase 8 |
|-----------|-------------------|---------------------|-------------------|-----------------|
| **Value** | HIGH | MEDIUM | LOW | HIGH |
| **Effort** | 2-3 days | 1-2 days | 1 week (wait) | 3-5 days |
| **Risk** | MEDIUM | LOW | LOW | MEDIUM |
| **Reversible** | YES (flag) | YES (config) | N/A | YES |
| **Builds on 7.4** | YES | YES | YES | PARTIAL |
| **Production Ready** | After validation | After validation | Enables status | Immediate |

**Recommendation**: **Option A (Regime Detection)** for maximum value and foundation for future work.

---

## Next Steps

### To Proceed with Option A (Regime Detection):

1. **Read regime_detector.py** to understand implementation
2. **Plan integration points** in forecaster.py and ensemble.py
3. **Design configuration schema** for regime-specific weights
4. **Create feature flag** in forecasting_config.yml
5. **Start implementation** with test-driven approach

### To Proceed with Option B (Weight Optimization):

1. **Extract validation data** from recent runs
2. **Run optimize_ensemble_weights.py** to get optimized weights
3. **Analyze weight changes** vs current configuration
4. **Validate on test set** to check for overfitting
5. **Update configuration** if improvement confirmed

### To Proceed with Option C (Audit Accumulation):

1. **Create automation script** for daily pipeline runs
2. **Set up scheduled task** (Windows Task Scheduler or cron)
3. **Monitor progress** weekly toward 20 audit goal
4. **Analyze audit trends** for production readiness

### To Skip to Phase 8:

1. **Review current system** for production gaps
2. **Define deployment requirements** (monitoring, alerting, recovery)
3. **Create Phase 8 planning document**
4. **Get stakeholder approval** for production rollout

---

## Resources

### Documentation
- [PHASE_7.4_FINAL_SUMMARY.md](PHASE_7.4_FINAL_SUMMARY.md) - Phase 7.4 completion details
- [AGENT_DEV_CHECKLIST.md](AGENT_DEV_CHECKLIST.md) - Updated with Phase 7.5 options
- [CLAUDE.md](../CLAUDE.md) - Agent workflow and best practices

### Code
- [forcester_ts/regime_detector.py](../forcester_ts/regime_detector.py) - Regime detection (Option A)
- [scripts/optimize_ensemble_weights.py](../scripts/optimize_ensemble_weights.py) - Weight optimization (Option B)
- [forcester_ts/ensemble.py](../forcester_ts/ensemble.py) - Current ensemble logic
- [forcester_ts/forecaster.py](../forcester_ts/forecaster.py) - Forecaster orchestration

### Configuration
- [config/forecasting_config.yml](../config/forecasting_config.yml) - Ensemble candidate weights

---

**Planning Document Created**: 2026-01-24
**Recommendation**: Option A - Regime Detection Integration
**Expected Timeline**: 2-3 days active development
**Next Action**: Read regime_detector.py and plan integration

---

## User Decision Required

**Question**: Which Phase 7.5 option should we proceed with?

**A. Regime Detection** (2-3 days, high value, adaptive weights)
**B. Weight Optimization** (1-2 days, medium value, static improvement)
**C. Audit Accumulation** (1 week wait, low value, production status)
**D. Skip to Phase 8** (3-5 days, production deployment focus)

Please indicate your preference, or request more analysis of any option.
