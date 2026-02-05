# Barbell Policy Test Plan - January 19, 2026

## Objective

Test if the barbell policy is **too conservative** (blocking profitable signals) or if the forecasting models genuinely need improvement by temporarily relaxing the policy thresholds.

## Background

### Previous Results (With Strict Policy)
- **Signal Pass Rate**: 28.6% (6/21 signals)
- **Passing Tickers**: AAPL (100%), MSFT (100%), MTN (100%)
- **Ensemble Approval**: 0% (15/15 forecasts blocked)
- **Policy Blocks**:
  - `RESEARCH_ONLY`: 11 forecasts (73%)
  - `DISABLE_DEFAULT`: 4 forecasts (27%)

### Policy Block Reasons
1. **No margin lift**: Ensemble RMSE improvement < 2% vs best single model
2. **RMSE regression**: Ensemble RMSE > 1.1x best single model (ratio: 1.22x - 1.68x)

## Hypothesis

**H0** (Null): Barbell policy is correctly blocking bad forecasts; models need fundamental improvement
**H1** (Alternative): Barbell policy is too strict; models are close to profitability but blocked by conservative thresholds

## Test Configuration Changes

### File: `config/forecaster_monitoring.yml`

#### Change 1: Promotion Margin (Line 51)
```yaml
# BEFORE
promotion_margin: 0.02  # Requires 2% RMSE improvement

# AFTER (TESTING ONLY)
promotion_margin: 0.0   # Allow any improvement
```

**Impact**: Eliminates "no margin lift" blocker → converts RESEARCH_ONLY to APPROVED

#### Change 2: Max RMSE Ratio (Line 42)
```yaml
# BEFORE
max_rmse_ratio_vs_baseline: 1.1  # Max 10% regression

# AFTER (TESTING ONLY)
max_rmse_ratio_vs_baseline: 1.5  # Allow up to 50% regression
```

**Impact**: Allows forecasts with ratio 1.22x - 1.68x to pass → converts DISABLE_DEFAULT to KEEP

### Backup Created
- Original config saved to: `config/forecaster_monitoring.yml.backup_20260119`
- **CRITICAL**: Revert changes after testing before production use

## Expected Outcomes

### Scenario A: Models Are Close to Profitability (H1 True)

If the barbell policy is too conservative:
- **Ensemble Approval Rate**: 0% → 60-80% (9-12 of 15 forecasts approved)
- **Signal Pass Rate**: 28.6% → 45-60% (9-13 of 21 signals)
- **Trade Execution**: 0 → 3-6 trades (AAPL, MSFT, NVDA)
- **Hypothetical P&L**: Positive or breakeven

**Implications**:
- ✅ Models are production-ready with relaxed policy
- ✅ Can negotiate policy thresholds (e.g., promotion_margin: 0.02 → 0.01)
- ✅ Fast path to profitability (Week 2 instead of Week 3)

### Scenario B: Models Need Fundamental Improvement (H0 True)

If forecasts are genuinely poor quality:
- **Ensemble Approval Rate**: 0% → 20-40% (3-6 of 15 forecasts approved)
- **Signal Pass Rate**: 28.6% → 35-45% (7-9 of 21 signals)
- **Trade Execution**: 0 → 1-2 trades (low confidence)
- **Hypothetical P&L**: Negative or highly volatile

**Implications**:
- ❌ Policy is correctly filtering bad forecasts
- ❌ Models need hyperparameter tuning (Phase 7.3)
- ❌ Slower path to profitability (Week 3-4 timeline)

### Scenario C: Mixed Results (Both H0 and H1 Partially True)

Some models pass, some still fail:
- **Ensemble Approval Rate**: 40-60% (6-9 of 15 forecasts)
- **Signal Pass Rate**: 35-50% (7-10 of 21 signals)
- **Per-Ticker Results**: AAPL/MSFT pass, others fail

**Implications**:
- 🔶 Policy is appropriate for average models
- 🔶 Focus tuning on failing tickers (AIG, BIL, KCB, etc.)
- 🔶 Selective production deployment (AAPL, MSFT, MTN only)

## Success Criteria

### Primary Metrics

| Metric | Baseline | Target (Success) | Measurement |
|--------|----------|------------------|-------------|
| Ensemble Approval Rate | 0% | >50% | Count APPROVED / total forecasts |
| Signal Pass Rate | 28.6% | >40% | Count PASS / total signals |
| Trade Executions | 0 | >2 | Count in trade_executions table |

### Secondary Metrics

| Metric | How to Measure | Success Threshold |
|--------|----------------|-------------------|
| Hypothetical P&L | Sum of realized_pnl | >$0 (profitable) |
| Win Rate | Profitable trades / total trades | >50% |
| Average Trade P&L | Total P&L / number of trades | >$50/trade |

## Test Execution

### Step 1: Run Pipeline (IN PROGRESS)
```bash
python scripts/run_etl_pipeline.py \
  --tickers AAPL,MSFT,NVDA \
  --start 2023-07-01 \
  --end 2026-01-19 \
  --execution-mode live \
  --enable-llm
```

**Expected Runtime**: 30-40 minutes
**Log File**: `pipeline_test_relaxed_policy.log`

### Step 2: Analyze Results
```bash
# Check ensemble approvals
grep "policy_decision" pipeline_test_relaxed_policy.log | grep -c "APPROVED"

# Check signal pass rate
python scripts/analyze_pipeline_run.py

# Check trade executions
python scripts/test_production_metrics.py
```

### Step 3: Calculate Hypothetical P&L
```sql
SELECT
  COUNT(*) as total_trades,
  SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
  SUM(realized_pnl) as total_pnl,
  AVG(realized_pnl) as avg_pnl_per_trade,
  MIN(realized_pnl) as worst_loss,
  MAX(realized_pnl) as best_win
FROM trade_executions
WHERE timestamp >= '2026-01-19'
  AND execution_mode = 'live'
```

### Step 4: Decision Matrix

Based on results, choose next action:

| Ensemble Approval | Signal Pass Rate | Trade P&L | Decision |
|-------------------|------------------|-----------|----------|
| >60% | >45% | Positive | ✅ **Negotiate policy** (promotion_margin: 0.01) |
| >60% | >45% | Negative | ⚠️ **Refine signal validation** thresholds |
| 40-60% | 35-45% | Positive | 🔶 **Selective deployment** (AAPL, MSFT only) |
| 40-60% | 35-45% | Negative | 🔶 **Mixed: tune top performers** |
| <40% | <35% | Any | ❌ **Models need work** (Phase 7.3 tuning) |

## Risks and Mitigations

### Risk 1: Test Data Contamination
**Mitigation**: Use `production_only=True` filter in all metrics

### Risk 2: Overfitting to Test Period
**Mitigation**: Cross-validate with multiple date ranges (2023 vs 2024 vs 2025)

### Risk 3: Accidental Production Deployment
**Mitigation**:
- Keep `execution_mode=live` (paper trading only)
- Backup config created: `forecaster_monitoring.yml.backup_20260119`
- Revert changes immediately after test

### Risk 4: False Positive (Lucky Trades)
**Mitigation**: Require minimum 5 trades for statistical significance

## Rollback Plan

If test shows negative results or system instability:

```bash
# 1. Stop any running pipelines
pkill -f run_etl_pipeline

# 2. Restore original config
cp config/forecaster_monitoring.yml.backup_20260119 config/forecaster_monitoring.yml

# 3. Verify restoration
git diff config/forecaster_monitoring.yml

# 4. Re-run baseline test
python scripts/run_etl_pipeline.py --tickers AAPL --start 2024-01-01 --end 2026-01-01 --execution-mode synthetic
```

## Post-Test Actions

### If Policy is Too Strict (Scenario A)
1. Negotiate policy thresholds:
   - `promotion_margin: 0.02` → `0.01` (1% improvement required)
   - `max_rmse_ratio_vs_baseline: 1.1` → `1.2` (allow 20% regression)
2. Document justification in config comments
3. Run full validation suite with new thresholds
4. Deploy to production with monitoring

### If Models Need Improvement (Scenario B)
1. Revert config to strict policy
2. Implement Phase 7.3: Hyperparameter tuning
   - Increase SARIMAX order: (2,1,1) → (3,1,2)
   - Tune SAMoSSA window: 40 → 60
   - Enable seasonal decomposition
3. Re-test with improved models
4. Iterate until >45% pass rate achieved

### If Mixed Results (Scenario C)
1. Implement per-ticker policy overrides in config
2. Deploy AAPL, MSFT, MTN to production
3. Continue tuning AIG, BIL, KCB, etc. in research mode
4. Gradual rollout as tickers pass validation

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Config changes | 10 min | ✅ Complete |
| Pipeline execution | 30-40 min | 🔄 In Progress |
| Results analysis | 15 min | ⏳ Pending |
| Decision & action | 30-60 min | ⏳ Pending |
| **Total** | **~2 hours** | 🎯 On Track |

## Success Definition

**Test is successful if**:
1. We gain actionable insight into policy vs model quality
2. We identify a clear path forward (negotiate policy OR tune models)
3. We can confidently estimate timeline to profitability

**Test is NOT successful if**:
1. Results are inconclusive (neither strongly positive nor negative)
2. New bugs or errors discovered
3. Trade execution logic broken

---

**Test Started**: 2026-01-19 23:50 UTC
**Test Owner**: Phase 8.2 - Barbell Policy Investigation
**Risk Level**: LOW (paper trading only, config backed up)
**Expected Completion**: 2026-01-20 01:50 UTC
