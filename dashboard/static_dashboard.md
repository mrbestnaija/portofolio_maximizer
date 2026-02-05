# 📊 Live Ensemble Performance Dashboard

> **Ensemble status (canonical, current)**: `../Documentation/ENSEMBLE_MODEL_STATUS.md` is the single source of truth for whether the time-series ensemble is currently active and what the audit gate decision is. This dashboard is a historical snapshot (see “Generated” timestamp) and should not be treated as current system status.

**Status**: Phase 7.3 GARCH Integration - Multi-Ticker Validation
**Generated**: 2026-01-21 19:45:00
**Auto-Refresh**: Every pipeline run updates this dashboard

---

## 🎯 Multi-Ticker Performance Summary

| Ticker | Status | GARCH Weight | RMSE Ratio | vs Target | Improvement |
|--------|--------|--------------|------------|-----------|-------------|
| **MSFT** | ✅ | 85.0% | **1.037** | **-0.063** | **+38.4%** |
| AAPL | ⚠️ | 85.0% | 1.470 | +0.370 | +12.6% |
| NVDA | ⚠️ | 0.0% (SAMoSSA) | 1.453 | +0.353 | +13.6% |
| **OVERALL** | 🎯 | 14% avg | **1.386** | **+0.286** | **+17.6%** |

---

## 📈 Progress to Target

```
Baseline: 1.682 → Current: 1.386 → Target: 1.100
[████████████████████████████░░░░░░░░░░░░░░░░░░░░] 50.9%
```

**Journey Progress**: 50.9% complete (from 1.682 baseline to 1.100 target)

---

## 🔑 Key Metrics

- **Total Tickers Analyzed**: 3 (AAPL, MSFT, NVDA)
- **GARCH-Dominant (≥50% weight)**: 2 of 14 ensembles (14.3%)
- **At Target (<1.1x RMSE)**: 1 of 3 tickers (33.3%) ✅
- **Average GARCH Weight**: 57% (when GARCH selected)
- **Best Performer**: MSFT at 1.037 (3.7% BETTER than target!)
- **Worst Performer**: NVDA at 1.682 (baseline level)

---

## 🔍 Per-Ticker Deep Dive

### ✅ MSFT - TARGET ACHIEVED!
```
Status: TARGET ACHIEVED (1.037 < 1.100)
GARCH Weight: 85%
RMSE Ratio: 1.037 vs 1.100 target (-0.063, 3.7% better!)
Improvement: 38.4% from baseline
Regime: Liquid, mean-reverting → GARCH optimal
```

**Why MSFT Succeeds**:
- High liquidity → stable volatility patterns
- Volatility clustering → GARCH captures well
- Mean-reverting behavior → matches GARCH assumptions
- **Action**: Use MSFT as reference case for GARCH selection rules

---

### ⚠️ AAPL - 36% to Target
```
Status: IMPROVING (36.4% of goal reached)
GARCH Weight: 85%
RMSE Ratio: 1.470 vs 1.100 target (+0.370 gap)
Improvement: 12.6% from baseline
Regime: Moderate volatility → GARCH helping but not optimal
```

**Why AAPL Underperforms**:
- Moderate improvement but not enough
- May need mixed ensemble vs pure GARCH
- Consider adding feature-based model for directional edge
- **Action**: Test ensemble weight optimization (reduce GARCH to 60-70%, blend with SAMoSSA)

---

### ⚠️ NVDA - 39% to Target (SAMoSSA-Only)
```
Status: IMPROVING (39.4% of goal reached)
GARCH Weight: 0% (SAMoSSA selected in all 12 folds)
RMSE Ratio: 1.453 avg (range: 1.223 - 1.682)
Improvement: 13.6% from baseline
Regime: Trending, high-vol → SAMoSSA appropriate
```

**Why NVDA Avoids GARCH**:
- Strong trends (AI boom/bust cycles)
- High volatility, non-stationary
- Structural breaks (AI narrative evolving)
- SAMoSSA's spectral decomposition better for trends
- **Action**: Phase 8 - Add PatchTST for trending regimes

---

## 🤖 Automated Recommendations

### 🔴 CRITICAL - None
✅ No critical issues detected

### 🟡 HIGH Priority

**1. Low GARCH Selection Rate**
- **Issue**: GARCH selected in only 2/14 ensembles (14.3%)
- **Root Cause**: SAMoSSA confidence (EVR ~0.95) always beats GARCH (AIC/BIC ~0.60) after normalization
- **Action**: Implement confidence calibration to balance scores
- **Code**:
  ```python
  # In ensemble.py derive_model_confidence()
  # Add quantile-based calibration instead of min-max normalization
  from scipy.stats import rankdata
  normalized_confidence = rankdata(raw_confidence) / len(raw_confidence)
  ```

### 🔵 MEDIUM Priority

**2. Performance Gap to Target**
- **Issue**: 2/3 tickers above target (avg gap: 0.362)
- **Action**: Implement ensemble weight optimization using holdout data
- **Code**:
  ```python
  # Add to EnsembleCoordinator
  from scipy.optimize import minimize

  def optimize_weights(self, holdout_forecasts, holdout_actuals):
      def objective(weights):
          ensemble_pred = sum(w * f for w, f in zip(weights, holdout_forecasts))
          return np.sqrt(np.mean((ensemble_pred - holdout_actuals) ** 2))

      result = minimize(objective, x0=[0.33, 0.33, 0.34],
                       bounds=[(0, 1)] * 3,
                       constraints={'type': 'eq', 'fun': lambda x: sum(x) - 1})
      return result.x
  ```

**3. AAPL Underperformance**
- **Issue**: AAPL has GARCH-dominant (85%) but RMSE ratio 1.470
- **Action**: Check if AAPL regime matches GARCH assumptions
- **Code**:
  ```bash
  # Test stationarity and volatility clustering
  python -c "
  import pandas as pd
  from statsmodels.tsa.stattools import adfuller
  prices = pd.read_parquet('data/training/training_AAPL_*.parquet')['Close']
  adf = adfuller(prices.pct_change().dropna())
  print(f'ADF p-value: {adf[1]:.4f} (stationary if <0.05)')
  "
  ```

### ⚪ LOW Priority

**4. NVDA Trending Regime**
- **Issue**: NVDA SAMoSSA-only but RMSE ratio 1.453 (39% to target)
- **Action**: Phase 8 - Add PatchTST for non-stationary trending markets
- **Benefit**: Transformers excel at capturing long-range dependencies and trends
- **Timeline**: Week 2 of Phase 8 implementation

---

## 🔄 Self-Iteration Status

**Current State**: ✅ HEALTHY
**Tickers at Target**: 1/3 (33%)
**System Readiness**: Ready for Phase 8 (Neural Forecasting)

### Iteration Triggers:
- ✅ GARCH integration validated (85% weight when selected)
- ✅ Multi-ticker validation complete (3 tickers)
- ✅ Performance improvement confirmed (17.6% overall)
- ✅ At least 1 ticker at target (MSFT)
- ⚠️ Optimization opportunities identified (see recommendations)

### Next Auto-Actions:
1. **If RMSE ratio regresses > 10%**: Auto-trigger diagnostics
   ```bash
   python scripts/run_ensemble_diagnostics.py
   ```

2. **If 2+ tickers reach target**: Begin Phase 8.1 (Infrastructure setup)
   ```bash
   pip install neuralforecast skforecast xgboost[gpu]
   ```

3. **If critical issues detected**: Send alert email
   ```python
   # Auto-email on critical issues (to be implemented)
   ```

---

## 📊 Historical Trend (Last 7 Days)

| Date | Avg RMSE Ratio | GARCH Selection % | Tickers at Target |
|------|----------------|-------------------|-------------------|
| 2026-01-21 | 1.386 | 14.3% | 1/3 (33%) |
| 2026-01-20 | 1.682 | 0% | 0/3 (0%) |
| *Baseline* | *1.682* | *0%* | *0/3 (0%)* |

**Trend**: ⬆️ Improving (17.6% gain in 1 day)

---

## 🎯 Targets and Goals

| Metric | Current | Target | Progress |
|--------|---------|--------|----------|
| Average RMSE Ratio | 1.386 | <1.100 | 🟨 50.9% |
| Tickers at Target | 1/3 | 3/3 | 🟥 33.0% |
| GARCH Selection Rate | 14.3% | 30-50% | 🟥 28.6% |
| Ensemble Improvement | +17.6% | +30% | 🟨 58.7% |

---

## 🚀 Roadmap to Full Target Achievement

### Short-Term (Phase 7.4-7.5) - 2 Weeks
- [ ] Implement confidence calibration (quantile-based)
- [ ] Add explicit regime detection (vol + trend features)
- [ ] Optimize AAPL ensemble weights (reduce GARCH to 60-70%)
- [ ] **Target**: 2/3 tickers at target, RMSE ratio <1.3

### Medium-Term (Phase 7.6-7.7) - 4 Weeks
- [ ] Implement dynamic weight optimization (scipy.optimize)
- [ ] Add model switching based on rolling performance
- [ ] Test on 10 tickers (expand validation)
- [ ] **Target**: 7/10 tickers at target, RMSE ratio <1.15

### Long-Term (Phase 8) - 7 Weeks
- [ ] Integrate PatchTST for trending regimes (NVDA)
- [ ] Add XGBoost GPU for directional features
- [ ] Implement real-time retraining triggers
- [ ] **Target**: 9/10 tickers at target, RMSE ratio <1.10

---

## 📚 Quick Reference

### Check Dashboard Status
```bash
python dashboard/live_ensemble_monitor.py
```

### Export Metrics to JSON
```bash
python dashboard/live_ensemble_monitor.py --export
# Output: dashboard/metrics.json
```

### Continuous Monitoring (60s refresh)
```bash
python dashboard/live_ensemble_monitor.py --watch
```

### Run Multi-Ticker Validation
```bash
python scripts/run_etl_pipeline.py \
  --tickers AAPL,MSFT,NVDA \
  --start 2024-07-01 \
  --end 2026-01-18 \
  --execution-mode live

python scripts/analyze_multi_ticker_results.py \
  logs/phase7.3_multi_ticker_validation.log
```

### Run Diagnostics on Specific Ticker
```bash
python scripts/run_ensemble_diagnostics.py \
  --ticker AAPL \
  --analyze-errors \
  --optimize-weights
```

---

## 🔗 Related Documentation

- [Phase 7.3 Complete](../Documentation/PHASE_7.3_COMPLETE.md) - GARCH integration details
- [Multi-Ticker Validation](../Documentation/PHASE_7.3_MULTI_TICKER_VALIDATION.md) - Full validation report
- [Phase 8 Plan](../Documentation/PHASE_8_NEURAL_FORECASTER_PLAN.md) - Neural forecaster roadmap
- [Implementation Checkpoint](../Documentation/implementation_checkpoint.md) - Architecture guidance

---

**Dashboard Version**: 1.0.0
**Last Updated**: 2026-01-21 19:45:00
**Next Review**: After next pipeline run or Phase 7.4 implementation

---

*This dashboard is self-iterative and updates automatically based on pipeline results. Recommendations are generated programmatically based on performance patterns.*
