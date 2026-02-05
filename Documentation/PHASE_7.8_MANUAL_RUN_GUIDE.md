# Phase 7.8: Manual Run Guide

**Objective**: Optimize ensemble weights for all remaining regimes (CRISIS, HIGH_VOL_TRENDING, MODERATE_MIXED)

**Status**: Ready for manual execution (extended runtime: 4-6 hours)

---

## Prerequisites

### 1. Activate Virtual Environment

```bash
# Windows
simpleTrader_env\Scripts\activate

# Linux/Mac
source simpleTrader_env/bin/activate
```

### 2. Verify Database

```bash
# Check database exists and has OHLCV data
sqlite3 data/portfolio_maximizer.db "SELECT COUNT(*) FROM ohlcv_data WHERE ticker='AAPL';"

# Expected: Several hundred rows (2023-01-01 to 2026-01-18)
```

---

## Phase 7.8 Command

### Extended Optimization (All Regimes)

```bash
python scripts/optimize_ensemble_weights.py \
    --source rolling_cv \
    --tickers AAPL \
    --db data/portfolio_maximizer.db \
    --start-date 2023-01-01 \
    --end-date 2026-01-18 \
    --horizon 5 \
    --min-train-size 180 \
    --step-size 10 \
    --max-folds 20 \
    --min-samples-per-regime 25 \
    --output data/phase7.8_optimized_weights.json \
    --update-config
```

**Parameters Explained**:
- `--start-date 2023-01-01`: Extended from 2024-07-01 (18 months → 3+ years)
- `--step-size 10`: Reduced from 20 (more folds, better coverage)
- `--max-folds 20`: Increased from 10 (more samples per regime)
- `--min-samples-per-regime 25`: Requires 25+ samples (5 folds × 5-day horizon)

**Expected Runtime**: 4-6 hours (SARIMAX order selection is slow)

---

## What to Expect

### Regimes to be Optimized

Based on Phase 7.5/7.7 observations:

| Regime | Expected Folds | Volatility Range | Trend Range | Likely Optimal Model |
|--------|----------------|------------------|-------------|----------------------|
| **MODERATE_TRENDING** | 8-10 | 23-27% | R²>0.78 | ✅ Already done (90% SAMOSSA) |
| **CRISIS** | 4-6 | 50-53% | 0.00-0.49 | GARCH or SARIMAX (defensive) |
| **HIGH_VOL_TRENDING** | 2-4 | 51%+ | R²>0.60 | SAMOSSA or MSSA-RL (complex patterns) |
| **MODERATE_MIXED** | 3-5 | 26-30% | 0.00-0.30 | Balanced mix |
| **MODERATE_RANGEBOUND** | 2-3 | <27% | <0.30 | GARCH (mean-reversion) |
| **LIQUID_RANGEBOUND** | 1-2 | <15% | <0.30 | GARCH (stable volatility) |

### Expected Output

**Console Output**:
```
================================================================================
PER-REGIME ENSEMBLE WEIGHT OPTIMIZATION (rolling_cv)
================================================================================

## MODERATE_TRENDING (samples=45, folds=9)
  samossa       90.0%
  sarimax        5.0%
  mssa_rl        5.0%
  RMSE: 19.2599 -> 6.7395 (+65.01%)

## CRISIS (samples=30, folds=6)
  garch         75.0%
  sarimax       20.0%
  samossa        5.0%
  RMSE: X.XXXX -> Y.YYYY (+ZZ.ZZ%)

## HIGH_VOL_TRENDING (samples=25, folds=5)
  samossa       60.0%
  mssa_rl       30.0%
  garch         10.0%
  RMSE: X.XXXX -> Y.YYYY (+ZZ.ZZ%)

[... additional regimes if sufficient samples ...]

================================================================================

## YAML Snippet (paste under forecasting.regime_detection)
regime_candidate_weights:
  MODERATE_TRENDING:
    - {sarimax: 0.05, samossa: 0.90, mssa_rl: 0.05}
  CRISIS:
    - {garch: 0.75, sarimax: 0.20, samossa: 0.05}
  HIGH_VOL_TRENDING:
    - {samossa: 0.60, mssa_rl: 0.30, garch: 0.10}
```

---

## Monitoring Progress

### Real-Time Monitoring

Open a second terminal and run:

```bash
# Monitor log output
tail -f logs/phase7.8_weight_optimization.log

# Or use grep to filter
tail -f logs/phase7.8_weight_optimization.log | grep -E "REGIME|Optimizing|RMSE"
```

### Check Fold Progress

```bash
# Count completed folds (each fold generates a regime line)
grep -c "REGIME detected" logs/phase7.8_weight_optimization.log

# Check which regimes are being detected
grep "REGIME detected" logs/phase7.8_weight_optimization.log | \
    cut -d'=' -f2 | cut -d',' -f1 | sort | uniq -c
```

**Example Output**:
```
     10 MODERATE_TRENDING
      6 CRISIS
      5 HIGH_VOL_TRENDING
      3 MODERATE_MIXED
```

---

## Troubleshooting

### Issue: Process Killed or Out of Memory

**Symptom**: Python process killed unexpectedly

**Solution**: Reduce max_folds or increase step_size

```bash
# Less intensive version (fewer folds)
python scripts/optimize_ensemble_weights.py \
    --source rolling_cv \
    --tickers AAPL \
    --start-date 2023-01-01 \
    --step-size 15 \     # Increased from 10
    --max-folds 15 \     # Reduced from 20
    [... other params ...]
```

### Issue: Insufficient Samples for Regime

**Symptom**: Warning message "No regimes had enough samples to optimise"

**Solution**: Lower `--min-samples-per-regime` or extend `--start-date`

```bash
# More lenient sample requirement
--min-samples-per-regime 20  # Reduced from 25

# Or extend history further
--start-date 2022-01-01  # Extended from 2023-01-01
```

### Issue: SARIMAX Taking Too Long

**Symptom**: Each fold takes 5+ minutes on SARIMAX order selection

**Expected**: This is normal! SARIMAX tests multiple (p,d,q) combinations per fold.

**Patience Required**: The optimization will complete, but may take the full 4-6 hours.

**Alternative**: If too slow, consider excluding SARIMAX (though this loses valuable diversity)

```bash
--models samossa mssa_rl  # Exclude SARIMAX (faster but less diverse)
```

---

## After Completion

### 1. Review Results

```bash
# View optimized weights
cat data/phase7.8_optimized_weights.json | jq

# Check RMSE improvements per regime
grep "RMSE:" logs/phase7.8_weight_optimization.log
```

### 1B. Fresh Data Regime Validation (Recommended)

```bash
# Fetch fresh parquet snapshots from yfinance
python scripts/fetch_fresh_data.py \
    --tickers AAPL,MSFT,NVDA \
    --start 2024-07-01 \
    --end 2026-01-18 \
    --output-dir data/raw

# Validate detected regime + weights against the fresh parquet
python scripts/validate_regime_on_fresh_data.py \
    --tickers AAPL,MSFT,NVDA \
    --output-dir data/raw \
    --regimes MODERATE_TRENDING,HIGH_VOL_TRENDING,CRISIS

# Optional: audit duplicated rows in the SQLite DB (safe to run anytime)
python scripts/audit_ohlcv_duplicates.py \
    --tickers AAPL,MSFT,NVDA \
    --export-deduped data/raw
```

### 2. Update Configuration

Copy the YAML snippet from console output to `config/forecasting_config.yml`:

```yaml
# config/forecasting_config.yml
regime_detection:
  enabled: true

  regime_candidate_weights:
    MODERATE_TRENDING:
      # Optimized for 23-27% volatility, strong trend (R²>0.78)
      - {samossa: 0.90, sarimax: 0.05, mssa_rl: 0.05}

    CRISIS:
      # Optimized for 50%+ volatility, extreme conditions
      - {garch: 0.75, sarimax: 0.20, samossa: 0.05}  # EXAMPLE - replace with actual

    HIGH_VOL_TRENDING:
      # Optimized for 50%+ volatility with strong trend
      - {samossa: 0.60, mssa_rl: 0.30, garch: 0.10}  # EXAMPLE - replace with actual

    # Add other regimes as optimized...
```

**Also update `config/pipeline_config.yml`** to maintain parity.

### 3. Validation Run

Test the fully optimized configuration:

```bash
python scripts/run_etl_pipeline.py \
    --tickers AAPL \
    --start 2024-07-01 \
    --end 2026-01-18 \
    --execution-mode auto
```

**Expected**:
- MODERATE_TRENDING folds use optimized weights (90% SAMOSSA)
- CRISIS folds use optimized weights (likely GARCH-dominant)
- HIGH_VOL_TRENDING folds use optimized weights (likely SAMOSSA/MSSA-RL)
- Overall RMSE regression reduced from +42% to ~+15-20%

### 4. Document Results

Create `Documentation/PHASE_7.8_ALL_REGIMES_OPTIMIZED.md`:

```markdown
# Phase 7.8: All Regimes Optimized

## Optimization Results

### MODERATE_TRENDING
- Weights: {samossa: 0.90, sarimax: 0.05, mssa_rl: 0.05}
- RMSE: 19.26 -> 6.74 (+65.0%)
- Samples: 45, Folds: 9

### CRISIS
- Weights: {garch: X.XX, sarimax: X.XX, ...}
- RMSE: X.XX -> Y.YY (+ZZ.Z%)
- Samples: NN, Folds: N

[... continue for each regime ...]

## Validation Results

[... AAPL validation with all optimized weights ...]

## Performance Comparison

| Phase | RMSE Ratio | Notes |
|-------|------------|-------|
| 7.4 | 1.043 | Baseline (regime detection disabled) |
| 7.5 | 1.483 | +42% regression (default weights) |
| 7.7 | TBD | Partial optimization (MODERATE_TRENDING only) |
| 7.8 | **TBD** | **Full optimization (all regimes)** |
```

### 5. Commit Results

```bash
git add config/forecasting_config.yml config/pipeline_config.yml \
        data/phase7.8_optimized_weights.json \
        Documentation/PHASE_7.8_ALL_REGIMES_OPTIMIZED.md

git commit -m "Phase 7.8: All-regime weight optimization complete

OPTIMIZATION RESULTS:
- Extended rolling CV (2023-01-01 to 2026-01-18, 3+ years)
- All regimes optimized: MODERATE_TRENDING, CRISIS, HIGH_VOL_TRENDING, etc.
- [Summarize RMSE improvements per regime]

CONFIGURATION UPDATES:
- config/forecasting_config.yml: Added optimized weights for all regimes
- config/pipeline_config.yml: Synchronized configuration

VALIDATION:
- AAPL test run: RMSE regression reduced from +42% to +XX%
- [Summarize key performance metrics]

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

git push origin master
```

---

## Alternative: Multi-Ticker Optimization

If you want ticker-specific weights (AAPL, MSFT, NVDA):

```bash
# Run separately for each ticker
for ticker in AAPL MSFT NVDA; do
    python scripts/optimize_ensemble_weights.py \
        --source rolling_cv \
        --tickers $ticker \
        --start-date 2023-01-01 \
        --step-size 10 \
        --max-folds 20 \
        --output data/phase7.8_${ticker}_optimized.json \
        --update-config
done
```

**Note**: This produces ticker-specific weights. You'd need to modify the config to support per-ticker regime weights (future enhancement).

---

## Quick Reference

| Task | Command |
|------|---------|
| Run Phase 7.8 optimization | `python scripts/optimize_ensemble_weights.py --source rolling_cv --tickers AAPL --start-date 2023-01-01 --step-size 10 --max-folds 20 --output data/phase7.8_optimized_weights.json --update-config` |
| Monitor progress | `tail -f logs/phase7.8_weight_optimization.log` |
| Count folds | `grep -c "REGIME detected" logs/phase7.8_weight_optimization.log` |
| Check regime distribution | `grep "REGIME detected" logs/*.log \| cut -d'=' -f2 \| sort \| uniq -c` |
| View results | `cat data/phase7.8_optimized_weights.json \| jq` |
| Validate config | `python scripts/run_etl_pipeline.py --tickers AAPL --start 2024-07-01 --execution-mode auto` |

---

## Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Data loading | 5-10 min | SQLite OHLCV extraction |
| CV fold execution | 3-5 hours | SARIMAX order selection is slow |
| Weight optimization | 5-10 min | scipy.optimize (fast) |
| **Total** | **4-6 hours** | Run overnight or during work hours |

---

## Success Criteria

✅ **Optimization Complete**:
- At least 3 regimes optimized (MODERATE_TRENDING, CRISIS, one other)
- Each regime has 25+ samples
- RMSE improvements > 20% for most regimes

✅ **Configuration Updated**:
- All optimized weights in forecasting_config.yml
- pipeline_config.yml synchronized

✅ **Validation Successful**:
- AAPL pipeline runs with optimized weights
- Overall RMSE regression < +25% (vs Phase 7.5's +42%)

---

**Prepared by**: Claude Sonnet 4.5
**Date**: 2026-01-25
**Phase**: 7.8 Preparation

**Status**: ✅ Ready for manual execution
**Estimated Completion**: 4-6 hours from start
