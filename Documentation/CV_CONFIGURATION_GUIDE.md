# Cross-Validation Configuration Guide

## Overview

The ETL pipeline now supports **configuration-driven cross-validation** where all CV parameters can be set in `config/pipeline_config.yml` and optionally overridden via CLI.

---

## Configuration File Structure

### Location
```
config/pipeline_config.yml
```

### Cross-Validation Section
```yaml
pipeline:
  data_split:
    # Default strategy (simple or cv)
    default_strategy: "simple"
    
    # Simple chronological split configuration
    simple_split:
      enabled: true
      train_ratio: 0.70
      validation_ratio: 0.15
      test_ratio: 0.15
      chronological: true
      shuffle: false
    
    # k-fold cross-validation configuration
    cross_validation:
      enabled: true
      n_splits: 5              # Number of folds
      test_size: 0.15          # Test set proportion
      gap: 0                   # Gap between train/val
      expanding_window: true   # Use expanding window
      window_strategy: "expanding"
      expected_coverage: 0.83  # Expected validation coverage
```

---

## Usage Patterns

### 1. Use Config Defaults (Recommended)

Set your preferred defaults in `config/pipeline_config.yml`, then run:

```bash
# Uses all config defaults (k=5, test_size=0.15, gap=0)
python scripts/run_etl_pipeline.py \
  --tickers AAPL \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --use-cv
```

**Output:**
```
✓ Using k-fold cross-validation (k=5)
  - Test size: 15%
  - Gap between train/val: 0 periods
  - Window strategy: expanding
  - Expected validation coverage: 83%
```

### 2. Override Config with CLI

Override specific parameters while keeping config defaults:

```bash
# Override k and test_size, keep gap from config
python scripts/run_etl_pipeline.py \
  --tickers AAPL \
  --start 2020-01-01 \
  --end 2024-01-01 \
  --use-cv \
  --n-splits 7 \
  --test-size 0.2
```

---

## CLI Parameters (All Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--use-cv` | flag | from config | Enable k-fold CV |
| `--n-splits` | int | 5 | Number of folds |
| `--test-size` | float | 0.15 | Test set proportion (0.0-1.0) |
| `--gap` | int | 0 | Gap between train/val periods |

---

## Configuration Priority

**Priority order** (highest to lowest):

1. **CLI arguments** (if provided)
2. **Config file values** (`pipeline_config.yml`)
3. **Hard-coded defaults** (fallback)

---

## Production Recommendations

### For Production Deployments

Set in `config/pipeline_config.yml`:

```yaml
data_split:
  default_strategy: "cv"  # Use CV by default
  
  cross_validation:
    n_splits: 7            # More folds = better validation
    test_size: 0.15        # Standard 15% test set
    gap: 1                 # Add 1-day gap to avoid look-ahead
    expanding_window: true
```

Then run without any flags:
```bash
python scripts/run_etl_pipeline.py --tickers AAPL,MSFT --include-frontier-tickers
```

---

## Benefits of Configuration-Driven Approach

✅ **No hard-coded values** - All parameters externalized
✅ **Team consistency** - Same config across team members
✅ **Easy tuning** - Edit config file, no code changes
✅ **CLI flexibility** - Override when needed
✅ **Version controlled** - Config changes tracked in git
✅ **Environment-specific** - Different configs for dev/prod

---

**Last Updated:** 2025-10-05
**Status:** Production Ready ✅
