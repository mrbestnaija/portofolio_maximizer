# EXP-R5-001: M3 Decision Template

**Prepared**: 2026-03-08
**Author**: Agent A
**Purpose**: Pre-written evaluation framework for ≥10 residual windows to prevent subjective interpretation when data arrives.

---

## Decision Threshold Summary

| Criterion | Threshold | Source |
|-----------|-----------|--------|
| Minimum evidence | `N_effective_audits >= 20` | Design Note §Promotion Contract |
| RMSE lift | `mean(rmse_ratio) <= 0.98` | Design Note §Promotion Contract |
| Residual prediction quality | `mean(corr_epsilon) >= 0.30` | Design Note §Promotion Contract (2026-03-08 revision) |
| Early abort trigger | `rmse_ratio > 1.02` for ≥5 consecutive windows | Design Note §Promotion Contract |

---

## M2 Interim Verdict (≥5 windows)

**Filled**: 2026-03-08 (Agent A Phase 3 backfill, 9 realized-metric windows)

| Metric | Value | Pass? |
|--------|-------|-------|
| `n_windows` | 9 | ≥5 required ✓ |
| `mean_rmse_ratio` | 1.7149 | ≤1.02? FAIL (high variance: good=0.68/0.90/0.96/0.99; bad=2.76/3.28/3.41) |
| `win_fraction_rmse` | 4/9 = 44% | % windows where resid_ens beats anchor |
| `mean_corr_epsilon` | -0.1447 | ≥0.10? FAIL (negative — residual corrects in wrong direction on average) |
| `mean_da_delta` | +0.015 (0.559-0.544) | positive ✓ (direction marginally improves) |
| `max_consecutive_abort` | 2 | < 5 threshold, no abort triggered |
| **Decision** | `inconclusive` — accumulate to 10 windows for formal M3 REDESIGN call | |

**Notes**:
- 5 of 9 windows have rmse_ratio > 1.02 (ensemble worse than anchor)
- Only 2 consecutive, so early abort not triggered
- Negative mean corr(ε, ε_hat) = residual model predicts corrections in wrong direction
- Pattern: CV fold windows (len=180/210) consistently have high rmse_ratio; larger datasets (len=601/1014) perform better
- Phase 2 proxy `corr_anchor_residual` was 0.705 (misleadingly positive); Phase 3 truth is -0.145

**Abort condition**: If `rmse_ratio > 1.02` in all 5 windows → abort, do not continue accumulation.

---

## M3 Full Evaluation (≥10 windows)

**COMPLETE — 11 windows** (2026-03-08, Bestman's code Phase 3 backfill):

| Window ID | dataset_end | len | rmse_anchor | rmse_resid_ens | rmse_ratio | da_anchor | da_resid | corr_epsilon |
|-----------|-------------|-----|-------------|----------------|------------|-----------|----------|--------------|
| fp=7be63fa6 | 2020-09-15 | 180 | 1.87 | 1.92 | 1.026 | 0.567 | 0.567 | +0.261 |
| fp=ff283488 | 2020-10-27 | 210 | 1.51 | 4.17 | 2.760 | 0.500 | 0.533 | -0.699 |
| fp=a3b65e5d | 2021-04-20 | 340 | 3.43 | 2.33 | 0.681 | 0.533 | 0.567 | -0.619 |
| fp=ad1bf2f6 | 2021-09-15 | 180 | 4.05 | 5.78 | 1.427 | 0.533 | 0.533 | +0.812 |
| fp=7bef3c48 | 2021-10-27 | 210 | 2.47 | 8.12 | 3.283 | 0.567 | 0.567 | -0.709 |
| fp=8ca36af2 | 2022-04-20 | 601 | 11.59 | 11.44 | 0.987 | 0.500 | 0.567 | +0.497 |
| fp=42b20eb8 | 2022-04-28 | 210 | ~4.3 | ~9.1 | 2.112 | 0.533 | 0.533 | -0.681 |
| fp=d74d9ba3 | 2022-10-20 | 732 | ~14.6 | ~15.1 | 1.035 | 0.567 | 0.567 | -0.845 |
| fp=3e01f88f | 2023-04-17 | 180 | 3.81 | 13.00 | 3.411 | 0.500 | 0.567 | -0.037 |
| fp=4e212b23 | 2023-05-29 | 210 | 7.92 | 7.60 | 0.960 | 0.567 | 0.533 | -0.574 |
| fp=51a38950 | 2023-11-20 | 1014 | 20.22 | 18.20 | 0.900 | 0.567 | 0.600 | -0.234 |

**Aggregate row** (means, 11 windows):

| n_windows | mean_rmse_ratio | win_fraction | mean_corr_epsilon | decision |
|-----------|-----------------|--------------|-------------------|----------|
| 11 | **1.6892** | 36% (4/11) | **-0.2571** | **REDESIGN** |

---

## Decision Rules (apply in order)

1. **ABORT** if ≥5 consecutive windows have `rmse_ratio > 1.02`.
   - Action: halt experiment, file redesign proposal.

2. **REDESIGN** if at ≥10 windows: `mean_rmse_ratio > 1.00` AND `mean_corr_epsilon < 0.10`.
   - Interpretation: residual model adds noise, not signal.
   - Action: review OOS protocol, consider longer oos_n floor.

3. **CONTINUE** if at ≥10 windows: `0.98 < mean_rmse_ratio <= 1.02` OR `mean_corr_epsilon >= 0.10`.
   - Interpretation: early signal; accumulate to 20 windows.

4. **PROMOTE CANDIDATE** if at ≥20 windows all of:
   - `mean_rmse_ratio <= 0.98`
   - `mean_corr_epsilon >= 0.30`
   - Action: human + Bestman's code review; promotion decision only.

---

## FORMAL M3 DECISION (2026-03-08, 11 windows)

**REDESIGN** — Rule 2 triggered.

| Check | Value | Threshold | Result |
|-------|-------|-----------|--------|
| n_windows | 11 | ≥10 | PASS |
| mean_rmse_ratio > 1.00 | 1.6892 | > 1.00 | TRUE |
| mean_corr_epsilon < 0.10 | -0.2571 | < 0.10 | TRUE |
| early_abort (5 consecutive) | 4 | ≥5 | NOT triggered |

**Rule 2 conditions both met: REDESIGN.**

### Redesign Findings

**Finding 1 — Training data floor too low for AR(1) residual fit**
Short CV fold windows (len=180/210) consistently produce high rmse_ratio (1.03-3.41). Larger datasets (len≥340) mostly stay below 1.10 except for volatile periods. The AR(1) residual model needs sufficient OOS history to estimate `phi` accurately. Recommend: `min_oos_n >= 60` before activating residual correction.

**Finding 2 — Negative mean corr(ε, ε_hat) = systematic wrong-direction corrections**
The residual model predicts positive corrections when actual errors are negative, and vice versa, in 8 of 11 windows. This suggests the AR(1) model is fitting the IN-SAMPLE residual autocorrelation, which reverses sign in OOS. Consider: lag-1 residual autocorrelation check — if `|phi| < 0.2` or OOS residual series is near white noise, skip correction entirely.

**Finding 3 — Phase 2 proxy (corr of forecasts) was misleading**
Phase 2 showed `corr_proxy=+0.705`. Phase 3 truth is `corr(ε,ε_hat)=-0.257`. The proxy measures correlation between two linearly-related forecasts (naturally high); it does not measure residual prediction quality. The proxy must not be used as a readiness gate.

### Redesign Actions Required

Before re-running as EXP-R5-002 (or revised R5-001):
1. Add `min_oos_n` guard in `forcester_ts/residual_ensemble.py` — skip correction if fewer than 60 OOS residuals available
2. Add phi validity check — skip correction if `|phi_hat| < 0.15` (near-zero autocorrelation = no persistent signal to exploit)
3. Remove Phase 2 proxy metric from readiness gates; `residual_experiment_metrics_present()` should require Phase 3 fields
4. Run redesigned experiment on ≥3 separate dataset ranges before activating

**Experiment status after M3: REDESIGN_REQUIRED. Do NOT promote or continue accumulating.**

---

## Evidence Independence Note

EXP-R5-001 tracking is **independent** of production gate health:
- `production_audit_gate` FAIL (`THIN_LINKAGE`, `EVIDENCE_HYGIENE_FAIL`) does NOT block experiment accumulation.
- Experiment `ABORT` does NOT affect production gate or live strategy.
- Both can be in FAIL/ABORT simultaneously without conflict.

---

## How to Populate

```bash
# Run quality pipeline with residual tracking
python scripts/run_quality_pipeline.py \
    --audit-dir logs/forecast_audits \
    --enable-residual-experiment

# Extract metrics from output JSON
python -c "
import json, pathlib
d = json.loads(pathlib.Path('visualizations/performance/residual_experiment_summary.json').read_text())
print('n_windows:', d['n_windows_with_residual_metrics'])
print('rmse_ratio_mean:', d['rmse_ratio_mean'])
print('corr_anchor_residual_mean:', d['corr_anchor_residual_mean'])
print('da_anchor_mean:', d['da_anchor_mean'])
print('da_residual_ensemble_mean:', d['da_residual_ensemble_mean'])
"
```

---

## Current State (2026-03-08, M2 complete — Phase 3 backfilled)

| Metric | Value | Notes |
|--------|-------|-------|
| `n_windows_with_realized_residual_metrics` | 9 | Phase 3 complete — realized prices computed |
| `m2_review_ready` | true | ≥5 realized windows achieved |
| `rmse_ratio_mean` | 1.7149 | Ensemble WORSE than anchor on average |
| `corr_anchor_residual_mean` | -0.1447 | Phase 3 truth: corr(ε, ε_hat) — negative, poor prediction quality |
| `da_residual_ensemble_mean` | 0.5593 | Marginal direction improvement (+1.5% vs anchor) |
| `early_abort_signal` | false | Max 2 consecutive above 1.02 (threshold = 5) |
| `n_active_audits` | 23 | 9 unique windows after dedup |
| Phase 2 proxy (old) | 0.705 | corr(y_hat_anchor, y_hat_ensemble) — misleading, now overwritten |
| Experiment status | REDESIGN_REQUIRED | M3 formal decision 2026-03-08: Rule 2 triggered at 11 windows |
| Realized price source | data/checkpoints/pipeline_20260308_184327_data_extraction_*.parquet | 2020-2024 AAPL synthetic series |
