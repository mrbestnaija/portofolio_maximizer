# R5 Ensemble Lift Diagnosis
Date: 2026-03-08
Owner: Agent A
Scope: Structural root-cause analysis of negative ensemble lift (CI=[-0.1139,-0.0572])

## Evidence Gathered

### Audit File Survey

Total audit files: ~400+, spanning 2026-02-24 through 2026-03-07.

Two format generations:

| Generation | Dates | evaluation_metrics | Source |
|---|---|---|---|
| Old (backtest) | 2026-02-24/25 | PRESENT | `evaluate()` called with realized prices |
| New (live/semi-live) | 2026-03-06/07 | ABSENT | `forecast()` only, no holdout available |

`ensemble_health_audit.py::extract_window_metrics()` skips any file without `artifacts.evaluation_metrics`.
`forecaster.py::_audit_history_stats()` likewise silently skips new-format files.

**Implication**: lift statistics and the internal preselection gate are effectively frozen on the 2026-02-25 backtest set. The 90%+ of files from 2026-03-06 contribute nothing to the gate.

### Per-Window RMSE Pattern (sample from 2026-02-25 backtest)

Window: 2025-02-25 to 2025-12-15, horizon 30, frequency B

| Model | RMSE | DA |
|---|---|---|
| garch | 3.5549 | 0.5517 |
| samossa | 4.0517 | 0.5517 |
| mssa_rl | 3.0782 | **best** |
| ensemble | 3.5060 | 0.5517 |

rmse_ratio = 3.506 / 3.078 = **1.139**
Ensemble weights used: garch 0.338, samossa 0.427, mssa_rl 0.235

**Critical observation**: directional_accuracy = 0.5517 for ALL models AND ensemble.
This means the three models produce nearly identical directional predictions.
When forecast correlation ~= 1, averaging adds noise without reducing error.

### Adaptive Weights (computed 2026-03-01)

From `config/forecasting_config.yml`:
```yaml
adaptive_candidate_weights:
  computed_at: '2026-03-01'
  weights:
  - garch: 0.348818
    samossa: 0.314581
    mssa_rl: 0.336602   # near-equal
  - garch: 0.7
    mssa_rl: 0.3
  - garch: 1.0
```

Near-equal weights (0.35/0.31/0.34) mean no model consistently dominates
across the 20-window recent set. Blending with near-equal weights always
produces RMSE near the mean of models, which is above the minimum.

### Preselection Gate Status (from 2026-03-06 GOOG audit)

```json
"ensemble_decision_reason": "preselection gate: recent RMSE ratio 1.133 > 1.000"
```

Gate correctly enforces DISABLE_DEFAULT. Live signals use best single model.
The gate threshold is `strict_preselection_max_rmse_ratio: 1.0`.

### Confidence Scoring vs RMSE Reality

From the 2026-03-06 GOOG window:
- garch confidence: 0.625
- samossa confidence: 0.65 (HIGHEST — but samossa RMSE is WORST)
- mssa_rl confidence: 0.40 (LOWEST — but mssa_rl RMSE is BEST in many windows)

`select_weights()` scores candidates by `sum(weight * confidence)`.
This selects candidates that put weight on high-confidence models —
but Platt calibration is trained on directional accuracy (binary),
not RMSE (continuous). A model can have high DA-confidence AND high RMSE.

## Root Causes (in order of impact)

### RC-1 (PRIMARY): Forecast correlation is not measured or penalized

The ensemble blender has no awareness of how correlated model forecast vectors are.
When all three models predict the same direction at similar magnitudes, the
ensemble output is a weighted average of near-identical predictions. Blending
does not reduce error in this case — it only averages RMSE levels.

Identical DA across all models (0.5517 in the sample window) confirms near-perfect
correlation in directional predictions for this window.

No code in `forcester_ts/ensemble.py` or `forcester_ts/forecaster.py` computes
or acts on inter-model forecast correlation.

### RC-2: Confidence scores are a poor weight proxy for RMSE

`select_weights()` selects the candidate with highest `sum(weight * confidence)`.
In practice, samossa (worst RMSE) gets highest confidence (0.65), so candidates
concentrating weight on samossa win the selection. This is backwards for RMSE.

Platt confidence was trained as a directional accuracy calibrator.
It is not a predictor of low RMSE.

### RC-3: Adaptive weights are near-uniform, providing no concentration benefit

When no model dominates across 20 recent windows, the adaptive weight formula
`exp(-lambda * mean_rmse / rmse_median)` with close RMSEs produces near-equal
weights. Equal weighting always yields RMSE between worst and best single model.

### RC-4: New audit files lack evaluation_metrics

Post-Phase 7.39 live runs write audit files without calling `evaluate()` (no
realized holdout available at forecast time). Both `ensemble_health_audit.py`
and `_audit_history_stats()` in the forecaster skip these files.

Gate decisions effectively run on stale 2026-02-25 backtest data only.

## Prescription

### P1 (Highest impact): Forecast diversity gate in ensemble.py

Before blending, compute pairwise Pearson correlation between model forecast
vectors. If max pairwise correlation > threshold (e.g. 0.90), skip ensemble
and fall through to best-confidence single model.

```python
# In select_weights() or in forecaster's blend step
def _max_forecast_correlation(forecasts: dict[str, np.ndarray]) -> float:
    series = [v for v in forecasts.values() if len(v) > 1]
    if len(series) < 2:
        return 0.0
    cors = []
    for i in range(len(series)):
        for j in range(i+1, len(series)):
            c = np.corrcoef(series[i], series[j])[0, 1]
            if math.isfinite(c):
                cors.append(abs(c))
    return max(cors) if cors else 0.0
```

If `_max_forecast_correlation(model_forecasts) > diversity_min_correlation_benefit`
(config, default 0.90), emit `ensemble_status="SKIP_HIGH_CORRELATION"` and use
best-confidence single model. The audit should log the correlation as a diagnostic.

This would block the ensemble in the majority of current windows (where all models
predict the same direction) and only allow it when genuine diversity exists.

### P2: Score candidates by RMSE history, not Platt confidence

In `select_weights()`, if audit history has recent per-model RMSE data available,
use `score = sum(weight * (1 / mean_rmse_i))` for the history-based candidate
and prefer it over confidence-scored candidates. Confidence-scored candidates
remain as fallback when history is insufficient.

### P3: Write evaluation_metrics to all audit formats

When realized prices are available (holdout mode, backtesting), ensure
`evaluate()` is called and `evaluation_metrics` is written. This feeds fresher
data to both `ensemble_health_audit.py` and `_audit_history_stats()`.

For live-mode runs (no realized holdout), add a note field:
```json
"evaluation_metrics_unavailable": "no realized holdout at forecast time"
```
so that `extract_window_metrics()` can distinguish "not evaluated" from
"evaluated and missing" rather than silently skipping.

### P4: Regime-conditional ensemble policy

From regime detection in audit files: HIGH_VOL_TRENDING is the dominant regime
in 2026-03-06 runs. In trending regimes, models all fit the same trend and
produce correlated forecasts. Regime-specific policy:
- HIGH_VOL_TRENDING / CRISIS: always use best single model (skip ensemble)
- LIQUID_RANGEBOUND / MODERATE_RANGEBOUND: allow ensemble (models disagree on direction)

This is lower confidence than P1 because we don't have per-regime lift statistics.

## Coordination

| Fix | Scope | Agent | Blocks |
|---|---|---|---|
| P1: Forecast diversity gate | `forcester_ts/ensemble.py` or `forcester_ts/forecaster.py` | A | Nothing |
| P2: RMSE-based candidate scoring | `forcester_ts/ensemble.py::select_weights()` | A | P3 (needs history) |
| P3: evaluation_metrics in all formats | `forcester_ts/forecaster.py` | A | Feeds P2 and gate |
| P4: Regime-conditional policy | `config/forecaster_monitoring.yml` + forecaster | A after evidence | Monday evidence window |

### Temporal Limitations

- P1 and P2 can be implemented now (no new trading data required)
- P4 requires per-regime lift statistics — only available after Monday evidence window
  if `fresh_linkage_included > 1` and `fresh_production_valid_matched >= 1`

### Spatial Limitations

- `evaluation_metrics` gap affects this workstation only — cloud/CI runs may
  or may not call `evaluate()` depending on pipeline configuration

### Technical Dependencies

- P1 requires model forecast vectors to be accessible at the blend step
  (they are: `self._latest_results.get("garch_forecast")` etc.)
- P2 requires `_audit_history_stats()` to return per-model mean RMSE
  (currently returns only ensemble ratios, not per-model values)
- P3 is a write-path change to the forecaster's audit serialization

## Current Gate Status

- R5 FAIL: CI=[-0.1139,-0.0572], definitively negative, n=162
- Live signals: correctly using best single model (ensemble DISABLE_DEFAULT)
- No PnL impact from R5 right now — the preselection gate is already blocking
  the bad ensemble from being used
- R5 lift improvement would unlock the ensemble as a valid signal source,
  potentially improving signal quality in diverse-forecast windows
