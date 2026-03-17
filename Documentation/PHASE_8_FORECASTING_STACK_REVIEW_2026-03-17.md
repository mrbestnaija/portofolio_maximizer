# Phase 8 — Forecasting Stack Critical Review & Remediation Plan
**Date**: 2026-03-17
**Baseline**: master@15f3a43 (Terminal B merged)
**WR baseline**: 38.1% / 42 round-trips / PF=0.59

---

## Audit Results — What the Review Claimed vs What the Code Actually Has

| # | Claim | Actual Status | Severity | File:Line |
|---|-------|--------------|----------|-----------|
| 1 | SARIMAX exogenous orthogonality missing | **CONFIRMED MISSING** — no PCA/LASSO/VIF | MEDIUM | forecaster.py:234-290 |
| 2 | Only ADF used for stationarity | **NOT CONFIRMED** — both ADF+KPSS run | LOW | forecaster.py:440-466 |
| 3 | No GARCH persistence guard / GJR fallback | **NOT CONFIRMED** — 0.97 threshold + GJR + EWMA chain exists | LOW | garch.py:270-346 |
| 4 | No GARCH convergence CI inflation | **NOT CONFIRMED** — 1.5x inflation on convergence_ok=False | LOW | garch.py:208-231, forecaster.py:1402-1425 |
| 5 | Residual diagnostics only in SARIMAX | **CONFIRMED** — LB/JB only in sarimax.py; GARCH/SAMOSSA/MSSA-RL unchecked | HIGH | sarimax.py:912-963 |
| 6 | Time-index alignment inadequate | **PARTIAL** — basic reindex+fill; no ensemble sync; gaps not flagged | MEDIUM | forecaster.py:386-395 |
| 7 | MSSA-RL RL policy is decorative | **CONFIRMED HIGH** — Q-table only modulates slope drift; no component/rank control | HIGH | mssa_rl.py:313-393 |
| 8 | No SSA window/rank bounds | **NOT CONFIRMED** — paper-based T//3 cap + dimension limits | LOW | samossa.py:381-399 |
| 9 | No model serialization version tracking | **CONFIRMED** — no Python/library fingerprint stored with pickled models | MEDIUM | model_snapshot_store.py |
| 10 | No deterministic backtesting | **PARTIAL** — synthetic mode seeded (seed=123); live GARCH optimizer non-deterministic | MEDIUM | run_etl_pipeline.py:325-357 |

**Bottom line**: 3 of 10 claims confirmed critical. 4 already implemented. 3 partial/real-but-lower-priority.

---

## Priority 1 — MSSA-RL Policy Is Decorative (HIGH)

### Problem
The RL Q-table (`mssa_rl.py:313-393`) is updated during `fit()` from realized returns but the Q-table:
- Does NOT control SSA rank selection (rank = cumulative variance ≥ 90%, deterministic)
- Does NOT control window length
- Does NOT control component weighting
- ONLY modulates the slope/drift direction in the forecast output via `q_direction_weight`

This means we're carrying the computational cost and diversity penalty of an "RL" model that is functionally a deterministic SSA with a drift multiplier. This is a primary contributor to the 38% WR: the ensemble is paying diversity cost (ensemble RMSE worse than best single in 92% of windows) without getting RL's stated benefit (adaptive component selection).

### Fix — Phase 8.1
Make the RL action space meaningful:
1. **Action 0 (mean_revert)**: Use only low-frequency components (top 25% of variance). De-trend aggressively.
2. **Action 1 (hold)**: Use standard component set (top 90% variance). Current behavior.
3. **Action 2 (trend_follow)**: Use all components including high-frequency. Add trend extrapolation.

Concrete code change in `mssa_rl.py`:
```python
# In forecast() — branch on Q-table action to select component set
action = np.argmax(q_row)  # Already computed
if action == 0:  # mean_revert
    n_components_use = max(1, n_components // 4)  # Low-freq only
elif action == 2:  # trend_follow
    n_components_use = n_components  # All components
else:
    n_components_use = max(1, int(n_components * 0.9))  # Standard
```

**Expected impact**: RL starts actually differentiating behavior across regimes. CRISIS regime (Q→action 0) produces smoother mean-reverting forecasts; trending regime (Q→action 2) includes high-frequency components.

---

## Priority 2 — Residual Diagnostics for All Models (HIGH)

### Problem
Ljung-Box and Jarque-Bera are run only in `sarimax.py:912-963`. GARCH, SAMOSSA, MSSA-RL produce residuals but none are checked. A GARCH model with autocorrelated residuals is mis-specified but will pass all gates silently.

### Fix — Phase 8.2
Add a shared `_run_residual_diagnostics(residuals: np.ndarray) -> dict` utility in `forcester_ts/` and call it from all model `fit()` methods:

```python
# forcester_ts/residual_diagnostics.py
def run_residual_diagnostics(residuals: np.ndarray, lags: int = 10) -> dict:
    """Returns dict with lb_pvalue, jb_pvalue, white_noise (bool)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import jarque_bera
    ...
    white_noise = (lb_p > 0.05) and (jb_p > 0.05)
    return {"lb_pvalue": lb_p, "jb_pvalue": jb_p, "white_noise": white_noise}
```

Store result in audit JSON under `"residual_diagnostics"`. Gate SNR check: if `white_noise=False` AND model is GARCH/SAMOSSA, log WARNING and inflate CI by 1.2x (softer than convergence failure's 1.5x).

---

## Priority 3 — ADF/KPSS Conflict Resolution (MEDIUM)

### Problem
Both ADF and KPSS run (`forecaster.py:440-466`) but when they disagree (ADF: stationary, KPSS: non-stationary = "conflicted unit root"), the code records both but takes no action. The conflicted case is common on financial time series with structural breaks.

### Fix — Phase 8.3
Add explicit conflict resolution:
```python
adf_stationary = adf_pvalue < 0.05
kpss_stationary = kpss_pvalue > 0.05  # KPSS: H0 = stationary

if adf_stationary and kpss_stationary:
    stationarity_verdict = "stationary"
    force_difference = False
elif not adf_stationary and not kpss_stationary:
    stationarity_verdict = "non_stationary"
    force_difference = True
else:
    stationarity_verdict = "conflicted"  # Structural break likely
    force_difference = True  # Conservative: difference anyway
```

Record `stationarity_verdict` in audit JSON. This prevents the current silent path where a series appears stationary to ADF (structural break inflates test statistic) but isn't.

---

## Priority 4 — SARIMAX Exogenous VIF Check (MEDIUM)

### Problem
`forecaster.py:234-290` uses `[ret_1, vol_10, mom_5, ema_gap_10, zscore_20]`. `vol_10` (10-day rolling std) and `zscore_20` (price vs 20-day mean / std) are highly correlated — both driven by realized volatility. Multicollinearity inflates standard errors and destabilizes coefficient estimates.

### Fix — Phase 8.4
Add VIF screening before SARIMAX fit:
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Drop features with VIF > 10 (standard threshold)
# Keep at most 3 features to limit degrees of freedom
```

Alternative: replace the 5-feature set with 3 pre-orthogonalized features:
- `ret_1` (price momentum, low VIF)
- `vol_10` (volatility regime)
- `macro_zscore` (PCA-1 of macro context if available, else `mom_5`)

---

## Priority 5 — Model Serialization Version Fingerprint (MEDIUM)

### Problem
`model_snapshot_store.py` pickles model objects with no metadata. A statsmodels version bump can change SARIMAX serialization format, causing silent deserialization failures or subtly different behavior.

### Fix — Phase 8.5
Store metadata sidecar alongside each pickle:
```python
# model_snapshot_store.py — alongside each .pkl write
metadata = {
    "python_version": sys.version,
    "statsmodels_version": statsmodels.__version__,
    "arch_version": arch.__version__,
    "numpy_version": np.__version__,
    "created_at": datetime.utcnow().isoformat(),
    "model_type": model_type,
    "ticker": ticker,
}
Path(pkl_path).with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))
```

On load: warn if stored version != current version. Block load if major version mismatch.

---

## Priority 6 — Ensemble Time-Index Sync (MEDIUM)

### Problem
Ensemble combines forecasts from GARCH (returns-domain), SAMOSSA (price-domain), and MSSA-RL (price-domain). If GARCH's forecast horizon ends at T+N-1 and SAMOSSA's at T+N (off-by-one from different frequency inference), the ensemble silently misaligns.

### Fix — Phase 8.6
Before ensemble weighting, assert all model forecast arrays have the same length and aligned index. If mismatch: log ERROR, use intersection, store `ensemble_index_mismatch=True` in audit JSON. Do not silently fill.

---

## Deferred (Phase 9+)

- **Deterministic backtesting for live data**: Requires pinned dataset snapshots. High infrastructure cost. Defer.
- **GARCH scipy optimizer seed**: `arch` library doesn't expose optimizer seed cleanly. Defer.
- **CI/CD environment fingerprint in PR checks**: Low urgency now that serialization metadata sidecar addresses the core risk.
- **Feature contracts with version bumps**: Useful but over-engineered for current team size.

---

## Integration With Active Work

| Item | Phase | Blocks |
|------|-------|--------|
| MSSA-RL component selection | 8.1 | Directional accuracy (Phase 8 WR improvement) |
| Residual diagnostics all models | 8.2 | GARCH SNR gate correctness (Phase 7.14-C) |
| ADF/KPSS conflict resolution | 8.3 | Regime detection accuracy (Phase 7.14-D) |
| SARIMAX VIF screening | 8.4 | SARIMAX stability (Phase 7.14 SARIMAX re-enable) |
| Serialization fingerprint | 8.5 | Ops resilience — no active phase dependency |
| Ensemble index sync | 8.6 | Ensemble integrity gate |

### Recommended implementation order
1. **8.1** (MSSA-RL) — most leverage on WR, code-only change
2. **8.2** (residual diagnostics) — shares infrastructure with 7.14-C GARCH hardening
3. **8.3** (ADF/KPSS verdict) — 10-line change in forecaster.py
4. Phase E (Platt wire) — from 7.14, parallel to 8.x work
5. **8.4** (SARIMAX VIF) — needed before SARIMAX re-enable
6. **8.5** (serialization metadata) — ops hygiene, any sprint
7. **8.6** (ensemble index sync) — low-risk defensive add
