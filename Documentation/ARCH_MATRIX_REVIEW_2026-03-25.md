# Critical Review: Data Structure and Matrix Architecture Document
**Date:** 2026-03-25
**Source doc:** "Deep Dive into Data Structure and Matrix Architecture for Portfolio Maximizer (v4.4) Models"
**Reviewed against:** actual codebase (`forcester_ts/`, Phase 10, commit 91d06bd)

---

## Summary Verdict

The document is a **useful conceptual primer** but contains **five significant inaccuracies** and **omits the three most architecturally important components** of the current system. It should not be used as implementation reference without the corrections below.

Rating by section:

| Section | Conceptual accuracy | Code accuracy | Notes |
|---------|-------------------|--------------|-------|
| SARIMAX | Good | Partial | Exogenous variable claim overstated |
| GARCH | Partial | Poor | Covariance matrix framing is wrong; implementation far richer |
| SAMoSSA / MSSA | Partial | Partial | Conflates two distinct models; misses diagonal averaging |
| RL Ensemble Weighting | Misleading | Incorrect | Conflates two unrelated mechanisms |

---

## 1. SARIMAX — What Is Accurate and What Is Not

### Correct
- The lag matrix formulation `X = [Y_{t-1}, Y_{t-2}, …, Y_{t-p}, X_exog]` correctly represents what `statsmodels.SARIMAX` builds internally.
- The seasonal lag matrix `X_seasonal = [Y_{t-s}, Y_{t-2s}, …]` with period `s` is accurate.
- Stationarity via differencing is required and handled (ADF test + `forced_d` wired in Phase 8.3).

### Inaccuracies / Gaps

**Exogenous variables are not macroeconomic factors.**
The document claims Column 2 contains "macroeconomic factors, weather, etc." In practice the `SARIMAXForecaster.fit()` accepts only `price_series` and optionally `returns_series` as exogenous input. Macro data is not wired to the forecaster at any point in the current pipeline.

```python
# sarimax.py — actual signature
def fit(self, series: pd.Series, returns: Optional[pd.Series] = None,
        forced_d: Optional[int] = None, ticker: str = "") -> "SARIMAXForecaster":
```

**Order selection is not manual grid search.**
The document describes a generic AR/MA/seasonal lag matrix but doesn't mention that `SARIMAXForecaster` performs AIC/BIC-guided automatic order search (modes: `"full"` / `"fast"`) bounded by `max_p`, `max_d`, `max_q`, `max_P`, `max_D`, `max_Q`. Results are cached per `(ticker, regime)` via `order_learner.py` for warm-start.

**"Dynamic Memory Buffer" improvement is already implemented.**
The suggestion of a sliding window buffer is addressed by `forcester_ts/parameter_cache.py` which persists fitted SARIMAX orders on disk, avoiding full re-search on each call.

**Missing: convergence and log-transform handling.**
`SARIMAXForecaster` wraps every fit attempt in `ConvergenceWarning` capture, retries with looser tolerances, and supports `log_transform=True` for price-level (non-return) inputs. None of this appears in the document.

---

## 2. GARCH — Significant Errors

### Correct
- The GARCH(1,1) variance equation `σ²_t = ω + α·Y²_{t-1} + β·σ²_{t-1}` is exact.
- The concept of squared-residuals as the input data structure is correct.

### Critical Error: The Diagonal Covariance Matrix is Wrong

The document presents:

```
Σ = diag(σ²_1, σ²_2, …, σ²_n)
```

This framing implies a static, cross-sectional covariance matrix across time periods — which contradicts the core purpose of GARCH. GARCH models a **scalar conditional variance path** `σ²_t` where each value depends causally on its predecessor. The output is a 1D time series of volatility estimates, not a diagonal matrix. The diagonal matrix framing belongs to multi-asset covariance estimation (DCC-GARCH), which is not implemented here.

**Correct representation:**

```
σ²_t  ∈ ℝ  for t = 1…T          (scalar, sequential)
σ²_t  = f(Y_{t-1}, σ²_{t-1})    (causal chain, not independent)
```

### Implementation Is Substantially Richer Than Described

The document describes a basic GARCH model. The actual `GARCHForecaster` (Phase 7.10b, 7.14-C) is:

| Feature | Document | Implementation |
|---------|----------|----------------|
| Mean model | Not mentioned | AR(1) — explicit directional signal |
| Distribution | Implied normal | Skewed-t (`dist='skewt'`) for fat tails |
| ADF pre-test | Not mentioned | Unit-root guard; auto-differences if needed |
| Convergence | Not mentioned | Captures `RuntimeWarning`, sets `_convergence_ok=False` |
| Fallback chain | Not mentioned | Primary GARCH → GJR-GARCH (asymmetric) → EWMA |
| CI inflation | Not mentioned | 1.5× half-width inflation when convergence failed |
| Order search | p=1,q=1 implied | AIC-guided search up to `max_p=3, max_q=3` |

**Fallback chain (actual, in order):**
```
arch GARCH(p,q) + skewt
  → if persistence ≥ 0.97 OR convergence failed:
      GJR-GARCH (asymmetric leverage term o=1)
        → if GJR also degenerate:
            EWMA (exponentially weighted moving average)
```

### Sparse Matrix Suggestion Is Inapplicable

"Use CSR format for parallel processing" — GARCH is an inherently sequential recursion. Each `σ²_t` requires `σ²_{t-1}`. CSR sparse matrices give no advantage here; the relevant parallelism is in the AIC-search loop over `(p, q, dist)` candidates, which is already bounded by the small grid size.

---

## 3. SAMoSSA / MSSA-RL — Two Distinct Models Conflated

### Critical Structural Error: These Are Two Separate Models

The document treats "MSSA/SAMoSSA" as a single model. The codebase contains **two separate forecasters** with different roles:

| File | Class | Basis | Purpose |
|------|-------|-------|---------|
| `samossa.py` | `SAMOSSAForecaster` | SSA + ARIMA hybrid | Price-level trend reconstruction |
| `mssa_rl.py` | `MSSARLForecaster` | SSA + Q-learning CUSUM | Change-point detection + regime-aware reconstruction |

Both use trajectory matrices and SVD, but the downstream logic is entirely different.

### SAMoSSA: What Is Correct and Missing

**Correct:** Trajectory matrix construction, SVD decomposition, reconstruction from top components.

**Missing: Two matrix types.** SAMoSSA supports both Page matrix and Hankel matrix construction, selected by `matrix_type`:

```python
# Page matrix (non-overlapping segments, shape L × K where K = T//L)
def _build_page_matrix(self, series): ...

# Hankel matrix (sliding window, shape L × (T-L+1))
def _build_hankel_matrix(self, series): ...
```

**Missing: Diagonal averaging.** After low-rank reconstruction from SVD components, the matrix must be converted back to a 1D time series. The document omits this step entirely. The actual method (`_diagonal_averaging`) averages anti-diagonal elements to recover the scalar sequence:

```
recon[t] = mean of all matrix[i,j] where i+j = t
```

This is a standard SSA step and failure to mention it leaves reconstruction mathematically incomplete.

**Missing: Residual ARIMA layer.** After SSA reconstruction, the SSA residuals `Y_t - Ŷ_SSA` are modelled with an `AR(1)` (configurable via `arima_order`). The final forecast is `Ŷ_SSA + AR_residual_forecast`. This is the "ARIMAX" in SAMoSSA.

**"Regularization of Singular Values" improvement is already implemented.**
Component count is controlled by `variance_target=0.90` (auto mode) — only components explaining up to 90% of trajectory variance are retained. Manual override via `n_components`.

### MSSA-RL: Key Differences from Document

**The document's description of SVD architecture is correct for MSSA-RL.** However, the RL component is for **internal reconstruction rank selection** (choosing between 25% / 90% / 100% variance ranks for three Q-actions: mean_revert / hold / trend_follow), NOT for ensemble weighting.

```python
# mssa_rl.py:204 — per-action reconstructions from the SAME SVD
self._recon_matrix_by_action = {
    0: _recon_for_rank(rank_25),   # mean_revert: low-frequency only
    1: _recon_for_rank(rank_90),   # hold: standard
    2: _recon_for_rank(rank_all),  # trend_follow: full fidelity
}
```

The Q-table `Q[(state, action)]` selects which reconstruction to use for forecasting based on CUSUM change-point detection signals. This is entirely internal to `MSSARLForecaster`.

---

## 4. RL Ensemble Weighting — Fundamentally Misrepresented

### What the Document Claims

The document describes a Q-learning agent with:
- State matrix of market features
- Action matrix of model weights
- Q-function `Q(S_t, A_t) = R_t + γ·max_{A'} Q(S_{t+1}, A')`
- Online weight updates through market interaction

### What Actually Exists

The `EnsembleCoordinator` is **not a reinforcement learning agent**. It is a **static candidate selection system**:

1. A predefined list of 15 candidate weight vectors is defined at construction (e.g., `{"sarimax": 0.50, "garch": 0.30, "mssa_rl": 0.20}`)
2. Each candidate is scored by:
   ```
   score = Σ_m (weight_m × confidence_m)
   ```
   where `confidence_m` is the current model's CV-estimated confidence
3. RMSE-rank hybrid scoring (Phase 10) normalizes each model's RMSE rank to `[0.05, 0.95]` to prevent SAMoSSA EVR dominance
4. DA (directional accuracy) penalties cap weight for models below `da_floor=0.10`
5. The highest-scoring candidate is selected — no state, no Q-table, no gradient updates

```python
# ensemble.py — actual selection (no RL)
for candidate in candidate_list:
    normalized = self._normalize(candidate)
    scaled = {m: w * model_confidence.get(m, 0.5) for m, w in normalized.items()}
    score = sum(scaled.values())
    scored_candidates.append((normalized, score))
selected = max(scored_candidates, key=lambda x: x[1])
```

**The Q-learning described in the document belongs to `MSSARLForecaster`** (reconstruction rank selection), not to `EnsembleCoordinator` (weight allocation). The document conflates these two unrelated mechanisms.

**Phase 7.17 adaptive candidates** are the closest thing to "learning" in the ensemble: `ensemble_health_audit.py` can write data-driven weight vectors based on historical performance which are prepended to the static list. But this is offline computation, not online RL.

---

## 5. Three Major Components Absent from Document

The document describes the model internals but misses the architectural components that connect them and determine actual trading performance.

### 5.1 Regime Detector (`regime_detector.py`)

A separate `RegimeDetector` runs before the ensemble to classify market conditions:

| Regime | Characteristics | Preferred models |
|--------|----------------|-----------------|
| LIQUID_RANGEBOUND | Low vol (<15%), weak trend | GARCH, SARIMAX |
| MODERATE_TRENDING | Medium vol, medium trend | Mixed ensemble |
| HIGH_VOL_TRENDING | High vol (>30%), strong trend | SAMoSSA, MSSA-RL, GARCH |
| CRISIS | Extreme vol, structural breaks | GARCH, SARIMAX (defensive) |

Feature extraction uses: realized volatility (annualized), Hurst exponent (R/S analysis), trend strength (OLS slope / residual std). The `RegimeConfig` is persisted to the `time_series_forecasts` DB table (`detected_regime`, `regime_confidence` columns — Phase 7.14-D).

### 5.2 RMSE-Rank Hybrid Confidence (Phase 10, `ensemble.py`)

The document describes confidence as a simple scalar per model. In Phase 10, confidence is a hybrid score:

```
conf_hybrid = 0.7 × conf_existing + 0.3 × conf_rmse_rank

where conf_rmse_rank = 1.0 - (rmse - min_rmse) / (max_rmse - min_rmse + ε)
clipped to [0.05, 0.95]
```

This prevents SAMoSSA's Explained Variance Ratio (~1.0 by SSA construction) from dominating when GARCH has lower RMSE on the actual forecast horizon.

### 5.3 Directional Classifier (`directional_classifier.py`, Phase 9)

A separate logistic regression layer (`CalibratedClassifierCV`) is trained on labeled historical data and provides `p_up ∈ [0,1]` — the probability that the next bar is positive. This gates signal generation independently of the ensemble forecast level.

Feature vector (20 features including `ensemble_pred_return`, CI half-width, SNR, regime flags, volatility metrics). Trained via TimeSeriesSplit walk-forward CV to prevent data leakage.

---

## 6. Actionable Corrections for Future Documentation

1. **SARIMAX:** Remove reference to macroeconomic exogenous variables. Replace "Dynamic Memory Buffer" improvement with description of existing `parameter_cache.py` warm-start. Add note on ADF-forced differencing.

2. **GARCH:** Remove the diagonal covariance matrix — replace with the scalar conditional variance path. Expand to cover the AR(1) mean model, skewt distribution, GJR fallback, convergence detection. Remove CSR/parallelism suggestion (inapplicable to sequential recursion).

3. **SAMoSSA:** Separate from MSSA-RL into two distinct sections. Add diagonal averaging step to reconstruction. Document the Page/Hankel matrix choice. Add the residual ARIMA layer.

4. **RL Ensemble:** Rename section to "Confidence-Weighted Candidate Selection." Replace Q-function formulation with the actual scoring equation. Move Q-learning description to the MSSA-RL section where it actually belongs.

5. **Add missing sections:**
   - Regime Detection and its influence on candidate ordering
   - RMSE-rank hybrid confidence scoring
   - Directional Classifier as a signal gate
   - Walk-forward CV as the RMSE estimation method (not train/test split)

---

## 7. What Is Genuinely Useful in the Document

The following are correct and worth retaining as conceptual orientation for new contributors:

- The general framing of time-series data as lagged design matrices is accurate for SARIMAX and provides intuition.
- The GARCH(1,1) variance recursion formula is exact and pedagogically clear.
- The SVD decomposition `X = UΣV^T` for SSA trajectory matrices is correct and the three-matrix interpretation (U = trends, Σ = strength, V = oscillations) is a useful mental model.
- The observation that stationarity checking is a prerequisite for SARIMAX is operationally correct and reflected in the ADF pre-test implementation.
- The "Exploration vs Exploitation" framing, though mislabeled as RL ensemble weighting, accurately describes the `prefer_diversified_candidate` / `diversity_tolerance` tradeoff in the actual coordinator.

---

*Reviewed by: engineering session 2026-03-25. Cross-checked against `forcester_ts/sarimax.py`, `garch.py`, `samossa.py`, `mssa_rl.py`, `ensemble.py`, `regime_detector.py`. Phase 10 (commit 91d06bd) is the reference version.*
