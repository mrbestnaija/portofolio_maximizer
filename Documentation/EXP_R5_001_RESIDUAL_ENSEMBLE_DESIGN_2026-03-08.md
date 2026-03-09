# EXP-R5-001: Residual Ensemble Around `mssa_rl` — Design Note

**Date**: 2026-03-08 (updated 2026-03-09: RC1–RC4 redesign applied)
**Author**: Agent A
**Scope**: Architecture specification for the residual ensemble research experiment.
**Status**: RC1–RC4 REDESIGN COMPLETE (commit d7ebacd) — see Appendix A below.
M3 decision was REDESIGN_REQUIRED (mean_rmse_ratio=1.689, mean_corr=-0.257, 11 windows).
Experiment reset: new accumulation cycle begins with corrected AR(1) behaviour.

---

## Motivation

The R5 lift gate is definitively negative (CI=[-0.1139, -0.0572], n=162). Root-cause diagnosis
(`R5_LIFT_DIAGNOSIS_2026-03-08.md`) shows that in most windows all three models produce
near-identical directional predictions, so a weighted average (the current ensemble) yields RMSE
close to the arithmetic mean of the individual RMSEs — always worse than the best single model.

The best single model is `mssa_rl` in the majority of audited windows. This experiment tests
whether a *targeted residual correction* anchored on `mssa_rl` can improve R5 lift
**without changing any live strategy, gate, or configuration**.

---

## Architecture

### Anchor model

`mssa_rl` → produces forecast series `y_hat_anchor[t]`.

### Residual target

```
epsilon[t] = y[t] - y_hat_anchor[t]
```

`epsilon[t]` is the per-step signed error of the anchor forecast against realized prices.

### Residual model

Predicts `epsilon_hat[t]` from:
- The same feature inputs available at forecast time.
- Optionally: regime tags (future extension — **not required for EXP-R5-001**).

For EXP-R5-001 the residual model is a lightweight linear or AR(1) model fitted to
`epsilon[t]` on the validation window, with out-of-sample predictions on the test window.

### Final forecast (research-only)

```
y_hat_resid_ens[t] = y_hat_anchor[t] + epsilon_hat[t]
```

This combines the anchor's structural trend with the residual correction's
bias/autocorrelation reduction.

---

## Critical Leakage Prevention (OOS Training Protocol)

The residual model **must not be trained on in-sample anchor predictions**.
Training on in-sample anchor residuals captures noise that perfectly correlates with
the training window — not the OOS error structure.

Four-step OOS protocol:

1. **Train anchor** (`mssa_rl`) on its training window `[t_0, t_train]`.
2. **Generate OOS anchor forecasts** on the validation window `(t_train, t_val]`.
   - These are genuinely out-of-sample: the anchor never saw these prices during fitting.
3. **Compute residuals** `epsilon[t] = y[t] - y_hat_anchor_oos[t]` over the validation window.
4. **Train residual model** on `epsilon[t]` from step 3 only.
   - Test-window residual predictions use the model fitted in step 4.

Invariant: **no residual training on in-sample anchor predictions.**

---

## Promotion Contract (for future decision — not enforced by Agent A)

The following thresholds must be met before EXP-R5-001 can be promoted to any
non-research context:

| Criterion | Threshold |
|-----------|-----------|
| Minimum evidence | `N_effective_audits >= 20` (horizon-consistent with R5) |
| Lift | `RMSE(resid_ens) / RMSE(mssa_rl) <= 0.98` |
| Residual prediction quality | `corr(epsilon[t], epsilon_hat[t]) >= 0.30` over ≥ 20 windows |

**Note on the correlation criterion (2026-03-08 revision):**

The original criterion was `corr(y_hat_anchor, y_hat_resid_ens) <= 0.90`.
This was removed because it is mathematically broken for a bias-correcting residual
model: `y_hat_resid_ens = y_hat_anchor + c` → Pearson corr = 1.0 regardless of
how well the bias is removed.  A correct bias corrector would always fail this gate.

The replacement criterion measures whether the residual model actually predicts the
anchor's out-of-sample errors: `corr(epsilon[t], epsilon_hat[t]) >= 0.30`.
A value ≥ 0.30 over ≥ 20 windows indicates the model captures structure in the
anchor's error series beyond a constant offset.  This is a Phase 3 metric (requires
realized prices) computed by Agent B at audit time.

**Early termination signal (C3):** If `rmse_ratio > 1.02` consistently across ≥ 5
consecutive windows, the residual correction is actively harmful — recommend
experiment redesign rather than continuing to accumulate useless audits.

- Promotion decisions are made by human + Claude Code review only.

---

## Implementation Reference

| Deliverable | File | Status |
|-------------|------|--------|
| A1 — This design note | `Documentation/EXP_R5_001_RESIDUAL_ENSEMBLE_DESIGN_2026-03-08.md` | DONE |
| A2 — Residual hook + ResidualModel | `forcester_ts/residual_ensemble.py` | DONE (Phase 2) |
| A3 — Unit tests (58 tests) | `tests/forcester_ts/test_residual_ensemble.py` | DONE |
| A0 — Config activation wire | `config/forecasting_config.yml` (inside + top-level) + `config/pipeline_config.yml` + `models/time_series_signal_generator.py` + `scripts/run_etl_pipeline.py` | DONE |
| A3.5 — Verification script | `scripts/verify_residual_experiment.py` | DONE |
| M1 — First active audit | `logs/forecast_audits/forecast_audit_20260308_174556.json` | DONE (2026-03-08) |

---

## Activation Instructions

**To activate the experiment** (generates first `residual_status="active"` audit):

```bash
# 1. Flip the flag in config
#    Edit config/forecasting_config.yml:
#      residual_experiment:
#        enabled: true   ← change from false

# 2. Run one forecast pipeline pass
python scripts/run_etl_pipeline.py --tickers AAPL --execution-mode synthetic

# 3. Verify the audit contains an active artifact
python scripts/verify_residual_experiment.py --audit-dir logs/forecast_audits
#    Expected output: [ACTIVE] ... residual_status: active

# 4. Collect experiment metrics across audit history
python scripts/run_quality_pipeline.py \
    --audit-dir logs/forecast_audits \
    --enable-residual-experiment
#    Output: visualizations/performance/residual_experiment_summary.json
```

---

## OOS Auto-Fit Protocol (Phase 2)

The ResidualModel is fitted automatically inside `TimeSeriesForecaster.fit()` using a
leave-last-out split of the training data:

- **OOS window**: `oos_n = max(forecast_horizon, 20)` — 20-point floor for reliable AR(1)
  (19 regression observations → ~12 df for 2-parameter model)
- **Minimum data requirement**: `3 × oos_n` bars (60 bars at default horizon=5)
- **Train split**: `price_series[:-oos_n]` fed to a temporary `MSSARLForecaster`
- **Val split**: `price_series[-oos_n:]` compared against the temporary anchor's forecast
- **Residuals**: `epsilon[t] = val_part[t] - anchor_oos[t]` (positional alignment)
- **Fallback**: any failure leaves `_residual_model = None` → `residual_status="inactive"`

The temporary anchor never replaces `self._mssa` — the main anchor is always fitted
on the full training window.

---

## Scope Restrictions

- Research-only: no impact on live trading, gates, or readiness.
- `config/forecasting_config.yml` `residual_experiment.enabled: false` by default.
- Default model behavior is bit-for-bit unchanged when flag is False.
- Promotion contract is advisory; Agent A does not enforce it at runtime.

---

## Agent B & C Handoff

**Agent B** (B1a done, B1b pending): `scripts/run_quality_pipeline.py --enable-residual-experiment`
is wired and functional.  B1a (Phase 2): verify `residual_status="active"` + non-null forecast lists.
B1b (Phase 3): compute `rmse_anchor`, `rmse_residual_ensemble`, `da_anchor`, `da_residual_ensemble`,
`corr(epsilon[t], epsilon_hat[t])` using realized prices — requires realized price feed at audit time.

**Agent C** (C1-C3): `EXP-R5-001 NOT RUN → IN PROGRESS` transition triggers when any audit file
contains `residual_status="active"` (verified via `verify_residual_experiment.py`).
Early-termination rule: `rmse_ratio > 1.02` across ≥ 5 consecutive windows → recommend redesign.

---

## Appendix A — RC1–RC4 Redesign (2026-03-09, commit d7ebacd)

### M3 root-cause (11-window evaluation)

Diagnosis of anti-signal confirmed by `scripts/_debug_anti_signal.py`:

- 9/11 windows had `corrections_all_same_sign = YES` (constant $7–$13 offsets, same direction every step)
- `mean_rmse_ratio = 1.689` (FAIL), `mean_corr(epsilon, epsilon_hat) = -0.257` (FAIL, need ≥ 0.30)
- Long-run mean `c / (1 - phi)` dominated corrections: e.g., end=2021-10-27 had phi=0.897, c=1.10, long-run mean=+10.63

The AR(1) was learning the DC bias introduced by the anchor mismatch (subset anchor `tmp_mssa` vs full-data `self._mssa`) — not genuine autocorrelation.

### Fixes applied to `forcester_ts/residual_ensemble.py` and `forcester_ts/forecaster.py`

**RC1 — Demeaning** (`fit_on_oos_residuals`):
```python
arr = arr - arr.mean()  # Remove DC offset; fit autocorrelation only
```

**RC3 — Proportional OOS slice** (`_fit_residual_model`):
```python
oos_n = len(cleaned) // 4   # was: max(horizon, 20) = 20-30
if oos_n < 20: return        # data floor: len >= 80 required
```

**RC4 — Phi gate** (`fit_on_oos_residuals`):
```python
_MIN_PHI = 0.15
if abs(self._phi) < _MIN_PHI:
    self.is_fitted = False
    self._skip_reason = f"phi_too_small ({self._phi:.4f} < {_MIN_PHI})"
    return self
```
Caller in `_fit_residual_model` checks `model.is_fitted`; leaves `_residual_model = None` if gate fired.

**RC2 — Anchor mismatch** (deferred): `tmp_mssa` (subset fit) vs `self._mssa` (full fit) still differ.
RC1 demeaning removes the mean bias from this mismatch.  Autocorrelation structure is assumed similar
across subset/full fits.  Full walk-forward residual generation is future work.

### New CANONICAL_FIELDS (observability)

| Field | Type | Meaning |
|-------|------|---------|
| `phi_hat` | float or None | AR(1) lag coefficient estimated by OLS |
| `intercept_hat` | float or None | AR(1) intercept (should be near 0 after RC1) |
| `n_train_residuals` | int or None | Number of demeaned OOS residuals used for fitting |
| `oos_n_used` | int or None | `len(cleaned) // 4` OOS slice size |
| `skip_reason` | str or None | Gate reason string when `is_fitted=False`; None if fitted |

Agent B: use `phi_hat` and `skip_reason` to distinguish "model fitted and applied" from "gate fired".
When `residual_status="active"` AND `skip_reason=None` AND `phi_hat >= 0.15`, correction was applied.

### Behaviour change summary for Agent B

| Scenario | Before RC1-RC4 | After RC1-RC4 |
|---------|---------------|--------------|
| Constant OOS residuals | AR(1) predicts the constant → anti-signal | Demeaned to zero → phi=0 → gate fires → inactive |
| Weak autocorrelation (phi < 0.15) | Correction applied regardless | Gate fires → inactive |
| Strong autocorrelation (phi >= 0.15) | Correction applied (possibly with DC bias) | Demeaned correction applied |
| Data < 80 points | oos_n = max(horizon, 20) = 20 | oos_n = len//4 < 20 → skip |
| Data >= 80 points | oos_n = 20 | oos_n = len//4 >= 20 (scales with data) |

### Re-accumulation cycle

The M3 REDESIGN decision resets the experiment.  New audits accumulating from 2026-03-09 onwards
will reflect the corrected behaviour.  M1/M2/M3 thresholds are unchanged:
- M1: first `residual_status="active"` audit
- M2: 9+ windows with realized residual metrics
- M3: 10+ windows — formal REDESIGN/PROMOTE/CONTINUE decision
