"""
forcester_ts/residual_ensemble.py

EXP-R5-001: Residual Ensemble Around mssa_rl (Research Experiment â€” Agent A).

PHASE STRUCTURE
---------------
Phase 1 (this file): Architecture plumbing only.
  - Canonical artifact schema defined.
  - `build_residual_ensemble()` combines anchor + optional residual.
  - `ResidualModel` stub exists â€” not yet trained (Phase 2).
  - Metric fields (rmse_*, da_*, corr_anchor_residual) are always None here;
    they are computed at audit time by Agent B (Phase 3).
  - OOS source guard on `compute_oos_residuals()`.

Phase 2: ResidualModel fitted on OOS anchor residuals â€” real epsilon_hat.
Phase 3: Metric emission (rmse_ratio, da_*, corr_anchor_residual) in audit layer.

SCOPE RESTRICTIONS (DO NOT REMOVE)
------------------------------------
- No changes to gate semantics or config/*.yml.
- All outputs are additional metadata for audits only.
- Default forecast behavior is UNCHANGED when experiment is inactive.
- Promotion contract is advisory; NOT enforced at runtime.

See: Documentation/EXP_R5_001_RESIDUAL_ENSEMBLE_DESIGN_2026-03-08.md
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = "EXP-R5-001"
ANCHOR_MODEL_ID = "mssa_rl"
CURRENT_PHASE = 2  # Updated: ResidualModel implemented.

# Canonical field set â€” single source of truth for the artifact schema.
# Agent B imports this via quality_pipeline_common; the instrumentation layer
# uses inactive_artifact().keys() as a cross-check.  Both must stay in sync.
CANONICAL_FIELDS: frozenset = frozenset({
    "experiment_id", "anchor_model_id", "phase",
    "residual_status", "residual_active", "reason",
    "residual_signal_valid", "correction_applied", "reason_code",
    "y_hat_anchor", "y_hat_residual_ensemble",
    "rmse_anchor", "rmse_residual_ensemble", "rmse_ratio",
    "da_anchor", "da_residual_ensemble", "corr_anchor_residual",
    "residual_mean", "residual_std", "n_corrected",
    "promotion_contract",
    # RC1/RC4 observability â€” always emitted; None when model not fitted.
    "phi_hat", "intercept_hat", "n_train_residuals", "oos_n_used", "skip_reason",
})

# Promotion contract thresholds â€” advisory, never enforced here.
#
# corr_threshold REMOVED (2026-03-08):
# The original criterion was corr(y_hat_anchor, y_hat_residual_ensemble) â‰¤ 0.90.
# For a bias-correcting residual model (the primary use case), the ensemble is
# anchor + constant_offset â†’ Pearson corr = 1.0 regardless of how well the bias
# is removed.  The criterion would always reject good corrections.
#
# Replacement: corr(epsilon[t], epsilon_hat[t]) â€” the correlation between actual
# OOS anchor errors and the residual model's predictions of those errors.  A value
# of â‰¥ 0.30 over â‰¥ 20 windows is the meaningful diversity signal.  This is a
# Phase 3 Agent B metric (requires realized prices); it is NOT computable here.
#
# Early-termination signal (C3): rmse_ratio > 1.02 across â‰¥ 5 consecutive windows
# indicates the residual correction is actively harmful â€” stop accumulating.
_PROMOTION_CONTRACT: Dict[str, Any] = {
    "rmse_ratio_threshold": 0.98,
    "residual_prediction_corr_threshold": 0.30,  # corr(epsilon[t], epsilon_hat[t]) â‰¥ 0.30
    "residual_prediction_corr_note": (
        "correlation between realized anchor errors and residual forecasts; "
        "requires realized prices â€” computed by Agent B at audit time"
    ),
    "early_stop_rmse_ratio": 1.02,  # stop if ratio > 1.02 across >= 5 windows
    "min_effective_audits": 20,
    "note": "advisory only â€” not enforced at runtime",
}


def inactive_artifact(
    *,
    reason: str = "experiment disabled",
    reason_code: str = "INACTIVE",
) -> Dict[str, Any]:
    """Return the canonical inactive artifact schema.

    All metric fields are explicitly None â€” unknown is not 0.0 and not PASS.
    Called by the forecaster when the experiment path is not active.
    """
    return {
        "experiment_id": EXPERIMENT_ID,
        "anchor_model_id": ANCHOR_MODEL_ID,
        "phase": CURRENT_PHASE,
        "residual_status": "inactive",
        "residual_active": False,
        "reason": reason,
        "residual_signal_valid": False,
        "correction_applied": False,
        "reason_code": reason_code,
        # Forecast series â€” None means "not computed", not "zero correction".
        "y_hat_anchor": None,
        "y_hat_residual_ensemble": None,
        # Phase 3 audit metrics â€” always None here; populated by Agent B.
        "rmse_anchor": None,
        "rmse_residual_ensemble": None,
        "rmse_ratio": None,
        "da_anchor": None,
        "da_residual_ensemble": None,
        "corr_anchor_residual": None,
        # Phase 1 diagnostics (set when active, None when inactive).
        "residual_mean": None,
        "residual_std": None,
        "n_corrected": 0,
        "promotion_contract": _PROMOTION_CONTRACT,
        # RC1/RC4 observability â€” None means model not fitted or gate fired.
        "phi_hat": None,
        "intercept_hat": None,
        "n_train_residuals": None,
        "oos_n_used": None,
        "skip_reason": None,
    }


# ---------------------------------------------------------------------------
# Public combiner â€” Phase 1
# ---------------------------------------------------------------------------

def build_residual_ensemble(
    anchor_forecast: pd.Series,
    residual_forecast: Optional[pd.Series],
) -> Dict[str, Any]:
    """Combine an anchor forecast with an optional residual correction.

    Phase 1 behaviour
    -----------------
    - Returns the canonical artifact schema in all cases.
    - When ``residual_forecast`` is None: output equals anchor bit-for-bit;
      all metric fields are None; ``residual_status = "inactive"``.
    - When ``residual_forecast`` is provided: y_hat_residual_ensemble =
      anchor + residual over the common index; ``residual_status = "active"``.
    - Metric fields (rmse_*, da_*, corr_anchor_residual) are always None â€”
      they require realized prices and are computed by Agent B at audit time.

    Parameters
    ----------
    anchor_forecast:
        Non-empty ``pd.Series`` from the anchor model (mssa_rl).
    residual_forecast:
        Series of predicted residuals ``epsilon_hat[t]``.  Pass ``None`` to
        disable the correction; output will be identical to anchor.

    Returns
    -------
    dict matching the canonical EXP-R5-001 artifact schema.
    """
    if not isinstance(anchor_forecast, pd.Series):
        raise TypeError(
            f"anchor_forecast must be a pd.Series, got {type(anchor_forecast).__name__}"
        )
    if anchor_forecast.empty:
        raise ValueError("anchor_forecast must not be empty")

    base = {
        "experiment_id": EXPERIMENT_ID,
        "anchor_model_id": ANCHOR_MODEL_ID,
        "phase": CURRENT_PHASE,
        "residual_signal_valid": False,
        "correction_applied": False,
        "reason_code": "INACTIVE",
        # Metric fields: always None in Phase 1 â€” populated by Agent B audit layer.
        "rmse_anchor": None,
        "rmse_residual_ensemble": None,
        "rmse_ratio": None,
        "da_anchor": None,
        "da_residual_ensemble": None,
        "corr_anchor_residual": None,
        "promotion_contract": _PROMOTION_CONTRACT,
        # RC1/RC4 observability â€” None here; populated by forecaster when model fitted.
        "phi_hat": None,
        "intercept_hat": None,
        "n_train_residuals": None,
        "oos_n_used": None,
        "skip_reason": None,
    }

    # --- No residual: output == anchor, fully inactive. ---
    if residual_forecast is None:
        return {
            **base,
            "residual_status": "inactive",
            "residual_active": False,
            "reason": "residual_forecast=None; output == anchor (no regression vs master)",
            "reason_code": "RESIDUAL_FORECAST_MISSING",
            "y_hat_anchor": anchor_forecast.copy(),
            "y_hat_residual_ensemble": anchor_forecast.copy(),
            "residual_mean": None,
            "residual_std": None,
            "n_corrected": 0,
        }

    if not isinstance(residual_forecast, pd.Series):
        raise TypeError(
            f"residual_forecast must be a pd.Series or None, "
            f"got {type(residual_forecast).__name__}"
        )

    # Align on common index (inner) to guard against mismatched horizons.
    common_idx = anchor_forecast.index.intersection(residual_forecast.index)
    if common_idx.empty:
        return {
            **base,
            "residual_status": "inactive",
            "residual_active": False,
            "reason": "residual_forecast index has no overlap with anchor â€” fallback to anchor",
            "reason_code": "RESIDUAL_INDEX_NO_OVERLAP",
            "y_hat_anchor": anchor_forecast.copy(),
            "y_hat_residual_ensemble": anchor_forecast.copy(),
            "residual_mean": None,
            "residual_std": None,
            "n_corrected": 0,
        }

    anchor_aligned = anchor_forecast.reindex(common_idx)
    residual_aligned = residual_forecast.reindex(common_idx)
    combined_values = anchor_aligned.values + residual_aligned.values
    combined_on_common = pd.Series(
        combined_values, index=common_idx, name="residual_ensemble"
    )

    # Pass through anchor for any horizon steps not covered by the residual.
    if len(common_idx) < len(anchor_forecast.index):
        extra_idx = anchor_forecast.index.difference(common_idx)
        combined_forecast = pd.concat(
            [combined_on_common, anchor_forecast.reindex(extra_idx)]
        ).sort_index()
    else:
        combined_forecast = combined_on_common

    resid_vals = residual_aligned.values.astype(float)
    n_corrected = int(len(common_idx))
    residual_mean = float(np.nanmean(resid_vals)) if n_corrected > 0 else None
    residual_std = float(np.nanstd(resid_vals)) if n_corrected > 0 else None

    return {
        **base,
        "residual_status": "active",
        "residual_active": True,
        "reason": f"residual correction applied over {n_corrected} steps",
        "residual_signal_valid": True,
        "correction_applied": True,
        "reason_code": "OK",
        "y_hat_anchor": anchor_forecast.copy(),
        "y_hat_residual_ensemble": combined_forecast,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "n_corrected": n_corrected,
    }


# ---------------------------------------------------------------------------
# OOS residual target helper â€” leakage-guarded
# ---------------------------------------------------------------------------

def compute_oos_residuals(
    anchor_oos_forecast: pd.Series,
    realized_prices: pd.Series,
    *,
    source: Literal["oos", "in_sample"] = "oos",
) -> pd.Series:
    """Compute out-of-sample residuals for training the residual model.

    ``epsilon[t] = y[t] - y_hat_anchor_oos[t]``

    Parameters
    ----------
    anchor_oos_forecast:
        OOS anchor forecast series.  Must be generated by an anchor that was
        already fitted on its training window and has NOT seen these prices.
    realized_prices:
        Realized prices / targets for the same horizon.
    source:
        Caller must pass ``"oos"`` (the default) to confirm the predictions
        are genuinely out-of-sample.  Passing ``"in_sample"`` raises
        ``ValueError`` immediately â€” this is the leakage guard.

    Returns
    -------
    ``pd.Series`` of residuals on the common index, name ``"oos_residuals"``.
    """
    if source == "in_sample":
        raise ValueError(
            "Leakage guard: source='in_sample' is forbidden. "
            "compute_oos_residuals() requires genuine OOS anchor predictions. "
            "Train the anchor first, then generate OOS forecasts on the held-out window."
        )
    if source != "oos":
        raise ValueError(f"source must be 'oos' or 'in_sample', got {source!r}")

    if not isinstance(anchor_oos_forecast, pd.Series) or not isinstance(
        realized_prices, pd.Series
    ):
        raise TypeError("Both inputs must be pd.Series")

    common_idx = anchor_oos_forecast.index.intersection(realized_prices.index)
    if common_idx.empty:
        raise ValueError(
            "anchor_oos_forecast and realized_prices share no common index entries"
        )

    residuals = (
        realized_prices.reindex(common_idx) - anchor_oos_forecast.reindex(common_idx)
    )
    residuals.name = "oos_residuals"
    return residuals


# ---------------------------------------------------------------------------
# ResidualModel â€” Phase 2: AR(1) fitted on OOS anchor residuals
# ---------------------------------------------------------------------------

class ResidualModel:
    """Lightweight AR(1) residual model for EXP-R5-001.

    Model: epsilon[t] = phi * epsilon[t-1] + c + noise

    Fitted via OLS (numpy.linalg.lstsq) on OOS anchor residuals only.
    phi is clamped to (-0.99, 0.99) to guarantee stationarity.

    Leakage contract (CRITICAL)
    ---------------------------
    ``fit_on_oos_residuals()`` is the ONLY valid fit path.
    Passing ``source="in_sample"`` raises ``ValueError`` immediately.
    In-sample residuals must never enter the fit â€” they capture noise
    from the anchor's own training window and leak target information.

    See: Documentation/EXP_R5_001_RESIDUAL_ENSEMBLE_DESIGN_2026-03-08.md
    """

    MODEL_ID = "resid_mssa_rl_v1"
    _PHI_CLAMP = 0.99  # Keep AR(1) stationary regardless of data
    _MIN_PHI = 0.15    # RC4: skip when autocorrelation is too weak (|phi| < threshold)
    # RC5: near-unit-root ceiling.  phi >= 0.90 means the AR(1) decays so slowly
    # that multi-step corrections converge toward the long-run mean over the full
    # horizon â€” indistinguishable from a constant-drift injection.
    _MAX_PHI_APPLICATION = 0.90
    # RC6: bias-dominated long-run mean.
    # LRM = intercept / (1 - |phi|).  If |LRM| > _MAX_LRM_STD_RATIO * residual_std
    # the model learned a constant offset, not autocorrelation structure.
    _MAX_LRM_STD_RATIO = 2.0
    # RC7: minimum 1-step directional accuracy on the training fold.
    # AR(1) that can't predict direction of its OWN training data should not be applied.
    _MIN_TRAIN_DA = 0.45
    # Minimum OOS residual points required for stable AR(1) validation.
    _MIN_OOS_POINTS = 20

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self._phi: float = 0.0        # AR(1) lag coefficient
        self._intercept: float = 0.0  # constant term
        self._last_residual: float = 0.0  # seed for multi-step prediction
        self._n_train: int = 0
        self._skip_reason: Optional[str] = None  # RC4 observability
        self._reason_code: Optional[str] = None
        self._anchor_rmse_proxy: Optional[float] = None
        self._residual_train_rmse: Optional[float] = None
        self._long_run_mean: Optional[float] = None
        self._local_rmse_anchor_val: Optional[float] = None
        self._local_rmse_corrected_val: Optional[float] = None

    def _skip(self, reason_code: str, reason: str) -> "ResidualModel":
        """Mark model as skipped with standardized reason code."""
        self.is_fitted = False
        self._reason_code = reason_code
        self._skip_reason = reason
        return self

    @staticmethod
    def _fit_ar1(arr: np.ndarray, phi_clamp: float) -> tuple[float, float]:
        """Fit AR(1) by OLS and return (intercept, phi)."""
        if len(arr) < 3:
            return 0.0, 0.0
        y = arr[1:]
        X = np.column_stack([np.ones(len(y)), arr[:-1]])
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        intercept = float(coeffs[0])
        phi = float(np.clip(coeffs[1], -phi_clamp, phi_clamp))
        return intercept, phi

    @staticmethod
    def _rollout_ar1(phi: float, intercept: float, seed: float, horizon: int) -> np.ndarray:
        """Generate AR(1) path for a fixed horizon."""
        out = np.zeros(horizon, dtype=float)
        eps = float(seed)
        for i in range(horizon):
            eps = phi * eps + intercept
            out[i] = eps
        return out

    def fit_on_oos_residuals(
        self,
        oos_residuals: pd.Series,
        *,
        source: Literal["oos", "in_sample"] = "oos",
    ) -> "ResidualModel":
        """Fit on out-of-sample anchor residuals.

        Parameters
        ----------
        oos_residuals:
            Series of ``epsilon[t] = y[t] - y_hat_anchor_oos[t]``, computed
            by ``compute_oos_residuals()``.  Must be OOS â€” not in-sample.
        source:
            Caller must pass ``"oos"`` to confirm leakage-free input.
            ``"in_sample"`` raises ``ValueError`` immediately.
        """
        if source == "in_sample":
            raise ValueError(
                "Leakage guard: source='in_sample' is forbidden. "
                "ResidualModel must be fitted on OOS anchor predictions only. "
                "Use compute_oos_residuals() to build the residual series first."
            )
        if source != "oos":
            raise ValueError(f"source must be 'oos' or 'in_sample', got {source!r}")
        if not isinstance(oos_residuals, pd.Series):
            raise TypeError("oos_residuals must be a pd.Series")

        arr_raw = oos_residuals.dropna().values.astype(float)
        self._n_train = len(arr_raw)
        self._reason_code = None
        self._skip_reason = None
        self._residual_train_rmse = None
        self._long_run_mean = None
        self._local_rmse_anchor_val = None
        self._local_rmse_corrected_val = None
        self._anchor_rmse_proxy = None

        if self._n_train < self._MIN_OOS_POINTS:
            return self._skip(
                "TOO_FEW_OOS_POINTS",
                f"n_train={self._n_train} < {self._MIN_OOS_POINTS}",
            )

        anchor_rmse_proxy = float(np.sqrt(np.mean(np.square(arr_raw))))
        self._anchor_rmse_proxy = anchor_rmse_proxy
        if not np.isfinite(anchor_rmse_proxy) or anchor_rmse_proxy <= 0.0:
            return self._skip(
                "NO_ANCHOR_RMSE_PROXY",
                f"anchor_rmse_proxy={anchor_rmse_proxy}",
            )

        # RC1: demean - fit autocorrelation only, not DC offset.
        arr = arr_raw - float(arr_raw.mean())
        self._intercept, self._phi = self._fit_ar1(arr, self._PHI_CLAMP)
        self._long_run_mean = float(self._intercept / max(1.0 - abs(self._phi), 1e-9))

        # RC4: skip correction when autocorrelation is too weak.
        if abs(self._phi) < self._MIN_PHI:
            return self._skip(
                "PHI_TOO_SMALL",
                f"phi_too_small ({self._phi:.4f} < {self._MIN_PHI}): "
                "autocorrelation too weak to apply correction",
            )

        # RC5: near-unit-root ceiling.
        if abs(self._phi) >= self._MAX_PHI_APPLICATION:
            return self._skip(
                "PHI_TOO_PERSISTENT",
                f"high_phi_near_unit_root ({self._phi:.4f} >= {self._MAX_PHI_APPLICATION}): "
                "multi-step correction converges to long-run mean over forecast horizon",
            )

        # RC6: long-run mean too large relative to anchor RMSE proxy.
        if (
            self._long_run_mean is not None
            and abs(self._long_run_mean) > self._MAX_LRM_STD_RATIO * anchor_rmse_proxy
        ):
            return self._skip(
                "LONG_RUN_MEAN_TOO_LARGE",
                f"bias_dominated_long_run_mean "
                f"(abs(long_run_mean)={abs(self._long_run_mean):.3f} > "
                f"{self._MAX_LRM_STD_RATIO} * anchor_rmse_proxy={anchor_rmse_proxy:.3f})",
            )

        # RC7: train directional usefulness.
        if len(arr) >= 4:
            _eps_pred_train = self._phi * arr[:-1] + self._intercept
            _train_err = arr[1:] - _eps_pred_train
            self._residual_train_rmse = float(np.sqrt(np.mean(np.square(_train_err))))
            _train_da = float(np.mean(np.sign(_eps_pred_train) == np.sign(arr[1:])))
            if _train_da < self._MIN_TRAIN_DA:
                return self._skip(
                    "POOR_TRAIN_DIRECTIONAL_USEFULNESS",
                    f"poor_train_directional_usefulness "
                    f"(train_da={_train_da:.3f} < {self._MIN_TRAIN_DA}): "
                    "AR(1) predicts wrong direction on its own training residuals",
                )

        # Final gate: local usefulness validation.
        holdout_n = max(5, min(20, len(arr) // 4))
        fit_n = len(arr) - holdout_n
        if fit_n < 10:
            return self._skip(
                "TOO_FEW_OOS_POINTS",
                f"fit_n={fit_n} too small for local usefulness validation",
            )
        fit_arr = arr[:fit_n]
        val_arr = arr[fit_n:]
        val_intercept, val_phi = self._fit_ar1(fit_arr, self._PHI_CLAMP)
        val_pred = self._rollout_ar1(val_phi, val_intercept, fit_arr[-1], len(val_arr))
        rmse_anchor_val = float(np.sqrt(np.mean(np.square(val_arr))))
        rmse_corrected_val = float(np.sqrt(np.mean(np.square(val_arr - val_pred))))
        self._local_rmse_anchor_val = rmse_anchor_val
        self._local_rmse_corrected_val = rmse_corrected_val
        if rmse_corrected_val >= rmse_anchor_val:
            return self._skip(
                "LOCAL_USEFULNESS_FAIL",
                f"local_usefulness_fail (rmse_corrected_val={rmse_corrected_val:.4f} "
                f">= rmse_anchor_val={rmse_anchor_val:.4f})",
            )

        self._last_residual = float(arr[-1]) if len(arr) > 0 else 0.0
        self._reason_code = "OK"
        self._skip_reason = None
        self.is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        last_residual: Optional[float] = None,
        *,
        index: Optional[pd.Index] = None,
    ) -> pd.Series:
        """Generate epsilon_hat for the forecast horizon.

        Parameters
        ----------
        horizon:
            Number of steps to forecast.
        last_residual:
            Seed residual for the AR(1) recursion.  Defaults to the last
            training residual (stored during ``fit_on_oos_residuals()``).
        index:
            Optional ``pd.Index`` to align the output with the anchor
            forecast.  Pass ``anchor_series.index`` to ensure overlap in
            ``build_residual_ensemble()``.  Defaults to ``RangeIndex``.

        Returns
        -------
        ``pd.Series`` of residual corrections named ``"residual_forecast"``.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "ResidualModel is not fitted. Call fit_on_oos_residuals() first."
            )
        eps = last_residual if last_residual is not None else self._last_residual
        phi = self._phi
        c = self._intercept
        forecasts: list[float] = []
        for _ in range(horizon):
            eps = phi * eps + c
            forecasts.append(eps)
        idx = index if index is not None else pd.RangeIndex(horizon)
        return pd.Series(forecasts, index=idx, name="residual_forecast")

