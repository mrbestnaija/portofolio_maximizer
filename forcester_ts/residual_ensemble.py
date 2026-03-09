"""
forcester_ts/residual_ensemble.py

EXP-R5-001: Residual Ensemble Around mssa_rl (Research Experiment — Agent A).

PHASE STRUCTURE
---------------
Phase 1 (this file): Architecture plumbing only.
  - Canonical artifact schema defined.
  - `build_residual_ensemble()` combines anchor + optional residual.
  - `ResidualModel` stub exists — not yet trained (Phase 2).
  - Metric fields (rmse_*, da_*, corr_anchor_residual) are always None here;
    they are computed at audit time by Agent B (Phase 3).
  - OOS source guard on `compute_oos_residuals()`.

Phase 2: ResidualModel fitted on OOS anchor residuals — real epsilon_hat.
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

# Canonical field set — single source of truth for the artifact schema.
# Agent B imports this via quality_pipeline_common; the instrumentation layer
# uses inactive_artifact().keys() as a cross-check.  Both must stay in sync.
CANONICAL_FIELDS: frozenset = frozenset({
    "experiment_id", "anchor_model_id", "phase",
    "residual_status", "residual_active", "reason",
    "y_hat_anchor", "y_hat_residual_ensemble",
    "rmse_anchor", "rmse_residual_ensemble", "rmse_ratio",
    "da_anchor", "da_residual_ensemble", "corr_anchor_residual",
    "residual_mean", "residual_std", "n_corrected",
    "promotion_contract",
    # RC1/RC4 observability — always emitted; None when model not fitted.
    "phi_hat", "intercept_hat", "n_train_residuals", "oos_n_used", "skip_reason",
})

# Promotion contract thresholds — advisory, never enforced here.
#
# corr_threshold REMOVED (2026-03-08):
# The original criterion was corr(y_hat_anchor, y_hat_residual_ensemble) ≤ 0.90.
# For a bias-correcting residual model (the primary use case), the ensemble is
# anchor + constant_offset → Pearson corr = 1.0 regardless of how well the bias
# is removed.  The criterion would always reject good corrections.
#
# Replacement: corr(epsilon[t], epsilon_hat[t]) — the correlation between actual
# OOS anchor errors and the residual model's predictions of those errors.  A value
# of ≥ 0.30 over ≥ 20 windows is the meaningful diversity signal.  This is a
# Phase 3 Agent B metric (requires realized prices); it is NOT computable here.
#
# Early-termination signal (C3): rmse_ratio > 1.02 across ≥ 5 consecutive windows
# indicates the residual correction is actively harmful — stop accumulating.
_PROMOTION_CONTRACT: Dict[str, Any] = {
    "rmse_ratio_threshold": 0.98,
    "residual_prediction_corr_threshold": 0.30,  # corr(epsilon[t], epsilon_hat[t]) ≥ 0.30
    "residual_prediction_corr_note": (
        "correlation between realized anchor errors and residual forecasts; "
        "requires realized prices — computed by Agent B at audit time"
    ),
    "early_stop_rmse_ratio": 1.02,  # stop if ratio > 1.02 across >= 5 windows
    "min_effective_audits": 20,
    "note": "advisory only — not enforced at runtime",
}


def inactive_artifact(*, reason: str = "experiment disabled") -> Dict[str, Any]:
    """Return the canonical inactive artifact schema.

    All metric fields are explicitly None — unknown is not 0.0 and not PASS.
    Called by the forecaster when the experiment path is not active.
    """
    return {
        "experiment_id": EXPERIMENT_ID,
        "anchor_model_id": ANCHOR_MODEL_ID,
        "phase": CURRENT_PHASE,
        "residual_status": "inactive",
        "residual_active": False,
        "reason": reason,
        # Forecast series — None means "not computed", not "zero correction".
        "y_hat_anchor": None,
        "y_hat_residual_ensemble": None,
        # Phase 3 audit metrics — always None here; populated by Agent B.
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
        # RC1/RC4 observability — None means model not fitted or gate fired.
        "phi_hat": None,
        "intercept_hat": None,
        "n_train_residuals": None,
        "oos_n_used": None,
        "skip_reason": None,
    }


# ---------------------------------------------------------------------------
# Public combiner — Phase 1
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
    - Metric fields (rmse_*, da_*, corr_anchor_residual) are always None —
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
        # Metric fields: always None in Phase 1 — populated by Agent B audit layer.
        "rmse_anchor": None,
        "rmse_residual_ensemble": None,
        "rmse_ratio": None,
        "da_anchor": None,
        "da_residual_ensemble": None,
        "corr_anchor_residual": None,
        "promotion_contract": _PROMOTION_CONTRACT,
        # RC1/RC4 observability — None here; populated by forecaster when model fitted.
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
            "reason": "residual_forecast index has no overlap with anchor — fallback to anchor",
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
        "y_hat_anchor": anchor_forecast.copy(),
        "y_hat_residual_ensemble": combined_forecast,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "n_corrected": n_corrected,
    }


# ---------------------------------------------------------------------------
# OOS residual target helper — leakage-guarded
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
        ``ValueError`` immediately — this is the leakage guard.

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
# ResidualModel — Phase 2: AR(1) fitted on OOS anchor residuals
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
    In-sample residuals must never enter the fit — they capture noise
    from the anchor's own training window and leak target information.

    See: Documentation/EXP_R5_001_RESIDUAL_ENSEMBLE_DESIGN_2026-03-08.md
    """

    MODEL_ID = "resid_mssa_rl_v1"
    _PHI_CLAMP = 0.99  # Keep AR(1) stationary regardless of data
    _MIN_PHI = 0.15    # RC4: skip correction when autocorrelation is too weak

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self._phi: float = 0.0        # AR(1) lag coefficient
        self._intercept: float = 0.0  # constant term
        self._last_residual: float = 0.0  # seed for multi-step prediction
        self._n_train: int = 0
        self._skip_reason: Optional[str] = None  # RC4 observability

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
            by ``compute_oos_residuals()``.  Must be OOS — not in-sample.
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

        arr = oos_residuals.dropna().values.astype(float)
        # RC1: demean — fit autocorrelation only, not DC offset.
        # A non-zero mean signals that the anchor has a systematic bias relative
        # to the OOS proxy anchor; applying that bias to the full anchor would
        # inject anti-signal.  Subtracting the mean lets AR(1) capture genuine
        # lag-1 structure only.
        arr = arr - arr.mean()
        self._n_train = len(arr)

        if len(arr) < 3:
            # Too few points for AR(1) — fall back to zero (demeaned constant).
            self._phi = 0.0
            self._intercept = 0.0
        else:
            # OLS: [1, epsilon[t-1]] -> epsilon[t]
            y = arr[1:]
            X = np.column_stack([np.ones(len(y)), arr[:-1]])
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self._intercept = float(coeffs[0])
            # Clamp phi to (-_PHI_CLAMP, +_PHI_CLAMP) to ensure stationarity.
            self._phi = float(
                np.clip(coeffs[1], -self._PHI_CLAMP, self._PHI_CLAMP)
            )

        # RC4: skip correction when autocorrelation is too weak.
        # |phi| < _MIN_PHI means the AR(1) is essentially white noise — applying
        # it would add noise without correcting anything systematic.
        if abs(self._phi) < self._MIN_PHI:
            self.is_fitted = False
            self._skip_reason = (
                f"phi_too_small ({self._phi:.4f} < {self._MIN_PHI}): "
                "autocorrelation too weak to apply correction"
            )
            return self

        self._last_residual = float(arr[-1]) if len(arr) > 0 else 0.0
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
