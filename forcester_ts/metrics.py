"""
Regression metric utilities shared across the forecasting stack.

All helpers accept pandas Series so they can preserve index alignment
when comparing model forecasts against realised prices.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

EPSILON = 1e-9


def _align_series(
    actual: pd.Series,
    predicted: pd.Series,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Align two series on a common index and drop NaNs."""

    if not isinstance(actual, pd.Series) or not isinstance(predicted, pd.Series):
        return None

    aligned = pd.concat(
        {"actual": actual, "predicted": predicted},
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        return None

    return aligned["actual"].to_numpy(), aligned["predicted"].to_numpy()


def rmse(actual: pd.Series, predicted: pd.Series) -> Optional[float]:
    """Root Mean Squared Error."""

    aligned = _align_series(actual, predicted)
    if aligned is None:
        return None

    actual_vals, predicted_vals = aligned
    return float(np.sqrt(np.mean((predicted_vals - actual_vals) ** 2)))


def smape(actual: pd.Series, predicted: pd.Series) -> Optional[float]:
    """Symmetric Mean Absolute Percentage Error."""

    aligned = _align_series(actual, predicted)
    if aligned is None:
        return None

    actual_vals, predicted_vals = aligned
    denominator = np.maximum(
        np.abs(actual_vals) + np.abs(predicted_vals),
        EPSILON,
    )
    return float(
        2.0 * np.mean(np.abs(predicted_vals - actual_vals) / denominator)
    )


def tracking_error(actual: pd.Series, predicted: pd.Series) -> Optional[float]:
    """
    Tracking error proxy for time-series forecasts.

    Defined as the standard deviation of the residuals (predicted - actual),
    which mirrors the classic portfolio tracking error formula.
    """

    aligned = _align_series(actual, predicted)
    if aligned is None:
        return None

    actual_vals, predicted_vals = aligned
    residual = predicted_vals - actual_vals
    return float(np.std(residual, ddof=0))


def directional_accuracy(actual: pd.Series, predicted: pd.Series) -> Optional[float]:
    """
    Directional accuracy of forecasts based on one-step price changes.

    Computes the fraction of periods where the sign of the forecasted
    price change matches the sign of the realised price change. Periods
    with zero change in both series are counted as correct.
    """
    aligned = _align_series(actual, predicted)
    if aligned is None:
        return None

    actual_vals, predicted_vals = aligned
    if actual_vals.size < 2:
        return None

    actual_diff = np.diff(actual_vals)
    pred_diff = np.diff(predicted_vals)
    # Avoid division-by-zero; treat exact zeros separately.
    actual_sign = np.sign(actual_diff)
    pred_sign = np.sign(pred_diff)
    correct = (actual_sign == pred_sign)
    return float(np.mean(correct)) if correct.size > 0 else None


def terminal_directional_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
) -> Optional[float]:
    """
    Terminal directional accuracy: did the forecast correctly predict whether
    the FINAL value of the horizon is above or below the FIRST value?

    This is the metric that maps directly to multi-step trade P&L:
      correct = sign(predicted[-1] - predicted[0]) == sign(actual[-1] - actual[0])

    Unlike 1-step DA (which averages bar-by-bar direction), terminal DA is a
    single binary outcome per window — it must be accumulated over ≥30 windows
    before it is statistically meaningful (SE = ±sqrt(0.25/n)).

    Returns 1.0 (correct) or 0.0 (incorrect) for a single window, or the
    mean over all aligned windows when the inputs contain multiple windows.
    """
    aligned = _align_series(actual, predicted)
    if aligned is None:
        return None

    actual_vals, predicted_vals = aligned
    if actual_vals.size < 2:
        return None

    actual_direction = float(np.sign(actual_vals[-1] - actual_vals[0]))
    pred_direction = float(np.sign(predicted_vals[-1] - predicted_vals[0]))
    # Flat forecast (pred == 0) or flat actuals (actual == 0) are treated as
    # incorrect since a flat prediction provides no directional signal.
    if pred_direction == 0 or actual_direction == 0:
        return 0.0
    return 1.0 if actual_direction == pred_direction else 0.0


def terminal_ci_coverage(
    actual: pd.Series,
    lower_ci: pd.Series,
    upper_ci: pd.Series,
) -> Optional[float]:
    """
    Terminal CI coverage: did the actual terminal price fall within the
    predicted CI at the terminal forecast step?

    Returns 1.0 if covered, 0.0 if not covered, None if data is insufficient.

    Accumulated over many windows, the mean of this metric is the empirical
    coverage rate. If the nominal CI is 80% but the empirical rate is 40%,
    the CI is too narrow by a factor of ~2x and SNR is inflated accordingly.
    """
    if not isinstance(actual, pd.Series) or actual.empty:
        return None
    if not isinstance(lower_ci, pd.Series) or lower_ci.empty:
        return None
    if not isinstance(upper_ci, pd.Series) or upper_ci.empty:
        return None

    # Use the last point where all three series have valid data
    combined = pd.concat(
        {"actual": actual, "lower": lower_ci, "upper": upper_ci},
        axis=1,
        join="inner",
    ).dropna()
    if combined.empty:
        return None

    last = combined.iloc[-1]
    actual_val = float(last["actual"])
    lower_val = float(last["lower"])
    upper_val = float(last["upper"])

    if not (np.isfinite(actual_val) and np.isfinite(lower_val) and np.isfinite(upper_val)):
        return None
    return 1.0 if lower_val <= actual_val <= upper_val else 0.0


def compute_regression_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    lower_ci: Optional[pd.Series] = None,
    upper_ci: Optional[pd.Series] = None,
) -> Optional[Dict[str, float]]:
    """Return all supported regression metrics for the provided series."""

    aligned = _align_series(actual, predicted)
    if aligned is None:
        return None

    tda = terminal_directional_accuracy(actual, predicted)
    ci_cov = terminal_ci_coverage(actual, lower_ci, upper_ci) if (
        lower_ci is not None and upper_ci is not None
    ) else None
    metrics = {
        "rmse": rmse(actual, predicted),
        "smape": smape(actual, predicted),
        "tracking_error": tracking_error(actual, predicted),
        "directional_accuracy": directional_accuracy(actual, predicted),
        "terminal_directional_accuracy": tda,
        "terminal_ci_coverage": ci_cov,
        "n_observations": int(
            pd.concat([actual, predicted], axis=1, join="inner").dropna().shape[0]
        ),
    }
    return {k: float(v) for k, v in metrics.items() if v is not None}


__all__ = [
    "compute_regression_metrics",
    "rmse",
    "smape",
    "tracking_error",
    "directional_accuracy",
    "terminal_directional_accuracy",
    "terminal_ci_coverage",
]
