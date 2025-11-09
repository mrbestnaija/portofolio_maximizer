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


def compute_regression_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> Optional[Dict[str, float]]:
    """Return all supported regression metrics for the provided series."""

    aligned = _align_series(actual, predicted)
    if aligned is None:
        return None

    metrics = {
        "rmse": rmse(actual, predicted),
        "smape": smape(actual, predicted),
        "tracking_error": tracking_error(actual, predicted),
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
]
