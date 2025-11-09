"""
SAMOSSA (Singular Spectrum Analysis + ARIMAX hybrid) forecaster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


@dataclass
class SAMOSSAConfig:
    window_length: int = 40
    n_components: int = 6
    use_residual_arima: bool = True
    min_series_length: int = 120
    forecast_horizon: int = 30


class SAMOSSAForecaster:
    """
    Simplified SAMOSSA implementation suitable for lightweight forecasts.

    It performs:
      1. Singular Spectrum Analysis decomposition.
      2. Optional ARIMA modelling of residuals via numpy polyfit.
      3. Reconstruction of the signal for the requested horizon.
    """

    def __init__(
        self,
        window_length: int = 40,
        n_components: int = 6,
        use_residual_arima: bool = True,
        min_series_length: int = 120,
    ) -> None:
        self.config = SAMOSSAConfig(
            window_length=window_length,
            n_components=n_components,
            use_residual_arima=use_residual_arima,
            min_series_length=min_series_length,
        )
        self._fitted = False
        self._reconstructed: Optional[pd.Series] = None
        self._residuals: Optional[pd.Series] = None
        self._explained_variance_ratio: float = 0.0
        self._last_index: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_trajectory_matrix(self, series: pd.Series) -> np.ndarray:
        L = self.config.window_length
        T = len(series)
        K = T - L + 1

        if K <= 0:
            raise ValueError("Window length larger than series length")

        trajectory = np.column_stack(
            [series.values[i : i + L] for i in range(K)]
        )  # L x K
        return trajectory

    def _ssa_decompose(self, series: pd.Series) -> np.ndarray:
        trajectory = self._build_trajectory_matrix(series)
        svd = TruncatedSVD(n_components=self.config.n_components, random_state=0)
        components = svd.fit_transform(trajectory)
        self._explained_variance_ratio = float(svd.explained_variance_ratio_.sum())
        return components @ svd.components_

    def _diagonal_averaging(self, matrix: np.ndarray) -> np.ndarray:
        L, K = matrix.shape
        T = L + K - 1
        recon = np.zeros(T)
        counts = np.zeros(T)

        for i in range(L):
            for j in range(K):
                recon[i + j] += matrix[i, j]
                counts[i + j] += 1

        return recon / counts

    def _fit_residual_trend(self, residuals: pd.Series) -> np.ndarray:
        if not self.config.use_residual_arima or len(residuals) < 10:
            return np.zeros(self.config.forecast_horizon)

        x = np.arange(len(residuals))
        coeffs = np.polyfit(x, residuals.values, deg=min(2, len(residuals) - 1))
        poly = np.poly1d(coeffs)
        future_x = np.arange(len(residuals), len(residuals) + self.config.forecast_horizon)
        return poly(future_x)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "SAMOSSAForecaster":
        if len(series) < self.config.min_series_length:
            raise ValueError("Series too short for SAMOSSA decomposition")

        cleaned = series.dropna()
        if len(cleaned) < self.config.min_series_length:
            raise ValueError("Insufficient non-NaN observations for SAMOSSA")

        self._last_index = cleaned.index[-1]

        window = min(self.config.window_length, len(cleaned) // 2)
        if window < 5:
            window = 5
        self.config.window_length = window

        matrix = self._ssa_decompose(cleaned)
        recon = self._diagonal_averaging(matrix)
        recon_series = pd.Series(recon, index=cleaned.index)

        residuals = cleaned - recon_series
        self._reconstructed = recon_series
        self._residuals = residuals
        self._fitted = True

        logger.info(
            "SAMOSSA fit complete (window=%s, components=%s, EVR=%.3f)",
            window,
            self.config.n_components,
            self._explained_variance_ratio,
        )
        return self

    def forecast(self, steps: int) -> Dict[str, Any]:
        if not self._fitted or self._reconstructed is None:
            raise ValueError("Fit must be called before forecast")

        base_forecast = np.full(steps, self._reconstructed.iloc[-1])
        residual_forecast = self._fit_residual_trend(self._residuals)

        combined = base_forecast + residual_forecast[:steps]

        if self._last_index is not None:
            future_index = pd.date_range(
                self._last_index + pd.Timedelta(days=1),
                periods=steps,
                freq=self._reconstructed.index.freq or "D",
            )
        else:
            future_index = pd.RangeIndex(start=0, stop=steps, step=1)

        forecast_series = pd.Series(combined, index=future_index)
        noise_level = float(self._residuals.std()) if self._residuals is not None else 0.0

        return {
            "forecast": forecast_series,
            "lower_ci": forecast_series - noise_level,
            "upper_ci": forecast_series + noise_level,
            "explained_variance_ratio": self._explained_variance_ratio,
            "window_length_used": self.config.window_length,
            "n_components": self.config.n_components,
        }

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "window_length_used": self.config.window_length,
            "n_components": self.config.n_components,
            "explained_variance_ratio": self._explained_variance_ratio,
            "use_residual_arima": self.config.use_residual_arima,
        }
