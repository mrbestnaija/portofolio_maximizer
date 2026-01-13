"""
SAMOSSA (Singular Spectrum Analysis + ARIMAX hybrid) forecaster.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.decomposition import TruncatedSVD

try:
    from statsmodels.tsa.ar_model import AutoReg
except Exception:  # pragma: no cover - optional dependency guard
    AutoReg = None

logger = logging.getLogger(__name__)


@dataclass
class SAMOSSAConfig:
    window_length: int = 40
    n_components: int = 6
    use_residual_arima: bool = True
    min_series_length: int = 120
    forecast_horizon: int = 30
    normalize: bool = True
    ar_order: int = 5
    matrix_type: Literal["page", "hankel"] = "page"


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
        forecast_horizon: int = 30,
        normalize: bool = True,
        ar_order: int = 5,
        matrix_type: Literal["page", "hankel"] = "page",
    ) -> None:
        self.config = SAMOSSAConfig(
            window_length=window_length,
            n_components=n_components,
            use_residual_arima=use_residual_arima,
            min_series_length=min_series_length,
            forecast_horizon=forecast_horizon,
            normalize=normalize,
            ar_order=ar_order,
            matrix_type=matrix_type,
        )
        self._fitted = False
        self._reconstructed: Optional[pd.Series] = None
        self._residuals: Optional[pd.Series] = None
        self._explained_variance_ratio: float = 0.0
        self._last_index: Optional[pd.Timestamp] = None
        self._trajectory_shape: Tuple[int, int] = (0, 0)
        self._target_freq: Optional[str] = None
        self._scale_mean: float = 0.0
        self._scale_std: float = 1.0
        self._residual_model: Any = None
        self._normalized_stats: Dict[str, float] = {"mean": 0.0, "std": 1.0}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_page_matrix(self, series: pd.Series) -> np.ndarray:
        L = self.config.window_length
        T = len(series)
        K = T // L
        if K <= 0:
            raise ValueError("Window length larger than series length")
        segments = [series.values[i * L : (i + 1) * L] for i in range(K)]
        return np.column_stack(segments)

    def _build_hankel_matrix(self, series: pd.Series) -> np.ndarray:
        L = self.config.window_length
        T = len(series)
        K = T - L + 1

        if K <= 0:
            raise ValueError("Window length larger than series length")

        return np.column_stack([series.values[i : i + L] for i in range(K)])

    def _ssa_decompose(self, series: pd.Series) -> np.ndarray:
        if self.config.matrix_type.lower() == "page":
            trajectory = self._build_page_matrix(series)
        else:
            trajectory = self._build_hankel_matrix(series)
        self._trajectory_shape = trajectory.shape
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

    def _fit_residual_trend(self, residuals: pd.Series, steps: int) -> np.ndarray:
        if not self.config.use_residual_arima or len(residuals) < 5:
            return np.zeros(steps)
        if self._residual_model is not None:
            try:
                start = len(residuals)
                end = start + steps - 1
                forecast = self._residual_model.predict(start=start, end=end)
                return np.asarray(forecast)
            except Exception as exc:  # pragma: no cover - fallback
                logger.debug("AutoReg residual forecast failed: %s", exc)
        x = np.arange(len(residuals))
        coeffs = np.polyfit(x, residuals.values, deg=min(2, len(residuals) - 1))
        poly = np.poly1d(coeffs)
        future_x = np.arange(len(residuals), len(residuals) + steps)
        return poly(future_x)

    def _fit_residual_model(self, residuals: pd.Series) -> None:
        if not self.config.use_residual_arima or AutoReg is None:
            self._residual_model = None
            return
        max_order = min(self.config.ar_order, max(1, len(residuals) // 4))
        if max_order < 1:
            self._residual_model = None
            return
        residuals = residuals.copy()
        residuals.index = pd.DatetimeIndex(residuals.index).tz_localize(None)
        freq = self._target_freq or residuals.index.freqstr or residuals.index.inferred_freq
        if freq:
            try:
                residuals.index = residuals.index.asfreq(freq)
            except Exception:
                residuals.index = pd.RangeIndex(start=0, stop=len(residuals), step=1)
        else:
            residuals.index = pd.RangeIndex(start=0, stop=len(residuals), step=1)
        try:
            self._residual_model = AutoReg(
                residuals, lags=max_order, old_names=False
            ).fit()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("AutoReg fitting failed for SAMOSSA residuals: %s", exc)
            self._residual_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "SAMOSSAForecaster":
        if len(series) < self.config.min_series_length:
            raise ValueError("Series too short for SAMOSSA decomposition")

        freq_hint = None
        try:
            freq_hint = series.attrs.get("_pm_freq_hint")
        except Exception:
            freq_hint = None

        cleaned = series.sort_index()
        cleaned = (
            cleaned.interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
        )
        cleaned = cleaned.dropna()
        if len(cleaned) < self.config.min_series_length:
            raise ValueError("Insufficient non-NaN observations for SAMOSSA")

        self._last_index = cleaned.index[-1]
        self._target_freq = cleaned.index.freqstr or cleaned.index.inferred_freq or freq_hint or "D"

        if self.config.normalize:
            self._scale_mean = float(cleaned.mean())
            std = float(cleaned.std())
            self._scale_std = std if std > 0 else 1.0
            normalized = (cleaned - self._scale_mean) / self._scale_std
            self._normalized_stats = {
                "mean": float(normalized.mean()),
                "std": float(normalized.std()) if len(normalized) > 1 else 1.0,
            }
        else:
            self._scale_mean = 0.0
            self._scale_std = 1.0
            normalized = cleaned.copy()
            self._normalized_stats = {
                "mean": float(normalized.mean()) if len(normalized) else 0.0,
                "std": float(normalized.std()) if len(normalized) > 1 else 1.0,
            }

        window_cap = max(5, int(np.sqrt(len(normalized))))
        window = min(self.config.window_length, window_cap)
        if window < 5:
            window = 5
        self.config.window_length = window

        usable_length = (len(normalized) // window) * window
        if usable_length < window:
            raise ValueError("Unable to construct SAMOSSA window from provided series")

        normalized_tail = normalized.iloc[-usable_length:]
        observed_tail = cleaned.iloc[-usable_length:]
        try:
            normalized_tail.index = normalized_tail.index.asfreq(self._target_freq)
            observed_tail.index = observed_tail.index.asfreq(self._target_freq)
        except Exception:
            pass

        matrix = self._ssa_decompose(normalized_tail)
        if self.config.matrix_type.lower() == "page":
            recon = matrix.reshape(-1, order="F")
        else:
            recon = self._diagonal_averaging(matrix)
        if recon.shape[0] != len(normalized_tail):
            recon = recon[: len(normalized_tail)]
        recon_series_norm = pd.Series(recon, index=normalized_tail.index)
        recon_series = recon_series_norm * self._scale_std + self._scale_mean

        residuals = observed_tail - recon_series
        self._reconstructed = recon_series
        self._residuals = residuals
        self._fit_residual_model(residuals)
        self._fitted = True

        logger.info(
            "SAMOSSA fit complete (matrix=%s, window=%s, components=%s, EVR=%.3f)",
            self.config.matrix_type,
            window,
            self.config.n_components,
            self._explained_variance_ratio,
        )
        return self

    def forecast(self, steps: int) -> Dict[str, Any]:
        if not self._fitted or self._reconstructed is None:
            raise ValueError("Fit must be called before forecast")

        base_forecast = np.full(steps, self._reconstructed.iloc[-1])
        residual_forecast = self._fit_residual_trend(self._residuals, steps)

        combined = base_forecast + residual_forecast[:steps]

        if self._last_index is not None:
            freq = self._target_freq or (self._reconstructed.index.freqstr if hasattr(self._reconstructed.index, "freqstr") else None) or "D"
            offset = to_offset(freq)
            start = self._last_index + offset
            future_index = pd.date_range(start=start, periods=steps, freq=freq)
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
            "residual_model_order": getattr(self._residual_model, "k_ar", 0) if self._residual_model is not None else 0,
        }

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "window_length_used": self.config.window_length,
            "n_components": self.config.n_components,
            "explained_variance_ratio": self._explained_variance_ratio,
            "use_residual_arima": self.config.use_residual_arima,
            "trajectory_matrix_shape": self._trajectory_shape,
            "scale_mean": self._scale_mean,
            "scale_std": self._scale_std,
            "normalized_mean": self._normalized_stats.get("mean"),
            "normalized_std": self._normalized_stats.get("std"),
        }
