"""
SAMOSSA (Singular Spectrum Analysis + ARIMAX hybrid) forecaster.
"""

from __future__ import annotations

import logging
import warnings as _warnings
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.decomposition import TruncatedSVD

from ._freq_compat import normalize_freq

try:
    from statsmodels.tsa.ar_model import AutoReg
except Exception:  # pragma: no cover - optional dependency guard
    AutoReg = None

try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA  # type: ignore[import]

    ARIMA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency guard
    _ARIMA = None  # type: ignore[assignment]
    ARIMA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SAMOSSAConfig:
    window_length: int = 40    # 0 = auto: min(T//3, 40); phase 7.10b: paper recommends <= T/3
    n_components: int = 6
    auto_select: bool = False
    variance_target: float = 0.90
    use_residual_arima: bool = True
    min_series_length: int = 120
    forecast_horizon: int = 30
    normalize: bool = True
    ar_order: int = 5
    matrix_type: Literal["page", "hankel"] = "page"
    # Phase 7.16: AR(1) replaces ARMA(1,1). Near-cancellation of AR/MA roots on
    # near-white-noise SSA residuals caused ConvergenceWarning on every fit.
    # Set to None to activate AIC-guided AR lag auto-search (AR1..max_ar_lag).
    arima_order: Optional[Tuple[int, int, int]] = (1, 0, 0)
    max_ar_lag: int = 4          # Phase 7.16: Max AR lag tried when arima_order=None
    trend_slope_bars: int = 10  # Phase 7.10b: bars over which directional slope signal is computed


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
        auto_select: bool = False,
        variance_target: float = 0.90,
        use_residual_arima: bool = True,
        min_series_length: int = 120,
        forecast_horizon: int = 30,
        normalize: bool = True,
        ar_order: int = 5,
        matrix_type: Literal["page", "hankel"] = "page",
        arima_order: Tuple[int, int, int] = (1, 0, 0),  # Phase 7.16: AR(1) — see SAMOSSAConfig
        trend_slope_bars: int = 10,
    ) -> None:
        self.config = SAMOSSAConfig(
            window_length=window_length,
            n_components=n_components,
            auto_select=auto_select,
            variance_target=variance_target,
            use_residual_arima=use_residual_arima,
            min_series_length=min_series_length,
            forecast_horizon=forecast_horizon,
            normalize=normalize,
            ar_order=ar_order,
            matrix_type=matrix_type,
            arima_order=tuple(arima_order) if arima_order else (1, 0, 1),  # type: ignore[arg-type]
            trend_slope_bars=max(3, int(trend_slope_bars)),
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
        self._trend_slope: float = 0.0
        self._trend_intercept: float = 0.0
        self._trend_strength: float = 0.0
        # Phase 8.2: residual diagnostics populated after fit() completes.
        self._residual_diagnostics: Dict[str, Any] = {}
        self._last_observed: Optional[float] = None

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
        max_components = min(trajectory.shape)
        if max_components <= 0:
            raise ValueError("Trajectory matrix too small for decomposition")

        n_components = int(self.config.n_components)
        auto_select = bool(self.config.auto_select or n_components <= 0)
        if auto_select:
            target = float(self.config.variance_target or 0.90)
            target = max(0.5, min(target, 0.999))
            svd = TruncatedSVD(n_components=max_components, random_state=0)
            transformed = svd.fit_transform(trajectory)
            explained = np.asarray(svd.explained_variance_ratio_, dtype=float)
            if explained.size == 0:
                self._explained_variance_ratio = 0.0
                self.config.n_components = 1
                return transformed @ svd.components_
            cumulative = np.cumsum(explained)
            n_select = int(np.searchsorted(cumulative, target) + 1)
            n_select = max(1, min(n_select, max_components))
            self.config.n_components = n_select
            self._explained_variance_ratio = float(cumulative[n_select - 1])
            return transformed[:, :n_select] @ svd.components_[:n_select, :]

        n_components = max(1, min(n_components, max_components))
        self.config.n_components = n_components
        svd = TruncatedSVD(n_components=n_components, random_state=0)
        components = svd.fit_transform(trajectory)
        self._explained_variance_ratio = float(svd.explained_variance_ratio_.sum())
        return components @ svd.components_

    def _estimate_trend(self, series: pd.Series, window: int) -> Tuple[float, float, float]:
        if series is None or len(series) < 2:
            last_val = float(series.iloc[-1]) if series is not None and len(series) else 0.0
            return 0.0, last_val, 0.0
        window = max(2, min(int(window), len(series)))
        tail = series.iloc[-window:]
        x = np.arange(len(tail), dtype=float)
        y = tail.values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        resid_std = float(np.std(resid)) if len(resid) > 1 else 0.0
        strength = abs(float(slope)) / (resid_std + 1e-8)
        return float(slope), float(intercept), float(strength)

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
                # ARIMA fitted model: use forecast() for out-of-sample steps.
                # AutoReg fitted model: use predict(start, end).
                # Try forecast() first (ARIMA API), then predict() (AutoReg API).
                if hasattr(self._residual_model, "forecast"):
                    fc = self._residual_model.forecast(steps=steps)
                    return np.asarray(fc)
                start = len(residuals)
                end = start + steps - 1
                forecast = self._residual_model.predict(start=start, end=end)
                return np.asarray(forecast)
            except Exception as exc:  # pragma: no cover - fallback
                logger.debug("Residual model forecast failed: %s", exc)
        # Last-resort: simple linear extrapolation from the last k residuals.
        # This replaces the old polyfit to avoid overfitting with degree-2 polynomial.
        k = min(20, len(residuals))
        recent = residuals.values[-k:]
        slope = float(np.polyfit(np.arange(k), recent, deg=1)[0]) if k >= 2 else 0.0
        return np.arange(1, steps + 1) * slope

    def _fit_residual_model(self, residuals: pd.Series) -> None:
        if not self.config.use_residual_arima:
            self._residual_model = None
            return
        if len(residuals) < 5:
            self._residual_model = None
            return

        residuals_plain = residuals.copy()
        residuals_plain.index = pd.RangeIndex(start=0, stop=len(residuals_plain), step=1)

        # Phase 7.16: AR lag auto-search when arima_order=None (AIC-guided).
        # When arima_order is None the config activates AIC-guided search from
        # AR(1) up to AR(max_ar_lag), using AutoReg for each lag candidate.
        if getattr(self.config, "arima_order", (1, 0, 0)) is None and AutoReg is not None:
            max_lag = max(1, int(getattr(self.config, "max_ar_lag", 4)))
            best_lag, best_aic, best_ar_fit = 1, float("inf"), None
            for lag in range(1, max_lag + 1):
                with _warnings.catch_warnings(record=True) as _w:
                    _warnings.simplefilter("always")
                    try:
                        _candidate = AutoReg(residuals_plain, lags=lag, old_names=False).fit()
                    except Exception:
                        continue
                if any("convergence" in str(w.message).lower() for w in _w):
                    continue
                try:
                    _cand_aic = float(_candidate.aic)
                except Exception:
                    _cand_aic = float("inf")
                if _cand_aic < best_aic:
                    best_aic, best_lag, best_ar_fit = _cand_aic, lag, _candidate
            self._learned_ar_lag = best_lag
            self._learned_ar_aic = best_aic
            if best_ar_fit is not None:
                self._residual_model = best_ar_fit
                logger.debug(
                    "SAMOSSA: AR lag auto-search selected AR(%d) (AIC=%.2f) on %d obs.",
                    best_lag, best_aic, len(residuals_plain),
                )
            else:
                self._residual_model = None
            return

        # Phase 7.10b: Use ARIMA(p,d,q) from config instead of AutoReg.
        # This actually uses the arima_order parameter (previously ignored despite config).
        # Phase 7.16: Capture convergence warnings so a non-converged ARIMA is NOT stored
        # as the residual model (previously only exceptions triggered the AutoReg fallback;
        # statsmodels emits ConvergenceWarning without raising, so the bad model was kept).
        if ARIMA_AVAILABLE and _ARIMA is not None:
            arima_order = getattr(self.config, "arima_order", (1, 0, 1)) or (1, 0, 0)
            try:
                with _warnings.catch_warnings(record=True) as _caught:
                    _warnings.simplefilter("always")
                    arima_fit = _ARIMA(residuals_plain, order=arima_order).fit()
                _convergence_warning = any(
                    issubclass(w.category, Warning)
                    and any(
                        kw in str(w.message).lower()
                        for kw in ("convergence", "nonstationary", "non-stationary",
                                   "non-invertible", "non_invertible")
                    )
                    for w in _caught
                )
                if _convergence_warning:
                    logger.debug(
                        "ARIMA%s residual fit issued convergence/stationarity warning; "
                        "falling back to AutoReg.",
                        arima_order,
                    )
                else:
                    self._residual_model = arima_fit
                    logger.debug(
                        "SAMOSSA: ARIMA%s residual model fitted on %d observations.",
                        arima_order,
                        len(residuals_plain),
                    )
                    return
            except Exception as arima_exc:
                logger.debug(
                    "ARIMA%s residual fit failed (%s); falling back to AutoReg.",
                    arima_order,
                    arima_exc,
                )

        # Fallback: AutoReg (original behaviour)
        if AutoReg is None:
            self._residual_model = None
            return
        max_order = min(self.config.ar_order, max(1, len(residuals) // 4))
        if max_order < 1:
            self._residual_model = None
            return
        try:
            self._residual_model = AutoReg(
                residuals_plain, lags=max_order, old_names=False
            ).fit()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("AutoReg fitting failed for SAMOSSA residuals: %s", exc)
            self._residual_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        series: pd.Series,
        order_learner=None,
        ticker: str = "",
        regime: str | None = None,
    ) -> "SAMOSSAForecaster":
        series_len = len(series)
        if series_len < self.config.min_series_length:
            adaptive_min = max(20, self.config.n_components * 5)
            if series_len < adaptive_min:
                raise ValueError("Series too short for SAMOSSA decomposition")
            logger.debug(
                "SAMOSSA: relaxing min_series_length from %s to %s for short series",
                self.config.min_series_length,
                series_len,
            )
            self.config.min_series_length = series_len

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
        self._target_freq = normalize_freq(
            cleaned.index.freqstr or cleaned.index.inferred_freq or freq_hint
        ) or "D"

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

        # Phase 7.10b: SAMoSSA paper recommends window_length <= T/3.
        # Auto mode (window_length=0/null): use T//3 directly.
        # Configured mode: cap at T//3 to enforce paper constraint; log override.
        series_t = len(normalized)
        paper_cap = max(5, series_t // 3)  # Paper hard limit; no arbitrary secondary cap
        configured_w = int(self.config.window_length) if self.config.window_length else 0
        if configured_w <= 0:
            # Auto mode: use paper recommendation directly
            window = paper_cap
        else:
            window = min(configured_w, paper_cap)
            if configured_w > paper_cap:
                logger.debug(
                    "SAMOSSA: configured window_length=%d exceeds T//3=%d; "
                    "capping at %d per paper constraint.",
                    configured_w, paper_cap, window,
                )
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

        try:
            self._last_observed = float(observed_tail.iloc[-1])
        except Exception:
            self._last_observed = None

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

        # Phase 8.2: run residual diagnostics and warn if not white noise.
        try:
            from .residual_diagnostics import run_residual_diagnostics  # pylint: disable=import-outside-toplevel
            self._residual_diagnostics = run_residual_diagnostics(residuals)
            if not self._residual_diagnostics.get("white_noise", True):
                logger.warning(
                    "SAMOSSA residuals fail white-noise check "
                    "(lb_p=%.4f, jb_p=%.4f, n=%d) — model may be mis-specified.",
                    self._residual_diagnostics.get("lb_pvalue") or 0.0,
                    self._residual_diagnostics.get("jb_pvalue") or 0.0,
                    self._residual_diagnostics.get("n", 0),
                )
        except Exception as _rd_exc:
            logger.debug("SAMOSSA residual_diagnostics failed: %s", _rd_exc)

        # Phase 7.16: Record learned AR lag in OrderLearner for future warm-start.
        if order_learner is not None and ticker:
            try:
                ar_lag = getattr(self, "_learned_ar_lag", None)
                ar_aic = getattr(self, "_learned_ar_aic", float("nan"))
                if ar_lag is None:
                    _ao = self.config.arima_order or (1, 0, 0)
                    ar_lag = int(_ao[0]) if _ao else 1
                    ar_aic = float("nan")
                order_learner.record_fit(
                    ticker=ticker, model_type="SAMOSSA_ARIMA", regime=regime,
                    order_params={"ar_lag": ar_lag},
                    aic=ar_aic, bic=float("nan"), n_obs=len(residuals),
                )
            except Exception as _oe:
                logger.debug("SAMOSSA OrderLearner.record_fit failed: %s", _oe)

        trend_window = min(max(30, self.config.window_length * 2), min(90, len(observed_tail)))
        self._trend_slope, self._trend_intercept, self._trend_strength = self._estimate_trend(
            observed_tail, trend_window
        )
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

        base_value = float(self._last_observed) if self._last_observed is not None else float(self._reconstructed.iloc[-1])
        base_forecast = np.full(steps, base_value)
        residual_forecast = self._fit_residual_trend(self._residuals, steps)

        combined = base_forecast + residual_forecast[:steps]
        trend_factor = min(1.0, max(0.0, self._trend_strength))
        if trend_factor >= 0.15 and np.isfinite(self._trend_slope):
            trend_steps = np.arange(1, steps + 1, dtype=float)
            combined = combined + (self._trend_slope * trend_steps * trend_factor)

        if self._last_index is not None:
            freq = normalize_freq(
                self._target_freq
                or (self._reconstructed.index.freqstr if hasattr(self._reconstructed.index, "freqstr") else None)
            ) or "D"
            offset = to_offset(freq)
            start = self._last_index + offset
            future_index = pd.date_range(start=start, periods=steps, freq=freq)
        else:
            future_index = pd.RangeIndex(start=0, stop=steps, step=1)

        forecast_series = pd.Series(combined, index=future_index)
        noise_level = float(self._residuals.std()) if self._residuals is not None else 0.0

        # Phase 7.10b: Directional slope signal from reconstructed trend component.
        # Use last trend_slope_bars bars of the reconstructed series to compute slope.
        # Positive slope -> BUY signal; negative -> SELL. Confidence = |slope|/noise.
        directional_signal = 0.0  # -1 / 0 / +1
        directional_confidence = 0.0
        if self._reconstructed is not None and len(self._reconstructed) >= 3:
            slope_bars = getattr(self.config, "trend_slope_bars", 10)
            k = min(slope_bars, len(self._reconstructed))
            recon_tail = self._reconstructed.values[-k:]
            if k >= 2:
                slope_val = float(np.polyfit(np.arange(k), recon_tail, deg=1)[0])
                noise_proxy = noise_level if noise_level > 0 else (float(np.std(recon_tail)) + 1e-8)
                directional_confidence = min(1.0, abs(slope_val) / noise_proxy)
                directional_signal = float(np.sign(slope_val))

        return {
            "forecast": forecast_series,
            "lower_ci": forecast_series - noise_level,
            "upper_ci": forecast_series + noise_level,
            "explained_variance_ratio": self._explained_variance_ratio,
            "window_length_used": self.config.window_length,
            "n_components": self.config.n_components,
            "residual_model_order": getattr(self._residual_model, "k_ar", 0) if self._residual_model is not None else 0,
            "directional_signal": directional_signal,
            "directional_confidence": directional_confidence,
            # Phase 8.2: residual diagnostics (Ljung-Box + Jarque-Bera)
            "residual_diagnostics": self._residual_diagnostics,
        }

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            "window_length_used": self.config.window_length,
            "n_components": self.config.n_components,
            "explained_variance_ratio": self._explained_variance_ratio,
            "trend_strength": self._trend_strength,
            "use_residual_arima": self.config.use_residual_arima,
            "trajectory_matrix_shape": self._trajectory_shape,
            "scale_mean": self._scale_mean,
            "scale_std": self._scale_std,
            "normalized_mean": self._normalized_stats.get("mean"),
            "normalized_std": self._normalized_stats.get("std"),
        }

    def load_fitted(self, snapshot: Any) -> "SAMOSSAForecaster":
        """
        Restore fitted state from a ModelSnapshotStore snapshot (skip-refit path).

        `snapshot` must be a dict with keys:
        {"reconstructed": pd.Series, "residual_model": fitted AutoReg/ARIMA,
         "config": SAMOSSAConfig dict, "scale_mean": float, "scale_std": float, ...}
        """
        if snapshot is None:
            return self
        try:
            if not isinstance(snapshot, dict):
                logger.debug("SAMOSSAForecaster.load_fitted: expected dict snapshot")
                return self
            self._reconstructed = snapshot.get("reconstructed")
            self._residuals = snapshot.get("residuals")
            self._residual_model = snapshot.get("residual_model")
            self._scale_mean = float(snapshot.get("scale_mean", 0.0))
            self._scale_std = float(snapshot.get("scale_std", 1.0))
            self._explained_variance_ratio = float(snapshot.get("evr", 0.0))
            self._trend_slope = float(snapshot.get("trend_slope", 0.0))
            self._trend_intercept = float(snapshot.get("trend_intercept", 0.0))
            self._trend_strength = float(snapshot.get("trend_strength", 0.0))
            self._last_index = snapshot.get("last_index")
            self._target_freq = snapshot.get("target_freq", "D")
            self._last_observed = snapshot.get("last_observed")
            self._normalized_stats = snapshot.get("normalized_stats", {"mean": 0.0, "std": 1.0})
            if snapshot.get("config_window_length") is not None:
                self.config.window_length = int(snapshot.get("config_window_length"))
            if snapshot.get("config_n_components") is not None:
                self.config.n_components = int(snapshot.get("config_n_components"))
            self._learned_ar_lag = snapshot.get("learned_ar_lag")
            self._learned_ar_aic = snapshot.get("learned_ar_aic")
            self._learned_ar_bic = snapshot.get("learned_ar_bic")
            self._fitted = True
            logger.debug("SAMOSSAForecaster.load_fitted: restored from snapshot")
        except Exception as exc:
            logger.warning("SAMOSSAForecaster.load_fitted failed: %s", exc)
        return self
