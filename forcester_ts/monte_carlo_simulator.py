"""
monte_carlo_simulator.py
------------------------
Lightweight Monte Carlo price-path simulation for forecast uncertainty.

This is intentionally additive: it consumes an existing point forecast plus a
dispersion estimate and returns empirical path summaries. It does not alter
model fitting, order-learning, or default-model selection.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """
    Simulate price paths from a point forecast and per-step volatility.

    Guardrails:
    - Clamp path count into a bounded range so callers cannot request
      trivially small or unboundedly large simulations.
    - Return explicit SKIP payloads when required inputs are absent.
    """

    MIN_PATHS = 250
    MAX_PATHS = 10000

    @classmethod
    def _normalize_path_count(cls, n_paths: int) -> int:
        try:
            parsed = int(n_paths)
        except Exception:
            parsed = cls.MIN_PATHS
        return max(cls.MIN_PATHS, min(cls.MAX_PATHS, parsed))

    @staticmethod
    def _normalize_alpha(alpha: float) -> float:
        try:
            alpha_val = float(alpha)
        except Exception:
            alpha_val = 0.05
        if not (0.0 < alpha_val < 1.0):
            return 0.05
        return alpha_val

    @staticmethod
    def _coerce_series(
        value: "pd.Series | np.ndarray | list[float] | tuple[float, ...] | None",
        *,
        index: pd.Index,
    ) -> pd.Series | None:
        if value is None:
            return None
        if isinstance(value, pd.Series):
            return value.reindex(index)
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1 or len(arr) == 0:
            return None
        n = min(len(arr), len(index))
        return pd.Series(arr[:n], index=index[:n])

    @staticmethod
    def _sigma_from_confidence_band(
        lower_ci: pd.Series | None,
        upper_ci: pd.Series | None,
        *,
        alpha: float,
    ) -> pd.Series | None:
        if lower_ci is None or upper_ci is None:
            return None
        if len(lower_ci) == 0 or len(upper_ci) == 0:
            return None
        alpha = MonteCarloSimulator._normalize_alpha(alpha)
        z_score = NormalDist().inv_cdf(1.0 - (alpha / 2.0))
        if not np.isfinite(z_score) or z_score <= 0.0:
            z_score = NormalDist().inv_cdf(0.975)
        half_width = (upper_ci.astype(float) - lower_ci.astype(float)) / 2.0
        sigma = (half_width / max(float(z_score), 1e-8)).abs()
        sigma = sigma.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return sigma

    def simulate_price_distribution(
        self,
        *,
        base_forecast: pd.Series,
        last_price: float,
        n_paths: int = 1000,
        seed: int | None = None,
        volatility: "pd.Series | np.ndarray | list[float] | tuple[float, ...] | None" = None,
        lower_ci: pd.Series | None = None,
        upper_ci: pd.Series | None = None,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        if not isinstance(base_forecast, pd.Series) or base_forecast.empty:
            return {"status": "SKIP", "reason": "base_forecast_missing"}
        try:
            last_price_val = float(last_price)
        except Exception:
            last_price_val = float("nan")
        if not np.isfinite(last_price_val) or last_price_val <= 0.0:
            return {"status": "SKIP", "reason": "last_price_missing"}

        forecast = base_forecast.astype(float)
        forecast = forecast.replace([np.inf, -np.inf], np.nan).dropna()
        if forecast.empty:
            return {"status": "SKIP", "reason": "base_forecast_missing"}

        sigma = self._coerce_series(volatility, index=forecast.index)
        sigma_source = "volatility_forecast"
        if sigma is None:
            lower = lower_ci.reindex(forecast.index) if isinstance(lower_ci, pd.Series) else None
            upper = upper_ci.reindex(forecast.index) if isinstance(upper_ci, pd.Series) else None
            sigma = self._sigma_from_confidence_band(lower, upper, alpha=alpha)
            sigma_source = "confidence_interval"
        if sigma is None:
            return {"status": "SKIP", "reason": "dispersion_inputs_missing"}

        aligned = pd.DataFrame({"forecast": forecast, "sigma": sigma}).dropna()
        if aligned.empty:
            return {"status": "SKIP", "reason": "dispersion_inputs_missing"}

        forecast = aligned["forecast"]
        sigma = aligned["sigma"].abs().clip(lower=1e-8, upper=3.0)
        idx = forecast.index

        prev_levels = np.empty(len(forecast), dtype=float)
        prev_levels[0] = last_price_val
        if len(forecast) > 1:
            prev_levels[1:] = forecast.iloc[:-1].to_numpy(dtype=float)
        prev_levels = np.maximum(np.abs(prev_levels), 1e-8)

        mean_returns = forecast.to_numpy(dtype=float) / prev_levels - 1.0
        mean_returns = np.clip(mean_returns, -0.95, 5.0)

        paths_used = self._normalize_path_count(n_paths)
        alpha_value = self._normalize_alpha(alpha)
        lower_quantile = alpha_value / 2.0
        upper_quantile = 1.0 - lower_quantile
        rng = np.random.default_rng(seed)
        shocks = rng.normal(
            loc=mean_returns,
            scale=sigma.to_numpy(dtype=float),
            size=(paths_used, len(forecast)),
        )
        shocks = np.clip(shocks, -0.95, 5.0)

        path_matrix = np.empty_like(shocks, dtype=float)
        running = np.full(paths_used, last_price_val, dtype=float)
        for step_idx in range(len(forecast)):
            running = running * (1.0 + shocks[:, step_idx])
            running = np.maximum(running, 1e-8)
            path_matrix[:, step_idx] = running

        expected_path = pd.Series(path_matrix.mean(axis=0), index=idx, name="mc_expected")
        median_path = pd.Series(np.quantile(path_matrix, 0.50, axis=0), index=idx, name="mc_median")
        lower_band = pd.Series(
            np.quantile(path_matrix, lower_quantile, axis=0),
            index=idx,
            name="mc_lower_ci",
        )
        upper_band = pd.Series(
            np.quantile(path_matrix, upper_quantile, axis=0),
            index=idx,
            name="mc_upper_ci",
        )
        stddev = pd.Series(path_matrix.std(axis=0), index=idx, name="mc_stddev")
        prob_up = pd.Series((path_matrix > last_price_val).mean(axis=0), index=idx, name="mc_prob_up")

        return {
            "status": "OK",
            "engine": "gaussian_path",
            "alpha": alpha_value,
            "confidence_level": 1.0 - alpha_value,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "paths_requested": int(n_paths) if isinstance(n_paths, (int, float)) else n_paths,
            "paths_used": paths_used,
            "seed": seed,
            "volatility_source": sigma_source,
            "expected_path": expected_path,
            "median_path": median_path,
            "lower_ci": lower_band,
            "upper_ci": upper_band,
            "stddev": stddev,
            "prob_up": prob_up,
        }
