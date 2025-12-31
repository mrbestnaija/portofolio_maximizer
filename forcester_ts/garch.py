"""
GARCH volatility model implementation moved from the ETL layer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    arch_model = None  # type: ignore[assignment]
    ARCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class GARCHForecaster:
    """Wrapper around `arch` to produce volatility forecasts."""

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        vol: str = "GARCH",
        dist: str = "normal",
        backend: Optional[str] = None,
    ) -> None:
        self.p = p
        self.q = q
        self.vol = vol
        self.dist = dist
        self.backend = (backend or ("arch" if ARCH_AVAILABLE else "ewma")).lower()
        if self.backend == "arch" and not ARCH_AVAILABLE:
            logger.warning("arch not installed; falling back to EWMA volatility model.")
            self.backend = "ewma"
        if self.backend not in {"arch", "ewma"}:
            raise ValueError(f"Unsupported GARCH backend '{backend}' (expected 'arch' or 'ewma').")

        self.model = None
        self.fitted_model = None
        self.forecast_results: Optional[Dict[str, Any]] = None
        self._scale_factor = 1.0  # Track scaling factor for rescaling forecasts
        self._fallback_state: Optional[Dict[str, Any]] = None

    def fit(self, returns: pd.Series) -> "GARCHForecaster":
        if returns.isna().all():
            raise ValueError("Returns series cannot be all NaNs")

        returns_clean = returns.dropna()
        if returns_clean.empty:
            raise ValueError("Returns series required for GARCH is empty after dropna")
        returns_clean = returns_clean.astype(float)

        backend = getattr(self, "backend", "arch")
        if backend != "arch":
            # Lightweight EWMA fallback when `arch` isn't installed. This keeps
            # the forecasting stack functional in minimal environments and for
            # unit tests while still reacting to recent volatility.
            lam = 0.94
            squared = np.square(returns_clean.to_numpy(dtype=float, copy=False))
            baseline_var = float(np.var(squared)) if squared.size else 0.0
            variance = float(np.var(returns_clean.to_numpy(dtype=float, copy=False), ddof=0))
            if not np.isfinite(variance) or variance <= 0.0:
                variance = max(1e-12, baseline_var)

            ewma_var = variance
            for r2 in squared:
                ewma_var = lam * ewma_var + (1.0 - lam) * float(r2)
            ewma_var = float(max(ewma_var, 1e-12))

            self._fallback_state = {
                "lambda": lam,
                "unconditional_variance": variance,
                "last_variance": ewma_var,
                "mean": float(returns_clean.mean()),
                "n_obs": int(len(returns_clean)),
            }
            # Sentinel so downstream callers treat the model as "fitted".
            self.fitted_model = True
            logger.info(
                "EWMA volatility model fitted successfully (lambda=%.3f, var=%.6g)",
                lam,
                ewma_var,
            )
            return self

        # Scale returns to improve GARCH convergence (recommended by arch library)
        # arch recommends values between 1 and 1000 for better convergence
        
        # Check if scaling is needed (check original scale)
        mean_abs = returns_clean.abs().mean()
        if mean_abs < 1.0 or mean_abs > 1000.0:
            # Scale to bring into recommended range (use 100x for typical returns ~0.001)
            scale_factor = 100.0
            returns_scaled = returns_clean * scale_factor
            self._scale_factor = scale_factor
        else:
            # Already in good range, no scaling needed
            self._scale_factor = 1.0
            returns_scaled = returns_clean

        self.model = arch_model(
            returns_scaled,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist,
            rescale=False,  # We handle scaling manually
        )
        self.fitted_model = self.model.fit(disp="off")
        logger.info(
            "GARCH(%s,%s) model fitted successfully (vol=%s, dist=%s)",
            self.p,
            self.q,
            self.vol,
            self.dist,
        )
        return self

    def forecast(self, steps: int) -> Dict[str, Any]:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        backend = getattr(self, "backend", "arch")
        if backend != "arch":
            state = self._fallback_state or {}
            last_var = float(state.get("last_variance") or 0.0)
            last_var = float(max(last_var, 1e-12))
            idx = pd.Index(range(1, int(steps) + 1), name="horizon")
            variance = pd.Series([last_var] * int(steps), index=idx)
            mean = pd.Series([float(state.get("mean") or 0.0)] * int(steps), index=idx)
            volatility = np.sqrt(variance)
            self.forecast_results = {
                "variance_forecast": variance,
                "mean_forecast": mean,
                "volatility": volatility,
                "steps": int(steps),
                "p": self.p,
                "q": self.q,
                "vol": self.vol,
                "dist": self.dist,
                "aic": None,
                "bic": None,
            }
            return self.forecast_results

        forecast_res = self.fitted_model.forecast(horizon=steps)
        variance = forecast_res.variance.iloc[-1]
        mean = forecast_res.mean.iloc[-1]

        # Rescale back if we scaled during fitting
        scale_factor = getattr(self, "_scale_factor", 1.0)
        if scale_factor != 1.0:
            # Variance scales as scale_factor^2, volatility and mean scale as scale_factor
            variance = variance / (scale_factor**2)
            mean = mean / scale_factor

        volatility = np.sqrt(variance)

        self.forecast_results = {
            "variance_forecast": variance,
            "mean_forecast": mean,
            "volatility": volatility,
            "steps": steps,
            "p": self.p,
            "q": self.q,
            "vol": self.vol,
            "dist": self.dist,
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic),
        }
        return self.forecast_results

    def get_model_summary(self) -> Dict[str, Any]:
        if self.fitted_model is None:
            return {}
        backend = getattr(self, "backend", "arch")
        if backend != "arch":
            state = self._fallback_state or {}
            return {
                "params": {"lambda": state.get("lambda"), "p": self.p, "q": self.q},
                "aic": None,
                "bic": None,
                "log_likelihood": None,
                "backend": backend,
            }
        return {
            "params": self.fitted_model.params.to_dict(),
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic),
            "log_likelihood": float(self.fitted_model.loglikelihood),
            "backend": backend,
        }
