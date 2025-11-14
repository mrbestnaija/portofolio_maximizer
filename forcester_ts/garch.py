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
    ) -> None:
        if not ARCH_AVAILABLE:
            raise ImportError("arch package required for GARCH modeling")

        self.p = p
        self.q = q
        self.vol = vol
        self.dist = dist
        self.model = None
        self.fitted_model = None
        self.forecast_results: Optional[Dict[str, Any]] = None
        self._scale_factor = 1.0  # Track scaling factor for rescaling forecasts

    def fit(self, returns: pd.Series) -> "GARCHForecaster":
        if returns.isna().all():
            raise ValueError("Returns series cannot be all NaNs")

        # Scale returns to improve GARCH convergence (recommended by arch library)
        # arch recommends values between 1 and 1000 for better convergence
        returns_clean = returns.dropna()
        
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

        forecast_res = self.fitted_model.forecast(horizon=steps)
        variance = forecast_res.variance.iloc[-1]
        mean = forecast_res.mean.iloc[-1]
        
        # Rescale back if we scaled during fitting
        scale_factor = getattr(self, '_scale_factor', 1.0)
        if scale_factor != 1.0:
            # Variance scales as scale_factor^2, volatility and mean scale as scale_factor
            variance = variance / (scale_factor ** 2)
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
        return {
            "params": self.fitted_model.params.to_dict(),
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic),
            "log_likelihood": float(self.fitted_model.loglikelihood),
        }
