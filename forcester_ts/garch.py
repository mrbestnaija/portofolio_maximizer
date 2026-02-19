"""
GARCH volatility model implementation moved from the ETL layer.

Phase 7.10b improvements:
  - AR(1) conditional mean model for directional signal (was 'Zero' / volatility only)
  - skewt distribution for fat tails + negative skew (was 'normal')
  - ADF stationarity test; auto-difference if unit root detected
  - GJR-GARCH asymmetric vol fallback before EWMA when persistence >= 0.97
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

try:
    from statsmodels.tsa.stattools import adfuller as _adfuller  # type: ignore[import]

    STATSMODELS_ADF_AVAILABLE = True
except Exception:  # pragma: no cover
    _adfuller = None  # type: ignore[assignment]
    STATSMODELS_ADF_AVAILABLE = False

logger = logging.getLogger(__name__)


class GARCHForecaster:
    """Wrapper around `arch` to produce volatility forecasts."""

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        vol: str = "GARCH",
        dist: str = "skewt",
        mean: str = "AR",
        backend: Optional[str] = None,
        *,
        auto_select: bool = True,
        max_p: int = 3,
        max_q: int = 3,
        enforce_stationarity: bool = True,
        igarch_fallback: str = "gjr",
    ) -> None:
        self.p = p
        self.q = q
        self.vol = vol
        self.dist = dist
        self.mean = mean  # Phase 7.10b: AR(1) mean model for directional signal
        self.enforce_stationarity = bool(enforce_stationarity)  # ADF pre-check
        self.igarch_fallback = str(igarch_fallback).lower()  # 'gjr' or 'ewma'
        self.backend = (backend or ("arch" if ARCH_AVAILABLE else "ewma")).lower()
        self.auto_select = bool(auto_select)
        self.max_p = int(max_p)
        self.max_q = int(max_q)
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
        self._differenced: bool = False  # Track if series was differenced for stationarity

    def fit(self, returns: pd.Series) -> "GARCHForecaster":
        if not getattr(self, "auto_select", True):
            raise ValueError("Manual GARCH orders are unsupported; set auto_select=True and use max_p/max_q caps.")

        if returns.isna().all():
            raise ValueError("Returns series cannot be all NaNs")

        returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if returns_clean.empty:
            raise ValueError("Returns series required for GARCH is empty after dropna")
        returns_clean = returns_clean.astype(float)

        # Guard against numerical overflow by clipping extreme tails.
        abs_returns = returns_clean.abs()
        abs_returns = abs_returns[np.isfinite(abs_returns)]
        if abs_returns.empty:
            raise ValueError("Returns series required for GARCH is empty after sanitization")
        try:
            p995 = float(np.nanpercentile(abs_returns, 99.5))
        except Exception:
            p995 = float(abs_returns.max())
        med = float(np.nanmedian(abs_returns)) if len(abs_returns) else 0.0
        cap = max(p995, med * 10.0) if med > 0 else p995
        if np.isfinite(cap) and cap > 0:
            returns_clean = returns_clean.clip(-cap, cap)

        backend = getattr(self, "backend", "arch")
        if backend != "arch":
            # Lightweight EWMA fallback when `arch` isn't installed. This keeps
            # the forecasting stack functional in minimal environments and for
            # unit tests while still reacting to recent volatility.
            return self._fit_ewma(returns_clean)

        # Phase 7.10b: ADF stationarity check. Financial returns are generally
        # stationary, but some series (especially level prices fed incorrectly)
        # may have a unit root. Auto-difference once if detected.
        if getattr(self, "enforce_stationarity", True) and STATSMODELS_ADF_AVAILABLE:
            try:
                adf_result = _adfuller(returns_clean.values, autolag="AIC")
                adf_pvalue = float(adf_result[1])
                if adf_pvalue > 0.05:
                    logger.warning(
                        "ADF test: unit root detected (p=%.4f); differencing returns once.",
                        adf_pvalue,
                    )
                    returns_clean = returns_clean.diff().dropna()
                    self._differenced = True
                    if returns_clean.empty:
                        raise ValueError("Returns series empty after ADF-motivated differencing")
            except Exception as adf_exc:  # pragma: no cover
                logger.debug("ADF stationarity check failed (%s); skipping.", adf_exc)

        # Scale returns to improve GARCH convergence (recommended by arch library).
        mean_abs = returns_clean.abs().mean()
        if mean_abs < 1.0 or mean_abs > 1000.0:
            scale_factor = 100.0
            returns_scaled = returns_clean * scale_factor
            self._scale_factor = scale_factor
        else:
            self._scale_factor = 1.0
            returns_scaled = returns_clean

        best_model = None
        best_fit = None
        best_aic = np.inf
        best_order = (self.p, self.q)

        # Phase 7.10b: Use configurable mean model (AR for directional signal).
        # 'AR' adds an AR(1) conditional mean; 'Zero' is the old default.
        mean_model = str(getattr(self, "mean", "AR"))
        # skewt may fail if arch version doesn't support it; fall back to 't' then 'normal'.
        dist_candidates = [self.dist]
        if self.dist == "skewt":
            dist_candidates += ["t", "normal"]
        elif self.dist == "t":
            dist_candidates += ["normal"]

        max_p = max(1, int(getattr(self, "max_p", self.p)))
        max_q = max(1, int(getattr(self, "max_q", self.q)))
        for p_candidate in range(1, max_p + 1):
            for q_candidate in range(1, max_q + 1):
                for dist_try in dist_candidates:
                    try:
                        model = arch_model(
                            returns_scaled,
                            mean=mean_model,
                            vol=self.vol,
                            p=p_candidate,
                            q=q_candidate,
                            dist=dist_try,
                            rescale=False,  # We handle scaling manually
                        )
                        fitted = model.fit(disp="off")
                        aic = float(getattr(fitted, "aic", np.inf))
                        if np.isfinite(aic) and aic < best_aic:
                            best_aic = aic
                            best_model = model
                            best_fit = fitted
                            best_order = (p_candidate, q_candidate)
                        break  # Use first dist that fits
                    except Exception:  # pragma: no cover - best-effort search
                        continue

        self.p, self.q = best_order
        self.model = best_model
        self.fitted_model = best_fit
        if self.fitted_model is None:
            logger.warning(
                "GARCH auto_select failed to fit any model; falling back to EWMA volatility."
            )
            self.backend = "ewma"
            return self._fit_ewma(returns_clean)
        # Phase 7.10b: IGARCH / unit-root guard -- if alpha+beta >= 0.97 the
        # variance process is near-non-stationary and forecasts diverge.
        # Try GJR-GARCH (asymmetric volatility; better for equity fat tails) first;
        # only fall back to EWMA if GJR also degenerates.
        persistence = self._garch_persistence()
        if persistence is not None and persistence >= 0.97:
            igarch_fallback = str(getattr(self, "igarch_fallback", "gjr")).lower()
            if igarch_fallback == "gjr" and ARCH_AVAILABLE:
                logger.warning(
                    "GARCH(%s,%s) persistence %.4f >= 0.97 (near-IGARCH); "
                    "attempting GJR-GARCH asymmetric model.",
                    self.p, self.q, persistence,
                )
                try:
                    gjr_model = arch_model(
                        returns_scaled,
                        mean=mean_model,
                        vol="GARCH",
                        p=self.p,
                        o=1,  # GJR leverage term
                        q=self.q,
                        dist=dist_candidates[0],
                        rescale=False,
                    )
                    gjr_fit = gjr_model.fit(disp="off")
                    gjr_persistence = sum(
                        float(v) for k, v in gjr_fit.params.items()
                        if k.startswith(("alpha[", "beta[")) and np.isfinite(float(v))
                    )
                    if gjr_persistence < 0.97:
                        self.model = gjr_model
                        self.fitted_model = gjr_fit
                        self.vol = "GJR-GARCH"
                        logger.info(
                            "GJR-GARCH fitted (persistence=%.4f); using asymmetric model.",
                            gjr_persistence,
                        )
                        return self
                except Exception as gjr_exc:
                    logger.debug("GJR-GARCH attempt failed (%s); falling back to EWMA.", gjr_exc)
            logger.warning(
                "GARCH persistence %.4f >= 0.97; GJR unavailable or also degenerate; "
                "falling back to EWMA to avoid divergent variance forecasts.",
                persistence,
            )
            self.backend = "ewma"
            return self._fit_ewma(returns_clean)

        logger.info(
            "GARCH(%s,%s) model fitted successfully (vol=%s, dist=%s, persistence=%.4f)",
            self.p,
            self.q,
            self.vol,
            self.dist,
            persistence if persistence is not None else 0.0,
        )
        return self

    def _garch_persistence(self) -> Optional[float]:
        """Sum of alpha + beta coefficients (persistence parameter).

        Returns None if the model is not fitted or params are unavailable.
        For stationary GARCH, persistence < 1.0; IGARCH has persistence >= 1.0.
        """
        if self.fitted_model is None or self.fitted_model is True:
            return None
        try:
            params = self.fitted_model.params
            alpha_sum = sum(
                float(v) for k, v in params.items()
                if k.startswith("alpha[") and np.isfinite(v)
            )
            beta_sum = sum(
                float(v) for k, v in params.items()
                if k.startswith("beta[") and np.isfinite(v)
            )
            return alpha_sum + beta_sum
        except Exception:
            return None

    def _fit_ewma(self, returns_clean: pd.Series) -> "GARCHForecaster":
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
                "igarch_fallback": True,
            }
        return {
            "params": self.fitted_model.params.to_dict(),
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic),
            "log_likelihood": float(self.fitted_model.loglikelihood),
            "persistence": self._garch_persistence(),
            "backend": backend,
            "mean_model": getattr(self, "mean", "AR"),
            "dist": self.dist,
            "differenced": bool(getattr(self, "_differenced", False)),
        }
