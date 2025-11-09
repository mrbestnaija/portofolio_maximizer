"""
SARIMAX forecasting utilities extracted from the ETL forecaster module.

Moving the logic here allows the same implementation to be reused by
other services (dashboards, backtesting) without coupling to `etl`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera

    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import guard
    STATSMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SARIMAXForecaster:
    """
    Seasonal ARIMA forecaster with automatic order selection and diagnostics.
    """

    def __init__(
        self,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
        seasonal_periods: Optional[int] = None,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        trend: str = "c",
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        auto_select: bool = True,
        manual_order: Optional[Tuple[int, int, int]] = None,
        manual_seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels required for SARIMAX forecasting")

        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal_periods = seasonal_periods
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.auto_select = auto_select
        self.manual_order = manual_order
        self.manual_seasonal_order = manual_seasonal_order

        self.model = None
        self.fitted_model = None
        self.best_order: Optional[Tuple[int, int, int]] = None
        self.best_seasonal_order: Optional[Tuple[int, int, int, int]] = None
        self.forecast_results: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_seasonality(data: pd.Series) -> int:
        if len(data) < 50:
            return 0

        for period in (12, 24, 52, 7):
            if len(data) >= 2 * period:
                autocorr = data.autocorr(lag=period)
                if abs(autocorr) > 0.3:
                    logger.info("Detected seasonality with period %s", period)
                    return period
        return 0

    @staticmethod
    def _test_stationarity(data: pd.Series) -> Tuple[bool, int]:
        try:
            adf_stat = adfuller(data.dropna())
            stationary = adf_stat[1] < 0.05
            if not stationary:
                diff_data = data.diff().dropna()
                if len(diff_data) > 10:
                    diff_stat = adfuller(diff_data)
                    if diff_stat[1] < 0.05:
                        return False, 1
            return stationary, 0
        except Exception as exc:  # pragma: no cover
            logger.warning("ADF stationarity test failed: %s", exc)
            return False, 1

    def _select_best_order(
        self,
        data: pd.Series,
        exogenous: Optional[pd.DataFrame] = None,
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        logger.info("Selecting optimal SARIMAX order...")

        stationary, recommend_d = self._test_stationarity(data)
        d = recommend_d if self.enforce_stationarity else 0

        seasonal_period = self.seasonal_periods or self._detect_seasonality(data)
        if seasonal_period == 0:
            seasonal_order = (0, 0, 0, 0)
        else:
            seasonal_order = (self.max_P, self.max_D, self.max_Q, seasonal_period)

        best_aic = np.inf
        best_order = None
        best_seasonal = None

        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                try:
                    test_order = (p, d, q)
                    model = SARIMAX(
                        data,
                        order=test_order,
                        seasonal_order=seasonal_order,
                        trend=self.trend,
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility,
                        exog=exogenous,
                    )
                    fitted = model.fit(disp=False)
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = test_order
                        best_seasonal = seasonal_order
                except Exception as exc:  # pragma: no cover
                    logger.debug("SARIMAX order %s failed: %s", (p, d, q), exc)
                    continue

        if best_order is None:
            raise RuntimeError("Unable to identify a stable SARIMAX order")

        logger.info(
            "Selected order %s seasonal %s with AIC %.2f",
            best_order,
            best_seasonal,
            best_aic,
        )
        return best_order, best_seasonal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        series: pd.Series,
        exogenous: Optional[pd.DataFrame] = None,
    ) -> "SARIMAXForecaster":
        if series.isna().all():
            raise ValueError("Series contains only NaNs")

        if not self.auto_select:
            if self.manual_order is None:
                raise ValueError("manual_order required when auto_select=False")
            self.best_order = self.manual_order
            self.best_seasonal_order = self.manual_seasonal_order or (0, 0, 0, 0)
        else:
            self.best_order, self.best_seasonal_order = self._select_best_order(
                series,
                exogenous,
            )

        self.model = SARIMAX(
            series,
            order=self.best_order,
            seasonal_order=self.best_seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            exog=exogenous,
        )
        self.fitted_model = self.model.fit(disp=False)
        logger.info(
            "SARIMAX model fitted successfully (order=%s, seasonal=%s)",
            self.best_order,
            self.best_seasonal_order,
        )
        return self

    def forecast(
        self,
        steps: int,
        exogenous: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        forecast_res = self.fitted_model.get_forecast(steps=steps, exog=exogenous)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=alpha)

        residuals = self.fitted_model.resid

        diagnostics = {
            "ljung_box_pvalue": None,
            "jarque_bera_pvalue": None,
            "residual_mean": float(residuals.mean()),
            "residual_std": float(residuals.std()),
        }

        if len(residuals) > 10:
            try:
                lb_df = acorr_ljungbox(
                    residuals, lags=min(10, len(residuals) // 4), return_df=True
                )
                diagnostics["ljung_box_pvalue"] = float(lb_df["lb_pvalue"].iloc[-1])
            except Exception:  # pragma: no cover
                pass

        if len(residuals) > 5:
            try:
                _, jb_pvalue = jarque_bera(residuals)
                diagnostics["jarque_bera_pvalue"] = float(jb_pvalue)
            except Exception:  # pragma: no cover
                pass

        forecast_ci = conf_int.rename(
            columns={
                conf_int.columns[0]: "lower_ci",
                conf_int.columns[1]: "upper_ci",
            }
        )

        z_score = stats.norm.ppf(1 - alpha / 2) if 0 < alpha < 1 else stats.norm.ppf(
            0.975
        )

        self.forecast_results = {
            "forecast": forecast_mean,
            "lower_ci": forecast_ci["lower_ci"],
            "upper_ci": forecast_ci["upper_ci"],
            "alpha": alpha,
            "z_score": float(z_score),
            "steps": steps,
            "model_order": self.best_order,
            "seasonal_order": self.best_seasonal_order,
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic),
            "diagnostics": diagnostics,
        }
        return self.forecast_results

    def get_model_summary(self) -> Dict[str, Any]:
        if self.fitted_model is None:
            return {}
        return {
            "order": self.best_order,
            "seasonal_order": self.best_seasonal_order,
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic),
            "log_likelihood": float(self.fitted_model.llf),
            "n_observations": int(self.fitted_model.nobs),
        }
