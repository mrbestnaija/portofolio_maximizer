"""
SARIMAX forecasting utilities extracted from the ETL forecaster module.

Moving the logic here allows the same implementation to be reused by
other services (dashboards, backtesting) without coupling to `etl`.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import guard
    STATSMODELS_AVAILABLE = False

try:
    from etl.warning_recorder import log_warning_records
except Exception:  # pragma: no cover - optional helper for standalone use
    def log_warning_records(records, context):
        return

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

    def _fit_model_instance(
        self,
        model: "SARIMAX",
        *,
        maxiter: Optional[int] = None,
        method: Optional[str] = None,
    ):
        """
        Fit a SARIMAX instance while trapping convergence warnings.
        Returns (result, converged_flag).
        """
        fit_kwargs: Dict[str, Any] = {"disp": False}
        if maxiter is not None:
            fit_kwargs["maxiter"] = maxiter
        if method is not None:
            fit_kwargs["method"] = method

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            result = model.fit(**fit_kwargs)

        log_warning_records(caught, "SARIMAXForecaster.fit_model_instance")
        triggered_warning = any(
            issubclass(warning.category, ConvergenceWarning) for warning in caught
        )
        mle_retvals = getattr(result, "mle_retvals", {}) or {}
        converged = bool(mle_retvals.get("converged", True)) and not triggered_warning
        return result, converged

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

        freq_hint = None
        try:
            freq_hint = data.attrs.get("_pm_freq_hint")
        except AttributeError:  # pragma: no cover - attrs optional
            freq_hint = None

        dates_arg = data.index if freq_hint is not None else None

        series_len = len(data)
        p_cap = min(self.max_p, 2 if series_len < 200 else self.max_p)
        q_cap = min(self.max_q, 2 if series_len < 200 else self.max_q)
        max_order_sum = 5 if series_len >= 400 else 3

        candidate_orders = []
        for p in range(p_cap + 1):
            for q in range(q_cap + 1):
                if p == 0 and q == 0:
                    candidate_orders.append((0, d, 0))
                    continue
                if p + q > max_order_sum:
                    continue
                candidate_orders.append((p, d, q))
        if (0, d, 0) not in candidate_orders:
            candidate_orders.insert(0, (0, d, 0))
        candidate_orders = list(dict.fromkeys(candidate_orders))

        seasonal_period = self.seasonal_periods or self._detect_seasonality(data)
        seasonal_candidates = [(0, 0, 0, 0)]
        if seasonal_period:
            seasonal_p_cap = min(self.max_P, 1 if series_len < 250 else self.max_P)
            seasonal_q_cap = min(self.max_Q, 1 if series_len < 250 else self.max_Q)
            seasonal_d = min(self.max_D, 1)
            for P in range(seasonal_p_cap + 1):
                for Q in range(seasonal_q_cap + 1):
                    if P == Q == 0:
                        seasonal_candidates.append((0, seasonal_d, 0, seasonal_period))
                        continue
                    if P + Q > 2:
                        continue
                    seasonal_candidates.append((P, seasonal_d, Q, seasonal_period))
        seasonal_candidates = list(dict.fromkeys(seasonal_candidates))

        best_aic = np.inf
        best_order: Optional[Tuple[int, int, int]] = None
        best_seasonal: Optional[Tuple[int, int, int, int]] = None

        for test_order in candidate_orders:
            for seasonal_order in seasonal_candidates:
                try:
                    model = SARIMAX(
                        data,
                        order=test_order,
                        seasonal_order=seasonal_order,
                        trend=self.trend,
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility,
                        exog=exogenous,
                        dates=dates_arg,
                        freq=freq_hint,
                    )
                    fitted, converged = self._fit_model_instance(model, maxiter=250)
                except Exception as exc:  # pragma: no cover
                    logger.debug(
                        "SARIMAX order %s seasonal %s failed to fit: %s",
                        test_order,
                        seasonal_order,
                        exc,
                    )
                    continue

                if not converged:
                    logger.debug(
                        "Skipping SARIMAX order %s seasonal %s due to non-convergence",
                        test_order,
                        seasonal_order,
                    )
                    continue

                if not np.isfinite(fitted.aic):
                    continue

                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = test_order
                    best_seasonal = seasonal_order

        if best_order is None or best_seasonal is None:
            logger.info(
                "Unable to identify a stable SARIMAX order via grid search; "
                "deploying fallback specifications per QUANT_TIME_SERIES_STACK."
            )
            fallback_specs = self._build_fallback_candidates(d, seasonal_period)
            for order_candidate, seasonal_candidate in fallback_specs:
                try:
                    model = SARIMAX(
                        data,
                        order=order_candidate,
                        seasonal_order=seasonal_candidate,
                        trend=self.trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        exog=exogenous,
                        dates=dates_arg,
                        freq=freq_hint,
                        simple_differencing=True,
                    )
                    fitted, converged = self._fit_model_instance(model, maxiter=200)
                except Exception as exc:  # pragma: no cover - fallback guard
                    logger.debug(
                        "Fallback SARIMAX order %s seasonal %s failed: %s",
                        order_candidate,
                        seasonal_candidate,
                        exc,
                    )
                    continue
                if converged and np.isfinite(fitted.aic):
                    best_order = order_candidate
                    best_seasonal = seasonal_candidate
                    best_aic = fitted.aic
                    break

            if best_order is None or best_seasonal is None:
                default_order = (0, max(d, 1), 0)
                default_seasonal = (0, 0, 0, 0)
                try:
                    model = SARIMAX(
                        data,
                        order=default_order,
                        seasonal_order=default_seasonal,
                        trend=self.trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        exog=exogenous,
                        dates=dates_arg,
                        freq=freq_hint,
                        simple_differencing=True,
                    )
                    fitted, converged = self._fit_model_instance(model, maxiter=150)
                    if not converged:
                        raise RuntimeError("Default SARIMAX fallback failed to converge")
                    best_order = default_order
                    best_seasonal = default_seasonal
                    best_aic = fitted.aic if np.isfinite(fitted.aic) else np.inf
                except Exception as exc:  # pragma: no cover - final guard
                    logger.error(
                        "Default SARIMAX fallback (0,1,0) failed: %s",
                        exc,
                    )
                    raise RuntimeError("Unable to identify a stable SARIMAX order") from exc

        logger.info(
            "Selected order %s seasonal %s with AIC %.2f",
            best_order,
            best_seasonal,
            best_aic,
        )
        return best_order, best_seasonal

    def _build_fallback_candidates(
        self,
        d: int,
        seasonal_period: int,
    ) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]]:
        base_d = max(d, 1)
        fallback = [
            ((1, base_d, 1), (0, 0, 0, 0)),
            ((1, base_d, 0), (0, 0, 0, 0)),
            ((0, base_d, 1), (0, 0, 0, 0)),
            ((0, base_d, 0), (0, 0, 0, 0)),
        ]
        if seasonal_period:
            fallback.extend(
                [
                    ((1, base_d, 1), (0, 1, 1, seasonal_period)),
                    ((1, base_d, 0), (0, 1, 1, seasonal_period)),
                ]
            )
        deduped: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = []
        for spec in fallback:
            if spec not in deduped:
                deduped.append(spec)
        return deduped

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

        freq_hint = None
        try:
            freq_hint = series.attrs.get("_pm_freq_hint")
        except AttributeError:  # pragma: no cover - attrs optional
            freq_hint = None
        if freq_hint is None:
            try:
                freq_hint = pd.infer_freq(series.index)
            except Exception:  # pragma: no cover - inference best effort
                freq_hint = None
        if freq_hint:
            try:
                series.attrs["_pm_freq_hint"] = freq_hint
            except Exception:
                pass

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

        dates_arg = series.index if freq_hint is not None else None
        primary_model = SARIMAX(
            series,
            order=self.best_order,
            seasonal_order=self.best_seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            exog=exogenous,
            dates=dates_arg,
            freq=freq_hint,
        )
        primary_result, converged = self._fit_model_instance(primary_model)

        if converged:
            self.model = primary_model
            self.fitted_model = primary_result
        else:
            logger.warning(
                "Primary SARIMAX fit did not converge; retrying with relaxed constraints."
            )
            fallback_model = SARIMAX(
                series,
                order=self.best_order,
                seasonal_order=self.best_seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
                exog=exogenous,
                dates=dates_arg,
                freq=freq_hint,
            )
            fallback_result, fallback_converged = self._fit_model_instance(
                fallback_model,
                maxiter=500,
                method="powell",
            )

            if fallback_converged:
                logger.info("Fallback SARIMAX fit converged with relaxed constraints.")
                self.model = fallback_model
                self.fitted_model = fallback_result
            else:
                logger.warning(
                    "Fallback SARIMAX fit still failed to converge; proceeding with primary fit outputs."
                )
                self.model = primary_model
                self.fitted_model = primary_result

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
