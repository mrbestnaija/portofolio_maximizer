"""
var_backtest.py
---------------
Value-at-Risk backtesting for GARCH volatility and SARIMAX confidence-interval
forecasts against realized returns.

Tests provided:
  - Kupiec POF (proportion of failures) — checks if violation rate matches alpha
  - Christoffersen (conditional coverage) — checks independence of violations
  - Pinball loss (quantile scoring) — measures calibration across tau levels

Usage:
    from forcester_ts.var_backtest import VaRBacktester
    bt = VaRBacktester()
    var_series = bt.compute_var(garch_vol, confidence_level=0.99)
    kupiec = bt.kupiec_test(actual_returns, var_series)
    pinball = bt.pinball_loss(actual_returns, {0.01: lower_q, 0.99: upper_q})
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TAUS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


class VaRBacktester:
    """
    Tail-risk calibration tests for forecasted VaR / quantile intervals.
    All methods accept pandas Series or numpy arrays.
    """

    def compute_var(
        self,
        volatility_forecast: "np.ndarray | pd.Series",
        confidence_level: float = 0.99,
        dist: str = "normal",
        nu: float | None = None,
        lambda_: float | None = None,
    ) -> "np.ndarray":
        """
        Compute VaR from a volatility forecast series.

        VaR_t = -sigma_t * q_alpha(dist)

        Args:
            volatility_forecast: GARCH sigma (std dev) forecasts
            confidence_level: e.g. 0.99 for 1% VaR
            dist: "normal" | "t" | "skewt"
            nu: degrees of freedom (t / skewt)
            lambda_: skewness (skewt only)

        Returns:
            VaR series (positive values; loss exceeds VaR when return < -VaR)
        """
        from scipy import stats

        sigma = np.asarray(volatility_forecast, dtype=float)
        alpha = 1.0 - confidence_level

        if dist == "normal":
            q = stats.norm.ppf(alpha)
        elif dist == "t" and nu is not None:
            q = stats.t.ppf(alpha, df=nu)
        elif dist == "skewt" and nu is not None:
            # Approximate skew-t quantile via t with same df
            q = stats.t.ppf(alpha, df=nu)
        else:
            q = stats.norm.ppf(alpha)

        return -sigma * q  # positive VaR

    def kupiec_test(
        self,
        actual_returns: "np.ndarray | pd.Series",
        var_series: "np.ndarray | pd.Series",
        confidence_level: float = 0.99,
    ) -> dict:
        """
        Kupiec Proportion of Failures (POF) test.

        H0: actual violation rate == alpha (= 1 - confidence_level)
        Test stat LR_POF ~ chi2(1) under H0.

        Returns dict: {violations, T, violation_rate, expected_rate,
                       lr_stat, p_value, reject_null}
        """
        from scipy import stats

        y = np.asarray(actual_returns, dtype=float)
        v = np.asarray(var_series, dtype=float)
        T = len(y)
        alpha = 1.0 - confidence_level

        # Violation = actual loss exceeds VaR
        violations_mask = y < -v
        n = int(violations_mask.sum())
        pi_hat = n / T if T > 0 else 0.0

        # LR_POF (Kupiec 1995)
        eps = 1e-12
        pi_hat_c = max(eps, min(1 - eps, pi_hat))
        alpha_c = max(eps, min(1 - eps, alpha))
        try:
            lr_pof = -2.0 * (
                (T - n) * math.log(1.0 - alpha_c) + n * math.log(alpha_c)
                - (T - n) * math.log(1.0 - pi_hat_c) - n * math.log(pi_hat_c)
            )
        except (ValueError, ZeroDivisionError):
            lr_pof = float("nan")

        p_value = float(stats.chi2.sf(lr_pof, df=1)) if lr_pof == lr_pof else float("nan")

        return {
            "violations": n,
            "T": T,
            "violation_rate": pi_hat,
            "expected_rate": alpha,
            "lr_stat": float(lr_pof),
            "p_value": p_value,
            "reject_null": (p_value < 0.05) if p_value == p_value else None,
        }

    def christoffersen_test(
        self,
        actual_returns: "np.ndarray | pd.Series",
        var_series: "np.ndarray | pd.Series",
        confidence_level: float = 0.99,
    ) -> dict:
        """
        Christoffersen (1998) conditional coverage test.

        Tests BOTH that the violation rate is correct (POF) AND that
        violations are independently distributed (no clustering).

        Returns dict: {n00, n01, n10, n11, pi01, pi11,
                       lr_ind, lr_cc, p_value_ind, p_value_cc,
                       reject_independence, reject_coverage}
        """
        from scipy import stats

        y = np.asarray(actual_returns, dtype=float)
        v = np.asarray(var_series, dtype=float)
        hits = (y < -v).astype(int)

        if len(hits) < 2:
            return {"error": "insufficient data for Christoffersen test"}

        # Transition counts
        n00 = int(((hits[:-1] == 0) & (hits[1:] == 0)).sum())
        n01 = int(((hits[:-1] == 0) & (hits[1:] == 1)).sum())
        n10 = int(((hits[:-1] == 1) & (hits[1:] == 0)).sum())
        n11 = int(((hits[:-1] == 1) & (hits[1:] == 1)).sum())

        eps = 1e-12
        pi01 = n01 / max(n00 + n01, 1)
        pi11 = n11 / max(n10 + n11, 1)
        pi_hat = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

        # LR independence
        try:
            lr_ind = -2.0 * (
                (n00 + n10) * math.log(max(1 - pi_hat, eps))
                + (n01 + n11) * math.log(max(pi_hat, eps))
                - n00 * math.log(max(1 - pi01, eps))
                - n01 * math.log(max(pi01, eps))
                - n10 * math.log(max(1 - pi11, eps))
                - n11 * math.log(max(pi11, eps))
            )
        except (ValueError, ZeroDivisionError):
            lr_ind = float("nan")

        # Combine with POF for conditional coverage
        kupiec = self.kupiec_test(actual_returns, var_series, confidence_level)
        lr_pof = kupiec["lr_stat"]
        lr_cc = float(lr_ind + lr_pof) if (lr_ind == lr_ind and lr_pof == lr_pof) else float("nan")

        p_ind = float(stats.chi2.sf(lr_ind, df=1)) if lr_ind == lr_ind else float("nan")
        p_cc = float(stats.chi2.sf(lr_cc, df=2)) if lr_cc == lr_cc else float("nan")

        return {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "pi01": float(pi01), "pi11": float(pi11),
            "lr_ind": float(lr_ind),
            "lr_cc": float(lr_cc),
            "p_value_ind": p_ind,
            "p_value_cc": p_cc,
            "reject_independence": (p_ind < 0.05) if p_ind == p_ind else None,
            "reject_coverage": (p_cc < 0.05) if p_cc == p_cc else None,
        }

    def pinball_loss(
        self,
        actual: "np.ndarray | pd.Series",
        quantile_forecasts: dict[float, "np.ndarray | pd.Series"],
    ) -> dict[float, float]:
        """
        Compute mean pinball (quantile) loss for each tau level.

        L(y, q, tau) = (y - q) * (tau - 1[y < q])

        Args:
            actual: realized values (returns, prices, etc.)
            quantile_forecasts: dict mapping tau → forecast series

        Returns:
            dict: {tau: mean_pinball_loss, ..., "mean": weighted_mean_across_taus}
        """
        y = np.asarray(actual, dtype=float)
        results: dict[float, float] = {}

        for tau, q_series in quantile_forecasts.items():
            q = np.asarray(q_series, dtype=float)
            n = min(len(y), len(q))
            if n == 0:
                results[float(tau)] = float("nan")
                continue
            diff = y[:n] - q[:n]
            loss = diff * (tau - (diff < 0).astype(float))
            results[float(tau)] = float(np.mean(loss))

        if results:
            finite = [v for v in results.values() if v == v]
            results["mean"] = float(np.mean(finite)) if finite else float("nan")

        return results

    @staticmethod
    def _has_forecast_input(value: "np.ndarray | pd.Series | None") -> bool:
        if value is None:
            return False
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return False
        if arr.size == 0:
            return False
        return bool(np.isfinite(arr).any())

    @staticmethod
    def _resolve_quantile_forecast(
        quantile_forecasts: "dict[float, np.ndarray | pd.Series] | None",
        tau: float,
    ) -> "np.ndarray | pd.Series | None":
        if not isinstance(quantile_forecasts, dict):
            return None
        try:
            tau_value = float(tau)
        except Exception:
            return None
        direct = quantile_forecasts.get(tau_value)
        if direct is not None and VaRBacktester._has_forecast_input(direct):
            return direct
        for raw_tau, series in quantile_forecasts.items():
            try:
                candidate = float(raw_tau)
            except Exception:
                continue
            if math.isclose(candidate, tau_value, rel_tol=0.0, abs_tol=1e-9) and VaRBacktester._has_forecast_input(series):
                return series
        return None

    def _compute_parametric_quantile(
        self,
        volatility_forecast: "np.ndarray | pd.Series",
        *,
        tau: float,
        dist: str,
        nu: float | None,
    ) -> "np.ndarray":
        cl_for_tau = 1.0 - tau if tau < 0.5 else tau
        var_t = self.compute_var(
            volatility_forecast,
            confidence_level=cl_for_tau,
            dist=dist,
            nu=nu,
        )
        return -var_t if tau < 0.5 else var_t

    def full_report(
        self,
        actual_returns: "np.ndarray | pd.Series",
        garch_vol: "np.ndarray | pd.Series | None",
        confidence_levels: tuple[float, ...] = (0.95, 0.99),
        taus: tuple[float, ...] = _DEFAULT_TAUS,
        dist: str = "normal",
        nu: float | None = None,
        quantile_forecasts: "dict[float, np.ndarray | pd.Series] | None" = None,
        coverage_quantile_forecasts: "dict[float, np.ndarray | pd.Series] | None" = None,
    ) -> dict:
        """
        Run all tests and return a unified dict for audit logging.

        When `quantile_forecasts` is supplied, provided taus are preferred for
        pinball loss.

        VaR coverage can be driven by a separate `coverage_quantile_forecasts`
        dict. If that argument is omitted, `quantile_forecasts` is reused for
        backward compatibility. Passing an explicit empty mapping disables
        coverage-side empirical quantile preference and preserves parametric
        VaR as the first live source.
        """
        report: dict = {"confidence_levels": {}, "pinball": {}, "pinball_sources": {}}
        coverage_quantiles = (
            quantile_forecasts
            if coverage_quantile_forecasts is None
            else coverage_quantile_forecasts
        )

        for cl in confidence_levels:
            cl_value = float(cl)
            lower_tail_tau = 1.0 - cl_value
            lower_q = self._resolve_quantile_forecast(coverage_quantiles, lower_tail_tau)
            source = "missing"
            if lower_q is not None:
                var_series = -np.asarray(lower_q, dtype=float)
                source = "empirical_quantile"
            elif self._has_forecast_input(garch_vol):
                var_series = self.compute_var(garch_vol, confidence_level=cl_value, dist=dist, nu=nu)
                source = "parametric_var"
            else:
                error = {
                    "error": (
                        f"missing lower-tail quantile forecast for tau={lower_tail_tau:.6f} "
                        "and volatility forecast unavailable"
                    )
                }
                report["confidence_levels"][cl_value] = {
                    "source": source,
                    "tau": lower_tail_tau,
                    "kupiec": dict(error),
                    "christoffersen": dict(error),
                }
                continue
            kupiec = self.kupiec_test(actual_returns, var_series, cl)
            cc = self.christoffersen_test(actual_returns, var_series, cl)
            report["confidence_levels"][cl_value] = {
                "source": source,
                "tau": lower_tail_tau,
                "kupiec": kupiec,
                "christoffersen": cc,
            }

        # Pinball: prefer supplied empirical quantiles, with explicit parametric fallback.
        q_forecasts = {}
        for tau in taus:
            tau_value = float(tau)
            quantile = self._resolve_quantile_forecast(quantile_forecasts, tau_value)
            if quantile is not None:
                q_forecasts[tau_value] = quantile
                report["pinball_sources"][tau_value] = "empirical_quantile"
            elif self._has_forecast_input(garch_vol):
                q_forecasts[tau_value] = self._compute_parametric_quantile(
                    garch_vol,
                    tau=tau_value,
                    dist=dist,
                    nu=nu,
                )
                report["pinball_sources"][tau_value] = "parametric_var"
            else:
                q_forecasts[tau_value] = np.array([], dtype=float)
                report["pinball_sources"][tau_value] = "missing"

        report["pinball"] = self.pinball_loss(actual_returns, q_forecasts)
        return report
