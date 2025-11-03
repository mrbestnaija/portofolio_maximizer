"""
Statistical testing utilities for validating trading strategies.

Implements the quantitative checks mandated in the sequenced plan:
- Significance testing against a benchmark
- Autocorrelation diagnostics (Ljung–Box & Durbin–Watson)
- Bootstrap confidence intervals for key risk metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy import stats

from etl.portfolio_math import calculate_max_drawdown


@dataclass
class BootstrapIntervals:
    """Bootstrap confidence intervals for strategy diagnostics."""

    sharpe_ratio: Tuple[float, float]
    max_drawdown: Tuple[float, float]
    samples: int
    confidence_level: float


class StatisticalTestSuite:
    """Collection of reusable statistical validation helpers."""

    def __init__(self, annualisation_factor: int = 252) -> None:
        self.annualisation_factor = annualisation_factor

    @staticmethod
    def _to_numpy(data: Iterable[float]) -> np.ndarray:
        array = np.asarray(list(data), dtype=float)
        if array.ndim != 1:
            raise ValueError("Expected a one-dimensional sequence of returns")
        if array.size == 0:
            raise ValueError("Input sequence is empty")
        return array

    def test_strategy_significance(
        self,
        strategy_returns: Iterable[float],
        benchmark_returns: Iterable[float],
        alpha: float = 0.05,
    ) -> Dict[str, float]:
        """
        Evaluate whether a strategy beats its benchmark.

        Returns a dictionary containing:
            p_value: Two-sided paired t-test p-value on excess returns.
            significant: Whether the null hypothesis is rejected.
            information_ratio: Annualised mean(excess)/std(excess).
            f_statistic: Variance ratio statistic (strategy/bmk).
            variance_ratio: The ratio itself for convenience.
        """
        strategy = self._to_numpy(strategy_returns)
        benchmark = self._to_numpy(benchmark_returns)
        if strategy.shape[0] != benchmark.shape[0]:
            raise ValueError("Strategy and benchmark returns must have equal length")

        excess = strategy - benchmark
        mean_excess = excess.mean()
        std_excess = excess.std(ddof=1)

        t_stat, p_value = stats.ttest_rel(strategy, benchmark)
        significant = bool(p_value < alpha)

        information_ratio = (
            (mean_excess * np.sqrt(self.annualisation_factor) / std_excess)
            if std_excess > 0
            else 0.0
        )

        var_strategy = np.var(strategy, ddof=1)
        var_benchmark = np.var(benchmark, ddof=1)
        f_statistic = var_strategy / var_benchmark if var_benchmark > 0 else np.inf

        return {
            "p_value": float(p_value),
            "significant": float(p_value) < alpha,
            "information_ratio": float(information_ratio),
            "variance_ratio": float(f_statistic),
            "f_statistic": float(f_statistic),
            "mean_excess": float(mean_excess),
        }

    def test_autocorrelation(
        self, returns: Iterable[float], lags: int = 10
    ) -> Dict[str, float]:
        """
        Run Ljung–Box and Durbin–Watson diagnostics on return autocorrelation.
        """
        series = self._to_numpy(returns)
        if lags <= 0:
            raise ValueError("lags must be greater than zero")
        if series.size <= lags:
            raise ValueError("Not enough observations for requested lags")

        mean = series.mean()
        centered = series - mean
        var = np.var(centered, ddof=0)

        # Autocorrelation coefficients up to `lags`
        acf = np.array(
            [
                np.sum(centered[: series.size - k] * centered[k:]) / (var * (series.size - k))
                for k in range(1, lags + 1)
            ]
        )

        lb_stat = series.size * (series.size + 2) * np.sum(
            (acf**2) / (series.size - np.arange(1, lags + 1))
        )
        lb_pvalue = stats.chi2.sf(lb_stat, df=lags)

        # Durbin–Watson
        diff = np.diff(series)
        dw_stat = np.sum(diff**2) / np.sum(centered**2)

        return {
            "ljung_box_stat": float(lb_stat),
            "ljung_box_p_value": float(lb_pvalue),
            "durbin_watson": float(dw_stat),
        }

    def bootstrap_validation(
        self,
        returns: Iterable[float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None,
    ) -> BootstrapIntervals:
        """
        Bootstrap confidence intervals for Sharpe ratio and max drawdown.
        """
        series = self._to_numpy(returns)

        rng = np.random.default_rng(random_state)
        sharpe_samples = np.empty(n_bootstrap)
        drawdown_samples = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            resample = rng.choice(series, size=series.size, replace=True)
            mean = resample.mean()
            std = resample.std(ddof=1)
            sharpe_samples[i] = (
                (mean / std) * np.sqrt(self.annualisation_factor) if std > 0 else 0.0
            )
            drawdown_samples[i] = calculate_max_drawdown(resample)

        alpha = (1 - confidence_level) / 2
        lower_q = alpha
        upper_q = 1 - alpha

        sharpe_ci = (
            float(np.quantile(sharpe_samples, lower_q)),
            float(np.quantile(sharpe_samples, upper_q)),
        )
        drawdown_ci = (
            float(np.quantile(drawdown_samples, lower_q)),
            float(np.quantile(drawdown_samples, upper_q)),
        )

        return BootstrapIntervals(
            sharpe_ratio=sharpe_ci,
            max_drawdown=drawdown_ci,
            samples=n_bootstrap,
            confidence_level=confidence_level,
        )
