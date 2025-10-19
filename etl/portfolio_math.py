"""
Enhanced Portfolio Mathematics Engine - Institutional Grade.

This module promotes the previously experimental `portfolio_math_enhanced`
implementation to production while preserving backwards compatible APIs used
throughout the codebase.

Key capabilities:
- Full suite of risk-adjusted performance metrics (Sharpe, Sortino, Calmar).
- Tail-risk analysis via VaR / CVaR / Expected Shortfall.
- Benchmark-relative analytics (alpha, beta, tracking error, information ratio).
- Kelly criterion sizing with safety caps.
- Robust covariance estimation (Ledoit–Wolf / OAS fallbacks).
- Markowitz and risk-parity optimisations.
- Bootstrap confidence intervals and stress testing helpers.
- Statistical significance testing for strategy validation.
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

DEFAULT_RISK_FREE_RATE = 0.02
TRADING_DAYS = 252

__all__ = [
    "calculate_returns",
    "calculate_enhanced_portfolio_metrics",
    "calculate_portfolio_metrics",
    "calculate_covariance_matrix",
    "calculate_robust_covariance_matrix",
    "calculate_kelly_fraction_correct",
    "optimize_portfolio_markowitz",
    "optimize_portfolio_risk_parity",
    "bootstrap_confidence_intervals",
    "calculate_max_drawdown",
    "calculate_sortino_ratio",
    "test_strategy_significance",
    "stress_test_portfolio",
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Vectorised log returns: r_t = ln(P_t / P_{t-1})."""
    prices = np.asarray(prices)
    if prices.ndim == 1:
        prices = prices.reshape(-1, 1)
    return np.diff(np.log(prices), axis=0)


def _annualise(series: np.ndarray) -> float:
    return float(series) * np.sqrt(TRADING_DAYS)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def calculate_enhanced_portfolio_metrics(
    returns: np.ndarray,
    weights: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    benchmark_returns: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Institutional-grade portfolio performance calculations.

    Args:
        returns: (T, N) array of asset returns.
        weights: (N,) array of portfolio weights.
        risk_free_rate: Annual risk-free rate.
        benchmark_returns: Optional benchmark returns for relative metrics.

    Returns:
        Dictionary containing comprehensive metrics.
    """
    returns = np.asarray(returns)
    weights = np.asarray(weights)

    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    if len(weights) != returns.shape[1]:
        raise ValueError("weights length must match number of assets in returns")

    portfolio_returns = returns @ weights
    if len(portfolio_returns) == 0:
        raise ValueError("returns must contain observations")

    total_return = np.prod(1 + portfolio_returns) - 1
    annual_return = (1 + total_return) ** (TRADING_DAYS / len(portfolio_returns)) - 1
    volatility = np.std(portfolio_returns) * np.sqrt(TRADING_DAYS)

    sharpe_ratio = (
        (annual_return - risk_free_rate) / volatility if volatility > 1e-8 else 0.0
    )

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_volatility = (
        np.std(downside_returns) * np.sqrt(TRADING_DAYS)
        if len(downside_returns) > 0
        else 0.0
    )
    sortino_ratio = (
        (annual_return - risk_free_rate) / downside_volatility
        if downside_volatility > 1e-8
        else 0.0
    )

    cumulative_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = 1 - cumulative_returns / running_max
    max_drawdown = float(np.max(drawdowns))
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 1e-8 else 0.0

    var_95 = float(np.percentile(portfolio_returns, 5))
    var_99 = float(np.percentile(portfolio_returns, 1))

    cvar_95 = float(portfolio_returns[portfolio_returns <= var_95].mean())
    cvar_99 = float(portfolio_returns[portfolio_returns <= var_99].mean())
    expected_shortfall = float(portfolio_returns[portfolio_returns < 0].mean())

    metrics: Dict[str, float] = {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar_ratio),
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "expected_shortfall": expected_shortfall,
        "periods": int(len(portfolio_returns)),
    }

    if benchmark_returns is not None:
        benchmark_returns = np.asarray(benchmark_returns)
        if benchmark_returns.shape[0] != portfolio_returns.shape[0]:
            raise ValueError("benchmark_returns must align with portfolio returns")

        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(TRADING_DAYS)
        information_ratio = (
            np.mean(excess_returns) * np.sqrt(TRADING_DAYS) / tracking_error
            if tracking_error > 1e-8
            else 0.0
        )

        slope, intercept, r_value, _, _ = stats.linregress(
            benchmark_returns, portfolio_returns
        )
        alpha = intercept * TRADING_DAYS
        beta = slope

        metrics.update(
            {
                "information_ratio": float(information_ratio),
                "alpha": float(alpha),
                "beta": float(beta),
                "r_squared": float(r_value**2),
                "tracking_error": float(tracking_error),
            }
        )

    return metrics


def calculate_portfolio_metrics(
    returns: np.ndarray,
    weights: np.ndarray,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, float]:
    """
    Backwards compatible wrapper returning the core subset of metrics.
    """
    metrics = calculate_enhanced_portfolio_metrics(
        returns=returns,
        weights=weights,
        risk_free_rate=risk_free_rate,
    )
    return {
        key: metrics[key]
        for key in (
            "total_return",
            "annual_return",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
            "periods",
        )
    }


def calculate_covariance_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Backwards compatible covariance matrix (empirical annualised).
    """
    return calculate_robust_covariance_matrix(returns, method="empirical")


# ---------------------------------------------------------------------------
# Risk modelling utilities
# ---------------------------------------------------------------------------
def calculate_kelly_fraction_correct(
    win_rate: float, avg_win: float, avg_loss: float
) -> float:
    """
    Correct Kelly Criterion implementation capped to 25% for prudence.
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b
    return float(max(0.0, min(kelly, 0.25)))


def calculate_robust_covariance_matrix(
    returns: np.ndarray, method: str = "ledoit_wolf"
) -> np.ndarray:
    """
    Estimate a robust annualised covariance matrix.
    """
    returns = np.asarray(returns)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    if method == "empirical":
        return np.cov(returns.T) * TRADING_DAYS

    estimator = None
    if method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf

            estimator = LedoitWolf()
        except ImportError:
            warnings.warn(
                "sklearn not available; falling back to empirical covariance.",
                RuntimeWarning,
            )
    elif method == "oas":
        try:
            from sklearn.covariance import OAS

            estimator = OAS()
        except ImportError:
            warnings.warn(
                "sklearn not available; falling back to empirical covariance.",
                RuntimeWarning,
            )
    else:
        raise ValueError(f"Unknown covariance estimation method: {method}")

    if estimator is None:
        return np.cov(returns.T) * TRADING_DAYS

    return estimator.fit(returns).covariance_ * TRADING_DAYS


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
def optimize_portfolio_markowitz(
    returns: np.ndarray,
    risk_aversion: float = 1.0,
    constraints: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Markowitz mean-variance optimisation with long-only constraints.
    """
    returns = np.asarray(returns)
    mu = np.mean(returns, axis=0)
    Sigma = np.cov(returns.T)
    n_assets = len(mu)

    def objective(w: np.ndarray) -> float:
        return 0.5 * risk_aversion * w.T @ Sigma @ w - w.T @ mu

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if constraints and "max_weight" in constraints:
        bounds = [(0.0, constraints["max_weight"]) for _ in range(n_assets)]
    else:
        bounds = [(0.0, 1.0) for _ in range(n_assets)]

    result = minimize(
        objective,
        x0=np.ones(n_assets) / n_assets,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )

    if not result.success:
        logger.warning("Markowitz optimisation failed: %s", result.message)
        return np.ones(n_assets) / n_assets, {"success": False, "message": result.message}

    return result.x, {"success": True, "fun": result.fun, "iterations": result.nit}


def optimize_portfolio_risk_parity(returns: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Risk parity optimisation producing equal risk contributions.
    """
    returns = np.asarray(returns)
    Sigma = np.cov(returns.T)
    n_assets = Sigma.shape[0]

    def objective(w: np.ndarray) -> float:
        portfolio_vol = np.sqrt(w.T @ Sigma @ w)
        if portfolio_vol < 1e-8:
            return 1e6

        risk_contributions = (w * (Sigma @ w)) / portfolio_vol
        target = portfolio_vol / n_assets
        return np.sum((risk_contributions - target) ** 2)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0) for _ in range(n_assets)]

    result = minimize(
        objective,
        x0=np.ones(n_assets) / n_assets,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )

    if not result.success:
        logger.warning("Risk parity optimisation failed: %s", result.message)
        return np.ones(n_assets) / n_assets, {"success": False, "message": result.message}

    return result.x, {"success": True, "fun": result.fun, "iterations": result.nit}


# ---------------------------------------------------------------------------
# Statistical tooling
# ---------------------------------------------------------------------------
def bootstrap_confidence_intervals(
    returns: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap confidence intervals for Sharpe, Sortino, and max drawdown.
    """
    returns = np.asarray(returns)
    if returns.size == 0:
        raise ValueError("returns must contain observations")

    metrics: List[Dict[str, float]] = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        sharpe = (
            (np.mean(sample) - DEFAULT_RISK_FREE_RATE / TRADING_DAYS)
            / np.std(sample)
            * np.sqrt(TRADING_DAYS)
            if np.std(sample) > 1e-8
            else 0.0
        )
        max_dd = calculate_max_drawdown(sample)
        sortino = calculate_sortino_ratio(sample)
        metrics.append({"sharpe": sharpe, "max_drawdown": max_dd, "sortino": sortino})

    alpha = (1 - confidence_level) / 2
    lower = alpha * 100
    upper = (1 - alpha) * 100

    sharpe_values = [m["sharpe"] for m in metrics]
    dd_values = [m["max_drawdown"] for m in metrics]
    sortino_values = [m["sortino"] for m in metrics]

    return {
        "sharpe_ci": (float(np.percentile(sharpe_values, lower)), float(np.percentile(sharpe_values, upper))),
        "max_dd_ci": (float(np.percentile(dd_values, lower)), float(np.percentile(dd_values, upper))),
        "sortino_ci": (float(np.percentile(sortino_values, lower)), float(np.percentile(sortino_values, upper))),
        "sharpe_std": float(np.std(sharpe_values)),
        "max_dd_std": float(np.std(dd_values)),
        "sortino_std": float(np.std(sortino_values)),
    }


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from equity curve."""
    returns = np.asarray(returns)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = 1 - cumulative / running_max
    return float(np.max(drawdowns))


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Sortino ratio using downside deviation."""
    returns = np.asarray(returns)
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 0.0
    return (
        (np.mean(returns) - risk_free_rate / TRADING_DAYS) / downside_std
        if downside_std > 1e-8
        else 0.0
    )


def test_strategy_significance(
    strategy_returns: np.ndarray, benchmark_returns: np.ndarray
) -> Dict[str, float]:
    """
    Hypothesis testing to determine if strategy outperforms the benchmark.
    """
    strategy_returns = np.asarray(strategy_returns)
    benchmark_returns = np.asarray(benchmark_returns)
    if strategy_returns.shape != benchmark_returns.shape:
        raise ValueError("strategy_returns and benchmark_returns must align")

    t_stat, p_value = stats.ttest_ind(strategy_returns, benchmark_returns, equal_var=False)

    excess = strategy_returns - benchmark_returns
    information_ratio = (
        np.mean(excess) / np.std(excess) if np.std(excess) > 1e-8 else 0.0
    )

    f_stat, f_p_value = stats.levene(strategy_returns, benchmark_returns)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "information_ratio": float(information_ratio),
        "f_statistic": float(f_stat),
        "f_p_value": float(f_p_value),
        "variance_equal": bool(f_p_value > 0.05),
    }

test_strategy_significance.__test__ = False  # Prevent pytest from collecting as a test.

# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------
def stress_test_portfolio(
    portfolio_returns: np.ndarray,
    scenarios: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Apply shock scenarios to portfolio returns and report stressed metrics.
    """
    portfolio_returns = np.asarray(portfolio_returns)
    results: Dict[str, Dict[str, float]] = {}

    for name, shock in scenarios.items():
        stressed = portfolio_returns + shock
        sharpe = (
            (np.mean(stressed) - DEFAULT_RISK_FREE_RATE / TRADING_DAYS)
            / np.std(stressed)
            * np.sqrt(TRADING_DAYS)
            if np.std(stressed) > 1e-8
            else 0.0
        )
        results[name] = {
            "shock_magnitude": float(shock),
            "stressed_sharpe": float(sharpe),
            "stressed_max_drawdown": float(calculate_max_drawdown(stressed)),
            "stressed_var_95": float(np.percentile(stressed, 5)),
            "stressed_var_99": float(np.percentile(stressed, 1)),
            "portfolio_loss": float(shock * len(portfolio_returns)),
        }

    return results
