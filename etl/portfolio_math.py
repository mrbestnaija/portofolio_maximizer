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

import math
import logging
import warnings
from typing import Any, Dict, Optional, Tuple, List

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
    # Phase 11 — Nigeria production extensions (additive, zero breakage)
    "omega_ratio",
    "omega_curve",
    "omega_robustness_summary",
    "payoff_asymmetry_ratio",
    "payoff_asymmetry_support_metrics",
    "fractional_kelly_fat_tail",
    "effective_ngn_return",
    "portfolio_metrics_ngn",
    "NGN_ANNUAL_INFLATION",
    "NGN_P2P_FRICTION",
    "DAILY_NGN_THRESHOLD",
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

    # Handle empty slices to avoid warnings
    negative_returns_95 = portfolio_returns[portfolio_returns <= var_95]
    negative_returns_99 = portfolio_returns[portfolio_returns <= var_99]
    negative_returns = portfolio_returns[portfolio_returns < 0]
    
    cvar_95 = float(negative_returns_95.mean()) if len(negative_returns_95) > 0 else 0.0
    cvar_99 = float(negative_returns_99.mean()) if len(negative_returns_99) > 0 else 0.0
    expected_shortfall = float(negative_returns.mean()) if len(negative_returns) > 0 else 0.0

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


# ---------------------------------------------------------------------------
# Phase 11 — Nigeria Production Extensions
# Additive only. No existing function modified. All 2208+ tests unaffected.
# Implementation date: 2026-04-06. Wiring (Phases B-E) begins after
# THIN_LINKAGE ≥ 10 (warmup expires 2026-04-15).
# ---------------------------------------------------------------------------

import os as _os

# ── NGN Calibration Constants ────────────────────────────────────────────────
# Update quarterly from CBN Statistical Bulletin and Bybit P2P median.
# Source: CBN MPC communiqués, Q1 2026 headline CPI + parallel-rate friction.
# Override via environment variables for live deployment.
NGN_ANNUAL_INFLATION: float = float(_os.getenv("NGN_ANNUAL_INFLATION", "0.28"))
NGN_P2P_FRICTION: float = float(_os.getenv("NGN_P2P_FRICTION", "0.03"))
# Daily compound hurdle: 28% inflation + 3% P2P friction, annualised to daily
DAILY_NGN_THRESHOLD: float = (
    (1.0 + NGN_ANNUAL_INFLATION + NGN_P2P_FRICTION) ** (1.0 / TRADING_DAYS) - 1.0
)


def omega_ratio(
    returns: pd.Series,
    threshold: float | None = None,
) -> float:
    """
    Omega ratio partitioned at the Nigeria-inflation-adjusted daily threshold.

    Replaces Sharpe as the primary barbell objective. Sharpe penalises
    asymmetric upside identically to downside — directly contradicting
    barbell philosophy. Omega is distribution-free and captures the
    full shape of the return distribution above/below the hurdle.

    Omega = sum(max(r - τ, 0)) / sum(max(τ - r, 0))

    τ defaults to DAILY_NGN_THRESHOLD — the true hurdle rate for a
    Nigeria-domiciled account (NGN inflation + USDT P2P withdrawal
    friction), not the USD risk-free rate used by Sharpe.

    Parameters
    ----------
    returns   : Daily return series from actual execution log (pd.Series)
    threshold : Daily threshold. None → DAILY_NGN_THRESHOLD

    Returns
    -------
    float — Omega ratio. >1 = portfolio beats NGN hurdle more than it misses.
            inf when zero losses above threshold. nan when series too short.

    Notes
    -----
    Requires ≥ 10 observations to be meaningful. The existing
    calculate_kelly_fraction_correct and Sharpe-based metrics are retained
    for backward compatibility with existing gate checks.
    """
    if returns is None or len(returns) < 10:
        return float("nan")

    tau = DAILY_NGN_THRESHOLD if threshold is None else threshold
    excess = pd.Series(returns) - tau
    gain = float(excess.clip(lower=0).sum())
    loss = float((-excess).clip(lower=0).sum())

    if loss == 0.0:
        return float("inf")
    return gain / loss


def _clean_return_series(returns: pd.Series | np.ndarray | list[float]) -> pd.Series:
    if returns is None:
        return pd.Series(dtype=float)
    return pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()


def omega_curve(
    returns: pd.Series,
    *,
    execution_drag_hurdle: float | None = None,
) -> Dict[str, Dict[str, float | None]]:
    """
    Omega values across a small hurdle ladder.

    The barbell argument only holds when the distribution still clears a
    realistic hurdle after execution drag. A single Omega point can flatter a
    strategy whose edge disappears once friction is added back in.
    """
    clean = _clean_return_series(returns)
    drag = None
    try:
        if execution_drag_hurdle is not None:
            drag = max(float(execution_drag_hurdle), 0.0)
    except (TypeError, ValueError):
        drag = None

    thresholds = {
        "zero": 0.0,
        "ngn_hurdle": DAILY_NGN_THRESHOLD,
        "cost_adjusted": (
            DAILY_NGN_THRESHOLD + drag
            if drag is not None
            else None
        ),
    }

    curve: Dict[str, Dict[str, float | None]] = {}
    for label, threshold in thresholds.items():
        if threshold is None:
            omega_value = None
        else:
            value = omega_ratio(clean, threshold)
            omega_value = None if (isinstance(value, float) and math.isnan(value)) else float(value)
        curve[label] = {
            "threshold": (float(threshold) if threshold is not None else None),
            "omega": omega_value,
        }
    return curve


def omega_robustness_summary(
    returns: pd.Series,
    *,
    execution_drag_hurdle: float | None = None,
) -> Dict[str, Any]:
    """
    Summarize whether Omega survives realistic hurdle escalation.

    `omega_ratio` remains the headline metric, but a barbell claim is only
    robust when Omega stays healthy at the NGN hurdle and after adding realized
    execution drag.
    """
    curve = omega_curve(returns, execution_drag_hurdle=execution_drag_hurdle)
    omega_zero = curve.get("zero", {}).get("omega")
    omega_hurdle = curve.get("ngn_hurdle", {}).get("omega")
    omega_cost = curve.get("cost_adjusted", {}).get("omega")
    cost_threshold = curve.get("cost_adjusted", {}).get("threshold")

    finite_points: List[float] = []
    for point in (omega_zero, omega_hurdle, omega_cost):
        if isinstance(point, (int, float)) and not math.isnan(point):
            finite_points.append(float(point))

    monotonicity_ok = True
    if len(finite_points) >= 2:
        for left, right in zip(finite_points, finite_points[1:]):
            if right > left + 1e-9:
                monotonicity_ok = False
                break

    omega_above_hurdle_margin = None
    if isinstance(omega_hurdle, (int, float)) and not math.isnan(float(omega_hurdle)):
        omega_above_hurdle_margin = float(omega_hurdle) - 1.0

    complete = cost_threshold is not None
    robustness_score: Optional[float] = None
    if complete and all(
        isinstance(point, (int, float)) and not math.isnan(float(point))
        for point in (omega_zero, omega_hurdle, omega_cost)
    ):
        omega_zero_f = max(float(omega_zero), 0.0)
        omega_hurdle_f = max(float(omega_hurdle), 0.0)
        omega_cost_f = max(float(omega_cost), 0.0)
        hurdle_strength = float(np.clip((omega_hurdle_f - 1.0) / 1.0, 0.0, 1.0))
        drag_strength = float(np.clip((omega_cost_f - 1.0) / 1.0, 0.0, 1.0))
        retention = float(np.clip(omega_cost_f / max(omega_hurdle_f, 1e-6), 0.0, 1.0))
        threshold_stability = float(np.clip(omega_hurdle_f / max(omega_zero_f, 1e-6), 0.0, 1.0))
        robustness_score = (
            0.40 * hurdle_strength
            + 0.30 * drag_strength
            + 0.20 * retention
            + 0.10 * threshold_stability
        )
        if not monotonicity_ok:
            robustness_score *= 0.50
        robustness_score = float(np.clip(robustness_score, 0.0, 1.0))

    return {
        "omega_curve": curve,
        "omega_robustness_score": robustness_score,
        "omega_monotonicity_ok": bool(monotonicity_ok),
        "omega_above_hurdle_margin": omega_above_hurdle_margin,
        "omega_robustness_complete": bool(complete),
        "execution_drag_hurdle": (
            float(execution_drag_hurdle)
            if execution_drag_hurdle is not None
            else None
        ),
    }


def payoff_asymmetry_ratio(returns: pd.Series) -> float:
    """
    Average win divided by average loss magnitude.

    This isolates payoff shape from hit-rate frequency. A low-win-rate barbell
    sleeve can still be structurally attractive when its winners are materially
    larger than its losers; this metric captures that engine directly.
    """
    if returns is None:
        return float("nan")

    s = pd.Series(returns).dropna()
    if s.empty:
        return float("nan")

    wins = s[s > 0]
    losses = s[s < 0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = abs(float(losses.mean())) if not losses.empty else 0.0

    if avg_loss == 0.0:
        return float("inf") if avg_win > 0.0 else 0.0
    return avg_win / avg_loss


def payoff_asymmetry_support_metrics(
    returns: pd.Series,
    *,
    trim_fraction: float = 0.10,
    min_wins: int = 3,
    min_losses: int = 3,
    max_winner_concentration_ratio: float = 0.60,
) -> Dict[str, Any]:
    """
    Support-aware payoff asymmetry diagnostics.

    Raw asymmetry stays visible because barbell systems *should* monetize rare
    fat-tail winners. But promotion-grade evidence must prove the asymmetry is
    not just one lucky outlier.
    """
    clean = _clean_return_series(returns)
    wins = clean[clean > 0].sort_values()
    losses = clean[clean < 0].abs().sort_values()

    raw_ratio = payoff_asymmetry_ratio(clean)
    n_wins = int(wins.size)
    n_losses = int(losses.size)
    gross_profit = float(wins.sum()) if n_wins else 0.0
    winner_concentration_ratio = (
        float(wins.iloc[-1] / gross_profit)
        if n_wins and gross_profit > 0.0
        else float("inf") if n_wins else 0.0
    )

    trim_fraction = float(np.clip(trim_fraction, 0.0, 0.49))
    trim_wins = int(math.floor(n_wins * trim_fraction))
    trim_losses = int(math.floor(n_losses * trim_fraction))
    trimmed_wins = wins.iloc[: max(n_wins - trim_wins, 0)] if trim_wins else wins
    trimmed_losses = losses.iloc[: max(n_losses - trim_losses, 0)] if trim_losses else losses

    if trimmed_losses.size == 0:
        trimmed_payoff_asymmetry = float("inf") if trimmed_wins.size > 0 else 0.0
    elif trimmed_wins.size == 0:
        trimmed_payoff_asymmetry = 0.0
    else:
        trimmed_payoff_asymmetry = float(trimmed_wins.mean() / max(float(trimmed_losses.mean()), 1e-12))

    support_ok = (
        n_wins >= int(min_wins)
        and n_losses >= int(min_losses)
        and winner_concentration_ratio <= float(max_winner_concentration_ratio)
    )

    if support_ok:
        if math.isinf(raw_ratio) and math.isfinite(trimmed_payoff_asymmetry):
            effective = float(trimmed_payoff_asymmetry)
        elif math.isinf(trimmed_payoff_asymmetry) and math.isfinite(raw_ratio):
            effective = float(raw_ratio)
        elif math.isinf(raw_ratio) and math.isinf(trimmed_payoff_asymmetry):
            effective = float("inf")
        else:
            effective = float(min(raw_ratio, trimmed_payoff_asymmetry))
    else:
        effective = 0.0

    return {
        "payoff_asymmetry": raw_ratio,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "winner_concentration_ratio": float(winner_concentration_ratio),
        "trimmed_payoff_asymmetry": float(trimmed_payoff_asymmetry),
        "payoff_asymmetry_support_ok": bool(support_ok),
        "payoff_asymmetry_effective": float(effective),
    }


def fractional_kelly_fat_tail(
    returns: pd.Series,
    risk_free: float | None = None,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Fractional Kelly with excess-kurtosis correction for fat-tailed assets.

    Extends the existing calculate_kelly_fraction_correct with a kurtosis
    dampener. Crypto and EM FX return distributions have excess kurtosis
    κ >> 3; applying full Kelly to such series is catastrophic.

    f* = [(μ - rf) / σ²] × [1 / (1 + max(κ-3, 0) / 4)] × λ

    Hard cap: position fraction clipped to [0.0, 0.20].

    Parameters
    ----------
    returns        : Daily return series from actual execution log (pd.Series)
    risk_free      : Daily risk-free rate. None → DAILY_NGN_THRESHOLD
    kelly_fraction : Fractional multiplier λ (default 0.25 = quarter-Kelly)

    Returns
    -------
    float — Position fraction in [0.0, 0.20].

    Notes
    -----
    Use alongside calculate_kelly_fraction_correct for cross-validation.
    Requires ≥ 30 observations; returns 0.01 (minimum stake) for short series.
    """
    if returns is None or len(returns) < 30:
        return 0.01

    rf = DAILY_NGN_THRESHOLD if risk_free is None else risk_free
    s = pd.Series(returns)
    mu = float(s.mean())
    sigma2 = float(s.var())
    kappa = float(s.kurtosis())  # pandas kurtosis = excess kurtosis (Fisher)

    if sigma2 == 0.0:
        return 0.0

    full_kelly = (mu - rf) / sigma2
    kurtosis_correction = 1.0 / (1.0 + max(kappa - 3.0, 0.0) / 4.0)
    f_star = full_kelly * kurtosis_correction * kelly_fraction

    return float(np.clip(f_star, 0.0, 0.20))


def effective_ngn_return(
    usd_return: float,
    ngn_usd_spot_change: float,
    withdrawal_friction: float | None = None,
) -> float:
    """
    Realised return in NGN terms after USDT-bridge conversion.

    Every USD PnL from OANDA/IC Markets/Bybit passes through a
    USDT → NGN P2P conversion on withdrawal. NGN has a structural
    devaluation drift — this is not symmetric noise.

    R_eff = R_USD + Δspot_NGN/USD − friction_per_day

    Parameters
    ----------
    usd_return          : Return denominated in USD (fractional, not %)
    ngn_usd_spot_change : Fractional change in NGN/USD spot rate for the period.
                          Positive = NGN weakened → favourable for USD holder.
    withdrawal_friction : P2P round-trip friction per day.
                          None → NGN_P2P_FRICTION / TRADING_DAYS

    Returns
    -------
    float — Effective daily return in NGN purchasing-power terms.
    """
    friction = (
        NGN_P2P_FRICTION / TRADING_DAYS
        if withdrawal_friction is None
        else withdrawal_friction
    )
    return float(usd_return + ngn_usd_spot_change - friction)


def portfolio_metrics_ngn(
    returns: pd.Series,
    *,
    execution_drag_hurdle: float | None = None,
) -> Dict[str, Any]:
    """
    Full Nigeria-calibrated metric set.

    Wraps calculate_enhanced_portfolio_metrics (existing, backward-compatible)
    and appends NGN-specific extensions. Does not replace or modify any
    existing function.

    Parameters
    ----------
    returns : Daily return series from actual execution log (pd.Series).
              Must be compatible with calculate_enhanced_portfolio_metrics.

    Returns
    -------
    Dict extending the base enhanced_metrics with:
      - omega_ratio           : Omega ratio vs NGN hurdle
      - payoff_asymmetry      : avg_win / |avg_loss| structural barbell engine
      - fractional_kelly_fat_tail : Quarter-Kelly with kurtosis correction
      - ngn_daily_threshold   : Current daily NGN hurdle rate
      - ngn_annual_hurdle_pct : Annualised hurdle in % (inflation + friction)
      - beats_ngn_hurdle      : bool, omega_ratio > 1.0

    Notes
    -----
    The "beats_ngn_hurdle" flag is the canonical Phase 11 success criterion.
    A system beating the NGN hurdle is outperforming the structural
    devaluation rate — the minimum acceptable bar for Nigeria deployment.
    """
    arr = np.array(returns).reshape(-1, 1)
    weights = np.ones(1)
    base: Dict[str, float] = calculate_enhanced_portfolio_metrics(arr, weights)

    omega = omega_ratio(returns)
    omega_robustness = omega_robustness_summary(
        returns,
        execution_drag_hurdle=execution_drag_hurdle,
    )
    asymmetry_support = payoff_asymmetry_support_metrics(returns)
    ngn_ext: Dict[str, object] = {
        "omega_ratio": omega,
        **omega_robustness,
        **asymmetry_support,
        "fractional_kelly_fat_tail": fractional_kelly_fat_tail(returns),
        "ngn_daily_threshold": DAILY_NGN_THRESHOLD,
        "ngn_annual_hurdle_pct": round(
            (NGN_ANNUAL_INFLATION + NGN_P2P_FRICTION) * 100.0, 1
        ),
        "beats_ngn_hurdle": (omega > 1.0) if not (isinstance(omega, float) and omega != omega) else False,
    }
    return {**base, **ngn_ext}
