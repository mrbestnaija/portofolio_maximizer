"""Portfolio mathematics engine (legacy implementation).

Kept for backward reference after promoting the enhanced engine.
"""
import numpy as np
from typing import Dict


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Vectorized log returns: r_t = ln(P_t / P_{t-1})."""
    return np.diff(np.log(prices), axis=0)


def calculate_portfolio_metrics(returns: np.ndarray, weights: np.ndarray,
                                risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Vectorized portfolio performance calculations."""
    portfolio_returns = returns @ weights
    total_return = np.prod(1 + portfolio_returns) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 1e-8 else 0.0

    cumulative_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = 1 - cumulative_returns / running_max
    max_drawdown = np.max(drawdowns)

    return {
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'periods': len(portfolio_returns)
    }


def calculate_covariance_matrix(returns: np.ndarray) -> np.ndarray:
    """Vectorized covariance matrix computation (annualised)."""
    return np.cov(returns.T) * 252
