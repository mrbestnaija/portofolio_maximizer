"""Portfolio mathematics engine with vectorized operations.

Mathematical Foundation:
- Portfolio return: R_p = w^T * r (dot product)
- Portfolio variance: σ_p² = w^T * Σ * w (quadratic form)
- Sharpe ratio: SR = (μ_p - r_f) / σ_p * √252
- Maximum drawdown: MDD = max(1 - P_t / max_{s≤t}(P_s))

Success Criteria:
- SPY 1-year return matches Yahoo Finance within 0.1%
- Covariance matrix computation O(n²) complexity
- All metrics vectorized (no explicit loops)
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Vectorized log returns: r_t = ln(P_t / P_{t-1})."""
    return np.diff(np.log(prices), axis=0)

def calculate_portfolio_metrics(returns: np.ndarray, weights: np.ndarray,
                                risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Vectorized portfolio performance calculations.

    Args:
        returns: (T, N) array of asset returns
        weights: (N,) array of portfolio weights
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with performance metrics
    """
    # Portfolio returns (vectorized dot product)
    portfolio_returns = returns @ weights

    # Annualized metrics
    total_return = np.prod(1 + portfolio_returns) - 1
    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = np.std(portfolio_returns) * np.sqrt(252)

    # Risk-adjusted metrics (with tolerance for near-zero volatility)
    sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 1e-8 else 0.0

    # Drawdown calculation (vectorized cumulative max)
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
    """Vectorized covariance matrix computation.

    Args:
        returns: (T, N) array of asset returns

    Returns:
        (N, N) covariance matrix
    """
    # Annualized covariance (vectorized)
    return np.cov(returns.T) * 252