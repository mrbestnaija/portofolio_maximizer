"""Tests for portfolio mathematics with vectorized operations."""
import pytest
import numpy as np
from etl.portfolio_math import (
    calculate_returns,
    calculate_portfolio_metrics,
    calculate_covariance_matrix
)

@pytest.fixture
def sample_prices():
    """Sample price data for 100 days."""
    np.random.seed(42)
    return np.random.lognormal(0, 0.02, 100).cumprod() * 100

@pytest.fixture
def sample_returns():
    """Sample returns for 3 assets over 252 days."""
    np.random.seed(42)
    return np.random.normal(0.0005, 0.01, (252, 3))

@pytest.fixture
def equal_weights():
    """Equal weights for 3 assets."""
    return np.array([1/3, 1/3, 1/3])

def test_calculate_returns(sample_prices):
    """Test vectorized log returns calculation."""
    returns = calculate_returns(sample_prices)

    assert len(returns) == len(sample_prices) - 1
    assert np.all(np.abs(returns) < 1)  # Sanity check
    assert not np.isnan(returns).any()

def test_calculate_portfolio_metrics(sample_returns, equal_weights):
    """Test vectorized portfolio metrics calculation."""
    metrics = calculate_portfolio_metrics(sample_returns, equal_weights)

    # Check all required metrics exist
    assert 'total_return' in metrics
    assert 'annual_return' in metrics
    assert 'volatility' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'periods' in metrics

    # Sanity checks
    assert metrics['periods'] == 252
    assert -1 <= metrics['total_return'] <= 10
    assert 0 <= metrics['volatility'] <= 1
    assert 0 <= metrics['max_drawdown'] <= 1

def test_sharpe_ratio_calculation(sample_returns, equal_weights):
    """Test Sharpe ratio calculation."""
    metrics = calculate_portfolio_metrics(
        sample_returns,
        equal_weights,
        risk_free_rate=0.02
    )

    # Sharpe should be reasonable for normal returns
    assert -5 <= metrics['sharpe_ratio'] <= 5

def test_max_drawdown_bounds(sample_returns, equal_weights):
    """Test maximum drawdown bounds."""
    metrics = calculate_portfolio_metrics(sample_returns, equal_weights)

    # MDD should be between 0 and 1
    assert 0 <= metrics['max_drawdown'] <= 1

def test_calculate_covariance_matrix(sample_returns):
    """Test vectorized covariance matrix computation."""
    cov_matrix = calculate_covariance_matrix(sample_returns)

    # Check shape
    assert cov_matrix.shape == (3, 3)

    # Check symmetry
    assert np.allclose(cov_matrix, cov_matrix.T)

    # Check positive semi-definite (eigenvalues ≥ 0)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    assert np.all(eigenvalues >= -1e-10)

def test_weights_validation(sample_returns):
    """Test portfolio metrics with different weight vectors."""
    # All weight in first asset
    weights_1 = np.array([1.0, 0.0, 0.0])
    metrics_1 = calculate_portfolio_metrics(sample_returns, weights_1)

    # All weight in second asset
    weights_2 = np.array([0.0, 1.0, 0.0])
    metrics_2 = calculate_portfolio_metrics(sample_returns, weights_2)

    # Results should be different
    assert metrics_1['total_return'] != metrics_2['total_return']

def test_zero_volatility_sharpe(equal_weights):
    """Test Sharpe ratio when volatility is zero."""
    # Constant returns (zero volatility)
    constant_returns = np.full((252, 3), 0.0001)

    metrics = calculate_portfolio_metrics(constant_returns, equal_weights)

    # Should handle zero volatility gracefully
    assert metrics['sharpe_ratio'] == 0.0