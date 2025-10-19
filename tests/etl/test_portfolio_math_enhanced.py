"""
Enhanced Portfolio Mathematics Tests - Professional Standards
Tests for institutional-grade mathematical functions

Focus: Mathematical correctness, statistical rigor, edge cases
Maximum: 500 lines (per Phase 4-6 guidelines)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from etl.portfolio_math import (
    calculate_enhanced_portfolio_metrics,
    calculate_kelly_fraction_correct,
    calculate_robust_covariance_matrix,
    optimize_portfolio_markowitz,
    optimize_portfolio_risk_parity,
    bootstrap_confidence_intervals,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    test_strategy_significance,
    stress_test_portfolio
)


class TestEnhancedPortfolioMetrics:
    """Test enhanced portfolio metrics calculations."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, (252, 3))  # 3 assets, 252 days
    
    @pytest.fixture
    def sample_weights(self):
        """Equal weight portfolio."""
        return np.array([0.4, 0.3, 0.3])
    
    @pytest.fixture
    def benchmark_returns(self):
        """Generate benchmark returns."""
        np.random.seed(43)
        return np.random.normal(0.0008, 0.015, 252)
    
    def test_enhanced_metrics_basic(self, sample_returns, sample_weights):
        """Test basic enhanced metrics calculation."""
        metrics = calculate_enhanced_portfolio_metrics(sample_returns, sample_weights)
        
        # Check all required metrics exist
        required_metrics = [
            'total_return', 'annual_return', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'calmar_ratio',
            'var_95', 'var_99', 'cvar_95', 'cvar_99', 'expected_shortfall'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Metric {metric} not numeric"
            assert not np.isnan(metrics[metric]), f"Metric {metric} is NaN"
    
    def test_enhanced_metrics_with_benchmark(self, sample_returns, sample_weights, benchmark_returns):
        """Test enhanced metrics with benchmark comparison."""
        metrics = calculate_enhanced_portfolio_metrics(
            sample_returns, sample_weights, benchmark_returns=benchmark_returns
        )
        
        # Check benchmark-relative metrics
        benchmark_metrics = [
            'information_ratio', 'alpha', 'beta', 'r_squared', 'tracking_error'
        ]
        
        for metric in benchmark_metrics:
            assert metric in metrics, f"Missing benchmark metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Benchmark metric {metric} not numeric"
            assert not np.isnan(metrics[metric]), f"Benchmark metric {metric} is NaN"
    
    def test_sortino_ratio_calculation(self, sample_returns, sample_weights):
        """Test Sortino ratio calculation."""
        metrics = calculate_enhanced_portfolio_metrics(sample_returns, sample_weights)
        
        # Sortino ratio should be >= Sharpe ratio (downside deviation <= total deviation)
        assert metrics['sortino_ratio'] >= metrics['sharpe_ratio'], \
            "Sortino ratio should be >= Sharpe ratio"
        
        # Sortino ratio should be positive for positive returns
        if metrics['annual_return'] > 0.02:  # Above risk-free rate
            assert metrics['sortino_ratio'] > 0, "Sortino ratio should be positive for positive returns"
    
    def test_var_cvar_calculation(self, sample_returns, sample_weights):
        """Test VaR and CVaR calculations."""
        metrics = calculate_enhanced_portfolio_metrics(sample_returns, sample_weights)
        
        # VaR 99% should be <= VaR 95%
        assert metrics['var_99'] <= metrics['var_95'], "VaR 99% should be <= VaR 95%"
        
        # CVaR should be <= VaR (expected shortfall <= VaR)
        assert metrics['cvar_95'] <= metrics['var_95'], "CVaR 95% should be <= VaR 95%"
        assert metrics['cvar_99'] <= metrics['var_99'], "CVaR 99% should be <= VaR 99%"
        
        # Expected shortfall should be negative (average of negative returns)
        assert metrics['expected_shortfall'] <= 0, "Expected shortfall should be <= 0"
    
    def test_calmar_ratio_calculation(self, sample_returns, sample_weights):
        """Test Calmar ratio calculation."""
        metrics = calculate_enhanced_portfolio_metrics(sample_returns, sample_weights)
        
        # Calmar ratio should be positive for positive returns
        if metrics['annual_return'] > 0:
            assert metrics['calmar_ratio'] > 0, "Calmar ratio should be positive for positive returns"
        
        # Calmar ratio should be finite
        assert np.isfinite(metrics['calmar_ratio']), "Calmar ratio should be finite"


class TestKellyCriterion:
    """Test Kelly Criterion implementation."""
    
    def test_kelly_criterion_correct(self):
        """Test correct Kelly Criterion calculation."""
        # Test case: 60% win rate, $100 avg win, $50 avg loss
        win_rate = 0.6
        avg_win = 100.0
        avg_loss = 50.0
        
        kelly = calculate_kelly_fraction_correct(win_rate, avg_win, avg_loss)
        
        # Expected: (2 * 0.6 - 0.4) / 2 = (1.2 - 0.4) / 2 = 0.4
        expected = (2 * 0.6 - 0.4) / 2
        capped_expected = min(expected, 0.25)
        assert abs(kelly - capped_expected) < 1e-6, f"Kelly fraction wrong: {kelly} (expected {capped_expected})"
    
    def test_kelly_criterion_edge_cases(self):
        """Test Kelly Criterion edge cases."""
        # Zero win rate
        kelly = calculate_kelly_fraction_correct(0.0, 100.0, 50.0)
        assert kelly == 0.0, "Kelly should be 0 for zero win rate"
        
        # 100% win rate
        kelly = calculate_kelly_fraction_correct(1.0, 100.0, 50.0)
        assert kelly == 0.0, "Kelly should be 0 for 100% win rate (invalid)"
        
        # Zero average loss
        kelly = calculate_kelly_fraction_correct(0.6, 100.0, 0.0)
        assert kelly == 0.0, "Kelly should be 0 for zero average loss"
        
        # Negative average loss
        kelly = calculate_kelly_fraction_correct(0.6, 100.0, -50.0)
        assert kelly == 0.0, "Kelly should be 0 for negative average loss"
    
    def test_kelly_criterion_capping(self):
        """Test Kelly Criterion capping at 25%."""
        # High win rate scenario that would exceed 25%
        win_rate = 0.9
        avg_win = 1000.0
        avg_loss = 10.0
        
        kelly = calculate_kelly_fraction_correct(win_rate, avg_win, avg_loss)
        assert kelly <= 0.25, "Kelly should be capped at 25%"
        assert kelly >= 0.0, "Kelly should be non-negative"


class TestPortfolioOptimization:
    """Test portfolio optimization functions."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for optimization."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, (252, 4))  # 4 assets
    
    def test_markowitz_optimization(self, sample_returns):
        """Test Markowitz mean-variance optimization."""
        weights, results = optimize_portfolio_markowitz(sample_returns)
        
        # Check weights sum to 1
        assert abs(np.sum(weights) - 1.0) < 1e-6, "Weights should sum to 1"
        
        # Check weights are non-negative
        assert np.all(weights >= 0), "All weights should be non-negative"
        
        # Check optimization succeeded
        assert results['success'], f"Optimization failed: {results.get('message', 'Unknown error')}"
    
    def test_risk_parity_optimization(self, sample_returns):
        """Test risk parity optimization."""
        weights, results = optimize_portfolio_risk_parity(sample_returns)
        
        # Check weights sum to 1
        assert abs(np.sum(weights) - 1.0) < 1e-6, "Weights should sum to 1"
        
        # Check weights are non-negative
        assert np.all(weights >= 0), "All weights should be non-negative"
        
        # Check optimization succeeded
        assert results['success'], f"Risk parity optimization failed: {results.get('message', 'Unknown error')}"
    
    def test_optimization_constraints(self, sample_returns):
        """Test optimization with constraints."""
        constraints = {'max_weight': 0.5}
        weights, results = optimize_portfolio_markowitz(sample_returns, constraints=constraints)
        
        # Check max weight constraint
        assert np.all(weights <= 0.5), "All weights should be <= 0.5"
        
        # Check optimization succeeded
        assert results['success'], "Constrained optimization should succeed"


class TestBootstrapConfidenceIntervals:
    """Test bootstrap confidence intervals."""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for bootstrap testing."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 100)
    
    def test_bootstrap_confidence_intervals(self, sample_returns):
        """Test bootstrap confidence intervals calculation."""
        ci_results = bootstrap_confidence_intervals(sample_returns, n_bootstrap=100)
        
        # Check all required metrics
        required_metrics = ['sharpe_ci', 'max_dd_ci', 'sortino_ci']
        for metric in required_metrics:
            assert metric in ci_results, f"Missing CI metric: {metric}"
            assert len(ci_results[metric]) == 2, f"CI {metric} should have 2 values"
            assert ci_results[metric][0] < ci_results[metric][1], f"CI {metric} should be ordered"
        
        # Check standard deviations
        std_metrics = ['sharpe_std', 'max_dd_std', 'sortino_std']
        for metric in std_metrics:
            assert metric in ci_results, f"Missing std metric: {metric}"
            assert ci_results[metric] > 0, f"Standard deviation {metric} should be positive"
    
    def test_bootstrap_different_confidence_levels(self, sample_returns):
        """Test bootstrap with different confidence levels."""
        ci_90 = bootstrap_confidence_intervals(sample_returns, confidence_level=0.90)
        ci_95 = bootstrap_confidence_intervals(sample_returns, confidence_level=0.95)
        
        # 90% CI should be narrower than 95% CI
        for metric in ['sharpe_ci', 'max_dd_ci', 'sortino_ci']:
            ci_90_width = ci_90[metric][1] - ci_90[metric][0]
            ci_95_width = ci_95[metric][1] - ci_95[metric][0]
            assert ci_90_width < ci_95_width, f"90% CI should be narrower than 95% CI for {metric}"


class TestStatisticalTests:
    """Test statistical significance tests."""
    
    @pytest.fixture
    def strategy_returns(self):
        """Generate strategy returns."""
        np.random.seed(42)
        return np.random.normal(0.002, 0.02, 100)  # Higher mean return
    
    @pytest.fixture
    def benchmark_returns(self):
        """Generate benchmark returns."""
        np.random.seed(43)
        return np.random.normal(0.001, 0.02, 100)  # Lower mean return
    
    def test_strategy_significance(self, strategy_returns, benchmark_returns):
        """Test strategy significance testing."""
        results = test_strategy_significance(strategy_returns, benchmark_returns)
        
        # Check all required metrics
        required_metrics = [
            't_statistic', 'p_value', 'significant', 'information_ratio',
            'f_statistic', 'f_p_value', 'variance_equal'
        ]
        
        for metric in required_metrics:
            assert metric in results, f"Missing significance metric: {metric}"
        
        # Check p-value is between 0 and 1
        assert 0 <= results['p_value'] <= 1, "P-value should be between 0 and 1"
        
        # Check significance is boolean
        assert isinstance(results['significant'], bool), "Significance should be boolean"
        
        # Check information ratio is finite
        assert np.isfinite(results['information_ratio']), "Information ratio should be finite"


class TestStressTesting:
    """Test stress testing functionality."""
    
    @pytest.fixture
    def portfolio_returns(self):
        """Generate portfolio returns for stress testing."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)
    
    def test_stress_test_scenarios(self, portfolio_returns):
        """Test stress testing under various scenarios."""
        scenarios = {
            'market_crash': -0.05,  # 5% daily loss
            'volatility_spike': -0.02,  # 2% daily loss
            'normal_stress': -0.01   # 1% daily loss
        }
        
        results = stress_test_portfolio(portfolio_returns, scenarios)
        
        # Check all scenarios are tested
        for scenario in scenarios:
            assert scenario in results, f"Missing stress test scenario: {scenario}"
            
            # Check required metrics
            required_metrics = [
                'shock_magnitude', 'stressed_sharpe', 'stressed_max_drawdown',
                'stressed_var_95', 'portfolio_loss'
            ]
            
            for metric in required_metrics:
                assert metric in results[scenario], f"Missing metric {metric} for scenario {scenario}"
        
        # Check that larger shocks result in worse metrics
        crash_results = results['market_crash']
        normal_results = results['normal_stress']
        
        assert crash_results['stressed_max_drawdown'] >= normal_results['stressed_max_drawdown'], \
            "Larger shock should result in larger drawdown"
        
        assert crash_results['portfolio_loss'] <= normal_results['portfolio_loss'], \
            "Larger shock should result in larger loss (more negative)"


class TestMathematicalRigor:
    """Test mathematical rigor and edge cases."""
    
    def test_zero_volatility_handling(self):
        """Test handling of zero volatility scenarios."""
        # Create returns with zero volatility
        returns = np.ones((100, 2)) * 0.001  # Constant returns
        weights = np.array([0.5, 0.5])
        
        metrics = calculate_enhanced_portfolio_metrics(returns, weights)
        
        # Sharpe ratio should be 0 for zero volatility
        assert metrics['sharpe_ratio'] == 0.0, "Sharpe ratio should be 0 for zero volatility"
        
        # Sortino ratio should be 0 for zero volatility
        assert metrics['sortino_ratio'] == 0.0, "Sortino ratio should be 0 for zero volatility"
    
    def test_negative_returns_handling(self):
        """Test handling of consistently negative returns."""
        # Create consistently negative returns
        returns = np.random.normal(-0.01, 0.02, (100, 2))  # Negative mean
        weights = np.array([0.5, 0.5])
        
        metrics = calculate_enhanced_portfolio_metrics(returns, weights)
        
        # All metrics should be finite
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                assert np.isfinite(metric_value), f"Metric {metric_name} should be finite"
    
    def test_single_asset_portfolio(self):
        """Test single asset portfolio."""
        returns = np.random.normal(0.001, 0.02, (100, 1))
        weights = np.array([1.0])
        
        metrics = calculate_enhanced_portfolio_metrics(returns, weights)
        
        # All metrics should be calculated correctly
        assert metrics['total_return'] is not None
        assert metrics['volatility'] > 0
        assert np.isfinite(metrics['sharpe_ratio'])
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create returns with known drawdown
        returns = np.array([0.1, -0.2, 0.05, -0.1, 0.15])  # 20% max drawdown
        
        max_dd = calculate_max_drawdown(returns)
        
        # Should be approximately 24.4%
        assert abs(max_dd - 0.244) < 0.01, f"Max drawdown wrong: {max_dd} (expected ~0.244)"
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        # Create returns with known downside
        returns = np.array([0.1, -0.05, 0.08, -0.02, 0.12])
        
        sortino = calculate_sortino_ratio(returns)
        
        # Should be positive for positive mean return
        assert sortino > 0, "Sortino ratio should be positive for positive returns"
        
        # Should be finite
        assert np.isfinite(sortino), "Sortino ratio should be finite"


# Line count: ~500 lines (comprehensive test coverage)

