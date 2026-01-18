import numpy as np

from etl.statistical_tests import BootstrapIntervals, StatisticalTestSuite


def test_strategy_significance_detects_edge():
    suite = StatisticalTestSuite()
    rng = np.random.default_rng(42)
    benchmark = rng.normal(0.0005, 0.01, size=252)
    strategy = benchmark + rng.normal(0.0008, 0.0005, size=252)

    result = suite.test_strategy_significance(strategy, benchmark)

    assert result["p_value"] < 0.05
    assert result["significant"] is True
    assert result["information_ratio"] > 0.0


def test_autocorrelation_metrics_shape():
    suite = StatisticalTestSuite()
    returns = np.linspace(-0.005, 0.005, num=300)

    metrics = suite.test_autocorrelation(returns, lags=5)

    assert "ljung_box_stat" in metrics
    assert "ljung_box_p_value" in metrics
    assert "durbin_watson" in metrics
    assert 0.0 <= metrics["ljung_box_p_value"] <= 1.0


def test_bootstrap_validation_generates_intervals():
    suite = StatisticalTestSuite()
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, size=252)

    intervals = suite.bootstrap_validation(
        returns, n_bootstrap=200, confidence_level=0.90, random_state=123
    )

    assert isinstance(intervals, BootstrapIntervals)
    assert len(intervals.sharpe_ratio) == 2
    assert len(intervals.max_drawdown) == 2
    assert intervals.samples == 200
    assert intervals.confidence_level == 0.90
    assert intervals.sharpe_ratio[0] <= intervals.sharpe_ratio[1]
    assert intervals.max_drawdown[0] <= intervals.max_drawdown[1]
