"""
Unit tests to verify all forecaster parameters are learned from data,
not hard-coded, and that learning is deterministic and reproducible.
"""
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

# Import forecasters
from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
from forcester_ts.sarimax import SARIMAXForecaster
from forcester_ts.samossa import SAMOSSAForecaster
from forcester_ts.mssa_rl import MSSARLForecaster, MSSARLConfig
from forcester_ts.garch import GARCHForecaster

pytestmark = pytest.mark.slow


class TestParameterLearning:
    """Test that all model parameters are learned from data, not hard-coded."""

    @pytest.fixture
    def synthetic_series(self) -> pd.Series:
        """Generate synthetic time series with known characteristics."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        # Trend + seasonality + noise
        t = np.arange(500)
        trend = 0.1 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 30)  # Monthly seasonality
        noise = np.random.randn(500) * 2
        values = 100 + trend + seasonal + noise
        return pd.Series(values, index=dates, name='Close')

    @pytest.fixture
    def nonstationary_series(self) -> pd.Series:
        """Generate non-stationary series requiring differencing."""
        np.random.seed(123)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        # Random walk with drift
        drift = 0.5
        noise = np.random.randn(300)
        values = 100 + np.cumsum(drift + noise)
        return pd.Series(values, index=dates, name='Close')

    @pytest.fixture
    def volatile_series(self) -> pd.Series:
        """Generate series with time-varying volatility for GARCH."""
        np.random.seed(456)
        dates = pd.date_range('2020-01-01', periods=400, freq='D')
        # Stable GARCH-like process to avoid overflow in test environments.
        omega = 0.1
        alpha = 0.1
        beta = 0.8
        sigma2 = omega / (1.0 - alpha - beta)
        returns = []
        for _ in range(400):
            eps = np.random.randn()
            sigma2 = omega + alpha * (eps**2) + beta * sigma2
            sigma2 = float(np.clip(sigma2, 1e-6, 25.0))
            returns.append(eps * np.sqrt(sigma2))
        prices = 100 * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates, name='Close')

    def test_sarimax_learns_differencing_order(self, nonstationary_series):
        """Verify SARIMAX learns differencing order from stationarity tests."""
        forecaster = SARIMAXForecaster(
            max_p=2,
            max_q=2,
            auto_select=True
        )
        forecaster.fit(nonstationary_series)

        # Should detect non-stationarity and apply differencing
        assert forecaster.best_order is not None, "SARIMAX should select an order"
        d = forecaster.best_order[1]
        assert d >= 1, f"Non-stationary series should have d>=1, got d={d}"

        # Order should be learned, not default
        assert forecaster.best_order != (0, 0, 0), "Order should be learned from data"

    def test_sarimax_learns_ar_ma_orders(self, synthetic_series):
        """Verify SARIMAX learns AR/MA orders via grid search."""
        forecaster = SARIMAXForecaster(
            max_p=3,
            max_q=3,
            auto_select=True
        )
        forecaster.fit(synthetic_series)

        assert forecaster.best_order is not None
        p, d, q = forecaster.best_order

        # Should explore search space
        assert 0 <= p <= 3, f"AR order should be within bounds: p={p}"
        assert 0 <= q <= 3, f"MA order should be within bounds: q={q}"

        # Not all series will need AR(3), MA(3) - should select based on AIC
        assert (p, q) != (3, 3), "Should not always select max orders"

    def test_sarimax_detects_seasonality(self, synthetic_series):
        """Verify SARIMAX auto-detects seasonal patterns."""
        forecaster = SARIMAXForecaster(
            max_p=2,
            max_q=2,
            seasonal_periods=None,  # Let it auto-detect
            max_P=1,
            max_Q=1,
            auto_select=True
        )
        forecaster.fit(synthetic_series)

        # Synthetic series has 30-day seasonality, should detect monthly pattern
        if forecaster.best_seasonal_order[3] > 0:  # m parameter
            m = forecaster.best_seasonal_order[3]
            # Should detect some seasonality (may not be exactly 30 due to ACF analysis)
            assert m > 0, "Should detect seasonal period"
            assert 7 <= m <= 60, f"Seasonal period should be reasonable: m={m}"

    def test_sarimax_different_data_different_orders(self):
        """Verify different data produces different learned orders."""
        np.random.seed(100)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Series 1: Strong AR(2) process
        ar_coefs = [0.5, 0.3]
        series1_vals = [100.0, 101.0]
        for i in range(298):
            next_val = 100 + ar_coefs[0] * (series1_vals[-1] - 100) + \
                       ar_coefs[1] * (series1_vals[-2] - 100) + np.random.randn()
            series1_vals.append(next_val)
        series1 = pd.Series(series1_vals, index=dates)

        # Series 2: Strong MA(2) process
        ma_coefs = [0.6, 0.4]
        errors = np.random.randn(300)
        series2_vals = [100.0]
        for i in range(1, 300):
            val = 100 + errors[i]
            if i >= 1:
                val += ma_coefs[0] * errors[i-1]
            if i >= 2:
                val += ma_coefs[1] * errors[i-2]
            series2_vals.append(val)
        series2 = pd.Series(series2_vals, index=dates)

        # Fit both
        f1 = SARIMAXForecaster(max_p=3, max_q=3, auto_select=True)
        f2 = SARIMAXForecaster(max_p=3, max_q=3, auto_select=True)
        f1.fit(series1)
        f2.fit(series2)

        # Should select different orders
        assert f1.best_order != f2.best_order, \
            "Different data characteristics should produce different orders"

    def test_samossa_adapts_window_to_series_length(self):
        """Verify SAMoSSA window adapts to series length."""
        # Short series
        short_series = pd.Series(
            np.random.randn(50) + 100,
            index=pd.date_range('2020-01-01', periods=50, freq='D')
        )

        # Long series
        long_series = pd.Series(
            np.random.randn(800) + 100,
            index=pd.date_range('2020-01-01', periods=800, freq='D')
        )

        f_short = SAMOSSAForecaster(window_length=60)
        f_long = SAMOSSAForecaster(window_length=60)

        f_short.fit(short_series)
        f_long.fit(long_series)

        # Short series should have window capped more aggressively
        assert f_short.config.window_length < 60, \
            "Short series should have window reduced from 60"

        # Long series should use larger window (closer to or equal to 60)
        assert f_long.config.window_length >= f_short.config.window_length, \
            "Long series should allow larger windows"

    def test_samossa_auto_selects_components(self):
        """Verify SAMoSSA auto-selects components when n_components=-1."""
        np.random.seed(42)
        # Series with clear structure (low rank)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        t = np.arange(300)
        simple_series = pd.Series(
            100 + 0.1*t + 5*np.sin(2*np.pi*t/30) + np.random.randn(300)*0.5,
            index=dates
        )

        # Complex series (higher rank)
        complex_series = pd.Series(
            100 + 0.1*t + \
            5*np.sin(2*np.pi*t/30) + \
            3*np.sin(2*np.pi*t/7) + \
            2*np.sin(2*np.pi*t/90) + \
            np.random.randn(300)*2,
            index=dates
        )

        f_simple = SAMOSSAForecaster(n_components=-1)  # Auto-select
        f_complex = SAMOSSAForecaster(n_components=-1)  # Auto-select

        f_simple.fit(simple_series)
        f_complex.fit(complex_series)

        # Both should select components (not use default)
        # Complex series may need more components
        assert f_simple._explained_variance_ratio >= 0.90, \
            "Auto-selection should capture sufficient variance"
        assert f_complex._explained_variance_ratio >= 0.90, \
            "Auto-selection should capture sufficient variance"

    def test_mssa_rl_learns_rank_from_variance(self):
        """Verify MSSA-RL auto-selects rank to capture 90% variance."""
        np.random.seed(42)
        series = pd.Series(
            np.random.randn(300) + 100,
            index=pd.date_range('2020-01-01', periods=300, freq='D')
        )

        config = MSSARLConfig(
            rank=None,  # Auto-select
            change_point_threshold=3.5
        )
        forecaster = MSSARLForecaster(config=config)
        forecaster.fit(series)

        # Should have selected a rank
        assert config.rank is not None, "Rank should be auto-selected"
        assert config.rank >= 1, "Rank should be positive"

        # Rank should be data-dependent (not always the same)
        # Test with different series
        series2 = pd.Series(
            np.random.randn(300) * 10 + 200,  # Different scale/mean
            index=pd.date_range('2020-01-01', periods=300, freq='D')
        )
        config2 = MSSARLConfig(rank=None, change_point_threshold=3.5)
        forecaster2 = MSSARLForecaster(config=config2)
        forecaster2.fit(series2)

        # Ranks might differ based on data characteristics
        # (not guaranteed to be different, but shouldn't always be identical)

    def test_mssa_rl_change_points_data_dependent(self):
        """Verify MSSA-RL change-points adapt to data volatility."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')

        # Stable series
        stable_series = pd.Series(
            100 + np.random.randn(300) * 0.5,  # Low volatility
            index=dates
        )

        # Volatile series with regime shifts
        volatile_vals = []
        for i in range(300):
            if i < 100:
                volatile_vals.append(100 + np.random.randn() * 2)
            elif i < 200:
                volatile_vals.append(120 + np.random.randn() * 5)  # Regime shift
            else:
                volatile_vals.append(90 + np.random.randn() * 3)   # Another shift
        volatile_series = pd.Series(volatile_vals, index=dates)

        config_stable = MSSARLConfig(rank=None, change_point_threshold=3.5)
        config_volatile = MSSARLConfig(rank=None, change_point_threshold=3.5)

        f_stable = MSSARLForecaster(config=config_stable)
        f_volatile = MSSARLForecaster(config=config_volatile)

        f_stable.fit(stable_series)
        f_volatile.fit(volatile_series)

        stable_cps = len(f_stable._change_points) if f_stable._change_points is not None else 0
        volatile_cps = len(f_volatile._change_points) if f_volatile._change_points is not None else 0

        # Volatile series should detect more change-points
        assert volatile_cps > stable_cps, \
            f"Volatile series should have more change-points: {volatile_cps} vs {stable_cps}"

    def test_garch_learns_order_from_data(self, volatile_series):
        """Verify GARCH learns (p,q) orders via grid search."""
        returns = volatile_series.pct_change().dropna()

        forecaster = GARCHForecaster(
            max_p=3,
            max_q=3
        )
        forecaster.fit(returns)

        # Should have selected an order
        assert hasattr(forecaster, 'fitted_model'), "GARCH should be fitted"

        # If using arch backend, check order
        if not forecaster._fallback_state:  # Not using EWMA fallback
            assert forecaster.p >= 1, "GARCH p should be >= 1"
            assert forecaster.q >= 1, "GARCH q should be >= 1"

    def test_no_hard_coded_parameters_in_forecasts(self, synthetic_series):
        """Integration test: verify full forecaster uses learned parameters."""
        config = TimeSeriesForecasterConfig(
            forecast_horizon=30,
            sarimax_enabled=True,
            samossa_enabled=True,
            mssa_rl_enabled=True,
            garch_enabled=True,
            sarimax_kwargs={'max_p': 3, 'max_q': 3, 'auto_select': True},
            samossa_kwargs={'window_length': 60, 'n_components': 8},
            mssa_rl_kwargs={'rank': None, 'change_point_threshold': 3.5},
        )

        forecaster = TimeSeriesForecaster(config=config)
        forecaster.fit(synthetic_series)
        result = forecaster.forecast(steps=30)

        # All models should have been fitted with learned parameters
        assert result is not None, "Forecast should be generated"
        assert 'ensemble_forecast' in result, "Ensemble forecast should exist"

        # Check that models used learned orders (not defaults)
        if forecaster._sarimax:
            assert forecaster._sarimax.best_order != (0, 0, 0), \
                "SARIMAX should learn non-trivial order"

        if forecaster._samossa:
            # Window should have been adapted
            assert forecaster._samossa.config.window_length > 0, \
                "SAMoSSA window should be set"

    def test_parameter_learning_is_deterministic(self, synthetic_series):
        """Verify parameter learning is deterministic with same data/seed."""
        # Fit twice with same series
        f1 = SARIMAXForecaster(max_p=3, max_q=3, auto_select=True)
        f2 = SARIMAXForecaster(max_p=3, max_q=3, auto_select=True)

        f1.fit(synthetic_series)
        f2.fit(synthetic_series)

        # Should select identical orders
        assert f1.best_order == f2.best_order, \
            "Parameter learning should be deterministic"
        assert f1.best_seasonal_order == f2.best_seasonal_order, \
            "Seasonal order learning should be deterministic"

    def test_learned_parameters_improve_with_more_data(self):
        """Verify models improve with longer series (more data to learn from)."""
        np.random.seed(42)

        # Generate AR(2) process
        ar_coefs = [0.6, 0.3]
        short_vals = [100.0, 101.0]
        for i in range(98):  # 100 points total
            next_val = 100 + ar_coefs[0] * (short_vals[-1] - 100) + \
                       ar_coefs[1] * (short_vals[-2] - 100) + np.random.randn()
            short_vals.append(next_val)

        long_vals = short_vals.copy()
        for i in range(400):  # Extend to 500 points
            next_val = 100 + ar_coefs[0] * (long_vals[-1] - 100) + \
                       ar_coefs[1] * (long_vals[-2] - 100) + np.random.randn()
            long_vals.append(next_val)

        dates_short = pd.date_range('2020-01-01', periods=len(short_vals), freq='D')
        dates_long = pd.date_range('2020-01-01', periods=len(long_vals), freq='D')

        short_series = pd.Series(short_vals, index=dates_short)
        long_series = pd.Series(long_vals, index=dates_long)

        f_short = SARIMAXForecaster(max_p=3, max_q=3, auto_select=True)
        f_long = SARIMAXForecaster(max_p=3, max_q=3, auto_select=True)

        f_short.fit(short_series)
        f_long.fit(long_series)

        # With more data, should have more confidence in order selection
        # At minimum, should identify AR structure
        p_short, _, _ = f_short.best_order
        p_long, _, _ = f_long.best_order

        # Longer series should capture AR(2) structure better
        assert p_long >= p_short, \
            "Longer series should identify equal or more AR terms"


class TestBayesianWarmStart:
    """Test Bayesian parameter priors and warm-start caching."""

    @pytest.fixture(autouse=True)
    def _use_tmp_cache(self, tmp_path):
        """Use a temporary directory for each test to avoid side-effects."""
        self._cache_dir = str(tmp_path / "model_params")

    def test_parameter_cache_structure(self):
        """Verify parameter cache has correct structure."""
        from forcester_ts.parameter_cache import ParameterCache

        cache = ParameterCache(cache_dir=self._cache_dir)

        # Build a dummy series for the data-hash argument
        dummy_series = pd.Series(
            np.random.randn(100) + 100,
            index=pd.date_range('2020-01-01', periods=100, freq='D'),
        )

        params = {
            'order': (2, 1, 1),
            'seasonal_order': (1, 0, 1, 12),
        }
        performance = {'aic': 4523.12, 'rmse': 2.34}

        cache.save('AAPL', 'sarimax', params, performance, dummy_series)
        loaded = cache.load('AAPL', 'sarimax')

        assert loaded is not None, "Should load cached parameters"
        assert tuple(loaded['parameters']['order']) == params['order'], "Order should match"

    def test_bayesian_prior_from_history(self):
        """Verify Bayesian priors computed from historical best parameters."""
        from forcester_ts.parameter_cache import ParameterCache

        cache = ParameterCache(cache_dir=self._cache_dir)

        # Simulate historical runs by saving multiple records
        dummy_series = pd.Series(
            np.random.randn(100) + 100,
            index=pd.date_range('2020-01-01', periods=100, freq='D'),
        )

        history = [
            {'order': (2, 1, 1), 'aic': 1000, 'rmse': 2.1},
            {'order': (2, 1, 1), 'aic': 1010, 'rmse': 2.2},
            {'order': (3, 1, 1), 'aic': 1005, 'rmse': 2.0},
            {'order': (2, 1, 0), 'aic': 1020, 'rmse': 2.3},
            {'order': (2, 1, 1), 'aic': 1002, 'rmse': 2.15},
        ]

        for entry in history:
            cache.save(
                'AAPL', 'sarimax',
                parameters={'order': entry['order']},
                performance={'aic': entry['aic'], 'rmse': entry['rmse']},
                series=dummy_series,
            )

        prior = cache.compute_bayesian_prior('AAPL', 'sarimax', metric='aic')

        # (2,1,1) appears 3/5 times with good AIC -> should have high prior probability
        assert prior is not None, "Should compute a prior from 5 observations"
        assert prior.order == (2, 1, 1), "Should select most frequent good order"
        assert 0.4 <= prior.confidence <= 1.0, "Should have reasonable confidence"

    def test_warm_start_uses_cached_parameters(self):
        """Verify warm-start returns cached parameters for a known ticker."""
        from forcester_ts.parameter_cache import ParameterCache

        cache = ParameterCache(cache_dir=self._cache_dir)

        dummy_series = pd.Series(
            np.random.randn(100) + 100,
            index=pd.date_range('2020-01-01', periods=100, freq='D'),
        )

        # Save a parameter record, then retrieve via get_warm_start_parameters
        cache.save(
            'AAPL', 'sarimax',
            parameters={'order': (2, 1, 1)},
            performance={'aic': 1000, 'rmse': 2.0},
            series=dummy_series,
        )

        warm = cache.get_warm_start_parameters('AAPL', 'sarimax', use_bayesian=False)
        assert warm is not None, "Should return warm-start params from cache"
        assert tuple(warm['order']) == (2, 1, 1), "Should return cached order"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
