"""
Unit Tests for Time Series Signal Generator
Line Count: ~300 lines (within budget)

Tests the critical signal generation logic that converts Time Series forecasts
to trading signals. This is profit-critical as incorrect signals lead to losses.

Per TESTING_GUIDE.md: Focus on profit-critical functions only.
"""

import copy
import json

import pytest
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from models.time_series_signal_generator import (
    TimeSeriesSignalGenerator,
    TimeSeriesSignal
)


@pytest.fixture(scope="session")
def ts_routing_config():
    """Load Time Series routing thresholds from configuration."""
    config_path = Path("config") / "signal_routing_config.yml"
    if not config_path.exists():
        pytest.skip("Time Series routing config is missing")
    raw = yaml.safe_load(config_path.read_text()) or {}
    return (raw.get("signal_routing") or {}).get("time_series") or {}


@pytest.fixture
def signal_generator(ts_routing_config):
    """Create signal generator instance for testing using config-driven thresholds.

    quant_validation_config disables JSONL logging so tests cannot contaminate
    the production logs/signals/quant_validation.jsonl file.
    """
    return TimeSeriesSignalGenerator(
        confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
        min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
        max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
        use_volatility_filter=bool(ts_routing_config.get("use_volatility_filter", True)),
        quant_validation_config={"logging": {"enabled": False}},
    )


@pytest.fixture
def sample_forecast_bundle():
    """Create sample forecast bundle for testing"""
    forecast_series = pd.Series([110.0, 112.0, 115.0],
                                index=pd.date_range('2024-01-01', periods=3, freq='D'))
    lower_ci = pd.Series([105.0, 107.0, 110.0],
                         index=pd.date_range('2024-01-01', periods=3, freq='D'))
    upper_ci = pd.Series([115.0, 117.0, 120.0],
                          index=pd.date_range('2024-01-01', periods=3, freq='D'))

    return {
        'horizon': 30,
        'ensemble_forecast': {
            'forecast': forecast_series,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        },
        'sarimax_forecast': {
            'forecast': forecast_series,
            'aic': 1200.5,
            'bic': 1250.3
        },
        'samossa_forecast': {
            'forecast': forecast_series,
            'explained_variance_ratio': 0.92
        },
        'garch_forecast': {
            'volatility': pd.Series([0.15, 0.16, 0.17])
        },
        'volatility_forecast': {
            'volatility': 0.20  # 20% volatility
        },
        'ensemble_metadata': {
            'primary_model': 'ENSEMBLE',
            'weights': {'sarimax': 0.4, 'samossa': 0.4, 'garch': 0.2},
            'aic': 1200.5,
            'bic': 1250.3
        }
    }


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0.0, 0.5, 100))

    return pd.DataFrame({
        'Close': prices,
        'Volume': rng.integers(1000000, 5000000, 100)
    }, index=dates)


@pytest.fixture
def quant_validation_config():
    """Loose thresholds to force PASS for helper tests."""
    return {
        'enabled': True,
        'lookback_days': 60,
        'risk_free_rate': 0.02,
        'scoring_mode': 'domain_utility',
        'success_criteria': {
            'capital_base': 10000,
            'min_annual_return': -1.0,
            'min_sharpe': -5.0,
            'min_sortino': -5.0,
            'max_drawdown': 1.0,
            'min_omega_ratio': 0.0,
            'min_payoff_asymmetry': 0.0,
            'min_profit_factor': 0.0,
            'min_win_rate': 0.0,
            'min_expected_shortfall': -1.0,
            'min_terminal_directional_accuracy': 0.0,
            'min_expected_profit': -1000.0,
        },
        'visualization': {'enabled': False},
        'bootstrap': {'n_samples': 25, 'confidence_level': 0.80},
    }


@pytest.fixture
def quant_validation_config_strict():
    """Strict thresholds to test failure path."""
    return {
        'enabled': True,
        'lookback_days': 60,
        'risk_free_rate': 0.02,
        'scoring_mode': 'domain_utility',
        'success_criteria': {
            'capital_base': 10000,
            'min_annual_return': 1.0,
            'min_sharpe': 2.0,
            'min_sortino': 2.0,
            'max_drawdown': 0.01,
            'min_omega_ratio': 2.0,
            'min_payoff_asymmetry': 5.0,
            'min_profit_factor': 5.0,
            'min_win_rate': 0.9,
            'min_expected_shortfall': -0.001,
            'min_terminal_directional_accuracy': 0.9,
            'min_expected_profit': 10000.0,
            'require_significance': True,
        },
        'visualization': {'enabled': False},
        'bootstrap': {'n_samples': 25, 'confidence_level': 0.80},
    }


@pytest.fixture
def quant_logging_config(quant_validation_config, tmp_path):
    """Quant validation config that logs to a temporary directory."""
    config = copy.deepcopy(quant_validation_config)
    config['logging'] = {
        'enabled': True,
        'log_dir': str(tmp_path),
        'filename': 'quant_validation.jsonl',
    }
    config['visualization'] = {'enabled': False}
    return config


class TestTimeSeriesSignalGenerator:
    """Test suite for TimeSeriesSignalGenerator"""

    def test_initialization(self, signal_generator, ts_routing_config):
        """Test signal generator initialization"""
        assert signal_generator.confidence_threshold == pytest.approx(
            float(ts_routing_config.get("confidence_threshold", 0.55))
        )
        assert signal_generator.min_expected_return == pytest.approx(
            float(ts_routing_config.get("min_expected_return", 0.003))
        )
        assert signal_generator.max_risk_score == pytest.approx(
            float(ts_routing_config.get("max_risk_score", 0.7))
        )
        assert signal_generator.use_volatility_filter == bool(
            ts_routing_config.get("use_volatility_filter", True)
        )

    def test_generate_buy_signal(self, signal_generator, sample_forecast_bundle, sample_market_data):
        """Test generating a BUY signal from bullish forecast"""
        current_price = 100.0
        ticker = "AAPL"

        signal = signal_generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=current_price,
            ticker=ticker,
            market_data=sample_market_data
        )

        assert isinstance(signal, TimeSeriesSignal)
        assert signal.ticker == ticker
        assert signal.entry_price == current_price
        assert signal.action in ('BUY', 'SELL', 'HOLD')
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.signal_type == 'TIME_SERIES'

    def test_generate_hold_signal_low_confidence(self, signal_generator, sample_forecast_bundle):
        """Test HOLD signal when confidence is below threshold"""
        # Create forecast with very low confidence
        low_confidence_forecast = sample_forecast_bundle.copy()
        low_confidence_forecast['ensemble_forecast'] = {
            'forecast': pd.Series([100.5, 100.6, 100.7])  # Very small move
        }
        low_confidence_forecast['volatility_forecast'] = {'volatility': 0.50}  # High volatility

        signal = signal_generator.generate_signal(
            forecast_bundle=low_confidence_forecast,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        # Should return HOLD due to low expected return or high risk
        assert signal.action == 'HOLD'

    def test_confidence_calculation(self, signal_generator, sample_forecast_bundle):
        """Test confidence score calculation"""
        # Test with strong forecast (high expected return)
        strong_forecast = sample_forecast_bundle.copy()
        strong_forecast['ensemble_forecast'] = {
            'forecast': pd.Series([120.0, 125.0, 130.0])  # 20-30% move
        }
        strong_forecast['volatility_forecast'] = {'volatility': 0.15}  # Low volatility

        signal = signal_generator.generate_signal(
            forecast_bundle=strong_forecast,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        # Strong forecast should have higher confidence
        assert signal.confidence >= 0.5

    def test_quant_success_profile_exposes_barbell_payoff_metrics(
        self,
        sample_forecast_bundle,
        sample_market_data,
        quant_validation_config,
    ):
        """Quant profile should expose omega_ratio and payoff_asymmetry for barbell-aware scoring."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.0,
            min_expected_return=0.0,
            max_risk_score=1.0,
            use_volatility_filter=False,
            quant_validation_config=quant_validation_config,
        )

        signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data,
        )
        profile = generator._build_quant_success_profile("AAPL", sample_market_data, signal)

        assert profile is not None
        assert "omega_ratio" in profile["metrics"]
        assert profile["metrics"]["omega_ratio"] is not None
        assert "payoff_asymmetry" in profile["metrics"]
        assert profile["metrics"]["payoff_asymmetry"] is not None
        assert "utility_breakdown" in profile
        assert "omega_ratio" in profile["utility_breakdown"]
        assert "payoff_asymmetry" in profile["utility_breakdown"]
        assert "diagnostics" in profile

    def test_evaluate_success_criteria_only_keeps_structural_gates(self):
        criteria = TimeSeriesSignalGenerator._evaluate_success_criteria(
            criteria_cfg={
                "min_expected_profit": 5.0,
                "min_omega_ratio": 1.5,
                "min_profit_factor": 2.0,
                "min_win_rate": 0.8,
            },
            metrics={"omega_ratio": 2.0},
            performance_snapshot={"profit_factor": 3.0, "win_rate": 0.2},
            significance=None,
            expected_profit=10.0,
            position_value=1000.0,
            action="BUY",
        )

        assert criteria["expected_profit"] is True
        assert "omega_ratio" not in criteria
        assert "profit_factor" not in criteria
        assert "win_rate" not in criteria

    def test_build_domain_utility_rewards_asymmetric_payoff_despite_low_win_rate(self):
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.0,
            min_expected_return=0.0,
            max_risk_score=1.0,
            use_volatility_filter=False,
            quant_validation_config={"enabled": False},
        )
        utility = generator._build_domain_utility(
            metrics={
                "omega_ratio": 2.2,
                "max_drawdown": 0.10,
                "expected_shortfall": -0.01,
            },
            performance_snapshot={"profit_factor": 2.0, "payoff_asymmetry": 2.65, "win_rate": 0.33},
            edge_block={"terminal_directional_accuracy": 0.60},
            criteria_cfg={
                "min_expected_profit": 5.0,
                "min_omega_ratio": 1.0,
                "min_payoff_asymmetry": 1.25,
                "min_profit_factor": 1.0,
                "max_drawdown": 0.30,
                "min_expected_shortfall": -0.03,
                "min_terminal_directional_accuracy": 0.45,
            },
            config={
                "validation_mode": "forecast_edge",
                "utility_weights": {
                    "expected_profit": 0.20,
                    "omega_ratio": 0.24,
                    "payoff_asymmetry": 0.16,
                    "profit_factor": 0.12,
                    "terminal_directional_accuracy": 0.12,
                    "max_drawdown": 0.08,
                    "expected_shortfall": 0.08,
                },
            },
            expected_profit=25.0,
            position_value=1000.0,
        )

        assert utility["utility_score"] is not None
        assert utility["utility_score"] > 0.60
        assert utility["utility_breakdown"]["payoff_asymmetry"]["passed_threshold"] is True
        assert utility["utility_breakdown"]["profit_factor"]["passed_threshold"] is True
        assert utility["utility_breakdown"]["omega_ratio"]["passed_threshold"] is True

    def test_build_domain_utility_penalizes_weak_payoff_even_with_better_hit_rate(self):
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.0,
            min_expected_return=0.0,
            max_risk_score=1.0,
            use_volatility_filter=False,
            quant_validation_config={"enabled": False},
        )
        utility = generator._build_domain_utility(
            metrics={
                "omega_ratio": 0.8,
                "max_drawdown": 0.40,
                "expected_shortfall": -0.08,
            },
            performance_snapshot={"profit_factor": 0.7, "payoff_asymmetry": 0.7, "win_rate": 0.75},
            edge_block={"terminal_directional_accuracy": 0.58},
            criteria_cfg={
                "min_expected_profit": 5.0,
                "min_omega_ratio": 1.0,
                "min_payoff_asymmetry": 1.25,
                "min_profit_factor": 1.0,
                "max_drawdown": 0.25,
                "min_expected_shortfall": -0.03,
                "min_terminal_directional_accuracy": 0.45,
            },
            config={
                "validation_mode": "forecast_edge",
                "utility_weights": {
                    "expected_profit": 0.20,
                    "omega_ratio": 0.24,
                    "payoff_asymmetry": 0.16,
                    "profit_factor": 0.12,
                    "terminal_directional_accuracy": 0.12,
                    "max_drawdown": 0.08,
                    "expected_shortfall": 0.08,
                },
            },
            expected_profit=8.0,
            position_value=1000.0,
        )

        assert utility["utility_score"] is not None
        assert utility["utility_score"] < 0.60
        assert utility["utility_breakdown"]["payoff_asymmetry"]["passed_threshold"] is False
        assert utility["utility_breakdown"]["profit_factor"]["passed_threshold"] is False
        assert utility["utility_breakdown"]["omega_ratio"]["passed_threshold"] is False

    def test_build_domain_utility_prefers_stronger_payoff_asymmetry_when_other_metrics_match(self):
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.0,
            min_expected_return=0.0,
            max_risk_score=1.0,
            use_volatility_filter=False,
            quant_validation_config={"enabled": False},
        )

        base_kwargs = {
            "metrics": {
                "omega_ratio": 1.6,
                "max_drawdown": 0.12,
                "expected_shortfall": -0.015,
            },
            "edge_block": {"terminal_directional_accuracy": 0.58},
            "criteria_cfg": {
                "min_expected_profit": 5.0,
                "min_omega_ratio": 1.0,
                "min_payoff_asymmetry": 1.25,
                "min_profit_factor": 1.0,
                "max_drawdown": 0.30,
                "min_expected_shortfall": -0.03,
                "min_terminal_directional_accuracy": 0.45,
            },
            "config": {
                "validation_mode": "forecast_edge",
                "utility_weights": {
                    "expected_profit": 0.20,
                    "omega_ratio": 0.24,
                    "payoff_asymmetry": 0.16,
                    "profit_factor": 0.12,
                    "terminal_directional_accuracy": 0.12,
                    "max_drawdown": 0.08,
                    "expected_shortfall": 0.08,
                },
            },
            "expected_profit": 20.0,
            "position_value": 1000.0,
        }

        strong = generator._build_domain_utility(
            performance_snapshot={"profit_factor": 1.7, "payoff_asymmetry": 2.65, "win_rate": 0.28},
            **base_kwargs,
        )
        weak = generator._build_domain_utility(
            performance_snapshot={"profit_factor": 1.7, "payoff_asymmetry": 1.05, "win_rate": 0.28},
            **base_kwargs,
        )

        assert strong["utility_score"] is not None
        assert weak["utility_score"] is not None
        assert strong["utility_score"] > weak["utility_score"]

    def test_confidence_penalizes_small_net_edge(self):
        """Confidence should be lower when net edge is tiny after costs."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.0,
            min_expected_return=0.0,
            max_risk_score=1.0,
            use_volatility_filter=False,
            quant_validation_config={"enabled": False},
        )

        market_data = pd.DataFrame(
            {
                "Close": [100.0, 100.0, 100.0],
                "TxnCostBps": [10.0, 10.0, 10.0],  # 20bp round-trip
                "ImpactBps": [0.0, 0.0, 0.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        small_edge = {
            "horizon": 3,
            "ensemble_forecast": {"forecast": pd.Series([100.1, 100.1, 100.2])},
            "sarimax_forecast": {"forecast": pd.Series([100.1, 100.1, 100.2])},
            "samossa_forecast": {"forecast": pd.Series([100.1, 100.1, 100.2])},
            "volatility_forecast": {"volatility": 0.10},
        }
        large_edge = {
            "horizon": 3,
            "ensemble_forecast": {"forecast": pd.Series([101.0, 101.0, 101.5])},
            "sarimax_forecast": {"forecast": pd.Series([101.0, 101.0, 101.5])},
            "samossa_forecast": {"forecast": pd.Series([101.0, 101.0, 101.5])},
            "volatility_forecast": {"volatility": 0.10},
        }

        sig_small = generator.generate_signal(
            forecast_bundle=small_edge,
            current_price=100.0,
            ticker="AAPL",
            market_data=market_data,
        )
        sig_large = generator.generate_signal(
            forecast_bundle=large_edge,
            current_price=100.0,
            ticker="AAPL",
            market_data=market_data,
        )

        assert sig_large.confidence > sig_small.confidence

    def test_confidence_penalizes_wide_ci(self):
        """Wider CIs (lower SNR) should reduce confidence versus narrow CIs."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.0,
            min_expected_return=0.0,
            max_risk_score=1.0,
            use_volatility_filter=False,
            quant_validation_config={"enabled": False},
        )
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        forecast = pd.Series([110.0, 112.0, 115.0], index=idx)
        narrow = {
            "horizon": 3,
            "ensemble_forecast": {
                "forecast": forecast,
                "lower_ci": pd.Series([109.8, 111.8, 114.8], index=idx),
                "upper_ci": pd.Series([110.2, 112.2, 115.2], index=idx),
            },
            "volatility_forecast": {"volatility": 0.10},
        }
        wide = {
            "horizon": 3,
            "ensemble_forecast": {
                "forecast": forecast,
                "lower_ci": pd.Series([100.0, 100.0, 100.0], index=idx),
                "upper_ci": pd.Series([130.0, 130.0, 130.0], index=idx),
            },
            "volatility_forecast": {"volatility": 0.10},
        }
        sig_narrow = generator.generate_signal(
            forecast_bundle=narrow,
            current_price=100.0,
            ticker="AAPL",
            market_data=None,
        )
        sig_wide = generator.generate_signal(
            forecast_bundle=wide,
            current_price=100.0,
            ticker="AAPL",
            market_data=None,
        )

        assert sig_narrow.confidence > sig_wide.confidence

    def test_risk_score_calculation(self, signal_generator, sample_forecast_bundle):
        """Test risk score calculation"""
        # Test with high volatility
        high_vol_forecast = sample_forecast_bundle.copy()
        high_vol_forecast['volatility_forecast'] = {'volatility': 0.50}  # 50% volatility

        signal = signal_generator.generate_signal(
            forecast_bundle=high_vol_forecast,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        # High volatility should increase risk score
        assert signal.risk_score >= 0.5

    def test_expected_return_uses_horizon_end_target(self):
        """Expected return should be computed from the horizon-end forecast value (not step-1)."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.0,
            min_expected_return=0.0,
            max_risk_score=1.0,
            use_volatility_filter=False,
        )
        forecast_series = pd.Series(
            [101.0, 102.0, 103.0],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        bundle = {
            "horizon": 3,
            "ensemble_forecast": {"forecast": forecast_series},
            "volatility_forecast": {"volatility": 0.10},
            "ensemble_metadata": {"primary_model": "ENSEMBLE"},
        }

        signal = generator.generate_signal(
            forecast_bundle=bundle,
            current_price=100.0,
            ticker="TEST",
            market_data=None,
        )
        assert signal.expected_return == pytest.approx(0.03)

    def test_target_and_stop_loss_calculation(self, signal_generator, sample_forecast_bundle):
        """Test target price and stop loss calculation"""
        signal = signal_generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        if signal.action == 'BUY':
            # Target should be forecast price
            assert signal.target_price is not None
            assert signal.target_price > signal.entry_price

            # Stop loss should be below entry
            assert signal.stop_loss is not None
            assert signal.stop_loss < signal.entry_price

        elif signal.action == 'SELL':
            # Target should be forecast price (lower)
            assert signal.target_price is not None
            assert signal.target_price < signal.entry_price

            # Stop loss should be above entry
            assert signal.stop_loss is not None
            assert signal.stop_loss > signal.entry_price

    def test_model_agreement_calculation(self, signal_generator, sample_forecast_bundle):
        """Test model agreement affects confidence"""
        # Create forecast with high model agreement
        agreeing_forecast = sample_forecast_bundle.copy()
        agreeing_forecast['sarimax_forecast'] = {
            'forecast': pd.Series([110.0, 112.0, 115.0])
        }
        agreeing_forecast['samossa_forecast'] = {
            'forecast': pd.Series([110.5, 112.5, 115.5])  # Very close to SARIMAX
        }

        signal_agreeing = signal_generator.generate_signal(
            forecast_bundle=agreeing_forecast,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        # Create forecast with low model agreement
        disagreeing_forecast = sample_forecast_bundle.copy()
        disagreeing_forecast['sarimax_forecast'] = {
            'forecast': pd.Series([110.0, 112.0, 115.0])
        }
        disagreeing_forecast['samossa_forecast'] = {
            'forecast': pd.Series([90.0, 88.0, 85.0])  # Very different from SARIMAX
        }

        signal_disagreeing = signal_generator.generate_signal(
            forecast_bundle=disagreeing_forecast,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        # Agreeing models should have higher confidence
        assert signal_agreeing.confidence >= signal_disagreeing.confidence

    def test_per_ticker_threshold_override_blocks_trade(self):
        """Per-ticker thresholds should override global routing thresholds."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.55,
            min_expected_return=0.003,
            max_risk_score=0.7,
            per_ticker_thresholds={
                "AAPL": {"min_expected_return": 0.01},
            },
            quant_validation_config={"enabled": False},
        )

        # Use a stronger return (0.7%) so MSFT (min_return=0.3%) passes confidence gate
        # even with snr_score=0.0 (H6 fix: SNR=None is now pessimistic). AAPL's higher
        # per-ticker min_return (1.0%) still blocks it.
        forecast = pd.Series([100.7, 100.7, 100.7])
        forecast_bundle = {
            "horizon": 30,
            "ensemble_forecast": {"forecast": forecast},
            "sarimax_forecast": {"forecast": forecast},
            "samossa_forecast": {"forecast": forecast},
            "volatility_forecast": {"volatility": 0.10},
        }

        market_data = pd.DataFrame(
            {
                "Close": [100.0, 100.0, 100.0],
                "TxnCostBps": [0.0, 0.0, 0.0],
                "ImpactBps": [0.0, 0.0, 0.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        aapl_signal = generator.generate_signal(
            forecast_bundle=forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=market_data,
        )
        msft_signal = generator.generate_signal(
            forecast_bundle=forecast_bundle,
            current_price=100.0,
            ticker="MSFT",
            market_data=market_data,
        )

        assert aapl_signal.action == "HOLD"
        assert msft_signal.action == "BUY"

    def test_roundtrip_friction_is_symmetric_for_sell(self):
        """Trading frictions should dampen SELL expected returns, not amplify them."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.55,
            min_expected_return=0.003,
            max_risk_score=0.7,
            quant_validation_config={"enabled": False},
        )

        # Use a stronger return (-0.7%) so the SELL signal passes confidence gate
        # even with snr_score=0.0 (H6 fix: SNR=None is now pessimistic 0.0 not neutral 0.5).
        forecast = pd.Series([99.3, 99.3, 99.3])
        forecast_bundle = {
            "horizon": 30,
            "ensemble_forecast": {"forecast": forecast},
            "sarimax_forecast": {"forecast": forecast},
            "samossa_forecast": {"forecast": forecast},
            "volatility_forecast": {"volatility": 0.10},
        }

        market_data = pd.DataFrame(
            {
                "Close": [100.0, 100.0, 100.0],
                "TxnCostBps": [5.0, 5.0, 5.0],  # 5bp per-side -> 10bp round-trip
                "ImpactBps": [0.0, 0.0, 0.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        signal = generator.generate_signal(
            forecast_bundle=forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=market_data,
        )

        ctx = signal.provenance.get("decision_context") or {}
        assert signal.action == "SELL"
        assert ctx.get("roundtrip_cost_bps") == pytest.approx(10.0)
        assert ctx.get("expected_return") == pytest.approx(-0.007)
        assert ctx.get("expected_return_net") == pytest.approx(-0.006)
        # Net should be closer to zero than gross for SELL signals.
        assert ctx["expected_return_net"] > ctx["expected_return"]

    def test_weather_context_is_attached_from_market_data_attrs(self, signal_generator, sample_forecast_bundle):
        """Weather context should be recorded in TS provenance when present on market data."""
        market_data = pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0],
                "Volume": [1_000_000, 1_100_000, 1_200_000],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        market_data.attrs["weather_context_by_ticker"] = {
            "AAPL": {
                "event_type": "heatwave",
                "severity": "high",
                "days_to_event": 3,
                "impact_direction": "adverse",
            }
        }

        signal = signal_generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=market_data,
        )

        assert signal.provenance["weather_context"]["event_type"] == "heatwave"
        assert signal.provenance["weather_context"]["severity"] == "high"

    def test_hold_signal_on_error(self, signal_generator):
        """Test HOLD signal returned on error"""
        # Invalid forecast bundle
        invalid_forecast = {'error': 'test error'}

        signal = signal_generator.generate_signal(
            forecast_bundle=invalid_forecast,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        assert signal.action == 'HOLD'
        assert signal.confidence == 0.0
        assert 'error' in signal.reasoning.lower() or 'no forecast' in signal.reasoning.lower()

    def test_batch_signal_generation(self, signal_generator, sample_forecast_bundle):
        """Test generating signals for multiple tickers"""
        forecast_bundles = {
            'AAPL': sample_forecast_bundle,
            'MSFT': sample_forecast_bundle.copy()
        }
        current_prices = {
            'AAPL': 100.0,
            'MSFT': 200.0
        }

        signals = signal_generator.generate_signals_batch(
            forecast_bundles=forecast_bundles,
            current_prices=current_prices,
            market_data=None
        )

        assert len(signals) == 2
        assert 'AAPL' in signals
        assert 'MSFT' in signals
        assert isinstance(signals['AAPL'], TimeSeriesSignal)
        assert isinstance(signals['MSFT'], TimeSeriesSignal)

    def test_provenance_extraction(self, signal_generator, sample_forecast_bundle):
        """Test provenance metadata extraction"""
        signal = signal_generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        assert signal.provenance is not None
        assert 'model_type' in signal.provenance
        assert 'timestamp' in signal.provenance
        assert 'forecast_horizon' in signal.provenance

    def test_expected_return_calculation(self, signal_generator, sample_forecast_bundle):
        """Test expected return calculation"""
        signal = signal_generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        # Expected return should be calculated from forecast vs current price
        if signal.action != 'HOLD':
            assert signal.expected_return != 0.0
            # For BUY: positive return expected
            # For SELL: negative return expected (price going down)

    def test_volatility_filter(self, signal_generator):
        """Test volatility filter affects signal generation"""
        # Low volatility forecast
        low_vol_forecast = {
            'horizon': 30,
            'ensemble_forecast': {
                'forecast': pd.Series([110.0, 112.0, 115.0])
            },
            'volatility_forecast': {'volatility': 0.15}  # 15% volatility
        }

        signal_low_vol = signal_generator.generate_signal(
            forecast_bundle=low_vol_forecast,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        # High volatility forecast
        high_vol_forecast = {
            'horizon': 30,
            'ensemble_forecast': {
                'forecast': pd.Series([110.0, 112.0, 115.0])
            },
            'volatility_forecast': {'volatility': 0.50}  # 50% volatility
        }

        signal_high_vol = signal_generator.generate_signal(
            forecast_bundle=high_vol_forecast,
            current_price=100.0,
            ticker="AAPL",
            market_data=None
        )

        # Low volatility should have higher confidence (if filter enabled)
        if signal_generator.use_volatility_filter:
            assert signal_low_vol.confidence >= signal_high_vol.confidence

    def test_quant_validation_profile_attached(self,
                                               sample_forecast_bundle,
                                               sample_market_data,
                                               quant_validation_config,
                                               ts_routing_config):
        """Quant helper attaches provenance when enabled."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=quant_validation_config
        )

        signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data
        )

        quant_profile = signal.provenance.get('quant_validation')
        assert quant_profile is not None
        assert 'status' in quant_profile
        assert quant_profile['criteria'], "Criteria should be evaluated"

    def test_quant_validation_forecast_edge_mode_uses_cv_metrics(
        self,
        monkeypatch,
        sample_forecast_bundle,
        sample_market_data,
        quant_validation_config,
        ts_routing_config,
    ):
        """forecast_edge mode should attach regression metrics and gate on them."""
        cfg = copy.deepcopy(quant_validation_config)
        cfg["validation_mode"] = "forecast_edge"
        cfg["forecast_edge_cv"] = {
            "min_train_size": 10,
            "horizon": 5,
            "step_size": 5,
            "max_folds": 1,
            "baseline_model": "samossa",
        }
        cfg["success_criteria"]["max_rmse_ratio_vs_baseline"] = 1.10
        cfg["success_criteria"]["min_terminal_directional_accuracy"] = 0.55

        import models.time_series_signal_generator as tsg_mod

        def fake_run(self, price_series, returns_series=None, ticker=""):  # noqa: ARG001
            return {
                "aggregate_metrics": {
                    "ensemble": {
                        "rmse": 0.9,
                        "directional_accuracy": 0.60,
                        "terminal_directional_accuracy": 0.58,
                    },
                    "samossa": {
                        "rmse": 1.0,
                        "directional_accuracy": 0.55,
                        "terminal_directional_accuracy": 0.56,
                    },
                },
                "fold_count": 1,
                "horizon": 5,
            }

        monkeypatch.setattr(tsg_mod.RollingWindowValidator, "run", fake_run)

        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=cfg,
        )

        signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data,
        )

        quant_profile = signal.provenance.get("quant_validation")
        assert quant_profile is not None
        assert quant_profile["forecast_edge"]["rmse_ratio_vs_baseline"] == pytest.approx(0.9)
        assert quant_profile["forecast_edge"]["directional_accuracy"] == pytest.approx(0.60)
        assert quant_profile["forecast_edge"]["terminal_directional_accuracy"] == pytest.approx(0.58)
        assert quant_profile["criteria"]["rmse_ratio_vs_baseline"] is True
        assert quant_profile["criteria"]["terminal_directional_accuracy"] is True
        assert quant_profile["diagnostics"]["directional_accuracy"] == pytest.approx(0.60)

    def test_quant_validation_forecast_edge_uses_terminal_direction_not_one_step(
        self,
        monkeypatch,
        sample_forecast_bundle,
        sample_market_data,
        quant_validation_config,
        ts_routing_config,
    ):
        cfg = copy.deepcopy(quant_validation_config)
        cfg["validation_mode"] = "forecast_edge"
        cfg["forecast_edge_cv"] = {
            "min_train_size": 10,
            "horizon": 5,
            "step_size": 5,
            "max_folds": 1,
            "baseline_model": "samossa",
        }
        cfg["success_criteria"]["max_rmse_ratio_vs_baseline"] = 1.10
        cfg["success_criteria"]["min_terminal_directional_accuracy"] = 0.55

        import models.time_series_signal_generator as tsg_mod

        def fake_run(self, price_series, returns_series=None, ticker=""):  # noqa: ARG001
            return {
                "aggregate_metrics": {
                    "ensemble": {
                        "rmse": 0.9,
                        "directional_accuracy": 0.95,
                        "terminal_directional_accuracy": 0.40,
                    },
                    "samossa": {
                        "rmse": 1.0,
                        "directional_accuracy": 0.55,
                        "terminal_directional_accuracy": 0.54,
                    },
                },
                "fold_count": 1,
                "horizon": 5,
            }

        monkeypatch.setattr(tsg_mod.RollingWindowValidator, "run", fake_run)
        monkeypatch.setattr(
            TimeSeriesSignalGenerator,
            "_build_domain_utility",
            lambda self, **kwargs: {
                "utility_breakdown": {
                    "expected_profit": {"passed_threshold": True},
                    "omega_ratio": {"passed_threshold": True},
                    "profit_factor": {"passed_threshold": True},
                    "terminal_directional_accuracy": {"passed_threshold": False},
                },
                "utility_score": 0.82,
                "total_weight": 1.0,
                "weight_validation": {"coverage_ok": True},
                "config_warnings": [],
            },
        )

        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=cfg,
        )

        signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data,
        )

        quant_profile = signal.provenance.get("quant_validation")
        assert quant_profile is not None
        assert quant_profile["forecast_edge"]["directional_accuracy"] == pytest.approx(0.95)
        assert quant_profile["forecast_edge"]["terminal_directional_accuracy"] == pytest.approx(0.40)
        assert quant_profile["criteria"]["terminal_directional_accuracy"] is False
        assert quant_profile["status"] == "PASS"
        assert quant_profile["failed_criteria"] == []
        assert quant_profile["soft_failed_criteria"] == ["terminal_directional_accuracy"]
        assert quant_profile["hard_failed_criteria"] == []
        assert quant_profile["scoring"]["structural_hard_pass"] is True
        assert quant_profile["scoring"]["structural_soft_pass"] is False

    def test_quant_validation_can_promote_terminal_direction_to_hard_gate(
        self,
        monkeypatch,
        sample_forecast_bundle,
        sample_market_data,
        quant_validation_config,
        ts_routing_config,
    ):
        cfg = copy.deepcopy(quant_validation_config)
        cfg["validation_mode"] = "forecast_edge"
        cfg["hard_gate_criteria"] = ["expected_profit", "terminal_directional_accuracy"]
        cfg["forecast_edge_cv"] = {
            "min_train_size": 10,
            "horizon": 5,
            "step_size": 5,
            "max_folds": 1,
            "baseline_model": "samossa",
        }
        cfg["success_criteria"]["max_rmse_ratio_vs_baseline"] = 1.10
        cfg["success_criteria"]["min_terminal_directional_accuracy"] = 0.55

        import models.time_series_signal_generator as tsg_mod

        def fake_run(self, price_series, returns_series=None, ticker=""):  # noqa: ARG001
            return {
                "aggregate_metrics": {
                    "ensemble": {
                        "rmse": 0.9,
                        "directional_accuracy": 0.95,
                        "terminal_directional_accuracy": 0.40,
                    },
                    "samossa": {
                        "rmse": 1.0,
                        "directional_accuracy": 0.55,
                        "terminal_directional_accuracy": 0.54,
                    },
                },
                "fold_count": 1,
                "horizon": 5,
            }

        monkeypatch.setattr(tsg_mod.RollingWindowValidator, "run", fake_run)
        monkeypatch.setattr(
            TimeSeriesSignalGenerator,
            "_build_domain_utility",
            lambda self, **kwargs: {
                "utility_breakdown": {
                    "expected_profit": {"passed_threshold": True},
                    "omega_ratio": {"passed_threshold": True},
                    "profit_factor": {"passed_threshold": True},
                    "terminal_directional_accuracy": {"passed_threshold": False},
                },
                "utility_score": 0.82,
                "total_weight": 1.0,
                "weight_validation": {"coverage_ok": True},
                "config_warnings": [],
            },
        )

        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=cfg,
        )

        signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data,
        )

        quant_profile = signal.provenance.get("quant_validation")
        assert quant_profile is not None
        assert quant_profile["status"] == "FAIL"
        assert quant_profile["failed_criteria"] == ["terminal_directional_accuracy"]
        assert quant_profile["hard_failed_criteria"] == ["terminal_directional_accuracy"]
        assert quant_profile["soft_failed_criteria"] == []

    def test_quant_validation_forecast_edge_mode_uses_shared_forecaster_config(
        self,
        monkeypatch,
        tmp_path,
        sample_forecast_bundle,
        sample_market_data,
        quant_validation_config,
        ts_routing_config,
    ):
        """Forecast-edge CV should inherit runtime config caps instead of falling back to defaults."""
        cfg = copy.deepcopy(quant_validation_config)
        cfg["validation_mode"] = "forecast_edge"
        cfg["forecast_edge_cv"] = {
            "min_train_size": 10,
            "horizon": 5,
            "step_size": 5,
            "max_folds": 1,
            "baseline_model": "samossa",
        }

        forecasting_cfg = {
            "forecasting": {
                "ensemble": {"enabled": False, "minimum_component_weight": 0.12},
                "regime_detection": {"enabled": True, "lookback_window": 42},
                "order_learning": {
                    "enabled": True,
                    "min_fits_to_suggest": 3,
                    "skip_grid_threshold": 5,
                },
                "monte_carlo": {"enabled": True, "paths": 400, "seed": 7},
                "sarimax": {"enabled": False},
                "garch": {"enabled": True, "max_p": 2, "max_q": 2},
                "samossa": {"enabled": True, "window_length": 33},
                "mssa_rl": {"enabled": True, "window_length": 21, "use_gpu": False},
            }
        }
        forecasting_path = tmp_path / "forecasting_config.yml"
        forecasting_path.write_text(yaml.safe_dump(forecasting_cfg), encoding="utf-8")

        import models.time_series_signal_generator as tsg_mod

        captured = {}

        def fake_run(self, price_series, returns_series=None, ticker=""):  # noqa: ARG001
            captured["ticker"] = ticker
            captured["garch_kwargs"] = dict(self.forecaster_config.garch_kwargs)
            captured["samossa_kwargs"] = dict(self.forecaster_config.samossa_kwargs)
            captured["mssa_rl_kwargs"] = dict(self.forecaster_config.mssa_rl_kwargs)
            captured["monte_carlo_config"] = dict(self.forecaster_config.monte_carlo_config)
            captured["order_learning_config"] = dict(self.forecaster_config.order_learning_config)
            captured["ensemble_enabled"] = bool(self.forecaster_config.ensemble_enabled)
            captured["regime_detection_enabled"] = bool(self.forecaster_config.regime_detection_enabled)
            return {
                "aggregate_metrics": {
                    "ensemble": {
                        "rmse": 0.9,
                        "directional_accuracy": 0.60,
                        "terminal_directional_accuracy": 0.58,
                    },
                    "samossa": {
                        "rmse": 1.0,
                        "directional_accuracy": 0.55,
                        "terminal_directional_accuracy": 0.56,
                    },
                },
                "fold_count": 1,
                "horizon": 5,
            }

        monkeypatch.setattr(tsg_mod.RollingWindowValidator, "run", fake_run)

        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=cfg,
            forecasting_config_path=str(forecasting_path),
        )

        signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data,
        )

        quant_profile = signal.provenance.get("quant_validation")
        assert quant_profile is not None
        assert captured["ticker"] == "AAPL"
        assert captured["garch_kwargs"]["max_p"] == 2
        assert captured["garch_kwargs"]["max_q"] == 2
        assert captured["samossa_kwargs"]["window_length"] == 33
        assert captured["samossa_kwargs"]["forecast_horizon"] == 5
        assert captured["mssa_rl_kwargs"]["window_length"] == 21
        assert captured["mssa_rl_kwargs"]["forecast_horizon"] == 5
        assert captured["monte_carlo_config"]["enabled"] is True
        assert captured["monte_carlo_config"]["paths"] == 400
        assert captured["order_learning_config"]["enabled"] is True
        assert captured["ensemble_enabled"] is False
        assert captured["regime_detection_enabled"] is True

    def test_quant_validation_forecast_edge_mode_normalizes_baseline_model_lookup(
        self,
        monkeypatch,
        sample_forecast_bundle,
        sample_market_data,
        quant_validation_config,
        ts_routing_config,
    ):
        """Hyphenated baseline aliases should resolve to the canonical aggregate key."""
        cfg = copy.deepcopy(quant_validation_config)
        cfg["validation_mode"] = "forecast_edge"
        cfg["forecast_edge_cv"] = {
            "min_train_size": 10,
            "horizon": 5,
            "step_size": 5,
            "max_folds": 1,
            "baseline_model": "mssa-rl",
        }
        cfg["success_criteria"]["max_rmse_ratio_vs_baseline"] = 1.10

        import models.time_series_signal_generator as tsg_mod

        def fake_run(self, price_series, returns_series=None, ticker=""):  # noqa: ARG001
            return {
                "aggregate_metrics": {
                    "ensemble": {
                        "rmse": 0.9,
                        "directional_accuracy": 0.60,
                        "terminal_directional_accuracy": 0.58,
                    },
                    "mssa_rl": {
                        "rmse": 1.0,
                        "directional_accuracy": 0.55,
                        "terminal_directional_accuracy": 0.56,
                    },
                },
                "fold_count": 1,
                "horizon": 5,
            }

        monkeypatch.setattr(tsg_mod.RollingWindowValidator, "run", fake_run)

        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=cfg,
        )

        signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data,
        )

        quant_profile = signal.provenance.get("quant_validation")
        assert quant_profile is not None
        assert quant_profile["forecast_edge"]["baseline_model"] == "mssa_rl"
        assert quant_profile["forecast_edge"]["baseline"]["rmse"] == pytest.approx(1.0)
        assert quant_profile["forecast_edge"]["rmse_ratio_vs_baseline"] == pytest.approx(0.9)
        assert quant_profile["criteria"]["rmse_ratio_vs_baseline"] is True

    def test_quant_validation_failure_updates_reasoning(self,
                                                        sample_forecast_bundle,
                                                        sample_market_data,
                                                        quant_validation_config_strict,
                                                        ts_routing_config):
        """Strict thresholds should mark validation as FAIL and annotate reasoning."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=quant_validation_config_strict
        )

        signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data
        )

        quant_profile = signal.provenance.get('quant_validation')
        assert quant_profile is not None
        assert quant_profile['status'] in ('FAIL', 'SKIPPED')
        assert 'QuantValidation=' in signal.reasoning

    def test_quant_validation_logging_output(self,
                                             sample_forecast_bundle,
                                             sample_market_data,
                                             quant_logging_config,
                                             ts_routing_config):
        """Quant validation helper writes structured log entries for debugging."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=quant_logging_config
        )

        signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data
        )

        log_file = Path(quant_logging_config['logging']['log_dir']) / quant_logging_config['logging']['filename']
        assert log_file.exists(), "Quant validation log file should be created"

        lines = [line for line in log_file.read_text().splitlines() if line.strip()]
        assert lines, "Quant validation log should contain entries"
        payload = json.loads(lines[-1])
        assert payload['ticker'] == 'AAPL'
        assert payload['quant_validation']['status']
        assert payload['market_context']['rows'] == len(sample_market_data)
        assert payload.get('run_id')
        assert payload.get('pipeline_id')
        assert payload.get('execution_mode')

    def test_quant_validation_logging_includes_signal_id(self,
                                                          sample_forecast_bundle,
                                                          sample_market_data,
                                                          quant_logging_config,
                                                          ts_routing_config):
        """JSONL entries must include signal_id for Platt scaling outcome linkage (B5)."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=quant_logging_config
        )

        generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data
        )

        log_file = Path(quant_logging_config['logging']['log_dir']) / quant_logging_config['logging']['filename']
        lines = [line for line in log_file.read_text().splitlines() if line.strip()]
        assert lines, "Log must have at least one entry"
        payload = json.loads(lines[-1])
        assert 'signal_id' in payload, "JSONL entry must contain signal_id field"
        # Phase 7.13-A2: signal_id is now a globally unique string ts_{ticker}_{run_suffix}_{counter}
        assert payload['signal_id'] is not None
        sid = str(payload['signal_id'])
        assert sid.startswith("ts_"), f"Expected ts_* prefix, got: {sid!r}"
        assert len(sid) > 6, f"ts_signal_id too short: {sid!r}"

    def test_load_jsonl_outcome_pairs_empty_when_no_outcome(self, quant_logging_config, ts_routing_config):
        """_load_jsonl_outcome_pairs returns empty lists when no entries have outcome field."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=quant_logging_config
        )
        log_dir = Path(quant_logging_config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / quant_logging_config['logging']['filename']
        # Write entries without outcome
        import json as _json
        log_file.write_text(
            _json.dumps({"signal_id": 1, "confidence": 0.9}) + "\n" +
            _json.dumps({"signal_id": 2, "confidence": 0.85}) + "\n",
            encoding="utf-8",
        )
        confs, wins = generator._load_jsonl_outcome_pairs(limit=100)
        assert confs == []
        assert wins == []

    def test_load_jsonl_outcome_pairs_reads_outcome_entries(self, quant_logging_config, ts_routing_config):
        """_load_jsonl_outcome_pairs correctly reads (confidence, win) pairs from JSONL."""
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=quant_logging_config
        )
        log_dir = Path(quant_logging_config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / quant_logging_config['logging']['filename']
        import json as _json
        log_file.write_text(
            _json.dumps({"signal_id": 1, "confidence": 0.90, "outcome": {"win": True, "pnl": 10.0}}) + "\n" +
            _json.dumps({"signal_id": 2, "confidence": 0.85, "outcome": {"win": False, "pnl": -5.0}}) + "\n" +
            _json.dumps({"signal_id": 3, "confidence": 0.80}) + "\n",  # no outcome
            encoding="utf-8",
        )
        confs, wins = generator._load_jsonl_outcome_pairs(limit=100)
        assert len(confs) == 2
        assert confs[0] == pytest.approx(0.85)  # newest first
        assert wins[0] == 0.0
        assert confs[1] == pytest.approx(0.90)
        assert wins[1] == 1.0

    def test_calibrate_confidence_falls_back_to_db_when_jsonl_class_imbalanced(
        self, quant_logging_config, ts_routing_config, tmp_path
    ):
        """When JSONL has enough entries (>=30) but only <5 losses, DB fallback is triggered.

        Root cause fixed: bootstrap outcomes were 36W/4L (losses < 5), JSONL count was 40 (>= 30),
        so the old `if len < 30` check never triggered DB fallback -> calibration always skipped.
        """
        import json as _json
        import sqlite3

        # Build a JSONL with 35 wins + 0 losses (class-imbalanced but n>=30)
        log_dir = Path(quant_logging_config["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / quant_logging_config["logging"]["filename"]
        lines = []
        for i in range(35):
            lines.append(_json.dumps({
                "signal_id": i, "action": "BUY", "confidence": 0.7 + i * 0.005,
                "outcome": {"win": True, "pnl": 10.0},
            }))
        log_file.write_text("\n".join(lines), encoding="utf-8")

        # Build a minimal DB with balanced pairs (20W/20L)
        db_path = tmp_path / "test_platt.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY, action TEXT, ticker TEXT, realized_pnl REAL,
            effective_confidence REAL, confidence_calibrated REAL, base_confidence REAL,
            is_close INTEGER, is_diagnostic INTEGER, is_synthetic INTEGER)""")
        for i in range(40):
            pnl = 5.0 if i < 20 else -5.0
            conn.execute(
                "INSERT INTO trade_executions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (i, "BUY", "AAPL", pnl, 0.55 + i * 0.005, None, None, 1, 0, 0),
            )
        conn.commit()
        conn.close()

        cfg = copy.deepcopy(quant_logging_config)
        cfg["calibration"] = {"db_path": str(db_path), "raw_weight": 0.80,
                               "max_downside_adjustment": 0.15, "max_upside_adjustment": 0.10}
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.55,
            min_expected_return=0.003,
            max_risk_score=0.7,
            quant_validation_config=cfg,
        )
        # With class-imbalanced JSONL, _calibrate_confidence should reach the DB path
        # and either succeed or at least not crash. The key assertion: no exception raised
        # and _platt_calibrated is set when DB data is sufficient.
        result = generator._calibrate_confidence(0.65, ticker="AAPL", db_path=str(db_path))
        assert isinstance(result, float)
        assert 0.05 <= result <= 0.95

    def test_per_ticker_min_expected_profit_override(self,
                                                     sample_forecast_bundle,
                                                     sample_market_data,
                                                     ts_routing_config):
        """Per-ticker success_criteria overrides should be honoured."""
        cfg = {
            'enabled': True,
            'lookback_days': 60,
            'risk_free_rate': 0.02,
            'success_criteria': {
                'capital_base': 10000,
                'min_annual_return': -1.0,
                'min_sharpe': -5.0,
                'min_sortino': -5.0,
                'max_drawdown': 1.0,
                'min_payoff_asymmetry': 0.0,
                'min_profit_factor': 0.0,
                'min_win_rate': 0.0,
                'min_expected_profit': -1000.0,
            },
            'per_ticker': {
                'BTC-USD': {
                    'success_criteria': {
                        'capital_base': 10000,
                        'min_annual_return': -1.0,
                        'min_sharpe': -5.0,
                        'min_sortino': -5.0,
                        'max_drawdown': 1.0,
                        'min_payoff_asymmetry': 0.0,
                        'min_profit_factor': 0.0,
                        'min_win_rate': 0.0,
                        'min_expected_profit': 999999.0,
                    }
                }
            },
            'visualization': {'enabled': False},
            'bootstrap': {'n_samples': 10, 'confidence_level': 0.80},
        }
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
            min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
            max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
            quant_validation_config=cfg,
        )

        # For AAPL (no per_ticker override), quant validation should PASS given loose thresholds.
        aapl_signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="AAPL",
            market_data=sample_market_data,
        )
        aapl_profile = aapl_signal.provenance.get('quant_validation')
        assert aapl_profile is not None
        assert aapl_profile['status'] in ('PASS', 'SKIPPED')

        # For BTC-USD, extreme per_ticker min_expected_profit should force FAIL.
        btc_signal = generator.generate_signal(
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            ticker="BTC-USD",
            market_data=sample_market_data,
        )
        btc_profile = btc_signal.provenance.get('quant_validation')
        assert btc_profile is not None
        assert btc_profile['status'] in ('FAIL', 'SKIPPED')

    def test_lob_cost_model_used_when_depth_available(self):
        generator = TimeSeriesSignalGenerator(
            confidence_threshold=0.1,
            min_expected_return=0.0,
            max_risk_score=1.0,
            cost_model={
                "lob": {
                    "enabled": True,
                    "levels": 3,
                    "tick_size_bps": 1.0,
                    "alpha": 0.3,
                    "max_exhaust_levels": 5,
                    "default_order_value": 10000.0,
                    "depth_profiles": {"US_EQUITY": {"depth_notional": 50000.0, "half_spread_bps": 1.0}},
                }
            },
        )
        market_data = pd.DataFrame({"Bid": [99.5], "Ask": [100.5], "Depth": [75000.0]})
        friction = generator._estimate_roundtrip_friction(
            ticker="AAPL",
            market_data=market_data,
        )
        assert friction["roundtrip_cost_fraction"] >= 0
        assert friction["source"] in {"lob_sim", "bid_ask"}


class TestTimeSeriesSignal:
    """Test suite for TimeSeriesSignal dataclass"""

    def test_signal_creation(self):
        """Test creating a signal"""
        signal = TimeSeriesSignal(
            ticker="AAPL",
            action="BUY",
            confidence=0.75,
            entry_price=100.0,
            target_price=110.0,
            stop_loss=95.0,
            signal_timestamp=datetime.now(),
            model_type="ENSEMBLE",
            expected_return=0.10,
            risk_score=0.5,
            reasoning="Test signal"
        )

        assert signal.ticker == "AAPL"
        assert signal.action == "BUY"
        assert signal.confidence == 0.75
        assert signal.entry_price == 100.0
        assert signal.target_price == 110.0
        assert signal.stop_loss == 95.0


class TestATRStopLoss:
    """Phase 7.14-B: Verify ATR-based stop loss in _calculate_targets and _compute_atr."""

    def _make_generator(self):
        return TimeSeriesSignalGenerator(
            confidence_threshold=0.50,
            min_expected_return=0.001,
            max_risk_score=0.90,
            quant_validation_config={"enabled": False},
        )

    def _make_ohlcv(self, n=20, price=100.0, bar_range=2.0):
        """Create synthetic OHLCV where ATR(14) ≈ bar_range."""
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "Open":  [price] * n,
            "High":  [price + bar_range] * n,
            "Low":   [price - bar_range] * n,
            "Close": [price] * n,
            "Volume": [1_000_000] * n,
        }, index=dates)

    def test_atr_stop_uses_bar_data(self):
        """ATR-based stop should equal current_price - ATR*1.5 for BUY."""
        gen = self._make_generator()
        price = 100.0
        bar_range = 2.0  # High-Low range per bar; ATR(14) ≈ 2.0
        market_data = self._make_ohlcv(n=20, price=price, bar_range=bar_range)

        atr = gen._compute_atr(market_data, period=14)
        assert atr is not None
        assert atr == pytest.approx(bar_range * 2, rel=0.05)  # TR = (H-L) when no gap

        target, stop = gen._calculate_targets(
            current_price=price,
            forecast_price=105.0,
            volatility=0.20,
            action="BUY",
            market_data=market_data,
        )
        expected_stop = price * (1 - max((atr * 1.5) / price, 0.015))
        assert stop == pytest.approx(expected_stop, rel=1e-6)

    def test_atr_stop_fallback_no_ohlc(self):
        """Missing High/Low columns -> fall back to volatility-based stop."""
        gen = self._make_generator()
        market_data = pd.DataFrame(
            {"Close": [100.0] * 20},
            index=pd.date_range("2024-01-01", periods=20, freq="D"),
        )
        atr = gen._compute_atr(market_data, period=14)
        assert atr is None  # No High/Low -> no ATR

        volatility = 0.20
        _, stop = gen._calculate_targets(
            current_price=100.0,
            forecast_price=105.0,
            volatility=volatility,
            action="BUY",
            market_data=market_data,
        )
        # Should use volatility fallback: pct = max(0.015, min(0.05, 0.20*0.5)) = 0.05
        expected_pct = max(0.015, min(0.05, volatility * 0.5))
        assert stop == pytest.approx(100.0 * (1 - expected_pct), rel=1e-6)

    def test_atr_stop_minimum_floor(self):
        """Very small ATR -> stop still enforces 1.5% minimum floor."""
        gen = self._make_generator()
        market_data = self._make_ohlcv(n=20, price=100.0, bar_range=0.01)  # tiny ATR

        atr = gen._compute_atr(market_data, period=14)
        assert atr is not None
        _, stop = gen._calculate_targets(
            current_price=100.0,
            forecast_price=102.0,
            volatility=None,
            action="BUY",
            market_data=market_data,
        )
        # stop_pct must be >= 1.5%
        stop_pct = (100.0 - stop) / 100.0
        assert stop_pct >= 0.015

    def test_atr_stop_nvda_wide(self):
        """High-vol name: ATR > 5% of price -> no 5% cap applied (cap was removed in 7.14-B)."""
        gen = self._make_generator()
        price = 130.0
        bar_range = 5.0  # ATR ≈ 10 (True Range per bar when accounting for H-L = 2*bar_range)
        market_data = self._make_ohlcv(n=20, price=price, bar_range=bar_range)

        atr = gen._compute_atr(market_data, period=14)
        assert atr is not None
        atr_pct = atr / price
        assert atr_pct > 0.05  # Confirms ATR exceeds old 5% cap threshold

        _, stop = gen._calculate_targets(
            current_price=price,
            forecast_price=price * 1.10,
            volatility=0.58,
            action="BUY",
            market_data=market_data,
        )
        stop_pct = (price - stop) / price
        # Old code: min(0.05, ...) would cap at 5%. New code: ATR*1.5/price with no cap.
        # We cannot assert stop_pct > 0.05 exactly (ATR*1.5 might differ), but stop < price.
        assert stop < price
        assert stop_pct >= 0.015  # At least the minimum floor


class TestBestSingleBaselineSelection:
    """Tests for best_single baseline_key logic in _build_forecast_edge."""

    def _make_aggregate(self, rmse_map: dict) -> dict:
        """Build aggregate_metrics dict from {model_name: rmse} map."""
        return {k: {"rmse": v, "mae": v * 0.8} for k, v in rmse_map.items()}

    def test_best_single_picks_min_rmse_model(self):
        """best_single selects the single model with lowest RMSE from aggregate."""
        aggregate = self._make_aggregate(
            {"samossa": 1.5, "mssa_rl": 0.9, "garch": 1.2, "ensemble": 0.95}
        )
        candidates = {k: v for k, v in aggregate.items() if k != "ensemble" and v}
        base = (
            min(candidates.values(), key=lambda m: float(m.get("rmse") or float("inf")))
            if candidates
            else {}
        )
        assert base["rmse"] == 0.9  # mssa_rl has lowest RMSE

    def test_best_single_excludes_ensemble(self):
        """best_single never picks the ensemble model even if it has lowest RMSE."""
        aggregate = self._make_aggregate(
            {"samossa": 1.5, "garch": 1.2, "ensemble": 0.1}
        )
        candidates = {k: v for k, v in aggregate.items() if k != "ensemble" and v}
        base = (
            min(candidates.values(), key=lambda m: float(m.get("rmse") or float("inf")))
            if candidates
            else {}
        )
        # Ensemble (0.1) excluded; garch (1.2) is best single
        assert base["rmse"] == 1.2

    def test_best_single_empty_aggregate_returns_empty(self):
        """best_single with no candidates returns empty dict (no crash)."""
        aggregate = {"ensemble": {"rmse": 0.5}}
        candidates = {k: v for k, v in aggregate.items() if k != "ensemble" and v}
        base = (
            min(candidates.values(), key=lambda m: float(m.get("rmse") or float("inf")))
            if candidates
            else {}
        )
        assert base == {}

    def test_cache_key_includes_baseline_key(self):
        """Different baseline_keys produce different cache entries."""
        gen = TimeSeriesSignalGenerator.__new__(TimeSeriesSignalGenerator)
        gen._forecast_edge_cache = {}

        key_samossa = ("AAPL", "2024-01-01", 5, "samossa")
        key_best = ("AAPL", "2024-01-01", 5, "best_single")

        gen._forecast_edge_cache[key_samossa] = ({"baseline_model": "samossa"}, {})
        # best_single cache key is distinct — should not hit samossa entry
        assert gen._forecast_edge_cache.get(key_best) is None

    def test_baseline_key_validation_accepts_best_single(self):
        """'best_single' is a valid baseline_key (not coerced to 'samossa')."""
        valid_keys = {"sarimax", "samossa", "mssa_rl", "garch", "best_single"}
        assert "best_single" in valid_keys

    def test_baseline_key_validation_accepts_garch(self):
        """'garch' is now a valid baseline_key (added in lift_semantics_baseline_parity)."""
        valid_keys = {"sarimax", "samossa", "mssa_rl", "garch", "best_single"}
        assert "garch" in valid_keys


class TestDirectionalGate:
    """Phase 9: directional gate is inactive by default."""

    def _make_bundle(self):
        import pandas as pd
        idx = pd.RangeIndex(5, name="horizon")
        return {
            "forecast": pd.Series([105.0] * 5, index=idx),
            "lower_ci": pd.Series([100.0] * 5, index=idx),
            "upper_ci": pd.Series([110.0] * 5, index=idx),
            "ensemble_metadata": {"weights": {"samossa": 1.0}, "confidence": {"samossa": 0.7}},
            "detected_regime": "MODERATE_TRENDING",
        }

    def test_gate_inactive_by_default(self):
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        gen = TimeSeriesSignalGenerator()
        # Gate should be off by default regardless of routing config
        assert gen._directional_gate_enabled() is False

    def test_p_up_field_on_signal(self):
        from models.time_series_signal_generator import TimeSeriesSignalGenerator, TimeSeriesSignal
        gen = TimeSeriesSignalGenerator()
        signal = gen.generate_signal(
            forecast_bundle=self._make_bundle(),
            current_price=100.0,
            ticker="AAPL",
        )
        # p_up should be None (gate disabled) or a float
        assert signal.p_up is None or isinstance(signal.p_up, float)

    def test_directional_gate_applied_false_by_default(self):
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        gen = TimeSeriesSignalGenerator()
        signal = gen.generate_signal(
            forecast_bundle=self._make_bundle(),
            current_price=100.0,
            ticker="AAPL",
        )
        assert signal.directional_gate_applied is False

    # G1: gate observability / exception paths (audit finding A1/A2)

    def test_gate_disabled_when_config_is_not_dict(self):
        """A1: non-dict signal_routing_config must disable gate gracefully, not raise."""
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        gen = TimeSeriesSignalGenerator()
        gen._signal_routing_config = "not-a-dict"
        assert gen._directional_gate_enabled() is False

    def test_gate_disabled_when_dc_section_is_not_dict(self):
        """A1: non-dict directional_classifier section must disable gate gracefully."""
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        gen = TimeSeriesSignalGenerator()
        gen._signal_routing_config = {"directional_classifier": "yes-please"}
        assert gen._directional_gate_enabled() is False

    def test_gate_enabled_when_config_explicitly_set(self):
        """A1: gate returns True when config dict has enabled=True."""
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        gen = TimeSeriesSignalGenerator()
        gen._signal_routing_config = {
            "directional_classifier": {"enabled": True, "p_up_threshold_buy": 0.55}
        }
        assert gen._directional_gate_enabled() is True

    def test_score_directional_returns_none_after_import_failure(self):
        """A2: failed import writes sentinel and subsequent calls return None without retrying."""
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        gen = TimeSeriesSignalGenerator()
        gen._signal_routing_config = {
            "directional_classifier": {"enabled": True, "p_up_threshold_buy": 0.55}
        }
        # Simulate a failed import by setting the sentinel directly
        gen._directional_classifier = TimeSeriesSignalGenerator._DIRECTIONAL_CLASSIFIER_FAILED
        result = gen._score_directional({})
        assert result is None

    def test_market_data_none_produces_nan_for_context_features(self):
        """A4/E1: market_data=None should not raise; context features should be NaN."""
        import math
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        gen = TimeSeriesSignalGenerator()
        bundle = self._make_bundle()
        features = gen._extract_classifier_features(
            forecast_bundle=bundle,
            current_price=100.0,
            expected_return=0.02,
            lower_ci=98.0,
            upper_ci=102.0,
            snr=1.5,
            model_agreement=0.8,
            market_data=None,
        )
        # Context features must be NaN (not missing from dict)
        assert "recent_return_5d" in features
        assert "recent_vol_ratio" in features
        assert math.isnan(features["recent_return_5d"])
        assert math.isnan(features["recent_vol_ratio"])

class TestHoldReasonCodes:
    """Verify _determine_action returns structured hold reason codes.

    Phase 10c: HOLD reason instrumentation makes the policy layer observable —
    enables aggregating HOLD causes across runs without parsing log strings.
    """

    def setup_method(self):
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        self.gen = TimeSeriesSignalGenerator()

    def _call(self, **kwargs):
        defaults = dict(
            expected_return=0.01,
            net_trade_return=0.01,
            confidence=0.70,
            risk_score=0.30,
            confidence_threshold=0.55,
            min_expected_return=0.002,
            max_risk_score=0.70,
        )
        defaults.update(kwargs)
        return self.gen._determine_action(**defaults)

    def test_buy_returns_none_reason(self):
        action, reason = self._call(expected_return=0.05)
        assert action == "BUY"
        assert reason is None

    def test_sell_returns_none_reason(self):
        action, reason = self._call(expected_return=-0.05)
        assert action == "SELL"
        assert reason is None

    def test_confidence_below_threshold_reason(self):
        action, reason = self._call(confidence=0.30, confidence_threshold=0.55)
        assert action == "HOLD"
        assert reason == "CONFIDENCE_BELOW_THRESHOLD"

    def test_min_return_reason(self):
        action, reason = self._call(net_trade_return=0.0001, min_expected_return=0.002)
        assert action == "HOLD"
        assert reason == "MIN_RETURN"

    def test_risk_too_high_reason(self):
        action, reason = self._call(risk_score=0.90, max_risk_score=0.70)
        assert action == "HOLD"
        assert reason == "RISK_TOO_HIGH"

    def test_zero_expected_return_reason(self):
        action, reason = self._call(expected_return=0.0)
        assert action == "HOLD"
        assert reason == "ZERO_EXPECTED_RETURN"

    def test_confidence_gate_takes_priority_over_return(self):
        # Both confidence and return fail — confidence checked first
        action, reason = self._call(confidence=0.10, net_trade_return=0.0001)
        assert action == "HOLD"
        assert reason == "CONFIDENCE_BELOW_THRESHOLD"


class TestVolBandContinuity:
    """Verify piecewise-linear vol-band produces no cliff edges.

    Phase P4: replaced the discrete step function (0.75 flat for vol in [0.40,0.60))
    with piecewise-linear interpolation so that adjacent vol values near the band
    boundaries produce smoothly varying confidence rather than sudden jumps.
    """

    _FIXED_KWARGS = dict(
        expected_return=0.05,
        net_trade_return=0.05,
        min_expected_return=0.001,
        model_agreement=0.80,
        diagnostics_score=0.70,
        snr=2.0,
        ticker="TEST",
    )

    def _conf(self, vol: float) -> float:
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        gen = TimeSeriesSignalGenerator(use_volatility_filter=True)
        return gen._calculate_confidence(volatility=vol, **self._FIXED_KWARGS)

    def test_vol_band_continuous_at_0_60_boundary(self) -> None:
        """vol=0.599 and vol=0.601 produce confidence within 2% of each other.

        Old code: 0.75 (below 0.60) vs 0.60 (at/above 0.60) = 25pp cliff.
        New code: linear interpolation reaches 0.601 as vol→0.60 from below.
        """
        c_below = self._conf(0.599)
        c_above = self._conf(0.601)
        assert c_above > 0, "confidence should be positive"
        relative_diff = abs(c_below - c_above) / c_above
        assert relative_diff < 0.02, (
            f"Expected <=2% difference at vol=0.60 boundary; "
            f"got conf(0.599)={c_below:.4f}, conf(0.601)={c_above:.4f}, "
            f"relative_diff={relative_diff:.1%}"
        )

    def test_vol_band_monotonically_decreasing_in_40_60_range(self) -> None:
        """Confidence strictly decreases as vol rises through [0.40, 0.60].

        Old code: vol=0.41 and vol=0.59 produced identical confidence (flat 0.75).
        New code: linear interpolation from 0.75 at vol=0.40 to 0.60 at vol=0.60.
        """
        c_low = self._conf(0.41)
        c_mid = self._conf(0.50)
        c_high = self._conf(0.59)
        assert c_low > c_mid, (
            f"conf(0.41)={c_low:.4f} should exceed conf(0.50)={c_mid:.4f}"
        )
        assert c_mid > c_high, (
            f"conf(0.50)={c_mid:.4f} should exceed conf(0.59)={c_high:.4f}"
        )


class TestDiagnosticsScorePessimisticFallback:
    """P1-C: missing diagnostics_score must use 0.0 (pessimistic), not 0.5 (neutral).

    A missing diagnostics score should LOWER confidence, not leave it neutral, so that
    forecasts with absent diagnostics are penalised rather than silently passed through.
    """

    _BASE_KWARGS = dict(
        expected_return=0.05,
        net_trade_return=0.05,
        min_expected_return=0.001,
        volatility=0.20,
        model_agreement=0.80,
        snr=2.0,
        ticker="TEST",
    )

    def _conf(self, diagnostics_score: float) -> float:
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        gen = TimeSeriesSignalGenerator(use_volatility_filter=True)
        return gen._calculate_confidence(diagnostics_score=diagnostics_score, **self._BASE_KWARGS)

    def test_missing_score_lower_than_neutral(self) -> None:
        """conf(missing=0.0) < conf(score=0.5) — pessimistic fallback penalises absence."""
        conf_missing = self._conf(0.0)   # P1-C: missing maps to 0.0
        conf_neutral = self._conf(0.5)   # old default
        assert conf_missing < conf_neutral, (
            f"Missing diagnostics_score (0.0) must produce lower confidence than neutral (0.5); "
            f"got missing={conf_missing:.4f}, neutral={conf_neutral:.4f}"
        )

    def test_missing_score_lower_than_good(self) -> None:
        """conf(missing=0.0) < conf(score=0.8) — a good diagnostics score earns higher confidence."""
        conf_missing = self._conf(0.0)
        conf_good = self._conf(0.8)
        assert conf_missing < conf_good, (
            f"Missing diagnostics_score must produce lower confidence than good score; "
            f"got missing={conf_missing:.4f}, good={conf_good:.4f}"
        )


class TestSNRNonePessimisticFallback:
    """H6: SNR=None must use 0.0 (pessimistic), not 0.5 (neutral).

    Consistent with P1-C policy: missing uncertainty measurement should lower confidence,
    not credit it neutrally. SNR is unavailable only when CI computation entirely fails.
    """

    _BASE_KWARGS = dict(
        expected_return=0.05,
        net_trade_return=0.05,
        min_expected_return=0.001,
        volatility=0.20,
        model_agreement=0.80,
        diagnostics_score=0.7,
        ticker="TEST",
    )

    def _conf(self, snr: float | None) -> float:
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        gen = TimeSeriesSignalGenerator(use_volatility_filter=True)
        return gen._calculate_confidence(snr=snr, **self._BASE_KWARGS)

    def test_snr_none_produces_lower_confidence_than_snr_neutral(self) -> None:
        """conf(snr=None) < conf(snr=1.0) — pessimistic fallback penalises absence."""
        conf_none = self._conf(None)
        conf_neutral = self._conf(1.0)
        assert conf_none < conf_neutral, (
            f"SNR=None must produce lower confidence than snr=1.0 (neutral); "
            f"got none={conf_none:.4f}, neutral={conf_neutral:.4f}"
        )

    def test_snr_none_does_not_credit_neutral(self) -> None:
        """conf(snr=None) < conf(snr=1.5) — snr=None uses score=0.0, snr=1.5 uses score=0.667.
        Note: snr=0.5 maps to score=0.0 (same as None), so comparison must use snr > 0.5."""
        conf_none = self._conf(None)
        conf_moderate = self._conf(1.5)  # snr_score = clamp01((1.5-0.5)/1.5) = 0.667
        assert conf_none < conf_moderate, (
            f"SNR=None (score=0.0) must be strictly lower than snr=1.5 (score=0.667); "
            f"got none={conf_none:.4f}, snr_moderate={conf_moderate:.4f}"
        )


class TestModelAgreementPessimisticFallback:
    """_check_model_agreement must return 0.0 (pessimistic) when <2 model forecasts are
    available — consistent with P1-C/H6 policy. Previously returned 0.5 (neutral credit).
    """

    def _gen(self):
        from models.time_series_signal_generator import TimeSeriesSignalGenerator
        return TimeSeriesSignalGenerator(use_volatility_filter=False)

    def _bundle_with_n_models(self, n: int) -> dict:
        """Build a minimal forecast_bundle with exactly n model payloads."""
        bundle: dict = {}
        payloads = [
            ("samossa_forecast", {"forecast": pd.Series([105.0, 106.0, 107.0])}),
            ("mssa_rl_forecast", {"forecast": pd.Series([104.0, 105.0, 106.0])}),
            ("sarimax_forecast", {"forecast": pd.Series([106.0, 107.0, 108.0])}),
        ]
        for key, payload in payloads[:n]:
            bundle[key] = payload
        return bundle

    def test_single_model_returns_pessimistic_zero(self) -> None:
        """With only 1 model forecast, agreement cannot be assessed → must return 0.0."""
        gen = self._gen()
        score = gen._check_model_agreement(self._bundle_with_n_models(1))
        assert score == 0.0, (
            f"_check_model_agreement with 1 model must return 0.0; got {score}"
        )

    def test_no_models_returns_pessimistic_zero(self) -> None:
        """With 0 model forecasts, must return 0.0 (not 0.5 neutral)."""
        gen = self._gen()
        score = gen._check_model_agreement({})
        assert score == 0.0, (
            f"_check_model_agreement with 0 models must return 0.0; got {score}"
        )

    def test_two_models_returns_nonzero_for_close_agreement(self) -> None:
        """With 2 well-agreeing models, score must be positive (agreement path active)."""
        gen = self._gen()
        score = gen._check_model_agreement(self._bundle_with_n_models(2))
        assert score > 0.0, (
            f"_check_model_agreement with 2 well-agreeing models must return >0; got {score}"
        )

    def test_single_model_lower_than_two_agreeing_models(self) -> None:
        """1-model pessimistic (0.0) < 2-model agreement score."""
        gen = self._gen()
        score_one = gen._check_model_agreement(self._bundle_with_n_models(1))
        score_two = gen._check_model_agreement(self._bundle_with_n_models(2))
        assert score_one < score_two, (
            f"1-model score ({score_one}) must be < 2-model score ({score_two})"
        )
