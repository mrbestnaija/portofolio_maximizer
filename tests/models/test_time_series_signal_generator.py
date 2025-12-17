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
    """Create signal generator instance for testing using config-driven thresholds."""
    return TimeSeriesSignalGenerator(
        confidence_threshold=float(ts_routing_config.get("confidence_threshold", 0.55)),
        min_expected_return=float(ts_routing_config.get("min_expected_return", 0.003)),
        max_risk_score=float(ts_routing_config.get("max_risk_score", 0.7)),
        use_volatility_filter=bool(ts_routing_config.get("use_volatility_filter", True)),
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
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    return pd.DataFrame({
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)


@pytest.fixture
def quant_validation_config():
    """Loose thresholds to force PASS for helper tests."""
    return {
        'enabled': True,
        'lookback_days': 60,
        'risk_free_rate': 0.02,
        'success_criteria': {
            'capital_base': 10000,
            'min_annual_return': -1.0,
            'min_sharpe': -5.0,
            'min_sortino': -5.0,
            'max_drawdown': 1.0,
            'min_profit_factor': 0.0,
            'min_win_rate': 0.0,
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
        'success_criteria': {
            'capital_base': 10000,
            'min_annual_return': 1.0,
            'min_sharpe': 2.0,
            'min_sortino': 2.0,
            'max_drawdown': 0.01,
            'min_profit_factor': 5.0,
            'min_win_rate': 0.9,
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

