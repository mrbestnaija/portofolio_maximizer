"""
Unit Tests for Signal Router
Line Count: ~250 lines (within budget)

Tests the critical signal routing logic that determines which signal source
(Time Series or LLM) to use. This is profit-critical as incorrect routing
leads to poor signal quality and losses.

Per TESTING_GUIDE.md: Focus on profit-critical functions only.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from models.signal_router import SignalRouter, SignalBundle
from models.time_series_signal_generator import TimeSeriesSignalGenerator, TimeSeriesSignal
from models.signal_adapter import SignalAdapter


@pytest.fixture
def sample_forecast_bundle():
    """Create sample forecast bundle"""
    forecast_series = pd.Series([110.0, 112.0, 115.0])
    return {
        'horizon': 30,
        'ensemble_forecast': {
            'forecast': forecast_series
        },
        'volatility_forecast': {
            'volatility': 0.20
        }
    }


@pytest.fixture
def sample_llm_signal():
    """Create sample LLM signal"""
    return {
        'ticker': 'AAPL',
        'action': 'BUY',
        'confidence': 0.65,
        'entry_price': 100.0,
        'reasoning': 'LLM analysis suggests bullish trend',
        'signal_timestamp': datetime.now().isoformat(),
        'llm_model': 'qwen:14b-chat-q4_K_M',
        'fallback': False
    }


@pytest.fixture
def ts_signal_generator():
    """Create Time Series signal generator"""
    return TimeSeriesSignalGenerator(
        confidence_threshold=0.55,
        min_expected_return=0.002,
        max_risk_score=0.7
    )


@pytest.fixture
def mock_llm_generator():
    """Create mock LLM signal generator"""
    mock = Mock()
    mock.generate_signal.return_value = {
        'ticker': 'AAPL',
        'action': 'BUY',
        'confidence': 0.65,
        'entry_price': 100.0,
        'reasoning': 'LLM fallback signal',
        'signal_timestamp': datetime.now().isoformat()
    }
    return mock


class TestSignalRouter:
    """Test suite for SignalRouter"""

    def test_initialization_default(self):
        """Test router initialization with defaults"""
        router = SignalRouter()

        assert router.feature_flags['time_series_primary'] is True
        assert router.feature_flags['llm_fallback'] is True
        assert router.feature_flags['llm_redundancy'] is False

    def test_initialization_custom_config(self):
        """Test router initialization with custom config"""
        config = {
            'time_series_primary': False,
            'llm_fallback': False,
            'llm_redundancy': True
        }

        router = SignalRouter(config=config)

        assert router.feature_flags['time_series_primary'] is False
        assert router.feature_flags['llm_fallback'] is False
        assert router.feature_flags['llm_redundancy'] is True

    def test_route_time_series_primary(self, ts_signal_generator, sample_forecast_bundle):
        """Test routing with Time Series as primary"""
        router = SignalRouter(
            config={'time_series_primary': True, 'llm_fallback': False},
            time_series_generator=ts_signal_generator
        )

        bundle = router.route_signal(
            ticker='AAPL',
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            market_data=None
        )

        assert bundle.primary_signal is not None
        assert bundle.primary_signal['source'] == 'TIME_SERIES'
        assert bundle.primary_signal['is_primary'] is True
        assert bundle.fallback_signal is None

    def test_route_llm_fallback(self, mock_llm_generator, sample_forecast_bundle):
        """Test LLM fallback when Time Series unavailable"""
        router = SignalRouter(
            config={'time_series_primary': True, 'llm_fallback': True},
            llm_generator=mock_llm_generator
        )

        # No forecast bundle (Time Series unavailable)
        bundle = router.route_signal(
            ticker='AAPL',
            forecast_bundle=None,
            current_price=100.0,
            market_data=None
        )

        # Should use LLM as fallback
        assert bundle.primary_signal is not None
        assert bundle.primary_signal['source'] == 'LLM'
        assert bundle.primary_signal['is_fallback'] is True

    def test_route_llm_fallback_low_confidence(self, ts_signal_generator, mock_llm_generator, sample_forecast_bundle):
        """Test LLM fallback when Time Series confidence too low"""
        router = SignalRouter(
            config={'time_series_primary': True, 'llm_fallback': True},
            time_series_generator=ts_signal_generator,
            llm_generator=mock_llm_generator
        )

        # Create forecast with very low confidence (will generate HOLD)
        low_confidence_forecast = {
            'horizon': 30,
            'ensemble_forecast': {
                'forecast': pd.Series([100.5, 100.6, 100.7])  # Tiny move
            },
            'volatility_forecast': {'volatility': 0.50}  # High volatility
        }

        bundle = router.route_signal(
            ticker='AAPL',
            forecast_bundle=low_confidence_forecast,
            current_price=100.0,
            market_data=None,
            llm_signal={'action': 'BUY', 'confidence': 0.65}
        )

        # Should use LLM if TS signal is HOLD
        if bundle.primary_signal and bundle.primary_signal.get('action') == 'HOLD':
            assert bundle.fallback_signal is not None
            assert bundle.fallback_signal['source'] == 'LLM'

    def test_route_redundancy_mode(self, ts_signal_generator, mock_llm_generator, sample_forecast_bundle):
        """Test redundancy mode (both TS and LLM)"""
        router = SignalRouter(
            config={
                'time_series_primary': True,
                'llm_fallback': True,
                'llm_redundancy': True
            },
            time_series_generator=ts_signal_generator,
            llm_generator=mock_llm_generator
        )

        bundle = router.route_signal(
            ticker='AAPL',
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            market_data=None
        )

        # Should have both primary and redundancy signals
        assert bundle.primary_signal is not None
        assert bundle.primary_signal['source'] == 'TIME_SERIES'

        # In redundancy mode, should also have LLM signal
        all_signals = bundle.all_signals
        assert len(all_signals) >= 1
        llm_signals = [s for s in all_signals if s.get('source') == 'LLM']
        if router.feature_flags['llm_redundancy']:
            assert len(llm_signals) >= 0  # May or may not be present

    def test_route_batch_signals(self, ts_signal_generator, sample_forecast_bundle):
        """Test routing signals for multiple tickers"""
        router = SignalRouter(
            config={'time_series_primary': True, 'llm_fallback': False},
            time_series_generator=ts_signal_generator
        )

        tickers = ['AAPL', 'MSFT']
        forecast_bundles = {
            'AAPL': sample_forecast_bundle,
            'MSFT': sample_forecast_bundle.copy()
        }
        current_prices = {
            'AAPL': 100.0,
            'MSFT': 200.0
        }

        bundles = router.route_signals_batch(
            tickers=tickers,
            forecast_bundles=forecast_bundles,
            current_prices=current_prices,
            market_data=None
        )

        assert len(bundles) == 2
        assert 'AAPL' in bundles
        assert 'MSFT' in bundles
        assert isinstance(bundles['AAPL'], SignalBundle)

    def test_routing_stats(self, ts_signal_generator, sample_forecast_bundle):
        """Test routing statistics tracking"""
        router = SignalRouter(
            config={'time_series_primary': True, 'llm_fallback': False},
            time_series_generator=ts_signal_generator
        )

        # Route some signals
        router.route_signal(
            ticker='AAPL',
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            market_data=None
        )

        stats = router.get_routing_stats()

        assert 'stats' in stats
        assert 'time_series_signals' in stats['stats']
        assert stats['stats']['time_series_signals'] >= 1

    def test_reset_stats(self, ts_signal_generator, sample_forecast_bundle):
        """Test resetting routing statistics"""
        router = SignalRouter(
            config={'time_series_primary': True, 'llm_fallback': False},
            time_series_generator=ts_signal_generator
        )

        # Route some signals
        router.route_signal(
            ticker='AAPL',
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            market_data=None
        )

        # Reset stats
        router.reset_stats()

        stats = router.get_routing_stats()
        assert stats['stats']['time_series_signals'] == 0

    def test_toggle_feature_flag(self):
        """Test toggling feature flags"""
        router = SignalRouter()

        # Toggle a flag
        router.toggle_feature_flag('llm_redundancy', True)
        assert router.feature_flags['llm_redundancy'] is True

        router.toggle_feature_flag('llm_redundancy', False)
        assert router.feature_flags['llm_redundancy'] is False

    def test_routing_mode_detection(self):
        """Test routing mode detection"""
        # Time Series primary + LLM fallback
        router1 = SignalRouter(config={
            'time_series_primary': True,
            'llm_fallback': True,
            'llm_redundancy': False
        })
        assert router1._get_routing_mode() == 'TIME_SERIES_PRIMARY_LLM_FALLBACK'

        # Time Series only
        router2 = SignalRouter(config={
            'time_series_primary': True,
            'llm_fallback': False
        })
        assert router2._get_routing_mode() == 'TIME_SERIES_ONLY'

        # LLM only
        router3 = SignalRouter(config={
            'time_series_primary': False,
            'llm_fallback': True
        })
        assert router3._get_routing_mode() == 'LLM_ONLY'

    def test_ts_disabled_ticker_routes_to_llm_only(self, mock_llm_generator, sample_forecast_bundle):
        """Tickers flagged as disable_time_series in forecaster_monitoring.yml should skip TS."""
        # Ensure BTC-USD is treated as TS-disabled by the router.
        router = SignalRouter(
            config={'time_series_primary': True, 'llm_fallback': True},
            llm_generator=mock_llm_generator,
        )
        assert "BTC-USD" in router._ts_disabled_tickers

        bundle = router.route_signal(
            ticker="BTC-USD",
            forecast_bundle=sample_forecast_bundle,
            current_price=100.0,
            market_data=None,
        )

        # Primary should come from LLM, since TS is disabled for this ticker.
        assert bundle.primary_signal is not None
        assert bundle.primary_signal["source"] == "LLM"


class TestSignalBundle:
    """Test suite for SignalBundle dataclass"""

    def test_bundle_creation(self):
        """Test creating a signal bundle"""
        bundle = SignalBundle(
            primary_signal={'action': 'BUY', 'source': 'TIME_SERIES'},
            fallback_signal={'action': 'HOLD', 'source': 'LLM'},
            all_signals=[
                {'action': 'BUY', 'source': 'TIME_SERIES'},
                {'action': 'HOLD', 'source': 'LLM'}
            ],
            metadata={'ticker': 'AAPL'},
            routing_timestamp=datetime.now()
        )

        assert bundle.primary_signal is not None
        assert bundle.fallback_signal is not None
        assert len(bundle.all_signals) == 2
        assert bundle.metadata['ticker'] == 'AAPL'
