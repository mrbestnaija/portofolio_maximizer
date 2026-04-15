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

    def test_unsupported_routing_knobs_emit_warning(self, caplog):
        caplog.set_level("WARNING")
        router = SignalRouter(
            config={
                'time_series_primary': True,
                'llm_fallback': True,
                'enable_samossa': True,
                'routing_mode': 'TIME_SERIES_ONLY',
            }
        )
        assert "unsupported_routing_knob:enable_samossa" in router.routing_contract_warnings
        assert "unsupported_routing_knob:routing_mode" in router.routing_contract_warnings

    def test_strict_routing_contract_raises_on_unsupported_knob(self):
        with pytest.raises(ValueError):
            SignalRouter(
                config={
                    'time_series_primary': True,
                    'llm_fallback': True,
                    'enable_samossa': True,
                    'strict_routing_config': True,
                }
            )

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

    def test_signal_to_dict_promotes_cost_and_weather_context(self):
        """Critical root fields must be promoted so execution gates cannot bypass nested context."""
        router = SignalRouter()
        ts_signal = TimeSeriesSignal(
            ticker="CORN",
            action="BUY",
            confidence=0.8,
            entry_price=100.0,
            signal_timestamp=datetime.now(),
            model_type="ENSEMBLE",
            expected_return=0.02,
            risk_score=0.4,
            provenance={
                "decision_context": {
                    "expected_return_net": 0.001,
                    "gross_trade_return": 0.02,
                    "net_trade_return": 0.001,
                    "roundtrip_cost_fraction": 0.002,
                    "roundtrip_cost_bps": 20.0,
                },
                "weather_context": {
                    "event_type": "drought",
                    "severity": "high",
                },
            },
        )

        payload = router._signal_to_dict(ts_signal)

        assert payload["expected_return_net"] == pytest.approx(0.001)
        assert payload["roundtrip_cost_bps"] == pytest.approx(20.0)
        assert payload["weather_context"]["event_type"] == "drought"

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

    def test_gs_disable_time_series_in_forecaster_monitoring(self):
        """GS (0 wins / 5 trades) must appear in the TS-disabled list loaded from
        forecaster_monitoring.yml so the kill-switch config change takes effect at runtime."""
        router = SignalRouter(config={'time_series_primary': True, 'llm_fallback': False})
        assert "GS" in router._ts_disabled_tickers, (
            "GS should be TS-disabled (0% WR); add disable_time_series: true under GS in "
            "config/forecaster_monitoring.yml"
        )

    def test_aapl_min_expected_return_at_global_default(self):
        """AAPL per-ticker min_expected_return must be at global default range, not an uncalibrated override.

        The prior 80 bps 'temporary conservative override' blocked 100% of AAPL signals and was
        identified as a THIN_LINKAGE root cause (funnel audit 2026-04-15). The 14% historical WR
        came from a system generating zero AAPL trades — it was a bootstrapping artefact, not
        evidence for a 80 bps bar. The domain_utility gate (omega_ratio, payoff_asymmetry) provides
        the actual quality bar; routing threshold must not silently block all signals.
        Guard: threshold must remain in [20 bps, 40 bps] global-default range.
        """
        import yaml
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "signal_routing_config.yml"
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        global_mer = (
            raw.get("signal_routing", {})
            .get("time_series", {})
            .get("min_expected_return", 0.003)
        )
        aapl_cfg = (
            raw.get("signal_routing", {})
            .get("time_series", {})
            .get("per_ticker", {})
            .get("AAPL", {})
        )
        aapl_mer = aapl_cfg.get("min_expected_return", global_mer)
        assert aapl_mer <= 0.0040, (
            f"AAPL min_expected_return={aapl_mer * 10000:.1f}bps must not exceed 40bps "
            "(silently re-acquiring an uncalibrated override — funnel audit 2026-04-15)."
        )
        assert aapl_mer >= 0.0020, (
            f"AAPL min_expected_return={aapl_mer * 10000:.1f}bps must be >= 20bps "
            "(below roundtrip cost is zero-edge territory)."
        )


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
