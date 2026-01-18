"""
Unit Tests for Signal Adapter
Line Count: ~150 lines (within budget)

Tests the signal adapter that provides unified interface for different signal sources.
This is critical for backward compatibility and preventing integration issues.

Per TESTING_GUIDE.md: Focus on profit-critical functions only.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from models.signal_adapter import SignalAdapter, UnifiedSignal
from models.time_series_signal_generator import TimeSeriesSignal


@pytest.fixture
def sample_ts_signal():
    """Create sample Time Series signal"""
    return TimeSeriesSignal(
        ticker='AAPL',
        action='BUY',
        confidence=0.75,
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        signal_timestamp=datetime.now(),
        model_type='ENSEMBLE',
        forecast_horizon=30,
        expected_return=0.10,
        risk_score=0.5,
        reasoning='Time Series forecast indicates bullish trend',
        provenance={'model_type': 'TIME_SERIES_ENSEMBLE'},
        signal_type='TIME_SERIES',
        volatility=0.20
    )


@pytest.fixture
def sample_llm_signal_dict():
    """Create sample LLM signal dictionary"""
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


class TestSignalAdapter:
    """Test suite for SignalAdapter"""

    def test_from_time_series_signal(self, sample_ts_signal):
        """Test converting Time Series signal to UnifiedSignal"""
        unified = SignalAdapter.from_time_series_signal(sample_ts_signal)

        assert isinstance(unified, UnifiedSignal)
        assert unified.ticker == 'AAPL'
        assert unified.action == 'BUY'
        assert unified.confidence == 0.75
        assert unified.entry_price == 100.0
        assert unified.source == 'TIME_SERIES'
        assert unified.model_type == 'ENSEMBLE'
        assert unified.target_price == 110.0
        assert unified.stop_loss == 95.0
        assert unified.expected_return == 0.10
        assert unified.risk_score == 0.5

    def test_from_llm_signal(self, sample_llm_signal_dict):
        """Test converting LLM signal to UnifiedSignal"""
        unified = SignalAdapter.from_llm_signal(sample_llm_signal_dict)

        assert isinstance(unified, UnifiedSignal)
        assert unified.ticker == 'AAPL'
        assert unified.action == 'BUY'
        assert unified.confidence == 0.65
        assert unified.entry_price == 100.0
        assert unified.source == 'LLM'
        assert unified.llm_model == 'qwen:14b-chat-q4_K_M'
        assert unified.fallback is False

    def test_to_legacy_dict(self, sample_ts_signal):
        """Test converting UnifiedSignal to legacy dict format"""
        unified = SignalAdapter.from_time_series_signal(sample_ts_signal)
        legacy_dict = SignalAdapter.to_legacy_dict(unified)

        assert isinstance(legacy_dict, dict)
        assert legacy_dict['ticker'] == 'AAPL'
        assert legacy_dict['action'] == 'BUY'
        assert legacy_dict['confidence'] == 0.75
        assert legacy_dict['entry_price'] == 100.0
        assert legacy_dict['source'] == 'TIME_SERIES'
        assert 'target_price' in legacy_dict
        assert 'stop_loss' in legacy_dict
        assert 'expected_return' in legacy_dict

    def test_normalize_time_series_signal(self, sample_ts_signal):
        """Test normalizing Time Series signal"""
        unified = SignalAdapter.normalize_signal(sample_ts_signal)

        assert isinstance(unified, UnifiedSignal)
        assert unified.source == 'TIME_SERIES'

    def test_normalize_llm_signal_dict(self, sample_llm_signal_dict):
        """Test normalizing LLM signal dict"""
        unified = SignalAdapter.normalize_signal(sample_llm_signal_dict)

        assert isinstance(unified, UnifiedSignal)
        assert unified.source == 'LLM'

    def test_normalize_unified_signal(self, sample_ts_signal):
        """Test normalizing already unified signal"""
        unified1 = SignalAdapter.from_time_series_signal(sample_ts_signal)
        unified2 = SignalAdapter.normalize_signal(unified1)

        # Should return the same signal
        assert unified2 is unified1

    def test_validate_signal_valid(self, sample_ts_signal):
        """Test validating a valid signal"""
        unified = SignalAdapter.from_time_series_signal(sample_ts_signal)
        is_valid, error = SignalAdapter.validate_signal(unified)

        assert is_valid is True
        assert error is None

    def test_validate_signal_missing_ticker(self):
        """Test validating signal with missing ticker"""
        signal = UnifiedSignal(
            ticker='',
            action='BUY',
            confidence=0.75,
            entry_price=100.0,
            signal_timestamp=datetime.now(),
            source='TIME_SERIES'
        )

        is_valid, error = SignalAdapter.validate_signal(signal)

        assert is_valid is False
        assert 'ticker' in error.lower()

    def test_validate_signal_invalid_action(self):
        """Test validating signal with invalid action"""
        signal = UnifiedSignal(
            ticker='AAPL',
            action='INVALID',
            confidence=0.75,
            entry_price=100.0,
            signal_timestamp=datetime.now(),
            source='TIME_SERIES'
        )

        is_valid, error = SignalAdapter.validate_signal(signal)

        assert is_valid is False
        assert 'action' in error.lower()

    def test_validate_signal_invalid_confidence(self):
        """Test validating signal with invalid confidence"""
        signal = UnifiedSignal(
            ticker='AAPL',
            action='BUY',
            confidence=1.5,  # Out of range
            entry_price=100.0,
            signal_timestamp=datetime.now(),
            source='TIME_SERIES'
        )

        is_valid, error = SignalAdapter.validate_signal(signal)

        assert is_valid is False
        assert 'confidence' in error.lower()

    def test_validate_signal_invalid_price(self):
        """Test validating signal with invalid entry price"""
        signal = UnifiedSignal(
            ticker='AAPL',
            action='BUY',
            confidence=0.75,
            entry_price=-10.0,  # Invalid
            signal_timestamp=datetime.now(),
            source='TIME_SERIES'
        )

        is_valid, error = SignalAdapter.validate_signal(signal)

        assert is_valid is False
        assert 'price' in error.lower()
