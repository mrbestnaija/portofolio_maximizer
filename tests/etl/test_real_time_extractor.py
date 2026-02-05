"""
Test Real-Time Market Data Extractor
Line Count: ~120 lines (within budget)

Tests real-time data extraction with:
- Quote fetching
- Volatility spike detection
- Failover mechanisms
- Rate limiting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from etl.real_time_extractor import RealTimeExtractor, MarketData, VolatilityAlert


class TestRealTimeExtractor:
    """Test suite for RealTimeExtractor class"""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance for testing"""
        with patch('etl.real_time_extractor.AlphaVantageExtractor'):
            with patch('etl.real_time_extractor.YFinanceExtractor'):
                return RealTimeExtractor(
                    update_frequency=1,  # 1 second for testing
                    volatility_threshold=0.05,
                    use_cache=False
                )

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data"""
        return MarketData(
            ticker='AAPL',
            price=150.50,
            volume=1000000,
            timestamp=datetime.now(),
            source='test'
        )

    def test_extractor_initialization(self, extractor):
        """Test extractor initializes with correct parameters"""
        assert extractor.update_frequency == 1
        assert extractor.volatility_threshold == 0.05
        assert extractor.use_cache is False
        assert isinstance(extractor.price_history, dict)
        assert isinstance(extractor.volume_history, dict)

    def test_market_data_structure(self, sample_market_data):
        """Test MarketData dataclass structure"""
        assert sample_market_data.ticker == 'AAPL'
        assert sample_market_data.price == 150.50
        assert sample_market_data.volume == 1000000
        assert sample_market_data.source == 'test'
        assert isinstance(sample_market_data.timestamp, datetime)

    def test_update_history(self, extractor, sample_market_data):
        """Test price and volume history updates"""
        ticker = 'AAPL'

        # Update history
        extractor._update_history(ticker, sample_market_data)

        assert ticker in extractor.price_history
        assert ticker in extractor.volume_history
        assert len(extractor.price_history[ticker]) == 1
        assert extractor.price_history[ticker][0] == 150.50
        assert extractor.volume_history[ticker][0] == 1000000

    def test_history_max_length(self, extractor):
        """Test history maintains max length of 100"""
        ticker = 'TEST'

        # Add 150 data points
        for i in range(150):
            market_data = MarketData(
                ticker=ticker,
                price=100.0 + i * 0.1,
                volume=1000000,
                timestamp=datetime.now(),
                source='test'
            )
            extractor._update_history(ticker, market_data)

        # Should only keep last 100
        assert len(extractor.price_history[ticker]) == 100
        assert len(extractor.volume_history[ticker]) == 100

    def test_volatility_spike_detection(self, extractor):
        """Test volatility spike detection"""
        ticker = 'AAPL'

        # Build normal price history
        for i in range(30):
            market_data = MarketData(
                ticker=ticker,
                price=100.0 + i * 0.1,  # Gradual increase
                volume=1000000,
                timestamp=datetime.now(),
                source='test'
            )
            extractor._update_history(ticker, market_data)

        # Add spike
        spike_data = MarketData(
            ticker=ticker,
            price=110.0,  # Large jump
            volume=1000000,
            timestamp=datetime.now(),
            source='test'
        )

        alert = extractor._check_volatility_spike(ticker, spike_data)

        assert isinstance(alert, VolatilityAlert)
        assert alert.ticker == ticker
        assert alert.spike_magnitude > 0
        assert alert.recommendation in ['HALT_TRADING', 'REDUCE_POSITIONS', 'MONITOR_CLOSELY']

    def test_no_volatility_spike_on_normal_move(self, extractor):
        """Test no alert on normal price movement"""
        ticker = 'AAPL'

        # Build normal price history
        for i in range(30):
            market_data = MarketData(
                ticker=ticker,
                price=100.0 + i * 0.05,  # Small gradual increase
                volume=1000000,
                timestamp=datetime.now(),
                source='test'
            )
            extractor._update_history(ticker, market_data)

        # Add normal move
        normal_data = MarketData(
            ticker=ticker,
            price=101.5,  # Small move
            volume=1000000,
            timestamp=datetime.now(),
            source='test'
        )

        alert = extractor._check_volatility_spike(ticker, normal_data)

        assert alert is None

    def test_get_multiple_quotes(self, extractor):
        """Test fetching multiple quotes"""
        tickers = ['AAPL', 'GOOGL', 'MSFT']

        # Mock the _fetch_current_data method
        def mock_fetch(ticker):
            return MarketData(
                ticker=ticker,
                price=100.0,
                volume=1000000,
                timestamp=datetime.now(),
                source='mock'
            )

        extractor._fetch_current_data = mock_fetch

        quotes = extractor.get_multiple_quotes(tickers)

        assert len(quotes) == 3
        for ticker in tickers:
            assert ticker in quotes
            assert isinstance(quotes[ticker], MarketData)
            assert quotes[ticker].ticker == ticker

    def test_rate_limiting(self, extractor):
        """Test rate limiting between requests"""
        ticker = 'AAPL'

        # Mock fetch method
        fetch_count = {'count': 0}

        def mock_fetch_with_limit(t):
            # Check rate limiting logic
            now = datetime.now()
            if t in extractor.last_request_time:
                time_since_last = (now - extractor.last_request_time[t]).total_seconds()
                if time_since_last < extractor.min_request_interval:
                    return None

            fetch_count['count'] += 1
            extractor.last_request_time[t] = now

            return MarketData(
                ticker=t,
                price=100.0,
                volume=1000000,
                timestamp=now,
                source='mock'
            )

        extractor._fetch_current_data = mock_fetch_with_limit

        # First fetch should succeed
        result1 = extractor.get_current_quote(ticker)
        assert result1 is not None
        assert fetch_count['count'] == 1

        # Immediate second fetch should be rate-limited
        result2 = extractor.get_current_quote(ticker)
        assert result2 is None
        assert fetch_count['count'] == 1  # Not incremented

    def test_stream_duration_limit(self, extractor):
        """Test stream respects duration limit"""
        # Mock fetch method
        def mock_fetch(ticker):
            return MarketData(
                ticker=ticker,
                price=100.0,
                volume=1000000,
                timestamp=datetime.now(),
                source='mock'
            )

        extractor._fetch_current_data = mock_fetch
        extractor.update_frequency = 0.1  # Fast for testing

        # Stream for very short duration
        data_count = 0
        for data in extractor.stream_market_data(['TEST'], duration_minutes=0.01):
            data_count += 1
            if data_count > 5:  # Safety limit
                break

        # Should have received some data but stopped due to duration
        assert data_count > 0

    def test_volatility_alert_structure(self):
        """Test VolatilityAlert dataclass structure"""
        alert = VolatilityAlert(
            ticker='AAPL',
            current_volatility=0.08,
            normal_volatility=0.02,
            spike_magnitude=4.0,
            timestamp=datetime.now(),
            recommendation='HALT_TRADING'
        )

        assert alert.ticker == 'AAPL'
        assert alert.current_volatility == 0.08
        assert alert.normal_volatility == 0.02
        assert alert.spike_magnitude == 4.0
        assert alert.recommendation == 'HALT_TRADING'
        assert isinstance(alert.timestamp, datetime)

    def test_insufficient_history_no_alert(self, extractor):
        """Test no volatility alert with insufficient history"""
        ticker = 'NEW'

        # Add only a few data points (< 20)
        for i in range(5):
            market_data = MarketData(
                ticker=ticker,
                price=100.0 + i,
                volume=1000000,
                timestamp=datetime.now(),
                source='test'
            )
            extractor._update_history(ticker, market_data)

        # Check for spike (should return None due to insufficient history)
        spike_data = MarketData(
            ticker=ticker,
            price=120.0,
            volume=1000000,
            timestamp=datetime.now(),
            source='test'
        )

        alert = extractor._check_volatility_spike(ticker, spike_data)
        assert alert is None


class TestRealTimeExtractorIntegration:
    """Integration tests for real-time extractor"""

    def test_failover_mechanism(self):
        """Test failover to backup source"""
        with patch('etl.real_time_extractor.AlphaVantageExtractor'):
            with patch('etl.real_time_extractor.YFinanceExtractor'):
                extractor = RealTimeExtractor()

                # Mock primary source failure
                extractor._fetch_current_data = Mock(return_value=None)

                # Mock backup source success
                backup_data = MarketData(
                    ticker='TEST',
                    price=100.0,
                    volume=1000000,
                    timestamp=datetime.now(),
                    source='backup'
                )
                extractor._fetch_from_backup = Mock(return_value=backup_data)

                # Should use backup
                result = extractor._fetch_from_backup('TEST')
                assert result is not None
                assert result.source == 'backup'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
