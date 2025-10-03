"""Unit tests for YFinanceExtractor caching mechanism.

Test Coverage:
- Cache hit/miss detection
- Cache freshness validation
- Cache coverage validation
- Network request reduction
- Auto-caching on fetch
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from etl.yfinance_extractor import YFinanceExtractor
from etl.data_storage import DataStorage


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    temp_dir = tempfile.mkdtemp()
    storage = DataStorage(base_path=temp_dir)
    yield storage
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    data = pd.DataFrame({
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(100, 200, len(dates)),
        'Low': np.random.uniform(100, 200, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    data.index.name = 'Date'
    return data


class TestCacheMechanism:
    """Test cache lookup and validation logic."""

    def test_cache_miss_no_storage(self):
        """Cache should miss when storage is None."""
        extractor = YFinanceExtractor(storage=None)
        result = extractor._check_cache('AAPL', datetime(2020, 1, 1), datetime(2020, 12, 31))
        assert result is None

    def test_cache_miss_no_files(self, temp_storage):
        """Cache should miss when no files exist."""
        extractor = YFinanceExtractor(storage=temp_storage)
        result = extractor._check_cache('AAPL', datetime(2020, 1, 1), datetime(2020, 12, 31))
        assert result is None

    def test_cache_miss_expired(self, temp_storage, sample_ohlcv_data):
        """Cache should miss when file is older than cache_hours."""
        # Save data
        temp_storage.save(sample_ohlcv_data, 'raw', 'AAPL')

        # Set cache validity to 0 hours (immediate expiration)
        extractor = YFinanceExtractor(storage=temp_storage, cache_hours=0)
        result = extractor._check_cache('AAPL', datetime(2020, 1, 1), datetime(2020, 12, 31))
        assert result is None

    def test_cache_miss_incomplete_coverage(self, temp_storage, sample_ohlcv_data):
        """Cache should miss when data doesn't cover requested range."""
        # Save data for 2020
        temp_storage.save(sample_ohlcv_data, 'raw', 'AAPL')

        # Request data for 2019-2021 (cache only has 2020)
        extractor = YFinanceExtractor(storage=temp_storage, cache_hours=24)
        result = extractor._check_cache('AAPL', datetime(2019, 1, 1), datetime(2021, 12, 31))
        assert result is None

    def test_cache_hit_valid_data(self, temp_storage, sample_ohlcv_data):
        """Cache should hit when data is fresh and covers range."""
        # Save data
        temp_storage.save(sample_ohlcv_data, 'raw', 'AAPL')

        # Request subset of cached range
        extractor = YFinanceExtractor(storage=temp_storage, cache_hours=24)
        result = extractor._check_cache('AAPL', datetime(2020, 6, 1), datetime(2020, 6, 30))

        assert result is not None
        assert not result.empty
        assert len(result) == 30  # June has 30 days
        assert result.index.min() >= pd.Timestamp('2020-06-01')
        assert result.index.max() <= pd.Timestamp('2020-06-30')

    def test_cache_hit_exact_range(self, temp_storage, sample_ohlcv_data):
        """Cache should hit and return exact range when requested."""
        temp_storage.save(sample_ohlcv_data, 'raw', 'AAPL')

        extractor = YFinanceExtractor(storage=temp_storage, cache_hours=24)
        result = extractor._check_cache('AAPL', datetime(2020, 1, 1), datetime(2020, 12, 31))

        assert result is not None
        assert len(result) == len(sample_ohlcv_data)


class TestCacheIntegration:
    """Test cache integration with extract_ohlcv."""

    def test_auto_caching_on_fetch(self, temp_storage, monkeypatch):
        """New data should be automatically cached after fetch."""
        # Mock fetch_ticker_data to avoid real network calls
        def mock_fetch(ticker, start, end, timeout):
            dates = pd.date_range(start, end, freq='D')
            return pd.DataFrame({
                'Open': np.random.uniform(100, 200, len(dates)),
                'High': np.random.uniform(100, 200, len(dates)),
                'Low': np.random.uniform(100, 200, len(dates)),
                'Close': np.random.uniform(100, 200, len(dates)),
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)

        import etl.yfinance_extractor
        monkeypatch.setattr(etl.yfinance_extractor, 'fetch_ticker_data', mock_fetch)

        extractor = YFinanceExtractor(storage=temp_storage, cache_hours=24)

        # First call should fetch and cache
        data1 = extractor.extract_ohlcv(['AAPL'], '2020-01-01', '2020-12-31')
        assert not data1.empty

        # Check that data was cached
        raw_files = list((temp_storage.base_path / 'raw').glob('*AAPL*.parquet'))
        assert len(raw_files) > 0

    def test_cache_hit_rate_logging(self, temp_storage, sample_ohlcv_data, monkeypatch, caplog):
        """Cache hit rate should be logged correctly."""
        # Pre-populate cache with AAPL
        temp_storage.save(sample_ohlcv_data, 'raw', 'AAPL')

        # Mock fetch for MSFT (cache miss)
        def mock_fetch(ticker, start, end, timeout):
            if ticker == 'MSFT':
                dates = pd.date_range(start, end, freq='D')
                return pd.DataFrame({
                    'Open': np.random.uniform(100, 200, len(dates)),
                    'High': np.random.uniform(100, 200, len(dates)),
                    'Low': np.random.uniform(100, 200, len(dates)),
                    'Close': np.random.uniform(100, 200, len(dates)),
                    'Volume': np.random.randint(1000000, 10000000, len(dates))
                }, index=dates)
            return pd.DataFrame()

        import etl.yfinance_extractor
        monkeypatch.setattr(etl.yfinance_extractor, 'fetch_ticker_data', mock_fetch)

        extractor = YFinanceExtractor(storage=temp_storage, cache_hours=24)

        # Fetch AAPL (cache hit) and MSFT (cache miss)
        with caplog.at_level('INFO'):
            data = extractor.extract_ohlcv(['AAPL', 'MSFT'], '2020-01-01', '2020-12-31')

        # Check cache hit rate in logs
        assert 'Cache HIT for AAPL' in caplog.text
        assert 'Cache performance: 1/2 hits (50.0% hit rate)' in caplog.text

    def test_no_duplicate_network_requests(self, temp_storage, sample_ohlcv_data, monkeypatch):
        """Cache hits should not trigger network requests."""
        temp_storage.save(sample_ohlcv_data, 'raw', 'AAPL')

        # Track fetch calls
        fetch_calls = []

        def mock_fetch(ticker, start, end, timeout):
            fetch_calls.append(ticker)
            return pd.DataFrame()

        import etl.yfinance_extractor
        monkeypatch.setattr(etl.yfinance_extractor, 'fetch_ticker_data', mock_fetch)

        extractor = YFinanceExtractor(storage=temp_storage, cache_hours=24)
        data = extractor.extract_ohlcv(['AAPL'], '2020-01-01', '2020-12-31')

        # Should not have called fetch (cache hit)
        assert 'AAPL' not in fetch_calls
        assert not data.empty


class TestCacheFreshness:
    """Test cache freshness validation."""

    def test_cache_validity_boundary(self, temp_storage, sample_ohlcv_data):
        """Test cache expiration at exact boundary."""
        # Save data
        filepath = temp_storage.save(sample_ohlcv_data, 'raw', 'AAPL')

        # Modify file timestamp to be exactly cache_hours old
        cache_hours = 24
        old_time = datetime.now() - timedelta(hours=cache_hours, seconds=1)
        old_timestamp = old_time.timestamp()
        filepath.parent.glob('*AAPL*.parquet').__next__().touch()
        import os
        os.utime(filepath, (old_timestamp, old_timestamp))

        extractor = YFinanceExtractor(storage=temp_storage, cache_hours=cache_hours)
        result = extractor._check_cache('AAPL', datetime(2020, 1, 1), datetime(2020, 12, 31))

        # Should be expired
        assert result is None
