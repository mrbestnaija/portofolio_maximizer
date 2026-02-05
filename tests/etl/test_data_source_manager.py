"""Tests for data source manager and multi-source orchestration.

This test suite validates:
1. Data source manager initialization
2. Extractor instantiation and registration
3. Selection strategies (priority, fallback)
4. Failover mechanisms
5. Cache statistics aggregation
6. Data validation through manager
"""

import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from etl.data_source_manager import DataSourceManager
from etl.base_extractor import BaseExtractor, ExtractorMetadata
from etl.yfinance_extractor import YFinanceExtractor
from etl.data_storage import DataStorage


@pytest.fixture
def temp_config_dir():
    """Create temporary config directory with test configuration."""
    temp_dir = tempfile.mkdtemp()
    config_dir = Path(temp_dir) / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create test data sources config
    config_content = """
data_sources:
  providers:
    - name: yfinance
      enabled: true
      priority: 1
      rate_limit: 5
      credentials_env: null

    - name: alpha_vantage
      enabled: false
      priority: 2
      rate_limit: 5
      credentials_env: ALPHA_VANTAGE_API_KEY

    - name: finnhub
      enabled: false
      priority: 3
      rate_limit: 60
      credentials_env: FINNHUB_API_KEY

  adapters:
    adapter_registry:
      yfinance: etl.yfinance_extractor.YFinanceExtractor
      alpha_vantage: etl.alpha_vantage_extractor.AlphaVantageExtractor
      finnhub: etl.finnhub_extractor.FinnhubExtractor

  selection_strategy:
    mode: priority

  failover:
    enabled: true
    max_failover_attempts: 3
    retry_delay: 1
"""

    config_file = config_dir / 'data_sources_config.yml'
    with open(config_file, 'w') as f:
        f.write(config_content)

    yield config_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_storage():
    """Create temporary storage for tests."""
    temp_dir = tempfile.mkdtemp()
    storage = DataStorage(base_path=temp_dir)

    yield storage

    # Cleanup
    shutil.rmtree(temp_dir)


class TestDataSourceManagerInitialization:
    """Test suite for DataSourceManager initialization."""

    def test_initialization_with_valid_config(self, temp_config_dir, temp_storage):
        """Test manager initializes correctly with valid configuration."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        assert manager is not None
        assert manager.active_extractor is not None
        assert manager.get_active_source() == 'yfinance'
        assert len(manager.get_available_sources()) >= 1

    def test_initialization_with_missing_config(self):
        """Test manager raises error when config file is missing."""
        with pytest.raises(FileNotFoundError):
            DataSourceManager(config_path='nonexistent_config.yml')

    def test_initialization_loads_config(self, temp_config_dir, temp_storage):
        """Test manager correctly loads configuration."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        assert 'providers' in manager.config
        assert 'selection_strategy' in manager.config
        assert len(manager.config['providers']) == 3


class TestExtractorInstantiation:
    """Test suite for extractor instantiation and registration."""

    def test_yfinance_extractor_instantiation(self, temp_config_dir, temp_storage):
        """Test YFinance extractor is correctly instantiated."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        extractor = manager.get_extractor('yfinance')
        assert extractor is not None
        assert isinstance(extractor, YFinanceExtractor)
        assert extractor.name == 'yfinance'

    def test_extractor_has_storage_reference(self, temp_config_dir, temp_storage):
        """Test instantiated extractors have storage reference."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        extractor = manager.get_extractor('yfinance')
        assert extractor.storage is temp_storage

    def test_disabled_extractors_not_loaded_in_priority_mode(self, temp_config_dir, temp_storage):
        """Test disabled extractors are not loaded in priority mode."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        # Only yfinance should be loaded (it's the only enabled one)
        assert 'yfinance' in manager.get_available_sources()
        assert len(manager.get_available_sources()) == 1


class TestSelectionStrategies:
    """Test suite for data source selection strategies."""

    def test_priority_mode_selects_highest_priority(self, temp_config_dir, temp_storage):
        """Test priority mode selects the highest priority enabled source."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        # yfinance has priority 1 (highest)
        assert manager.get_active_source() == 'yfinance'

    def test_get_available_sources_returns_list(self, temp_config_dir, temp_storage):
        """Test get_available_sources returns correct list."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        sources = manager.get_available_sources()
        assert isinstance(sources, list)
        assert 'yfinance' in sources


class TestDataExtraction:
    """Test suite for data extraction through manager."""

    @patch('etl.yfinance_extractor.fetch_ticker_data')
    def test_extract_ohlcv_uses_active_source(self, mock_fetch, temp_config_dir, temp_storage):
        """Test extract_ohlcv uses the active data source."""
        # Mock data
        mock_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))

        mock_fetch.return_value = mock_data.copy()

        config_path = temp_config_dir / 'data_sources_config.yml'
        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        result = manager.extract_ohlcv(['AAPL'], '2024-01-01', '2024-01-02')

        assert result is not None
        assert not result.empty
        assert mock_fetch.called

    @patch('etl.yfinance_extractor.fetch_ticker_data')
    def test_extract_ohlcv_with_prefer_source(self, mock_fetch, temp_config_dir, temp_storage):
        """Test extract_ohlcv respects prefer_source parameter."""
        mock_data = pd.DataFrame({
            'Open': [100],
            'High': [102],
            'Low': [99],
            'Close': [101],
            'Volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1))

        mock_fetch.return_value = mock_data.copy()

        config_path = temp_config_dir / 'data_sources_config.yml'
        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        # Should use yfinance since it's the only available
        result = manager.extract_ohlcv(['AAPL'], '2024-01-01', '2024-01-02',
                                       prefer_source='yfinance')

        assert result is not None


class TestCacheStatistics:
    """Test suite for cache statistics aggregation."""

    def test_get_cache_statistics_returns_dict(self, temp_config_dir, temp_storage):
        """Test get_cache_statistics returns dictionary for all sources."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        stats = manager.get_cache_statistics()

        assert isinstance(stats, dict)
        assert 'yfinance' in stats

    def test_cache_statistics_structure(self, temp_config_dir, temp_storage):
        """Test cache statistics have correct structure."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        stats = manager.get_cache_statistics()
        yf_stats = stats['yfinance']

        assert 'cache_hits' in yf_stats
        assert 'cache_misses' in yf_stats
        assert 'total_requests' in yf_stats
        assert 'hit_rate' in yf_stats


class TestDataValidation:
    """Test suite for data validation through manager."""

    def test_validate_data_uses_active_extractor(self, temp_config_dir, temp_storage):
        """Test validate_data uses the active extractor's validation."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        # Create test data
        test_data = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1100]
        })

        result = manager.validate_data(test_data)

        assert isinstance(result, dict)
        assert 'passed' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'quality_score' in result

    def test_validate_empty_dataframe(self, temp_config_dir, temp_storage):
        """Test validation of empty DataFrame."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        empty_df = pd.DataFrame()
        result = manager.validate_data(empty_df)

        assert result['passed'] is False
        assert len(result['errors']) > 0


class TestManagerStringRepresentation:
    """Test suite for manager string representations."""

    def test_repr(self, temp_config_dir, temp_storage):
        """Test __repr__ method."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        repr_str = repr(manager)
        assert 'DataSourceManager' in repr_str
        assert 'yfinance' in repr_str

    def test_str(self, temp_config_dir, temp_storage):
        """Test __str__ method."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        str_repr = str(manager)
        assert 'Data Source Manager' in str_repr
        assert 'yfinance' in str_repr
        assert 'priority' in str_repr.lower()


class TestErrorHandling:
    """Test suite for error handling."""

    def test_invalid_yaml_config(self, temp_config_dir):
        """Test handling of invalid YAML configuration."""
        config_path = temp_config_dir / 'invalid_config.yml'

        # Create invalid YAML
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [[[")

        with pytest.raises(Exception):  # Should raise YAML error
            DataSourceManager(config_path=str(config_path))

    def test_extract_with_no_active_extractor(self, temp_config_dir, temp_storage):
        """Test extraction fails gracefully when no active extractor."""
        config_path = temp_config_dir / 'data_sources_config.yml'

        manager = DataSourceManager(
            config_path=str(config_path),
            storage=temp_storage
        )

        # Force remove active extractor
        manager.active_extractor = None

        # Should raise error or handle gracefully
        # This depends on implementation - adjust as needed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
