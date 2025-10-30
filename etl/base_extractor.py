"""Base extractor abstract class for platform-agnostic data source implementation.

This module provides the abstract base class that all data extractors must implement,
ensuring a consistent interface across different data sources (yfinance, Alpha Vantage,
Finnhub, etc.).

Design Pattern: Abstract Factory + Strategy Pattern
- Each data source implements the BaseExtractor interface
- DataSourceManager selects and instantiates the appropriate extractor

Mathematical Foundation:
- Interface contract: All extractors must return standardized OHLCV format
- Data quality guarantee: QC(data) ≥ threshold for all sources
- Cache efficiency: η = n_cached / n_total across all sources
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractorMetadata:
    """Metadata container for extraction operations."""

    def __init__(self, ticker: str, source: str, extraction_timestamp: datetime,
                 data_start_date: datetime, data_end_date: datetime,
                 row_count: int, cache_hit: bool):
        """Initialize extraction metadata.

        Args:
            ticker: Stock ticker symbol
            source: Data source name (yfinance, alpha_vantage, etc.)
            extraction_timestamp: When extraction occurred
            data_start_date: First date in returned data
            data_end_date: Last date in returned data
            row_count: Number of rows returned
            cache_hit: Whether data came from cache
        """
        self.ticker = ticker
        self.source = source
        self.extraction_timestamp = extraction_timestamp
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        self.row_count = row_count
        self.cache_hit = cache_hit

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'ticker': self.ticker,
            'source': self.source,
            'extraction_timestamp': self.extraction_timestamp.isoformat(),
            'data_start_date': self.data_start_date.isoformat() if self.data_start_date else None,
            'data_end_date': self.data_end_date.isoformat() if self.data_end_date else None,
            'row_count': self.row_count,
            'cache_hit': self.cache_hit
        }


class BaseExtractor(ABC):
    """Abstract base class for all data source extractors.

    All data source implementations (yfinance, Alpha Vantage, Finnhub, etc.)
    must inherit from this class and implement the required methods.

    Design Requirements:
    1. Standardized OHLCV output format
    2. Built-in caching support
    3. Comprehensive error handling
    4. Quality validation
    5. Metadata tracking
    """

    def __init__(self, name: str, timeout: int = 30, cache_hours: int = 24,
                 storage=None, **kwargs):
        """Initialize base extractor.

        Args:
            name: Data source name (e.g., 'yfinance', 'alpha_vantage')
            timeout: Request timeout in seconds
            cache_hours: Cache validity duration in hours
            storage: DataStorage instance for cache operations
            **kwargs: Additional source-specific parameters
        """
        self.name = name
        self.timeout = timeout
        self.cache_hours = cache_hours
        self.storage = storage
        self.kwargs = kwargs

        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_partials = 0

        logger.info(f"Initialized {self.name} extractor (cache: {cache_hours}h)")

    @abstractmethod
    def extract_ohlcv(self, tickers: List[str], start_date: str,
                      end_date: str) -> pd.DataFrame:
        """Extract OHLCV data for given tickers and date range.

        This is the primary method that all extractors must implement.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with standardized columns:
            - Index: DatetimeIndex
            - Columns: Open, High, Low, Close, Volume, Adj Close (optional), ticker

        Raises:
            NotImplementedError: If subclass doesn't implement this method
            ValueError: If invalid parameters provided
            RuntimeError: If extraction fails
        """
        pass

    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return validation report.

        Args:
            data: DataFrame to validate

        Returns:
            Dictionary containing validation results:
            {
                'passed': bool,
                'errors': List[str],
                'warnings': List[str],
                'quality_score': float (0-1),
                'metrics': {
                    'missing_rate': float,
                    'outlier_count': int,
                    'gap_count': int
                }
            }

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        pass

    @abstractmethod
    def get_metadata(self, ticker: str, data: pd.DataFrame) -> ExtractorMetadata:
        """Generate metadata for extracted data.

        Args:
            ticker: Stock ticker symbol
            data: Extracted DataFrame

        Returns:
            ExtractorMetadata object with extraction details

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        pass

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache metrics:
            {
                'cache_hits': int,
                'cache_misses': int,
                'total_requests': int,
                'hit_rate': float (0-1)
            }
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0

        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_partial_hits': self._cache_partials,
            'total_requests': total,
            'hit_rate': hit_rate
        }

    def _increment_cache_hit(self) -> None:
        """Increment cache hit counter."""
        self._cache_hits += 1

    def _increment_cache_partial(self) -> None:
        """Increment partial cache hit counter."""
        self._cache_partials += 1

    def _increment_cache_miss(self) -> None:
        """Increment cache miss counter."""
        self._cache_misses += 1

    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to consistent format.

        Ensures all extractors return data with the same column names:
        Open, High, Low, Close, Volume, Adj Close (optional)

        Args:
            data: DataFrame with potentially non-standard column names

        Returns:
            DataFrame with standardized column names
        """
        # Column name mapping (handles various naming conventions)
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Adj Close',
            'adjusted close': 'Adj Close',
            'adj_close': 'Adj Close',
            'adjusted_close': 'Adj Close'
        }

        # Create lowercase version for matching
        data_copy = data.copy()
        rename_dict = {}

        for col in data_copy.columns:
            col_lower = str(col).lower()
            if col_lower in column_mapping:
                rename_dict[col] = column_mapping[col_lower]

        if rename_dict:
            data_copy.rename(columns=rename_dict, inplace=True)

        return data_copy

    def _flatten_multiindex(self, data: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns to single level.

        Some data sources (like yfinance) return MultiIndex columns.
        This method flattens them for consistency.

        Args:
            data: DataFrame with potentially MultiIndex columns

        Returns:
            DataFrame with single-level column index
        """
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data

    def _check_required_columns(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Verify that required OHLCV columns are present.

        Args:
            data: DataFrame to check

        Returns:
            Tuple of (all_present: bool, missing_columns: List[str])
        """
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required if col not in data.columns]

        return len(missing) == 0, missing

    def __repr__(self) -> str:
        """String representation of extractor."""
        return f"{self.__class__.__name__}(name='{self.name}', cache={self.cache_hours}h)"

    def __str__(self) -> str:
        """Human-readable string representation."""
        stats = self.get_cache_statistics()
        return (f"{self.name} Extractor\n"
                f"  Cache: {self.cache_hours}h validity\n"
                f"  Stats: {stats['cache_hits']}/{stats['total_requests']} hits "
                f"({stats['hit_rate']:.1%})")
