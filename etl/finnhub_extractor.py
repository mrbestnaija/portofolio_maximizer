"""Finnhub data extractor with full API integration.

This module provides production-ready Finnhub data extraction with:
- Full API integration with stock/candle endpoint
- Intelligent caching with 24-hour validity
- Rate limiting (60 calls/minute for free tier)
- Exponential backoff retry logic
- Comprehensive data validation
- Unix timestamp conversion

API Documentation: https://finnhub.io/docs/api

Mathematical Foundation:
- OHLCV standardization: [Open, High, Low, Close, Volume]
- Rate limiting: λ = 60 requests/minute (free tier)
- Cache efficiency: η = n_cached / n_total
- Timestamp conversion: t_datetime = datetime.from_timestamp(t_unix)
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from etl.base_extractor import BaseExtractor, ExtractorMetadata

logger = logging.getLogger(__name__)


class FinnhubExtractor(BaseExtractor):
    """Finnhub extractor with full API integration.

    Features:
    - Stock candles (OHLCV) API endpoint
    - 60 requests/minute rate limiting (free tier)
    - Exponential backoff retry (3 attempts)
    - Cache-first strategy with 24h validity
    - Unix timestamp to datetime conversion
    - Comprehensive validation and metadata
    """

    def __init__(self, name: str = 'finnhub', timeout: int = 30,
                 cache_hours: int = 24, storage=None, config_path: str = None,
                 api_key: str = None, **kwargs):
        """Initialize Finnhub extractor.

        Args:
            name: Data source name (default: 'finnhub')
            timeout: Request timeout in seconds
            cache_hours: Cache validity duration in hours
            storage: DataStorage instance for cache operations
            config_path: Path to Finnhub config file
            api_key: Finnhub API key (if not in .env)
            **kwargs: Additional parameters
        """
        super().__init__(name=name, timeout=timeout, cache_hours=cache_hours,
                        storage=storage, **kwargs)

        # Load configuration
        if config_path is None:
            config_path = 'config/finnhub_config.yml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get API key from environment or parameter
        # SECURITY: Use secret_loader for secure secret management
        from etl.secret_loader import load_secret
        api_key_env = self.config['extraction']['authentication']['api_key_env']
        self.api_key = api_key or load_secret(api_key_env)

        if not self.api_key:
            raise ValueError(
                f"Finnhub API key not found. "
                f"Set {api_key_env} in .env file or pass as parameter."
            )

        # API configuration
        self.base_url = self.config['extraction']['source']['base_url']
        self.endpoint = self.config['extraction']['data']['endpoint']
        self.resolution = self.config['extraction']['data']['resolution']

        # Rate limiting configuration
        rate_config = self.config['extraction']['rate_limiting']
        self.rate_limit_enabled = rate_config['enabled']
        self.requests_per_minute = rate_config['requests_per_minute']
        self.delay_between_requests = rate_config['delay_between_requests']

        # Network configuration
        network_config = self.config['extraction']['network']
        self.max_retries = network_config['max_retries']
        self.retry_delay = network_config['retry_delay_seconds']
        self.backoff_factor = network_config['backoff_factor']

        # Column mapping
        self.column_mapping = self.config['extraction']['data']['column_mapping']

        # Response configuration
        response_config = self.config['extraction']['response']
        self.success_status = response_config['success_status']
        self.convert_timestamps = response_config['convert_timestamps']

        # Request tracking for rate limiting
        self.last_request_time = None

        logger.info(f"Finnhub extractor initialized successfully")
        logger.info(f"  - Rate limit: {self.requests_per_minute} requests/minute")
        logger.info(f"  - Cache: {cache_hours} hours validity")
        logger.info(f"  - Resolution: {self.resolution} (daily)")

    def extract_ohlcv(self, tickers: List[str], start_date: str,
                      end_date: str) -> pd.DataFrame:
        """Extract OHLCV data from Finnhub API.

        Implements cache-first strategy with automatic failover to API.
        Rate limiting ensures compliance with API limits (60 calls/minute).

        Mathematical Foundation:
        - Cache hit rate: η = n_cached / n_total
        - Rate limit: λ ≤ 60 requests/minute

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with columns: [Open, High, Low, Close, Volume]
            MultiIndex: (ticker, date)

        Raises:
            ValueError: If API key is invalid or tickers list is empty
            requests.RequestException: If API request fails after retries
        """
        if not tickers:
            raise ValueError("Tickers list cannot be empty")

        logger.info(f"Extracting data for {len(tickers)} ticker(s) from Finnhub")

        all_data = []
        cache_hits = 0

        for i, ticker in enumerate(tickers, 1):
            logger.info(f"[{i}/{len(tickers)}] Processing {ticker}")

            # Check cache first
            if self.storage:
                cached_data = self._check_cache(ticker, start_date, end_date)
                if cached_data is not None:
                    logger.info(f"  ✓ Cache HIT for {ticker}")
                    all_data.append(cached_data)
                    cache_hits += 1
                    continue

            logger.info(f"  ✗ Cache MISS for {ticker}, fetching from API...")

            # Fetch from API with rate limiting and retries
            try:
                ticker_data = self._fetch_with_retry(ticker, start_date, end_date)

                if ticker_data.empty:
                    logger.warning(f"  ⚠ No data for {ticker} in date range {start_date} to {end_date}")
                    continue

                # Add ticker column
                ticker_data['Ticker'] = ticker

                # Cache the data
                if self.storage:
                    self._save_to_cache(ticker, ticker_data)

                all_data.append(ticker_data)
                logger.info(f"  ✓ Fetched {len(ticker_data)} rows for {ticker}")

            except Exception as e:
                logger.error(f"  ✗ Failed to fetch {ticker}: {str(e)}")
                continue

        # Combine all tickers
        if not all_data:
            raise ValueError(f"No data retrieved for any ticker")

        combined_data = pd.concat(all_data, axis=0)

        # Create MultiIndex (ticker, date)
        if 'Ticker' in combined_data.columns:
            combined_data = combined_data.reset_index()
            combined_data = combined_data.set_index(['Ticker', 'Date'])
            combined_data = combined_data.sort_index()

        # Log cache performance
        cache_hit_rate = cache_hits / len(tickers) * 100
        logger.info(f"Cache performance: {cache_hits}/{len(tickers)} hits ({cache_hit_rate:.1f}%)")

        return combined_data

    def _fetch_with_retry(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data with exponential backoff retry logic.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string

        Returns:
            DataFrame with OHLCV data

        Raises:
            requests.RequestException: If all retries fail
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                # Rate limiting
                self._apply_rate_limit()

                # Fetch from API
                data = self._get_stock_candles(ticker, start_date, end_date)

                # Validate non-empty
                if data.empty:
                    raise ValueError(f"No data returned for {ticker}")

                return data

            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"All {self.max_retries} retries failed for {ticker}")
                    raise

                delay = self.retry_delay * (self.backoff_factor ** (attempt - 1))
                logger.warning(f"Attempt {attempt} failed: {str(e)}")
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)

    def _get_stock_candles(self, ticker: str, start_date: str, end_date: str,
                          resolution: str = None) -> pd.DataFrame:
        """Fetch stock candles (OHLCV) from Finnhub API.

        API Endpoint: /stock/candle
        Returns: OHLCV data with Unix timestamps

        Args:
            ticker: Stock ticker symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            resolution: Time resolution (D=daily, W=weekly, M=monthly)

        Returns:
            DataFrame with standardized OHLCV columns

        Raises:
            requests.RequestException: If API request fails
            ValueError: If response is invalid or empty
        """
        # Convert dates to Unix timestamps
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

        # Use provided resolution or default from config
        resolution = resolution or self.resolution

        # Build request URL
        url = f"{self.base_url}{self.endpoint}"
        params = {
            'symbol': ticker,
            'resolution': resolution,
            'from': start_timestamp,
            'to': end_timestamp,
            'token': self.api_key
        }

        logger.debug(f"API request: {url} with symbol={ticker}, resolution={resolution}")

        # Make request
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()

        # Check response status
        if 's' in data:
            if data['s'] == self.success_status:
                logger.debug(f"API returned success status for {ticker}")
            elif data['s'] == 'no_data':
                logger.warning(f"No data available for {ticker}")
                return pd.DataFrame()
            else:
                raise ValueError(f"API returned error status: {data.get('s')}")
        else:
            raise ValueError(f"Invalid API response format. Keys: {list(data.keys())}")

        # Extract OHLCV arrays
        required_keys = ['c', 'h', 'l', 'o', 'v', 't']
        if not all(key in data for key in required_keys):
            missing = [key for key in required_keys if key not in data]
            raise ValueError(f"Missing required keys in API response: {missing}")

        # Create DataFrame from arrays
        df = pd.DataFrame({
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data['v'],
            'timestamp': data['t']
        })

        # Convert Unix timestamps to datetime
        if self.convert_timestamps:
            df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.drop('timestamp', axis=1)
            df = df.set_index('Date')

        # Sort by date
        df = df.sort_index()

        # Ensure numeric types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _apply_rate_limit(self):
        """Apply rate limiting between API requests.

        Free tier: 60 requests/minute → 1 second between requests
        Premium tier: 300 requests/minute → 0.2 seconds between requests
        """
        if not self.rate_limit_enabled:
            return

        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.delay_between_requests:
                sleep_time = self.delay_between_requests - elapsed
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _check_cache(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Check local cache for recent data.

        Inherits cache logic from BaseExtractor via storage.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date string
            end_date: End date string

        Returns:
            Cached DataFrame if valid, None otherwise
        """
        if not self.storage:
            return None

        # Check for cached files
        cache_files = list(self.storage.base_path.glob(f'raw/{ticker}_*.parquet'))
        if not cache_files:
            return None

        # Find most recent cache file
        cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        cache_file = cache_files[0]

        # Check freshness
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age > (self.cache_hours * 3600):
            logger.debug(f"Cache expired for {ticker} ({file_age/3600:.1f} hours old)")
            return None

        # Load and validate coverage
        cached_data = pd.read_parquet(cache_file)

        # Check date coverage
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        cache_start = cached_data.index.min()
        cache_end = cached_data.index.max()

        tolerance = pd.Timedelta(days=3)
        if (cache_start <= start_dt + tolerance) and (cache_end >= end_dt - tolerance):
            return cached_data
        else:
            logger.debug(f"Cache coverage insufficient for {ticker}")
            return None

    def _save_to_cache(self, ticker: str, data: pd.DataFrame):
        """Save data to cache.

        Args:
            ticker: Stock ticker symbol
            data: DataFrame to cache
        """
        if not self.storage:
            return

        timestamp = datetime.now().strftime('%Y%m%d')
        cache_path = self.storage.base_path / 'raw' / f'{ticker}_{timestamp}.parquet'
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove Ticker column before caching
        data_to_cache = data.copy()
        if 'Ticker' in data_to_cache.columns:
            data_to_cache = data_to_cache.drop('Ticker', axis=1)

        data_to_cache.to_parquet(cache_path, compression='snappy', index=True)
        logger.debug(f"Cached {len(data_to_cache)} rows for {ticker}")

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality.

        Checks:
        - Price positivity (Open, High, Low, Close > 0)
        - Volume non-negativity (Volume ≥ 0)
        - Price relationships (Low ≤ Close ≤ High)
        - Outlier detection (Z-score > 3σ)
        - Missing data rate

        Args:
            data: DataFrame to validate

        Returns:
            Validation report with errors and warnings
        """
        validation_report = {
            'status': 'valid',
            'errors': [],
            'warnings': [],
            'metrics': {}
        }

        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            validation_report['errors'].append(f"Missing required columns: {missing_cols}")
            validation_report['status'] = 'invalid'
            return validation_report

        # Price positivity
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if (data[col] <= 0).any():
                validation_report['errors'].append(f"{col} has non-positive values")

        # Volume non-negativity
        if (data['Volume'] < 0).any():
            validation_report['errors'].append("Volume has negative values")

        # Price relationships
        if ((data['Low'] > data['High']).any() or
            (data['Close'] < data['Low']).any() or
            (data['Close'] > data['High']).any()):
            validation_report['warnings'].append("Invalid price relationships detected")

        # Missing data
        missing_rate = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        validation_report['metrics']['missing_data_rate'] = missing_rate
        if missing_rate > 0.10:
            validation_report['warnings'].append(f"High missing data rate: {missing_rate:.2%}")

        # Outlier detection (Z-score method)
        for col in price_cols:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outlier_rate = (z_scores > 3).sum() / len(data)
            if outlier_rate > 0.01:
                validation_report['warnings'].append(
                    f"{col} has {outlier_rate:.2%} outliers (>3σ)"
                )

        if validation_report['errors']:
            validation_report['status'] = 'invalid'
        elif validation_report['warnings']:
            validation_report['status'] = 'valid_with_warnings'

        return validation_report

    def get_metadata(self, ticker: str, data: pd.DataFrame) -> ExtractorMetadata:
        """Generate metadata about the extraction.

        Args:
            ticker: Stock ticker symbol
            data: Extracted DataFrame

        Returns:
            ExtractorMetadata object with source info and quality metrics
        """
        validation_report = self.validate_data(data)

        # Calculate quality score
        quality_score = 1.0
        if validation_report['errors']:
            quality_score -= 0.5
        quality_score -= len(validation_report['warnings']) * 0.1
        quality_score = max(0.0, min(1.0, quality_score))

        metadata = ExtractorMetadata(
            source_name=self.name,
            ticker=ticker,
            start_date=data.index.min(),
            end_date=data.index.max(),
            num_records=len(data),
            extraction_timestamp=datetime.now(),
            columns=list(data.columns),
            quality_score=quality_score,
            validation_report=validation_report
        )

        return metadata

    def __repr__(self) -> str:
        """String representation."""
        return f"FinnhubExtractor(name='{self.name}', rate_limit={self.requests_per_minute}/min)"

    def __str__(self) -> str:
        """Human-readable string."""
        return (f"Finnhub Extractor\n"
                f"  Resolution: {self.resolution}\n"
                f"  Rate limit: {self.requests_per_minute} requests/minute\n"
                f"  Cache: {self.cache_hours} hours\n"
                f"  Status: Production-ready")
