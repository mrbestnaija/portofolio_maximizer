"""Yahoo Finance data extraction with vectorized quality validation.

Mathematical Foundation:
- Data completeness: missing_rate = 1 - (valid_rows / total_expected_rows)
- Price continuity: gap_threshold = 3� of log returns
- Volume validation: detect zero-volume days (market holidays)

Success Criteria:
- 10+ years daily data for 10 liquid ETFs
- <1% missing values across all tickers
- <5 data gaps per ticker
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """Decorator for exponential backoff retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=30.0)
def fetch_ticker_data(ticker: str, start_date: datetime, end_date: datetime,
                      timeout: int = 30) -> pd.DataFrame:
    """Fetch OHLCV data for single ticker with robust network error handling.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date for data fetch
        end_date: End date for data fetch
        timeout: Request timeout in seconds

    Returns:
        DataFrame with OHLCV data or empty DataFrame on failure
    """
    try:
        # Configure yfinance session with timeout
        ticker_obj = yf.Ticker(ticker)
        ticker_obj.session.request = lambda *args, **kwargs: (
            kwargs.update({'timeout': timeout}),
            ticker_obj.session.request(*args, **kwargs)
        )[1]

        data = yf.download(ticker, start=start_date, end=end_date,
                          progress=False, timeout=timeout)

        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()

        # Flatten MultiIndex columns (yfinance returns MultiIndex for single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data['ticker'] = ticker
        return data

    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Network error fetching {ticker}: {e}")
        raise  # Re-raise for retry decorator
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()

def vectorized_quality_check(data: pd.DataFrame) -> Dict[str, float]:
    """Vectorized data quality metrics calculation."""
    total_rows = len(data)
    missing_rate = data.isnull().sum().sum() / (total_rows * len(data.columns)) if total_rows > 0 else 1.0

    # Vectorized log returns for gap detection
    close_prices = data['Close'].values
    log_returns = np.diff(np.log(close_prices))
    return_std = np.std(log_returns)
    gaps = np.sum(np.abs(log_returns) > 3 * return_std)

    # Zero volume detection (vectorized)
    zero_volume_days = np.sum(data['Volume'].values == 0)

    return {
        'total_rows': total_rows,
        'missing_rate': missing_rate,
        'price_gaps': int(gaps),
        'zero_volume_days': int(zero_volume_days)
    }

def extract_multi_ticker(tickers: List[str], years: int = 10,
                        rate_limit_delay: float = 0.5,
                        timeout: int = 30) -> Tuple[pd.DataFrame, Dict]:
    """Extract and validate data for multiple tickers with rate limiting.

    Args:
        tickers: List of ticker symbols
        years: Number of years of historical data
        rate_limit_delay: Delay between requests in seconds
        timeout: Request timeout in seconds

    Returns:
        Tuple of (combined DataFrame, quality report dict)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)

    all_data = []
    quality_report = {}
    failed_tickers = []

    for i, ticker in enumerate(tickers):
        logger.info(f"Fetching {ticker} ({i+1}/{len(tickers)})...")

        try:
            data = fetch_ticker_data(ticker, start_date, end_date, timeout=timeout)

            if not data.empty:
                quality = vectorized_quality_check(data)
                quality_report[ticker] = quality
                all_data.append(data)

                logger.info(f"{ticker}: {quality['total_rows']} rows, "
                           f"{quality['missing_rate']:.2%} missing, "
                           f"{quality['price_gaps']} gaps")
            else:
                failed_tickers.append(ticker)

        except Exception as e:
            logger.error(f"Failed to process {ticker} after retries: {e}")
            failed_tickers.append(ticker)

        # Rate limiting between requests
        if i < len(tickers) - 1:
            time.sleep(rate_limit_delay)

    if failed_tickers:
        logger.warning(f"Failed tickers ({len(failed_tickers)}): {', '.join(failed_tickers)}")

    combined = pd.concat(all_data) if all_data else pd.DataFrame()
    return combined, quality_report

class YFinanceExtractor:
    """Wrapper class for yfinance extraction operations with network robustness and caching."""

    def __init__(self, timeout: int = 30, rate_limit_delay: float = 0.5,
                 retention_years: int = 10, cleanup_days: int = 7,
                 cache_hours: int = 24, storage=None):
        """Initialize extractor with network configuration, retention policy, and caching.

        Args:
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests in seconds
            retention_years: Years of historical data to maintain locally
            cleanup_days: Auto-cleanup files older than this (days)
            cache_hours: Cache validity duration in hours (default: 24h)
            storage: DataStorage instance for cache operations (optional)
        """
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.retention_years = retention_years
        self.cleanup_days = cleanup_days
        self.cache_hours = cache_hours
        self.storage = storage

    def _check_cache(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Check local cache for recent data matching date range.

        Mathematical Foundation:
        - Cache validity: t_now - t_file ≤ cache_hours × 3600s
        - Coverage check: [cache_start, cache_end] ⊇ [start_date, end_date]

        Args:
            ticker: Stock ticker symbol
            start_date: Requested start date
            end_date: Requested end date

        Returns:
            Cached DataFrame if valid and complete, None otherwise
        """
        if not self.storage:
            return None

        try:
            # Vectorized file lookup: most recent first
            stage_path = self.storage.base_path / 'raw'
            pattern = f"*{ticker}*.parquet"
            files = sorted(stage_path.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)

            if not files:
                return None

            # Check freshness (vectorized time delta)
            latest_file = files[0]
            file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)

            if file_age.total_seconds() > self.cache_hours * 3600:
                logger.info(f"Cache MISS for {ticker}: expired (age: {file_age.total_seconds()/3600:.1f}h)")
                return None

            # Load and validate coverage (vectorized boolean indexing)
            cached_data = pd.read_parquet(latest_file)
            cache_start, cache_end = cached_data.index.min(), cached_data.index.max()

            # Coverage check with 3-day tolerance for non-trading days (weekends, holidays)
            # Allow cache if it covers requested range ±3 days
            tolerance = timedelta(days=3)
            start_ok = cache_start <= start_date + tolerance
            end_ok = cache_end >= end_date - tolerance

            if start_ok and end_ok:
                filtered = cached_data[(cached_data.index >= start_date) & (cached_data.index <= end_date)]
                logger.info(f"Cache HIT for {ticker}: {len(filtered)} rows (age: {file_age.total_seconds()/3600:.1f}h)")
                return filtered
            else:
                logger.info(f"Cache MISS for {ticker}: incomplete coverage (need: {start_date} to {end_date}, have: {cache_start} to {cache_end})")
                return None

        except Exception as e:
            logger.warning(f"Cache lookup failed for {ticker}: {e}")
            return None

    def extract_with_retention(self, tickers: List[str], storage) -> Tuple[Dict, Dict]:
        """Extract data with 10-year retention and auto-cleanup.

        Returns:
            Tuple of (saved_files dict, quality_report dict)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.retention_years*365)

        data, quality = extract_multi_ticker(tickers, years=self.retention_years,
                                            rate_limit_delay=self.rate_limit_delay,
                                            timeout=self.timeout)

        saved_files = {}
        for ticker in tickers:
            ticker_data = data[data['ticker'] == ticker] if not data.empty else pd.DataFrame()
            if not ticker_data.empty:
                filepath = storage.save(ticker_data.drop('ticker', axis=1), 'raw', ticker)
                saved_files[ticker] = str(filepath)

        # Auto-cleanup old files (vectorized in storage)
        storage.cleanup_old_files('raw', retention_days=self.cleanup_days)

        return saved_files, quality

    def extract_ohlcv(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Extract OHLCV data with cache-first strategy.

        Mathematical Foundation:
        - Cache hit rate: η = n_cached / n_total
        - Network efficiency: reduce API calls by factor of (1 - η)

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Combined DataFrame with OHLCV data for all tickers
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        all_data = []
        cache_hits = 0
        cache_misses = 0

        for i, ticker in enumerate(tickers):
            try:
                # Cache-first strategy: check local storage before network request
                cached_data = self._check_cache(ticker, start, end)

                if cached_data is not None and not cached_data.empty:
                    # Cache HIT: use local data
                    cached_data['ticker'] = ticker
                    all_data.append(cached_data)
                    cache_hits += 1
                else:
                    # Cache MISS: fetch from network
                    data = fetch_ticker_data(ticker, start, end, timeout=self.timeout)
                    if not data.empty:
                        data['ticker'] = ticker
                        all_data.append(data)
                        cache_misses += 1

                        # Save to cache for future use (if storage available)
                        if self.storage:
                            try:
                                data_to_save = data.drop('ticker', axis=1) if 'ticker' in data.columns else data
                                # Flatten MultiIndex columns if present (yfinance returns MultiIndex)
                                if isinstance(data_to_save.columns, pd.MultiIndex):
                                    data_to_save.columns = data_to_save.columns.get_level_values(0)
                                self.storage.save(data_to_save, 'raw', ticker)
                            except Exception as e:
                                logger.warning(f"Failed to cache {ticker}: {e}")

                    # Rate limiting (only for network requests)
                    if i < len(tickers) - 1 and cached_data is None:
                        time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Skipping {ticker} after failed retries: {e}")
                cache_misses += 1
                continue

        # Log cache performance
        total = cache_hits + cache_misses
        if total > 0:
            hit_rate = cache_hits / total
            logger.info(f"Cache performance: {cache_hits}/{total} hits ({hit_rate:.1%} hit rate)")

        if not all_data:
            return pd.DataFrame()

        # Reset index before concat to avoid duplicate indices
        combined = pd.concat([df.reset_index() for df in all_data], ignore_index=True)

        # Set Date as index
        if 'Date' in combined.columns:
            combined.set_index('Date', inplace=True)

        return combined