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
import json
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import time
from functools import wraps

from etl.base_extractor import BaseExtractor, ExtractorMetadata

logger = logging.getLogger(__name__)

def _is_unrecoverable_ticker_error(error: Exception) -> bool:
    """Return True when the ticker is clearly unavailable/delisted."""
    try:
        from yfinance.shared import YFTzMissingError, YFPricesMissingError  # type: ignore
        if isinstance(error, (YFTzMissingError, YFPricesMissingError)):
            return True
    except Exception:
        # yfinance internals may change; fall back to string matching
        pass

    message = str(error).lower()
    substrings = [
        "possibly delisted",
        "no timezone found",
        "no price data found",
        "no data fetched",
    ]
    return any(text in message for text in substrings)


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
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            timeout=timeout,
            auto_adjust=False,
        )

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
        if _is_unrecoverable_ticker_error(e):
            logger.warning(f"Skipping {ticker}: {e} (likely stale/delisted)")
            return pd.DataFrame()

        logger.error(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()

def vectorized_quality_check(data: pd.DataFrame) -> Dict[str, float]:
    """Vectorized data quality metrics calculation."""
    total_rows = len(data)
    missing_rate = data.isnull().sum().sum() / (total_rows * len(data.columns)) if total_rows > 0 else 1.0
    gaps = 0
    return_std = 0.0

    # Vectorized log returns for gap detection (guard for non-positive/short series)
    close_series = data['Close'] if 'Close' in data.columns else pd.Series(dtype=float)
    close_series = close_series.dropna()
    positive_close = close_series[close_series > 0]
    if len(positive_close) >= 2:
        log_returns = np.diff(np.log(positive_close.to_numpy(dtype=float, copy=False)))
        if log_returns.size:
            return_std = float(np.std(log_returns))
            if return_std > 0:
                gaps = int(np.sum(np.abs(log_returns) > 3 * return_std))

    # Zero volume detection (vectorized)
    zero_volume_days = int(np.sum(data['Volume'].values == 0)) if 'Volume' in data.columns else 0

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

class YFinanceExtractor(BaseExtractor):
    """Yahoo Finance extractor with network robustness and caching.

    Inherits from BaseExtractor to provide standardized interface for data extraction.
    """

    def __init__(self, name: str = 'yfinance', timeout: int = 30,
                 rate_limit_delay: float = 0.5, retention_years: int = 10,
                 cleanup_days: int = 7, cache_hours: int = 24, storage=None,
                 failure_backoff_hours: int = 6, **kwargs):
        """Initialize Yahoo Finance extractor.

        Args:
            name: Data source name (default: 'yfinance')
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests in seconds
            retention_years: Years of historical data to maintain locally
            cleanup_days: Auto-cleanup files older than this (days)
            cache_hours: Cache validity duration in hours (default: 24h)
            storage: DataStorage instance for cache operations (optional)
            **kwargs: Additional parameters (for BaseExtractor compatibility)
        """
        # Initialize parent class
        super().__init__(name=name, timeout=timeout, cache_hours=cache_hours,
                        storage=storage, **kwargs)

        # YFinance-specific attributes
        self.rate_limit_delay = rate_limit_delay
        self.retention_years = retention_years
        self.cleanup_days = cleanup_days
        self._cache_events: Dict[str, Dict[str, Any]] = {}
        self.failure_backoff_hours = failure_backoff_hours
        self._recent_failures: Dict[str, datetime] = {}

    def _should_skip_ticker(self, ticker: str) -> bool:
        """Skip tickers that recently failed to reduce noisy retries."""
        failure_time = self._recent_failures.get(ticker)
        if not failure_time:
            return False

        elapsed = datetime.now() - failure_time
        if elapsed <= timedelta(hours=self.failure_backoff_hours):
            self._record_cache_event(ticker, "skipped", 0)
            logger.warning(
                "Skipping %s: last failure %.1fh ago (backoff %sh)",
                ticker,
                elapsed.total_seconds() / 3600,
                self.failure_backoff_hours,
            )
            return True

        # Backoff window expired; allow fetch and clear flag
        self._recent_failures.pop(ticker, None)
        return False

    def _mark_failure(self, ticker: str, reason: str) -> None:
        """Mark ticker as failed to avoid noisy repeated downloads."""
        self._recent_failures[ticker] = datetime.now()
        self._record_cache_event(ticker, "failed", 0)
        logger.warning("Marked %s as failed (reason: %s); will back off", ticker, reason)

    def _load_cache_metadata(self, parquet_path) -> Dict[str, Any]:
        """Load metadata saved alongside cached parquet file."""
        metadata_path = parquet_path.with_suffix('.meta.json')
        if not metadata_path.exists():
            return {}

        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

        return meta

    def _record_cache_event(self, ticker: str, status: str, rows: int = 0) -> None:
        """Capture cache status for diagnostics."""
        self._cache_events[ticker] = {"status": status, "rows": rows}

    def _check_cache(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
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
            self._record_cache_event(ticker, "miss", 0)
            return None

        try:
            # Vectorized file lookup: most recent first
            stage_path = self.storage.base_path / 'raw'
            pattern = f"*{ticker}*.parquet"
            files = sorted(stage_path.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)

            if not files:
                self._record_cache_event(ticker, "miss", 0)
                return None

            # Check freshness (vectorized time delta)
            latest_file = files[0]
            file_age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)

            if file_age.total_seconds() > self.cache_hours * 3600:
                logger.info(f"Cache MISS for {ticker}: expired (age: {file_age.total_seconds()/3600:.1f}h)")
                self._record_cache_event(ticker, "miss", 0)
                return None

            # Load and validate coverage (vectorized boolean indexing)
            cached_data = pd.read_parquet(latest_file)
            cache_start, cache_end = cached_data.index.min(), cached_data.index.max()

            # Coverage check with 3-day tolerance for non-trading days (weekends, holidays)
            # Allow cache if it covers requested range ±3 days
            tolerance = timedelta(days=3)
            start_ok = cache_start <= start_date + tolerance
            end_ok = cache_end >= end_date - tolerance

            filtered = cached_data[(cached_data.index >= start_date) & (cached_data.index <= end_date)]

            if start_ok and end_ok:
                logger.info(
                    f"Cache HIT for {ticker}: {len(filtered)} rows (age: {file_age.total_seconds()/3600:.1f}h)"
                )
                self._record_cache_event(ticker, "full", len(filtered))
                return filtered

            cache_meta = self._load_cache_metadata(latest_file)
            requested_start = cache_meta.get('requested_start')
            requested_end = cache_meta.get('requested_end')

            partial_allowed = False
            if requested_start and requested_end:
                try:
                    requested_start_dt = pd.to_datetime(requested_start)
                    requested_end_dt = pd.to_datetime(requested_end)
                except Exception:
                    requested_start_dt = None
                    requested_end_dt = None

                if requested_start_dt and requested_end_dt:
                    if requested_start_dt <= start_date and requested_end_dt >= end_date:
                        partial_allowed = True

            if partial_allowed:
                gaps = []
                if not start_ok:
                    gaps.append(f"start gap ({cache_start.date()} vs {start_date.date()})")
                if not end_ok:
                    gaps.append(f"end gap ({cache_end.date()} vs {end_date.date()})")
                gap_msg = ", ".join(gaps) if gaps else "partial coverage"
                logger.info(
                    f"Cache HIT (partial) for {ticker}: {len(filtered)} rows available; {gap_msg}. "
                    "Returning cached range to avoid redundant downloads."
                )
                self._record_cache_event(ticker, "partial", len(filtered))
                return filtered

            logger.info(
                f"Cache MISS for {ticker}: incomplete coverage (need: {start_date} to {end_date}, "
                f"have: {cache_start} to {cache_end})"
            )
            self._record_cache_event(ticker, "miss", 0)
            return None

        except Exception as e:
            logger.warning(f"Cache lookup failed for {ticker}: {e}")
            self._record_cache_event(ticker, "miss", 0)
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
                ticker_df = ticker_data.drop('ticker', axis=1)
                data_start = ticker_df.index.min().to_pydatetime() if len(ticker_df) > 0 else None
                data_end = ticker_df.index.max().to_pydatetime() if len(ticker_df) > 0 else None
                metadata = {
                    'requested_start': start_date.isoformat(),
                    'requested_end': end_date.isoformat(),
                    'data_start': data_start.isoformat() if data_start else None,
                    'data_end': data_end.isoformat() if data_end else None,
                    'row_count': int(len(ticker_df)),
                }
                filepath = storage.save(ticker_df, 'raw', ticker, metadata=metadata)
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
        cache_partial_hits = 0
        self._cache_events.clear()
        failed_tickers: List[str] = []

        for i, ticker in enumerate(tickers):
            try:
                if self._should_skip_ticker(ticker):
                    failed_tickers.append(ticker)
                    continue

                # Cache-first strategy: check local storage before network request
                cached_data = self._check_cache(ticker, start, end)
                event = self._cache_events.get(ticker, {})
                coverage = event.get("status")

                if cached_data is not None and not cached_data.empty:
                    # Cache HIT: use local data
                    cached_data['ticker'] = ticker
                    all_data.append(cached_data)
                    cache_hits += 1
                    if coverage == "partial":
                        cache_partial_hits += 1
                    continue

                # Cache MISS: fetch from network
                data = fetch_ticker_data(ticker, start, end, timeout=self.timeout)
                cache_misses += 1
                if not data.empty:
                    data['ticker'] = ticker
                    all_data.append(data)

                    # Save to cache for future use (if storage available)
                    if self.storage:
                        try:
                            data_to_save = data.drop('ticker', axis=1) if 'ticker' in data.columns else data
                            # Flatten MultiIndex columns if present (yfinance returns MultiIndex)
                            if isinstance(data_to_save.columns, pd.MultiIndex):
                                data_to_save.columns = data_to_save.columns.get_level_values(0)

                            data_start = data_to_save.index.min().to_pydatetime() if len(data_to_save) > 0 else None
                            data_end = data_to_save.index.max().to_pydatetime() if len(data_to_save) > 0 else None
                            metadata = {
                                'requested_start': start.isoformat(),
                                'requested_end': end.isoformat(),
                                'data_start': data_start.isoformat() if data_start else None,
                                'data_end': data_end.isoformat() if data_end else None,
                                'row_count': int(len(data_to_save)),
                            }

                            self.storage.save(data_to_save, 'raw', ticker, metadata=metadata)
                            self._record_cache_event(ticker, "refreshed", len(data_to_save))
                        except Exception as e:
                            logger.warning(f"Failed to cache {ticker}: {e}")
                else:
                    failed_tickers.append(ticker)
                    self._mark_failure(ticker, "no data returned from yfinance")

                # Rate limiting (only for network requests)
                if i < len(tickers) - 1:
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                failed_tickers.append(ticker)
                self._mark_failure(ticker, str(e))
                logger.error(f"Skipping {ticker} after failed retries: {e}")
                cache_misses += 1
                continue

        # Update cache statistics (for BaseExtractor)
        self._cache_hits += cache_hits
        self._cache_misses += cache_misses
        self._cache_partials += cache_partial_hits

        # Log cache performance
        total = cache_hits + cache_misses
        if total > 0:
            hit_rate = cache_hits / total
            partial_note = f" ({cache_partial_hits} partial)" if cache_partial_hits else ""
            logger.info(f"Cache performance: {cache_hits}/{total} hits ({hit_rate:.1%} hit rate){partial_note}")

        if not all_data:
            if failed_tickers:
                logger.warning(
                    "No data returned; failed tickers: %s",
                    ", ".join(sorted(set(failed_tickers))),
                )
            return pd.DataFrame()
        elif failed_tickers:
            logger.warning(
                "Partial success; failed tickers skipped: %s",
                ", ".join(sorted(set(failed_tickers))),
            )

        # Reset index before concat to avoid duplicate indices
        combined = pd.concat([df.reset_index() for df in all_data], ignore_index=True)

        # Set Date as index
        if 'Date' in combined.columns:
            combined.set_index('Date', inplace=True)

        # Standardize columns and flatten MultiIndex (from BaseExtractor)
        combined = self._flatten_multiindex(combined)
        combined = self._standardize_columns(combined)

        return combined

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return validation report.

        Implements BaseExtractor abstract method.

        Args:
            data: DataFrame to validate

        Returns:
            Dictionary containing validation results
        """
        errors = []
        warnings = []
        metrics = {}

        # Check if data is empty
        if data.empty:
            errors.append("DataFrame is empty")
            return {
                'passed': False,
                'errors': errors,
                'warnings': warnings,
                'quality_score': 0.0,
                'metrics': metrics
            }

        # Calculate missing data rate
        total_rows = len(data)
        total_cells = total_rows * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        missing_rate = missing_cells / total_cells if total_cells > 0 else 1.0
        metrics['missing_rate'] = missing_rate

        if missing_rate > 0.01:  # >1% missing
            warnings.append(f"High missing data rate: {missing_rate:.2%}")

        # Vectorized price validation (if Close column exists)
        if 'Close' in data.columns:
            close_prices = data['Close'].dropna()
            if len(close_prices) > 0:
                # Check for non-positive prices
                if (close_prices <= 0).any():
                    errors.append("Non-positive prices detected")

                # Detect price gaps using log returns
                log_returns = np.diff(np.log(close_prices))
                if len(log_returns) > 0:
                    return_std = np.std(log_returns)
                    gaps = np.sum(np.abs(log_returns) > 3 * return_std)
                    metrics['gap_count'] = int(gaps)

                    if gaps > 5:
                        warnings.append(f"High number of price gaps: {gaps}")

        # Volume validation
        if 'Volume' in data.columns:
            volumes = data['Volume'].dropna()
            if len(volumes) > 0:
                # Check for negative volumes
                if (volumes < 0).any():
                    errors.append("Negative volumes detected")

                # Zero volume detection
                zero_volume_days = int(np.sum(volumes == 0))
                metrics['zero_volume_days'] = zero_volume_days

                if zero_volume_days > total_rows * 0.1:  # >10% zero volume
                    warnings.append(f"High zero-volume days: {zero_volume_days}")

        # Outlier detection (Z-score method)
        outlier_count = 0
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                values = data[col].dropna()
                if len(values) > 0:
                    z_scores = np.abs((values - values.mean()) / values.std())
                    outlier_count += int(np.sum(z_scores > 3))

        metrics['outlier_count'] = outlier_count
        if outlier_count > total_rows * 0.05:  # >5% outliers
            warnings.append(f"High outlier count: {outlier_count}")

        # Calculate quality score (0-1)
        quality_score = 1.0
        quality_score -= min(missing_rate * 10, 0.5)  # Missing data penalty
        quality_score -= min(metrics.get('gap_count', 0) / 100, 0.3)  # Gap penalty
        quality_score -= min(outlier_count / total_rows, 0.2)  # Outlier penalty
        quality_score = max(0.0, quality_score)

        metrics['quality_score'] = quality_score

        # Determine pass/fail
        passed = len(errors) == 0 and quality_score >= 0.7

        return {
            'passed': passed,
            'errors': errors,
            'warnings': warnings,
            'quality_score': quality_score,
            'metrics': metrics
        }

    def get_metadata(self, ticker: str, data: pd.DataFrame) -> ExtractorMetadata:
        """Generate metadata for extracted data.

        Implements BaseExtractor abstract method.

        Args:
            ticker: Stock ticker symbol
            data: Extracted DataFrame

        Returns:
            ExtractorMetadata object with extraction details
        """
        extraction_timestamp = datetime.now()

        if data.empty:
            return ExtractorMetadata(
                ticker=ticker,
                source=self.name,
                extraction_timestamp=extraction_timestamp,
                data_start_date=None,
                data_end_date=None,
                row_count=0,
                cache_hit=False
            )

        # Get date range from index
        data_start_date = data.index.min().to_pydatetime() if hasattr(data.index.min(), 'to_pydatetime') else data.index.min()
        data_end_date = data.index.max().to_pydatetime() if hasattr(data.index.max(), 'to_pydatetime') else data.index.max()

        return ExtractorMetadata(
            ticker=ticker,
            source=self.name,
            extraction_timestamp=extraction_timestamp,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            row_count=len(data),
            cache_hit=False  # Set by caller if from cache
        )
