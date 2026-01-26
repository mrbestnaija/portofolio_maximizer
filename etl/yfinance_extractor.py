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
import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import time
from functools import wraps
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

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
                      timeout: int = 30,
                      interval: Optional[str] = None,
                      auto_adjust: bool = False) -> pd.DataFrame:
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
        download_kwargs: Dict[str, Any] = {
            "start": start_date,
            "end": end_date,
            "progress": False,
            "timeout": timeout,
            "auto_adjust": auto_adjust,
        }
        if interval:
            download_kwargs["interval"] = interval

        data = yf.download(ticker, **download_kwargs)

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
                 failure_backoff_hours: int = 6, config_path: Optional[str] = None, **kwargs):
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
        resolved_config_path = config_path or kwargs.pop("config_path", None)
        self.config_path = Path(resolved_config_path) if resolved_config_path else None
        self.config: Dict[str, Any] = self._load_config(self.config_path) if self.config_path else {}
        self._configure_provider_cache()

        # Apply config-driven defaults (fall back to constructor args when missing).
        self.interval = self._get_config_value(("extraction", "data", "interval"))
        env_interval = os.getenv("YFINANCE_INTERVAL") or os.getenv("YFINANCE_DATA_INTERVAL")
        if env_interval:
            self.interval = str(env_interval).strip()
        self.auto_adjust = bool(self._get_config_value(("extraction", "data", "auto_adjust"), False) or False)

        cache_hours = int(self._get_config_value(("extraction", "cache", "cache_hours"), cache_hours) or cache_hours)
        timeout = int(self._get_config_value(("extraction", "network", "timeout_seconds"), timeout) or timeout)

        self.cache_tolerance_days = int(
            self._get_config_value(("extraction", "cache", "tolerance_days"), 3) or 3
        )
        self.min_rows_required = int(
            self._get_config_value(("extraction", "quality_checks", "min_rows_required"), 10) or 10
        )
        cleanup_days = int(
            self._get_config_value(("extraction", "cache", "retention_days"), cleanup_days) or cleanup_days
        )
        self.auto_cleanup_cache = bool(
            self._get_config_value(("extraction", "cache", "auto_cleanup"), False) or False
        )

        enabled_rate_limit = self._get_config_value(("extraction", "rate_limiting", "enabled"), True)
        configured_delay = self._get_config_value(("extraction", "rate_limiting", "delay_between_tickers"))
        rpm = self._get_config_value(("extraction", "rate_limiting", "requests_per_minute"))
        if not enabled_rate_limit:
            configured_delay = 0
            rpm = None

        rate_limit_delay = self._resolve_rate_limit_delay(
            default_delay=rate_limit_delay,
            configured_delay=configured_delay,
            requests_per_minute=rpm,
        )

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
        # Reduce noise for known sentinel/fake tickers used in tests or housekeeping.
        self._quiet_failure_tickers = {"DELISTED", "MISSING", "INVALID"}

    def _configure_provider_cache(self) -> None:
        """Route yfinance's internal sqlite caches to a writable location."""
        cache_dir = os.getenv("YFINANCE_CACHE_DIR") or os.getenv("YFINANCE_TZ_CACHE_DIR")
        if not cache_dir and os.name == "posix":
            is_wsl = bool(os.getenv("WSL_DISTRO_NAME"))
            if not is_wsl:
                try:
                    is_wsl = "microsoft" in Path("/proc/version").read_text().lower()
                except OSError:
                    is_wsl = False
            if is_wsl:
                cache_dir = str(Path(os.getenv("WSL_SQLITE_TMP", "/tmp")) / "py-yfinance")

        if not cache_dir:
            return

        try:
            cache_path = Path(cache_dir).expanduser()
            cache_path.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(cache_path))
            logger.info("yfinance cache dir set to %s", cache_path)
        except Exception as exc:
            logger.warning("Unable to set yfinance cache dir (%s): %s", cache_dir, exc)

    def _should_skip_ticker(self, ticker: str) -> bool:
        """Skip tickers that recently failed to reduce noisy retries."""
        failure_time = self._recent_failures.get(ticker)
        if not failure_time:
            return False

        elapsed = datetime.now() - failure_time
        if elapsed <= timedelta(hours=self.failure_backoff_hours):
            self._record_cache_event(ticker, "skipped", 0)
            log_fn = logger.info if ticker in self._quiet_failure_tickers else logger.warning
            log_fn(
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
        log_fn = logger.info if ticker in self._quiet_failure_tickers else logger.warning
        log_fn("Marked %s as failed (reason: %s); will back off", ticker, reason)

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

    @staticmethod
    def _load_config(config_path: Optional[Path]) -> Dict[str, Any]:
        """Load the provider YAML config file (if available)."""
        if not config_path:
            return {}
        if yaml is None:
            logger.debug("pyyaml unavailable; skipping yfinance config load from %s", config_path)
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            return loaded if isinstance(loaded, dict) else {}
        except FileNotFoundError:
            logger.warning("YFinance config file not found: %s", config_path)
            return {}
        except Exception as exc:
            logger.warning("Failed to load yfinance config from %s: %s", config_path, exc)
            return {}

    def _get_config_value(self, path: Tuple[str, ...], default: Any = None) -> Any:
        node: Any = self.config
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node.get(key)
        return default if node is None else node

    @staticmethod
    def _resolve_rate_limit_delay(
        default_delay: float,
        configured_delay: Any,
        requests_per_minute: Any,
    ) -> float:
        delay = default_delay
        if configured_delay is not None:
            try:
                delay = float(configured_delay)
            except Exception:
                delay = default_delay

        min_delay_from_rpm: Optional[float] = None
        if requests_per_minute is not None:
            try:
                rpm = float(requests_per_minute)
                if rpm > 0:
                    min_delay_from_rpm = 60.0 / rpm
            except Exception:
                min_delay_from_rpm = None

        if min_delay_from_rpm is not None:
            delay = max(delay, min_delay_from_rpm)

        return max(0.0, delay)

    def _effective_cache_hours(self) -> float:
        """
        Adjust cache TTL for market hours/weekends to avoid needless churn.
        """
        base = float(getattr(self, "cache_hours", 0) or 0)
        now = datetime.utcnow()
        weekday = now.weekday()
        hour = now.hour
        # US market hours approx 13:30-20:00 UTC; keep base TTL there, extend off-hours/weekends.
        if weekday >= 5:
            return max(base, base * 2 or 48.0)  # weekends: allow 48h+ if base missing
        if hour < 12 or hour > 21:
            return max(base, base * 1.5)
        return base

    def _write_cache_perf_artifact(self, *, tickers: List[str], start: datetime, end: datetime) -> None:
        if str(os.getenv("CACHE_PERF_ARTIFACTS") or "0") != "1":
            return
        try:
            out_dir = Path("logs") / "performance"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "source": self.name,
                "tickers": list(tickers),
                "requested_start": start.isoformat(),
                "requested_end": end.isoformat(),
                "cache_hours": float(getattr(self, "cache_hours", 0) or 0),
                "effective_cache_hours": float(self._effective_cache_hours()),
                "events": dict(self._cache_events),
            }
            path = out_dir / f"cache_perf_{ts}.json"
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            logger.info("Wrote cache perf artifact: %s", path)
        except Exception:
            logger.debug("Unable to write cache perf artifact", exc_info=True)

    def _load_latest_cached_frame(self, ticker: str) -> Optional[pd.DataFrame]:
        if not self.storage:
            return None
        try:
            stage_path = self.storage.base_path / 'raw'
            files = sorted(
                stage_path.glob(f"*{ticker}*.parquet"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            if not files:
                return None
            return pd.read_parquet(files[0])
        except Exception:
            return None

    def _maybe_fetch_tail_delta(
        self,
        ticker: str,
        cached_data: pd.DataFrame,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch missing tail bars when cache is partial and delta updates are allowed.
        """
        if cached_data is None or cached_data.empty:
            return None
        try:
            last_ts = pd.to_datetime(cached_data.index.max())
        except Exception:
            return None

        # Only fetch when we clearly lag the requested end.
        if last_ts >= end_date:
            return None

        # Buffer by 1 day to cover boundary gaps.
        start = last_ts - timedelta(days=1)
        try:
            delta_df = fetch_ticker_data(
                ticker=ticker,
                start_date=start,
                end_date=end_date,
                timeout=self.timeout,
                interval=self.interval,
                auto_adjust=self.auto_adjust,
            )
        except Exception:
            return None

        if delta_df is None or delta_df.empty:
            return None

        if isinstance(delta_df.columns, pd.MultiIndex):
            delta_df.columns = delta_df.columns.get_level_values(0)
        delta_df["ticker"] = ticker
        combined = pd.concat([cached_data, delta_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined

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
        if getattr(self, "cache_hours", 0) <= 0:
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
            file_age_seconds = max(0.0, time.time() - latest_file.stat().st_mtime)
            file_age = timedelta(seconds=file_age_seconds)

            effective_cache_hours = self._effective_cache_hours()
            if file_age_seconds >= effective_cache_hours * 3600:
                # With delta-refresh enabled, allow using stale cache as a base so
                # we can fetch only the missing tail bars.
                if str(os.getenv("ENABLE_CACHE_DELTAS") or "0") == "1":
                    logger.info(
                        "Cache STALE for %s (age %.1fh), allowing delta refresh base",
                        ticker,
                        file_age.total_seconds() / 3600,
                    )
                else:
                    logger.info(
                        "Cache MISS for %s: expired (age: %.1fh >= %.1fh effective TTL)",
                        ticker,
                        file_age.total_seconds() / 3600,
                        effective_cache_hours,
                    )
                    self._record_cache_event(ticker, "miss", 0)
                    return None

            # Load and validate coverage (vectorized boolean indexing)
            cached_data = pd.read_parquet(latest_file)
            cache_start, cache_end = cached_data.index.min(), cached_data.index.max()

            # Coverage check with tolerance for non-trading days (weekends, holidays)
            tolerance = timedelta(days=self.cache_tolerance_days)
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
            t_cycle = time.perf_counter()
            try:
                if self._should_skip_ticker(ticker):
                    failed_tickers.append(ticker)
                    continue

                # Cache-first strategy: check local storage before network request
                t_cache0 = time.perf_counter()
                cached_data = self._check_cache(ticker, start, end)
                t_cache1 = time.perf_counter()
                event = self._cache_events.get(ticker, {})
                coverage = event.get("status")

                if cached_data is not None and not cached_data.empty:
                    # Cache HIT: use local data (optionally patched with tail delta)
                    if str(os.getenv("ENABLE_CACHE_DELTAS") or "0") == "1":
                        patched = self._maybe_fetch_tail_delta(ticker, cached_data, end)
                        if patched is not None and not patched.empty:
                            cached_data = patched
                            coverage = "delta_refresh"
                    cached_data['ticker'] = ticker
                    all_data.append(cached_data)
                    cache_hits += 1
                    if coverage == "partial":
                        cache_partial_hits += 1
                    self._cache_events.setdefault(ticker, {}).update(
                        {
                            "cache_check_ms": round((t_cache1 - t_cache0) * 1000.0, 3),
                            "total_ms": round((time.perf_counter() - t_cycle) * 1000.0, 3),
                        }
                    )
                    continue

                # Cache MISS: fetch from network
                data = fetch_ticker_data(
                    ticker,
                    start,
                    end,
                    timeout=self.timeout,
                    interval=self.interval,
                    auto_adjust=self.auto_adjust,
                )
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

                self._cache_events.setdefault(ticker, {}).update(
                    {
                        "cache_check_ms": round((t_cache1 - t_cache0) * 1000.0, 3),
                        "total_ms": round((time.perf_counter() - t_cycle) * 1000.0, 3),
                    }
                )

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

        quiet_failed = {t for t in failed_tickers if t in self._quiet_failure_tickers}
        noisy_failed = {t for t in failed_tickers if t not in self._quiet_failure_tickers}

        if not all_data:
            if noisy_failed:
                logger.warning(
                    "No data returned; failed tickers: %s",
                    ", ".join(sorted(noisy_failed)),
                )
            if quiet_failed:
                logger.info(
                    "No data returned; skipped quiet tickers: %s",
                    ", ".join(sorted(quiet_failed)),
                )
            return pd.DataFrame()
        elif failed_tickers:
            if noisy_failed:
                logger.warning(
                    "Partial success; failed tickers skipped: %s",
                    ", ".join(sorted(noisy_failed)),
                )
            if quiet_failed:
                logger.info(
                    "Partial success; skipped quiet tickers: %s",
                    ", ".join(sorted(quiet_failed)),
                )

        # Reset index before concat to avoid duplicate indices.
        combined = pd.concat([df.reset_index() for df in all_data], ignore_index=True)

        # Preserve timestamps across intervals (daily uses `Date`, intraday uses `Datetime`).
        datetime_col = None
        for candidate in ("Date", "Datetime", "date", "datetime"):
            if candidate in combined.columns:
                datetime_col = candidate
                break
        if datetime_col:
            combined[datetime_col] = pd.to_datetime(combined[datetime_col], errors="coerce")
            combined = combined.loc[combined[datetime_col].notna()].copy()
            combined.set_index(datetime_col, inplace=True)
            combined.index.name = "Date"
            combined.sort_index(inplace=True)

        # Standardize columns and flatten MultiIndex (from BaseExtractor)
        combined = self._flatten_multiindex(combined)
        combined = self._standardize_columns(combined)

        if self.storage and self.auto_cleanup_cache and self.cleanup_days > 0:
            try:
                self.storage.cleanup_old_files('raw', retention_days=self.cleanup_days)
            except Exception as exc:
                logger.debug("Auto-cleanup skipped: %s", exc)

        self._write_cache_perf_artifact(tickers=tickers, start=start, end=end)
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
        if self.min_rows_required and total_rows < self.min_rows_required:
            warnings.append(f"Low observation count: {total_rows} < {self.min_rows_required}")
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
