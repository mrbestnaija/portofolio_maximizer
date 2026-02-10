"""OpenBB Platform multi-provider data extractor with priority-based fallback.

Provides a unified interface to multiple financial data providers via the OpenBB SDK.
Implements the BaseExtractor contract with:
- Priority-based provider fallback chain (yfinance -> polygon -> alpha_vantage -> finnhub)
- Per-provider independent rate limiting
- Parquet cache-first strategy (24h TTL, reuses existing data/raw/ directory)
- Nigeria market support (.NG suffix, LUNO crypto pairs)
- Lazy OpenBB SDK initialization to avoid import-time side effects

Provider Fallback Flow:
    Cache check --> HIT? Return
      |  MISS
      v
    yfinance (p1) --> polygon (p2) --> alpha_vantage (p3) --> finnhub (p4)
      |  All fail?
      v
    Return empty DataFrame, log error

Mathematical Foundation:
- Cache validity: t_now - t_file <= cache_hours * 3600s
- Coverage check: [cache_start, cache_end] >= [start_date, end_date]
- Missing rate: 1 - (valid_rows / total_expected_rows)
- Price continuity: gap_threshold = 3 sigma of log returns
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from etl.base_extractor import BaseExtractor, ExtractorMetadata
from etl.secret_loader import (
    bootstrap_dotenv,
    load_alpha_vantage_key,
    load_finnhub_key,
    load_luno_credentials,
    load_polygon_key,
    load_secret,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OpenBBProviderConfig:
    """Configuration for a single OpenBB data provider."""

    name: str
    enabled: bool = True
    priority: int = 99
    credentials_env: Optional[str] = None
    rate_limit_rpm: int = 60


@dataclass
class OpenBBConfig:
    """Full configuration for the OpenBB extractor."""

    providers: List[OpenBBProviderConfig] = field(default_factory=list)
    cache_hours: int = 24
    cache_dir: str = "data/raw"
    min_rows: int = 10
    max_missing_rate: float = 0.05
    max_gap_days: int = 5
    retry_max: int = 2
    retry_base_delay: float = 2.0
    retry_max_delay: float = 30.0
    nigeria_enabled: bool = True
    nigeria_suffix: str = ".NG"
    luno_pairs: List[str] = field(
        default_factory=lambda: ["XBTNGN", "ETHNGN", "XRPNGN"]
    )
    luno_api_key_env: str = "API_KEY_LUNO"
    luno_api_id_env: str = "API_ID_LUNO"
    log_provider_attempts: bool = True
    log_fallback_events: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OpenBBConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Populated OpenBBConfig instance.
        """
        if yaml is None:
            logger.warning("PyYAML not installed; using default OpenBB config")
            return cls._defaults()

        path = Path(path)
        if not path.exists():
            logger.warning("OpenBB config not found at %s; using defaults", path)
            return cls._defaults()

        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.error("Failed to parse %s: %s", path, exc)
            return cls._defaults()

        obb_cfg = raw.get("openbb", {})

        # Parse providers
        providers: List[OpenBBProviderConfig] = []
        for p in obb_cfg.get("providers", []):
            providers.append(
                OpenBBProviderConfig(
                    name=p.get("name", "unknown"),
                    enabled=p.get("enabled", True),
                    priority=p.get("priority", 99),
                    credentials_env=p.get("credentials_env"),
                    rate_limit_rpm=p.get("rate_limit_rpm", 60),
                )
            )
        # Sort by priority ascending
        providers.sort(key=lambda x: x.priority)

        cache_cfg = obb_cfg.get("cache", {})
        val_cfg = obb_cfg.get("validation", {})
        retry_cfg = obb_cfg.get("retry", {})
        ng_cfg = obb_cfg.get("nigeria", {})
        log_cfg = obb_cfg.get("logging", {})

        return cls(
            providers=providers,
            cache_hours=cache_cfg.get("cache_hours", 24),
            cache_dir=cache_cfg.get("cache_dir", "data/raw"),
            min_rows=val_cfg.get("min_rows", 10),
            max_missing_rate=val_cfg.get("max_missing_rate", 0.05),
            max_gap_days=val_cfg.get("max_gap_days", 5),
            retry_max=retry_cfg.get("max_retries", 2),
            retry_base_delay=retry_cfg.get("base_delay_seconds", 2.0),
            retry_max_delay=retry_cfg.get("max_delay_seconds", 30.0),
            nigeria_enabled=ng_cfg.get("enabled", True),
            nigeria_suffix=ng_cfg.get("ticker_suffix", ".NG"),
            luno_pairs=ng_cfg.get("luno_pairs", {}).get("pairs", [])
            if isinstance(ng_cfg.get("luno_pairs"), dict)
            else ng_cfg.get("luno_pairs", ["XBTNGN", "ETHNGN", "XRPNGN"]),
            luno_api_key_env=ng_cfg.get("luno_credentials", {}).get(
                "api_key_env", "API_KEY_LUNO"
            ),
            luno_api_id_env=ng_cfg.get("luno_credentials", {}).get(
                "api_id_env", "API_ID_LUNO"
            ),
            log_provider_attempts=log_cfg.get("log_provider_attempts", True),
            log_fallback_events=log_cfg.get("log_fallback_events", True),
        )

    @classmethod
    def _defaults(cls) -> "OpenBBConfig":
        """Return sensible defaults when YAML is unavailable."""
        return cls(
            providers=[
                OpenBBProviderConfig("yfinance", True, 1, None, 60),
                OpenBBProviderConfig("polygon", True, 2, "MASSIVE_API_KEY", 5),
                OpenBBProviderConfig(
                    "alpha_vantage", True, 3, "ALPHA_VANTAGE_API_KEY", 5
                ),
                OpenBBProviderConfig("finnhub", True, 4, "FINNHUB_API_KEY", 60),
            ]
        )


# ---------------------------------------------------------------------------
# Per-provider rate limiter
# ---------------------------------------------------------------------------

class PerProviderRateLimiter:
    """Independent per-provider rate limiting.

    Each provider has its own RPM budget, enforced via simple time-based
    token bucket (one token per interval).
    """

    def __init__(self, providers: List[OpenBBProviderConfig]) -> None:
        self._intervals: Dict[str, float] = {}
        self._last_call: Dict[str, float] = {}
        for p in providers:
            rpm = max(p.rate_limit_rpm, 1)
            self._intervals[p.name] = 60.0 / rpm
            self._last_call[p.name] = 0.0

    def wait_if_needed(self, provider_name: str) -> float:
        """Block until the rate limit window for *provider_name* has elapsed.

        Returns:
            Seconds actually waited (0.0 if no wait was needed).
        """
        interval = self._intervals.get(provider_name, 0.0)
        if interval <= 0:
            return 0.0

        elapsed = time.monotonic() - self._last_call.get(provider_name, 0.0)
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
            self._last_call[provider_name] = time.monotonic()
            return remaining

        self._last_call[provider_name] = time.monotonic()
        return 0.0

    def record_call(self, provider_name: str) -> None:
        """Record that a call was just made (updates timestamp)."""
        self._last_call[provider_name] = time.monotonic()


# ---------------------------------------------------------------------------
# OpenBB Extractor
# ---------------------------------------------------------------------------

# OpenBB provider -> credential env-var name mapping
_PROVIDER_CREDENTIAL_MAP: Dict[str, str] = {
    "polygon": "MASSIVE_API_KEY",
    "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
    "finnhub": "FINNHUB_API_KEY",
}

# OpenBB credential setting names (obb.user.credentials.<name>)
_OBB_CREDENTIAL_NAMES: Dict[str, str] = {
    "polygon": "polygon_api_key",
    "alpha_vantage": "alpha_vantage_api_key",
    "finnhub": "finnhub_api_key",
}


class OpenBBExtractor(BaseExtractor):
    """Multi-provider OHLCV extractor using the OpenBB SDK.

    Inherits from BaseExtractor and implements the three required methods:
    - extract_ohlcv()
    - validate_data()
    - get_metadata()

    Usage:
        extractor = OpenBBExtractor()
        df = extractor.extract_ohlcv(["AAPL", "MSFT"], "2024-01-01", "2024-06-01")
        report = extractor.validate_data(df)
        meta = extractor.get_metadata("AAPL", df)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        storage=None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenBB extractor.

        Args:
            config_path: Path to openbb_config.yml. Defaults to config/openbb_config.yml.
            storage: Optional DataStorage instance for cache operations.
            **kwargs: Passed through to BaseExtractor.
        """
        # Bootstrap .env so credentials are available
        bootstrap_dotenv()

        # Resolve config path
        if config_path is None:
            project_root = Path(__file__).resolve().parents[1]
            config_path = project_root / "config" / "openbb_config.yml"
        self.config_path = Path(config_path)

        # Load configuration
        self.config = OpenBBConfig.from_yaml(self.config_path)

        # Initialize BaseExtractor
        super().__init__(
            name="openbb",
            timeout=kwargs.pop("timeout", 30),
            cache_hours=self.config.cache_hours,
            storage=storage,
            **kwargs,
        )

        # Load credentials per provider
        self._credentials: Dict[str, Optional[str]] = {}
        for provider in self.config.providers:
            if provider.credentials_env:
                self._credentials[provider.name] = load_secret(
                    provider.credentials_env
                )
            else:
                self._credentials[provider.name] = None  # e.g. yfinance (free)

        # Rate limiter
        self._rate_limiter = PerProviderRateLimiter(self.config.providers)

        # Lazy SDK handle
        self._obb = None
        self._obb_initialized = False

        # Cache directory
        project_root = Path(__file__).resolve().parents[1]
        self._cache_dir = project_root / self.config.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Track provider usage for metadata
        self._last_provider_used: Dict[str, str] = {}

        logger.info(
            "OpenBBExtractor initialized with %d providers (cache: %dh)",
            len(self.config.providers),
            self.config.cache_hours,
        )

    # ------------------------------------------------------------------
    # Lazy OpenBB SDK initialization
    # ------------------------------------------------------------------

    def _init_obb(self) -> Any:
        """Lazily import and configure the OpenBB SDK.

        Returns:
            The obb module handle, or None if import fails.
        """
        if self._obb_initialized:
            return self._obb

        try:
            from openbb import obb  # type: ignore

            # Configure credentials for each provider
            for provider_name, cred_value in self._credentials.items():
                if cred_value and provider_name in _OBB_CREDENTIAL_NAMES:
                    obb_attr = _OBB_CREDENTIAL_NAMES[provider_name]
                    try:
                        setattr(obb.user.credentials, obb_attr, cred_value)
                        logger.debug(
                            "Set OpenBB credential for %s", provider_name
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to set credential for %s: %s",
                            provider_name,
                            exc,
                        )

            self._obb = obb
            self._obb_initialized = True
            logger.info("OpenBB SDK initialized successfully")
            return self._obb

        except ImportError:
            logger.error(
                "OpenBB SDK not installed. Run: pip install 'openbb>=4.0.0'"
            )
            self._obb_initialized = True  # Don't retry
            return None
        except Exception as exc:
            logger.error("Failed to initialize OpenBB SDK: %s", exc)
            self._obb_initialized = True
            return None

    # ------------------------------------------------------------------
    # Nigeria market helpers
    # ------------------------------------------------------------------

    def _is_nigeria_ticker(self, ticker: str) -> bool:
        """Check if a ticker is a Nigerian market instrument."""
        if not self.config.nigeria_enabled:
            return False
        upper = ticker.upper()
        if upper.endswith(self.config.nigeria_suffix):
            return True
        if upper in [p.upper() for p in self.config.luno_pairs]:
            return True
        return False

    def _resolve_nigeria_ticker(self, ticker: str) -> Tuple[str, str]:
        """Resolve a Nigeria ticker to its canonical form and source.

        Args:
            ticker: Raw ticker string.

        Returns:
            Tuple of (resolved_ticker, source_hint) where source_hint is
            'ngx' for stock exchange or 'luno' for crypto pairs.
        """
        upper = ticker.upper()

        # LUNO crypto pair
        if upper in [p.upper() for p in self.config.luno_pairs]:
            return upper, "luno"

        # NGX stock (strip suffix for OpenBB query)
        if upper.endswith(self.config.nigeria_suffix):
            base = upper[: -len(self.config.nigeria_suffix)]
            return base, "ngx"

        return ticker, "unknown"

    # ------------------------------------------------------------------
    # Cache operations
    # ------------------------------------------------------------------

    def _cache_path_for(self, ticker: str) -> Path:
        """Return the expected parquet cache file path for a ticker."""
        safe_ticker = ticker.replace("/", "_").replace(".", "_")
        return self._cache_dir / f"openbb_{safe_ticker}.parquet"

    def _check_cache(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Check local parquet cache for recent data.

        Args:
            ticker: Stock ticker symbol.
            start_date: Requested start date.
            end_date: Requested end date.

        Returns:
            Cached DataFrame if valid and fresh, None otherwise.
        """
        if self.config.cache_hours <= 0:
            self._increment_cache_miss()
            return None

        cache_file = self._cache_path_for(ticker)
        if not cache_file.exists():
            self._increment_cache_miss()
            return None

        # Freshness check
        file_age_seconds = max(0.0, time.time() - cache_file.stat().st_mtime)
        if file_age_seconds >= self.config.cache_hours * 3600:
            logger.info(
                "Cache MISS for %s: expired (age: %.1fh >= %dh TTL)",
                ticker,
                file_age_seconds / 3600,
                self.config.cache_hours,
            )
            self._increment_cache_miss()
            return None

        try:
            cached = pd.read_parquet(cache_file)
        except Exception as exc:
            logger.warning("Failed to read cache for %s: %s", ticker, exc)
            self._increment_cache_miss()
            return None

        if cached.empty:
            self._increment_cache_miss()
            return None

        # Ensure DatetimeIndex
        if not isinstance(cached.index, pd.DatetimeIndex):
            try:
                cached.index = pd.to_datetime(cached.index)
            except Exception:
                self._increment_cache_miss()
                return None

        # Coverage check with 3-day tolerance for non-trading days
        tolerance = timedelta(days=3)
        cache_start = cached.index.min()
        cache_end = cached.index.max()

        start_ok = cache_start <= pd.Timestamp(start_date) + tolerance
        end_ok = cache_end >= pd.Timestamp(end_date) - tolerance

        if start_ok and end_ok:
            filtered = cached[
                (cached.index >= str(start_date))
                & (cached.index <= str(end_date))
            ]
            logger.info(
                "Cache HIT for %s: %d rows (age: %.1fh)",
                ticker,
                len(filtered),
                file_age_seconds / 3600,
            )
            self._increment_cache_hit()
            return filtered

        logger.info(
            "Cache MISS for %s: coverage gap (have %s-%s, need %s-%s)",
            ticker,
            cache_start.date(),
            cache_end.date(),
            start_date,
            end_date,
        )
        self._increment_cache_miss()
        return None

    def _save_cache(self, ticker: str, data: pd.DataFrame) -> None:
        """Save data to parquet cache."""
        if data.empty:
            return
        try:
            cache_file = self._cache_path_for(ticker)
            data.to_parquet(cache_file, engine="pyarrow")
            logger.debug("Cached %d rows for %s -> %s", len(data), ticker, cache_file)
        except Exception as exc:
            logger.warning("Failed to save cache for %s: %s", ticker, exc)

    # ------------------------------------------------------------------
    # Provider fetch logic
    # ------------------------------------------------------------------

    def _fetch_single_ticker(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single ticker, trying providers in priority order.

        Args:
            ticker: Ticker symbol.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).

        Returns:
            DataFrame with OHLCV data or empty DataFrame if all providers fail.
        """
        obb = self._init_obb()
        if obb is None:
            logger.error("OpenBB SDK unavailable; cannot fetch %s", ticker)
            return pd.DataFrame()

        enabled_providers = [p for p in self.config.providers if p.enabled]

        # Skip providers without credentials (except free ones like yfinance)
        available_providers = []
        for p in enabled_providers:
            if p.credentials_env is None:
                available_providers.append(p)
            elif self._credentials.get(p.name):
                available_providers.append(p)
            else:
                if self.config.log_provider_attempts:
                    logger.debug(
                        "Skipping %s: no credentials for %s",
                        p.name,
                        p.credentials_env,
                    )

        if not available_providers:
            logger.error("No available providers for %s", ticker)
            return pd.DataFrame()

        last_error: Optional[Exception] = None

        for provider_cfg in available_providers:
            provider_name = provider_cfg.name

            if self.config.log_provider_attempts:
                logger.info(
                    "Trying provider %s (priority %d) for %s",
                    provider_name,
                    provider_cfg.priority,
                    ticker,
                )

            # Rate limit
            waited = self._rate_limiter.wait_if_needed(provider_name)
            if waited > 0:
                logger.debug(
                    "Rate-limited %s: waited %.2fs", provider_name, waited
                )

            try:
                result = obb.equity.price.historical(
                    symbol=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    provider=provider_name,
                )
                self._rate_limiter.record_call(provider_name)

                # Convert OBBject to DataFrame
                df = self._openbb_result_to_dataframe(result)

                if df.empty:
                    if self.config.log_fallback_events:
                        logger.warning(
                            "Provider %s returned empty data for %s",
                            provider_name,
                            ticker,
                        )
                    continue

                if len(df) < self.config.min_rows:
                    if self.config.log_fallback_events:
                        logger.warning(
                            "Provider %s returned only %d rows for %s (min: %d)",
                            provider_name,
                            len(df),
                            ticker,
                            self.config.min_rows,
                        )
                    continue

                # Standardize column names
                df = self._standardize_openbb_output(df)
                df["ticker"] = ticker

                self._last_provider_used[ticker] = provider_name
                logger.info(
                    "Provider %s returned %d rows for %s",
                    provider_name,
                    len(df),
                    ticker,
                )
                return df

            except Exception as exc:
                last_error = exc
                self._rate_limiter.record_call(provider_name)
                if self.config.log_fallback_events:
                    logger.warning(
                        "Provider %s failed for %s: %s", provider_name, ticker, exc
                    )
                continue

        # All providers failed
        logger.error(
            "All providers failed for %s. Last error: %s", ticker, last_error
        )
        return pd.DataFrame()

    def _openbb_result_to_dataframe(self, result: Any) -> pd.DataFrame:
        """Convert an OpenBB OBBject result to a pandas DataFrame.

        Args:
            result: OBBject returned by obb.equity.price.historical().

        Returns:
            DataFrame with DatetimeIndex.
        """
        try:
            # OpenBB v4+ returns OBBject with .to_df() method
            if hasattr(result, "to_df"):
                df = result.to_df()
            elif hasattr(result, "results") and result.results:
                df = pd.DataFrame([r.model_dump() for r in result.results])
            else:
                return pd.DataFrame()

            if df.empty:
                return df

            # Set date as index
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    pass

            df.sort_index(inplace=True)
            return df

        except Exception as exc:
            logger.warning("Failed to convert OBBject to DataFrame: %s", exc)
            return pd.DataFrame()

    def _standardize_openbb_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map OpenBB lowercase columns to the project's Title case standard.

        Handles provider-specific column variations (e.g., 'adj_close',
        'adjusted_close', 'vwap').

        Args:
            df: Raw DataFrame from OpenBB.

        Returns:
            DataFrame with standardized column names (Open, High, Low, Close, Volume).
        """
        # Use BaseExtractor's built-in standardizer first
        df = self._standardize_columns(df)

        # Additional OpenBB-specific mappings
        extra_mapping = {
            "vwap": "VWAP",
            "transactions": "Transactions",
            "adj_close": "Adj Close",
            "adjusted_close": "Adj Close",
            "split_ratio": "Split_Ratio",
            "dividend": "Dividend",
        }

        rename_dict = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in extra_mapping and col != extra_mapping[col_lower]:
                rename_dict[col] = extra_mapping[col_lower]

        if rename_dict:
            df = df.rename(columns=rename_dict)

        return df

    # ------------------------------------------------------------------
    # BaseExtractor required methods
    # ------------------------------------------------------------------

    def extract_ohlcv(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Extract OHLCV data for given tickers and date range.

        Uses cache-first strategy: checks parquet cache, falls back to
        multi-provider fetch via OpenBB SDK.

        Args:
            tickers: List of ticker symbols.
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.

        Returns:
            DataFrame with standardized columns (Open, High, Low, Close,
            Volume) and DatetimeIndex.
        """
        if not tickers:
            logger.warning("No tickers provided")
            return pd.DataFrame()

        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        all_frames: List[pd.DataFrame] = []

        for ticker in tickers:
            ticker = ticker.strip().upper()

            # Nigeria market handling
            if self._is_nigeria_ticker(ticker):
                resolved, source_hint = self._resolve_nigeria_ticker(ticker)
                logger.info(
                    "Nigeria ticker %s resolved to %s (source: %s)",
                    ticker,
                    resolved,
                    source_hint,
                )
                if source_hint == "luno":
                    logger.info(
                        "LUNO crypto pair %s - OpenBB crypto endpoint not yet integrated",
                        resolved,
                    )
                    continue
                # Use resolved ticker for NGX stocks
                fetch_ticker = resolved
            else:
                fetch_ticker = ticker

            # Cache check
            cached = self._check_cache(fetch_ticker, start_dt, end_dt)
            if cached is not None and not cached.empty:
                if "ticker" not in cached.columns:
                    cached["ticker"] = ticker
                all_frames.append(cached)
                continue

            # Fetch from providers
            df = self._fetch_single_ticker(fetch_ticker, start_date, end_date)
            if df.empty:
                logger.warning("No data obtained for %s", ticker)
                continue

            # Ensure ticker column uses original ticker name
            df["ticker"] = ticker

            # Save to cache
            self._save_cache(fetch_ticker, df)

            all_frames.append(df)

        if not all_frames:
            logger.warning("No data extracted for any ticker")
            return pd.DataFrame()

        combined = pd.concat(all_frames, axis=0)

        # Ensure DatetimeIndex
        if not isinstance(combined.index, pd.DatetimeIndex):
            try:
                combined.index = pd.to_datetime(combined.index)
            except Exception:
                pass

        combined.sort_index(inplace=True)

        # Verify required columns
        has_all, missing = self._check_required_columns(combined)
        if not has_all:
            logger.warning("Missing required columns: %s", missing)

        logger.info(
            "Extracted %d total rows for %d tickers",
            len(combined),
            len(tickers),
        )
        return combined

    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return validation report.

        Checks:
        - Empty data
        - Missing value rate
        - Price positivity (Open, High, Low, Close > 0)
        - Volume non-negativity
        - Trading day gaps exceeding threshold

        Args:
            data: DataFrame to validate.

        Returns:
            Validation report dictionary.
        """
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {
            "missing_rate": 0.0,
            "outlier_count": 0,
            "gap_count": 0,
        }

        if data.empty:
            return {
                "passed": False,
                "errors": ["Empty DataFrame"],
                "warnings": [],
                "quality_score": 0.0,
                "metrics": metrics,
            }

        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        present_cols = [c for c in ohlcv_cols if c in data.columns]

        # Missing rate
        if present_cols:
            total_cells = len(data) * len(present_cols)
            missing_cells = data[present_cols].isna().sum().sum()
            missing_rate = float(missing_cells / total_cells) if total_cells > 0 else 0.0
            metrics["missing_rate"] = missing_rate

            if missing_rate > self.config.max_missing_rate:
                errors.append(
                    f"Missing rate {missing_rate:.2%} exceeds threshold "
                    f"{self.config.max_missing_rate:.2%}"
                )

        # Price positivity
        price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in data.columns]
        for col in price_cols:
            neg_count = (data[col].dropna() <= 0).sum()
            if neg_count > 0:
                errors.append(f"{col} has {neg_count} non-positive values")

        # Volume non-negativity
        if "Volume" in data.columns:
            neg_vol = (data["Volume"].dropna() < 0).sum()
            if neg_vol > 0:
                warnings.append(f"Volume has {neg_vol} negative values")

        # Gap detection (trading day gaps)
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            diffs = data.index.to_series().diff().dropna()
            # Gaps > max_gap_days (excluding weekends)
            gap_threshold = timedelta(days=self.config.max_gap_days)
            gaps = diffs[diffs > gap_threshold]
            metrics["gap_count"] = len(gaps)
            if len(gaps) > 0:
                warnings.append(
                    f"{len(gaps)} gap(s) exceeding {self.config.max_gap_days} days"
                )

        # Quality score
        score = 1.0
        score -= min(metrics["missing_rate"] * 5, 0.5)  # up to 0.5 penalty
        score -= min(len(errors) * 0.15, 0.3)  # up to 0.3 penalty
        score -= min(metrics["gap_count"] * 0.05, 0.2)  # up to 0.2 penalty
        score = max(score, 0.0)

        passed = len(errors) == 0

        return {
            "passed": passed,
            "errors": errors,
            "warnings": warnings,
            "quality_score": round(score, 3),
            "metrics": metrics,
        }

    def get_metadata(self, ticker: str, data: pd.DataFrame) -> ExtractorMetadata:
        """Generate metadata for extracted data.

        Args:
            ticker: Stock ticker symbol.
            data: Extracted DataFrame.

        Returns:
            ExtractorMetadata object with extraction details.
        """
        now = datetime.utcnow()

        if data.empty:
            return ExtractorMetadata(
                ticker=ticker,
                source=f"openbb:{self._last_provider_used.get(ticker, 'none')}",
                extraction_timestamp=now,
                data_start_date=now,
                data_end_date=now,
                row_count=0,
                cache_hit=False,
            )

        # Determine if this was a cache hit
        cache_hit = self._cache_hits > 0 and ticker not in self._last_provider_used

        data_start = data.index.min()
        data_end = data.index.max()

        # Convert pandas Timestamp to datetime if needed
        if hasattr(data_start, "to_pydatetime"):
            data_start = data_start.to_pydatetime()
        if hasattr(data_end, "to_pydatetime"):
            data_end = data_end.to_pydatetime()

        return ExtractorMetadata(
            ticker=ticker,
            source=f"openbb:{self._last_provider_used.get(ticker, 'cache')}",
            extraction_timestamp=now,
            data_start_date=data_start,
            data_end_date=data_end,
            row_count=len(data),
            cache_hit=cache_hit,
        )

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    def get_available_providers(self) -> List[Dict[str, Any]]:
        """List configured providers and their availability status.

        Returns:
            List of dicts with provider info.
        """
        result = []
        for p in self.config.providers:
            has_creds = True
            if p.credentials_env:
                has_creds = bool(self._credentials.get(p.name))
            result.append(
                {
                    "name": p.name,
                    "enabled": p.enabled,
                    "priority": p.priority,
                    "has_credentials": has_creds,
                    "rate_limit_rpm": p.rate_limit_rpm,
                }
            )
        return result

    def __repr__(self) -> str:
        n_providers = len(
            [p for p in self.config.providers if p.enabled]
        )
        return (
            f"OpenBBExtractor(providers={n_providers}, "
            f"cache={self.config.cache_hours}h)"
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint for standalone OpenBB extraction."""
    parser = argparse.ArgumentParser(
        description="OpenBB multi-provider OHLCV extractor"
    )
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated ticker symbols (e.g., AAPL,MSFT)",
    )
    parser.add_argument(
        "--start", required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to openbb_config.yml (default: config/openbb_config.yml)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache lookup (always fetch fresh data)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    tickers = [t.strip() for t in args.tickers.split(",")]

    extractor = OpenBBExtractor(config_path=args.config)

    if args.no_cache:
        extractor.config.cache_hours = 0

    print(f"[INFO] Extracting {len(tickers)} ticker(s): {tickers}")
    print(f"[INFO] Date range: {args.start} to {args.end}")
    print(f"[INFO] Available providers:")
    for p in extractor.get_available_providers():
        status = "[OK]" if p["has_credentials"] or p["name"] == "yfinance" else "[NO KEY]"
        print(f"  {p['priority']}. {p['name']} {status} (RPM: {p['rate_limit_rpm']})")

    df = extractor.extract_ohlcv(tickers, args.start, args.end)

    if df.empty:
        print("[ERROR] No data extracted")
        return

    print(f"\n[SUCCESS] Extracted {len(df)} rows")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"[INFO] Date range: {df.index.min()} to {df.index.max()}")

    # Validation
    report = extractor.validate_data(df)
    print(f"\n[VALIDATION] Passed: {report['passed']}")
    print(f"[VALIDATION] Quality score: {report['quality_score']}")
    if report["errors"]:
        for e in report["errors"]:
            print(f"  [ERROR] {e}")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"  [WARN] {w}")

    # Cache stats
    stats = extractor.get_cache_statistics()
    print(f"\n[CACHE] Hits: {stats['cache_hits']}, Misses: {stats['cache_misses']}, "
          f"Hit rate: {stats['hit_rate']:.1%}")


if __name__ == "__main__":
    main()
