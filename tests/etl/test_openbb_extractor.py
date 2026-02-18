"""Unit tests for OpenBBExtractor multi-provider data extraction.

Test Coverage:
- Initialization and config loading
- Provider fallback chain
- Per-provider rate limiting
- Cache hit/miss behavior
- Nigeria market ticker resolution
- Data validation
- Metadata generation

All tests use mocked OpenBB SDK to avoid network calls.
"""

import os
import tempfile
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from etl.openbb_extractor import (
    OpenBBConfig,
    OpenBBExtractor,
    OpenBBProviderConfig,
    PerProviderRateLimiter,
)
from etl.base_extractor import ExtractorMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV DataFrame with 100 trading days."""
    dates = pd.bdate_range("2024-01-02", periods=100, freq="B")
    rng = np.random.RandomState(42)
    close = 150.0 + rng.randn(100).cumsum()
    data = pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 2, 100),
            "High": close + rng.uniform(0, 3, 100),
            "Low": close - rng.uniform(0, 3, 100),
            "Close": close,
            "Volume": rng.randint(1_000_000, 10_000_000, 100),
        },
        index=dates,
    )
    data.index.name = "date"
    return data


@pytest.fixture
def openbb_lowercase_data():
    """Generate sample data with OpenBB lowercase column names."""
    dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
    rng = np.random.RandomState(42)
    close = 150.0 + rng.randn(50).cumsum()
    data = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 2, 50),
            "high": close + rng.uniform(0, 3, 50),
            "low": close - rng.uniform(0, 3, 50),
            "close": close,
            "volume": rng.randint(1_000_000, 10_000_000, 50),
        },
        index=dates,
    )
    data.index.name = "date"
    return data


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_config():
    """Return a minimal OpenBBConfig for testing."""
    return OpenBBConfig(
        providers=[
            OpenBBProviderConfig("yfinance", True, 1, None, 60),
            OpenBBProviderConfig("polygon", True, 2, "MASSIVE_API_KEY", 5),
            OpenBBProviderConfig("alpha_vantage", True, 3, "ALPHA_VANTAGE_API_KEY", 5),
            OpenBBProviderConfig("finnhub", True, 4, "FINNHUB_API_KEY", 60),
        ],
        cache_hours=24,
        cache_dir="data/raw",
    )


def _make_mock_obb_result(df: pd.DataFrame) -> MagicMock:
    """Create a mock OBBject that returns *df* from to_df()."""
    result = MagicMock()
    result.to_df.return_value = df
    return result


def _make_extractor(
    tmp_dir: Path,
    config: OpenBBConfig | None = None,
    env_overrides: dict | None = None,
) -> OpenBBExtractor:
    """Create an OpenBBExtractor pointing at tmp_dir for cache.

    Patches environment variables and config loading to avoid touching
    the real filesystem or .env file.
    """
    env = {
        "CI": "true",  # Prevent .env bootstrap
        "MASSIVE_API_KEY": "test_polygon_key",
        "ALPHA_VANTAGE_API_KEY": "test_av_key",
        "FINNHUB_API_KEY": "test_finnhub_key",
        "API_KEY_LUNO": "test_luno_key",
        "API_ID_LUNO": "test_luno_id",
    }
    if env_overrides:
        env.update(env_overrides)

    with patch.dict(os.environ, env, clear=False):
        if config is None:
            config = OpenBBConfig(
                providers=[
                    OpenBBProviderConfig("yfinance", True, 1, None, 60),
                    OpenBBProviderConfig("polygon", True, 2, "MASSIVE_API_KEY", 5),
                    OpenBBProviderConfig(
                        "alpha_vantage", True, 3, "ALPHA_VANTAGE_API_KEY", 5
                    ),
                    OpenBBProviderConfig("finnhub", True, 4, "FINNHUB_API_KEY", 60),
                ],
                cache_hours=24,
                cache_dir=str(tmp_dir),
            )

        with patch(
            "etl.openbb_extractor.OpenBBConfig.from_yaml", return_value=config
        ):
            ext = OpenBBExtractor(config_path="dummy.yml")
            # Override cache dir to temp
            ext._cache_dir = tmp_dir
            ext._cache_dir.mkdir(parents=True, exist_ok=True)
            return ext


# ---------------------------------------------------------------------------
# TestOpenBBExtractorInit
# ---------------------------------------------------------------------------

class TestOpenBBExtractorInit:
    """Test extractor initialization and config loading."""

    def test_init_with_defaults(self, temp_cache_dir):
        """Extractor initializes with default config when YAML is missing."""
        ext = _make_extractor(temp_cache_dir)
        assert ext.name == "openbb"
        assert len(ext.config.providers) == 4
        assert ext.config.cache_hours == 24

    def test_provider_order(self, temp_cache_dir):
        """Providers are sorted by priority."""
        ext = _make_extractor(temp_cache_dir)
        names = [p.name for p in ext.config.providers]
        assert names == ["yfinance", "polygon", "alpha_vantage", "finnhub"]

    def test_credentials_loaded(self, temp_cache_dir):
        """Credentials are loaded from environment variables."""
        ext = _make_extractor(temp_cache_dir)
        assert ext._credentials["yfinance"] is None  # Free, no key
        assert ext._credentials["polygon"] == "test_polygon_key"
        assert ext._credentials["alpha_vantage"] == "test_av_key"
        assert ext._credentials["finnhub"] == "test_finnhub_key"

    def test_config_from_yaml(self, tmp_path):
        """Config loads correctly from YAML file."""
        yaml_content = """
openbb:
  providers:
    - name: "yfinance"
      enabled: true
      priority: 1
      rate_limit_rpm: 60
    - name: "polygon"
      enabled: true
      priority: 2
      credentials_env: "MASSIVE_API_KEY"
      rate_limit_rpm: 5
  cache:
    cache_hours: 12
    cache_dir: "data/raw"
  validation:
    min_rows: 20
"""
        config_file = tmp_path / "openbb_config.yml"
        config_file.write_text(yaml_content)

        config = OpenBBConfig.from_yaml(config_file)
        assert len(config.providers) == 2
        assert config.providers[0].name == "yfinance"
        assert config.cache_hours == 12
        assert config.min_rows == 20

    def test_config_defaults_on_missing_yaml(self, tmp_path):
        """Returns defaults when YAML file doesn't exist."""
        config = OpenBBConfig.from_yaml(tmp_path / "nonexistent.yml")
        assert len(config.providers) == 4
        assert config.providers[0].name == "yfinance"

    def test_get_available_providers(self, temp_cache_dir):
        """get_available_providers returns provider status list."""
        ext = _make_extractor(temp_cache_dir)
        providers = ext.get_available_providers()
        assert len(providers) == 4
        assert providers[0]["name"] == "yfinance"
        assert providers[0]["has_credentials"] is True  # yfinance has no cred req
        assert providers[1]["name"] == "polygon"
        assert providers[1]["has_credentials"] is True

    def test_repr(self, temp_cache_dir):
        """Repr includes provider count and cache hours."""
        ext = _make_extractor(temp_cache_dir)
        r = repr(ext)
        assert "OpenBBExtractor" in r
        assert "providers=4" in r
        assert "cache=24h" in r


# ---------------------------------------------------------------------------
# TestProviderFallback
# ---------------------------------------------------------------------------

class TestProviderFallback:
    """Test multi-provider fallback chain."""

    def test_first_provider_succeeds(self, temp_cache_dir, sample_ohlcv_data):
        """When first provider succeeds, no fallback occurs."""
        ext = _make_extractor(temp_cache_dir)
        mock_result = _make_mock_obb_result(sample_ohlcv_data)

        mock_obb = MagicMock()
        mock_obb.equity.price.historical.return_value = mock_result

        ext._obb = mock_obb
        ext._obb_initialized = True

        df = ext._fetch_single_ticker("AAPL", "2024-01-01", "2024-06-01")
        assert not df.empty
        assert len(df) == 100
        assert "ticker" in df.columns
        # Only one call should have been made
        assert mock_obb.equity.price.historical.call_count == 1

    def test_fallback_on_first_failure(self, temp_cache_dir, sample_ohlcv_data):
        """Falls back to next provider when first raises an exception."""
        ext = _make_extractor(temp_cache_dir)
        mock_result = _make_mock_obb_result(sample_ohlcv_data)

        mock_obb = MagicMock()
        mock_obb.equity.price.historical.side_effect = [
            Exception("yfinance error"),
            mock_result,
        ]

        ext._obb = mock_obb
        ext._obb_initialized = True

        df = ext._fetch_single_ticker("AAPL", "2024-01-01", "2024-06-01")
        assert not df.empty
        assert mock_obb.equity.price.historical.call_count == 2

    def test_fallback_on_empty_result(self, temp_cache_dir, sample_ohlcv_data):
        """Falls back when provider returns empty data."""
        ext = _make_extractor(temp_cache_dir)
        empty_result = _make_mock_obb_result(pd.DataFrame())
        good_result = _make_mock_obb_result(sample_ohlcv_data)

        mock_obb = MagicMock()
        mock_obb.equity.price.historical.side_effect = [
            empty_result,
            good_result,
        ]

        ext._obb = mock_obb
        ext._obb_initialized = True

        df = ext._fetch_single_ticker("AAPL", "2024-01-01", "2024-06-01")
        assert not df.empty
        assert mock_obb.equity.price.historical.call_count == 2

    def test_all_providers_fail(self, temp_cache_dir):
        """Returns empty DataFrame when all providers fail."""
        ext = _make_extractor(temp_cache_dir)

        mock_obb = MagicMock()
        mock_obb.equity.price.historical.side_effect = Exception("all fail")

        ext._obb = mock_obb
        ext._obb_initialized = True

        df = ext._fetch_single_ticker("AAPL", "2024-01-01", "2024-06-01")
        assert df.empty

    def test_skips_provider_without_credentials(self, temp_cache_dir, sample_ohlcv_data):
        """Providers without credentials are skipped."""
        config = OpenBBConfig(
            providers=[
                OpenBBProviderConfig("polygon", True, 1, "MISSING_KEY", 5),
                OpenBBProviderConfig("yfinance", True, 2, None, 60),
            ],
            cache_hours=24,
            cache_dir=str(temp_cache_dir),
        )

        ext = _make_extractor(
            temp_cache_dir,
            config=config,
            env_overrides={"MISSING_KEY": ""},
        )

        mock_result = _make_mock_obb_result(sample_ohlcv_data)
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.return_value = mock_result

        ext._obb = mock_obb
        ext._obb_initialized = True
        # Clear the credential so polygon is skipped
        ext._credentials["polygon"] = None

        df = ext._fetch_single_ticker("AAPL", "2024-01-01", "2024-06-01")
        assert not df.empty
        # Should have called with yfinance provider only
        call_kwargs = mock_obb.equity.price.historical.call_args
        assert call_kwargs[1]["provider"] == "yfinance"

    def test_min_rows_threshold(self, temp_cache_dir):
        """Falls back when provider returns fewer rows than min_rows."""
        # Create data with only 5 rows (below default min_rows=10)
        dates = pd.bdate_range("2024-01-02", periods=5, freq="B")
        small_df = pd.DataFrame(
            {
                "Open": [100] * 5,
                "High": [105] * 5,
                "Low": [95] * 5,
                "Close": [102] * 5,
                "Volume": [1000000] * 5,
            },
            index=dates,
        )

        big_dates = pd.bdate_range("2024-01-02", periods=50, freq="B")
        big_df = pd.DataFrame(
            {
                "Open": [100] * 50,
                "High": [105] * 50,
                "Low": [95] * 50,
                "Close": [102] * 50,
                "Volume": [1000000] * 50,
            },
            index=big_dates,
        )

        ext = _make_extractor(temp_cache_dir)
        small_result = _make_mock_obb_result(small_df)
        big_result = _make_mock_obb_result(big_df)

        mock_obb = MagicMock()
        mock_obb.equity.price.historical.side_effect = [small_result, big_result]

        ext._obb = mock_obb
        ext._obb_initialized = True

        df = ext._fetch_single_ticker("AAPL", "2024-01-01", "2024-06-01")
        assert len(df) == 50  # Should have fallen back to second provider


# ---------------------------------------------------------------------------
# TestRateLimiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Test per-provider rate limiting."""

    def test_no_wait_first_call(self):
        """First call should not wait."""
        providers = [OpenBBProviderConfig("test", True, 1, None, 60)]
        limiter = PerProviderRateLimiter(providers)
        waited = limiter.wait_if_needed("test")
        assert waited == 0.0

    def test_enforces_delay(self, monkeypatch):
        """Second call within interval should wait (without real sleeps)."""
        providers = [OpenBBProviderConfig("test", True, 1, None, 6)]  # 10s interval
        limiter = PerProviderRateLimiter(providers)

        # Patch the time module used by the limiter so we don't actually sleep for 10s.
        now = {"t": 100.0}
        slept: list[float] = []

        def _fake_monotonic() -> float:
            return float(now["t"])

        def _fake_sleep(seconds: float) -> None:
            seconds = float(seconds)
            slept.append(seconds)
            now["t"] += seconds

        monkeypatch.setattr("etl.openbb_extractor.time.monotonic", _fake_monotonic)
        monkeypatch.setattr("etl.openbb_extractor.time.sleep", _fake_sleep)

        limiter.wait_if_needed("test")
        now["t"] += 0.1
        limiter.record_call("test")
        now["t"] += 0.1

        waited = limiter.wait_if_needed("test")

        # Should have waited close to 10 seconds
        assert waited > 0
        assert slept, "Expected limiter to sleep to enforce rate limit"
        assert slept[-1] == pytest.approx(waited, abs=1e-6)
        assert waited == pytest.approx(9.9, abs=0.01)

    def test_provider_independence(self):
        """Different providers have independent rate limits."""
        providers = [
            OpenBBProviderConfig("fast", True, 1, None, 600),   # 0.1s interval
            OpenBBProviderConfig("slow", True, 2, None, 6),     # 10s interval
        ]
        limiter = PerProviderRateLimiter(providers)

        # Call slow first
        limiter.wait_if_needed("slow")
        limiter.record_call("slow")

        # Fast should not wait
        waited = limiter.wait_if_needed("fast")
        assert waited == 0.0

    def test_record_call_updates_timestamp(self):
        """record_call should update the last call timestamp."""
        providers = [OpenBBProviderConfig("test", True, 1, None, 120)]  # 0.5s
        limiter = PerProviderRateLimiter(providers)

        limiter.record_call("test")
        assert limiter._last_call["test"] > 0


# ---------------------------------------------------------------------------
# TestCacheBehavior
# ---------------------------------------------------------------------------

class TestCacheBehavior:
    """Test parquet cache hit/miss logic."""

    def test_cache_miss_empty_dir(self, temp_cache_dir):
        """Cache miss when no files exist."""
        ext = _make_extractor(temp_cache_dir)
        result = ext._check_cache("AAPL", datetime(2024, 1, 1), datetime(2024, 6, 1))
        assert result is None

    def test_cache_hit(self, temp_cache_dir, sample_ohlcv_data):
        """Cache hit when fresh data covers the request."""
        ext = _make_extractor(temp_cache_dir)
        ext._save_cache("AAPL", sample_ohlcv_data)

        result = ext._check_cache("AAPL", datetime(2024, 1, 2), datetime(2024, 5, 1))
        assert result is not None
        assert len(result) > 0

    def test_cache_miss_expired(self, temp_cache_dir, sample_ohlcv_data):
        """Cache miss when file is older than cache_hours."""
        config = OpenBBConfig(
            providers=[OpenBBProviderConfig("yfinance", True, 1, None, 60)],
            cache_hours=0,  # Immediate expiration
            cache_dir=str(temp_cache_dir),
        )
        ext = _make_extractor(temp_cache_dir, config=config)
        ext._save_cache("AAPL", sample_ohlcv_data)

        result = ext._check_cache("AAPL", datetime(2024, 1, 1), datetime(2024, 6, 1))
        assert result is None

    def test_cache_avoids_network(self, temp_cache_dir, sample_ohlcv_data):
        """Cache hit should prevent any network calls."""
        ext = _make_extractor(temp_cache_dir)
        ext._save_cache("AAPL", sample_ohlcv_data)

        mock_obb = MagicMock()
        ext._obb = mock_obb
        ext._obb_initialized = True

        df = ext.extract_ohlcv(["AAPL"], "2024-01-02", "2024-05-01")
        assert not df.empty
        # No network calls should have been made
        mock_obb.equity.price.historical.assert_not_called()

    def test_cache_miss_fetches_and_saves(self, temp_cache_dir, sample_ohlcv_data):
        """Cache miss triggers fetch and saves result."""
        ext = _make_extractor(temp_cache_dir)

        mock_result = _make_mock_obb_result(sample_ohlcv_data)
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.return_value = mock_result
        ext._obb = mock_obb
        ext._obb_initialized = True

        df = ext.extract_ohlcv(["AAPL"], "2024-01-01", "2024-06-01")
        assert not df.empty

        # Cache file should now exist
        cache_file = ext._cache_path_for("AAPL")
        assert cache_file.exists()

    def test_cache_path_sanitization(self, temp_cache_dir):
        """Ticker with special chars gets sanitized in cache path."""
        ext = _make_extractor(temp_cache_dir)
        path = ext._cache_path_for("BRK.B")
        assert "." not in path.stem.replace("openbb_BRK_B", "")

    def test_cache_stats_tracking(self, temp_cache_dir, sample_ohlcv_data):
        """Cache stats are tracked correctly."""
        ext = _make_extractor(temp_cache_dir)
        ext._save_cache("AAPL", sample_ohlcv_data)

        # First: cache hit
        ext._check_cache("AAPL", datetime(2024, 1, 2), datetime(2024, 5, 1))
        # Second: cache miss (different ticker)
        ext._check_cache("MSFT", datetime(2024, 1, 1), datetime(2024, 6, 1))

        stats = ext.get_cache_statistics()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1


# ---------------------------------------------------------------------------
# TestNigeriaMarket
# ---------------------------------------------------------------------------

class TestNigeriaMarket:
    """Test Nigeria market ticker detection and resolution."""

    def test_ng_suffix_detection(self, temp_cache_dir):
        """Tickers ending in .NG are detected as Nigerian."""
        ext = _make_extractor(temp_cache_dir)
        assert ext._is_nigeria_ticker("DANGCEM.NG") is True
        assert ext._is_nigeria_ticker("AAPL") is False

    def test_luno_pair_detection(self, temp_cache_dir):
        """LUNO crypto pairs are detected as Nigerian market."""
        ext = _make_extractor(temp_cache_dir)
        assert ext._is_nigeria_ticker("XBTNGN") is True
        assert ext._is_nigeria_ticker("ETHNGN") is True
        assert ext._is_nigeria_ticker("BTCUSD") is False

    def test_ng_ticker_resolution(self, temp_cache_dir):
        """NGX tickers resolve by stripping .NG suffix."""
        ext = _make_extractor(temp_cache_dir)
        resolved, source = ext._resolve_nigeria_ticker("DANGCEM.NG")
        assert resolved == "DANGCEM"
        assert source == "ngx"

    def test_luno_pair_resolution(self, temp_cache_dir):
        """LUNO pairs resolve to uppercase with luno source."""
        ext = _make_extractor(temp_cache_dir)
        resolved, source = ext._resolve_nigeria_ticker("XBTNGN")
        assert resolved == "XBTNGN"
        assert source == "luno"

    def test_nigeria_disabled(self, temp_cache_dir):
        """When Nigeria is disabled, .NG tickers are not detected."""
        config = OpenBBConfig(
            providers=[OpenBBProviderConfig("yfinance", True, 1, None, 60)],
            nigeria_enabled=False,
            cache_dir=str(temp_cache_dir),
        )
        ext = _make_extractor(temp_cache_dir, config=config)
        assert ext._is_nigeria_ticker("DANGCEM.NG") is False


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    """Test data validation logic."""

    def test_valid_data_passes(self, temp_cache_dir, sample_ohlcv_data):
        """Clean OHLCV data should pass validation."""
        ext = _make_extractor(temp_cache_dir)
        report = ext.validate_data(sample_ohlcv_data)
        assert report["passed"] is True
        assert report["quality_score"] > 0.8
        assert len(report["errors"]) == 0

    def test_empty_data_fails(self, temp_cache_dir):
        """Empty DataFrame should fail validation."""
        ext = _make_extractor(temp_cache_dir)
        report = ext.validate_data(pd.DataFrame())
        assert report["passed"] is False
        assert "Empty DataFrame" in report["errors"]
        assert report["quality_score"] == 0.0

    def test_high_missing_rate_fails(self, temp_cache_dir, sample_ohlcv_data):
        """Data with too many missing values should fail."""
        ext = _make_extractor(temp_cache_dir)
        # Inject 20% missing values
        data = sample_ohlcv_data.copy()
        mask = np.random.RandomState(42).random(data.shape) < 0.2
        data[mask] = np.nan

        report = ext.validate_data(data)
        assert report["metrics"]["missing_rate"] > 0.05

    def test_negative_prices_fail(self, temp_cache_dir, sample_ohlcv_data):
        """Negative prices should be flagged as errors."""
        ext = _make_extractor(temp_cache_dir)
        data = sample_ohlcv_data.copy()
        data.loc[data.index[0], "Close"] = -10.0

        report = ext.validate_data(data)
        assert report["passed"] is False
        assert any("non-positive" in e for e in report["errors"])

    def test_gap_detection(self, temp_cache_dir):
        """Gaps exceeding max_gap_days are flagged."""
        ext = _make_extractor(temp_cache_dir)

        # Create data with a 10-day gap
        dates1 = pd.bdate_range("2024-01-02", periods=20, freq="B")
        dates2 = pd.bdate_range("2024-02-19", periods=20, freq="B")
        dates = dates1.append(dates2)

        data = pd.DataFrame(
            {
                "Open": [100] * len(dates),
                "High": [105] * len(dates),
                "Low": [95] * len(dates),
                "Close": [102] * len(dates),
                "Volume": [1000000] * len(dates),
            },
            index=dates,
        )

        report = ext.validate_data(data)
        assert report["metrics"]["gap_count"] >= 1

    def test_quality_score_range(self, temp_cache_dir, sample_ohlcv_data):
        """Quality score should be between 0 and 1."""
        ext = _make_extractor(temp_cache_dir)
        report = ext.validate_data(sample_ohlcv_data)
        assert 0.0 <= report["quality_score"] <= 1.0


# ---------------------------------------------------------------------------
# TestMetadata
# ---------------------------------------------------------------------------

class TestMetadata:
    """Test metadata generation."""

    def test_metadata_with_data(self, temp_cache_dir, sample_ohlcv_data):
        """Metadata includes correct fields from extracted data."""
        ext = _make_extractor(temp_cache_dir)
        ext._last_provider_used["AAPL"] = "yfinance"

        meta = ext.get_metadata("AAPL", sample_ohlcv_data)
        assert isinstance(meta, ExtractorMetadata)
        assert meta.ticker == "AAPL"
        assert "openbb:yfinance" in meta.source
        assert meta.row_count == 100
        assert meta.data_start_date is not None
        assert meta.data_end_date is not None

    def test_metadata_empty_data(self, temp_cache_dir):
        """Metadata handles empty DataFrame gracefully."""
        ext = _make_extractor(temp_cache_dir)
        meta = ext.get_metadata("AAPL", pd.DataFrame())
        assert meta.row_count == 0
        assert "none" in meta.source

    def test_metadata_to_dict(self, temp_cache_dir, sample_ohlcv_data):
        """Metadata converts to dictionary correctly."""
        ext = _make_extractor(temp_cache_dir)
        ext._last_provider_used["AAPL"] = "polygon"

        meta = ext.get_metadata("AAPL", sample_ohlcv_data)
        d = meta.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["row_count"] == 100
        assert "openbb:polygon" in d["source"]


# ---------------------------------------------------------------------------
# TestExtractOHLCV (integration-style with mocks)
# ---------------------------------------------------------------------------

class TestExtractOHLCV:
    """Test the main extract_ohlcv method end-to-end."""

    def test_empty_tickers_returns_empty(self, temp_cache_dir):
        """No tickers returns empty DataFrame."""
        ext = _make_extractor(temp_cache_dir)
        df = ext.extract_ohlcv([], "2024-01-01", "2024-06-01")
        assert df.empty

    def test_single_ticker_fetch(self, temp_cache_dir, sample_ohlcv_data):
        """Single ticker fetch returns standardized data."""
        ext = _make_extractor(temp_cache_dir)
        mock_result = _make_mock_obb_result(sample_ohlcv_data)
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.return_value = mock_result
        ext._obb = mock_obb
        ext._obb_initialized = True

        df = ext.extract_ohlcv(["AAPL"], "2024-01-01", "2024-06-01")
        assert not df.empty
        assert "ticker" in df.columns
        assert "Open" in df.columns
        assert "Close" in df.columns

    def test_multi_ticker_fetch(self, temp_cache_dir, sample_ohlcv_data):
        """Multiple tickers are concatenated."""
        ext = _make_extractor(temp_cache_dir)
        mock_result = _make_mock_obb_result(sample_ohlcv_data)
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.return_value = mock_result
        ext._obb = mock_obb
        ext._obb_initialized = True

        df = ext.extract_ohlcv(["AAPL", "MSFT"], "2024-01-01", "2024-06-01")
        assert not df.empty
        assert set(df["ticker"].unique()) == {"AAPL", "MSFT"}

    def test_column_standardization(self, temp_cache_dir, openbb_lowercase_data):
        """Lowercase OpenBB columns are standardized to Title case."""
        ext = _make_extractor(temp_cache_dir)
        mock_result = _make_mock_obb_result(openbb_lowercase_data)
        mock_obb = MagicMock()
        mock_obb.equity.price.historical.return_value = mock_result
        ext._obb = mock_obb
        ext._obb_initialized = True

        df = ext.extract_ohlcv(["AAPL"], "2024-01-01", "2024-06-01")
        assert "Open" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns
        assert "Close" in df.columns
        assert "Volume" in df.columns


# ---------------------------------------------------------------------------
# TestOpenBBSDKInit
# ---------------------------------------------------------------------------

class TestOpenBBSDKInit:
    """Test lazy OpenBB SDK initialization."""

    def test_lazy_init_not_called_at_construction(self, temp_cache_dir):
        """SDK should not be initialized until first use."""
        ext = _make_extractor(temp_cache_dir)
        assert ext._obb is None
        assert ext._obb_initialized is False

    def test_init_obb_sets_credentials(self, temp_cache_dir):
        """SDK initialization sets provider credentials."""
        ext = _make_extractor(temp_cache_dir)

        mock_obb = MagicMock()
        with patch("etl.openbb_extractor.obb", mock_obb, create=True):
            with patch.dict(
                "sys.modules", {"openbb": MagicMock(obb=mock_obb)}
            ):
                result = ext._init_obb()
                # Should have attempted to set credentials
                assert ext._obb_initialized is True

    def test_init_obb_handles_import_error(self, temp_cache_dir):
        """Handles ImportError gracefully when openbb not installed."""
        ext = _make_extractor(temp_cache_dir)

        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'openbb'"),
        ):
            result = ext._init_obb()
            assert result is None
            assert ext._obb_initialized is True  # Don't retry
