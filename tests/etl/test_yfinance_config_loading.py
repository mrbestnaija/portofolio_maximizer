"""Unit tests for YFinanceExtractor provider-config integration."""

import pandas as pd

from etl.yfinance_extractor import YFinanceExtractor


def test_yfinance_extractor_applies_config_and_passes_interval(tmp_path, monkeypatch):
    config_path = tmp_path / "yfinance_config.yml"
    config_path.write_text(
        """
extraction:
  cache:
    cache_hours: 7
    tolerance_days: 1
    auto_cleanup: false
    retention_days: 30
  network:
    timeout_seconds: 99
  rate_limiting:
    enabled: true
    delay_between_tickers: 0.0
    requests_per_minute: 600
  data:
    interval: "1wk"
  quality_checks:
    min_rows_required: 10
""".lstrip(),
        encoding="utf-8",
    )

    calls = []

    def mock_fetch_ticker_data(ticker, start_date, end_date, timeout=30, **kwargs):
        calls.append(
            {
                "ticker": ticker,
                "timeout": timeout,
                **kwargs,
            }
        )
        index = pd.date_range(start_date, end_date, freq="D", name="Date")
        return pd.DataFrame(
            {
                "Open": [1.0] * len(index),
                "High": [1.0] * len(index),
                "Low": [1.0] * len(index),
                "Close": [1.0] * len(index),
                "Volume": [1] * len(index),
            },
            index=index,
        )

    import etl.yfinance_extractor as yfinance_extractor

    monkeypatch.setattr(yfinance_extractor, "fetch_ticker_data", mock_fetch_ticker_data)

    extractor = YFinanceExtractor(storage=None, config_path=str(config_path))
    assert extractor.cache_hours == 7
    assert extractor.timeout == 99
    assert extractor.interval == "1wk"
    assert extractor.cache_tolerance_days == 1
    assert extractor.rate_limit_delay == 0.1  # rpm-derived floor: 60/600

    data = extractor.extract_ohlcv(["AAPL"], "2020-01-01", "2020-01-03")
    assert not data.empty
    assert calls and calls[0]["ticker"] == "AAPL"
    assert calls[0]["timeout"] == 99
    assert calls[0]["interval"] == "1wk"


def test_yfinance_extractor_respects_env_interval_override(tmp_path, monkeypatch):
    config_path = tmp_path / "yfinance_config.yml"
    config_path.write_text(
        """
extraction:
  data:
    interval: "1d"
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("YFINANCE_INTERVAL", "1h")
    extractor = YFinanceExtractor(storage=None, config_path=str(config_path))
    assert extractor.interval == "1h"
