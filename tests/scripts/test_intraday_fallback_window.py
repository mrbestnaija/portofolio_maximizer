from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from scripts import run_auto_trader


def test_prepare_market_window_uses_intraday_min_lookback(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_now = dt.datetime(2024, 1, 10, 12, 0, 0, tzinfo=run_auto_trader.UTC)

    class FakeDateTime(dt.datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fixed_now if tz is None else fixed_now.astimezone(tz)

    monkeypatch.setattr(run_auto_trader, "datetime", FakeDateTime)

    calls: dict[str, str] = {}

    class DummyManager:
        def __init__(self, interval: str) -> None:
            self.active_extractor = type("Extractor", (), {"interval": interval, "name": "test"})()

        def extract_ohlcv(self, tickers, start_date: str, end_date: str):
            calls["start_date"] = start_date
            calls["end_date"] = end_date
            return pd.DataFrame()

    # Intraday interval should use intraday min lookback days instead of 365.
    run_auto_trader._prepare_market_window(DummyManager("1h"), ["AAPL"], lookback_days=7)
    assert calls["end_date"] == "2024-01-10"
    assert calls["start_date"] == "2023-12-11"  # max(7, 30) days lookback

    # Daily interval retains the original 365-day minimum.
    run_auto_trader._prepare_market_window(DummyManager("1d"), ["AAPL"], lookback_days=7)
    assert calls["start_date"] == "2023-01-10"

