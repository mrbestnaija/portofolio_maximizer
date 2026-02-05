import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from scripts import backfill_signal_validation as bsv


UTC = timezone.utc


def test_ensure_utc_datetime_normalizes_common_inputs() -> None:
    naive_dt = datetime(2024, 1, 2, 3, 4, 5)
    aware_dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    only_date = date(2024, 1, 2)

    from_naive = bsv._ensure_utc_datetime(naive_dt)
    from_aware = bsv._ensure_utc_datetime(aware_dt)
    from_date = bsv._ensure_utc_datetime(only_date)

    assert from_naive.tzinfo == UTC
    assert from_naive.replace(tzinfo=None) == naive_dt

    assert from_aware.tzinfo == UTC
    assert from_aware == aware_dt

    assert from_date.tzinfo == UTC
    assert from_date.date() == only_date


def test_ensure_utc_datetime_parses_iso_strings() -> None:
    as_date = bsv._ensure_utc_datetime("2024-01-02")
    as_dt = bsv._ensure_utc_datetime("2024-01-02T03:04:05")
    as_dt_z = bsv._ensure_utc_datetime("2024-01-02T03:04:05Z")

    assert as_date.tzinfo == UTC
    assert as_date.date() == date(2024, 1, 2)

    assert as_dt.tzinfo == UTC
    assert as_dt.year == 2024 and as_dt.month == 1 and as_dt.day == 2

    assert as_dt_z.tzinfo == UTC
    assert as_dt_z.hour == 3 and as_dt_z.minute == 4 and as_dt_z.second == 5


def test_ensure_utc_datetime_handles_space_and_broken_T() -> None:
    spaced = bsv._ensure_utc_datetime("2025-12-04 17:42:07")
    assert spaced.tzinfo == UTC
    assert spaced.year == 2025 and spaced.month == 12 and spaced.day == 4
    assert spaced.hour == 17 and spaced.minute == 42 and spaced.second == 7

    broken = bsv._ensure_utc_datetime("2025-12-04 17:42:07T00:00:00")
    assert broken.tzinfo == UTC
    assert broken.year == 2025 and broken.month == 12 and broken.day == 4
    # The original time component before the extra "T00:00:00" should be preserved.
    assert broken.hour == 17 and broken.minute == 42 and broken.second == 7


def test_load_market_data_returns_indexed_frame(tmp_path: Path) -> None:
    db_path = tmp_path / "ohlcv_test.db"
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE ohlcv_data (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO ohlcv_data (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("AAPL", "2024-01-01", 1.0, 2.0, 0.5, 1.5, 100.0),
            )

        df = bsv.load_market_data(
            conn=conn,
            ticker="AAPL",
            signal_date=datetime(2024, 1, 2, tzinfo=UTC),
            lookback_days=5,
        )
    finally:
        conn.close()

    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_build_signal_dict_uses_utc_timestamp() -> None:
    class FakeRow(dict):
        def keys(self):
            return super().keys()

    row = FakeRow(
        {
            "ticker": "AAPL",
            "action": "BUY",
            "confidence": 0.9,
            "reasoning": "Test",
            "risk_level": None,
            "signal_date": "2024-01-02",
        }
    )

    payload = bsv.build_signal_dict(row)

    assert payload["ticker"] == "AAPL"
    assert payload["action"] == "BUY"
    assert payload["risk_level"] == "medium"

    ts = datetime.fromisoformat(payload["signal_timestamp"])
    assert ts.tzinfo == UTC
