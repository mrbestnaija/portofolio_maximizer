"""Tests for etl.timestamp_utils -- UTC normalization helpers."""

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from etl.timestamp_utils import ensure_utc, utc_now, ensure_utc_index


# ---------------------------------------------------------------------------
# ensure_utc
# ---------------------------------------------------------------------------


class TestEnsureUtc:
    def test_none_returns_none(self):
        assert ensure_utc(None) is None

    def test_empty_string_returns_none(self):
        assert ensure_utc("") is None
        assert ensure_utc("   ") is None

    def test_naive_datetime_assumes_utc(self):
        naive = datetime(2026, 1, 15, 12, 0, 0)
        result = ensure_utc(naive)
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.hour == 12

    def test_utc_datetime_passthrough(self):
        utc_dt = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_utc(utc_dt)
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result == utc_dt

    def test_aware_datetime_converts_to_utc(self):
        est = timezone(timedelta(hours=-5))
        aware = datetime(2026, 1, 15, 12, 0, 0, tzinfo=est)
        result = ensure_utc(aware)
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.hour == 17  # 12 EST = 17 UTC

    def test_iso_string_with_offset(self):
        result = ensure_utc("2026-01-15T12:00:00+00:00")
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.hour == 12

    def test_iso_string_with_z_suffix(self):
        result = ensure_utc("2026-01-15T12:00:00Z")
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.hour == 12

    def test_naive_iso_string_assumes_utc(self):
        result = ensure_utc("2026-01-15T12:00:00")
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.hour == 12

    def test_iso_string_with_nonzero_offset(self):
        result = ensure_utc("2026-01-15T12:00:00+05:30")
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.hour == 6  # 12:00 IST = 06:30 UTC
        assert result.minute == 30

    def test_pd_timestamp_naive(self):
        ts = pd.Timestamp("2026-01-15 12:00:00")
        result = ensure_utc(ts)
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.hour == 12

    def test_pd_timestamp_aware(self):
        ts = pd.Timestamp("2026-01-15 12:00:00", tz="US/Eastern")
        result = ensure_utc(ts)
        assert result is not None
        assert result.tzinfo == timezone.utc
        assert result.hour == 17  # 12 EST = 17 UTC

    def test_pd_nat_returns_none(self):
        assert ensure_utc(pd.NaT) is None

    def test_unparseable_string_returns_none(self):
        assert ensure_utc("not-a-date") is None

    def test_integer_returns_none(self):
        assert ensure_utc(12345) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# utc_now
# ---------------------------------------------------------------------------


class TestUtcNow:
    def test_is_aware(self):
        now = utc_now()
        assert now.tzinfo == timezone.utc

    def test_is_recent(self):
        now = utc_now()
        delta = abs((datetime.now(timezone.utc) - now).total_seconds())
        assert delta < 2.0


# ---------------------------------------------------------------------------
# ensure_utc_index
# ---------------------------------------------------------------------------


class TestEnsureUtcIndex:
    def test_naive_index_becomes_utc(self):
        idx = pd.DatetimeIndex(["2026-01-15", "2026-01-16"])
        result = ensure_utc_index(idx)
        assert result.tzinfo is not None
        assert str(result.tzinfo) == "UTC"

    def test_aware_index_converts_to_utc(self):
        idx = pd.DatetimeIndex(["2026-01-15", "2026-01-16"]).tz_localize("US/Eastern")
        result = ensure_utc_index(idx)
        assert str(result.tzinfo) == "UTC"

    def test_utc_index_passthrough(self):
        idx = pd.DatetimeIndex(["2026-01-15", "2026-01-16"]).tz_localize("UTC")
        result = ensure_utc_index(idx)
        assert str(result.tzinfo) == "UTC"
        assert len(result) == 2

    def test_preserves_length(self):
        idx = pd.DatetimeIndex(["2026-01-15", "2026-01-16", "2026-01-17"])
        result = ensure_utc_index(idx)
        assert len(result) == 3
