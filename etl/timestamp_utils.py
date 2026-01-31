"""
Timestamp normalization utilities for Portfolio Maximizer.

All internal timestamps in the execution and persistence layers must be
tz-aware UTC datetime objects.  This module provides the canonical helpers
used at system boundaries (data ingestion, DB reads, signal creation) to
enforce that invariant.

Self-correcting: tz-naive inputs are assumed UTC rather than rejected.
"""

from datetime import datetime, timezone
from typing import Optional, Union

import pandas as pd


def ensure_utc(ts: Union[datetime, pd.Timestamp, str, None]) -> Optional[datetime]:
    """Normalize any timestamp representation to a tz-aware UTC datetime.

    Rules
    -----
    - ``None`` / ``pd.NaT`` -> ``None``
    - tz-aware datetime/Timestamp -> convert to UTC, return as ``datetime``
    - tz-naive datetime/Timestamp -> assume UTC, attach ``tzinfo``
    - ISO string -> parse, then apply above rules
    - Unparseable -> ``None`` (caller decides fallback)
    """
    if ts is None:
        return None

    # pd.NaT is a singleton but isinstance(pd.NaT, pd.Timestamp) is True
    # in some pandas versions, so check explicitly.
    try:
        if pd.isna(ts):
            return None
    except (TypeError, ValueError):
        pass

    # --- str path ---
    if isinstance(ts, str):
        ts = ts.strip()
        if not ts:
            return None
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            try:
                parsed = pd.to_datetime(ts, errors="coerce")
                if parsed is pd.NaT:
                    return None
                ts = parsed.to_pydatetime()
            except Exception:
                return None

    # --- pd.Timestamp path ---
    if isinstance(ts, pd.Timestamp):
        if ts is pd.NaT:
            return None
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC").to_pydatetime()
        return ts.tz_localize("UTC").to_pydatetime()

    # --- datetime path ---
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            return ts.astimezone(timezone.utc)
        return ts.replace(tzinfo=timezone.utc)

    return None


def utc_now() -> datetime:
    """Return the current moment as a tz-aware UTC datetime."""
    return datetime.now(timezone.utc)


def ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize a pandas DatetimeIndex to UTC.

    - tz-aware -> ``tz_convert('UTC')``
    - tz-naive -> ``tz_localize('UTC')``
    """
    if idx.tzinfo is not None:
        return idx.tz_convert("UTC")
    return idx.tz_localize("UTC")
