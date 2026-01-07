from __future__ import annotations

import pandas as pd
import pytest

from scripts import run_auto_trader


def _make_frame(last_close: float) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [last_close - 1] * 5,
            "High": [last_close + 1] * 5,
            "Low": [last_close - 2] * 5,
            "Close": [last_close - 0.4, last_close - 0.2, last_close - 0.1, last_close, last_close],
            "Volume": [100, 100, 100, 100, 100],
        },
        index=idx,
    )


def test_generate_forecasts_bulk_parallel_matches_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = {f"T{i:02d}": _make_frame(100.0 + i) for i in range(12)}

    def fake_forecast(frame: pd.DataFrame, horizon: int):
        last = float(frame["Close"].iloc[-1])
        return ({"horizon": horizon, "forecast": last + horizon}, last)

    monkeypatch.setattr(run_auto_trader, "_generate_time_series_forecast", fake_forecast)

    seq = run_auto_trader._generate_forecasts_bulk(frames, 7, parallel=False)
    par = run_auto_trader._generate_forecasts_bulk(frames, 7, parallel=True, max_workers=4)

    assert seq == par
