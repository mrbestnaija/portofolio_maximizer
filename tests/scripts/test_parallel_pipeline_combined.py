from __future__ import annotations

import pandas as pd
import pytest

from scripts import run_auto_trader


class _DummyPreprocessor:
    def handle_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame


def _make_frame(seed: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    base = 100 + seed
    return pd.DataFrame(
        {
            "Open": [base] * 6,
            "High": [base + 1] * 6,
            "Low": [base - 1] * 6,
            "Close": [base, base + 0.1, base + 0.2, base + 0.3, base + 0.4, base + 0.5],
            "Volume": [100] * 6,
        },
        index=idx,
    )


def test_parallel_combined_matches_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [_make_frame(i) for i in range(8)]
    entries = [{"ticker": f"T{i}", "frame": frames[i], "order": i} for i in range(len(frames))]
    preprocessor = _DummyPreprocessor()

    def fake_forecast(frame: pd.DataFrame, horizon: int, **_kwargs):
        last = float(frame["Close"].iloc[-1])
        return ({"horizon": horizon, "forecast": last + horizon}, last)

    monkeypatch.setattr(run_auto_trader, "_generate_time_series_forecast", fake_forecast)

    seq = run_auto_trader._build_candidates_with_forecasts(
        entries,
        preprocessor=preprocessor,
        horizon=5,
        parallel=False,
        max_workers=None,
    )
    par = run_auto_trader._build_candidates_with_forecasts(
        entries,
        preprocessor=preprocessor,
        horizon=5,
        parallel=True,
        max_workers=3,
    )

    assert [c["ticker"] for c in seq] == [c["ticker"] for c in par]
    for left, right in zip(seq, par):
        assert left["quality"]["quality_score"] == right["quality"]["quality_score"]
        assert left["mid_price"] == right["mid_price"]
        assert left["forecast_bundle"] == right["forecast_bundle"]
        assert left["current_price"] == right["current_price"]
