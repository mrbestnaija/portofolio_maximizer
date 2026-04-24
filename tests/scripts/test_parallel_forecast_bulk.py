from __future__ import annotations

import types

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

    def fake_forecast(frame: pd.DataFrame, horizon: int, **_kwargs):
        last = float(frame["Close"].iloc[-1])
        return ({"horizon": horizon, "forecast": last + horizon}, last)

    monkeypatch.setattr(run_auto_trader, "_generate_time_series_forecast", fake_forecast)

    seq = run_auto_trader._generate_forecasts_bulk(frames, 7, parallel=False)
    par = run_auto_trader._generate_forecasts_bulk(frames, 7, parallel=True, max_workers=4)

    assert seq == par


def test_generate_forecasts_bulk_forwards_execution_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = {f"T{i:02d}": _make_frame(100.0 + i) for i in range(4)}
    seen: list[str | None] = []

    def fake_forecast(frame: pd.DataFrame, horizon: int, **kwargs):
        seen.append(kwargs.get("execution_mode"))
        last = float(frame["Close"].iloc[-1])
        return ({"horizon": horizon, "forecast": last + horizon}, last)

    monkeypatch.setattr(run_auto_trader, "_generate_time_series_forecast", fake_forecast)

    run_auto_trader._generate_forecasts_bulk(frames, 7, parallel=False, execution_mode="live")
    assert seen == ["live"] * len(frames)

    seen.clear()
    run_auto_trader._generate_forecasts_bulk(
        frames,
        7,
        parallel=True,
        max_workers=2,
        execution_mode="synthetic",
    )
    assert len(seen) == len(frames)
    assert set(seen) == {"synthetic"}


def test_edge_safe_runtime_defaults_to_serial_and_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENABLE_PARALLEL_FORECASTS", raising=False)
    monkeypatch.delenv("ENABLE_PARALLEL_TICKER_PROCESSING", raising=False)
    monkeypatch.delenv("ENABLE_PARALLEL_TICKERS", raising=False)
    monkeypatch.setenv("PMX_EDGE_SAFE_RUNTIME", "1")

    profile = run_auto_trader._resolve_parallel_runtime_profile()

    assert profile["edge_safe_runtime"] is True
    assert profile["parallel_forecasts"] is False
    assert profile["parallel_ticker_processing"] is False


def test_edge_safe_runtime_allows_explicit_gpu_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PMX_EDGE_SAFE_RUNTIME", "1")
    monkeypatch.setenv("ENABLE_GPU_PARALLEL", "1")
    monkeypatch.setattr(run_auto_trader, "_GPU_PARALLEL_ENABLED", None)
    monkeypatch.setattr(
        run_auto_trader,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
    )

    assert run_auto_trader._gpu_parallel_enabled() is True
