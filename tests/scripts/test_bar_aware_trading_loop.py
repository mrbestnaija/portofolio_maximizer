from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import run_auto_trader


def _make_raw_window(bar_ts: pd.Timestamp) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [
            bar_ts - pd.Timedelta(days=2),
            bar_ts - pd.Timedelta(days=1),
            bar_ts,
        ]
    )
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1_000_000, 1_000_000, 1_000_000],
        },
        index=idx,
    )


def test_bar_timestamp_gate_skips_same_bar(tmp_path: Path) -> None:
    gate = run_auto_trader.BarTimestampGate(state_path=tmp_path / "bar_state.json", persist=False)
    ts = pd.Timestamp("2024-01-10")

    is_new, previous, current = gate.check("AAPL", ts)
    assert is_new is True
    assert previous is None
    assert current

    is_new, previous, current_2 = gate.check("AAPL", ts)
    assert is_new is False
    assert previous == current_2

    is_new, _, _ = gate.check("AAPL", ts + pd.Timedelta(days=1))
    assert is_new is True


def test_bar_aware_loop_skips_second_cycle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"execute": 0}
    bar_ts = pd.Timestamp("2024-01-10")
    raw_window = _make_raw_window(bar_ts)

    # Keep all test artifacts under tmp_path.
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "portfolio_test.db"))
    monkeypatch.setattr(run_auto_trader, "EXECUTION_LOG_PATH", tmp_path / "execution_log.jsonl")
    monkeypatch.setattr(run_auto_trader, "RUN_SUMMARY_LOG_PATH", tmp_path / "run_summary.jsonl")
    monkeypatch.setattr(run_auto_trader, "DASHBOARD_DATA_PATH", tmp_path / "dashboard_data.json")
    monkeypatch.setattr(run_auto_trader, "_emit_dashboard_png", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_auto_trader.atexit, "register", lambda *_args, **_kwargs: None)

    # Avoid writing provenance artifacts to repo paths.
    import etl.database_manager as dbm

    def _raise_skip(*_args, **_kwargs):
        raise RuntimeError("skip provenance artifact emission")

    monkeypatch.setattr(dbm.DatabaseManager, "get_data_provenance_summary", _raise_skip)

    # Deterministic ticker universe and window; no network dependency.
    class DummyUniverse:
        def __init__(self, tickers: list[str]) -> None:
            self.tickers = tickers
            self.universe_source = "test"
            self.active_source = "test"

    monkeypatch.setattr(
        run_auto_trader,
        "resolve_ticker_universe",
        lambda base_tickers, include_frontier, active_source, use_discovery: DummyUniverse(base_tickers),
    )

    class DummyDataSourceManager:
        def __init__(self, *args, **kwargs) -> None:
            self.active_extractor = type("Extractor", (), {"name": "test"})()

        def get_active_source(self) -> str:
            return "test"

    monkeypatch.setattr(run_auto_trader, "DataSourceManager", DummyDataSourceManager)
    monkeypatch.setattr(run_auto_trader, "_prepare_market_window", lambda *_args, **_kwargs: raw_window)
    monkeypatch.setattr(run_auto_trader, "_validate_market_window", lambda *_args, **_kwargs: True)

    def fake_forecast(price_frame: pd.DataFrame, horizon: int):
        series = pd.Series(
            [101.0, 102.0],
            index=pd.date_range("2024-01-11", periods=2, freq="D"),
        )
        return {"horizon": horizon, "ensemble_forecast": {"forecast": series}}, float(price_frame["Close"].iloc[-1])

    monkeypatch.setattr(run_auto_trader, "_generate_time_series_forecast", fake_forecast)

    def fake_execute_signal(*, ticker: str, **_kwargs):
        calls["execute"] += 1
        return {"ticker": ticker, "status": "EXECUTED"}

    monkeypatch.setattr(run_auto_trader, "_execute_signal", fake_execute_signal)

    callback = getattr(run_auto_trader.main, "callback", None)
    assert callable(callback)
    callback(
        tickers="AAPL",
        include_frontier_tickers=False,
        lookback_days=30,
        forecast_horizon=2,
        initial_capital=10000.0,
        cycles=2,
        sleep_seconds=0,
        bar_aware=True,
        persist_bar_state=False,
        bar_state_path=str(tmp_path / "bar_state.json"),
        enable_llm=False,
        llm_model="test",
        verbose=False,
    )
    assert calls["execute"] == 1
