from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_horizon_consistent_backtest import Window, run_horizon_backtest


def test_run_horizon_backtest_writes_json_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_simulate_candidate(**kwargs):
        captured.update(kwargs)
        return {"total_return": 12.0, "profit_factor": 1.5, "win_rate": 0.6, "max_drawdown": 0.1, "total_trades": 3}

    monkeypatch.setattr("scripts.run_horizon_consistent_backtest.simulate_candidate", fake_simulate_candidate)

    class DummyDB:
        pass

    report_path = tmp_path / "report.json"
    payload = run_horizon_backtest(
        db_manager=DummyDB(),  # type: ignore[arg-type]
        tickers=["AAPL"],
        window=Window(start_date="2024-01-01", end_date="2024-02-01"),
        candidate_params={"forecast_horizon": 5},
        guardrails={"confidence_threshold": 0.5},
        initial_capital=10000.0,
        report_path=report_path,
    )

    assert report_path.exists()
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written["window"]["start_date"] == "2024-01-01"
    assert written["tickers"] == ["AAPL"]
    assert written["metrics"]["profit_factor"] == 1.5
    assert payload["metrics"]["total_trades"] == 3
    assert captured["tickers"] == ["AAPL"]
