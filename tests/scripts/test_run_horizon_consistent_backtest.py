from __future__ import annotations

import json
from pathlib import Path

from scripts.run_horizon_consistent_backtest import Window, run_horizon_backtest


class _Cursor:
    def execute(self, sql: str) -> None:  # noqa: ARG002
        return None

    def fetchone(self):
        return ("2024-01-31",)


class _FakeDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.cursor = _Cursor()


def test_run_horizon_backtest_writes_provenance(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "portfolio.db"
    db_path.write_bytes(b"fake-db")
    config_path = tmp_path / "signal_routing_config.yml"
    config_path.write_text("signal_routing: {}\n", encoding="utf-8")
    report_path = tmp_path / "horizon_backtest.json"

    monkeypatch.setattr(
        "scripts.run_horizon_consistent_backtest.simulate_candidate",
        lambda **kwargs: {  # noqa: ARG005
            "profit_factor": 1.2,
            "win_rate": 0.55,
            "total_trades": 4,
            "total_return": 123.0,
            "max_drawdown": 0.1,
        },
    )

    payload = run_horizon_backtest(
        db_manager=_FakeDB(db_path),
        tickers=["AAPL", "MSFT"],
        window=Window(start_date="2024-01-01", end_date="2024-01-31"),
        candidate_params={"forecast_horizon": 10},
        guardrails={"min_confidence": 0.5},
        initial_capital=25000.0,
        report_path=report_path,
        config_paths=[config_path],
    )

    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("dataset_hash", "db_max_ohlcv_date", "config_hash", "git_commit", "config_paths"):
        assert key in payload["provenance"]
        assert key in saved["provenance"]
    assert payload["provenance"]["db_max_ohlcv_date"] == "2024-01-31"
    assert payload["provenance"]["config_paths"] == [str(config_path.resolve())]
