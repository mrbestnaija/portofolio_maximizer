import json
from pathlib import Path

from scripts.check_dashboard_health import (
    _load_json,
    _summarize_forecaster_health,
    _summarize_tickers,
)


def test_load_json_parses_valid_payload(tmp_path):
    path = tmp_path / "dashboard_data.json"
    payload = {"meta": {"run_id": "test_run"}}
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = _load_json(path)
    assert loaded["meta"]["run_id"] == "test_run"


def test_summarize_forecaster_health_handles_missing_block(capsys):
    payload = {}
    _summarize_forecaster_health(payload)
    out = capsys.readouterr().out
    assert "Forecaster Health" in out
    assert "no forecaster_health block" in out


def test_summarize_tickers_computes_basic_stats(capsys):
    payload = {
        "signals": [
            {
                "ticker": "AAPL",
                "status": "EXECUTED",
                "realized_pnl": 10.0,
            },
            {
                "ticker": "AAPL",
                "status": "EXECUTED",
                "realized_pnl": -5.0,
            },
            {
                "ticker": "MSFT",
                "status": "EXECUTED",
                "realized_pnl": 0.0,
            },
        ]
    }
    monitoring_cfg = {
        "quant_validation": {"min_profit_factor": 1.0, "min_win_rate": 0.0},
        "per_ticker": {},
    }

    _summarize_tickers(payload, monitoring_cfg)
    out = capsys.readouterr().out

    # AAPL should have 2 trades; MSFT 1 trade.
    assert "AAPL" in out
    assert "2" in out  # trades column for AAPL
    assert "MSFT" in out
    assert "1" in out  # trades column for MSFT

