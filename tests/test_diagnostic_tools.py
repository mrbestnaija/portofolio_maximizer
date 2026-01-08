import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from scripts import liquidate_open_trades, generate_signal_routing_overrides


def _init_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            trade_date DATE,
            action TEXT NOT NULL,
            shares REAL NOT NULL,
            price REAL NOT NULL,
            commission REAL,
            signal_id INTEGER,
            realized_pnl REAL,
            realized_pnl_pct REAL,
            holding_period_days INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def test_liquidate_open_trades_marks_realized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "trades.db"
    _init_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO trade_executions (ticker, action, shares, price, commission) VALUES (?,?,?,?,?)",
        ("AAPL", "BUY", 10, 10.0, 1.0),
    )
    cur.execute(
        "INSERT INTO trade_executions (ticker, action, shares, price, commission) VALUES (?,?,?,?,?)",
        ("CL=F", "SELL", 5, 20.0, 2.0),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        liquidate_open_trades,
        "_fetch_prices",
        lambda tickers: {"AAPL": 12.0, "CL=F": 18.0},
    )

    runner = CliRunner()
    result = runner.invoke(liquidate_open_trades.main, ["--db-path", str(db_path)])
    assert result.exit_code == 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, realized_pnl, realized_pnl_pct FROM trade_executions ORDER BY ticker"
    )
    rows = cur.fetchall()
    conn.close()

    aapl = rows[0]
    clf = rows[1]
    assert pytest.approx(aapl[1], rel=1e-6) == 19.0  # (12-10)*10 - 1
    assert pytest.approx(aapl[2], rel=1e-6) == 0.19
    assert pytest.approx(clf[1], rel=1e-6) == 8.0  # (20-18)*5 - 2
    assert pytest.approx(clf[2], rel=1e-6) == 0.08


def test_generate_signal_routing_overrides(tmp_path: Path) -> None:
    proposals = {
        "time_series_thresholds": [
            {
                "ticker": "CL=F",
                "confidence_threshold": 0.5,
                "min_expected_return": 0.001,
                "total_trades": 29,
                "win_rate": 0.86,
                "profit_factor": 38.6,
                "annualized_pnl": 1000.0,
            }
        ],
        "transaction_costs": [
            {
                "group": "US_EQUITY",
                "roundtrip_cost_median_bps": 10.0,
                "suggested_roundtrip_cost_bps": 12.5,
            }
        ],
    }
    proposals_path = tmp_path / "proposals.json"
    proposals_path.write_text(json.dumps(proposals), encoding="utf-8")
    output = tmp_path / "overrides.yml"

    runner = CliRunner()
    result = runner.invoke(
        generate_signal_routing_overrides.main,
        ["--proposals-path", str(proposals_path), "--output", str(output)],
    )
    assert result.exit_code == 0

    content = output.read_text(encoding="utf-8")
    assert "CL=F" in content
    assert "US_EQUITY" in content
