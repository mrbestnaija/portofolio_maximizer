from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts import cleanup_orphaned_positions as mod


def test_cleanup_orphaned_positions_cli_help_runs_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "cleanup_orphaned_positions.py"), "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "ModuleNotFoundError" not in result.stderr


def test_cleanup_orphaned_positions_only_ids_filters_workset(tmp_path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "portfolio.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                trade_date TEXT,
                action TEXT,
                shares REAL,
                price REAL,
                realized_pnl REAL,
                realized_pnl_pct REAL,
                entry_price REAL,
                exit_price REAL,
                holding_period_days INTEGER,
                exit_reason TEXT,
                is_close INTEGER DEFAULT 0
            )
            """
        )
        conn.executemany(
            "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, realized_pnl, is_close) "
            "VALUES (?, ?, ?, 'BUY', 1.0, ?, NULL, 0)",
            [
                (1, "AAPL", "2026-04-15", 266.58),
                (2, "NVDA", "2026-04-15", 198.99),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    dates = pd.date_range("2026-04-14", periods=20, freq="D", tz="UTC")
    market = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(20)],
            "High": [101.0 + i for i in range(20)],
            "Low": [99.0 + i for i in range(20)],
            "Close": [100.5 + i for i in range(20)],
        },
        index=dates,
    )
    monkeypatch.setattr(mod.yf, "download", lambda *args, **kwargs: market)

    mod.main(dry_run=True, db_path=str(db_path), only_ids={1})
    out = capsys.readouterr().out

    assert "   1 AAPL" in out
    assert "   2 NVDA" not in out
