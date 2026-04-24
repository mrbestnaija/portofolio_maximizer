from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts import retire_zombie_opens as mod


def _make_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            action TEXT NOT NULL,
            shares REAL NOT NULL,
            price REAL NOT NULL,
            is_close INTEGER NOT NULL DEFAULT 0,
            is_synthetic INTEGER NOT NULL DEFAULT 0,
            execution_mode TEXT NOT NULL DEFAULT 'live',
            entry_trade_id INTEGER,
            ts_signal_id TEXT
        );
        CREATE TABLE portfolio_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            shares INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            entry_timestamp TEXT NOT NULL,
            stop_loss REAL,
            target_price REAL,
            max_holding_days INTEGER,
            holding_bars INTEGER DEFAULT 0,
            entry_bar_timestamp TEXT,
            last_bar_timestamp TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO portfolio_state (ticker, shares, entry_price, entry_timestamp) "
        "VALUES (?, ?, ?, ?)",
        [
            ("AAPL", 1, 266.58, "2026-04-15T00:00:00+00:00"),
            ("NVDA", 4, 198.98, "2026-04-15T00:00:00+00:00"),
            ("AMZN", 3, 249.84, "2026-04-16T00:00:00+00:00"),
        ],
    )
    rows = [
        (1, "AAPL", "2026-04-15", "BUY", 1.0, 266.58, 0, 0, "live", None, "ts_AAPL_0001"),
        (2, "NVDA", "2026-04-15", "BUY", 1.0, 198.98, 0, 0, "live", None, "ts_NVDA_0001"),
        (3, "NVDA", "2026-04-16", "BUY", 1.0, 198.98, 0, 0, "live", None, "ts_NVDA_0002"),
        (4, "NVDA", "2026-04-16", "BUY", 1.0, 198.98, 0, 0, "live", None, "ts_NVDA_0003"),
        (5, "NVDA", "2026-04-17", "BUY", 1.0, 198.98, 0, 0, "live", None, "ts_NVDA_0004"),
        (6, "NVDA", "2026-04-17", "BUY", 1.0, 198.98, 0, 0, "live", None, "ts_NVDA_0005"),
        (7, "NVDA", "2026-04-17", "BUY", 1.0, 198.98, 0, 0, "live", None, "ts_NVDA_0006"),
        (8, "AMZN", "2026-04-16", "BUY", 1.0, 249.84, 0, 0, "live", None, "ts_AMZN_0001"),
        (9, "AMZN", "2026-04-16", "BUY", 1.0, 249.84, 0, 0, "live", None, "ts_AMZN_0002"),
        (10, "AMZN", "2026-04-17", "BUY", 1.0, 249.84, 0, 0, "live", None, "ts_AMZN_0003"),
        (11, "AMZN", "2026-04-17", "BUY", 1.0, 249.84, 0, 0, "live", None, "ts_AMZN_0004"),
        (12, "AMZN", "2026-04-17", "BUY", 1.0, 249.84, 0, 0, "live", None, "ts_AMZN_0005"),
    ]
    conn.executemany(
        "INSERT INTO trade_executions (id, ticker, trade_date, action, shares, price, is_close, "
        "is_synthetic, execution_mode, entry_trade_id, ts_signal_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_retire_zombie_opens_dry_run_uses_live_portfolio_state(tmp_path, capsys):
    db_path = tmp_path / "portfolio.db"
    _make_db(db_path)

    mod.main(apply=False, db_path=db_path)
    out = capsys.readouterr().out

    assert "Live portfolio_state snapshot:" in out
    assert "AAPL: 1 share(s)" in out
    assert "NVDA: 4 share(s)" in out
    assert "AMZN: 3 share(s)" in out
    assert "RETIRE id=   2" in out
    assert "RETIRE id=   3" in out
    assert "RETIRE id=   8" in out
    assert "RETIRE id=   9" in out


def test_retire_zombie_opens_apply_marks_surplus_rows_synthetic(tmp_path):
    db_path = tmp_path / "portfolio.db"
    _make_db(db_path)

    mod.main(apply=True, db_path=db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        retire_rows = conn.execute(
            "SELECT id, is_synthetic, execution_mode FROM trade_executions "
            "WHERE id IN (2, 3, 8, 9) ORDER BY id"
        ).fetchall()
        keep_rows = conn.execute(
            "SELECT id, is_synthetic, execution_mode FROM trade_executions "
            "WHERE id IN (1, 4, 5, 6, 7, 10, 11, 12) ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    assert all(int(row["is_synthetic"]) == 1 for row in retire_rows)
    assert all(str(row["execution_mode"]) == "zombie_retired" for row in retire_rows)
    assert all(int(row["is_synthetic"]) == 0 for row in keep_rows)
