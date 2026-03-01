from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts import purge_invalid_order_learner_rows as mod


DDL = """
CREATE TABLE model_order_stats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT    NOT NULL,
    model_type   TEXT    NOT NULL,
    regime       TEXT,
    order_params TEXT    NOT NULL,
    n_fits       INTEGER DEFAULT 0,
    aic_sum      REAL    DEFAULT 0.0,
    bic_sum      REAL    DEFAULT 0.0,
    best_aic     REAL,
    last_used    DATE,
    first_seen   DATE,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "orders.db"
    conn = sqlite3.connect(db_path)
    conn.execute(DDL)
    conn.executemany(
        """
        INSERT INTO model_order_stats
            (ticker, model_type, regime, order_params, n_fits, aic_sum, bic_sum, best_aic, last_used, first_seen)
        VALUES (?, 'GARCH', '__none__', '{}', ?, 100.0, 110.0, 90.0, DATE('now'), DATE('now'))
        """,
        [
            ("AAPL", 3),
            ("Close", 4),
            ("price", 2),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_purge_invalid_rows_dry_run_leaves_db_unchanged(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path)

    result = mod.purge_invalid_rows(db_path, apply=False)

    assert result["invalid_rows"] == 2
    assert result["deleted_rows"] == 0
    assert result["invalid_tickers"] == {"Close": 1, "price": 1}

    conn = sqlite3.connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM model_order_stats").fetchone()[0]
    conn.close()
    assert total == 3


def test_purge_invalid_rows_apply_deletes_only_invalid_rows(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path)

    result = mod.purge_invalid_rows(db_path, apply=True)

    assert result["invalid_rows"] == 2
    assert result["deleted_rows"] == 2

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT ticker FROM model_order_stats ORDER BY ticker"
    ).fetchall()
    conn.close()
    assert rows == [("AAPL",)]


def test_main_json_emits_machine_readable_payload(tmp_path: Path, capsys) -> None:
    db_path = _make_db(tmp_path)

    exit_code = mod.main(["--db", str(db_path), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["invalid_rows"] == 2
    assert payload["deleted_rows"] == 0
