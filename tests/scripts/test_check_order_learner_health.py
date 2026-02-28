from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts import check_order_learner_health as mod


MODEL_ORDER_STATS_DDL = """
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


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(MODEL_ORDER_STATS_DDL)
    conn.execute(
        """
        CREATE TABLE time_series_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            model_type TEXT,
            aic REAL,
            created_at TIMESTAMP
        )
        """
    )
    return conn


def test_check_coverage_warns_when_no_qualified_rows() -> None:
    conn = _make_conn()
    conn.execute(
        """
        INSERT INTO model_order_stats
            (ticker, model_type, regime, order_params, n_fits, aic_sum, bic_sum, best_aic, last_used, first_seen)
        VALUES ('AAPL', 'GARCH', '__none__', '{}', 1, 120.0, 130.0, 120.0, DATE('now'), DATE('now'))
        """
    )
    result = mod.check_coverage(conn, min_fits=3)
    conn.close()

    assert result["status"] == "WARN"
    assert result["total_entries"] == 1
    assert result["qualified_entries"] == 0


def test_check_coverage_ok_when_qualified_rows_exist() -> None:
    conn = _make_conn()
    conn.execute(
        """
        INSERT INTO model_order_stats
            (ticker, model_type, regime, order_params, n_fits, aic_sum, bic_sum, best_aic, last_used, first_seen)
        VALUES ('AAPL', 'GARCH', '__none__', '{}', 3, 360.0, 390.0, 110.0, DATE('now'), DATE('now'))
        """
    )
    result = mod.check_coverage(conn, min_fits=3)
    conn.close()

    assert result["status"] == "OK"
    assert result["qualified_entries"] == 1


def test_check_aic_drift_uses_best_cached_entry_per_model() -> None:
    conn = _make_conn()
    conn.execute(
        """
        INSERT INTO time_series_forecasts (ticker, model_type, aic, created_at)
        VALUES ('AAPL', 'SARIMAX', 100.0, DATETIME('now'))
        """
    )
    conn.executemany(
        """
        INSERT INTO model_order_stats
            (ticker, model_type, regime, order_params, n_fits, aic_sum, bic_sum, best_aic, last_used, first_seen)
        VALUES (?, 'SARIMAX', '__none__', ?, 3, ?, 0.0, ?, DATE('now'), DATE('now'))
        """,
        [
            ("AAPL", '{"order":[1,1,1]}', 315.0, 105.0),
            ("AAPL", '{"order":[2,1,2]}', 480.0, 160.0),
        ],
    )
    result = mod.check_aic_drift(conn, lookback_days=30, drift_threshold=0.10)
    conn.close()

    assert result["status"] == "OK"
    assert result["alerts"] == []


def test_run_health_check_missing_db_fails_closed(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing.db"
    assert mod.run_health_check(str(missing_db)) == 1


def test_check_snapshot_store_errors_on_invalid_manifest_path(tmp_path: Path, monkeypatch) -> None:
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "manifest.json").write_text(
        json.dumps({"AAPL|GARCH|NONE": {"path": "../evil.pkl"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "forcester_ts.model_snapshot_store.ModelSnapshotStore.DEFAULT_SNAPSHOT_DIR",
        snap_dir,
    )

    result = mod.check_snapshot_store()

    assert result["status"] == "ERROR"
    assert result["invalid_paths"] == ["../evil.pkl"]
