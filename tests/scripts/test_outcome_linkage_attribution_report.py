from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts.outcome_linkage_attribution_report import build_report


def _seed_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                trade_date TEXT,
                bar_timestamp TEXT,
                action TEXT,
                price REAL,
                realized_pnl REAL,
                exit_reason TEXT,
                ts_signal_id TEXT,
                holding_period_days INTEGER,
                entry_trade_id INTEGER,
                entry_price REAL,
                exit_price REAL,
                is_close INTEGER,
                is_diagnostic INTEGER DEFAULT 0,
                is_synthetic INTEGER DEFAULT 0
            );
            CREATE VIEW production_closed_trades AS
            SELECT *
            FROM trade_executions
            WHERE is_close = 1
              AND COALESCE(is_diagnostic, 0) = 0
              AND COALESCE(is_synthetic, 0) = 0;
            """
        )
        conn.execute(
            """
            INSERT INTO trade_executions
            (id, ticker, trade_date, bar_timestamp, action, price, ts_signal_id, is_close)
            VALUES
            (1, 'AAPL', '2026-03-01', '2026-03-01T00:00:00Z', 'BUY', 100.0, 'ts_AAPL_1', 0)
            """
        )
        conn.execute(
            """
            INSERT INTO trade_executions
            (id, ticker, trade_date, bar_timestamp, action, price, realized_pnl, exit_reason,
             ts_signal_id, holding_period_days, entry_trade_id, entry_price, exit_price, is_close)
            VALUES
            (2, 'AAPL', '2026-03-03', '2026-03-03T00:00:00Z', 'SELL', 105.0, 5.0, 'TIME_EXIT',
             'ts_AAPL_1', 2, 1, 100.0, 105.0, 1)
            """
        )
        conn.execute(
            """
            INSERT INTO trade_executions
            (id, ticker, trade_date, bar_timestamp, action, price, ts_signal_id, is_close)
            VALUES
            (3, 'MSFT', '2026-03-01', '2026-03-01T01:00:00Z', 'BUY', 200.0, 'legacy_2026_01', 0)
            """
        )
        conn.execute(
            """
            INSERT INTO trade_executions
            (id, ticker, trade_date, bar_timestamp, action, price, realized_pnl, exit_reason,
             ts_signal_id, holding_period_days, entry_trade_id, entry_price, exit_price, is_close)
            VALUES
            (4, 'MSFT', '2026-03-04', '2026-03-04T01:00:00Z', 'SELL', 198.0, -2.0, 'STOP_LOSS',
             'legacy_2026_01', 3, 3, 200.0, 198.0, 1)
            """
        )
        conn.commit()
    finally:
        conn.close()


def _seed_audit(audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": {
            "ticker": "AAPL",
            "forecast_horizon": 6,
            "end": "2026-03-01T00:00:00Z",
        },
        "signal_context": {
            "ts_signal_id": "ts_AAPL_1",
            "ticker": "AAPL",
            "run_id": "test_run",
            "entry_ts": "2026-03-01T00:00:00Z",
            "forecast_horizon": 6,
        },
    }
    (audit_dir / "forecast_audit_test.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_report_links_closed_trade_to_forecast_audit(tmp_path: Path) -> None:
    db_path = tmp_path / "pmx.db"
    audit_dir = tmp_path / "audits"
    _seed_db(db_path)
    _seed_audit(audit_dir)

    payload = build_report(db_path=db_path, audit_dir=audit_dir, limit=10)
    summary = payload["summary"]
    records = payload["records"]

    assert summary["total_closed_trades"] == 2
    assert summary["linked_closed_trades"] == 1
    assert summary["linked_trade_ratio"] == 0.5
    assert summary["total_ts_trades"] == 1
    assert summary["linked_ts_trades"] == 1
    assert summary["linked_ts_trade_ratio"] == 1.0
    assert summary["ts_trade_coverage"] == 0.5
    assert summary["linked_correct_direction_negative_count"] == 0
    assert summary["high_integrity_violation_count"] == 0
    assert summary["close_before_entry_count"] == 0
    assert summary["closed_missing_exit_reason_count"] == 0

    assert len(records) == 2
    ts_rec = next(r for r in records if r["ts_signal_id"] == "ts_AAPL_1")
    legacy_rec = next(r for r in records if r["ts_signal_id"] == "legacy_2026_01")

    assert ts_rec["outcome_linked"] is True
    assert ts_rec["forecast_direction"] == "BUY"
    assert ts_rec["realized_direction"] == "UP"
    assert ts_rec["direction_match"] is True
    assert ts_rec["audit_file"] == "forecast_audit_test.json"

    assert legacy_rec["outcome_linked"] is False
    assert legacy_rec["realized_direction"] == "DOWN"


def test_build_report_excursion_stub_flag(tmp_path: Path) -> None:
    """Phase 7.39: excursion_data_available must be False in summary (MAE/MFE not stored).
    All records must have excursion_min_pct and excursion_max_pct as None."""
    db_path = tmp_path / "pmx_exc.db"
    audit_dir = tmp_path / "audits"
    _seed_db(db_path)
    _seed_audit(audit_dir)

    payload = build_report(db_path=db_path, audit_dir=audit_dir, limit=10)
    summary = payload["summary"]
    records = payload["records"]

    assert summary["excursion_data_available"] is False
    for rec in records:
        assert rec["excursion_min_pct"] is None, f"excursion_min_pct should be None, got {rec['excursion_min_pct']}"
        assert rec["excursion_max_pct"] is None, f"excursion_max_pct should be None, got {rec['excursion_max_pct']}"


def test_build_report_flags_lifecycle_integrity_violations(tmp_path: Path) -> None:
    db_path = tmp_path / "pmx_integrity.db"
    audit_dir = tmp_path / "audits"
    _seed_db(db_path)
    _seed_audit(audit_dir)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            INSERT INTO trade_executions
            (id, ticker, trade_date, bar_timestamp, action, price, ts_signal_id, is_close)
            VALUES
            (10, 'NVDA', '2026-03-05', '2026-03-05T00:00:00Z', 'BUY', 300.0, 'legacy_nvda', 0)
            """
        )
        conn.execute(
            """
            INSERT INTO trade_executions
            (id, ticker, trade_date, bar_timestamp, action, price, realized_pnl, exit_reason,
             ts_signal_id, holding_period_days, entry_trade_id, entry_price, exit_price, is_close)
            VALUES
            (11, 'NVDA', '2026-03-04', '2026-03-04T00:00:00Z', 'SELL', 295.0, -5.0, NULL,
             'legacy_nvda', 1, 10, 300.0, 295.0, 1)
            """
        )
        conn.commit()
    finally:
        conn.close()

    payload = build_report(db_path=db_path, audit_dir=audit_dir, limit=20)
    summary = payload["summary"]
    assert summary["high_integrity_violation_count"] >= 1
    assert summary["close_before_entry_count"] >= 1
    assert summary["closed_missing_exit_reason_count"] >= 1
