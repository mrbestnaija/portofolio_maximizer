"""Tests for scripts/update_platt_outcomes.py — Platt scaling outcome reconciliation."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jsonl(entries: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_execution_log(entries: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


def _make_db(path: Path, trades: List[Dict[str, Any]]) -> None:
    """Create minimal trade_executions table and populate with given rows.

    Phase 7.13-A2: table includes ts_signal_id TEXT column.
    Tests pass integer signal_id values; _make_db stores str(signal_id) in ts_signal_id
    so the reconcile() query (which now targets ts_signal_id) matches correctly.
    """
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            signal_id INTEGER,
            ts_signal_id TEXT,
            ticker TEXT,
            trade_date TEXT,
            realized_pnl REAL,
            realized_pnl_pct REAL,
            is_close INTEGER,
            entry_trade_id INTEGER,
            holding_period_days INTEGER
        )"""
    )
    for t in trades:
        raw_sid = t.get("signal_id")
        ts_sid = t.get("ts_signal_id")
        if ts_sid is None and raw_sid is not None:
            ts_sid = str(raw_sid)
        cur.execute(
            "INSERT INTO trade_executions (signal_id, ts_signal_id, ticker, trade_date, realized_pnl, realized_pnl_pct, is_close, entry_trade_id, holding_period_days) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                raw_sid,
                ts_sid,
                t.get("ticker"),
                t.get("trade_date"),
                t.get("realized_pnl"),
                t.get("realized_pnl_pct"),
                t.get("is_close", 1),
                t.get("entry_trade_id"),
                t.get("holding_period_days"),
            ),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReconcile:
    def test_entry_with_matching_closed_trade_gets_outcome(self, tmp_path, capsys):
        """Entry with signal_id matching a closed trade should receive outcome field."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        _make_db(db, [{"signal_id": 42, "realized_pnl": 25.50, "realized_pnl_pct": 0.025, "is_close": 1}])
        _make_jsonl([{"signal_id": 42, "confidence": 0.9, "ticker": "AAPL"}], log)

        total, updated, already_done = reconcile(db_path=db, log_path=log, dry_run=False)

        assert total == 1
        assert updated == 1
        assert already_done == 0
        output = capsys.readouterr().out
        assert "matched=1" in output
        assert "matched_new=1" in output
        assert "already_done=0" in output

        entries = _read_jsonl(log)
        assert "outcome" in entries[0]
        assert entries[0]["outcome"]["win"] is True
        assert abs(entries[0]["outcome"]["pnl"] - 25.50) < 0.01
        assert entries[0]["outcome"]["pnl_pct"] == pytest.approx(0.025)

    def test_entry_without_signal_id_is_skipped(self, tmp_path):
        """Entry without signal_id should not be modified."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        _make_db(db, [])
        _make_jsonl([{"confidence": 0.8, "ticker": "MSFT"}], log)

        total, updated, already_done = reconcile(db_path=db, log_path=log, dry_run=False)

        assert updated == 0
        entries = _read_jsonl(log)
        assert "outcome" not in entries[0]

    def test_entry_already_has_outcome_is_not_overwritten(self, tmp_path, capsys):
        """Entry that already has outcome should be left unchanged."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        existing_outcome = {"win": True, "pnl": 10.0, "pnl_pct": 0.01}
        _make_db(db, [{"signal_id": 7, "realized_pnl": -5.0, "realized_pnl_pct": -0.005, "is_close": 1}])
        _make_jsonl([{"signal_id": 7, "confidence": 0.85, "outcome": existing_outcome}], log)

        total, updated, already_done = reconcile(db_path=db, log_path=log, dry_run=False)

        assert updated == 0
        assert already_done == 1
        output = capsys.readouterr().out
        assert "matched=1" in output
        assert "matched_new=0" in output
        assert "already_done=1" in output
        entries = _read_jsonl(log)
        # Original outcome unchanged
        assert entries[0]["outcome"] == existing_outcome

    def test_dry_run_does_not_write_file(self, tmp_path):
        """dry_run=True should compute updates but not modify the file."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        _make_db(db, [{"signal_id": 1, "realized_pnl": -8.0, "realized_pnl_pct": -0.008, "is_close": 1}])
        original = [{"signal_id": 1, "confidence": 0.9}]
        _make_jsonl(original, log)
        original_mtime = log.stat().st_mtime

        total, updated, already_done = reconcile(db_path=db, log_path=log, dry_run=True)

        assert updated == 1
        # File must not be modified
        assert log.stat().st_mtime == original_mtime
        entries = _read_jsonl(log)
        assert "outcome" not in entries[0]

    def test_db_not_found_exits_with_error(self, tmp_path, capsys):
        """Missing DB should trigger sys.exit(1)."""
        from scripts.update_platt_outcomes import reconcile

        missing_db = tmp_path / "no_such.db"
        log = tmp_path / "qv.jsonl"
        _make_jsonl([{"signal_id": 1, "confidence": 0.8}], log)

        with pytest.raises(SystemExit) as exc_info:
            reconcile(db_path=missing_db, log_path=log, dry_run=False)

        assert exc_info.value.code == 1

    def test_multiple_closes_same_signal_id_uses_largest_pnl(self, tmp_path):
        """When multiple closes share a signal_id, the one with abs(pnl) wins."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        # Two closing trades for signal 5 — larger absolute pnl wins
        _make_db(db, [
            {"signal_id": 5, "realized_pnl": 3.0, "realized_pnl_pct": 0.003, "is_close": 1},
            {"signal_id": 5, "realized_pnl": -50.0, "realized_pnl_pct": -0.05, "is_close": 1},
        ])
        _make_jsonl([{"signal_id": 5, "confidence": 0.92}], log)

        reconcile(db_path=db, log_path=log, dry_run=False)

        entries = _read_jsonl(log)
        # -50 has larger abs value → should be selected
        assert entries[0]["outcome"]["win"] is False
        assert abs(entries[0]["outcome"]["pnl"] - (-50.0)) < 0.01

    def test_losing_trade_outcome_win_is_false(self, tmp_path):
        """Negative pnl should produce outcome['win'] = False."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        _make_db(db, [{"signal_id": 9, "realized_pnl": -12.5, "realized_pnl_pct": -0.012, "is_close": 1}])
        _make_jsonl([{"signal_id": 9, "confidence": 0.88}], log)

        reconcile(db_path=db, log_path=log, dry_run=False)

        entries = _read_jsonl(log)
        assert entries[0]["outcome"]["win"] is False
        assert entries[0]["outcome"]["pnl"] == pytest.approx(-12.5)

    def test_no_matching_trade_leaves_entry_unchanged(self, tmp_path):
        """Entry with signal_id that has no matching trade should remain without outcome."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        _make_db(db, [])  # empty trades
        _make_jsonl([{"signal_id": 99, "confidence": 0.75}], log)

        total, updated, _ = reconcile(db_path=db, log_path=log, dry_run=False)

        assert updated == 0
        entries = _read_jsonl(log)
        assert "outcome" not in entries[0]

    def test_mixed_entries_only_matched_updated(self, tmp_path):
        """Batch with partial matches — only entries with matching closed trades get outcome."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        _make_db(db, [{"signal_id": 10, "realized_pnl": 5.0, "realized_pnl_pct": 0.005, "is_close": 1}])
        entries = [
            {"signal_id": 10, "confidence": 0.82},  # has match
            {"signal_id": 20, "confidence": 0.91},  # no match
            {"confidence": 0.70},                   # no signal_id
        ]
        _make_jsonl(entries, log)

        total, updated, already_done = reconcile(db_path=db, log_path=log, dry_run=False)

        assert total == 3
        assert updated == 1
        rows = _read_jsonl(log)
        assert "outcome" in rows[0]
        assert "outcome" not in rows[1]
        assert "outcome" not in rows[2]

    def test_jsonl_not_found_returns_zero(self, tmp_path, capsys):
        """Missing JSONL should return (0, 0, 0) without error."""
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        _make_db(db, [])
        missing_log = tmp_path / "no_such.jsonl"

        total, updated, done = reconcile(db_path=db, log_path=missing_log, dry_run=False)

        assert total == 0
        assert updated == 0
        assert done == 0

    def test_missing_signal_id_can_match_by_symbol_and_time(self, tmp_path):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"
        execution_log = tmp_path / "execution.jsonl"

        _make_db(
            db,
            [
                {
                    "signal_id": None,
                    "ts_signal_id": "ts_close_1",
                    "ticker": "AAPL",
                    "trade_date": "2024-01-06",
                    "realized_pnl": 12.0,
                    "realized_pnl_pct": 0.012,
                    "is_close": 1,
                }
            ],
        )
        _make_execution_log(
            [
                {
                    "run_id": "run_1",
                    "ticker": "AAPL",
                    "signal_timestamp": "2024-01-01T00:00:00",
                    "status": "EXECUTED",
                    "executed": True,
                }
            ],
            execution_log,
        )
        _make_jsonl(
            [
                {
                    "ticker": "AAPL",
                    "forecast_time": "2024-01-01T00:00:00",
                    "forecast_horizon": 5,
                    "confidence": 0.8,
                    "run_id": "run_1",
                }
            ],
            log,
        )

        total, updated, already_done = reconcile(
            db_path=db,
            log_path=log,
            execution_log_path=execution_log,
            dry_run=False,
        )

        assert total == 1
        assert updated == 1
        assert already_done == 0
        entries = _read_jsonl(log)
        assert entries[0]["outcome"]["win"] is True

    def test_time_mismatch_blocks_outcome_even_with_matching_signal_id(self, tmp_path, capsys):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"
        execution_log = tmp_path / "execution.jsonl"

        _make_db(
            db,
            [
                {
                    "signal_id": 42,
                    "ticker": "AAPL",
                    "trade_date": "2024-01-20",
                    "realized_pnl": 25.5,
                    "realized_pnl_pct": 0.025,
                    "is_close": 1,
                }
            ],
        )
        _make_execution_log(
            [
                {
                    "ts_signal_id": "42",
                    "status": "EXECUTED",
                    "executed": True,
                }
            ],
            execution_log,
        )
        _make_jsonl(
            [
                {
                    "signal_id": 42,
                    "ticker": "AAPL",
                    "forecast_time": "2024-01-01T00:00:00",
                    "forecast_horizon": 5,
                    "confidence": 0.9,
                }
            ],
            log,
        )

        total, updated, already_done = reconcile(
            db_path=db,
            log_path=log,
            execution_log_path=execution_log,
            dry_run=False,
        )

        assert total == 1
        assert updated == 0
        assert already_done == 0
        entries = _read_jsonl(log)
        assert "outcome" not in entries[0]
        assert "TIME_MISMATCH" in capsys.readouterr().out

    def test_duplicate_stable_key_does_not_double_match(self, tmp_path, capsys):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"
        execution_log = tmp_path / "execution.jsonl"

        _make_db(
            db,
            [
                {
                    "signal_id": 10,
                    "ticker": "MSFT",
                    "trade_date": "2024-01-06",
                    "realized_pnl": 7.5,
                    "realized_pnl_pct": 0.0075,
                    "is_close": 1,
                }
            ],
        )
        _make_execution_log(
            [
                {
                    "ts_signal_id": "10",
                    "status": "EXECUTED",
                    "executed": True,
                }
            ],
            execution_log,
        )
        _make_jsonl(
            [
                {
                    "signal_id": 10,
                    "ticker": "MSFT",
                    "forecast_time": "2024-01-01T00:00:00",
                    "forecast_horizon": 5,
                    "confidence": 0.7,
                },
                {
                    "signal_id": 10,
                    "ticker": "MSFT",
                    "forecast_time": "2024-01-01T00:00:00",
                    "forecast_horizon": 5,
                    "confidence": 0.71,
                },
            ],
            log,
        )

        total, updated, already_done = reconcile(
            db_path=db,
            log_path=log,
            execution_log_path=execution_log,
            dry_run=False,
        )

        assert total == 2
        assert updated == 1
        assert already_done == 0
        rows = _read_jsonl(log)
        assert "outcome" in rows[0]
        assert "outcome" not in rows[1]
        assert "DUPLICATE_KEY" in capsys.readouterr().out

    def test_timestamp_and_forecast_edge_horizon_enable_symbol_time_fallback(self, tmp_path):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        _make_db(
            db,
            [
                {
                    "signal_id": None,
                    "ts_signal_id": "ts_close_1",
                    "ticker": "AAPL",
                    "trade_date": "2024-01-06",
                    "realized_pnl": 12.0,
                    "realized_pnl_pct": 0.012,
                    "is_close": 1,
                }
            ],
        )
        _make_jsonl(
            [
                {
                    "ticker": "AAPL",
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "quant_validation": {
                        "forecast_edge": {"horizon": 5},
                    },
                    "confidence": 0.8,
                }
            ],
            log,
        )

        total, updated, already_done = reconcile(db_path=db, log_path=log, dry_run=False)

        assert total == 1
        assert updated == 1
        assert already_done == 0
        entries = _read_jsonl(log)
        assert entries[0]["outcome"]["win"] is True

    def test_future_expected_close_is_marked_not_yet_eligible(self, tmp_path, capsys):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        _make_db(db, [])
        _make_jsonl(
            [
                {
                    "ticker": "MSFT",
                    "signal_id": "ts_future_1",
                    "timestamp": "2999-01-01T00:00:00+00:00",
                    "quant_validation": {
                        "forecast_edge": {"horizon": 5},
                    },
                    "confidence": 0.8,
                }
            ],
            log,
        )

        total, updated, already_done = reconcile(db_path=db, log_path=log, dry_run=False)

        assert total == 1
        assert updated == 0
        assert already_done == 0
        output = capsys.readouterr().out
        assert "not_yet_eligible=1" in output
        assert "eligibility_window=" in output
        assert "\"earliest_expected_close_date\"" in output
        entries = _read_jsonl(log)
        assert "outcome" not in entries[0]

    def test_execution_log_reject_skips_pending_outcome_lookup(self, tmp_path, capsys):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"
        execution_log = tmp_path / "execution.jsonl"

        _make_db(
            db,
            [{"signal_id": 77, "ticker": "AAPL", "trade_date": "2024-01-06", "realized_pnl": 9.0, "is_close": 1}],
        )
        _make_execution_log(
            [
                {
                    "ts_signal_id": "77",
                    "status": "REJECTED",
                    "executed": False,
                    "reason": "Non-actionable signal",
                }
            ],
            execution_log,
        )
        _make_jsonl(
            [{"signal_id": 77, "ticker": "AAPL", "forecast_time": "2024-01-01T00:00:00", "forecast_horizon": 5}],
            log,
        )

        total, updated, already_done = reconcile(
            db_path=db,
            log_path=log,
            execution_log_path=execution_log,
            dry_run=False,
        )

        assert total == 1
        assert updated == 0
        assert already_done == 0
        assert "outcome" not in _read_jsonl(log)[0]
        assert "NOT_EXECUTED" in capsys.readouterr().out

    def test_missing_execution_log_entry_is_not_treated_as_pending(self, tmp_path, capsys):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"
        execution_log = tmp_path / "execution.jsonl"

        _make_db(
            db,
            [{"signal_id": 88, "ticker": "MSFT", "trade_date": "2024-01-06", "realized_pnl": 4.0, "is_close": 1}],
        )
        _make_execution_log([], execution_log)
        _make_jsonl(
            [{"signal_id": 88, "ticker": "MSFT", "forecast_time": "2024-01-01T00:00:00", "forecast_horizon": 5}],
            log,
        )

        total, updated, already_done = reconcile(
            db_path=db,
            log_path=log,
            execution_log_path=execution_log,
            dry_run=False,
        )

        assert total == 1
        assert updated == 0
        assert already_done == 0
        assert "outcome" not in _read_jsonl(log)[0]
        assert "NO_EXECUTION_RECORD" in capsys.readouterr().out

    def test_execution_gate_can_fallback_to_unique_run_and_ticker(self, tmp_path):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"
        execution_log = tmp_path / "execution.jsonl"

        _make_db(
            db,
            [{"signal_id": 111, "ticker": "AAPL", "trade_date": "2024-01-06", "realized_pnl": 6.0, "is_close": 1}],
        )
        _make_execution_log(
            [
                {
                    "run_id": "run_2",
                    "ticker": "AAPL",
                    "status": "EXECUTED",
                    "executed": True,
                }
            ],
            execution_log,
        )
        _make_jsonl(
            [
                {
                    "signal_id": 111,
                    "ticker": "AAPL",
                    "run_id": "run_2",
                }
            ],
            log,
        )

        total, updated, already_done = reconcile(
            db_path=db,
            log_path=log,
            execution_log_path=execution_log,
            dry_run=False,
        )

        assert total == 1
        assert updated == 1
        assert already_done == 0
        assert _read_jsonl(log)[0]["outcome"]["win"] is True

    def test_not_yet_eligible_reports_visible_execution_log(self, tmp_path, capsys):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"
        execution_log = tmp_path / "execution.jsonl"

        _make_db(db, [])
        _make_execution_log(
            [
                {
                    "ts_signal_id": "ts_future_1",
                    "status": "EXECUTED",
                    "executed": True,
                }
            ],
            execution_log,
        )
        _make_jsonl(
            [
                {
                    "signal_id": "ts_future_1",
                    "ticker": "AAPL",
                    "forecast_time": "2099-01-01T00:00:00+00:00",
                    "forecast_horizon": 30,
                }
            ],
            log,
        )

        total, updated, already_done = reconcile(
            db_path=db,
            log_path=log,
            execution_log_path=execution_log,
            dry_run=False,
        )

        assert total == 1
        assert updated == 0
        assert already_done == 0
        output = capsys.readouterr().out
        assert "NOT_YET_ELIGIBLE" in output
        assert "execution_log_loaded=1" in output
        assert "eligibility_window=" in output

    def test_open_only_leg_is_classified_as_lifecycle_lag(self, tmp_path, capsys):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"
        execution_log = tmp_path / "execution.jsonl"

        _make_db(
            db,
            [
                {
                    "signal_id": 99,
                    "ticker": "NVDA",
                    "trade_date": "2024-01-02",
                    "realized_pnl": None,
                    "realized_pnl_pct": None,
                    "is_close": 0,
                }
            ],
        )
        _make_execution_log(
            [
                {
                    "ts_signal_id": "99",
                    "status": "EXECUTED",
                    "executed": True,
                }
            ],
            execution_log,
        )
        _make_jsonl(
            [{"signal_id": 99, "ticker": "NVDA", "forecast_time": "2024-01-01T00:00:00", "forecast_horizon": 5}],
            log,
        )

        total, updated, already_done = reconcile(
            db_path=db,
            log_path=log,
            execution_log_path=execution_log,
            dry_run=False,
        )

        assert total == 1
        assert updated == 0
        assert already_done == 0
        assert "outcome" not in _read_jsonl(log)[0]
        assert "OPEN_ONLY_LIFECYCLE_LAG" in capsys.readouterr().out

    def test_malformed_execution_log_is_error_and_not_loaded(self, tmp_path, capsys):
        from scripts.update_platt_outcomes import reconcile

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"
        execution_log = tmp_path / "execution.jsonl"

        _make_db(
            db,
            [{"signal_id": 321, "ticker": "AAPL", "trade_date": "2024-01-06", "realized_pnl": 10.0, "is_close": 1}],
        )
        execution_log.write_text("{bad json}\nnot-json\n", encoding="utf-8")
        _make_jsonl(
            [{"signal_id": 321, "ticker": "AAPL", "forecast_time": "2024-01-01T00:00:00+00:00", "forecast_horizon": 5}],
            log,
        )

        total, updated, already_done = reconcile(
            db_path=db,
            log_path=log,
            execution_log_path=execution_log,
            dry_run=False,
        )

        assert total == 1
        assert updated == 0
        assert already_done == 0
        output = capsys.readouterr().out
        assert "execution_log_loaded=0" in output
        assert "execution_log_integrity=ERROR_NO_VALID_EVENTS" in output
        assert "evidence_status=ERROR" in output
        assert "\"lines_parsed\": 0" in output
        assert "\"lines_bad\": 2" in output
        assert "EXECUTION_LOG_CORRUPT" in output
        assert "evidence_status=OK" not in output
        assert "outcome" not in _read_jsonl(log)[0]

    def test_same_day_future_close_is_not_yet_eligible_at_timestamp_level(
        self,
        tmp_path,
        capsys,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from scripts import update_platt_outcomes as mod

        db = tmp_path / "test.db"
        log = tmp_path / "qv.jsonl"

        _make_db(db, [])
        _make_jsonl(
            [
                {
                    "signal_id": "ts_future_same_day",
                    "ticker": "MSFT",
                    "forecast_time": "2026-03-05T07:18:27+00:00",
                    "forecast_horizon": 0,
                }
            ],
            log,
        )
        monkeypatch.setattr(mod, "now_utc", lambda: datetime(2026, 3, 5, 5, 18, 27, tzinfo=timezone.utc))

        total, updated, already_done = mod.reconcile(db_path=db, log_path=log, dry_run=False)

        assert total == 1
        assert updated == 0
        assert already_done == 0
        output = capsys.readouterr().out
        assert "not_yet_eligible=1" in output
        assert "NOT_YET_ELIGIBLE" in output
        assert "expected_close_ts_in_future" in output
        assert "outcome" not in _read_jsonl(log)[0]
