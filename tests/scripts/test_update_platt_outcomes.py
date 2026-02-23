"""Tests for scripts/update_platt_outcomes.py — Platt scaling outcome reconciliation."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
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


def _make_db(path: Path, trades: List[Dict[str, Any]]) -> None:
    """Create minimal trade_executions table and populate with given rows."""
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            signal_id INTEGER,
            realized_pnl REAL,
            realized_pnl_pct REAL,
            is_close INTEGER
        )"""
    )
    for t in trades:
        cur.execute(
            "INSERT INTO trade_executions (signal_id, realized_pnl, realized_pnl_pct, is_close) "
            "VALUES (?, ?, ?, ?)",
            (t.get("signal_id"), t.get("realized_pnl"), t.get("realized_pnl_pct"), t.get("is_close", 1)),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReconcile:
    def test_entry_with_matching_closed_trade_gets_outcome(self, tmp_path):
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

    def test_entry_already_has_outcome_is_not_overwritten(self, tmp_path):
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
