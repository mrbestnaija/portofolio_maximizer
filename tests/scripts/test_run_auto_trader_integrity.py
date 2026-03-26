from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from scripts.run_auto_trader import _tagged_only_cross_mode_contamination


def _make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE trade_executions (id INTEGER PRIMARY KEY, is_contaminated INTEGER DEFAULT 0)"
    )
    return conn


def test_tagged_only_cross_mode_contamination_returns_true_for_quarantined_rows():
    conn = _make_db()
    conn.executemany(
        "INSERT INTO trade_executions (id, is_contaminated) VALUES (?, ?)",
        [(252, 1), (255, 1)],
    )
    violations = [
        SimpleNamespace(
            severity="HIGH",
            check_name="CROSS_MODE_CONTAMINATION",
            affected_ids=[252, 255],
        )
    ]
    assert _tagged_only_cross_mode_contamination(violations, conn) is True


def test_tagged_only_cross_mode_contamination_rejects_untagged_rows():
    conn = _make_db()
    conn.executemany(
        "INSERT INTO trade_executions (id, is_contaminated) VALUES (?, ?)",
        [(252, 1), (255, 0)],
    )
    violations = [
        SimpleNamespace(
            severity="HIGH",
            check_name="CROSS_MODE_CONTAMINATION",
            affected_ids=[252, 255],
        )
    ]
    assert _tagged_only_cross_mode_contamination(violations, conn) is False


def test_tagged_only_cross_mode_contamination_rejects_mixed_high_violations():
    conn = _make_db()
    conn.execute("INSERT INTO trade_executions (id, is_contaminated) VALUES (?, ?)", (252, 1))
    violations = [
        SimpleNamespace(
            severity="HIGH",
            check_name="CROSS_MODE_CONTAMINATION",
            affected_ids=[252],
        ),
        SimpleNamespace(
            severity="HIGH",
            check_name="ORPHANED_POSITION",
            affected_ids=[249],
        ),
    ]
    assert _tagged_only_cross_mode_contamination(violations, conn) is False
