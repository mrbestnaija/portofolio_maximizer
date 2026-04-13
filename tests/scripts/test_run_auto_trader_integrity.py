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


# ---------------------------------------------------------------------------
# B2: funnel audit dedup guard
# ---------------------------------------------------------------------------

def test_write_funnel_audit_dedup_same_key_writes_once(tmp_path, monkeypatch):
    """Calling _write_funnel_audit_entry twice with the same (ts_signal_id, reason)
    must produce exactly one line in funnel_audit.jsonl (dedup guard — B2 fix)."""
    import scripts.run_auto_trader as mod

    log_path = tmp_path / "funnel_audit.jsonl"
    monkeypatch.setattr(mod, "FUNNEL_AUDIT_LOG_PATH", log_path)
    # Reset the module-level dedup set so prior test state doesn't interfere
    mod._FUNNEL_LOGGED.clear()

    mod._write_funnel_audit_entry(
        ticker="AAPL", ts_signal_id="ts_AAPL_0001", reason="SNR_GATE",
        confidence=0.50, snr=1.2, expected_return=0.001,
    )
    mod._write_funnel_audit_entry(
        ticker="AAPL", ts_signal_id="ts_AAPL_0001", reason="SNR_GATE",
        confidence=0.50, snr=1.2, expected_return=0.001,
    )

    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1, (
        f"Expected 1 funnel entry for duplicate (tsid, reason); got {len(lines)}"
    )


def test_write_funnel_audit_different_reason_writes_both(tmp_path, monkeypatch):
    """Different reason codes for the same ts_signal_id must each produce one entry."""
    import scripts.run_auto_trader as mod

    log_path = tmp_path / "funnel_audit.jsonl"
    monkeypatch.setattr(mod, "FUNNEL_AUDIT_LOG_PATH", log_path)
    mod._FUNNEL_LOGGED.clear()

    mod._write_funnel_audit_entry(
        ticker="AAPL", ts_signal_id="ts_AAPL_0002", reason="CONFIDENCE_GATE",
        confidence=0.50, snr=None, expected_return=None,
    )
    mod._write_funnel_audit_entry(
        ticker="AAPL", ts_signal_id="ts_AAPL_0002", reason="SNR_GATE",
        confidence=0.50, snr=1.2, expected_return=None,
    )

    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2, (
        f"Expected 2 funnel entries for distinct reasons; got {len(lines)}"
    )
