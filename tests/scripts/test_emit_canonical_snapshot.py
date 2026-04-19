"""Tests for emit_canonical_snapshot.py

Pins three invariants:
1. schema_version field is present and == 2
2. All canonical keys are present in the output
3. roi_ann_pct matches compute_utilization() output (utilization backend consistency)
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

import scripts.emit_canonical_snapshot as mod
from scripts.emit_canonical_snapshot import emit_snapshot, SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_db(tmp_path):
    """Minimal DB sufficient for emit_snapshot to run without errors."""
    db = tmp_path / "canonical_test.db"
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE portfolio_cash_state (
            id INTEGER PRIMARY KEY,
            cash REAL,
            initial_capital REAL,
            updated_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE trade_executions (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            trade_date TEXT,
            action TEXT,
            shares REAL,
            price REAL,
            total_value REAL,
            realized_pnl REAL,
            holding_period_days REAL,
            is_close INTEGER DEFAULT 0,
            is_synthetic INTEGER DEFAULT 0,
            is_diagnostic INTEGER DEFAULT 0,
            entry_trade_id INTEGER,
            exit_reason TEXT
        )
    """)
    cur.execute("""
        CREATE VIEW production_closed_trades AS
        SELECT * FROM trade_executions
        WHERE is_close = 1
          AND COALESCE(is_synthetic, 0) = 0
          AND COALESCE(is_diagnostic, 0) = 0
    """)

    cur.execute("INSERT INTO portfolio_cash_state VALUES (1, 24000.0, 25000.0, '2026-01-01')")

    # Two round-trips
    cur.executemany(
        "INSERT INTO trade_executions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (1, "NVDA", "2026-01-10", "BUY",  4, 200.0, 800.0, None, None, 0, 0, 0, None, None),
            (2, "NVDA", "2026-01-12", "SELL", 4, 204.0, 816.0, 80.0, 2.0,  1, 0, 0, 1,    "TAKE_PROFIT"),
            (3, "AAPL", "2026-01-15", "BUY",  2, 250.0, 500.0, None, None, 0, 0, 0, None, None),
            (4, "AAPL", "2026-01-19", "SELL", 2, 230.0, 460.0,-40.0, 4.0,  1, 0, 0, 3,    "STOP_LOSS"),
        ],
    )
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCanonicalSnapshot:

    def test_schema_version_present_and_correct(self, minimal_db):
        """schema_version must be present and equal SCHEMA_VERSION (currently 2)."""
        snapshot = emit_snapshot(minimal_db)
        assert "schema_version" in snapshot, "schema_version field missing from snapshot"
        assert snapshot["schema_version"] == SCHEMA_VERSION == 2

    def test_all_canonical_keys_present(self, minimal_db):
        """All top-level canonical keys must be present."""
        snapshot = emit_snapshot(minimal_db)
        required = {"schema_version", "generated_utc", "closed_pnl", "capital",
                    "open_risk", "utilization", "summary", "source_contract"}
        missing = required - set(snapshot.keys())
        assert not missing, f"Missing canonical keys: {missing}"

    def test_roi_ann_matches_compute_utilization(self, minimal_db):
        """roi_ann_pct in snapshot must match compute_utilization() directly."""
        from scripts.compute_capital_utilization import compute_utilization
        util_direct = compute_utilization(minimal_db)

        snapshot = emit_snapshot(minimal_db)
        util_via_snapshot = snapshot.get("utilization", {})

        assert util_via_snapshot.get("roi_ann_pct") == pytest.approx(
            util_direct["roi_ann_pct"], rel=0.01
        ), (
            f"roi_ann_pct mismatch: snapshot={util_via_snapshot.get('roi_ann_pct')}, "
            f"direct={util_direct['roi_ann_pct']}"
        )

    def test_closed_pnl_matches_production_closed_trades(self, minimal_db):
        """closed_pnl section must read from production_closed_trades view."""
        snapshot = emit_snapshot(minimal_db)
        pnl = snapshot["closed_pnl"]
        assert pnl["n_trips"] == 2
        assert pnl["total_pnl"] == pytest.approx(80.0 - 40.0, abs=0.01)
        assert pnl["source"] == "production_closed_trades"

    def test_capital_from_portfolio_cash_state(self, minimal_db):
        """capital section must read from portfolio_cash_state.initial_capital."""
        snapshot = emit_snapshot(minimal_db)
        cap = snapshot["capital"]
        assert cap["capital"] == 25000.0
        assert cap["source"] == "portfolio_cash_state"

    def test_summary_gap_to_hurdle_computed(self, minimal_db):
        """summary.gap_to_hurdle_pp must be 28.0 - ann_roi_pct."""
        snapshot = emit_snapshot(minimal_db)
        s = snapshot["summary"]
        if s.get("ann_roi_pct") is not None:
            expected_gap = round(28.0 - s["ann_roi_pct"], 2)
            assert s["gap_to_hurdle_pp"] == pytest.approx(expected_gap, abs=0.01)
        assert s["ngn_hurdle_pct"] == 28.0

    def test_output_json_parseable(self, minimal_db, tmp_path):
        """Output JSON must round-trip without errors."""
        out_path = tmp_path / "canonical_snapshot_latest.json"
        snapshot = emit_snapshot(minimal_db)
        out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        reloaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert reloaded["schema_version"] == SCHEMA_VERSION

    def test_gate_artifact_prefers_audit_gate_path(self, minimal_db, tmp_path, monkeypatch):
        """The canonical snapshot must read the audit_gate artifact path first."""
        repo_root = tmp_path / "repo"
        audit_gate = repo_root / "logs" / "audit_gate"
        legacy_gate = repo_root / "logs"
        audit_gate.mkdir(parents=True, exist_ok=True)
        legacy_gate.mkdir(parents=True, exist_ok=True)
        (legacy_gate / "production_gate_latest.json").write_text(
            json.dumps({"posture": "LEGACY_ONLY"}),
            encoding="utf-8",
        )
        (audit_gate / "production_gate_latest.json").write_text(
            json.dumps({"posture": "AUDIT_GATE", "phase3_ready": True}),
            encoding="utf-8",
        )

        monkeypatch.setattr(mod, "ROOT", repo_root)
        monkeypatch.setattr(
            mod,
            "_run_utilization",
            lambda db_path, capital: {"roi_ann_pct": 9.86, "trades_per_day": 0.5, "deployment_pct": 1.83},
        )

        snapshot = mod.emit_snapshot(minimal_db)
        assert snapshot["gate"]["artifact_path"] == str(audit_gate / "production_gate_latest.json")
        assert snapshot["source_contract"]["canonical"]["gate_artifact"] == str(
            audit_gate / "production_gate_latest.json"
        )
        assert Path(snapshot["source_contract"]["ui_only"]["metrics_summary"]).as_posix().endswith(
            "visualizations/performance/metrics_summary.json"
        )
