from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from scripts import prometheus_alert_exporter as mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_metrics_snapshot_reads_canonical_artifacts(tmp_path) -> None:
    dashboard = tmp_path / "dashboard_data.json"
    production_gate = tmp_path / "production_gate_latest.json"
    audit_db = tmp_path / "dashboard_audit.db"
    now = datetime(2026, 4, 2, 6, 0, 0, tzinfo=timezone.utc)

    _write_json(
        dashboard,
        {
            "meta": {
                "ts": "2026-04-02T05:59:30Z",
                "data_origin": "mixed",
            },
            "trade_events": [],
            "signals": [{"ticker": "AAPL"}],
            "price_series": {"AAPL": [{"date": "2026-04-02", "close": 100.0}]},
            "robustness": {"status": "WARN"},
            "alerts": ["Artifact binding failed."],
        },
    )
    _write_json(
        production_gate,
        {
            "timestamp_utc": "2026-04-02T05:58:00Z",
            "phase3_ready": False,
            "artifact_binding": {"pass": False},
            "production_profitability_gate": {"status": "FAIL"},
            "profitability_proof": {
                "evidence_progress": {
                    "remaining_closed_trades": 12,
                    "remaining_trading_days": 6,
                }
            },
        },
    )

    conn = sqlite3.connect(audit_db)
    try:
        conn.execute(
            """
            CREATE TABLE dashboard_snapshots(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                run_id TEXT,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO dashboard_snapshots(created_at, run_id, payload_json) VALUES (?,?,?)",
            ("2026-04-02T05:59:00Z", "RID-1", "{}"),
        )
        conn.commit()
    finally:
        conn.close()

    snapshot = mod.collect_metrics_snapshot(
        dashboard_json=dashboard,
        production_gate_json=production_gate,
        audit_db=audit_db,
        now=now,
    )

    assert snapshot["dashboard"]["present"] is True
    assert snapshot["dashboard"]["signals_total"] == 1
    assert snapshot["dashboard"]["trade_events_total"] == 0
    assert snapshot["dashboard"]["origin"] == "mixed"
    assert snapshot["gate"]["status"] == "FAIL"
    assert snapshot["gate"]["artifact_binding_pass"] == 0
    assert snapshot["gate"]["remaining_closed_trades"] == 12
    assert snapshot["audit"]["snapshot_count"] == 1
    assert snapshot["alerts"]["production_gate_fail"] == 1
    assert snapshot["alerts"]["data_origin_not_live"] == 1


def test_render_metrics_emits_expected_alert_gauges(tmp_path) -> None:
    dashboard = tmp_path / "missing_dashboard.json"
    production_gate = tmp_path / "missing_gate.json"
    audit_db = tmp_path / "missing_audit.db"
    snapshot = mod.collect_metrics_snapshot(
        dashboard_json=dashboard,
        production_gate_json=production_gate,
        audit_db=audit_db,
        now=datetime(2026, 4, 2, 6, 0, 0, tzinfo=timezone.utc),
    )

    text = mod.render_metrics(snapshot)

    assert "pmx_dashboard_payload_present 0" in text
    assert "pmx_production_gate_artifact_present 0" in text
    assert "pmx_alert_dashboard_payload_missing 1" in text
    assert "pmx_alert_production_gate_missing 1" in text
    assert "pmx_alert_production_gate_stale 0" in text
    assert "pmx_alert_dashboard_audit_error 0" in text
    assert 'pmx_dashboard_origin_state{state="unknown"} 1' not in text
    assert 'pmx_dashboard_origin_state{state="live"}' not in text
    assert 'pmx_dashboard_origin_state{origin="unknown"} 1' in text


def test_collect_metrics_snapshot_flags_stale_gate_using_bridge_policy(tmp_path, monkeypatch) -> None:
    dashboard = tmp_path / "dashboard_data.json"
    production_gate = tmp_path / "production_gate_latest.json"
    now = datetime(2026, 4, 2, 6, 0, 0, tzinfo=timezone.utc)

    _write_json(
        dashboard,
        {
            "meta": {"ts": "2026-04-02T05:59:30Z", "data_origin": "live"},
            "trade_events": [],
            "signals": [],
            "price_series": {},
        },
    )
    _write_json(
        production_gate,
        {
            "timestamp_utc": "2026-04-02T05:40:00Z",
            "production_profitability_gate": {"status": "PASS"},
        },
    )
    monkeypatch.setenv("PMX_PRODUCTION_GATE_MAX_AGE_MINUTES", "5")

    snapshot = mod.collect_metrics_snapshot(
        dashboard_json=dashboard,
        production_gate_json=production_gate,
        audit_db=tmp_path / "missing_audit.db",
        now=now,
    )

    assert snapshot["gate"]["status"] == "PASS"
    assert snapshot["alerts"]["production_gate_fail"] == 0
    assert snapshot["alerts"]["production_gate_stale"] == 1
    assert snapshot["policies"]["production_gate_stale_seconds"] == 300.0


def test_collect_metrics_snapshot_flags_audit_db_error(tmp_path) -> None:
    dashboard = tmp_path / "dashboard_data.json"
    audit_db = tmp_path / "dashboard_audit.db"
    _write_json(
        dashboard,
        {
            "meta": {"ts": "2026-04-02T05:59:30Z", "data_origin": "live"},
            "trade_events": [],
            "signals": [],
            "price_series": {},
        },
    )
    audit_db.write_text("not a sqlite database", encoding="utf-8")

    snapshot = mod.collect_metrics_snapshot(
        dashboard_json=dashboard,
        production_gate_json=tmp_path / "missing_gate.json",
        audit_db=audit_db,
        now=datetime(2026, 4, 2, 6, 0, 0, tzinfo=timezone.utc),
    )

    assert snapshot["audit"]["present"] is True
    assert snapshot["audit"]["status"] == "ERROR"
    assert snapshot["alerts"]["dashboard_audit_missing"] == 0
    assert snapshot["alerts"]["dashboard_audit_error"] == 1
