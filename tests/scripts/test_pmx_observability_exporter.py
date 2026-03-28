from __future__ import annotations

import json
import threading
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts import pmx_observability_exporter as mod


FIXED_NOW = datetime(2026, 3, 28, 8, 30, tzinfo=timezone.utc)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _runner_ok(cmd: list[str]):
    joined = " ".join(cmd)
    if "openclaw_remote_workflow.py" in joined:
        payload = {
            "timestamp_utc": "2026-03-28T08:29:30Z",
            "overall": "OK",
            "gateway_reachable": True,
            "primary_channel": "whatsapp",
            "primary_status": "OK",
            "channels_status_elapsed_ms": 4321,
            "recovery_mode": "channels_status_timeout_softened",
        }
        return 0, payload, json.dumps(payload), ""
    if "check_model_improvement.py" in joined:
        payload = {
            "timestamp_utc": "2026-03-28T08:25:00Z",
            "results": [
                {"layer": 1, "name": "Forecast Quality", "status": "FAIL"},
                {"layer": 2, "name": "Gate Status", "status": "PASS"},
            ],
        }
        return 1, payload, "[WARNING] noisy prelude\n" + json.dumps(payload), ""
    raise AssertionError(f"unexpected command: {joined}")


def test_exporter_builds_required_metrics_from_existing_artifacts(tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard_data.json"
    metrics_summary = tmp_path / "metrics_summary.json"
    production_gate = tmp_path / "production_gate_latest.json"
    maintenance = tmp_path / "openclaw_maintenance_latest.json"
    cron_jobs = tmp_path / "jobs.json"
    state_path = tmp_path / "logs" / "exporter_state.json"

    _write_json(
        dashboard,
        {"meta": {"generated_utc": "2026-03-28T08:29:00Z", "ts": "2026-03-28T08:29:00Z"}},
    )
    _write_json(metrics_summary, {"generated_utc": "2026-03-28T08:28:00Z", "status": "WARN"})
    _write_json(
        production_gate,
        {
            "timestamp_utc": "2026-03-28T08:27:00Z",
            "phase3_ready": True,
            "phase3_reason": "READY",
            "profitability_proof": {
                "evidence_progress": {
                    "closed_trades": 40,
                    "remaining_trading_days": 11,
                }
            },
        },
    )
    _write_json(
        maintenance,
        {
            "timestamp_utc": "2026-03-28T08:26:00Z",
            "steps": {
                "fast_supervisor": {
                    "action": "soft_timeout_skip",
                    "reason": "channels_status_timeout_softened",
                },
                "gateway_health": {
                    "warnings": ["gateway_detached_listener_conflict"],
                    "primary_channel_issue_final": None,
                },
                "channels_status_snapshot": {
                    "channels": {
                        "whatsapp": {
                            "reconnectAttempts": 1,
                        }
                    }
                },
            },
        },
    )
    _write_json(
        cron_jobs,
        {
            "jobs": [
                {
                    "id": "job-1",
                    "name": "[P0] Production Gate Check",
                    "enabled": True,
                    "schedule": {"kind": "cron", "expr": "0 7 * * *"},
                    "state": {
                        "lastStatus": "success",
                        "lastRunAtMs": int(datetime(2026, 3, 28, 7, 0, tzinfo=timezone.utc).timestamp() * 1000),
                        "consecutiveErrors": 0,
                    },
                }
            ]
        },
    )

    exporter = mod.ObservabilityExporter(
        dashboard_path=dashboard,
        metrics_summary_path=metrics_summary,
        production_gate_path=production_gate,
        maintenance_path=maintenance,
        cron_jobs_path=cron_jobs,
        db_path=tmp_path / "pmx.db",
        state_path=state_path,
        command_runner=_runner_ok,
        sqlite_checker=lambda _: (True, None),
        now_provider=lambda: FIXED_NOW,
    )
    exporter.refresh(force=True)

    text = exporter.get_metrics_text()
    assert "pmx_openclaw_gateway_up" in text
    assert "pmx_openclaw_primary_channel_up" in text
    assert "pmx_cron_job_last_success_unixtime" in text
    assert "pmx_dashboard_snapshot_age_seconds" in text
    assert "pmx_production_gate_pass 1" in text
    assert "pmx_proof_runway_closed_trades 40" in text
    assert "pmx_sqlite_health_ok{component=\"sqlite\"} 1" in text
    assert "pmx_openclaw_recovery_events_total{component=\"openclaw\"} 3" in text

    health = exporter.get_health_payload()
    assert health["status"] == "ok"
    assert health["warning_count"] == 0


def test_exporter_fails_soft_when_noncritical_inputs_are_missing_or_corrupt(tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard_data.json"
    maintenance = tmp_path / "openclaw_maintenance_latest.json"
    cron_jobs = tmp_path / "jobs.json"

    _write_json(
        dashboard,
        {"meta": {"ts": "2026-03-28T08:29:00Z"}},
    )
    maintenance.write_text("{not valid json", encoding="utf-8")
    _write_json(cron_jobs, {"jobs": []})

    exporter = mod.ObservabilityExporter(
        dashboard_path=dashboard,
        metrics_summary_path=tmp_path / "missing_metrics_summary.json",
        production_gate_path=tmp_path / "missing_production_gate.json",
        maintenance_path=maintenance,
        cron_jobs_path=cron_jobs,
        db_path=tmp_path / "missing.db",
        state_path=tmp_path / "logs" / "exporter_state.json",
        command_runner=lambda cmd: (0, {"timestamp_utc": "2026-03-28T08:29:00Z", "gateway_reachable": True, "primary_channel": "whatsapp", "primary_status": "OK", "channels_status_elapsed_ms": 1111}, "{}", ""),
        sqlite_checker=lambda _: (False, "missing:pmx.db"),
        now_provider=lambda: FIXED_NOW,
    )
    exporter.refresh(force=True)

    health = exporter.get_health_payload()
    assert health["status"] == "ok"
    assert health["warning_count"] >= 1
    assert any("missing" in item or "corrupt" in item for item in health["warnings"])
    assert "pmx_dashboard_snapshot_expected_refresh_seconds" in exporter.get_metrics_text()


def test_exporter_persists_last_successful_cron_timestamp(tmp_path: Path) -> None:
    cron_jobs = tmp_path / "jobs.json"
    state_path = tmp_path / "logs" / "exporter_state.json"
    success_ms = int(datetime(2026, 3, 28, 7, 0, tzinfo=timezone.utc).timestamp() * 1000)

    _write_json(
        cron_jobs,
        {
            "jobs": [
                {
                    "id": "job-1",
                    "name": "[P0] Production Gate Check",
                    "enabled": True,
                    "schedule": {"kind": "cron", "expr": "0 7 * * *"},
                    "state": {"lastStatus": "success", "lastRunAtMs": success_ms, "consecutiveErrors": 0},
                }
            ]
        },
    )
    exporter = mod.ObservabilityExporter(
        dashboard_path=tmp_path / "d1.json",
        metrics_summary_path=tmp_path / "m1.json",
        production_gate_path=tmp_path / "g1.json",
        maintenance_path=tmp_path / "o1.json",
        cron_jobs_path=cron_jobs,
        db_path=tmp_path / "pmx.db",
        state_path=state_path,
        command_runner=lambda cmd: (0, {"timestamp_utc": "2026-03-28T08:29:00Z", "gateway_reachable": True, "primary_channel": "whatsapp", "primary_status": "OK", "channels_status_elapsed_ms": 1111}, "{}", ""),
        sqlite_checker=lambda _: (True, None),
        now_provider=lambda: FIXED_NOW,
    )
    exporter.refresh(force=True)

    _write_json(
        cron_jobs,
        {
            "jobs": [
                {
                    "id": "job-1",
                    "name": "[P0] Production Gate Check",
                    "enabled": True,
                    "schedule": {"kind": "cron", "expr": "0 7 * * *"},
                    "state": {"lastStatus": "error", "lastRunAtMs": success_ms + 5000, "consecutiveErrors": 2},
                }
            ]
        },
    )
    second = mod.ObservabilityExporter(
        dashboard_path=tmp_path / "d2.json",
        metrics_summary_path=tmp_path / "m2.json",
        production_gate_path=tmp_path / "g2.json",
        maintenance_path=tmp_path / "o2.json",
        cron_jobs_path=cron_jobs,
        db_path=tmp_path / "pmx.db",
        state_path=state_path,
        command_runner=lambda cmd: (0, {"timestamp_utc": "2026-03-28T08:29:00Z", "gateway_reachable": True, "primary_channel": "whatsapp", "primary_status": "OK", "channels_status_elapsed_ms": 1111}, "{}", ""),
        sqlite_checker=lambda _: (True, None),
        now_provider=lambda: FIXED_NOW,
    )
    second.refresh(force=True)
    assert int(second._persisted_state["cron_last_success_ms"]["job-1"]) == success_ms


def test_http_endpoints_return_metrics_and_green_health_for_stale_noncritical_inputs(tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard_data.json"
    _write_json(dashboard, {"meta": {"generated_utc": "2026-03-28T08:29:00Z"}})
    exporter = mod.ObservabilityExporter(
        dashboard_path=dashboard,
        metrics_summary_path=tmp_path / "missing.json",
        production_gate_path=tmp_path / "missing_gate.json",
        maintenance_path=tmp_path / "missing_maintenance.json",
        cron_jobs_path=tmp_path / "missing_jobs.json",
        db_path=tmp_path / "missing.db",
        state_path=tmp_path / "logs" / "exporter_state.json",
        command_runner=lambda cmd: (0, {"timestamp_utc": "2026-03-28T08:29:00Z", "gateway_reachable": True, "primary_channel": "whatsapp", "primary_status": "OK", "channels_status_elapsed_ms": 1111}, "{}", ""),
        sqlite_checker=lambda _: (False, "missing:db"),
        now_provider=lambda: FIXED_NOW,
    )
    exporter.refresh(force=True)

    mod._ExporterHandler.exporter = exporter
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod._ExporterHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/healthz") as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["status"] == "ok"
        with urllib.request.urlopen(base + "/metrics") as resp:
            assert resp.status == 200
            text = resp.read().decode("utf-8")
            assert "pmx_dashboard_snapshot_expected_refresh_seconds" in text
    finally:
        server.shutdown()
        server.server_close()
