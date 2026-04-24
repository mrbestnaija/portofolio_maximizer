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


def _write_required_jobs_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "jobs:",
                "  production_gate_check:",
                "    name: \"[P0] Production Gate Check\"",
                "    required_for_green: true",
                "    severity: P0",
                "    expected_cadence_seconds: 86400",
                "  signal_linkage_monitor:",
                "    name: \"[P1] Signal Linkage Monitor\"",
                "    required_for_green: false",
                "    severity: P1",
                "    expected_cadence_seconds: 86400",
            ]
        ),
        encoding="utf-8",
    )


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
            "fallback_ready": ["telegram"],
            "fallback_ready_count": 1,
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
        {
            "meta": {"generated_utc": "2026-03-28T08:29:00Z", "ts": "2026-03-28T08:29:00Z"},
            "pnl": {"absolute": 123.45, "pct": 0.0123},
            "win_rate": 0.6,
            "trade_count": 25,
            "positions": {
                "MSFT": {"shares": 2, "entry_price": 100.0},
                "NVDA": {"shares": -1, "entry_price": 250.0},
            },
            "latency": {"ts_ms": 1875.0, "llm_ms": 2450.0},
            "forecaster_health": {
                "thresholds": {
                    "profit_factor_min": 1.1,
                    "win_rate_min": 0.52,
                    "rmse_ratio_max": 1.1,
                },
                "metrics": {
                    "profit_factor": 1.4,
                    "win_rate": 0.58,
                    "rmse": {"ensemble": 48.0, "baseline": 52.0, "ratio": 0.9231},
                },
                "status": {
                    "profit_factor_ok": True,
                    "win_rate_ok": True,
                    "rmse_ok": True,
                },
            },
            "quant_validation_health": {
                "total": 100.0,
                "pass_count": 94.0,
                "fail_count": 6.0,
                "fail_fraction": 0.06,
                "negative_expected_profit_fraction": 0.02,
                "max_fail_fraction": 0.85,
                "max_negative_expected_profit_fraction": 0.5,
            },
            "routing": {"ts_signals": 12, "llm_signals": 3, "fallback_used": 1},
            "quality": {"average": 0.98, "minimum": 0.95},
            "equity": [{"t": "start", "v": 25000.0}, {"t": "end", "v": 25123.45}],
            "equity_realized": [{"t": "start", "v": 25000.0}, {"t": "end", "v": 25088.9}],
            "signals": [
                {"ticker": "MSFT", "signal_confidence": 0.55, "expected_return": 0.01, "shares": 2},
                {
                    "ticker": "NVDA",
                    "signal_confidence": 0.9,
                    "effective_confidence": 0.9,
                    "expected_return": -0.05,
                    "shares": 1,
                    "mid_slippage_bp": -25.992073132190733,
                },
            ],
            "trade_events": [
                {"kind": "open", "shares": 2},
                {"kind": "close", "shares": 1, "slippage": 0.0015},
                {"kind": "close", "shares": 1, "slippage": 0.002599207313219073},
            ],
            "operator_console": {
                "status": "WARN",
                "maintenance": {
                    "reconnect_attempts": 2,
                    "recovery_mode": "gateway_detached_listener_conflict",
                },
                "activity": {"short_circuit_events": 4, "tool_calls": 11},
                "issues": [{"kind": "latency"}, {"kind": "proof_runway"}],
            },
        },
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
                    "payload": {"kind": "system"},
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
    assert "pmx_dashboard_pnl_absolute 123.45" in text
    assert "pmx_dashboard_open_positions_count 2" in text
    assert "pmx_dashboard_long_positions_count 1" in text
    assert "pmx_dashboard_short_positions_count 1" in text
    assert "pmx_dashboard_position_gross_notional 450" in text
    assert "pmx_dashboard_position_net_notional -50" in text
    assert "pmx_dashboard_latency_ts_ms 1875" in text
    assert "pmx_dashboard_equity_last 25123.45" in text
    assert "pmx_dashboard_equity_realized_last 25088.9" in text
    assert "pmx_dashboard_signal_count 2" in text
    assert "pmx_dashboard_trade_event_count 3" in text
    assert "pmx_dashboard_latest_signal_confidence 0.9" in text
    assert "pmx_dashboard_latest_signal_expected_return -0.05" in text
    assert "pmx_dashboard_latest_signal_shares 1" in text
    assert "pmx_dashboard_latest_signal_mid_slippage_bp -25.992073" in text
    assert "pmx_dashboard_latest_trade_shares 1" in text
    assert "pmx_dashboard_latest_trade_slippage_bp 25.992073" in text
    assert "pmx_forecaster_rmse_ratio 0.9231" in text
    assert "pmx_forecaster_failed_checks 0" in text
    assert "pmx_quant_validation_fail_fraction 0.06" in text
    assert "pmx_operator_console_status_code 1" in text
    assert "pmx_operator_short_circuit_events 4" in text
    assert "pmx_operator_tool_calls_recent 11" in text
    assert "pmx_operator_issue_count 2" in text
    assert "pmx_operator_reconnect_attempts 2" in text
    assert "pmx_operator_recovery_mode_code 4" in text
    assert "pmx_production_gate_pass 1" in text
    assert "pmx_proof_runway_closed_trades 40" in text
    assert "pmx_sqlite_health_ok{component=\"sqlite\"} 1" in text
    assert "pmx_openclaw_recovery_events_total{component=\"openclaw\"} 3" in text

    health = exporter.get_health_payload()
    assert health["status"] == "ok"
    assert health["warning_count"] == 0


def test_exporter_surfaces_malformed_cron_jobs_and_fallback_readiness(tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard_data.json"
    metrics_summary = tmp_path / "metrics_summary.json"
    production_gate = tmp_path / "production_gate_latest.json"
    maintenance = tmp_path / "openclaw_maintenance_latest.json"
    cron_jobs = tmp_path / "jobs.json"
    state_path = tmp_path / "logs" / "exporter_state.json"

    _write_json(dashboard, {"meta": {"generated_utc": "2026-03-28T08:29:00Z"}})
    _write_json(metrics_summary, {"generated_utc": "2026-03-28T08:28:00Z", "status": "OK"})
    _write_json(
        production_gate,
        {
            "timestamp_utc": "2026-03-28T08:27:00Z",
            "phase3_ready": True,
            "phase3_reason": "READY",
            "profitability_proof": {"evidence_progress": {"closed_trades": 40, "remaining_trading_days": 11}},
        },
    )
    _write_json(
        maintenance,
        {
            "timestamp_utc": "2026-03-28T08:26:00Z",
            "steps": {
                "fast_supervisor": {"action": "soft_timeout_skip", "reason": "channels_status_timeout_softened"},
                "gateway_health": {"rpc_ok": True, "service_status": "running", "warnings": []},
            },
        },
    )
    _write_json(
        cron_jobs,
        {
            "jobs": [
                {
                    "id": "bad-job",
                    "name": "[P2] Gate and Readiness Check",
                    "agentId": "trading",
                    "enabled": True,
                    "schedule": {"kind": "cron", "expr": "*/10 * * * *"},
                    "payload": {"kind": "agentTurn", "message": "do work"},
                    "delivery": {"channel": "whatsapp"},
                    "state": {"lastStatus": "pending", "consecutiveErrors": 0},
                },
                {
                    "id": "good-job",
                    "name": "[P1] Healthy Job",
                    "agentId": "ops",
                    "enabled": True,
                    "schedule": {"kind": "cron", "expr": "0 * * * *"},
                    "sessionTarget": "isolated",
                    "payload": {"kind": "agentTurn", "message": "ok"},
                    "delivery": {"channel": "whatsapp", "fallback": {"channel": "telegram", "to": "telegram:6515478488"}},
                    "state": {"lastStatus": "success", "consecutiveErrors": 0},
                },
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
    assert 'pmx_openclaw_fallback_ready_count{component="openclaw"} 1' in text
    assert 'pmx_openclaw_cron_jobs_total{component="cron"} 2' in text
    assert 'pmx_openclaw_cron_invalid_session_target_total{component="cron"} 1' in text
    assert 'pmx_openclaw_cron_malformed_jobs_total{component="cron"} 1' in text
    assert 'pmx_openclaw_cron_delivery_fallback_ready_count{component="cron"} 1' in text

    health = exporter.get_health_payload()
    assert any("cron_invalid_session_target_count:1" in warning for warning in health["warnings"])


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
                    "payload": {"kind": "system"},
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
                    "payload": {"kind": "system"},
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


def test_exporter_filters_cron_metrics_to_required_job_inventory(tmp_path: Path) -> None:
    cron_jobs = tmp_path / "jobs.json"
    required_jobs = tmp_path / "observability_required_jobs.yml"
    _write_required_jobs_config(required_jobs)
    _write_json(
        cron_jobs,
        {
            "jobs": [
                {
                    "id": "prod-gate",
                    "name": "[P0] Production Gate Check",
                    "enabled": True,
                    "schedule": {"kind": "cron", "expr": "0 7 * * *"},
                    "payload": {"kind": "system"},
                    "state": {"lastStatus": "success", "lastRunAtMs": int(FIXED_NOW.timestamp() * 1000)},
                },
                {
                    "id": "linkage",
                    "name": "[P1] Signal Linkage Monitor",
                    "enabled": True,
                    "schedule": {"kind": "cron", "expr": "0 8 * * *"},
                    "payload": {"kind": "system"},
                    "state": {"lastStatus": "success", "lastRunAtMs": int(FIXED_NOW.timestamp() * 1000)},
                },
                {
                    "id": "unknown",
                    "name": "[P1] Something Else",
                    "enabled": True,
                    "schedule": {"kind": "cron", "expr": "0 9 * * *"},
                    "payload": {"kind": "system"},
                    "state": {"lastStatus": "success", "lastRunAtMs": int(FIXED_NOW.timestamp() * 1000)},
                },
            ]
        },
    )

    exporter = mod.ObservabilityExporter(
        dashboard_path=tmp_path / "dashboard.json",
        metrics_summary_path=tmp_path / "metrics.json",
        production_gate_path=tmp_path / "gate.json",
        maintenance_path=tmp_path / "maintenance.json",
        cron_jobs_path=cron_jobs,
        required_cron_jobs_path=required_jobs,
        db_path=tmp_path / "pmx.db",
        state_path=tmp_path / "logs" / "exporter_state.json",
        command_runner=lambda cmd: (0, {"timestamp_utc": "2026-03-28T08:29:00Z", "gateway_reachable": True, "primary_channel": "whatsapp", "primary_status": "OK", "channels_status_elapsed_ms": 1111}, "{}", ""),
        sqlite_checker=lambda _: (True, None),
        now_provider=lambda: FIXED_NOW,
    )

    cron_payload, warnings = exporter._collect_cron_jobs(now=FIXED_NOW)
    assert warnings == []
    jobs = cron_payload["jobs"]
    assert [row["job"] for row in jobs] == ["p0_production_gate_check"]
    assert jobs[0]["enabled"] == 1.0
    assert jobs[0]["expected_interval_seconds"] == 86400.0


def test_exporter_uses_strict_phase3_readiness_for_gate_metric(tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard_data.json"
    production_gate = tmp_path / "production_gate_latest.json"
    _write_json(dashboard, {"meta": {"generated_utc": "2026-03-28T08:29:00Z"}})
    _write_json(
        production_gate,
        {
            "timestamp_utc": "2026-03-28T08:27:00Z",
            "phase3_ready": True,
            "phase3_reason": "READY",
            "phase3_strict_ready": False,
            "phase3_strict_reason": "READY,GATE_SEMANTICS_INCONCLUSIVE_ALLOWED",
            "production_profitability_gate": {
                "status": "PASS",
                "pass": True,
                "strict_pass": False,
                "gate_semantics_status": "INCONCLUSIVE_ALLOWED",
            },
            "profitability_proof": {
                "evidence_progress": {
                    "closed_trades": 40,
                    "remaining_trading_days": 11,
                }
            },
        },
    )

    exporter = mod.ObservabilityExporter(
        dashboard_path=dashboard,
        metrics_summary_path=tmp_path / "metrics.json",
        production_gate_path=production_gate,
        maintenance_path=tmp_path / "maintenance.json",
        cron_jobs_path=tmp_path / "jobs.json",
        db_path=tmp_path / "pmx.db",
        state_path=tmp_path / "logs" / "exporter_state.json",
        command_runner=lambda cmd: (0, {"timestamp_utc": "2026-03-28T08:29:00Z", "gateway_reachable": True, "primary_channel": "whatsapp", "primary_status": "OK", "channels_status_elapsed_ms": 1111}, "{}", ""),
        sqlite_checker=lambda _: (True, None),
        now_provider=lambda: FIXED_NOW,
    )

    exporter.refresh(force=True)

    text = exporter.get_metrics_text()
    assert "pmx_production_gate_pass 0" in text
    assert 'pmx_production_gate_status_code 3' in text


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
