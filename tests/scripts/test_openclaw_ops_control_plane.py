from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import openclaw_ops_control_plane as ops


def _base_snapshot() -> dict:
    return {
        "components": {
            "gateway": {
                "healthy": True,
                "rpc_ok": True,
                "primary_issue": "",
            },
            "primary_channel": {
                "healthy": True,
                "linked": True,
                "running": True,
                "connected": True,
                "relink_required": False,
                "issue_reason": "",
            },
            "dashboard": {
                "healthy": True,
                "url": "http://127.0.0.1:8000/visualizations/live_dashboard.html",
            },
            "watcher": {
                "healthy": True,
                "running": True,
            },
            "runtime": {
                "included": False,
                "openclaw_exec_env_ok": True,
            },
            "readiness": {
                "readiness_status": "FAIL",
                "ready_now": False,
                "overall_passed": False,
                "phase3_ready": True,
                "phase3_reason": "phase3 looks good",
                "skipped_gate_labels": ["run_all_gates"],
            },
            "maintenance": {
                "last_fast_supervisor_restart_at": "2026-03-14T16:00:00+00:00",
                "last_gateway_restart_at": "2026-03-14T16:05:00+00:00",
            },
        },
        "readiness_payload": {
            "blockers": [
                {
                    "source": "security",
                    "code": "prompt_injection_block_disabled",
                    "detail": "OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS is disabled.",
                },
                {
                    "source": "capital_readiness",
                    "code": "capital_readiness_failed",
                    "detail": "win rate too low; profit factor too low; lift negative",
                },
            ],
            "warnings": [],
        },
        "runtime_payload": None,
    }


def test_extract_ops_intent_request_maps_supported_commands() -> None:
    assert ops.extract_ops_intent_request("what is broken")["command"] == "status"
    assert ops.extract_ops_intent_request("show runtime")["include_runtime"] is True
    heal_gateway = ops.extract_ops_intent_request("heal gateway")
    assert heal_gateway["command"] == "recover"
    assert heal_gateway["targets"] == ["gateway"]
    heal_dashboard = ops.extract_ops_intent_request("heal dashboard")
    assert heal_dashboard["targets"] == ["dashboard"]
    heal_watcher = ops.extract_ops_intent_request("heal watcher")
    assert heal_watcher["targets"] == ["watcher"]


def test_normalize_requested_targets_rejects_invalid_target() -> None:
    with pytest.raises(ValueError, match="unsupported recovery targets"):
        ops._normalize_requested_targets("gateway,trades")


def test_classify_ops_issues_separates_governance_security_and_economics() -> None:
    snapshot = _base_snapshot()

    grouped = ops.classify_ops_issues(snapshot)

    governance = grouped["human_governance_action_required"]
    security = grouped["human_security_action_required"]
    economics = grouped["human_economic_action_required"]

    assert any(row["code"] == "overall_passed_false" for row in governance)
    assert any(row["code"] == "prompt_injection_block_disabled" for row in security)
    assert any(row["code"] == "capital_readiness_failed" for row in economics)

    overall = ops._build_overall_status(issue_classes=grouped, components=snapshot["components"])
    assert overall["status"] == "WARN"
    assert overall["service_health"] == "healthy"
    assert overall["production_status"] == "blocked"


def test_classify_ops_issues_marks_dashboard_and_watcher_as_recoverable() -> None:
    snapshot = _base_snapshot()
    snapshot["components"]["dashboard"]["healthy"] = False
    snapshot["components"]["watcher"]["healthy"] = False
    snapshot["components"]["watcher"]["running"] = False

    grouped = ops.classify_ops_issues(snapshot)
    service_rows = grouped["recoverable_service_failure"]

    assert any(row["target"] == "dashboard" for row in service_rows)
    assert any(row["target"] == "watcher" for row in service_rows)


def test_build_operator_summary_distinguishes_service_health_from_production_ready() -> None:
    snapshot = _base_snapshot()
    issue_classes = ops.classify_ops_issues(snapshot)
    overall = ops._build_overall_status(issue_classes=issue_classes, components=snapshot["components"])

    summary = ops.build_operator_summary(
        initial_issue_classes=issue_classes,
        final_issue_classes=issue_classes,
        attempted_actions=[],
        completed_actions=[],
        components=snapshot["components"],
        overall_status=overall,
    )

    assert summary["headline"] == "Services healthy; production is still blocked."
    assert "Service health=healthy production_status=blocked" in summary["message"]


def test_decide_notification_respects_cooldown_and_recovery_transition() -> None:
    snapshot = _base_snapshot()
    grouped = ops.classify_ops_issues(snapshot)
    summary = {"message": "OpenClaw ops: blocked"}
    future = datetime.now(timezone.utc) + timedelta(minutes=10)
    state = {
        "last_status": "issue",
        "last_issue_signature": ops._fingerprint_payload(
            [{"class": row["class"], "code": row["code"], "target": row["target"]} for row in ops._flatten_issue_classes(grouped)]
        ),
        "next_notify_at_utc": future.isoformat(),
    }

    suppressed = ops._decide_notification(
        final_issue_classes=grouped,
        operator_summary=summary,
        state=state,
        cooldown_seconds=900,
    )
    assert suppressed["should_notify"] is False
    assert suppressed["reason"] == "cooldown_active"
    assert suppressed["next_notify_at_utc"] == future.isoformat()

    recovered = ops._decide_notification(
        final_issue_classes={label: [] for label in ops.ISSUE_CLASSES},
        operator_summary={"message": "OpenClaw ops: recovered"},
        state={"last_status": "issue", "last_issue_signature": "abc"},
        cooldown_seconds=900,
    )
    assert recovered["should_notify"] is True
    assert recovered["reason"] == "recovered"


def test_run_ops_command_marks_gateway_recovery_completed(tmp_path, monkeypatch) -> None:
    first = _base_snapshot()
    first["components"]["gateway"]["healthy"] = False
    first["components"]["gateway"]["rpc_ok"] = False
    first["components"]["primary_channel"]["running"] = False
    second = _base_snapshot()

    snapshots = iter([first, second])
    monkeypatch.setattr(ops, "collect_ops_snapshot", lambda **kwargs: next(snapshots))
    monkeypatch.setattr(
        ops,
        "_run_recovery_targets",
        lambda **kwargs: [
            {
                "target": "gateway",
                "ok": True,
                "summary": "Attempted gateway maintenance recovery.",
                "returncode": 0,
                "command": ["python", "scripts/openclaw_maintenance.py"],
                "stdout_tail": [],
                "stderr_tail": [],
            }
        ],
    )
    monkeypatch.setattr(
        ops,
        "_send_notification",
        lambda **kwargs: {"attempted": False, "sent": False, "reason": "notifications_disabled", "results": []},
    )

    verdict = ops.run_ops_command(
        command_name="sweep",
        primary_channel="whatsapp",
        gate_artifact_path=tmp_path / "gate_status_latest.json",
        db_path=tmp_path / "portfolio_maximizer.db",
        dashboard_port=8000,
        include_runtime=False,
        timeout_seconds=5.0,
        apply_safe_recovery=True,
        explicit_targets=[],
        maintenance_report_path=tmp_path / "openclaw_maintenance_latest.json",
        maintenance_state_path=tmp_path / "openclaw_maintenance_state.json",
        watcher_json_path=tmp_path / "live_denominator_latest.json",
        watcher_tickers="AAPL,MSFT",
        watcher_cycles=30,
        watcher_sleep_seconds=86400,
        notify_on_change=False,
        notify_targets="",
        notify_to="",
        cooldown_seconds=900,
        verdict_path=tmp_path / "openclaw_ops_control_plane_latest.json",
        state_path=tmp_path / "openclaw_ops_control_plane_state.json",
    )

    assert verdict["completed_actions"][0]["target"] == "gateway"
    assert verdict["overall_status"]["service_health"] == "healthy"
    assert verdict["overall_status"]["status"] == "WARN"
    assert Path(tmp_path / "openclaw_ops_control_plane_latest.json").exists()


def test_collect_ops_snapshot_surfaces_gate_decomposition_components(tmp_path, monkeypatch) -> None:
    readiness_payload = {
        "readiness_status": "FAIL",
        "ready_now": False,
        "summary": {"blocker_count": 2, "warning_count": 0},
        "gate_artifact": {
            "overall_passed": False,
            "phase3_ready": False,
            "phase3_reason": "GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL",
            "skipped_gate_labels": [],
        },
        "gate_decomposition": {
            "component_status": [
                {"name": "PERFORMANCE_BLOCKER", "pass": False},
                {"name": "LINKAGE_BLOCKER", "pass": False},
                {"name": "HYGIENE_BLOCKER", "pass": True},
            ],
            "reason_breakdown": {"available": True},
            "refresh_result": {"refreshed": True, "reason": "source_timestamp_mismatch"},
        },
        "openclaw_exec_env": {"ok": True},
        "blockers": [],
        "warnings": [],
    }

    def fake_call_quietly(fn, /, *args, **kwargs):
        if fn is ops.readiness_mod.collect_openclaw_production_readiness:
            return readiness_payload, [], []
        return {"status": "ok", "checks": [], "failed_checks": [], "check_count": 0}, [], []

    monkeypatch.setattr(ops, "_call_quietly", fake_call_quietly)
    monkeypatch.setattr(
        ops,
        "_collect_dashboard_components",
        lambda **kwargs: {"dashboard": {"healthy": True, "url": "http://127.0.0.1:8000"}, "watcher": {"healthy": True, "running": True}},
    )
    monkeypatch.setattr(
        ops,
        "_collect_primary_channel_components",
        lambda **kwargs: {
            "gateway": {"healthy": True, "rpc_ok": True, "primary_issue": ""},
            "primary_channel": {"healthy": True, "linked": True, "running": True, "connected": True, "relink_required": False, "issue_reason": ""},
            "maintenance": {},
        },
    )

    snapshot = ops.collect_ops_snapshot(
        gate_artifact_path=tmp_path / "gate_status_latest.json",
        db_path=tmp_path / "portfolio_maximizer.db",
        primary_channel="whatsapp",
        dashboard_port=8000,
        include_runtime=False,
        timeout_seconds=5.0,
        maintenance_report_path=tmp_path / "openclaw_maintenance_latest.json",
        maintenance_state_path=tmp_path / "openclaw_maintenance_state.json",
        watcher_json_path=tmp_path / "live_denominator_latest.json",
    )

    readiness = snapshot["components"]["readiness"]
    assert readiness["blocker_components"] == [
        {"name": "PERFORMANCE_BLOCKER", "pass": False},
        {"name": "LINKAGE_BLOCKER", "pass": False},
        {"name": "HYGIENE_BLOCKER", "pass": True},
    ]
    assert readiness["reason_breakdown_available"] is True
    assert readiness["decomposition_refreshed"] is True
    assert readiness["decomposition_refresh_reason"] == "source_timestamp_mismatch"


def test_recover_gateway_uses_repo_python(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(cmd: list[str], *, timeout_seconds: float) -> dict[str, object]:
        captured["cmd"] = list(cmd)
        captured["timeout_seconds"] = timeout_seconds
        return {"ok": True, "returncode": 0, "command": list(cmd), "stdout_tail": [], "stderr_tail": []}

    monkeypatch.setattr(ops, "_repo_python_bin", lambda: r"C:\repo\simpleTrader_env\Scripts\python.exe")
    monkeypatch.setattr(ops, "_run_command", fake_run_command)

    result = ops._recover_gateway(
        primary_channel="whatsapp",
        timeout_seconds=30.0,
        maintenance_report_path=tmp_path / "openclaw_maintenance_latest.json",
    )

    assert result["ok"] is True
    assert captured["cmd"][0] == r"C:\repo\simpleTrader_env\Scripts\python.exe"


def test_recover_dashboard_like_passes_repo_python(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class Result:
        bridge_running = True
        server_running = True
        watcher_running = True
        started_bridge = False
        started_server = False
        started_watcher = False
        warnings: list[str] = []

    def fake_ensure_dashboard_stack(**kwargs):
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(ops, "_repo_python_bin", lambda: r"C:\repo\simpleTrader_env\Scripts\python.exe")
    monkeypatch.setattr(ops.dashboard_mod, "_ensure_dashboard_stack", fake_ensure_dashboard_stack)

    result = ops._recover_dashboard_like(
        ensure_live_watcher=True,
        dashboard_port=8000,
        db_path=tmp_path / "portfolio_maximizer.db",
        watcher_tickers="AAPL,MSFT",
        watcher_cycles=30,
        watcher_sleep_seconds=60,
    )

    assert result["ok"] is True
    assert captured["python_bin"] == r"C:\repo\simpleTrader_env\Scripts\python.exe"
