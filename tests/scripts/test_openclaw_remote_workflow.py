from __future__ import annotations

import json
from pathlib import Path

from scripts import openclaw_remote_workflow as workflow


def test_gateway_accepts_loopback_for_channel_driven_whatsapp_remote_dev() -> None:
    cfg = {
        "gateway": {
            "mode": "local",
            "bind": "loopback",
        }
    }
    payload = {
        "channels": {
            "whatsapp": {
                "configured": True,
                "linked": True,
                "running": True,
                "connected": True,
            }
        },
        "channelAccounts": {
            "whatsapp": [
                {
                    "accountId": "default",
                    "enabled": True,
                    "linked": True,
                    "running": True,
                    "connected": True,
                }
            ]
        },
        "channelDefaultAccountId": {"whatsapp": "default"},
    }

    original = workflow._gateway_local_ping
    workflow._gateway_local_ping = lambda: (True, "ok")
    try:
        result = workflow._check_gateway(cfg, primary_channel="whatsapp", channels_payload=payload)
    finally:
        workflow._gateway_local_ping = original

    assert result["status"] == "OK"
    assert result["access_mode"] == "channel-driven"
    assert "channel-driven via whatsapp" in result["detail"]


def test_check_channels_uses_live_runtime_state_not_static_config() -> None:
    cfg = {
        "channels": {
            "whatsapp": {"enabled": True},
            "telegram": {"enabled": True},
            "discord": {"enabled": True},
        }
    }
    payload = {
        "channels": {
            "whatsapp": {
                "configured": True,
                "linked": True,
                "running": True,
                "connected": True,
            },
            "telegram": {"configured": True, "running": True},
            "discord": {
                "configured": True,
                "running": False,
                "lastError": "Failed to resolve Discord application id",
            },
        },
        "channelAccounts": {
            "whatsapp": [
                {
                    "accountId": "default",
                    "enabled": True,
                    "linked": True,
                    "running": True,
                    "connected": True,
                }
            ],
            "telegram": [{"accountId": "default", "enabled": True, "running": True}],
            "discord": [{"accountId": "default", "enabled": True, "running": False, "lastError": "boom"}],
        },
        "channelDefaultAccountId": {
            "whatsapp": "default",
            "telegram": "default",
            "discord": "default",
        },
    }

    result = workflow._check_channels(cfg, primary_channel="whatsapp", channels_payload=payload)

    assert result["status"] == "OK"
    assert result["primary_status"] == "OK"
    assert result["channels"]["discord"]["status"] == "WARN"
    assert result["fallback_ready"] == ["telegram"]


def test_gateway_softens_probe_failure_when_recent_maintenance_shows_recovered_channel() -> None:
    cfg = {
        "gateway": {
            "mode": "local",
            "bind": "loopback",
        }
    }
    maintenance = {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "steps": {
            "fast_supervisor": {
                "action": "soft_timeout_skip",
                "reason": "channels_status_timeout_softened",
            },
            "gateway_health": {
                "rpc_ok": True,
                "service_status": "running",
                "primary_channel_issue_final": None,
                "warnings": [],
            },
            "channels_status_snapshot": {
                "channels": {
                    "whatsapp": {
                        "configured": True,
                        "linked": True,
                        "running": True,
                        "connected": True,
                        "lastError": None,
                    }
                }
            },
        },
    }

    original = workflow._gateway_local_ping
    workflow._gateway_local_ping = lambda: (False, "timeout")
    try:
        result = workflow._check_gateway(
            cfg,
            primary_channel="whatsapp",
            channels_payload=None,
            maintenance_payload=maintenance,
            maintenance_age_minutes=1.0,
        )
    finally:
        workflow._gateway_local_ping = original

    assert result["status"] == "WARN"
    assert "probe degraded" in result["issues"][0]
    assert result["recovery_context"]["mode"] == "channels_status_timeout_softened"


def test_check_channels_uses_recent_maintenance_snapshot_as_recovering_context() -> None:
    cfg = {
        "channels": {
            "whatsapp": {"enabled": True},
            "telegram": {"enabled": True},
            "discord": {"enabled": False},
        }
    }
    maintenance = {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "steps": {
            "fast_supervisor": {
                "action": "gateway_restart_triggered",
                "reason": "channels_status_call_failed:rc=124",
            },
            "gateway_health": {
                "rpc_ok": True,
                "service_status": "running",
                "primary_channel_issue_final": None,
                "warnings": [],
            },
            "channels_status_snapshot": {
                "channels": {
                    "whatsapp": {
                        "configured": True,
                        "linked": True,
                        "running": True,
                        "connected": True,
                        "lastError": None,
                    },
                    "telegram": {
                        "configured": True,
                        "running": True,
                        "lastError": None,
                    },
                }
            },
        },
    }

    result = workflow._check_channels(
        cfg,
        primary_channel="whatsapp",
        channels_payload=None,
        maintenance_payload=maintenance,
        maintenance_age_minutes=1.0,
    )

    assert result["status"] == "WARN"
    assert result["primary_status"] == "RECOVERING"
    assert result["channels"]["whatsapp"]["source"] == "maintenance_snapshot"
    assert result["fallback_ready"] == ["telegram"]


def test_channel_test_uses_inferred_whatsapp_target_and_send_helper() -> None:
    calls: list[tuple[str, str, str]] = []

    original_infer = workflow._infer_whatsapp_target
    original_send = workflow._send_openclaw_message
    workflow._infer_whatsapp_target = lambda cfg: "+2347000000000"
    workflow._send_openclaw_message = lambda *, channel, to, message: (calls.append((channel, to, message)) or (True, 0, "ok"))
    try:
        rc = workflow.cmd_channel_test(as_json=True)
    finally:
        workflow._infer_whatsapp_target = original_infer
        workflow._send_openclaw_message = original_send

    assert rc == 0
    assert calls
    assert calls[0][0] == "whatsapp"
    assert calls[0][1] == "+2347000000000"
    assert "[PMX WhatsApp Remote Dev Test]" in calls[0][2]


def test_bridge_test_runs_orchestrator_against_whatsapp_target() -> None:
    calls: list[tuple[str, str, str]] = []

    original_infer = workflow._infer_whatsapp_target
    original_bridge = workflow._run_bridge_test
    workflow._infer_whatsapp_target = lambda cfg: "+2347000000000"
    workflow._run_bridge_test = lambda *, channel, reply_to, message, timeout=120.0: (calls.append((channel, reply_to, message)) or (0, "ok", ""))
    try:
        rc = workflow.cmd_bridge_test(as_json=True)
    finally:
        workflow._infer_whatsapp_target = original_infer
        workflow._run_bridge_test = original_bridge

    assert rc == 0
    assert calls == [("whatsapp", "+2347000000000", "status")]


def test_cron_health_reports_missing_session_target_without_crashing(tmp_path: Path, monkeypatch) -> None:
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "bad-job",
                        "name": "[P2] Gate and Readiness Check",
                        "agentId": "trading",
                        "enabled": True,
                        "schedule": {"kind": "cron", "expr": "*/10 * * * *"},
                        "payload": {
                            "kind": "agentTurn",
                            "message": "do work",
                        },
                        "delivery": {"channel": "whatsapp"},
                        "state": {"consecutiveErrors": 0, "lastStatus": "pending"},
                    },
                    {
                        "id": "good-job",
                        "name": "[P1] Healthy Job",
                        "agentId": "ops",
                        "enabled": True,
                        "schedule": {"kind": "cron", "expr": "0 * * * *"},
                        "sessionTarget": "isolated",
                        "payload": {
                            "kind": "agentTurn",
                            "message": "ok",
                        },
                        "delivery": {"channel": "whatsapp", "fallback": {"channel": "telegram", "to": "+2347"}},
                        "state": {"consecutiveErrors": 0, "lastStatus": "success"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(workflow, "CRON_JOBS_PATH", jobs_path)

    rc = workflow.cmd_cron_health(as_json=True)
    assert rc == 2

    # Re-run through the summary helper to keep the assertion focused on product behavior.
    summary = workflow._check_cron_jobs()
    assert summary["status"] == "FAIL"
    assert summary["jobs_invalid"] == 1
    assert summary["invalid_session_target_count"] == 1
    assert summary["delivery_fallback_ready_count"] == 1


def test_cron_health_surfaces_warn_rows_in_validation_output(tmp_path: Path, monkeypatch, capsys) -> None:
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "warn-job",
                        "name": "[P2] Missing Payload",
                        "agentId": "trading",
                        "enabled": True,
                        "schedule": {"kind": "cron", "expr": "*/10 * * * *"},
                        "sessionTarget": "isolated",
                        "delivery": {"channel": "whatsapp"},
                        "state": {"consecutiveErrors": 0, "lastStatus": "pending"},
                    },
                    {
                        "id": "good-job",
                        "name": "[P1] Healthy Job",
                        "agentId": "ops",
                        "enabled": True,
                        "schedule": {"kind": "cron", "expr": "0 * * * *"},
                        "sessionTarget": "isolated",
                        "payload": {
                            "kind": "agentTurn",
                            "message": "ok",
                        },
                        "delivery": {"channel": "whatsapp", "fallback": {"channel": "telegram", "to": "+2347"}},
                        "state": {"consecutiveErrors": 0, "lastStatus": "success"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(workflow, "CRON_JOBS_PATH", jobs_path)

    rc = workflow.cmd_cron_health(as_json=True)
    stdout = capsys.readouterr().out
    result = json.loads(stdout)

    assert rc == 1
    assert result["jobs"][0]["validation_status"] == "WARN"
    assert "payload_missing" in result["jobs"][0]["validation_issues"]


def test_cron_health_rejects_non_string_session_target_without_hiding_it(tmp_path: Path, monkeypatch, capsys) -> None:
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "bad-job",
                        "name": "[P2] Gate and Readiness Check",
                        "agentId": "trading",
                        "enabled": True,
                        "schedule": {"kind": "cron", "expr": "*/10 * * * *"},
                        "sessionTarget": 123,
                        "payload": {
                            "kind": "agentTurn",
                            "message": "do work",
                        },
                        "delivery": {"channel": "whatsapp"},
                        "state": {"consecutiveErrors": 0, "lastStatus": "pending"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(workflow, "CRON_JOBS_PATH", jobs_path)

    rc = workflow.cmd_cron_health(as_json=True)
    stdout = capsys.readouterr().out
    result = json.loads(stdout)

    assert rc == 2
    assert result["jobs"][0]["validation_status"] == "FAIL"
    assert "sessionTarget_not_string" in result["jobs"][0]["validation_issues"]


def test_cron_health_surfaces_stale_python_path_count(tmp_path: Path, monkeypatch, capsys) -> None:
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "rewrite-job",
                        "name": "[P1] Rewrite Stale Python Path",
                        "agentId": "training",
                        "enabled": True,
                        "schedule": {"kind": "cron", "expr": "0 3 * * *"},
                        "sessionTarget": "isolated",
                        "payload": {
                            "kind": "agentTurn",
                            "message": ".\\simpleTrader_env\\Scripts\\python.exe scripts\\check_classifier_readiness.py --json",
                        },
                        "delivery": {"channel": "whatsapp", "fallback": {"channel": "telegram", "to": "+2347"}},
                        "state": {"consecutiveErrors": 0, "lastStatus": "pending"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(workflow, "CRON_JOBS_PATH", jobs_path)

    rc = workflow.cmd_cron_health(as_json=True)
    stdout = capsys.readouterr().out
    result = json.loads(stdout)

    assert rc == 0
    assert result["summary"]["stale_python_path_count"] == 1
