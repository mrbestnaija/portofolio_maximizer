from __future__ import annotations

import json

from scripts import openclaw_maintenance as om


def _dns_down_channels_payload() -> dict:
    return {
        "channels": {
            "whatsapp": {
                "configured": True,
                "linked": True,
                "running": False,
                "connected": False,
                "lastError": '{"error":{"data":{"code":"ENOTFOUND","hostname":"web.whatsapp.com"}}}',
                "lastDisconnect": {
                    "status": 428,
                    "error": "status=428 Precondition Required Connection Terminated",
                    "loggedOut": False,
                },
            }
        },
        "channelAccounts": {
            "whatsapp": [
                {
                    "accountId": "default",
                    "enabled": True,
                    "configured": True,
                    "linked": True,
                    "running": False,
                    "connected": False,
                    "lastError": '{"error":{"data":{"code":"ENOTFOUND","hostname":"web.whatsapp.com"}}}',
                    "lastDisconnect": {
                        "status": 428,
                        "error": "status=428 Precondition Required Connection Terminated",
                        "loggedOut": False,
                    },
                }
            ]
        },
        "channelDefaultAccountId": {"whatsapp": "default"},
    }


def _handshake_timeout_channels_payload() -> dict:
    return {
        "channels": {
            "whatsapp": {
                "configured": True,
                "linked": True,
                "running": False,
                "connected": False,
                "lastError": '{"error":{"output":{"statusCode":408,"payload":{"message":"WebSocket Error (Opening handshake has timed out)"}}}}',
            }
        },
        "channelAccounts": {
            "whatsapp": [
                {
                    "accountId": "default",
                    "enabled": True,
                    "configured": True,
                    "linked": True,
                    "running": False,
                    "connected": False,
                    "lastError": '{"error":{"output":{"statusCode":408,"payload":{"message":"WebSocket Error (Opening handshake has timed out)"}}}}',
                }
            ]
        },
        "channelDefaultAccountId": {"whatsapp": "default"},
    }


def _healthy_channels_payload() -> dict:
    return {
        "channels": {
            "whatsapp": {
                "configured": True,
                "linked": True,
                "running": True,
                "connected": True,
                "lastError": "",
            }
        },
        "channelAccounts": {
            "whatsapp": [
                {
                    "accountId": "default",
                    "enabled": True,
                    "configured": True,
                    "linked": True,
                    "running": True,
                    "connected": True,
                    "lastError": "",
                }
            ]
        },
        "channelDefaultAccountId": {"whatsapp": "default"},
    }


def test_derive_status_marks_primary_unresolved_as_fail() -> None:
    assert om._derive_status([]) == "PASS"
    assert om._derive_status(["gateway_restart_failed"]) == "WARN"
    assert om._derive_status(["primary_channel_unresolved:whatsapp_dns_resolution_failed"]) == "FAIL"


def test_normalize_openclaw_command_resets_self_reference() -> None:
    base, warning = om._normalize_openclaw_command(
        [
            "python",
            "C:/repo/scripts/openclaw_maintenance.py",
            "--watch",
            "--watch-interval",
            "30",
        ]
    )
    assert warning == "openclaw_command_self_reference_reset"
    assert any(str(x).lower() == "openclaw" for x in base)


def test_gateway_health_dns_issue_skips_restart_and_reenable(monkeypatch) -> None:
    calls: list[list[str]] = []
    payload = _dns_down_channels_payload()

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args == ["gateway", "status"]:
            gateway_payload = {
                "rpc": {"ok": True},
                "service": {"runtime": {"status": "running", "state": "Running"}},
            }
            return om._CmdResult(True, 0, ["openclaw", "gateway", "status"], "", ""), gateway_payload
        raise AssertionError(f"Unexpected json call args: {args}")

    def fake_run_openclaw(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        calls.append(list(args))
        if args[:3] == ["--no-color", "channels", "logs"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "line1\nline2", "")
        return om._CmdResult(True, 0, ["openclaw", *args], "", "")

    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(om, "_run_openclaw", fake_run_openclaw)
    monkeypatch.setattr(om, "_resolve_dns", lambda hostname: {"hostname": hostname, "ok": False, "addresses": [], "error": "nx"})

    out = om._gateway_health_and_heal(
        oc_base=["openclaw"],
        channels_payload=payload,
        primary_channel="whatsapp",
        apply=True,
        restart_on_rpc_failure=True,
        recheck_delay_seconds=0.0,
        attempt_primary_reenable=True,
        primary_restart_attempts=3,
    )

    assert "skip_restart_due_to_dns_failure" in out["warnings"]
    assert "primary_channel_unresolved:whatsapp_dns_resolution_failed" in out["errors"]
    assert out.get("channel_logs_tail")
    assert not any("restart" in step for step in out["actions"])
    assert not any(cmd[:3] == ["--no-color", "config", "set"] for cmd in calls)


def test_gateway_health_reenable_avoids_legacy_whatsapp_enabled_key(monkeypatch) -> None:
    calls: list[list[str]] = []
    payload = _handshake_timeout_channels_payload()

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args == ["gateway", "status"]:
            gateway_payload = {
                "rpc": {"ok": True},
                "service": {"runtime": {"status": "running", "state": "Running"}},
            }
            return om._CmdResult(True, 0, ["openclaw", "gateway", "status"], "", ""), gateway_payload
        if args in (["channels", "status"], ["channels", "status", "--probe"]):
            return om._CmdResult(True, 0, ["openclaw", "channels", "status"], "", ""), payload
        raise AssertionError(f"Unexpected json call args: {args}")

    def fake_run_openclaw(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        calls.append(list(args))
        if args == ["gateway", "restart"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "", "")
        if args[:3] == ["--no-color", "channels", "logs"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "line1\nline2", "")
        if args[:3] == ["--no-color", "config", "set"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "", "")
        raise AssertionError(f"Unexpected call args: {args}")

    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(om, "_run_openclaw", fake_run_openclaw)
    monkeypatch.setattr(om, "_resolve_dns", lambda hostname: {"hostname": hostname, "ok": True, "addresses": ["127.0.0.1"], "error": ""})

    out = om._gateway_health_and_heal(
        oc_base=["openclaw"],
        channels_payload=payload,
        primary_channel="whatsapp",
        apply=True,
        restart_on_rpc_failure=True,
        recheck_delay_seconds=0.0,
        attempt_primary_reenable=True,
        primary_restart_attempts=1,
    )

    assert "primary_account_disabled:default" in out["actions"]
    assert "primary_account_enabled:default" in out["actions"]
    assert any(
        cmd[:4] == ["--no-color", "config", "set", "channels.whatsapp.accounts.default.enabled"]
        for cmd in calls
    )
    assert not any(
        cmd[:4] == ["--no-color", "config", "set", "channels.whatsapp.enabled"]
        for cmd in calls
    )


def test_gateway_health_dns_reprobe_recovered_then_restarts(monkeypatch) -> None:
    calls: list[list[str]] = []
    dns_payload = _dns_down_channels_payload()
    handshake_payload = _handshake_timeout_channels_payload()
    healthy_payload = _healthy_channels_payload()

    status_probe_count = {"value": 0}

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args == ["gateway", "status"]:
            gateway_payload = {
                "rpc": {"ok": True},
                "service": {"runtime": {"status": "running", "state": "Running"}},
            }
            return om._CmdResult(True, 0, ["openclaw", "gateway", "status"], "", ""), gateway_payload
        if args == ["channels", "status", "--probe"]:
            status_probe_count["value"] += 1
            if status_probe_count["value"] == 1:
                return om._CmdResult(True, 0, ["openclaw", "channels", "status", "--probe"], "", ""), handshake_payload
            return om._CmdResult(True, 0, ["openclaw", "channels", "status", "--probe"], "", ""), healthy_payload
        raise AssertionError(f"Unexpected json call args: {args}")

    def fake_run_openclaw(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        calls.append(list(args))
        if args == ["gateway", "restart"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "", "")
        if args[:3] == ["--no-color", "channels", "logs"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "line1\nline2", "")
        raise AssertionError(f"Unexpected call args: {args}")

    dns_responses = iter(
        [
            {"hostname": "web.whatsapp.com", "ok": False, "addresses": [], "error": "dns-down"},
            {"hostname": "web.whatsapp.com", "ok": False, "addresses": [], "error": "dns-down"},
            {"hostname": "web.whatsapp.com", "ok": True, "addresses": ["102.132.104.60"], "error": ""},
        ]
    )

    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(om, "_run_openclaw", fake_run_openclaw)
    monkeypatch.setattr(om, "_resolve_dns", lambda hostname: next(dns_responses))
    monkeypatch.setattr(om.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("OPENCLAW_DNS_REPROBE_ATTEMPTS", "2")
    monkeypatch.setenv("OPENCLAW_DNS_REPROBE_BASE_DELAY_SECONDS", "0")

    out = om._gateway_health_and_heal(
        oc_base=["openclaw"],
        channels_payload=dns_payload,
        primary_channel="whatsapp",
        apply=True,
        restart_on_rpc_failure=True,
        recheck_delay_seconds=0.0,
        attempt_primary_reenable=False,
        primary_restart_attempts=1,
    )

    assert "dns_reprobe_recovered" in out["actions"]
    assert "gateway_restart_primary_channel_recovery" in out["actions"]
    assert out["primary_channel_issue_final"] is None
    assert not out["errors"]
    assert any(cmd == ["gateway", "restart"] for cmd in calls)


def test_gateway_health_reports_delivery_recovery_signals(monkeypatch) -> None:
    payload = _healthy_channels_payload()

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args == ["gateway", "status"]:
            gateway_payload = {
                "rpc": {"ok": True},
                "service": {"runtime": {"status": "running", "state": "Running"}},
                "port": {"status": "busy", "listeners": [{"pid": 12345}]},
            }
            return om._CmdResult(True, 0, ["openclaw", "gateway", "status"], "", ""), gateway_payload
        raise AssertionError(f"Unexpected json call args: {args}")

    def fake_run_openclaw(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args[:3] == ["--no-color", "channels", "logs"]:
            return om._CmdResult(
                True,
                0,
                ["openclaw", *args],
                "\n".join(
                    [
                        "Decrypted message with closed session",
                        "Waiting 25000ms before retrying delivery",
                        "Recovery time budget exceeded — 1 entries deferred to next restart",
                    ]
                ),
                "",
            )
        raise AssertionError(f"Unexpected call args: {args}")

    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(om, "_run_openclaw", fake_run_openclaw)
    monkeypatch.setattr(om, "_resolve_dns", lambda hostname: {"hostname": hostname, "ok": True, "addresses": ["127.0.0.1"], "error": ""})
    monkeypatch.setenv("OPENCLAW_DELIVERY_RETRY_WARN_MS", "20000")

    out = om._gateway_health_and_heal(
        oc_base=["openclaw"],
        channels_payload=payload,
        primary_channel="whatsapp",
        apply=False,
        restart_on_rpc_failure=False,
        recheck_delay_seconds=0.0,
        attempt_primary_reenable=False,
        primary_restart_attempts=1,
    )

    diag = out.get("delivery_recovery")
    assert isinstance(diag, dict)
    assert int(diag.get("signal_count") or 0) >= 3
    assert "whatsapp_closed_session_signal_detected" in out["warnings"]
    assert "delivery_recovery_budget_exceeded" in out["warnings"]
    assert "delivery_retry_wait_high:25000ms" in out["warnings"]
    assert "run: openclaw channels login --channel whatsapp --account default --verbose" in out["manual_actions"]


def test_gateway_health_skips_restart_when_cooldown_active(monkeypatch) -> None:
    calls: list[list[str]] = []
    payload = _handshake_timeout_channels_payload()
    state = {"last_gateway_restart_at": om._utc_now_iso()}

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args == ["gateway", "status"]:
            gateway_payload = {
                "rpc": {"ok": True},
                "service": {"runtime": {"status": "running", "state": "Running"}},
            }
            return om._CmdResult(True, 0, ["openclaw", "gateway", "status"], "", ""), gateway_payload
        raise AssertionError(f"Unexpected json call args: {args}")

    def fake_run_openclaw(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        calls.append(list(args))
        if args[:3] == ["--no-color", "channels", "logs"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "", "")
        raise AssertionError(f"Unexpected call args: {args}")

    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(om, "_run_openclaw", fake_run_openclaw)
    monkeypatch.setattr(om, "_resolve_dns", lambda hostname: {"hostname": hostname, "ok": True, "addresses": ["127.0.0.1"], "error": ""})

    out = om._gateway_health_and_heal(
        oc_base=["openclaw"],
        channels_payload=payload,
        primary_channel="whatsapp",
        apply=True,
        restart_on_rpc_failure=True,
        recheck_delay_seconds=0.0,
        attempt_primary_reenable=False,
        primary_restart_attempts=2,
        state=state,
        gateway_restart_cooldown_seconds=600.0,
    )

    assert any(str(x).startswith("gateway_restart_cooldown_active:") for x in out["warnings"])
    assert not any(cmd == ["gateway", "restart"] for cmd in calls)


def test_gateway_health_recovers_detached_listener_conflict(monkeypatch) -> None:
    calls: list[list[str]] = []
    payload = _handshake_timeout_channels_payload()
    healthy_payload = _healthy_channels_payload()
    gateway_status_calls = {"value": 0}

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args == ["gateway", "status"]:
            gateway_status_calls["value"] += 1
            if gateway_status_calls["value"] == 1:
                gateway_payload = {
                    "rpc": {"ok": True},
                    "service": {"runtime": {"status": "stopped", "state": "Ready"}},
                    "gateway": {"port": 18789},
                    "port": {
                        "status": "busy",
                        "listeners": [
                            {
                                "pid": 12345,
                                "command": "node.exe",
                                "commandLine": "\"C:\\Program Files\\nodejs\\node.exe\" C:\\Users\\Bestman\\AppData\\Roaming\\npm\\node_modules\\openclaw\\dist\\index.js gateway --port 18789",
                            }
                        ],
                    },
                }
            else:
                gateway_payload = {
                    "rpc": {"ok": False},
                    "service": {"runtime": {"status": "stopped", "state": "Ready"}},
                    "gateway": {"port": 18789},
                    "port": {"status": "free", "listeners": []},
                }
            return om._CmdResult(True, 0, ["openclaw", "gateway", "status"], "", ""), gateway_payload
        if args == ["channels", "status", "--probe"]:
            return om._CmdResult(True, 0, ["openclaw", "channels", "status", "--probe"], "", ""), healthy_payload
        raise AssertionError(f"Unexpected json call args: {args}")

    def fake_run_openclaw(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        calls.append(list(args))
        if args in (["gateway", "stop"], ["gateway", "restart"]):
            return om._CmdResult(True, 0, ["openclaw", *args], "", "")
        if args[:3] == ["--no-color", "channels", "logs"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "", "")
        raise AssertionError(f"Unexpected call args: {args}")

    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(om, "_run_openclaw", fake_run_openclaw)
    monkeypatch.setattr(om, "_pid_running", lambda pid: pid == 12345)
    monkeypatch.setattr(om, "_terminate_pid", lambda pid: om._CmdResult(True, 0, ["taskkill", "/PID", str(pid)], "", ""))
    monkeypatch.setattr(om, "_resolve_dns", lambda hostname: {"hostname": hostname, "ok": True, "addresses": ["127.0.0.1"], "error": ""})

    out = om._gateway_health_and_heal(
        oc_base=["openclaw"],
        channels_payload=payload,
        primary_channel="whatsapp",
        apply=True,
        restart_on_rpc_failure=True,
        recheck_delay_seconds=0.0,
        attempt_primary_reenable=False,
        primary_restart_attempts=1,
    )

    assert "gateway_stop_detached_listener_recovery" in out["actions"]
    assert "gateway_listener_terminated:12345" in out["actions"]
    assert "gateway_restart_primary_channel_recovery" in out["actions"]
    assert out["primary_channel_issue_final"] is None
    assert any(cmd == ["gateway", "stop"] for cmd in calls)
    assert any(cmd == ["gateway", "restart"] for cmd in calls)


def test_gateway_health_does_not_kill_unverified_listener(monkeypatch) -> None:
    calls: list[list[str]] = []
    payload = _handshake_timeout_channels_payload()

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args == ["gateway", "status"]:
            gateway_payload = {
                "rpc": {"ok": True},
                "service": {"runtime": {"status": "stopped", "state": "Ready"}},
                "gateway": {"port": 18789},
                "port": {
                    "status": "busy",
                    "listeners": [
                        {
                            "pid": 22222,
                            "command": "python.exe",
                            "commandLine": "python -m http.server 18789",
                        }
                    ],
                },
            }
            return om._CmdResult(True, 0, ["openclaw", "gateway", "status"], "", ""), gateway_payload
        raise AssertionError(f"Unexpected json call args: {args}")

    def fake_run_openclaw(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        calls.append(list(args))
        if args[:3] == ["--no-color", "channels", "logs"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "", "")
        raise AssertionError(f"Unexpected call args: {args}")

    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(om, "_run_openclaw", fake_run_openclaw)
    monkeypatch.setattr(om, "_resolve_dns", lambda hostname: {"hostname": hostname, "ok": True, "addresses": ["127.0.0.1"], "error": ""})

    out = om._gateway_health_and_heal(
        oc_base=["openclaw"],
        channels_payload=payload,
        primary_channel="whatsapp",
        apply=True,
        restart_on_rpc_failure=True,
        recheck_delay_seconds=0.0,
        attempt_primary_reenable=False,
        primary_restart_attempts=1,
    )

    assert "gateway_detached_listener_conflict" in out["warnings"]
    assert "skip_restart_due_to_listener_conflict" in out["warnings"]
    assert out["primary_channel_issue_final"] == "whatsapp_handshake_timeout"
    assert not any(cmd == ["gateway", "stop"] for cmd in calls)
    assert not any(cmd == ["gateway", "restart"] for cmd in calls)


def test_lock_holder_matches_process_rejects_pid_reuse(monkeypatch) -> None:
    holder = {
        "pid": 17116,
        "mode": "watch",
        "command": "python C:/repo/scripts/openclaw_maintenance.py --watch --apply",
    }

    monkeypatch.setattr(om, "_pid_running", lambda pid: pid == 17116)
    monkeypatch.setattr(
        om,
        "_process_command_line",
        lambda pid: "C:\\WINDOWS\\system32\\svchost.exe -k LocalService -p -s NPSMSvc" if pid == 17116 else "",
    )

    assert om._lock_holder_matches_process(holder) is False


def test_acquire_run_lock_reclaims_stale_lock_when_pid_reused(monkeypatch, tmp_path) -> None:
    lock_path = tmp_path / "openclaw_maintenance.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 17116,
                "mode": "watch",
                "created_at_utc": om._utc_now_iso(),
                "token": "old-token",
                "command": "python C:/repo/scripts/openclaw_maintenance.py --watch --apply",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(om, "_pid_running", lambda pid: pid == 17116)
    monkeypatch.setattr(
        om,
        "_process_command_line",
        lambda pid: "C:\\WINDOWS\\system32\\svchost.exe -k LocalService -p -s NPSMSvc" if pid == 17116 else "",
    )

    attempt = om._acquire_run_lock(
        lock_path=lock_path,
        mode="run",
        wait_seconds=0.0,
        stale_seconds=60,
    )

    assert attempt.acquired is True
    assert attempt.lock is not None
    assert attempt.reason == "acquired"


def test_acquire_run_lock_preserves_live_matching_watch_holder(monkeypatch, tmp_path) -> None:
    lock_path = tmp_path / "openclaw_maintenance.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 22222,
                "mode": "watch",
                "created_at_utc": om._utc_now_iso(),
                "token": "watch-token",
                "command": "python C:/repo/scripts/openclaw_maintenance.py --watch --apply",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(om, "_pid_running", lambda pid: pid == 22222)
    monkeypatch.setattr(
        om,
        "_process_command_line",
        lambda pid: "python C:/repo/scripts/openclaw_maintenance.py --watch --apply" if pid == 22222 else "",
    )

    attempt = om._acquire_run_lock(
        lock_path=lock_path,
        mode="run",
        wait_seconds=0.0,
        stale_seconds=60,
    )

    assert attempt.acquired is False
    assert attempt.reason == "held"


def test_cleanup_stale_session_locks_handles_sessions_glob_error(monkeypatch, tmp_path) -> None:
    agents_root = tmp_path / ".openclaw" / "agents"
    agents_root.mkdir(parents=True)

    monkeypatch.setattr(om.Path, "home", staticmethod(lambda: tmp_path))
    original_glob = om.Path.glob

    def fake_glob(self, pattern):  # type: ignore[no-untyped-def]
        if pattern == "*/sessions":
            raise OSError(87, "The parameter is incorrect")
        return original_glob(self, pattern)

    monkeypatch.setattr(om.Path, "glob", fake_glob)

    out = om._cleanup_stale_session_locks(apply=False, session_stale_seconds=7200)

    assert any(str(x).startswith("scan_sessions_root_failed:") for x in out["errors"])


def test_reconcile_bound_direct_sessions_resets_wrong_agent_and_stale_model(monkeypatch, tmp_path) -> None:
    openclaw_root = tmp_path / ".openclaw"
    (openclaw_root / "agents" / "ops" / "sessions").mkdir(parents=True)
    (openclaw_root / "agents" / "trading" / "sessions").mkdir(parents=True)
    (openclaw_root / "openclaw.json").write_text(
        json.dumps(
            {
                "acp": {"defaultAgent": "ops"},
                "agents": {
                    "list": [
                        {"id": "ops", "model": "ollama/qwen3:8b"},
                        {"id": "trading", "model": "ollama/qwen3:8b"},
                    ]
                },
                "bindings": [
                    {"agentId": "ops", "match": {"channel": "whatsapp", "accountId": "default"}}
                ],
            }
        ),
        encoding="utf-8",
    )
    ops_index = openclaw_root / "agents" / "ops" / "sessions" / "sessions.json"
    trading_index = openclaw_root / "agents" / "trading" / "sessions" / "sessions.json"
    ops_key = "agent:ops:whatsapp:direct:+2348061573767"
    trading_key = "agent:trading:whatsapp:direct:+2348061573767"
    ops_index.write_text(
        json.dumps({ops_key: {"model": "ollama/qwen3.5:27b-pruned"}}),
        encoding="utf-8",
    )
    trading_index.write_text(
        json.dumps({trading_key: {"model": "ollama/qwen3.5:27b-pruned"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(om.Path, "home", staticmethod(lambda: tmp_path))

    preview = om._reconcile_bound_direct_sessions(primary_channel="whatsapp", apply=False)

    assert preview["bound_agent_id"] == "ops"
    assert preview["expected_model"] == "ollama/qwen3:8b"
    assert preview["peers_with_conflicts"] == 1
    assert preview["duplicate_wrong_agent_keys"] == 1
    assert preview["refreshed_bound_keys"] == 1
    assert preview["session_indexes_written"] == 0

    out = om._reconcile_bound_direct_sessions(primary_channel="whatsapp", apply=True)

    assert out["session_indexes_written"] == 2
    assert out["updated_agents"] == ["ops", "trading"]
    assert json.loads(ops_index.read_text(encoding="utf-8")) == {}
    assert json.loads(trading_index.read_text(encoding="utf-8")) == {}


def test_watch_mode_survives_cycle_exception(monkeypatch, tmp_path) -> None:
    report_file = tmp_path / "openclaw_maintenance_report.json"
    state_file = tmp_path / "openclaw_maintenance_state.json"
    lock_file = tmp_path / "openclaw_maintenance.lock.json"

    cleanup_calls = {"count": 0}

    def fake_cleanup(*, apply: bool, session_stale_seconds: int) -> dict:
        del apply, session_stale_seconds
        cleanup_calls["count"] += 1
        if cleanup_calls["count"] >= 2:
            raise RuntimeError("simulated_cleanup_failure")
        return {"errors": []}

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args == ["channels", "status", "--probe"]:
            payload = _healthy_channels_payload()
            return om._CmdResult(True, 0, ["openclaw", "channels", "status", "--probe"], "", ""), payload
        raise AssertionError(f"Unexpected json call args: {args}")

    def fake_gateway_health_and_heal(**_kwargs):
        return {
            "errors": [],
            "warnings": [],
            "actions": [],
            "manual_actions": [],
            "primary_channel_issue_final": None,
        }

    sleep_calls = {"count": 0}

    def fake_sleep(_seconds: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise KeyboardInterrupt()

    mono = {"value": 0.0}

    def fake_monotonic() -> float:
        mono["value"] += 31.0
        return mono["value"]

    monkeypatch.setattr(om, "_cleanup_stale_session_locks", fake_cleanup)
    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(om, "_gateway_health_and_heal", fake_gateway_health_and_heal)
    monkeypatch.setattr(om.time, "sleep", fake_sleep)
    monkeypatch.setattr(om.time, "monotonic", fake_monotonic)

    rc = om.main(
        [
            "--watch",
            "--no-fast-supervisor",
            "--watch-interval",
            "30",
            "--report-file",
            str(report_file),
            "--state-file",
            str(state_file),
            "--lock-file",
            str(lock_file),
        ]
    )

    assert rc == 0
    payload = json.loads(report_file.read_text(encoding="utf-8"))
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    assert payload.get("status") == "WARN"
    assert payload.get("watch_cycle") == 2
    assert any(str(x).startswith("watch_cycle_exception:RuntimeError:simulated_cleanup_failure") for x in warnings)


def test_fast_supervisor_tick_triggers_restart_at_threshold(monkeypatch) -> None:
    probe_calls = {"count": 0}
    run_calls: list[list[str]] = []
    state: dict[str, object] = {}
    fast_state: dict[str, object] = {"consecutive_failures": 1}

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        probe_calls["count"] += 1
        if args == ["channels", "status", "--probe"]:
            if probe_calls["count"] == 1:
                return om._CmdResult(False, 1, ["openclaw", *args], "", "closed"), None
            return om._CmdResult(True, 0, ["openclaw", *args], "", ""), _healthy_channels_payload()
        raise AssertionError(f"Unexpected json call args: {args}")

    def fake_run_openclaw(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        run_calls.append(list(args))
        if args == ["gateway", "restart"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "", "")
        raise AssertionError(f"Unexpected call args: {args}")

    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(om, "_run_openclaw", fake_run_openclaw)
    monkeypatch.setattr(om.time, "sleep", lambda _s: None)

    out = om._fast_supervisor_tick(
        oc_base=["openclaw"],
        primary_channel="whatsapp",
        apply=True,
        state_map=state,
        fast_state=fast_state,
        failure_threshold=2,
        restart_cooldown_seconds=0.0,
        probe_timeout_seconds=8.0,
        post_restart_recheck_seconds=0.0,
    )

    assert out["action"] == "gateway_restart_triggered"
    assert fast_state["consecutive_failures"] == 0
    assert "last_fast_supervisor_restart_at" in state
    assert any(cmd == ["gateway", "restart"] for cmd in run_calls)


def test_fast_supervisor_tick_healthy_resets_failures(monkeypatch) -> None:
    state: dict[str, object] = {}
    fast_state: dict[str, object] = {"consecutive_failures": 3}

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args == ["channels", "status", "--probe"]:
            return om._CmdResult(True, 0, ["openclaw", *args], "", ""), _healthy_channels_payload()
        raise AssertionError(f"Unexpected json call args: {args}")

    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(
        om,
        "_run_openclaw",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("restart should not be called for healthy probe")),
    )

    out = om._fast_supervisor_tick(
        oc_base=["openclaw"],
        primary_channel="whatsapp",
        apply=True,
        state_map=state,
        fast_state=fast_state,
        failure_threshold=2,
        restart_cooldown_seconds=20.0,
        probe_timeout_seconds=8.0,
        post_restart_recheck_seconds=0.0,
    )

    assert out["ok"] is True
    assert out["action"] == "none"
    assert out["consecutive_failures"] == 0
    assert fast_state["consecutive_failures"] == 0


def test_fast_supervisor_tick_softens_probe_timeout_when_gateway_is_healthy(monkeypatch) -> None:
    state: dict[str, object] = {}
    fast_state: dict[str, object] = {"consecutive_failures": 1}

    def fake_run_openclaw_json(*, oc_base, args, timeout_seconds=20.0):
        del oc_base, timeout_seconds
        if args == ["channels", "status", "--probe"]:
            return om._CmdResult(False, 124, ["openclaw", *args], "", "timeout"), None
        if args == ["gateway", "status"]:
            payload = {
                "rpc": {"ok": True},
                "service": {"runtime": {"status": "running", "state": "Running"}},
            }
            return om._CmdResult(True, 0, ["openclaw", *args], "", ""), payload
        raise AssertionError(f"Unexpected json call args: {args}")

    monkeypatch.setattr(om, "_run_openclaw_json", fake_run_openclaw_json)
    monkeypatch.setattr(
        om,
        "_run_openclaw",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("restart should not be called for softened timeout")),
    )

    out = om._fast_supervisor_tick(
        oc_base=["openclaw"],
        primary_channel="whatsapp",
        apply=True,
        state_map=state,
        fast_state=fast_state,
        failure_threshold=2,
        restart_cooldown_seconds=20.0,
        probe_timeout_seconds=8.0,
        post_restart_recheck_seconds=0.0,
    )

    assert out["ok"] is True
    assert out["action"] == "soft_timeout_skip"
    assert out["reason"] == "channels_status_timeout_softened"
    assert out["consecutive_failures"] == 0
    assert fast_state["consecutive_failures"] == 0
