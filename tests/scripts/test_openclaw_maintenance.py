from __future__ import annotations

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


def test_derive_status_marks_primary_unresolved_as_fail() -> None:
    assert om._derive_status([]) == "PASS"
    assert om._derive_status(["gateway_restart_failed"]) == "WARN"
    assert om._derive_status(["primary_channel_unresolved:whatsapp_dns_resolution_failed"]) == "FAIL"


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
