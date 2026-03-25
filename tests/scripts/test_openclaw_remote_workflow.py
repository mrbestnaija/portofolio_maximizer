from __future__ import annotations

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
