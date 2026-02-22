from __future__ import annotations

import json

from scripts import openclaw_regression_gate as gate


def _healthy_payload() -> dict:
    return {
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
                    "running": True,
                    "connected": True,
                }
            ]
        },
    }


def _down_payload() -> dict:
    return {
        "channels": {
            "whatsapp": {
                "configured": True,
                "linked": True,
                "running": False,
                "connected": False,
            }
        },
        "channelAccounts": {
            "whatsapp": [
                {
                    "accountId": "default",
                    "enabled": True,
                    "running": False,
                    "connected": False,
                }
            ]
        },
    }


def test_channel_ready_accepts_running_connected_whatsapp() -> None:
    ok, reason = gate._channel_ready(_healthy_payload(), "whatsapp")
    assert ok is True
    assert reason == "ok"


def test_channel_ready_rejects_down_channel() -> None:
    ok, reason = gate._channel_ready(_down_payload(), "whatsapp")
    assert ok is False
    assert reason == "channel_not_running"


def test_run_regression_gate_skips_when_cli_missing_if_allowed(monkeypatch) -> None:
    def fake_run(cmd: list[str], *, timeout_seconds: float):
        del timeout_seconds
        return gate._CmdResult(False, 127, cmd, "", "not found")

    monkeypatch.setattr(gate, "_run", fake_run)

    ok, report = gate.run_regression_gate(
        openclaw_command="openclaw",
        python_bin="python",
        primary_channel="whatsapp",
        timeout_seconds=5.0,
        allow_missing_openclaw=True,
    )

    assert ok is True
    assert report["status"] == "SKIP"
    assert "openclaw_cli_missing" in report["warnings"]


def test_run_regression_gate_fails_when_channel_not_ready(monkeypatch) -> None:
    payload = _down_payload()

    def fake_run(cmd: list[str], *, timeout_seconds: float):
        del timeout_seconds
        if "channels" in cmd and "status" in cmd:
            return gate._CmdResult(True, 0, cmd, json.dumps(payload), "")
        return gate._CmdResult(True, 0, cmd, "", "")

    monkeypatch.setattr(gate, "_run", fake_run)

    ok, report = gate.run_regression_gate(
        openclaw_command="openclaw",
        python_bin="python",
        primary_channel="whatsapp",
        timeout_seconds=5.0,
        allow_missing_openclaw=False,
    )

    assert ok is False
    assert report["status"] == "FAIL"
    assert any(str(x).startswith("primary_channel_not_ready:") for x in report["errors"])


def test_run_regression_gate_passes_when_probe_and_maintenance_pass(monkeypatch) -> None:
    payload = _healthy_payload()

    def fake_run(cmd: list[str], *, timeout_seconds: float):
        del timeout_seconds
        if "channels" in cmd and "status" in cmd:
            return gate._CmdResult(True, 0, cmd, json.dumps(payload), "")
        if "openclaw_maintenance.py" in " ".join(cmd):
            return gate._CmdResult(True, 0, cmd, "[openclaw_maintenance] status=PASS", "")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(gate, "_run", fake_run)

    ok, report = gate.run_regression_gate(
        openclaw_command="openclaw",
        python_bin="python",
        primary_channel="whatsapp",
        timeout_seconds=5.0,
        allow_missing_openclaw=False,
    )

    assert ok is True
    assert report["status"] == "PASS"
    assert not report["errors"]
