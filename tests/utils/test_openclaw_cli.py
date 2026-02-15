from __future__ import annotations

from utils.openclaw_cli import build_message_send_command, parse_openclaw_targets, send_message


def test_build_message_send_command_basic() -> None:
    cmd = build_message_send_command(command="openclaw", to="target", message="hello")
    idx = cmd.index("message")
    assert cmd[idx : idx + 2] == ["message", "send"]
    assert "--target" in cmd
    assert "--message" in cmd


def test_send_message_missing_binary_is_handled() -> None:
    result = send_message(
        to="target",
        message="hello",
        command="definitely-not-a-real-openclaw-binary-123",
        timeout_seconds=0.5,
    )
    assert result.ok is False
    assert result.returncode == 127


def test_parse_openclaw_targets_e164_implies_whatsapp() -> None:
    targets = parse_openclaw_targets("+15551234567")
    assert targets == [("whatsapp", "+15551234567")]


def test_parse_openclaw_targets_channel_prefix_is_respected() -> None:
    targets = parse_openclaw_targets("telegram:@mychannel, discord:channel:123")
    assert targets == [("telegram", "@mychannel"), ("discord", "channel:123")]


def test_parse_openclaw_targets_default_channel_applies_to_bare_targets() -> None:
    targets = parse_openclaw_targets("@me", default_channel="telegram")
    assert targets == [("telegram", "@me")]
