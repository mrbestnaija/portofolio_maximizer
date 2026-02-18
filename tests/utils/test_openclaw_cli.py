from __future__ import annotations

import re
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from utils.openclaw_cli import (
    _clear_stuck_gateway_sessions,
    _is_retryable_error,
    _is_session_lock_error,
    _is_missing_listener_error,
    build_agent_turn_command,
    build_message_send_command,
    OpenClawResult,
    parse_openclaw_targets,
    run_agent_turn,
    send_message,
)


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


# ---------------------------------------------------------------------------
# Stuck session detection and auto-kill tests
# ---------------------------------------------------------------------------


class TestStuckSessionDetection:
    """Safety tests: prevent stuck sessions from blocking the gateway."""

    STUCK_LOG_OUTPUT = textwrap.dedent("""\
        2026-02-18T06:24:12.656Z warn diagnostic stuck session: sessionId=67ca0549 sessionKey=unknown state=processing age=2539s queueDepth=0
        2026-02-18T06:24:42.668Z debug diagnostic heartbeat: webhooks=0/0/0 active=1 waiting=0 queued=1
        2026-02-18T06:24:42.670Z warn diagnostic stuck session: sessionId=67ca0549 sessionKey=unknown state=processing age=2569s queueDepth=0
    """)

    HEALTHY_LOG_OUTPUT = textwrap.dedent("""\
        2026-02-18T06:24:42.668Z debug diagnostic heartbeat: webhooks=0/0/0 active=0 waiting=0 queued=0
    """)

    def test_stuck_session_age_regex_extracts_ages(self) -> None:
        """The regex used to detect stuck sessions must parse age values correctly."""
        ages = [int(m) for m in re.findall(r"stuck session:.*?age=(\d+)s", self.STUCK_LOG_OUTPUT)]
        assert ages == [2539, 2569], f"Expected [2539, 2569], got {ages}"

    def test_stuck_session_age_regex_no_match_on_healthy_logs(self) -> None:
        """Healthy logs must not trigger stuck session detection."""
        ages = [int(m) for m in re.findall(r"stuck session:.*?age=(\d+)s", self.HEALTHY_LOG_OUTPUT)]
        assert ages == []

    @patch("utils.openclaw_cli.subprocess.run")
    def test_clear_stuck_gateway_sessions_triggers_on_high_age(self, mock_run: MagicMock) -> None:
        """When a stuck session exceeds threshold, gateway must be force-restarted."""
        log_proc = MagicMock()
        log_proc.stdout = self.STUCK_LOG_OUTPUT
        log_proc.stderr = ""

        stop_proc = MagicMock()
        start_proc = MagicMock()
        kill_proc = MagicMock()

        mock_run.side_effect = [log_proc, stop_proc, kill_proc, start_proc]

        result = _clear_stuck_gateway_sessions(
            command="openclaw",
            max_age_seconds=300,
        )
        assert result is True, "Should return True when stuck session found and gateway restarted"
        # Verify gateway stop was called
        calls = mock_run.call_args_list
        assert any("stop" in str(c) for c in calls), "Gateway stop must be called"

    @patch("utils.openclaw_cli.subprocess.run")
    def test_clear_stuck_gateway_sessions_no_trigger_on_healthy(self, mock_run: MagicMock) -> None:
        """When no stuck sessions exist, gateway must NOT be restarted."""
        log_proc = MagicMock()
        log_proc.stdout = self.HEALTHY_LOG_OUTPUT
        log_proc.stderr = ""

        mock_run.return_value = log_proc

        result = _clear_stuck_gateway_sessions(
            command="openclaw",
            max_age_seconds=300,
        )
        assert result is False, "Should return False when no stuck sessions"
        assert mock_run.call_count == 1, "Only the log fetch should be called"

    @patch("utils.openclaw_cli.subprocess.run")
    def test_clear_stuck_gateway_sessions_no_trigger_below_threshold(self, mock_run: MagicMock) -> None:
        """Sessions below the age threshold must NOT trigger a restart."""
        log_proc = MagicMock()
        log_proc.stdout = "stuck session: sessionId=abc state=processing age=120s queueDepth=0"
        log_proc.stderr = ""

        mock_run.return_value = log_proc

        result = _clear_stuck_gateway_sessions(
            command="openclaw",
            max_age_seconds=300,
        )
        assert result is False, "120s < 300s threshold should not trigger restart"

    def test_clear_stuck_gateway_sessions_handles_exception_gracefully(self) -> None:
        """If the openclaw binary doesn't exist, function must not crash."""
        result = _clear_stuck_gateway_sessions(
            command="definitely-not-a-real-binary-12345",
            max_age_seconds=300,
        )
        assert result is False


class TestRunAgentTurnSafety:
    """Safety tests: run_agent_turn must not hang or create orphaned sessions."""

    def test_run_agent_turn_timeout_produces_returncode_124(self) -> None:
        """Agent turns that exceed timeout must return error code 124, not hang."""
        result = run_agent_turn(
            to="+15551234567",
            message="test",
            command="definitely-not-a-real-binary-12345",
            timeout_seconds=1.0,
        )
        # FileNotFoundError -> returncode 127
        assert result.ok is False
        assert result.returncode in (124, 127), f"Expected 124 (timeout) or 127 (not found), got {result.returncode}"

    def test_build_agent_turn_command_includes_timeout_flag(self) -> None:
        """The --timeout flag must always be present to prevent gateway-side hangs."""
        cmd = build_agent_turn_command(
            command="openclaw",
            to="+15551234567",
            message="test",
            cli_timeout_seconds=45.0,
        )
        assert "--timeout" in cmd, "Agent turn command must include --timeout flag"
        idx = cmd.index("--timeout")
        timeout_val = float(cmd[idx + 1])
        assert timeout_val == 45.0

    def test_build_agent_turn_command_cli_timeout_capped(self) -> None:
        """CLI timeout must be capped to prevent indefinite hangs."""
        cmd = build_agent_turn_command(
            command="openclaw",
            to="+15551234567",
            message="test",
            cli_timeout_seconds=120.0,
        )
        idx = cmd.index("--timeout")
        timeout_val = float(cmd[idx + 1])
        assert timeout_val <= 600.0, "CLI timeout must not exceed 600s"


class TestErrorClassification:
    """Safety tests: error classification must correctly identify retryable vs fatal errors."""

    def test_gateway_timeout_is_retryable(self) -> None:
        result = OpenClawResult(ok=False, returncode=1, command=[], stdout="", stderr="gateway timeout after 60000ms")
        assert _is_retryable_error(result) is True

    def test_missing_listener_detected(self) -> None:
        result = OpenClawResult(
            ok=False, returncode=1, command=[], stdout="",
            stderr="Error: No active WhatsApp Web listener (account: default). Start the gateway.",
        )
        assert _is_missing_listener_error(result) is True

    def test_session_lock_detected(self) -> None:
        result = OpenClawResult(ok=False, returncode=1, command=[], stdout="session file locked", stderr="")
        assert _is_session_lock_error(result) is True

    def test_auth_error_not_retryable(self) -> None:
        result = OpenClawResult(ok=False, returncode=1, command=[], stdout="", stderr="api key invalid")
        assert _is_retryable_error(result) is False

    def test_tool_not_supported_not_retryable(self) -> None:
        result = OpenClawResult(ok=False, returncode=1, command=[], stdout="", stderr="model does not support tools")
        assert _is_retryable_error(result) is False

    @patch("utils.openclaw_cli.time.sleep", return_value=None)
    @patch("utils.openclaw_cli.subprocess.run")
    def test_send_message_recovers_missing_listener_via_gateway_restart(
        self, mock_run: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        send_fail = MagicMock(returncode=1, stdout="", stderr="No active WhatsApp Web listener (account: default).")
        status_not_ready = MagicMock(
            returncode=0,
            stdout=(
                '{"channels":{"whatsapp":{"running":false,"connected":false}},'
                '"channelAccounts":{"whatsapp":[{"enabled":true,"running":false,"connected":false}]}}'
            ),
            stderr="",
        )
        restart_ok = MagicMock(returncode=0, stdout="restarted", stderr="")
        status_ready = MagicMock(
            returncode=0,
            stdout=(
                '{"channels":{"whatsapp":{"running":true,"connected":true}},'
                '"channelAccounts":{"whatsapp":[{"enabled":true,"running":true,"connected":true}]}}'
            ),
            stderr="",
        )
        send_ok = MagicMock(returncode=0, stdout='{"ok":true}', stderr="")
        mock_run.side_effect = [send_fail, status_not_ready, restart_ok, status_ready, send_ok]

        with patch.dict("utils.openclaw_cli._listener_recovery_state", {"last_restart_monotonic": 0.0}, clear=True):
            with patch.dict("utils.openclaw_cli.os.environ", {"OPENCLAW_AUTO_RECOVER_LISTENER": "1"}, clear=False):
                result = send_message(
                    to="+15551234567",
                    message="hello",
                    command="openclaw",
                    timeout_seconds=5.0,
                    max_retries=0,
                    skip_dedup=True,
                    skip_rate_limit=True,
                )

        assert result.ok is True
        calls = [" ".join(str(x) for x in (call.args[0] if call.args else [])) for call in mock_run.call_args_list]
        assert any("gateway restart" in cmd for cmd in calls), "Expected gateway restart recovery call"

    @patch("utils.openclaw_cli.time.sleep", return_value=None)
    @patch("utils.openclaw_cli.subprocess.run")
    def test_send_message_missing_listener_dns_hint_on_failed_recovery(
        self, mock_run: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        send_fail = MagicMock(returncode=1, stdout="", stderr="No active WhatsApp Web listener (account: default).")
        status_fail = MagicMock(
            returncode=1,
            stdout="",
            stderr="WebSocket Error (getaddrinfo ENOTFOUND web.whatsapp.com)",
        )
        mock_run.side_effect = [send_fail, status_fail]

        with patch.dict("utils.openclaw_cli._listener_recovery_state", {"last_restart_monotonic": 0.0}, clear=True):
            with patch.dict("utils.openclaw_cli.os.environ", {"OPENCLAW_AUTO_RECOVER_LISTENER": "1"}, clear=False):
                result = send_message(
                    to="+15551234567",
                    message="hello",
                    command="openclaw",
                    timeout_seconds=5.0,
                    max_retries=0,
                    skip_dedup=True,
                    skip_rate_limit=True,
                )

        assert result.ok is False
        assert "DNS lookup to web.whatsapp.com failed" in result.stderr

    @patch("utils.openclaw_cli.time.sleep", return_value=None)
    @patch("utils.openclaw_cli.time.monotonic", return_value=1000.0)
    @patch("utils.openclaw_cli.subprocess.run")
    def test_send_message_missing_listener_respects_recovery_cooldown(
        self, mock_run: MagicMock, _mock_monotonic: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        send_fail = MagicMock(returncode=1, stdout="", stderr="No active WhatsApp Web listener (account: default).")
        status_not_ready = MagicMock(
            returncode=0,
            stdout=(
                '{"channels":{"whatsapp":{"running":false,"connected":false}},'
                '"channelAccounts":{"whatsapp":[{"enabled":true,"running":false,"connected":false}]}}'
            ),
            stderr="",
        )
        mock_run.side_effect = [send_fail, status_not_ready]

        with patch.dict("utils.openclaw_cli._listener_recovery_state", {"last_restart_monotonic": 999.0}, clear=True):
            with patch.dict(
                "utils.openclaw_cli.os.environ",
                {
                    "OPENCLAW_AUTO_RECOVER_LISTENER": "1",
                    "OPENCLAW_LISTENER_RECOVERY_COOLDOWN_SECONDS": "600",
                },
                clear=False,
            ):
                result = send_message(
                    to="+15551234567",
                    message="hello",
                    command="openclaw",
                    timeout_seconds=5.0,
                    max_retries=0,
                    skip_dedup=True,
                    skip_rate_limit=True,
                )

        assert result.ok is False
        assert "cooldown" in result.stderr.lower()
        calls = [" ".join(str(x) for x in (call.args[0] if call.args else [])) for call in mock_run.call_args_list]
        assert not any("gateway restart" in cmd for cmd in calls)


class TestWorkspaceBootstrapContract:
    """Safety tests: workspace bootstrap files must fit within OpenClaw's bootstrap limit.

    When bootstrap files exceed the limit, OpenClaw truncates them, causing the
    LLM to lose critical context. This leads to hallucinated tool calls
    (wrong paths, missing args) and stuck sessions that block the gateway queue.

    Ref: 2026-02-18 incident - SOUL.md (11K) truncated at 8K limit caused
    9 consecutive failed tool calls and 2000s+ stuck sessions.
    """

    # OpenClaw default bootstrapMaxChars (must match openclaw.json config)
    BOOTSTRAP_MAX_CHARS = 20000

    def test_individual_workspace_files_within_limit(self) -> None:
        """No single workspace file should exceed the bootstrap limit."""
        from pathlib import Path

        workspace_files = ["SOUL.md", "AGENTS.md", "TOOLS.md", "IDENTITY.md", "USER.md"]
        project_root = Path(__file__).resolve().parents[2]

        for fname in workspace_files:
            fpath = project_root / fname
            if not fpath.exists():
                continue
            size = len(fpath.read_text(encoding="utf-8"))
            assert size <= self.BOOTSTRAP_MAX_CHARS, (
                f"{fname} is {size} chars (limit={self.BOOTSTRAP_MAX_CHARS}). "
                f"OpenClaw will truncate it, causing LLM context loss and stuck sessions. "
                f"Trim the file or increase agents.defaults.bootstrapMaxChars."
            )

    def test_total_workspace_files_within_budget(self) -> None:
        """Total workspace bootstrap payload must fit within a reasonable token budget.

        At ~4 chars per token, 20K chars = ~5K tokens. Keep total under 25K chars
        to prevent slow prefill on small models (qwen3:8b).
        """
        from pathlib import Path

        workspace_files = ["SOUL.md", "AGENTS.md", "TOOLS.md", "IDENTITY.md", "USER.md"]
        project_root = Path(__file__).resolve().parents[2]

        total = 0
        for fname in workspace_files:
            fpath = project_root / fname
            if fpath.exists():
                total += len(fpath.read_text(encoding="utf-8"))

        max_total = 25000  # ~6K tokens at 4 chars/token
        assert total <= max_total, (
            f"Total workspace files = {total} chars (limit={max_total}). "
            f"This causes slow prefill (~{total // 4000}K extra tokens) on small models. "
            f"Trim workspace files or split into skill-specific context."
        )


class TestSessionStoreContract:
    """Safety tests: session store must stay bounded and not accumulate stale sessions."""

    MAX_SESSIONS_THRESHOLD = 50  # Alert if session store grows beyond this

    def test_session_store_exists_and_is_valid_json(self) -> None:
        """Session store must be valid JSON (not corrupted by concurrent writes)."""
        import json
        from pathlib import Path

        store_path = Path.home() / ".openclaw" / "agents" / "main" / "sessions" / "sessions.json"
        if not store_path.exists():
            pytest.skip("OpenClaw session store not found (CI or fresh install)")

        data = json.loads(store_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), "Session store must be a JSON object"

    def test_session_store_not_bloated(self) -> None:
        """Session store must not accumulate unbounded stale sessions."""
        import json
        from pathlib import Path

        store_path = Path.home() / ".openclaw" / "agents" / "main" / "sessions" / "sessions.json"
        if not store_path.exists():
            pytest.skip("OpenClaw session store not found (CI or fresh install)")

        data = json.loads(store_path.read_text(encoding="utf-8"))
        count = len(data)
        assert count <= self.MAX_SESSIONS_THRESHOLD, (
            f"Session store has {count} entries (threshold={self.MAX_SESSIONS_THRESHOLD}). "
            f"Stale sessions are accumulating and will block the gateway queue."
        )
