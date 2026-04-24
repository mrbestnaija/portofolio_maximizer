from __future__ import annotations

from scripts import openclaw_notify as on
from utils.openclaw_cli import OpenClawResult
import utils.openclaw_cli as oc


_SUPERVISED_CONFLICT_TEXT = (
    "Gateway failed to start: gateway already running (pid 217512); lock timeout after 5000ms\n"
    "If the gateway is supervised, stop it with: openclaw gateway stop"
)


def test_gateway_supervision_conflict_classifier_detects_supervised_restart_conflict() -> None:
    assert on._is_gateway_supervision_conflict_text(_SUPERVISED_CONFLICT_TEXT) is True
    assert on._is_gateway_supervision_conflict_text("connection refused") is False


def test_main_send_mode_suppresses_verbose_tail_for_supervised_gateway_conflict(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(on, "_write_run_log", lambda **kwargs: None)
    monkeypatch.setattr(
        on,
        "send_message_multi",
        lambda **kwargs: [
            OpenClawResult(
                ok=False,
                returncode=1,
                command=["openclaw", "message", "send"],
                stdout="",
                stderr=_SUPERVISED_CONFLICT_TEXT,
            )
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "openclaw_notify.py",
            "--targets",
            "whatsapp:+15551234567",
            "--message",
            "hello",
            "--no-infer-to",
        ],
    )

    rc = on.main()

    err = capsys.readouterr().err
    assert rc == 1
    assert "already running under supervision" in err
    assert "FAILED (exit=" not in err
    assert "stderr (tail)" not in err


def test_main_prompt_mode_suppresses_verbose_tail_for_supervised_gateway_conflict(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(on, "_write_run_log", lambda **kwargs: None)
    monkeypatch.setattr(
        oc,
        "run_agent_turn",
        lambda **kwargs: OpenClawResult(
            ok=False,
            returncode=1,
            command=["openclaw", "agent"],
            stdout="",
            stderr=_SUPERVISED_CONFLICT_TEXT,
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "openclaw_notify.py",
            "--prompt",
            "--to",
            "+15551234567",
            "--message",
            "hello",
            "--no-infer-to",
        ],
    )

    rc = on.main()

    err = capsys.readouterr().err
    assert rc == 1
    assert "already running under supervision" in err
    assert "FAILED (exit=" not in err
    assert "stderr (tail)" not in err


def test_main_prompt_mode_passes_trusted_high_risk_flag(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(on, "_write_run_log", lambda **kwargs: None)
    monkeypatch.setattr(
        oc,
        "run_agent_turn",
        lambda **kwargs: captured.update(kwargs) or OpenClawResult(
            ok=True,
            returncode=0,
            command=["openclaw", "agent"],
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "openclaw_notify.py",
            "--prompt",
            "--to",
            "+15551234567",
            "--message",
            "hello",
            "--approve-high-risk",
            "--no-infer-to",
        ],
    )

    rc = on.main()

    assert rc == 0
    assert captured["approve_high_risk"] is True
