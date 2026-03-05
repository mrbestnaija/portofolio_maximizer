from __future__ import annotations

import json
from pathlib import Path

from scripts import project_runtime_status as mod


def _write_openclaw(tmp_path: Path, payload: dict) -> None:
    cfg_path = tmp_path / ".openclaw" / "openclaw.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")


def test_exec_env_check_flags_invalid_host(monkeypatch, tmp_path) -> None:
    _write_openclaw(tmp_path, {"tools": {"exec": {"host": "invalid"}}})
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is False
    assert check["signals"] == ["invalid_exec_host"]


def test_exec_env_check_flags_invalid_sandbox_mode(monkeypatch, tmp_path) -> None:
    _write_openclaw(
        tmp_path,
        {
            "tools": {"exec": {"host": "sandbox"}},
            "agents": {"defaults": {"sandbox": {"mode": "off"}}},
        },
    )
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is False
    assert check["signals"] == ["invalid_sandbox_mode"]


def test_exec_env_check_flags_missing_acp_default(monkeypatch, tmp_path) -> None:
    _write_openclaw(
        tmp_path,
        {
            "tools": {"exec": {"host": "gateway"}},
            "agents": {"defaults": {"sandbox": {"mode": "non-main"}}},
        },
    )
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is False
    assert check["signals"] == ["missing_acp_default_agent"]


def test_exec_env_check_passes_with_valid_values(monkeypatch, tmp_path) -> None:
    _write_openclaw(
        tmp_path,
        {
            "tools": {"exec": {"host": "sandbox"}},
            "agents": {"defaults": {"sandbox": {"mode": "non-main"}}},
            "acp": {"defaultAgent": "training-agent-01"},
        },
    )
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is True
    assert check["signals"] == ["exec_env_valid"]
