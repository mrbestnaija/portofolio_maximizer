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


def test_collect_runtime_status_runs_production_gate_in_unattended_profile(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "portfolio_maximizer.db").write_text("", encoding="utf-8")

    captured: list[tuple[str, list[str]]] = []

    def fake_run_check(name, cmd, timeout_seconds, **kwargs):
        del timeout_seconds, kwargs
        captured.append((str(name), list(cmd)))
        return {
            "name": str(name),
            "ok": True,
            "returncode": 0,
            "duration_seconds": 0.0,
            "command": " ".join(cmd),
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "_run_check", fake_run_check)
    monkeypatch.setattr(
        mod,
        "_openclaw_exec_environment_check",
        lambda: {
            "name": "openclaw_exec_env",
            "ok": True,
            "returncode": 0,
            "duration_seconds": 0.0,
            "command": "validate",
            "stdout": "ok",
            "stderr": "",
            "signals": ["exec_env_valid"],
        },
    )

    payload = mod.collect_runtime_status(timeout_seconds=1.0)
    assert payload["status"] == "ok"

    by_name = {name: cmd for name, cmd in captured}
    prod_cmd = by_name["production_gate"]
    assert "--unattended-profile" in prod_cmd
