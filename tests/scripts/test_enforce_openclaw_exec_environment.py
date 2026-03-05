from __future__ import annotations

import json
from pathlib import Path

import scripts.enforce_openclaw_exec_environment as mod


def test_enforce_exec_environment_sets_host_sandbox_mode_and_acp_agent() -> None:
    cfg = {
        "agents": {
            "list": [
                {"id": "ops", "default": False},
                {"id": "training", "default": True},
            ],
            "defaults": {"sandbox": {"mode": "off"}},
        }
    }

    out, changes = mod.enforce_exec_environment(
        cfg,
        preferred_host="sandbox",
        sandbox_mode_for_sandbox_host="non-main",
        ensure_acp_default_agent=True,
        preferred_agent="",
    )

    assert out["tools"]["exec"]["host"] == "sandbox"
    assert out["agents"]["defaults"]["sandbox"]["mode"] == "non-main"
    assert out["acp"]["defaultAgent"] == "training"
    assert changes


def test_enforce_exec_environment_gateway_host_keeps_existing_sandbox_mode() -> None:
    cfg = {
        "agents": {"defaults": {"sandbox": {"mode": "off", "scope": "agent"}}},
        "tools": {"exec": {"host": "sandbox"}},
    }

    out, _changes = mod.enforce_exec_environment(
        cfg,
        preferred_host="gateway",
        sandbox_mode_for_sandbox_host="non-main",
        ensure_acp_default_agent=False,
        preferred_agent="",
    )

    assert out["tools"]["exec"]["host"] == "gateway"
    assert out["agents"]["defaults"]["sandbox"]["mode"] == "off"


def test_enforce_config_file_uses_exec_host_conf_when_host_not_provided(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "openclaw.json"
    cfg_path.write_text(json.dumps({"agents": {"defaults": {"sandbox": {"mode": "off"}}}}), encoding="utf-8")

    conf_path = tmp_path / "exec_host.conf"
    conf_path.write_text("tools.exec.host=sandbox\n", encoding="utf-8")
    monkeypatch.setattr(mod, "DEFAULT_EXEC_HOST_CONF", conf_path)

    report = mod.enforce_config_file(
        config_path=cfg_path,
        preferred_host="",
        sandbox_mode_for_sandbox_host="non-main",
        ensure_acp_default_agent=True,
        preferred_agent="",
        dry_run=False,
    )

    assert report["ok"] is True
    assert report["changed"] is True

    updated = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert updated["tools"]["exec"]["host"] == "sandbox"
    assert updated["agents"]["defaults"]["sandbox"]["mode"] == "non-main"
    assert updated["acp"]["defaultAgent"] == "ops"


def test_enforce_config_file_missing_config_returns_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    report = mod.enforce_config_file(
        config_path=missing,
        preferred_host="sandbox",
        sandbox_mode_for_sandbox_host="non-main",
        ensure_acp_default_agent=True,
        preferred_agent="",
        dry_run=False,
    )
    assert report["ok"] is False
    assert str(report["error"]).startswith("config_missing:")
