from __future__ import annotations

import json
from pathlib import Path

from scripts import project_runtime_status as mod


def _write_openclaw(tmp_path: Path, payload: dict) -> None:
    cfg_path = tmp_path / ".openclaw" / "openclaw.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")


def _base_ok_check(name: str) -> dict:
    return {
        "name": name,
        "ok": True,
        "returncode": 0,
        "duration_seconds": 0.0,
        "command": name,
        "stdout": "",
        "stderr": "",
    }


def _write_valid_dashboard(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "generated_utc": "2099-01-01T00:00:00Z",
                    "ts": "2099-01-01T00:00:00Z",
                    "payload_schema_version": 2,
                    "payload_required_sections": [
                        "meta",
                        "pnl",
                        "signals",
                        "trade_events",
                        "price_series",
                        "robustness",
                        "live_denominator",
                        "quant_validation",
                    ],
                },
                "pnl": {"absolute": 1.0, "pct": 0.01},
                "signals": [],
                "trade_events": [],
                "price_series": {},
                "robustness": {
                    "overall_status": "OK",
                    "freshness_status": "FRESH",
                    "sidecar_age_minutes": {},
                },
                "live_denominator": {"status": "WAITING"},
                "quant_validation": {},
            }
        ),
        encoding="utf-8",
    )


def _write_valid_persistence_status(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2099-01-01T00:00:00Z",
                "status_contract": {
                    "schema_version": 1,
                    "max_age_seconds": 172800,
                    "reconciled_components": [
                        "dashboard.bridge",
                        "dashboard.http_server",
                        "dashboard.live_watcher",
                    ],
                },
                "dashboard": {
                    "bridge": {"running": True},
                    "http_server": {"running": True},
                    "live_watcher": {"running": True},
                    "warnings": [],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_valid_production_gate_artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "phase3_ready": True,
                "phase3_reason": "READY",
                "lift_inconclusive_allowed": False,
                "warmup_expired": True,
            }
        ),
        encoding="utf-8",
    )


def test_exec_env_check_flags_invalid_host(monkeypatch, tmp_path) -> None:
    _write_openclaw(tmp_path, {"tools": {"exec": {"host": "invalid"}}})
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is False
    assert check["signals"] == ["invalid_exec_host"]


def test_exec_env_check_accepts_utf8_bom(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / ".openclaw" / "openclaw.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tools": {"exec": {"host": "node"}},
        "agents": {"defaults": {"sandbox": {"mode": "non-main"}}},
        "acp": {"defaultAgent": "ops"},
    }
    cfg_path.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)
    if hasattr(mod, "_node_host_available"):
        monkeypatch.setattr(mod, "_node_host_available", lambda timeout_seconds=6.0: True)

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is True
    assert check["signals"] == ["exec_env_valid"]


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
    monkeypatch.setattr(mod, "_docker_sandbox_available", lambda timeout_seconds=5.0: True)

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is True
    assert check["signals"] == ["exec_env_valid"]


def test_exec_env_check_flags_invalid_agent_override(monkeypatch, tmp_path) -> None:
    _write_openclaw(
        tmp_path,
        {
            "tools": {"exec": {"host": "sandbox"}},
            "agents": {
                "defaults": {"sandbox": {"mode": "non-main"}},
                "list": [
                    {"id": "ops", "tools": {"profile": "full"}, "sandbox": {"mode": "off"}},
                    {"id": "notifier", "tools": {"profile": "messaging", "deny": ["exec"]}},
                ],
            },
            "acp": {"defaultAgent": "ops"},
        },
    )
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(mod, "_docker_sandbox_available", lambda timeout_seconds=5.0: True)

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is False
    assert check["signals"] == ["invalid_sandbox_mode"]
    assert "ops" in check["stderr"]


def test_exec_env_check_flags_unavailable_docker_sandbox(monkeypatch, tmp_path) -> None:
    _write_openclaw(
        tmp_path,
        {
            "tools": {"exec": {"host": "sandbox"}},
            "agents": {"defaults": {"sandbox": {"mode": "non-main"}}},
            "acp": {"defaultAgent": "ops"},
        },
    )
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(mod, "_docker_sandbox_available", lambda timeout_seconds=5.0: False)

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is False
    assert check["signals"] == ["sandbox_runtime_unavailable"]


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


def test_collect_runtime_status_uses_longer_timeout_for_production_gate(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "portfolio_maximizer.db").write_text("", encoding="utf-8")

    captured: list[tuple[str, float]] = []

    def fake_run_check(name, cmd, timeout_seconds, **kwargs):
        del cmd, kwargs
        captured.append((str(name), float(timeout_seconds)))
        return {
            "name": str(name),
            "ok": True,
            "returncode": 0,
            "duration_seconds": 0.0,
            "command": str(name),
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
    by_name = {name: timeout for name, timeout in captured}
    assert by_name["production_gate"] >= 240.0


def test_collect_runtime_status_strict_fails_on_inconclusive_allowed_gate(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")
    (project_root / "logs" / "audit_gate").mkdir(parents=True, exist_ok=True)
    (project_root / "logs" / "audit_gate" / "production_gate_latest.json").write_text(
        json.dumps(
            {
                "phase3_ready": True,
                "phase3_reason": "READY",
                "lift_inconclusive_allowed": True,
                "warmup_expired": False,
            }
        ),
        encoding="utf-8",
    )

    def fake_run_check(name, cmd, timeout_seconds, **kwargs):
        del cmd, timeout_seconds, kwargs
        if name == "production_gate":
            check = _base_ok_check(name)
            check["stdout"] = "Gate status    : PASS (semantics=INCONCLUSIVE_ALLOWED)"
            return check
        return _base_ok_check(name)

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "_run_check", fake_run_check)
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(mod, "_collect_observability_stack_check", lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"})

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=True)
    assert payload["status"] == "degraded"
    assert "strict_production_gate" in payload["failed_checks"]


def test_collect_runtime_status_strict_fails_on_stale_dashboard_payload(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    stale_dashboard = project_root / "visualizations" / "dashboard_data.json"
    stale_dashboard.parent.mkdir(parents=True, exist_ok=True)
    stale_dashboard.write_text(
        json.dumps(
            {
                "meta": {"generated_utc": "2020-01-01T00:00:00Z", "ts": "2020-01-01T00:00:00Z"},
                "pnl": {"absolute": 1.0, "pct": 0.01},
                "signals": [],
                "trade_events": [],
                "price_series": {},
                "robustness": {"overall_status": "OK", "freshness_status": "FRESH", "sidecar_age_minutes": {}},
                "live_denominator": {"status": "WAITING"},
                "quant_validation": {},
            }
        ),
        encoding="utf-8",
    )
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", stale_dashboard)
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(mod, "_collect_observability_stack_check", lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"})

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=True)
    assert payload["status"] == "degraded"
    assert "strict_dashboard_payload" in payload["failed_checks"]


def test_collect_runtime_status_strict_fails_on_stale_persistence_status(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    stale_status = project_root / "logs" / "persistence_manager_status.json"
    stale_status.parent.mkdir(parents=True, exist_ok=True)
    stale_status.write_text(
        json.dumps(
            {
                "timestamp_utc": "2020-01-01T00:00:00Z",
                "status_contract": {
                    "schema_version": 1,
                    "max_age_seconds": 60,
                    "reconciled_components": [
                        "dashboard.bridge",
                        "dashboard.http_server",
                        "dashboard.live_watcher",
                    ],
                },
                "dashboard": {
                    "bridge": {"running": True},
                    "http_server": {"running": True},
                    "live_watcher": {"running": True},
                    "warnings": [],
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", stale_status)
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(mod, "_collect_observability_stack_check", lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"})

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=True)
    assert payload["status"] == "degraded"
    assert "strict_persistence_manager_status" in payload["failed_checks"]


def test_collect_runtime_status_strict_fails_when_sidecar_stack_disagrees(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(
        mod,
        "_collect_observability_stack_check",
        lambda timeout_seconds: {
            "name": "observability_stack",
            "ok": False,
            "returncode": 1,
            "duration_seconds": 0.0,
            "command": "status_observability_stack.ps1 -Json -RequireCurrent",
            "stdout": "status=degraded",
            "stderr": "stack_status=degraded",
            "stack_status": "degraded",
        },
    )

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=True)
    assert payload["status"] == "degraded"
    assert "observability_stack" in payload["failed_checks"]
    assert "strict_observability_agreement" in payload["failed_checks"]
