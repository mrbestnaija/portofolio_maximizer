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
                "timestamp_utc": "2099-01-01T00:00:00Z",
                "phase3_ready": True,
                "phase3_reason": "READY",
                "lift_inconclusive_allowed": False,
                "warmup_expired": True,
            }
        ),
        encoding="utf-8",
    )


def _write_valid_canonical_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "gate": {
                    "freshness_status": {
                        "status": "fresh",
                        "age_minutes": 15.0,
                        "expected_max_age_minutes": 1440.0,
                        "last_expected_emission_utc": "2026-04-18T20:00:00Z",
                        "last_actual_emission_utc": "2026-04-18T19:45:00Z",
                    },
                    "warmup_state": {
                        "posture": "expired",
                        "deadline_utc": "2026-04-24T20:00:00Z",
                        "matched_needed": 0,
                    },
                    "trajectory_alarm": {
                        "active": False,
                        "days_to_deadline": 5.0,
                        "matched_needed": 0,
                        "expected_closes_remaining": 0.0,
                        "shortfall": 0.0,
                    },
                    "post_deadline_time_to_10_estimate": {
                        "status": "inactive",
                        "estimated_days": None,
                    },
                    "posture": "GENUINE_PASS",
                    "gate_artifact_age_minutes": 15.0,
                },
                "summary": {
                    "ann_roi_pct": 9.86,
                    "roi_ann_pct": 9.86,
                    "deployment_pct": 1.83,
                    "objective_score": 18.05,
                    "objective_valid": True,
                    "ngn_hurdle_pct": 28.0,
                    "gap_to_hurdle_pp": 18.14,
                    "evidence_health": "clean",
                    "unattended_gate": "PASS",
                    "unattended_ready": True,
                },
                "utilization": {"roi_ann_pct": 9.86, "deployment_pct": 1.83},
                "alpha_objective": {
                    "roi_ann_pct": 9.86,
                    "deployment_pct": 1.83,
                    "objective_score": 18.05,
                    "objective_valid": True,
                },
                "thin_linkage": {
                    "matched_current": 10,
                    "matched_needed": 0,
                    "trajectory_alarm": {
                        "active": False,
                        "days_to_deadline": 5.0,
                        "matched_needed": 0,
                        "expected_closes_remaining": 0.0,
                        "shortfall": 0.0,
                    },
                    "post_deadline_time_to_10_estimate": {
                        "status": "inactive",
                        "estimated_days": None,
                        "covered_lot_term_days": None,
                        "new_round_trip_term_days": None,
                        "covered_lots_remaining": 0,
                        "matched_needed": 0,
                        "covered_lot_daily_close_rate": 0.0,
                        "new_round_trip_daily_rate": 0.0,
                    },
                },
                "source_contract": {
                    "status": "clean",
                    "canonical_sources": [
                        {"metric": "closed_pnl", "source_file": "production_closed_trades", "query_or_key": "production_closed_trades"},
                        {"metric": "capital", "source_file": "portfolio_cash_state", "query_or_key": "portfolio_cash_state.initial_capital"},
                        {"metric": "open_risk", "source_file": "trade_executions", "query_or_key": "trade_executions WHERE is_close=0"},
                        {"metric": "utilization", "source_file": "scripts/compute_capital_utilization.py", "query_or_key": "scripts.compute_capital_utilization.compute_utilization"},
                    ],
                    "allowlisted_readers": [
                        "scripts/emit_canonical_snapshot.py",
                        "scripts/project_runtime_status.py",
                        "scripts/institutional_unattended_gate.py",
                    ],
                    "violations_found": [],
                    "scan_timestamp_utc": "2026-04-18T12:00:00Z",
                    "canonical": {"closed_pnl": "production_closed_trades"},
                    "ui_only": {"metrics_summary": "visualizations/performance/metrics_summary.json"},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_valid_runtime_status(path: Path, *, ts: str = "2099-01-01T00:00:00Z") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp_utc": ts,
                "status": "ok",
                "strict_mode": False,
                "check_count": 0,
                "failed_checks": [],
                "automation_ready_for_thin_linkage": True,
            }
        ),
        encoding="utf-8",
    )


def _write_valid_run_auto_trader_artifact(path: Path, *, quarantined: bool = False, ts: str = "2099-01-01T00:00:00Z") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "meta": {"ts": ts},
                "runtime_status": {
                    "eligibility_snapshot_status": "QUARANTINED" if quarantined else "READY",
                    "eligibility_snapshot_statuses": ["QUARANTINED"] if quarantined else ["READY"],
                    "quarantined_cycle_count": 1 if quarantined else 0,
                    "quarantine_reason": "Eligibility snapshot computation failed" if quarantined else None,
                },
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


def test_exec_env_check_resolves_project_home_fallback_when_wsl_home_missing(monkeypatch, tmp_path) -> None:
    project_root = (
        tmp_path
        / "Users"
        / "Bestman"
        / "personal_projects"
        / "portfolio_maximizer_v45"
        / "portfolio_maximizer_v45"
    )
    cfg_path = project_root.parents[2] / ".openclaw" / "openclaw.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tools": {"exec": {"host": "gateway"}},
        "agents": {"defaults": {"sandbox": {"mode": "non-main"}}},
        "acp": {"defaultAgent": "ops"},
    }
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path / "wsl-home")

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is True
    assert check["signals"] == ["exec_env_valid"]
    assert str(cfg_path) in check["command"]
    assert str(cfg_path) in check["checked_paths"]


def test_exec_env_check_flags_missing_openclaw_config(monkeypatch, tmp_path) -> None:
    project_root = (
        tmp_path
        / "Users"
        / "Bestman"
        / "personal_projects"
        / "portfolio_maximizer_v45"
        / "portfolio_maximizer_v45"
    )
    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path / "wsl-home")

    check = mod._openclaw_exec_environment_check()
    assert check["ok"] is False
    assert check["signals"] == ["openclaw_config_missing"]
    assert "openclaw config missing" in check["stderr"]


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


def test_collect_runtime_status_uses_cached_production_gate_artifact_without_live_run(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    canonical_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    _write_valid_canonical_snapshot(canonical_snapshot)

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
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", canonical_snapshot)
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
    assert all(name != "production_gate" for name, _ in captured)
    prod_check = next(check for check in payload["checks"] if check["name"] == "production_gate")
    assert prod_check["ok"] is True
    assert prod_check["command"] == f"validate {mod.DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH.name}"
    assert "semantics=" in prod_check["stdout"]
    assert prod_check["artifact_age_minutes"] is not None


def test_collect_runtime_status_fails_closed_when_production_gate_artifact_stale(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    _write_valid_canonical_snapshot(project_root / "logs" / "canonical_snapshot_latest.json")
    gate_artifact = project_root / "logs" / "audit_gate" / "production_gate_latest.json"
    payload = json.loads(gate_artifact.read_text(encoding="utf-8"))
    payload["timestamp_utc"] = "2020-01-01T00:00:00Z"
    gate_artifact.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", gate_artifact)
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", project_root / "logs" / "canonical_snapshot_latest.json")
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(
        mod,
        "_openclaw_exec_environment_check",
        lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]},
    )
    monkeypatch.setattr(
        mod,
        "_collect_observability_stack_check",
        lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"},
    )

    payload = mod.collect_runtime_status(timeout_seconds=1.0)
    assert payload["status"] == "degraded"
    assert "production_gate" in payload["failed_checks"]
    prod_check = next(check for check in payload["checks"] if check["name"] == "production_gate")
    assert prod_check["ok"] is False
    assert "production_gate_artifact_stale" in prod_check["stderr"]


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
    canonical_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    _write_valid_canonical_snapshot(canonical_snapshot)
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", canonical_snapshot)
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
    canonical_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    _write_valid_canonical_snapshot(canonical_snapshot)
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", canonical_snapshot)
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
    canonical_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    _write_valid_canonical_snapshot(canonical_snapshot)
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", canonical_snapshot)
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
    canonical_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    _write_valid_canonical_snapshot(canonical_snapshot)
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", canonical_snapshot)
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


def test_collect_runtime_status_strict_fails_on_bad_canonical_snapshot(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    bad_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    bad_snapshot.parent.mkdir(parents=True, exist_ok=True)
    bad_snapshot.write_text(json.dumps({"schema_version": 1, "summary": {}}), encoding="utf-8")

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", bad_snapshot)
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(mod, "_collect_observability_stack_check", lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"})

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=True)
    assert payload["status"] == "degraded"
    assert "strict_canonical_snapshot" in payload["failed_checks"]


def test_collect_runtime_status_strict_passes_with_clean_canonical_snapshot(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    clean_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    _write_valid_canonical_snapshot(clean_snapshot)

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", clean_snapshot)
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(mod, "_collect_observability_stack_check", lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"})

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=True)
    assert payload["status"] == "ok"
    strict_canonical = next(check for check in payload["checks"] if check["name"] == "strict_canonical_snapshot")
    assert strict_canonical["ok"] is True
    assert strict_canonical["evidence_health"] == "clean"
    assert strict_canonical["gate_artifact_age_minutes"] == 15.0


def test_collect_runtime_status_strict_fails_on_degraded_evidence_health(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    degraded_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    degraded_snapshot.parent.mkdir(parents=True, exist_ok=True)
    degraded_snapshot.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "gate": {
                    "freshness_status": {
                        "status": "fresh",
                        "age_minutes": 15.0,
                        "expected_max_age_minutes": 1440.0,
                    },
                    "warmup_state": {
                        "posture": "expired",
                        "deadline_utc": "2026-04-24T20:00:00Z",
                        "matched_needed": 0,
                    },
                    "trajectory_alarm": {"active": False},
                    "post_deadline_time_to_10_estimate": {"status": "inactive", "estimated_days": None},
                },
                "summary": {
                    "ann_roi_pct": 9.86,
                    "roi_ann_pct": 9.86,
                    "deployment_pct": 1.83,
                    "objective_score": 18.05,
                    "objective_valid": True,
                    "ngn_hurdle_pct": 28.0,
                    "gap_to_hurdle_pp": 18.14,
                    "evidence_health": "degraded",
                    "unattended_gate": "PASS",
                    "unattended_ready": True,
                },
                "utilization": {"roi_ann_pct": 9.86, "deployment_pct": 1.83},
                "alpha_objective": {
                    "roi_ann_pct": 9.86,
                    "deployment_pct": 1.83,
                    "objective_score": 18.05,
                    "objective_valid": True,
                },
                "thin_linkage": {"matched_current": 10, "matched_needed": 0},
                "source_contract": {
                    "status": "clean",
                    "canonical_sources": [
                        {"metric": "closed_pnl", "source_file": "production_closed_trades", "query_or_key": "production_closed_trades"}
                    ],
                    "allowlisted_readers": ["scripts/project_runtime_status.py"],
                    "violations_found": [],
                    "scan_timestamp_utc": "2026-04-18T12:00:00Z",
                    "canonical": {"closed_pnl": "production_closed_trades"},
                    "ui_only": {"metrics_summary": "visualizations/performance/metrics_summary.json"},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", degraded_snapshot)
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(mod, "_collect_observability_stack_check", lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"})

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=True)
    assert payload["status"] == "degraded"
    strict_canonical = next(check for check in payload["checks"] if check["name"] == "strict_canonical_snapshot")
    assert strict_canonical["ok"] is False
    assert "degraded_evidence_health:degraded" in strict_canonical["stderr"]


def test_collect_runtime_status_strict_fails_on_stale_gate_artifact_minutes(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    stale_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    stale_snapshot.parent.mkdir(parents=True, exist_ok=True)
    stale_snapshot.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "gate": {
                    "freshness_status": {
                        "status": "stale",
                        "age_minutes": 120.0,
                        "expected_max_age_minutes": 60.0,
                    },
                    "warmup_state": {
                        "posture": "expired",
                        "deadline_utc": "2026-04-24T20:00:00Z",
                        "matched_needed": 0,
                    },
                    "trajectory_alarm": {"active": False},
                    "post_deadline_time_to_10_estimate": {"status": "inactive", "estimated_days": None},
                },
                "summary": {
                    "ann_roi_pct": 9.86,
                    "roi_ann_pct": 9.86,
                    "deployment_pct": 1.83,
                    "objective_score": 18.05,
                    "objective_valid": True,
                    "ngn_hurdle_pct": 28.0,
                    "gap_to_hurdle_pp": 18.14,
                    "evidence_health": "clean",
                    "unattended_gate": "PASS",
                    "unattended_ready": True,
                },
                "utilization": {"roi_ann_pct": 9.86, "deployment_pct": 1.83},
                "alpha_objective": {
                    "roi_ann_pct": 9.86,
                    "deployment_pct": 1.83,
                    "objective_score": 18.05,
                    "objective_valid": True,
                },
                "thin_linkage": {"matched_current": 10, "matched_needed": 0},
                "source_contract": {
                    "status": "clean",
                    "canonical_sources": [
                        {"metric": "closed_pnl", "source_file": "production_closed_trades", "query_or_key": "production_closed_trades"}
                    ],
                    "allowlisted_readers": ["scripts/project_runtime_status.py"],
                    "violations_found": [],
                    "scan_timestamp_utc": "2026-04-18T12:00:00Z",
                    "canonical": {"closed_pnl": "production_closed_trades"},
                    "ui_only": {"metrics_summary": "visualizations/performance/metrics_summary.json"},
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", stale_snapshot)
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(mod, "_collect_observability_stack_check", lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"})

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=True)
    assert payload["status"] == "degraded"
    strict_canonical = next(check for check in payload["checks"] if check["name"] == "strict_canonical_snapshot")
    assert strict_canonical["ok"] is False
    assert "freshness_not_fresh:stale" in strict_canonical["stderr"]


def test_collect_runtime_status_exposes_automation_ready_for_thin_linkage(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    runtime_status_path = project_root / "logs" / "runtime_status_latest.json"
    run_auto_trader_path = project_root / "logs" / "automation" / "run_auto_trader_latest.json"
    _write_valid_runtime_status(runtime_status_path)
    _write_valid_run_auto_trader_artifact(run_auto_trader_path, quarantined=False)
    ready_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    _write_valid_canonical_snapshot(ready_snapshot)
    payload = json.loads(ready_snapshot.read_text(encoding="utf-8"))
    payload["thin_linkage"]["matched_current"] = 10
    payload["thin_linkage"]["matched_needed"] = 10
    payload["gate"]["warmup_state"]["matched_needed"] = 10
    payload["gate"]["freshness_status"]["status"] = "fresh"
    payload["summary"]["evidence_health"] = "clean"
    ready_snapshot.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_RUNTIME_STATUS_PATH", runtime_status_path)
    monkeypatch.setattr(mod, "DEFAULT_RUN_AUTO_TRADER_ARTIFACT_PATH", run_auto_trader_path)
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", ready_snapshot)
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(mod, "_collect_observability_stack_check", lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"})

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=False)
    assert payload["status"] == "ok"
    assert payload["automation_ready_for_thin_linkage"] is True
    detail = payload["automation_ready_for_thin_linkage_detail"]
    assert detail["ready"] is True
    assert detail["latest_cycle_status"] == "READY"
    assert detail["thin_linkage_matched_current"] == 10
    assert detail["thin_linkage_matched_needed"] == 10
    assert detail["runtime_status_source"].endswith("runtime_status_latest.json")


def test_collect_runtime_status_falls_back_to_run_auto_trader_meta_timestamp(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    run_auto_trader_path = project_root / "logs" / "automation" / "run_auto_trader_latest.json"
    _write_valid_run_auto_trader_artifact(run_auto_trader_path, quarantined=False)
    ready_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    _write_valid_canonical_snapshot(ready_snapshot)
    payload = json.loads(ready_snapshot.read_text(encoding="utf-8"))
    payload["thin_linkage"]["matched_current"] = 10
    payload["thin_linkage"]["matched_needed"] = 10
    payload["gate"]["warmup_state"]["matched_needed"] = 10
    payload["gate"]["freshness_status"]["status"] = "fresh"
    payload["summary"]["evidence_health"] = "clean"
    ready_snapshot.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_RUNTIME_STATUS_PATH", project_root / "logs" / "runtime_status_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_RUN_AUTO_TRADER_ARTIFACT_PATH", run_auto_trader_path)
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", ready_snapshot)
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(mod, "_collect_observability_stack_check", lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"})

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=False)
    detail = payload["automation_ready_for_thin_linkage_detail"]
    assert payload["automation_ready_for_thin_linkage"] is True
    assert detail["ready"] is True
    assert detail["runtime_status_source"].endswith("run_auto_trader_latest.json")
    assert detail["runtime_status_age_seconds"] is not None
    assert "runtime_status_missing" not in detail["reasons"]


def test_collect_runtime_status_marks_automation_ready_false_when_latest_cycle_quarantined(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "repo"
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "portfolio_maximizer.db").write_text("", encoding="utf-8")
    _write_valid_dashboard(project_root / "visualizations" / "dashboard_data.json")
    _write_valid_persistence_status(project_root / "logs" / "persistence_manager_status.json")
    _write_valid_production_gate_artifact(project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    runtime_status_path = project_root / "logs" / "runtime_status_latest.json"
    run_auto_trader_path = project_root / "logs" / "automation" / "run_auto_trader_latest.json"
    _write_valid_runtime_status(runtime_status_path)
    _write_valid_run_auto_trader_artifact(run_auto_trader_path, quarantined=True)
    ready_snapshot = project_root / "logs" / "canonical_snapshot_latest.json"
    _write_valid_canonical_snapshot(ready_snapshot)
    payload = json.loads(ready_snapshot.read_text(encoding="utf-8"))
    payload["thin_linkage"]["matched_current"] = 10
    payload["thin_linkage"]["matched_needed"] = 20
    payload["gate"]["warmup_state"]["matched_needed"] = 20
    payload["summary"]["evidence_health"] = "clean"
    ready_snapshot.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(mod, "DEFAULT_DASHBOARD_PATH", project_root / "visualizations" / "dashboard_data.json")
    monkeypatch.setattr(mod, "DEFAULT_PERSISTENCE_STATUS_PATH", project_root / "logs" / "persistence_manager_status.json")
    monkeypatch.setattr(mod, "DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH", project_root / "logs" / "audit_gate" / "production_gate_latest.json")
    monkeypatch.setattr(mod, "DEFAULT_RUNTIME_STATUS_PATH", runtime_status_path)
    monkeypatch.setattr(mod, "DEFAULT_RUN_AUTO_TRADER_ARTIFACT_PATH", run_auto_trader_path)
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", ready_snapshot)
    monkeypatch.setattr(mod, "_run_check", lambda name, cmd, timeout_seconds, **kwargs: _base_ok_check(name))
    monkeypatch.setattr(mod, "_openclaw_exec_environment_check", lambda: {**_base_ok_check("openclaw_exec_env"), "signals": ["exec_env_valid"]})
    monkeypatch.setattr(mod, "_collect_observability_stack_check", lambda timeout_seconds: {**_base_ok_check("observability_stack"), "stack_status": "ok"})

    payload = mod.collect_runtime_status(timeout_seconds=1.0, strict=False)
    assert payload["automation_ready_for_thin_linkage"] is False
    detail = payload["automation_ready_for_thin_linkage_detail"]
    assert detail["ready"] is False
    assert "latest_cycle_quarantined" in detail["reasons"]
