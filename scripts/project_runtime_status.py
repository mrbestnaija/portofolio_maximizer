#!/usr/bin/env python3
"""
PowerShell-safe runtime status snapshot for Portfolio Maximizer.

Why this exists:
- OpenClaw `exec` runs in Windows PowerShell on some hosts.
- PowerShell 5 does not support `&&`, so chained shell commands can fail.
- This script runs key checks sequentially from Python without shell chaining.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import dashboard_db_bridge as dashboard_bridge

try:
    from scripts.production_gate_contract import (
        gate_semantics_status as _gate_semantics_status,
        phase3_strict_ready as _phase3_strict_ready,
    )
except Exception:  # pragma: no cover - script execution path fallback
    from production_gate_contract import (  # type: ignore
        gate_semantics_status as _gate_semantics_status,
        phase3_strict_ready as _phase3_strict_ready,
    )

VALID_EXEC_HOSTS = {"sandbox", "gateway", "node"}
VALID_SANDBOX_MODES_FOR_SANDBOX_HOST = {"non-main", "all"}
DEFAULT_DASHBOARD_PATH = PROJECT_ROOT / "visualizations" / "dashboard_data.json"
DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH = PROJECT_ROOT / "logs" / "audit_gate" / "production_gate_latest.json"
DEFAULT_PERSISTENCE_STATUS_PATH = PROJECT_ROOT / "logs" / "persistence_manager_status.json"
_PRODUCTION_GATE_SEMANTICS_RE = re.compile(r"semantics=([A-Z_]+)")


def _trim(text: str, max_chars: int = 1400) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "\n...[truncated]..."


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _agent_allows_exec(agent: dict[str, Any]) -> bool:
    tools = _as_dict(agent.get("tools"))
    deny = {str(item).strip().lower() for item in _as_list(tools.get("deny")) if str(item).strip()}
    if "exec" in deny or "group:runtime" in deny:
        return False

    allow = {str(item).strip().lower() for item in _as_list(tools.get("allow")) if str(item).strip()}
    if allow:
        return "exec" in allow or "group:runtime" in allow

    profile = str(tools.get("profile") or "").strip().lower()
    return profile != "messaging"


def _invalid_sandbox_override_agents(cfg: dict[str, Any]) -> list[str]:
    tools = _as_dict(cfg.get("tools"))
    exec_cfg = _as_dict(tools.get("exec"))
    host = str(exec_cfg.get("host") or "").strip().lower()
    if host != "sandbox":
        return []

    agents = _as_dict(cfg.get("agents"))
    invalid: list[str] = []
    for agent in _as_list(agents.get("list")):
        if not isinstance(agent, dict) or not _agent_allows_exec(agent):
            continue
        sandbox = _as_dict(agent.get("sandbox"))
        mode = str(sandbox.get("mode") or "").strip().lower()
        if mode and mode not in VALID_SANDBOX_MODES_FOR_SANDBOX_HOST:
            aid = str(agent.get("id") or "?").strip() or "?"
            invalid.append(aid)
    return invalid


def _docker_sandbox_available(timeout_seconds: float = 5.0) -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return int(proc.returncode) == 0


def _node_host_available(timeout_seconds: float = 6.0) -> bool:
    """Return True if at least one OpenClaw node is paired and reachable.

    Calls ``openclaw nodes list --json`` with a short timeout. Any failure
    (missing CLI, non-zero exit, empty list, JSON parse error) is treated as
    unavailable — matching the pattern of ``_docker_sandbox_available``.
    """
    try:
        proc = subprocess.run(
            ["openclaw", "nodes", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    if int(proc.returncode) != 0:
        return False
    try:
        payload = json.loads(proc.stdout or "")
    except Exception:
        return False
    # Accept a top-level list or a dict with a "nodes" key.
    if isinstance(payload, list):
        nodes = payload
    elif isinstance(payload, dict):
        nodes = payload.get("nodes") or []
    else:
        return False
    return bool(nodes)


def _read_openclaw_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _read_json_file(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.exists():
        return {}, f"missing:{path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {}, f"unreadable:{path.name}:{exc}"
    return (payload, None) if isinstance(payload, dict) else ({}, f"invalid:{path.name}:root_not_object")


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _make_check(
    name: str,
    *,
    ok: bool,
    returncode: int,
    command: str,
    stdout: str = "",
    stderr: str = "",
    duration_seconds: float = 0.0,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "ok": ok,
        "returncode": int(returncode),
        "duration_seconds": round(float(duration_seconds), 3),
        "command": command,
        "stdout": _trim(stdout),
        "stderr": _trim(stderr),
    }
    payload.update(extra)
    return payload


def _run_check(
    name: str,
    cmd: list[str],
    timeout_seconds: float,
    *,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.time()
    env = dict(os.environ)
    for key, value in (env_overrides or {}).items():
        text = str(value or "").strip()
        if text:
            env[key] = text
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            env=env,
        )
        duration = time.time() - start
        return {
            "name": name,
            "ok": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "duration_seconds": round(duration, 3),
            "command": " ".join(cmd),
            "stdout": _trim(proc.stdout or ""),
            "stderr": _trim(proc.stderr or ""),
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start
        return {
            "name": name,
            "ok": False,
            "returncode": 124,
            "duration_seconds": round(duration, 3),
            "command": " ".join(cmd),
            "stdout": _trim(exc.stdout if isinstance(exc.stdout, str) else ""),
            "stderr": _trim((exc.stderr if isinstance(exc.stderr, str) else "") or "timeout"),
        }
    except FileNotFoundError as exc:
        duration = time.time() - start
        return {
            "name": name,
            "ok": False,
            "returncode": 127,
            "duration_seconds": round(duration, 3),
            "command": " ".join(cmd),
            "stdout": "",
            "stderr": _trim(str(exc)),
        }


def _extract_production_gate_semantics(
    check: dict[str, Any],
    artifact: dict[str, Any],
) -> str:
    artifact_semantics = _gate_semantics_status(artifact)
    if artifact_semantics:
        return artifact_semantics
    for field in ("stdout", "stderr"):
        text = str(check.get(field) or "")
        match = _PRODUCTION_GATE_SEMANTICS_RE.search(text)
        if match:
            return match.group(1)
    if (
        bool(check.get("ok"))
        and bool(artifact.get("phase3_ready"))
        and bool(artifact.get("lift_inconclusive_allowed"))
        and not bool(artifact.get("warmup_expired", True))
    ):
        return "INCONCLUSIVE_ALLOWED"
    if bool(check.get("ok")) and bool(_phase3_strict_ready(artifact)):
        return "READY"
    return ""


def _strict_production_gate_check(base_check: dict[str, Any]) -> dict[str, Any]:
    artifact, err = _read_json_file(DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH)
    semantics = _extract_production_gate_semantics(base_check, artifact)
    errors: list[str] = []
    if not bool(base_check.get("ok")):
        errors.append("base_check_failed")
    if err:
        errors.append(err)
    if semantics == "INCONCLUSIVE_ALLOWED":
        errors.append("warmup_only_pass")
    return _make_check(
        "strict_production_gate",
        ok=not errors,
        returncode=0 if not errors else 1,
        command=f"validate {DEFAULT_PRODUCTION_GATE_ARTIFACT_PATH.name}",
        stdout=f"semantics={semantics or 'UNKNOWN'}",
        stderr="; ".join(errors),
        semantics=semantics or "UNKNOWN",
        phase3_ready=bool(_phase3_strict_ready(artifact)),
        warmup_expired=bool(artifact.get("warmup_expired", True)),
    )


def _strict_dashboard_payload_check() -> dict[str, Any]:
    report = dashboard_bridge.validate_dashboard_payload_file(DEFAULT_DASHBOARD_PATH)
    return _make_check(
        "strict_dashboard_payload",
        ok=bool(report.get("ok")),
        returncode=0 if bool(report.get("ok")) else 1,
        command=f"validate {DEFAULT_DASHBOARD_PATH.name}",
        stdout=(
            f"generated_utc={report.get('generated_utc')} age_seconds={report.get('age_seconds')} "
            f"threshold_seconds={report.get('freshness_threshold_seconds')}"
        ),
        stderr="; ".join(str(item) for item in report.get("errors", [])),
        age_seconds=report.get("age_seconds"),
        freshness_threshold_seconds=report.get("freshness_threshold_seconds"),
        missing_keys=report.get("missing_keys", []),
    )


def _strict_persistence_manager_status_check() -> dict[str, Any]:
    payload, err = _read_json_file(DEFAULT_PERSISTENCE_STATUS_PATH)
    errors: list[str] = []
    if err:
        errors.append(err)
    contract = payload.get("status_contract") if isinstance(payload.get("status_contract"), dict) else {}
    if not contract:
        errors.append("missing_status_contract")
    max_age_seconds = 0
    try:
        max_age_seconds = max(int(contract.get("max_age_seconds") or 0), 0)
    except Exception:
        max_age_seconds = 0
    if max_age_seconds <= 0:
        errors.append("invalid_status_contract.max_age_seconds")

    generated = _parse_iso_datetime(payload.get("timestamp_utc"))
    if generated is None:
        errors.append("missing_timestamp_utc")
    age_seconds: float | None = None
    if generated is not None:
        age_seconds = max(0.0, (datetime.now(timezone.utc) - generated).total_seconds())
        if max_age_seconds > 0 and age_seconds > max_age_seconds:
            errors.append(
                f"stale_persistence_status:age_seconds={age_seconds:.1f}:threshold_seconds={max_age_seconds}"
            )

    dashboard = payload.get("dashboard") if isinstance(payload.get("dashboard"), dict) else {}
    bridge = dashboard.get("bridge") if isinstance(dashboard.get("bridge"), dict) else {}
    server = dashboard.get("http_server") if isinstance(dashboard.get("http_server"), dict) else {}
    watcher = dashboard.get("live_watcher") if isinstance(dashboard.get("live_watcher"), dict) else {}
    required_components = {
        "bridge": bool(bridge.get("running")),
        "http_server": bool(server.get("running")),
        "live_watcher": bool(watcher.get("running")),
    }
    for name, running in required_components.items():
        if not running:
            errors.append(f"{name}_not_running")

    warnings = dashboard.get("warnings") if isinstance(dashboard.get("warnings"), list) else []

    return _make_check(
        "strict_persistence_manager_status",
        ok=not errors,
        returncode=0 if not errors else 1,
        command=f"validate {DEFAULT_PERSISTENCE_STATUS_PATH.name}",
        stdout=(
            f"timestamp_utc={payload.get('timestamp_utc')} age_seconds={round(age_seconds, 3) if age_seconds is not None else None} "
            f"max_age_seconds={max_age_seconds}"
        ),
        stderr="; ".join(errors),
        age_seconds=round(age_seconds, 3) if age_seconds is not None else None,
        max_age_seconds=max_age_seconds,
        required_components=required_components,
        warnings=[str(item) for item in warnings if str(item).strip()],
    )


def _collect_observability_stack_check(*, timeout_seconds: float) -> dict[str, Any]:
    cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(PROJECT_ROOT / "scripts" / "status_observability_stack.ps1"),
        "-Json",
        "-RequireCurrent",
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
        duration = time.time() - start
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start
        return _make_check(
            "observability_stack",
            ok=False,
            returncode=124,
            command=" ".join(cmd),
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr if isinstance(exc.stderr, str) else "") or "timeout",
            duration_seconds=duration,
        )
    except FileNotFoundError as exc:
        duration = time.time() - start
        return _make_check(
            "observability_stack",
            ok=False,
            returncode=127,
            command=" ".join(cmd),
            stderr=str(exc),
            duration_seconds=duration,
        )

    payload: dict[str, Any] = {}
    parse_error = ""
    try:
        payload = json.loads(proc.stdout or "")
        if not isinstance(payload, dict):
            payload = {}
            parse_error = "stack_payload_not_object"
    except Exception as exc:
        parse_error = f"stack_payload_unreadable:{exc}"
    stack_status = str(payload.get("status") or "").strip().lower()
    errors: list[str] = []
    if parse_error:
        errors.append(parse_error)
    if stack_status != "ok":
        errors.append(f"stack_status={stack_status or 'unknown'}")
    if int(proc.returncode) != 0:
        errors.append(f"returncode={int(proc.returncode)}")
    if str(proc.stderr or "").strip():
        errors.append(_trim(proc.stderr or "", 400))
    return _make_check(
        "observability_stack",
        ok=not errors,
        returncode=int(proc.returncode),
        command=" ".join(cmd),
        stdout=(
            f"status={stack_status or 'unknown'} "
            f"legacy_sidecar_count={payload.get('legacy_sidecar_count')} "
            f"optional_missing_count={payload.get('optional_missing_count')}"
        ),
        stderr="; ".join(errors),
        duration_seconds=duration,
        stack_status=stack_status or "unknown",
        legacy_sidecar_count=int(payload.get("legacy_sidecar_count") or 0),
        optional_missing_count=int(payload.get("optional_missing_count") or 0),
    )


def _strict_observability_agreement_check(
    *,
    base_checks: list[dict[str, Any]],
    strict_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    base_ok = all(bool(check.get("ok")) for check in base_checks)
    stack_check = next((check for check in strict_checks if check.get("name") == "observability_stack"), None)
    stack_ok = bool(stack_check and stack_check.get("ok"))
    pmx_contract_failures = [
        str(check.get("name"))
        for check in strict_checks
        if check.get("name") != "observability_stack" and not bool(check.get("ok"))
    ]
    mismatches: list[str] = []
    if base_ok and not stack_ok:
        mismatches.append("runtime_ok_but_sidecar_stack_not_green")
    if stack_ok and pmx_contract_failures:
        mismatches.append("sidecar_stack_green_but_pmx_contract_not_green")
    if not base_ok and stack_ok:
        mismatches.append("sidecar_stack_green_but_base_runtime_degraded")
    if not mismatches and (not base_ok or not stack_ok or pmx_contract_failures):
        mismatches.append("strict_green_requirements_not_met")

    return _make_check(
        "strict_observability_agreement",
        ok=not mismatches,
        returncode=0 if not mismatches else 1,
        command="reconcile strict runtime vs sidecar stack",
        stdout=(
            f"base_runtime_ok={int(base_ok)} sidecar_stack_ok={int(stack_ok)} "
            f"pmx_contract_failures={','.join(pmx_contract_failures) if pmx_contract_failures else 'none'}"
        ),
        stderr="; ".join(mismatches),
    )


def _openclaw_exec_environment_check() -> dict[str, Any]:
    cfg_path = Path.home() / ".openclaw" / "openclaw.json"
    check = {
        "name": "openclaw_exec_env",
        "ok": False,
        "returncode": 1,
        "duration_seconds": 0.0,
        "command": f"validate {cfg_path}",
        "stdout": "",
        "stderr": "",
        "signals": [],
    }
    if not cfg_path.exists():
        check["stderr"] = f"openclaw config missing: {cfg_path}"
        return check
    try:
        cfg = _read_openclaw_json(cfg_path)
    except Exception as exc:
        check["stderr"] = f"openclaw config unreadable: {exc}"
        return check

    tools = cfg.get("tools", {}) if isinstance(cfg.get("tools"), dict) else {}
    exec_cfg = tools.get("exec", {}) if isinstance(tools.get("exec"), dict) else {}
    host = str(exec_cfg.get("host") or "").strip().lower()
    if host not in VALID_EXEC_HOSTS:
        check["signals"] = ["invalid_exec_host"]
        check["stderr"] = "tools.exec.host missing/invalid"
        return check

    agents = cfg.get("agents", {}) if isinstance(cfg.get("agents"), dict) else {}
    defaults = agents.get("defaults", {}) if isinstance(agents.get("defaults"), dict) else {}
    sandbox = defaults.get("sandbox", {}) if isinstance(defaults.get("sandbox"), dict) else {}
    sandbox_mode = str(sandbox.get("mode") or "").strip().lower()
    if host == "sandbox" and sandbox_mode not in VALID_SANDBOX_MODES_FOR_SANDBOX_HOST:
        check["signals"] = ["invalid_sandbox_mode"]
        check["stderr"] = "agents.defaults.sandbox.mode invalid for sandbox host"
        return check

    if host == "sandbox" and not _docker_sandbox_available():
        check["signals"] = ["sandbox_runtime_unavailable"]
        check["stderr"] = "docker daemon unavailable for sandbox host"
        return check

    if host == "node" and not _node_host_available():
        check["signals"] = ["node_host_unavailable"]
        check["stderr"] = (
            "tools.exec.host=node but no paired node found via 'openclaw nodes list'; "
            "run: python scripts/enforce_openclaw_exec_environment.py --host gateway"
        )
        return check

    invalid_override_agents = _invalid_sandbox_override_agents(cfg)
    if invalid_override_agents:
        check["signals"] = ["invalid_sandbox_mode"]
        check["stderr"] = (
            "sandbox disabled by agent override(s): " + ",".join(sorted(invalid_override_agents))
        )
        return check

    acp = cfg.get("acp", {}) if isinstance(cfg.get("acp"), dict) else {}
    default_agent = str(acp.get("defaultAgent") or "").strip()
    if not default_agent:
        check["signals"] = ["missing_acp_default_agent"]
        check["stderr"] = "acp.defaultAgent missing"
        return check

    check["ok"] = True
    check["returncode"] = 0
    check["signals"] = ["exec_env_valid"]
    check["stdout"] = (
        f"host={host} sandbox_mode={sandbox_mode or '<unset>'} "
        f"acp.defaultAgent={default_agent} invalid_agent_overrides=0"
    )
    return check


def collect_runtime_status(*, timeout_seconds: float = 90.0, strict: bool = False) -> dict[str, Any]:
    py = sys.executable
    db_path = PROJECT_ROOT / "data" / "portfolio_maximizer.db"
    whitelist_ids = str(
        os.getenv("INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS", "66")
    ).strip() or "66"
    production_gate_timeout_seconds = max(float(timeout_seconds), float(os.getenv("PROJECT_RUNTIME_STATUS_PRODUCTION_GATE_TIMEOUT_SECONDS", "240")))

    checks: list[dict[str, Any]] = []

    if db_path.exists():
        checks.append(
            _run_check(
                "pnl_integrity",
                [py, "-m", "integrity.pnl_integrity_enforcer", "--db", str(db_path)],
                timeout_seconds=timeout_seconds,
                env_overrides={
                    "INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS": whitelist_ids,
                },
            )
        )
    else:
        checks.append(
            {
                "name": "pnl_integrity",
                "ok": False,
                "returncode": 2,
                "duration_seconds": 0.0,
                "command": f"{py} -m integrity.pnl_integrity_enforcer --db {db_path}",
                "stdout": "",
                "stderr": "Database not found; skipped integrity enforcer.",
            }
        )

    checks.append(
        _run_check(
            "production_gate",
            [
                py,
                str(PROJECT_ROOT / "scripts" / "production_audit_gate.py"),
                "--unattended-profile",
            ],
            timeout_seconds=production_gate_timeout_seconds,
        )
    )
    checks.append(
        _run_check(
            "error_monitor",
            [py, str(PROJECT_ROOT / "scripts" / "error_monitor.py"), "--check"],
            timeout_seconds=timeout_seconds,
        )
    )
    checks.append(_openclaw_exec_environment_check())

    if strict:
        production_gate_check = next(
            (check for check in checks if check.get("name") == "production_gate"),
            {},
        )
        strict_checks = [
            _strict_production_gate_check(production_gate_check),
            _strict_dashboard_payload_check(),
            _strict_persistence_manager_status_check(),
            _collect_observability_stack_check(timeout_seconds=timeout_seconds),
        ]
        strict_checks.append(
            _strict_observability_agreement_check(
                base_checks=checks,
                strict_checks=strict_checks,
            )
        )
        checks.extend(strict_checks)

    failed = [c["name"] for c in checks if not bool(c.get("ok"))]
    return {
        "status": "ok" if not failed else "degraded",
        "failed_checks": failed,
        "check_count": len(checks),
        "strict_mode": bool(strict),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks,
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="PowerShell-safe PMX runtime status snapshot")
    p.add_argument("--timeout-seconds", type=float, default=90.0, help="Per-check timeout")
    p.add_argument("--strict", action="store_true", help="Enforce strict-green observability contract.")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = p.parse_args(argv)

    payload = collect_runtime_status(timeout_seconds=float(args.timeout_seconds), strict=bool(args.strict))
    if args.pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, separators=(",", ":")))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
