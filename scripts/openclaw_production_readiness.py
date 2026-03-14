#!/usr/bin/env python3
"""
Local-only OpenClaw production-readiness snapshot for Portfolio Maximizer.

This script answers one operational question in a tool-friendly way:
"What is blocking a safe, production-ready OpenClaw + PMX setup right now?"

It combines:
- PMX capital-readiness verdicts
- OpenClaw exec-environment validity
- OpenClaw gateway/channel regression health
- Local-model availability and local-only enforcement
- Security posture for unattended agent turns

The output is intentionally JSON-stable so the OpenClaw bridge can consume it
without scraping human-readable logs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from etl.secret_loader import bootstrap_dotenv

    bootstrap_dotenv()
except Exception:
    pass

from scripts import capital_readiness_check as capital_mod
from scripts import openclaw_regression_gate as regression_mod
from scripts import project_runtime_status as runtime_mod


FALSEY_ENV_VALUES = {"0", "false", "no", "off"}
DEFAULT_APPROVAL_TOKEN = "PMX_APPROVE_HIGH_RISK"
DEFAULT_GATE_ARTIFACT = PROJECT_ROOT / "logs" / "gate_status_latest.json"
DEFAULT_PRODUCTION_GATE_ARTIFACT = PROJECT_ROOT / "logs" / "audit_gate" / "production_gate_latest.json"
DEFAULT_OPENCLAW_JSON_RELATIVE = Path(".openclaw") / "openclaw.json"


def _default_openclaw_json_path() -> Path:
    return Path.home() / DEFAULT_OPENCLAW_JSON_RELATIVE


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _tail_lines(text: str, *, limit: int = 24) -> list[str]:
    lines = [str(line) for line in str(text or "").splitlines() if str(line).strip()]
    if len(lines) <= limit:
        return lines
    return lines[-limit:]


def _call_quietly(fn, /, *args, **kwargs) -> tuple[Any, list[str], list[str]]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        result = fn(*args, **kwargs)
    return result, _tail_lines(stdout_buf.getvalue()), _tail_lines(stderr_buf.getvalue())


def _env_enabled(name: str, *, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw not in FALSEY_ENV_VALUES


def _approval_token_value() -> str:
    token = str(os.getenv("OPENCLAW_AUTONOMY_APPROVAL_TOKEN", "")).strip()
    return token or DEFAULT_APPROVAL_TOKEN


def _issue(*, source: str, code: str, detail: str) -> dict[str, str]:
    return {
        "source": str(source or "").strip() or "unknown",
        "code": str(code or "").strip() or "unknown",
        "detail": str(detail or "").strip() or "unspecified",
    }


def _dedupe_issues(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("source") or "").strip(),
            str(row.get("code") or "").strip(),
            str(row.get("detail") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "source": key[0] or "unknown",
                "code": key[1] or "unknown",
                "detail": key[2] or "unspecified",
            }
        )
    return out


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _artifact_age_hours(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        return round((time.time() - path.stat().st_mtime) / 3600.0, 2)
    except Exception:
        return None


def _artifact_mtime_utc(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _tool_capable_model_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    return "qwen3" in low or "qwen2.5" in low


def _tool_capable_model_ref(ref: str) -> bool:
    low = str(ref or "").strip().lower()
    if not low.startswith("ollama/"):
        return False
    return _tool_capable_model_name(low)


def _normalize_ollama_base_url(base_url: str) -> str:
    raw = str(base_url or "").strip() or "http://127.0.0.1:11434/v1"
    text = raw.rstrip("/")
    if text.lower().endswith("/v1"):
        return text[: -len("/v1")]
    return text


def _discover_ollama_models(base_url: str, timeout_seconds: float = 2.0) -> list[str]:
    api_base = _normalize_ollama_base_url(base_url)
    url = f"{api_base}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=max(0.5, float(timeout_seconds))) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)
    except (OSError, urllib.error.URLError, ValueError):
        return []

    models = payload.get("models") if isinstance(payload, dict) else None
    out: list[str] = []
    if isinstance(models, list):
        for row in models:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if name:
                out.append(name)
    return out


def _gate_artifact_snapshot(path: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "timestamp_utc": None,
        "overall_passed": None,
        "phase3_ready": None,
        "phase3_reason": "",
        "age_hours": None,
        "mtime_utc": _artifact_mtime_utc(path),
        "status_stage": "",
        "skipped_optional_gates": None,
        "max_skipped_optional_gates": None,
        "skipped_gate_labels": [],
    }
    if not path.exists():
        return snapshot

    payload = _read_json_file(path)
    if not payload:
        snapshot["parse_error"] = True
        return snapshot

    snapshot["age_hours"] = _artifact_age_hours(path)
    snapshot["timestamp_utc"] = payload.get("timestamp_utc")
    snapshot["overall_passed"] = payload.get("overall_passed")
    snapshot["phase3_ready"] = payload.get("phase3_ready")
    snapshot["phase3_reason"] = str(payload.get("phase3_reason") or "")
    snapshot["status_stage"] = str(payload.get("status_stage") or "")
    snapshot["skipped_optional_gates"] = payload.get("skipped_optional_gates")
    snapshot["max_skipped_optional_gates"] = payload.get("max_skipped_optional_gates")
    skipped = payload.get("skipped_gate_labels")
    if isinstance(skipped, list):
        snapshot["skipped_gate_labels"] = [str(x) for x in skipped if str(x).strip()]
    return snapshot


def _production_gate_snapshot(path: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "phase3_ready": None,
        "phase3_reason": "",
        "warmup_expired": None,
        "gate_semantics_status": "",
        "pass_semantics_version": None,
        "age_hours": None,
        "mtime_utc": _artifact_mtime_utc(path),
        "timestamp_utc": None,
        "parse_error": False,
    }
    if not path.exists():
        return snapshot

    payload = _read_json_file(path)
    if not payload:
        snapshot["parse_error"] = True
        return snapshot

    gate_block = (
        payload.get("production_profitability_gate")
        if isinstance(payload.get("production_profitability_gate"), dict)
        else {}
    )
    snapshot["age_hours"] = _artifact_age_hours(path)
    snapshot["timestamp_utc"] = payload.get("timestamp_utc")
    snapshot["phase3_ready"] = payload.get("phase3_ready")
    snapshot["phase3_reason"] = str(payload.get("phase3_reason") or "")
    snapshot["warmup_expired"] = payload.get("warmup_expired")
    snapshot["gate_semantics_status"] = str(gate_block.get("gate_semantics_status") or "")
    snapshot["pass_semantics_version"] = payload.get("pass_semantics_version")
    return snapshot


def _refresh_production_gate_artifact(
    *,
    artifact_path: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    blockers: list[dict[str, str]] = []
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    before_mtime = artifact_path.stat().st_mtime if artifact_path.exists() else None
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "production_audit_gate.py"),
        "--unattended-profile",
        "--output-json",
        str(artifact_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(10.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        blockers.append(
            _issue(
                source="gate_truth",
                code="production_gate_refresh_timeout",
                detail=f"Timed out refreshing production gate artifact after {float(timeout_seconds):.1f}s.",
            )
        )
        return {
            "attempted": True,
            "ok": False,
            "returncode": 124,
            "command": cmd,
            "stdout_tail": _tail_lines(exc.stdout if isinstance(exc.stdout, str) else ""),
            "stderr_tail": _tail_lines((exc.stderr if isinstance(exc.stderr, str) else "") or "timeout"),
        }, blockers
    except FileNotFoundError as exc:
        blockers.append(
            _issue(
                source="gate_truth",
                code="production_gate_refresh_missing",
                detail=f"Unable to launch production_audit_gate.py: {exc}",
            )
        )
        return {
            "attempted": True,
            "ok": False,
            "returncode": 127,
            "command": cmd,
            "stdout_tail": [],
            "stderr_tail": [str(exc)],
        }, blockers

    snapshot = _production_gate_snapshot(artifact_path)
    refreshed_after_run = (
        artifact_path.exists()
        and not bool(snapshot.get("parse_error"))
        and (
            before_mtime is None
            or artifact_path.stat().st_mtime >= float(before_mtime)
        )
    )
    ok = refreshed_after_run and int(proc.returncode) in {0, 1}
    if not ok:
        blockers.append(
            _issue(
                source="gate_truth",
                code="production_gate_refresh_failed",
                detail="Refreshing production_gate_latest.json failed or did not produce a parseable artifact.",
            )
        )
    return {
        "attempted": True,
        "ok": ok,
        "returncode": int(proc.returncode),
        "command": cmd,
        "stdout_tail": _tail_lines(proc.stdout or ""),
        "stderr_tail": _tail_lines(proc.stderr or ""),
        "artifact_refreshed": refreshed_after_run,
    }, blockers


def _openclaw_model_posture(config_path: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    snapshot: dict[str, Any] = {
        "config_path": str(config_path),
        "config_present": config_path.exists(),
        "local_only_effective": _env_enabled("OPENCLAW_LOCAL_ONLY", default=True),
        "remote_providers": [],
        "primary_model": "",
        "fallback_models": [],
        "allowlist_models": [],
        "image_primary": "",
        "ollama_base_url": "",
        "ollama_reachable": False,
        "discovered_models": [],
        "tool_model_configured": False,
        "tool_model_available": False,
    }
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not config_path.exists():
        blockers.append(
            _issue(
                source="openclaw_models",
                code="openclaw_config_missing",
                detail=f"OpenClaw config missing: {config_path}",
            )
        )
        return snapshot, blockers, warnings

    payload = _read_json_file(config_path)
    if not payload:
        blockers.append(
            _issue(
                source="openclaw_models",
                code="openclaw_config_unreadable",
                detail=f"OpenClaw config unreadable: {config_path}",
            )
        )
        return snapshot, blockers, warnings

    providers = payload.get("models", {}).get("providers", {})
    providers = providers if isinstance(providers, dict) else {}
    defaults = payload.get("agents", {}).get("defaults", {})
    defaults = defaults if isinstance(defaults, dict) else {}
    model_defaults = defaults.get("model", {})
    model_defaults = model_defaults if isinstance(model_defaults, dict) else {}
    image_defaults = defaults.get("imageModel", {})
    image_defaults = image_defaults if isinstance(image_defaults, dict) else {}

    primary_model = str(model_defaults.get("primary") or "").strip()
    fallbacks_raw = model_defaults.get("fallbacks")
    fallback_models = [str(x) for x in fallbacks_raw if str(x).strip()] if isinstance(fallbacks_raw, list) else []

    allowlist_raw = defaults.get("models")
    allowlist_models: list[str] = []
    if isinstance(allowlist_raw, dict):
        allowlist_models = [str(k) for k in allowlist_raw.keys() if str(k).strip()]
    elif isinstance(allowlist_raw, list):
        allowlist_models = [str(x) for x in allowlist_raw if str(x).strip()]

    image_primary = str(image_defaults.get("primary") or "").strip()
    remote_providers = sorted([str(k) for k in providers.keys() if str(k).strip() and str(k).strip() != "ollama"])
    remote_fallbacks = [row for row in fallback_models if not row.startswith("ollama/")]
    remote_allowlist = [row for row in allowlist_models if not row.startswith("ollama/")]

    ollama_provider = providers.get("ollama") if isinstance(providers.get("ollama"), dict) else {}
    ollama_base_url = (
        str(os.getenv("OPENCLAW_OLLAMA_BASE_URL") or "").strip()
        or str(os.getenv("OLLAMA_HOST") or "").strip()
        or str(ollama_provider.get("baseUrl") or "").strip()
        or "http://127.0.0.1:11434/v1"
    )
    discovered_models = _discover_ollama_models(ollama_base_url)

    tool_model_configured = any(
        _tool_capable_model_ref(ref)
        for ref in [primary_model, *fallback_models, *allowlist_models]
    )
    tool_model_available = any(_tool_capable_model_name(name) for name in discovered_models)

    snapshot.update(
        {
            "remote_providers": remote_providers,
            "primary_model": primary_model,
            "fallback_models": fallback_models,
            "allowlist_models": allowlist_models,
            "image_primary": image_primary,
            "ollama_base_url": ollama_base_url,
            "ollama_reachable": bool(discovered_models),
            "discovered_models": discovered_models[:12],
            "tool_model_configured": tool_model_configured,
            "tool_model_available": tool_model_available,
        }
    )

    if not snapshot["local_only_effective"]:
        blockers.append(
            _issue(
                source="openclaw_models",
                code="openclaw_local_only_disabled",
                detail="OPENCLAW_LOCAL_ONLY is disabled; remote/cloud model exposure is allowed.",
            )
        )
    if remote_providers:
        blockers.append(
            _issue(
                source="openclaw_models",
                code="openclaw_remote_providers",
                detail=f"Remote providers configured: {', '.join(remote_providers)}",
            )
        )
    if not primary_model:
        blockers.append(
            _issue(
                source="openclaw_models",
                code="openclaw_primary_missing",
                detail="agents.defaults.model.primary is not configured.",
            )
        )
    elif not primary_model.startswith("ollama/"):
        blockers.append(
            _issue(
                source="openclaw_models",
                code="openclaw_primary_remote",
                detail=f"Primary model is not local-only: {primary_model}",
            )
        )
    if remote_fallbacks:
        blockers.append(
            _issue(
                source="openclaw_models",
                code="openclaw_remote_fallbacks",
                detail=f"Remote fallback chain detected: {', '.join(remote_fallbacks)}",
            )
        )
    if remote_allowlist:
        warnings.append(
            _issue(
                source="openclaw_models",
                code="openclaw_remote_allowlist",
                detail=f"Remote models remain allowlisted: {', '.join(remote_allowlist[:6])}",
            )
        )
    if not tool_model_configured:
        blockers.append(
            _issue(
                source="openclaw_models",
                code="tool_model_not_configured",
                detail="No local tool-capable qwen model is configured for OpenClaw agent turns.",
            )
        )
    if not discovered_models:
        blockers.append(
            _issue(
                source="openclaw_models",
                code="ollama_unreachable",
                detail=f"Ollama is unreachable or returned no models at {ollama_base_url}.",
            )
        )
    elif not tool_model_available:
        blockers.append(
            _issue(
                source="openclaw_models",
                code="tool_model_not_available",
                detail="Ollama is up, but no local qwen tool-capable model is currently available.",
            )
        )
    if image_primary and not image_primary.startswith("ollama/"):
        warnings.append(
            _issue(
                source="openclaw_models",
                code="remote_image_model",
                detail=f"imageModel.primary uses a remote model: {image_primary}",
            )
        )

    return snapshot, blockers, warnings


def _security_posture() -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    approval_token = _approval_token_value()
    autonomy_guard_enabled = _env_enabled("OPENCLAW_AUTONOMY_GUARD_ENABLED", default=True)
    approval_required = _env_enabled("OPENCLAW_AUTONOMY_REQUIRE_APPROVAL_TOKEN", default=True)
    injection_block_enabled = _env_enabled("OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS", default=True)
    policy_prefix_enabled = _env_enabled("OPENCLAW_AUTONOMY_POLICY_PREFIX_ENABLED", default=True)
    runtime_pip_enabled = _env_enabled("PMX_ALLOW_RUNTIME_PIP_INSTALL", default=False)
    non_default_approval_token_configured = bool(
        approval_token and approval_token != DEFAULT_APPROVAL_TOKEN
    )

    snapshot = {
        "autonomy_guard_enabled": autonomy_guard_enabled,
        "approval_token_required": approval_required,
        "prompt_injection_block_enabled": injection_block_enabled,
        "policy_prefix_enabled": policy_prefix_enabled,
        "runtime_pip_install_enabled": runtime_pip_enabled,
        "non_default_approval_token_configured": non_default_approval_token_configured,
    }
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not autonomy_guard_enabled:
        blockers.append(
            _issue(
                source="security",
                code="autonomy_guard_disabled",
                detail="OPENCLAW_AUTONOMY_GUARD_ENABLED is disabled.",
            )
        )
    if not approval_required:
        blockers.append(
            _issue(
                source="security",
                code="approval_token_not_required",
                detail="OPENCLAW_AUTONOMY_REQUIRE_APPROVAL_TOKEN is disabled.",
            )
        )
    if not injection_block_enabled:
        blockers.append(
            _issue(
                source="security",
                code="prompt_injection_block_disabled",
                detail="OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS is disabled.",
            )
        )
    if runtime_pip_enabled:
        blockers.append(
            _issue(
                source="security",
                code="runtime_pip_install_enabled",
                detail="PMX_ALLOW_RUNTIME_PIP_INSTALL is enabled, which weakens unattended production hardening.",
            )
        )
    if approval_required and not non_default_approval_token_configured:
        blockers.append(
            _issue(
                source="security",
                code="weak_approval_token",
                detail="OPENCLAW_AUTONOMY_APPROVAL_TOKEN is unset or still using the default token.",
            )
        )
    if not policy_prefix_enabled:
        warnings.append(
            _issue(
                source="security",
                code="policy_prefix_disabled",
                detail="OPENCLAW_AUTONOMY_POLICY_PREFIX_ENABLED is disabled.",
            )
        )

    return snapshot, blockers, warnings


def _openclaw_exec_env_posture() -> tuple[dict[str, Any], list[dict[str, str]]]:
    check = runtime_mod._openclaw_exec_environment_check()
    blockers: list[dict[str, str]] = []
    if not bool(check.get("ok")):
        signals = check.get("signals") if isinstance(check.get("signals"), list) else []
        signal_text = ",".join(str(x) for x in signals if str(x).strip()) or "unknown"
        detail = str(check.get("stderr") or "").strip() or f"signals={signal_text}"
        blockers.append(
            _issue(
                source="openclaw_exec_env",
                code="openclaw_exec_env_invalid",
                detail=f"{detail} ({signal_text})",
            )
        )
    return check, blockers


def _openclaw_regression_posture(timeout_seconds: float) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    ok, report = regression_mod.run_regression_gate(
        openclaw_command=str(os.getenv("OPENCLAW_COMMAND", "openclaw")),
        python_bin=sys.executable,
        primary_channel=str(os.getenv("OPENCLAW_CHANNEL", "whatsapp")).strip().lower() or "whatsapp",
        timeout_seconds=max(5.0, float(timeout_seconds)),
        allow_missing_openclaw=True,
    )
    status = str(report.get("status") or "FAIL").upper()
    if status == "FAIL":
        errors = report.get("errors") if isinstance(report.get("errors"), list) else []
        detail = "; ".join(str(x) for x in errors[:4]) or "OpenClaw regression gate failed."
        blockers.append(
            _issue(
                source="openclaw_regression",
                code="openclaw_regression_failed",
                detail=detail,
            )
        )
    elif status == "SKIP":
        warnings.append(
            _issue(
                source="openclaw_regression",
                code="openclaw_cli_missing",
                detail="OpenClaw CLI is unavailable, so channel regression health was skipped.",
            )
        )
    elif not ok:
        blockers.append(
            _issue(
                source="openclaw_regression",
                code="openclaw_regression_failed",
                detail="OpenClaw regression gate returned a non-success status.",
            )
        )
    return report, blockers, warnings


def _capital_readiness_snapshot() -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    try:
        result, captured_stdout, captured_stderr = _call_quietly(capital_mod.run_capital_readiness)
    except Exception as exc:
        blockers.append(
            _issue(
                source="capital_readiness",
                code="capital_readiness_error",
                detail=f"Capital readiness execution failed: {exc}",
            )
        )
        return {
            "ready": False,
            "verdict": "FAIL",
            "reasons": [],
            "warnings": [],
            "metrics": {},
        }, blockers, warnings

    reasons = result.get("reasons") if isinstance(result.get("reasons"), list) else []
    advisory = result.get("warnings") if isinstance(result.get("warnings"), list) else []
    verdict = str(result.get("verdict") or "FAIL")
    result["captured_output"] = {
        "stdout_tail": captured_stdout,
        "stderr_tail": captured_stderr,
    }
    if not bool(result.get("ready")):
        blockers.append(
            _issue(
                source="capital_readiness",
                code="capital_readiness_failed",
                detail="; ".join(str(x) for x in reasons[:3]) or f"Capital readiness verdict={verdict}",
            )
        )
    for row in advisory[:3]:
        warnings.append(
            _issue(
                source="capital_readiness",
                code="capital_readiness_warning",
                detail=str(row),
            )
        )
    return result, blockers, warnings


def _gate_truth_posture(
    *,
    gate_artifact_path: Path,
    production_gate_artifact_path: Path,
    refresh_production_gate: bool,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    refresh_result: Optional[dict[str, Any]] = None

    if refresh_production_gate:
        refresh_result, refresh_blockers = _refresh_production_gate_artifact(
            artifact_path=production_gate_artifact_path,
            timeout_seconds=max(20.0, float(timeout_seconds)),
        )
        blockers.extend(refresh_blockers)

    gate_artifact = _gate_artifact_snapshot(gate_artifact_path)
    production_gate = _production_gate_snapshot(production_gate_artifact_path)

    gate_time = _parse_ts(gate_artifact.get("timestamp_utc")) or _parse_ts(gate_artifact.get("mtime_utc"))
    production_time = _parse_ts(production_gate.get("timestamp_utc")) or _parse_ts(production_gate.get("mtime_utc"))
    freshest_phase3_source = "gate_status_latest"
    effective_phase3_ready = gate_artifact.get("phase3_ready")
    effective_phase3_reason = gate_artifact.get("phase3_reason")

    if production_time and (gate_time is None or production_time >= gate_time):
        freshest_phase3_source = "production_gate_latest"
        effective_phase3_ready = production_gate.get("phase3_ready")
        effective_phase3_reason = production_gate.get("phase3_reason")

    skipped_optional = gate_artifact.get("skipped_optional_gates")
    max_skipped = gate_artifact.get("max_skipped_optional_gates")
    if isinstance(skipped_optional, int) and isinstance(max_skipped, int) and skipped_optional > max_skipped:
        blockers.append(
            _issue(
                source="gate_truth",
                code="gate_skip_policy_failed",
                detail=(
                    f"gate_status_latest.json reports skipped_optional_gates={skipped_optional} "
                    f"> max_skipped_optional_gates={max_skipped}; overall_passed is fail-closed by policy."
                ),
            )
        )

    if gate_time and production_time and production_time > gate_time:
        if gate_artifact.get("phase3_ready") != production_gate.get("phase3_ready"):
            blockers.append(
                _issue(
                    source="gate_truth",
                    code="stale_gate_artifact_phase3_drift",
                    detail=(
                        "gate_status_latest.json is older than production_gate_latest.json and they disagree on "
                        f"phase3_ready ({gate_artifact.get('phase3_ready')} vs {production_gate.get('phase3_ready')})."
                    ),
                )
            )
        elif str(gate_artifact.get("phase3_reason") or "").strip() != str(production_gate.get("phase3_reason") or "").strip():
            warnings.append(
                _issue(
                    source="gate_truth",
                    code="stale_gate_artifact_phase3_reason_drift",
                    detail=(
                        "gate_status_latest.json is older than production_gate_latest.json and phase3_reason differs "
                        f"({gate_artifact.get('phase3_reason')} vs {production_gate.get('phase3_reason')})."
                    ),
                )
            )
        else:
            warnings.append(
                _issue(
                    source="gate_truth",
                    code="gate_artifact_older_than_production_gate",
                    detail="gate_status_latest.json is older than production_gate_latest.json; refresh canonical gates before trusting readiness.",
                )
            )

    snapshot = {
        "gate_artifact": gate_artifact,
        "production_gate_artifact": production_gate,
        "refresh_result": refresh_result,
        "freshest_phase3_source": freshest_phase3_source,
        "effective_phase3_ready": effective_phase3_ready,
        "effective_phase3_reason": effective_phase3_reason,
        "drift_detected": bool(blockers or warnings),
    }
    return snapshot, blockers, warnings


def _fresh_runtime_snapshot(timeout_seconds: float) -> tuple[Optional[dict[str, Any]], list[dict[str, str]]]:
    blockers: list[dict[str, str]] = []
    try:
        payload, captured_stdout, captured_stderr = _call_quietly(
            runtime_mod.collect_runtime_status,
            timeout_seconds=max(5.0, float(timeout_seconds)),
        )
    except Exception as exc:
        blockers.append(
            _issue(
                source="runtime_status",
                code="runtime_status_error",
                detail=f"Runtime snapshot failed: {exc}",
            )
        )
        return None, blockers

    if isinstance(payload, dict):
        payload["captured_output"] = {
            "stdout_tail": captured_stdout,
            "stderr_tail": captured_stderr,
        }

    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    for check in checks:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or "").strip()
        if not name or bool(check.get("ok")):
            continue
        if name == "openclaw_exec_env":
            continue
        detail = str(check.get("stderr") or "").strip() or str(check.get("stdout") or "").strip() or "check failed"
        blockers.append(
            _issue(
                source="runtime_status",
                code=f"runtime_check_failed:{name}",
                detail=f"{name}: {detail[:220]}",
            )
        )
    return payload, blockers


def _build_human_action_guides(
    *,
    blockers: list[dict[str, str]],
    gate_truth: dict[str, Any],
) -> list[dict[str, Any]]:
    codes = {str(row.get("code") or "").strip() for row in blockers if isinstance(row, dict)}
    guides: list[dict[str, Any]] = []

    def add(guide: dict[str, Any]) -> None:
        guide_id = str(guide.get("id") or "").strip()
        if not guide_id:
            return
        if any(str(row.get("id") or "").strip() == guide_id for row in guides):
            return
        guides.append(guide)

    if "weak_approval_token" in codes:
        add(
            {
                "id": "approval_token",
                "title": "Set a non-default approval token",
                "owner": "human",
                "can_auto_fix": False,
                "why_blocked": "Unattended OpenClaw turns are still using the default approval token.",
                "steps": [
                    "Choose a strong random token and store it outside the repo.",
                    "Set OPENCLAW_AUTONOMY_APPROVAL_TOKEN in the shell or service environment that launches OpenClaw.",
                    "Rerun the readiness snapshot to verify the blocker clears.",
                ],
                "commands": [
                    "$env:OPENCLAW_AUTONOMY_APPROVAL_TOKEN='<strong-random-token>'",
                    "python scripts/openclaw_production_readiness.py --json",
                    "python scripts/openclaw_ops_control_plane.py status --json",
                ],
                "cli_hint": "python scripts/openclaw_production_readiness.py --action-guide approval_token",
            }
        )

    if codes & {"gate_skip_policy_failed", "stale_gate_artifact_phase3_drift"}:
        add(
            {
                "id": "canonical_gate_truth",
                "title": "Refresh canonical gate truth without skip flags",
                "owner": "human",
                "can_auto_fix": False,
                "why_blocked": "The current gate artifact is stale or came from a degraded run that skipped too many optional gates.",
                "steps": [
                    "Coordinate with the parallel refactor lane before rerunning shared gates.",
                    "Run the canonical gate stack with no skip flags so gate_status_latest.json reflects a full evidence pass.",
                    "Compare the refreshed gate artifact with production_gate_latest.json and rerun readiness.",
                ],
                "commands": [
                    "python scripts/run_all_gates.py --json",
                    "python scripts/openclaw_production_readiness.py --json",
                ],
                "notes": [
                    "Do not use --skip-forecast-gate, --skip-profitability-gate, or --skip-institutional-gate for final evidence.",
                    f"Freshest phase3 source right now: {gate_truth.get('freshest_phase3_source')}",
                ],
                "cli_hint": "python scripts/openclaw_production_readiness.py --action-guide canonical_gate_truth",
            }
        )

    if "capital_readiness_failed" in codes:
        add(
            {
                "id": "capital_readiness",
                "title": "Triage economics and evidence depth",
                "owner": "human",
                "can_auto_fix": False,
                "why_blocked": "Trade quality and lift evidence are failing hard readiness rules.",
                "steps": [
                    "Review the latest capital-readiness verdict and identify which rules are failing.",
                    "Confirm whether the current gate artifact is fresh enough to trust before making strategy changes.",
                    "Only after evidence is current, investigate strategy quality, lift, and sample size.",
                ],
                "commands": [
                    "python scripts/capital_readiness_check.py --json",
                    "python scripts/openclaw_production_readiness.py --refresh-production-gate --json",
                ],
                "cli_hint": "python scripts/openclaw_production_readiness.py --action-guide capital_readiness",
            }
        )

    return guides


def _recommendations_for_issues(blockers: list[dict[str, str]], warnings: list[dict[str, str]]) -> list[str]:
    rows = [*blockers, *warnings]
    recs: list[str] = []

    def add(msg: str) -> None:
        text = str(msg or "").strip()
        if text and text not in recs:
            recs.append(text)

    codes = {str(row.get("code") or "") for row in rows}

    if "openclaw_exec_env_invalid" in codes:
        add("Run `python scripts/enforce_openclaw_exec_environment.py` and then `python scripts/project_runtime_status.py --pretty`.")
    if codes & {
        "openclaw_local_only_disabled",
        "openclaw_remote_providers",
        "openclaw_primary_remote",
        "openclaw_remote_fallbacks",
        "tool_model_not_configured",
        "tool_model_not_available",
        "ollama_unreachable",
    }:
        add("Keep OpenClaw local-only and re-apply the model chain with `python scripts/openclaw_models.py apply`; ensure `qwen3:8b` is available in Ollama.")
    if codes & {
        "autonomy_guard_disabled",
        "approval_token_not_required",
        "prompt_injection_block_disabled",
        "weak_approval_token",
        "runtime_pip_install_enabled",
    }:
        add(
            "Harden unattended execution: set `OPENCLAW_AUTONOMY_GUARD_ENABLED=1`, "
            "`OPENCLAW_AUTONOMY_REQUIRE_APPROVAL_TOKEN=1`, "
            "`OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS=1`, "
            "use a non-default `OPENCLAW_AUTONOMY_APPROVAL_TOKEN`, and keep `PMX_ALLOW_RUNTIME_PIP_INSTALL=0`."
        )
        add("Show the exact human steps with `python scripts/openclaw_production_readiness.py --action-guide approval_token`.")
    if codes & {"openclaw_regression_failed", "openclaw_cli_missing"}:
        add("Validate messaging health with `python scripts/openclaw_regression_gate.py --json` and repair channels via `python scripts/openclaw_maintenance.py --strict`.")
    if codes & {"gate_skip_policy_failed", "stale_gate_artifact_phase3_drift"}:
        add("Refresh canonical gate truth without skip flags; see `python scripts/openclaw_production_readiness.py --action-guide canonical_gate_truth`.")
    if codes & {"capital_readiness_failed", "runtime_check_failed:production_gate"}:
        add("Do not promote to production yet; use `python scripts/run_all_gates.py --json` and `python scripts/capital_readiness_check.py --json` to confirm current blockers and evidence gaps.")
        add("For a focused human triage checklist, run `python scripts/openclaw_production_readiness.py --action-guide capital_readiness`.")
    if codes & {"runtime_status_error", "capital_readiness_error"}:
        add("Fix the failing local diagnostics before trusting readiness claims; this snapshot is only as good as the underlying checks.")

    return recs[:6]


def collect_openclaw_production_readiness(
    *,
    config_path: Optional[Path] = None,
    gate_artifact_path: Path = DEFAULT_GATE_ARTIFACT,
    production_gate_artifact_path: Path = DEFAULT_PRODUCTION_GATE_ARTIFACT,
    fresh_runtime: bool = False,
    refresh_production_gate: bool = False,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    effective_config_path = config_path or _default_openclaw_json_path()

    gate_truth, gate_truth_blockers, gate_truth_warnings = _gate_truth_posture(
        gate_artifact_path=gate_artifact_path,
        production_gate_artifact_path=production_gate_artifact_path,
        refresh_production_gate=bool(refresh_production_gate),
        timeout_seconds=float(timeout_seconds),
    )
    gate_artifact = gate_truth.get("gate_artifact") if isinstance(gate_truth.get("gate_artifact"), dict) else _gate_artifact_snapshot(gate_artifact_path)
    blockers.extend(gate_truth_blockers)
    warnings.extend(gate_truth_warnings)

    model_posture, model_blockers, model_warnings = _openclaw_model_posture(effective_config_path)
    blockers.extend(model_blockers)
    warnings.extend(model_warnings)

    security_posture, security_blockers, security_warnings = _security_posture()
    blockers.extend(security_blockers)
    warnings.extend(security_warnings)

    exec_env_check, exec_env_blockers = _openclaw_exec_env_posture()
    blockers.extend(exec_env_blockers)

    regression_report, regression_blockers, regression_warnings = _openclaw_regression_posture(
        timeout_seconds=min(20.0, max(5.0, float(timeout_seconds)))
    )
    blockers.extend(regression_blockers)
    warnings.extend(regression_warnings)

    capital_readiness, capital_blockers, capital_warnings = _capital_readiness_snapshot()
    blockers.extend(capital_blockers)
    warnings.extend(capital_warnings)

    runtime_snapshot: Optional[dict[str, Any]] = None
    if fresh_runtime:
        runtime_snapshot, runtime_blockers = _fresh_runtime_snapshot(timeout_seconds=max(5.0, float(timeout_seconds)))
        blockers.extend(runtime_blockers)

    blockers = _dedupe_issues(blockers)
    warnings = _dedupe_issues(warnings)
    human_action_guides = _build_human_action_guides(blockers=blockers, gate_truth=gate_truth)
    recommendations = _recommendations_for_issues(blockers, warnings)

    readiness_status = "FAIL" if blockers else ("WARN" if warnings else "PASS")
    ready_now = readiness_status == "PASS"

    return {
        "status": "PASS",
        "action": "assess_production_readiness",
        "timestamp_utc": _utc_now_iso(),
        "readiness_status": readiness_status,
        "ready_now": ready_now,
        "summary": {
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "fresh_runtime": bool(fresh_runtime),
            "human_action_count": len(human_action_guides),
        },
        "blockers": blockers,
        "warnings": warnings,
        "recommendations": recommendations,
        "human_action_guides": human_action_guides,
        "gate_artifact": gate_artifact,
        "production_gate_artifact": gate_truth.get("production_gate_artifact"),
        "gate_truth": gate_truth,
        "capital_readiness": capital_readiness,
        "openclaw_exec_env": exec_env_check,
        "openclaw_regression": regression_report,
        "security_posture": security_posture,
        "model_posture": model_posture,
        "runtime_snapshot": runtime_snapshot,
    }


def _print_human_summary(payload: dict[str, Any]) -> None:
    readiness_status = str(payload.get("readiness_status") or "UNKNOWN")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    recommendations = payload.get("recommendations") if isinstance(payload.get("recommendations"), list) else []
    gate_truth = payload.get("gate_truth") if isinstance(payload.get("gate_truth"), dict) else {}
    action_guides = payload.get("human_action_guides") if isinstance(payload.get("human_action_guides"), list) else []

    print(
        f"[openclaw_production_readiness] readiness={readiness_status} "
        f"blockers={summary.get('blocker_count', len(blockers))} "
        f"warnings={summary.get('warning_count', len(warnings))}"
    )
    if gate_truth:
        print(
            "[openclaw_production_readiness] gate_truth "
            f"effective_phase3_ready={gate_truth.get('effective_phase3_ready')} "
            f"source={gate_truth.get('freshest_phase3_source')} "
            f"drift_detected={gate_truth.get('drift_detected')}"
        )
    for row in blockers[:6]:
        if not isinstance(row, dict):
            continue
        print(
            f"[openclaw_production_readiness] blocker "
            f"{row.get('source', 'unknown')}:{row.get('code', 'unknown')} - {row.get('detail', '')}"
        )
    for row in warnings[:4]:
        if not isinstance(row, dict):
            continue
        print(
            f"[openclaw_production_readiness] warning "
            f"{row.get('source', 'unknown')}:{row.get('code', 'unknown')} - {row.get('detail', '')}"
        )
    for rec in recommendations[:4]:
        print(f"[openclaw_production_readiness] next: {rec}")
    for guide in action_guides[:3]:
        if not isinstance(guide, dict):
            continue
        hint = str(guide.get("cli_hint") or "").strip()
        if hint:
            print(f"[openclaw_production_readiness] human_action: {hint}")


def _print_action_guides(payload: dict[str, Any], *, requested_guide: str) -> None:
    guides = payload.get("human_action_guides") if isinstance(payload.get("human_action_guides"), list) else []
    if requested_guide and requested_guide != "all":
        guides = [row for row in guides if isinstance(row, dict) and str(row.get("id") or "").strip() == requested_guide]

    if not guides:
        print("[openclaw_production_readiness] No matching human-action guides for the current blockers.")
        return

    print("[openclaw_production_readiness] Human Action Guide")
    for guide in guides:
        if not isinstance(guide, dict):
            continue
        print(f"[openclaw_production_readiness] guide {guide.get('id')} - {guide.get('title')}")
        print(f"  why: {guide.get('why_blocked')}")
        for step in guide.get("steps", [])[:4]:
            print(f"  step: {step}")
        for cmd in guide.get("commands", [])[:4]:
            print(f"  cmd : {cmd}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_default_openclaw_json_path()),
        help="Path to openclaw.json (default: ~/.openclaw/openclaw.json).",
    )
    parser.add_argument(
        "--gate-artifact",
        default=str(DEFAULT_GATE_ARTIFACT),
        help="Path to logs/gate_status_latest.json.",
    )
    parser.add_argument(
        "--production-gate-artifact",
        default=str(DEFAULT_PRODUCTION_GATE_ARTIFACT),
        help="Path to logs/audit_gate/production_gate_latest.json.",
    )
    parser.add_argument(
        "--fresh-runtime",
        action="store_true",
        help="Run a fresh runtime snapshot in addition to artifact-based readiness checks.",
    )
    parser.add_argument(
        "--refresh-production-gate",
        action="store_true",
        help="Refresh production_gate_latest.json before evaluating readiness.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=20.0,
        help="Timeout budget for OpenClaw regression and fresh runtime checks.",
    )
    parser.add_argument(
        "--action-guide",
        default="",
        help="Show a human-action guide for unresolved blockers: approval_token, canonical_gate_truth, capital_readiness, or all.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    try:
        payload = collect_openclaw_production_readiness(
            config_path=Path(str(args.config)),
            gate_artifact_path=Path(str(args.gate_artifact)),
            production_gate_artifact_path=Path(str(args.production_gate_artifact)),
            fresh_runtime=bool(args.fresh_runtime),
            refresh_production_gate=bool(args.refresh_production_gate),
            timeout_seconds=float(args.timeout_seconds),
        )
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "action": "assess_production_readiness",
            "error": str(exc),
            "timestamp_utc": _utc_now_iso(),
        }
        if bool(args.json):
            print(json.dumps(failure, indent=2))
        else:
            print(f"[openclaw_production_readiness] FAIL - {exc}", file=sys.stderr)
        return 2

    requested_guide = str(args.action_guide or "").strip().lower()
    if requested_guide:
        guides = payload.get("human_action_guides") if isinstance(payload.get("human_action_guides"), list) else []
        if requested_guide != "all":
            guides = [
                row
                for row in guides
                if isinstance(row, dict) and str(row.get("id") or "").strip() == requested_guide
            ]
        guide_payload = {
            "status": "PASS",
            "action": "production_readiness_action_guide",
            "timestamp_utc": _utc_now_iso(),
            "requested_guide": requested_guide,
            "readiness_status": payload.get("readiness_status"),
            "guides": guides,
        }
        if bool(args.json):
            print(json.dumps(guide_payload, indent=2))
        else:
            _print_action_guides(payload, requested_guide=requested_guide)
        return 0

    if bool(args.json):
        print(json.dumps(payload, indent=2))
    else:
        _print_human_summary(payload)
    return 0 if bool(payload.get("ready_now")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
