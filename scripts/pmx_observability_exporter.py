#!/usr/bin/env python3
"""
Read-only Prometheus exporter for PMX observability on Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD_PATH = PROJECT_ROOT / "visualizations" / "dashboard_data.json"
DEFAULT_METRICS_SUMMARY_PATH = PROJECT_ROOT / "visualizations" / "performance" / "metrics_summary.json"
DEFAULT_PRODUCTION_GATE_PATH = PROJECT_ROOT / "logs" / "audit_gate" / "production_gate_latest.json"
DEFAULT_OPENCLAW_MAINTENANCE_PATH = PROJECT_ROOT / "logs" / "automation" / "openclaw_maintenance_latest.json"
DEFAULT_CRON_JOBS_PATH = Path.home() / ".openclaw" / "cron" / "jobs.json"
DEFAULT_SQLITE_DB_PATH = PROJECT_ROOT / "data" / "portfolio_maximizer.db"
DEFAULT_EXPORTER_STATE_PATH = PROJECT_ROOT / "logs" / "observability" / "exporter_state.json"

DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 9765
DEFAULT_ARTIFACT_INTERVAL_SECONDS = 15.0
DEFAULT_RUNTIME_INTERVAL_SECONDS = 30.0
DEFAULT_HEAVY_INTERVAL_SECONDS = 300.0
DEFAULT_LOOP_SLEEP_SECONDS = 1.0
DEFAULT_DASHBOARD_EXPECTED_REFRESH_SECONDS = 60.0

ALLOWED_LABEL_KEYS = ("severity", "component", "channel", "job")
PRODUCTION_GATE_STATUS_CODES = {
    "PASS": 0,
    "READY": 0,
    "GREEN": 0,
    "WARN": 1,
    "INCONCLUSIVE": 1,
    "FAIL": 2,
    "RED": 2,
    "MISSING": 3,
    "ERROR": 3,
    "UNKNOWN": 3,
}
GENERIC_STATUS_CODES = {
    "PASS": 0,
    "OK": 0,
    "WARN": 1,
    "FAIL": 2,
    "ERROR": 3,
    "MISSING": 3,
    "UNKNOWN": 3,
}
MODEL_STATUS_CODES = {
    "PASS": 0,
    "WARN": 1,
    "FAIL": 2,
    "SKIP": 3,
    "ERROR": 4,
    "UNKNOWN": 4,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _sanitize_labels(labels: Optional[Dict[str, Any]]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for key, value in (labels or {}).items():
        text_key = str(key or "").strip()
        if not text_key or text_key not in ALLOWED_LABEL_KEYS:
            continue
        text_value = str(value or "").strip()
        if not text_value:
            continue
        cleaned[text_key] = text_value
    return cleaned


def _format_metric_value(value: float) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if int(value) == value:
        return str(int(value))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_unixtime(value: Optional[datetime]) -> Optional[float]:
    if value is None:
        return None
    return float(value.timestamp())


def _json_best_effort(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty output")
    try:
        return json.loads(text)
    except Exception:
        obj_start = text.find("{")
        obj_end = text.rfind("}")
        if obj_start >= 0 and obj_end > obj_start:
            return json.loads(text[obj_start : obj_end + 1])
        arr_start = text.find("[")
        arr_end = text.rfind("]")
        if arr_start >= 0 and arr_end > arr_start:
            return json.loads(text[arr_start : arr_end + 1])
        raise


def _read_json_file(path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
    if not path.exists():
        return {}, f"missing:{path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {}, f"corrupt:{path.name}:{exc}"
    if not isinstance(payload, dict):
        return {}, f"invalid:{path.name}:root_not_object"
    return payload, None


def _candidate_datetimes(payload: Dict[str, Any], *paths: Tuple[str, ...]) -> List[datetime]:
    values: List[datetime] = []
    for path in paths:
        cursor: Any = payload
        for part in path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(part)
        parsed = _parse_iso_datetime(cursor)
        if parsed:
            values.append(parsed)
    return values


def _artifact_timestamp(
    payload: Dict[str, Any],
    path: Path,
    *json_paths: Tuple[str, ...],
) -> Optional[datetime]:
    candidates = _candidate_datetimes(payload, *json_paths)
    if candidates:
        return candidates[0]
    if path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return None


def _artifact_age_seconds(timestamp: Optional[datetime]) -> Optional[float]:
    if timestamp is None:
        return None
    return max(0.0, (_utc_now() - timestamp).total_seconds())


def _python_executable() -> str:
    override = str(os.getenv("PMX_OBSERVABILITY_PYTHON") or "").strip()
    if override:
        return override
    repo_venv = PROJECT_ROOT / "simpleTrader_env" / "Scripts" / "python.exe"
    if repo_venv.exists():
        return str(repo_venv)
    return sys.executable


def _run_json_command(
    cmd: List[str],
    *,
    timeout_seconds: float,
    cwd: Optional[Path] = None,
) -> Tuple[int, Optional[Any], str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, OSError) as exc:
        return 127, None, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return 124, None, stdout, (stderr or "timeout")

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    try:
        payload = _json_best_effort(stdout)
    except Exception:
        payload = None
    return int(proc.returncode), payload, stdout, stderr


def _check_sqlite_db(db_path: Path) -> Tuple[bool, Optional[str]]:
    if not db_path.exists():
        return False, f"missing:{db_path.name}"
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=2.0)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return False, str(exc)
    return True, None


def _slugify_job(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return slug or "job"


def _severity_for_job(name: str) -> str:
    match = re.match(r"\[(P\d)\]", (name or "").strip())
    return match.group(1) if match else "OTHER"


def _cron_interval_seconds(schedule: Dict[str, Any]) -> Optional[float]:
    if not isinstance(schedule, dict):
        return None
    kind = str(schedule.get("kind") or "").strip().lower()
    if kind != "cron":
        return None
    expr = str(schedule.get("expr") or "").strip()
    parts = expr.split()
    if len(parts) != 5:
        return None
    minute, hour, dom, month, dow = parts
    if minute.startswith("*/") and hour == "*" and dom == "*" and month == "*" and dow == "*":
        try:
            return float(int(minute[2:]) * 60)
        except Exception:
            return None
    if hour.startswith("*/") and minute.isdigit() and dom == "*" and month == "*" and dow == "*":
        try:
            return float(int(hour[2:]) * 3600)
        except Exception:
            return None
    if minute.isdigit() and hour.isdigit() and dom == "*" and month == "*" and dow == "*":
        return 86400.0
    if minute.isdigit() and hour.isdigit() and dom == "*" and month == "*" and dow != "*":
        return 7 * 86400.0
    return None


def _worst_model_status(results: Iterable[Dict[str, Any]]) -> str:
    order = {"PASS": 0, "WARN": 1, "FAIL": 2, "SKIP": 3, "ERROR": 4, "UNKNOWN": 4}
    worst_name = "PASS"
    worst_score = -1
    for row in results:
        status = str(row.get("status") or "UNKNOWN").strip().upper()
        score = order.get(status, 4)
        if score > worst_score:
            worst_name = status
            worst_score = score
    return worst_name


@dataclass
class MetricRegistry:
    meta: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    samples: Dict[str, List[Tuple[Dict[str, str], float]]] = field(default_factory=dict)

    def add(
        self,
        name: str,
        value: float,
        *,
        help_text: str,
        metric_type: str = "gauge",
        labels: Optional[Dict[str, Any]] = None,
    ) -> None:
        clean_labels = _sanitize_labels(labels)
        self.meta.setdefault(name, (help_text, metric_type))
        self.samples.setdefault(name, []).append((clean_labels, float(value)))

    def render(self) -> str:
        lines: List[str] = []
        for name in sorted(self.samples):
            help_text, metric_type = self.meta[name]
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            for labels, value in self.samples[name]:
                label_text = ""
                if labels:
                    parts = [
                        f'{key}="{_escape_label_value(str(labels[key]))}"'
                        for key in sorted(labels)
                    ]
                    label_text = "{" + ",".join(parts) + "}"
                lines.append(f"{name}{label_text} {_format_metric_value(value)}")
        return "\n".join(lines) + ("\n" if lines else "")


class ObservabilityExporter:
    def __init__(
        self,
        *,
        dashboard_path: Path = DEFAULT_DASHBOARD_PATH,
        metrics_summary_path: Path = DEFAULT_METRICS_SUMMARY_PATH,
        production_gate_path: Path = DEFAULT_PRODUCTION_GATE_PATH,
        maintenance_path: Path = DEFAULT_OPENCLAW_MAINTENANCE_PATH,
        cron_jobs_path: Path = DEFAULT_CRON_JOBS_PATH,
        db_path: Path = DEFAULT_SQLITE_DB_PATH,
        state_path: Path = DEFAULT_EXPORTER_STATE_PATH,
        artifacts_interval_seconds: float = DEFAULT_ARTIFACT_INTERVAL_SECONDS,
        runtime_interval_seconds: float = DEFAULT_RUNTIME_INTERVAL_SECONDS,
        heavy_interval_seconds: float = DEFAULT_HEAVY_INTERVAL_SECONDS,
        dashboard_expected_refresh_seconds: float = DEFAULT_DASHBOARD_EXPECTED_REFRESH_SECONDS,
        command_runner: Optional[
            Callable[[List[str]], Tuple[int, Optional[Any], str, str]]
        ] = None,
        sqlite_checker: Optional[Callable[[Path], Tuple[bool, Optional[str]]]] = None,
        now_provider: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.dashboard_path = Path(dashboard_path)
        self.metrics_summary_path = Path(metrics_summary_path)
        self.production_gate_path = Path(production_gate_path)
        self.maintenance_path = Path(maintenance_path)
        self.cron_jobs_path = Path(cron_jobs_path)
        self.db_path = Path(db_path)
        self.state_path = Path(state_path)
        self.artifacts_interval_seconds = float(artifacts_interval_seconds)
        self.runtime_interval_seconds = float(runtime_interval_seconds)
        self.heavy_interval_seconds = float(heavy_interval_seconds)
        self.dashboard_expected_refresh_seconds = float(dashboard_expected_refresh_seconds)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._refresh_in_progress = False
        self._command_runner = command_runner or self._default_command_runner
        self._sqlite_checker = sqlite_checker or _check_sqlite_db
        self._now_provider = now_provider or _utc_now
        self._snapshot: Dict[str, Any] = {
            "status": "starting",
            "timestamp_utc": _utc_now_iso(),
            "warnings": [],
            "collectors": {},
        }
        self._metrics_text = ""
        self._next_due = {
            "artifacts": 0.0,
            "runtime": 0.0,
            "heavy": 0.0,
        }
        self._persisted_state = self._load_state()

    def _now(self) -> datetime:
        return self._now_provider()

    def _default_command_runner(self, cmd: List[str]) -> Tuple[int, Optional[Any], str, str]:
        timeout_seconds = 180.0 if "check_model_improvement.py" in " ".join(cmd) else 30.0
        return _run_json_command(cmd, timeout_seconds=timeout_seconds, cwd=PROJECT_ROOT)

    def _load_state(self) -> Dict[str, Any]:
        payload, err = _read_json_file(self.state_path)
        if err:
            return {
                "cron_last_success_ms": {},
                "recovery_events_total": 0,
                "last_maintenance_timestamp": "",
            }
        return {
            "cron_last_success_ms": payload.get("cron_last_success_ms", {}) if isinstance(payload.get("cron_last_success_ms"), dict) else {},
            "recovery_events_total": int(payload.get("recovery_events_total") or 0),
            "last_maintenance_timestamp": str(payload.get("last_maintenance_timestamp") or ""),
        }

    def _save_state(self) -> None:
        _ensure_dir(self.state_path)
        self.state_path.write_text(
            json.dumps(self._persisted_state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def start(self, *, loop_sleep_seconds: float = DEFAULT_LOOP_SLEEP_SECONDS) -> None:
        self.refresh(force=True)
        self._thread = threading.Thread(
            target=self._run_loop,
            kwargs={"loop_sleep_seconds": float(loop_sleep_seconds)},
            daemon=True,
            name="pmx-observability-exporter",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _run_loop(self, *, loop_sleep_seconds: float) -> None:
        while not self._stop_event.wait(max(0.2, loop_sleep_seconds)):
            self.refresh(force=False)

    def refresh(self, *, force: bool) -> None:
        now = self._now().timestamp()
        with self._lock:
            if self._refresh_in_progress:
                return
            due = [
                name
                for name, threshold in self._next_due.items()
                if force or now >= threshold
            ]
            if not due:
                return
            self._refresh_in_progress = True

        try:
            collected: Dict[str, Dict[str, Any]] = {}
            if "artifacts" in due:
                collected["artifacts"] = self._collect_artifacts()
            if "runtime" in due:
                collected["runtime"] = self._collect_runtime()
            if "heavy" in due:
                collected["heavy"] = self._collect_heavy()

            with self._lock:
                snapshot = dict(self._snapshot)
                snapshot.setdefault("collectors", {})
                snapshot["timestamp_utc"] = self._now().isoformat()
                warnings: List[str] = []

                for name, payload in collected.items():
                    snapshot["collectors"][name] = payload
                    warnings.extend(str(item) for item in payload.get("warnings", []) if str(item).strip())
                    if name == "artifacts":
                        self._next_due[name] = now + self.artifacts_interval_seconds
                    elif name == "runtime":
                        self._next_due[name] = now + self.runtime_interval_seconds
                    elif name == "heavy":
                        self._next_due[name] = now + self.heavy_interval_seconds

                snapshot["warnings"] = sorted(set(warnings))
                snapshot["status"] = "ok"
                self._snapshot = snapshot
                self._metrics_text = self._build_metrics_text(snapshot)
                self._save_state()
        finally:
            with self._lock:
                self._refresh_in_progress = False

    def get_metrics_text(self) -> str:
        with self._lock:
            return self._metrics_text

    def get_health_payload(self) -> Dict[str, Any]:
        with self._lock:
            snapshot = json.loads(json.dumps(self._snapshot))
        collectors = snapshot.get("collectors", {})
        healthy = bool(collectors)
        return {
            "status": "ok" if healthy else "starting",
            "timestamp_utc": snapshot.get("timestamp_utc"),
            "warning_count": len(snapshot.get("warnings", [])),
            "warnings": snapshot.get("warnings", []),
            "collectors": {
                name: {
                    "status": data.get("status"),
                    "warning_count": len(data.get("warnings", [])),
                    "last_collected_utc": data.get("timestamp_utc"),
                }
                for name, data in collectors.items()
                if isinstance(data, dict)
            },
        }

    def _collect_artifacts(self) -> Dict[str, Any]:
        now = self._now()
        warnings: List[str] = []

        dashboard_payload, dashboard_err = _read_json_file(self.dashboard_path)
        if dashboard_err:
            warnings.append(dashboard_err)
        dashboard_ts = _artifact_timestamp(
            dashboard_payload,
            self.dashboard_path,
            ("meta", "generated_utc"),
            ("meta", "ts"),
        )

        metrics_payload, metrics_err = _read_json_file(self.metrics_summary_path)
        if metrics_err:
            warnings.append(metrics_err)
        metrics_ts = _artifact_timestamp(
            metrics_payload,
            self.metrics_summary_path,
            ("generated_utc",),
            ("generated_at",),
        )

        gate_payload, gate_err = _read_json_file(self.production_gate_path)
        if gate_err:
            warnings.append(gate_err)
        gate_ts = _artifact_timestamp(
            gate_payload,
            self.production_gate_path,
            ("timestamp_utc",),
            ("artifact_binding", "summary_generated_utc"),
        )
        phase3_reason = str(
            gate_payload.get("phase3_reason")
            or gate_payload.get("status")
            or ("READY" if gate_payload.get("phase3_ready") else "FAIL")
        ).strip().upper() or "UNKNOWN"
        evidence_progress = (
            gate_payload.get("profitability_proof", {}).get("evidence_progress", {})
            if isinstance(gate_payload.get("profitability_proof"), dict)
            else {}
        )

        maintenance_payload, maintenance_err = _read_json_file(self.maintenance_path)
        if maintenance_err:
            warnings.append(maintenance_err)
        maintenance_ts = _artifact_timestamp(
            maintenance_payload,
            self.maintenance_path,
            ("timestamp_utc",),
        )
        recovery_keys = self._maintenance_recovery_keys(maintenance_payload)
        maintenance_timestamp_text = str(maintenance_payload.get("timestamp_utc") or "").strip()
        if maintenance_timestamp_text and maintenance_timestamp_text != self._persisted_state.get("last_maintenance_timestamp"):
            self._persisted_state["last_maintenance_timestamp"] = maintenance_timestamp_text
            self._persisted_state["recovery_events_total"] = int(
                self._persisted_state.get("recovery_events_total", 0)
            ) + len(recovery_keys)

        metrics_status = str(metrics_payload.get("status") or "UNKNOWN").strip().upper()

        return {
            "status": "ok",
            "timestamp_utc": now.isoformat(),
            "warnings": warnings,
            "dashboard": {
                "generated_unixtime": _to_unixtime(dashboard_ts),
                "age_seconds": _artifact_age_seconds(dashboard_ts),
            },
            "metrics_summary": {
                "generated_unixtime": _to_unixtime(metrics_ts),
                "status": metrics_status,
                "status_code": GENERIC_STATUS_CODES.get(metrics_status, 3),
            },
            "production_gate": {
                "generated_unixtime": _to_unixtime(gate_ts),
                "pass": 1.0 if bool(gate_payload.get("phase3_ready")) else 0.0,
                "status_code": float(PRODUCTION_GATE_STATUS_CODES.get(phase3_reason, 3)),
                "status_text": phase3_reason,
                "closed_trades": float(evidence_progress.get("closed_trades") or 0.0),
                "remaining_days": float(evidence_progress.get("remaining_trading_days") or 0.0),
            },
            "maintenance": {
                "generated_unixtime": _to_unixtime(maintenance_ts),
                "recovery_event_count": float(self._persisted_state.get("recovery_events_total", 0)),
                "recovery_keys": recovery_keys,
            },
        }

    def _maintenance_recovery_keys(self, payload: Dict[str, Any]) -> List[str]:
        steps = payload.get("steps", {}) if isinstance(payload.get("steps"), dict) else {}
        fast = steps.get("fast_supervisor", {}) if isinstance(steps.get("fast_supervisor"), dict) else {}
        gateway = steps.get("gateway_health", {}) if isinstance(steps.get("gateway_health"), dict) else {}
        channels = (
            steps.get("channels_status_snapshot", {}).get("channels", {})
            if isinstance(steps.get("channels_status_snapshot"), dict)
            else {}
        )
        gateway_warnings = [str(item) for item in gateway.get("warnings", [])] if isinstance(gateway.get("warnings"), list) else []
        keys: List[str] = []
        if (
            str(fast.get("action") or "") == "soft_timeout_skip"
            or str(fast.get("reason") or "") == "channels_status_timeout_softened"
        ):
            keys.append("channels_status_timeout_softened")
        if (
            str(fast.get("action") or "") == "gateway_restart_triggered"
            and not gateway.get("primary_channel_issue_final")
        ):
            keys.append("gateway_restart_recovered")
        if (
            str(gateway.get("primary_channel_issue") or "") == "whatsapp_handshake_timeout"
            and not gateway.get("primary_channel_issue_final")
        ):
            keys.append("whatsapp_handshake_recovered")
        if "gateway_detached_listener_conflict" in gateway_warnings:
            keys.append("gateway_detached_listener_conflict")
        whatsapp = channels.get("whatsapp") if isinstance(channels, dict) else {}
        reconnect_attempts = 0
        if isinstance(whatsapp, dict):
            try:
                reconnect_attempts = int(whatsapp.get("reconnectAttempts") or 0)
            except Exception:
                reconnect_attempts = 0
        if reconnect_attempts > 0:
            keys.append("whatsapp_reconnect")
        deduped: List[str] = []
        seen = set()
        for item in keys:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def _collect_runtime(self) -> Dict[str, Any]:
        now = self._now()
        warnings: List[str] = []

        rc, payload, stdout, stderr = self._command_runner(
            [
                _python_executable(),
                str(PROJECT_ROOT / "scripts" / "openclaw_remote_workflow.py"),
                "health",
                "--json",
            ]
        )
        if not isinstance(payload, dict):
            warnings.append(f"openclaw_health_error:rc={rc}")
            if stderr.strip():
                warnings.append(f"openclaw_health_stderr:{stderr.strip()[:140]}")
            if stdout.strip() and not stderr.strip():
                warnings.append(f"openclaw_health_stdout:{stdout.strip()[:140]}")
            payload = {}

        sqlite_ok, sqlite_error = self._sqlite_checker(self.db_path)
        if sqlite_error:
            warnings.append(f"sqlite:{sqlite_error}")

        cron_payload, cron_warnings = self._collect_cron_jobs(now=now)
        warnings.extend(cron_warnings)

        primary_channel = str(payload.get("primary_channel") or "whatsapp").strip().lower() or "whatsapp"
        primary_status = str(payload.get("primary_status") or "UNKNOWN").strip().upper() or "UNKNOWN"
        primary_up = 1.0 if primary_status == "OK" else 0.0
        gateway_up = 1.0 if bool(payload.get("gateway_reachable")) else 0.0
        channels_latency_ms = float(payload.get("channels_status_elapsed_ms") or 0.0)

        return {
            "status": "ok",
            "timestamp_utc": now.isoformat(),
            "warnings": warnings,
            "openclaw": {
                "observed_unixtime": _to_unixtime(_parse_iso_datetime(payload.get("timestamp_utc")) or now),
                "gateway_up": gateway_up,
                "primary_channel": primary_channel,
                "primary_up": primary_up,
                "channels_status_latency_ms": channels_latency_ms,
                "recovery_mode": str(payload.get("recovery_mode") or "unknown"),
            },
            "sqlite": {
                "ok": 1.0 if sqlite_ok else 0.0,
            },
            "cron": cron_payload,
        }

    def _collect_cron_jobs(self, *, now: datetime) -> Tuple[Dict[str, Any], List[str]]:
        payload, err = _read_json_file(self.cron_jobs_path)
        warnings: List[str] = []
        if err:
            warnings.append(err)
            return {"jobs": []}, warnings

        jobs = payload.get("jobs", [])
        if not isinstance(jobs, list):
            warnings.append("invalid:jobs.json:jobs_not_list")
            return {"jobs": []}, warnings

        job_rows: List[Dict[str, Any]] = []
        cache = self._persisted_state.setdefault("cron_last_success_ms", {})
        now_ms = int(now.timestamp() * 1000)
        for job in jobs:
            if not isinstance(job, dict):
                continue
            state = job.get("state", {}) if isinstance(job.get("state"), dict) else {}
            schedule = job.get("schedule", {}) if isinstance(job.get("schedule"), dict) else {}
            name = str(job.get("name") or job.get("id") or "job").strip()
            job_label = _slugify_job(name)
            job_id = str(job.get("id") or job_label).strip()
            severity = _severity_for_job(name)
            enabled = 1.0 if bool(job.get("enabled", False)) else 0.0
            last_run_ms = int(state.get("lastRunAtMs") or 0)
            last_status = str(state.get("lastStatus") or state.get("lastRunStatus") or "unknown").strip().lower()
            if last_status == "success" and last_run_ms > 0:
                cache[job_id] = last_run_ms
            last_success_ms = int(cache.get(job_id) or 0)
            expected_interval_seconds = _cron_interval_seconds(schedule) or 0.0
            stale_threshold_seconds = expected_interval_seconds * 2.0 if expected_interval_seconds > 0 else 0.0
            freshness_lag_seconds = (
                max(0.0, (now_ms - last_success_ms) / 1000.0) if last_success_ms > 0 else 0.0
            )
            job_rows.append(
                {
                    "job": job_label,
                    "severity": severity,
                    "enabled": enabled,
                    "last_run_unixtime": last_run_ms / 1000.0 if last_run_ms > 0 else 0.0,
                    "last_success_unixtime": last_success_ms / 1000.0 if last_success_ms > 0 else 0.0,
                    "last_status_code": float(GENERIC_STATUS_CODES.get(last_status.upper(), 3)),
                    "consecutive_errors": float(state.get("consecutiveErrors") or 0.0),
                    "expected_interval_seconds": expected_interval_seconds,
                    "stale_threshold_seconds": stale_threshold_seconds,
                    "freshness_lag_seconds": freshness_lag_seconds,
                }
            )
        return {"jobs": job_rows}, warnings

    def _collect_heavy(self) -> Dict[str, Any]:
        now = self._now()
        warnings: List[str] = []
        rc, payload, stdout, stderr = self._command_runner(
            [
                _python_executable(),
                str(PROJECT_ROOT / "scripts" / "check_model_improvement.py"),
                "--json",
            ]
        )
        if not isinstance(payload, dict):
            warnings.append(f"model_improvement_error:rc={rc}")
            if stderr.strip():
                warnings.append(f"model_improvement_stderr:{stderr.strip()[:140]}")
            if stdout.strip() and not stderr.strip():
                warnings.append(f"model_improvement_stdout:{stdout.strip()[:140]}")
            return {
                "status": "ok",
                "timestamp_utc": now.isoformat(),
                "warnings": warnings,
                "model_improvement": {
                    "status_code": float(MODEL_STATUS_CODES["ERROR"]),
                    "observed_unixtime": now.timestamp(),
                },
            }
        results = payload.get("results", [])
        worst = _worst_model_status(results if isinstance(results, list) else [])
        observed = _parse_iso_datetime(payload.get("timestamp_utc")) or now
        return {
            "status": "ok",
            "timestamp_utc": now.isoformat(),
            "warnings": warnings,
            "model_improvement": {
                "status_code": float(MODEL_STATUS_CODES.get(worst, 4)),
                "observed_unixtime": observed.timestamp(),
            },
        }

    def _build_metrics_text(self, snapshot: Dict[str, Any]) -> str:
        registry = MetricRegistry()
        registry.add(
            "pmx_dashboard_snapshot_expected_refresh_seconds",
            self.dashboard_expected_refresh_seconds,
            help_text="Expected PMX dashboard refresh cadence in seconds.",
        )

        artifacts = snapshot.get("collectors", {}).get("artifacts", {})
        if isinstance(artifacts, dict):
            dashboard = artifacts.get("dashboard", {}) if isinstance(artifacts.get("dashboard"), dict) else {}
            if dashboard.get("age_seconds") is not None:
                registry.add(
                    "pmx_dashboard_snapshot_age_seconds",
                    float(dashboard["age_seconds"]),
                    help_text="Age of the latest dashboard snapshot in seconds.",
                )
            if dashboard.get("generated_unixtime") is not None:
                registry.add(
                    "pmx_dashboard_snapshot_generated_unixtime",
                    float(dashboard["generated_unixtime"]),
                    help_text="Unix timestamp for the latest dashboard snapshot generation time.",
                )

            metrics_summary = artifacts.get("metrics_summary", {}) if isinstance(artifacts.get("metrics_summary"), dict) else {}
            if metrics_summary.get("generated_unixtime") is not None:
                registry.add(
                    "pmx_metrics_summary_generated_unixtime",
                    float(metrics_summary["generated_unixtime"]),
                    help_text="Unix timestamp for the latest performance metrics summary artifact.",
                )
            registry.add(
                "pmx_metrics_summary_status_code",
                float(metrics_summary.get("status_code") or 3.0),
                help_text="Status code for metrics_summary.json (0=PASS/OK,1=WARN,2=FAIL,3=MISSING/ERROR).",
            )

            production_gate = artifacts.get("production_gate", {}) if isinstance(artifacts.get("production_gate"), dict) else {}
            registry.add(
                "pmx_production_gate_pass",
                float(production_gate.get("pass") or 0.0),
                help_text="Whether the latest production gate artifact is passing (1=pass, 0=fail).",
            )
            registry.add(
                "pmx_production_gate_status_code",
                float(production_gate.get("status_code") or 3.0),
                help_text="Production gate status code (0=ready/pass,1=warn/inconclusive,2=fail/red,3=missing/error).",
            )
            if production_gate.get("generated_unixtime") is not None:
                registry.add(
                    "pmx_production_gate_generated_unixtime",
                    float(production_gate["generated_unixtime"]),
                    help_text="Unix timestamp for the latest production gate artifact.",
                )
            registry.add(
                "pmx_proof_runway_closed_trades",
                float(production_gate.get("closed_trades") or 0.0),
                help_text="Closed trades counted toward production profitability proof runway.",
            )
            registry.add(
                "pmx_proof_runway_remaining_days",
                float(production_gate.get("remaining_days") or 0.0),
                help_text="Remaining trading days required by the profitability proof runway.",
            )

            maintenance = artifacts.get("maintenance", {}) if isinstance(artifacts.get("maintenance"), dict) else {}
            registry.add(
                "pmx_openclaw_recovery_events_total",
                float(maintenance.get("recovery_event_count") or 0.0),
                help_text="Cumulative recovery events observed from OpenClaw maintenance artifacts.",
                metric_type="counter",
                labels={"component": "openclaw"},
            )

        runtime = snapshot.get("collectors", {}).get("runtime", {})
        if isinstance(runtime, dict):
            openclaw = runtime.get("openclaw", {}) if isinstance(runtime.get("openclaw"), dict) else {}
            primary_channel = str(openclaw.get("primary_channel") or "whatsapp")
            registry.add(
                "pmx_openclaw_gateway_up",
                float(openclaw.get("gateway_up") or 0.0),
                help_text="Whether the OpenClaw gateway is reachable (1=yes, 0=no).",
                labels={"component": "openclaw"},
            )
            registry.add(
                "pmx_openclaw_primary_channel_up",
                float(openclaw.get("primary_up") or 0.0),
                help_text="Whether the primary OpenClaw channel is healthy (1=yes, 0=no).",
                labels={"channel": primary_channel},
            )
            registry.add(
                "pmx_openclaw_channels_status_latency_ms",
                float(openclaw.get("channels_status_latency_ms") or 0.0),
                help_text="Latency of the latest openclaw channels.status call in milliseconds.",
                labels={"channel": primary_channel},
            )
            if openclaw.get("observed_unixtime") is not None:
                registry.add(
                    "pmx_openclaw_health_observed_unixtime",
                    float(openclaw["observed_unixtime"]),
                    help_text="Unix timestamp for the latest OpenClaw health observation.",
                    labels={"channel": primary_channel},
                )

            sqlite_payload = runtime.get("sqlite", {}) if isinstance(runtime.get("sqlite"), dict) else {}
            registry.add(
                "pmx_sqlite_health_ok",
                float(sqlite_payload.get("ok") or 0.0),
                help_text="Whether the primary SQLite database is readable (1=yes, 0=no).",
                labels={"component": "sqlite"},
            )

            cron_payload = runtime.get("cron", {}) if isinstance(runtime.get("cron"), dict) else {}
            for job in cron_payload.get("jobs", []) if isinstance(cron_payload.get("jobs"), list) else []:
                if not isinstance(job, dict):
                    continue
                labels = {
                    "job": str(job.get("job") or "job"),
                    "severity": str(job.get("severity") or "OTHER"),
                }
                registry.add(
                    "pmx_cron_job_enabled",
                    float(job.get("enabled") or 0.0),
                    help_text="Whether the cron job is enabled (1=yes, 0=no).",
                    labels=labels,
                )
                registry.add(
                    "pmx_cron_job_last_run_unixtime",
                    float(job.get("last_run_unixtime") or 0.0),
                    help_text="Unix timestamp for the latest observed cron run.",
                    labels=labels,
                )
                registry.add(
                    "pmx_cron_job_last_success_unixtime",
                    float(job.get("last_success_unixtime") or 0.0),
                    help_text="Unix timestamp for the latest successful cron run seen by the exporter.",
                    labels=labels,
                )
                registry.add(
                    "pmx_cron_job_consecutive_errors",
                    float(job.get("consecutive_errors") or 0.0),
                    help_text="Consecutive error count for the cron job.",
                    labels=labels,
                )
                registry.add(
                    "pmx_cron_job_last_status_code",
                    float(job.get("last_status_code") or 3.0),
                    help_text="Cron job last status code (0=PASS/OK,1=WARN,2=FAIL,3=MISSING/ERROR).",
                    labels=labels,
                )
                registry.add(
                    "pmx_cron_job_expected_interval_seconds",
                    float(job.get("expected_interval_seconds") or 0.0),
                    help_text="Expected cron cadence in seconds derived from the configured cron expression.",
                    labels=labels,
                )
                registry.add(
                    "pmx_cron_job_stale_threshold_seconds",
                    float(job.get("stale_threshold_seconds") or 0.0),
                    help_text="Staleness threshold in seconds for the cron job.",
                    labels=labels,
                )
                registry.add(
                    "pmx_cron_job_freshness_lag_seconds",
                    float(job.get("freshness_lag_seconds") or 0.0),
                    help_text="Seconds since the exporter last saw a successful run for the cron job.",
                    labels=labels,
                )

        heavy = snapshot.get("collectors", {}).get("heavy", {})
        if isinstance(heavy, dict):
            model = heavy.get("model_improvement", {}) if isinstance(heavy.get("model_improvement"), dict) else {}
            registry.add(
                "pmx_model_improvement_status_code",
                float(model.get("status_code") or 4.0),
                help_text="Model improvement checker status code (0=PASS,1=WARN,2=FAIL,3=SKIP,4=ERROR).",
                labels={"component": "model_quality"},
            )
            if model.get("observed_unixtime") is not None:
                registry.add(
                    "pmx_model_improvement_observed_unixtime",
                    float(model.get("observed_unixtime") or 0.0),
                    help_text="Unix timestamp for the latest model improvement observation.",
                    labels={"component": "model_quality"},
                )

        return registry.render()


class _ExporterHandler(BaseHTTPRequestHandler):
    exporter: ObservabilityExporter

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/metrics"):
            self._send_metrics()
            return
        if self.path.startswith("/healthz"):
            self._send_health()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(b'{"status":"not_found"}')

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/shutdown"):
            payload = json.dumps({"status": "shutting_down"}).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(b'{"status":"not_found"}')

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _send_metrics(self) -> None:
        payload = self.exporter.get_metrics_text().encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_health(self) -> None:
        payload = json.dumps(self.exporter.get_health_payload(), sort_keys=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def build_exporter_from_args(args: argparse.Namespace) -> ObservabilityExporter:
    return ObservabilityExporter(
        dashboard_path=Path(args.dashboard_path),
        metrics_summary_path=Path(args.metrics_summary_path),
        production_gate_path=Path(args.production_gate_path),
        maintenance_path=Path(args.maintenance_path),
        cron_jobs_path=Path(args.cron_jobs_path),
        db_path=Path(args.db_path),
        state_path=Path(args.state_path),
        artifacts_interval_seconds=float(args.artifacts_interval_seconds),
        runtime_interval_seconds=float(args.runtime_interval_seconds),
        heavy_interval_seconds=float(args.heavy_interval_seconds),
        dashboard_expected_refresh_seconds=float(args.dashboard_expected_refresh_seconds),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PMX Prometheus exporter")
    parser.add_argument("--bind", default=DEFAULT_BIND, help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port (default: 9765)")
    parser.add_argument("--dashboard-path", default=str(DEFAULT_DASHBOARD_PATH))
    parser.add_argument("--metrics-summary-path", default=str(DEFAULT_METRICS_SUMMARY_PATH))
    parser.add_argument("--production-gate-path", default=str(DEFAULT_PRODUCTION_GATE_PATH))
    parser.add_argument("--maintenance-path", default=str(DEFAULT_OPENCLAW_MAINTENANCE_PATH))
    parser.add_argument("--cron-jobs-path", default=str(DEFAULT_CRON_JOBS_PATH))
    parser.add_argument("--db-path", default=str(DEFAULT_SQLITE_DB_PATH))
    parser.add_argument("--state-path", default=str(DEFAULT_EXPORTER_STATE_PATH))
    parser.add_argument("--artifacts-interval-seconds", type=float, default=DEFAULT_ARTIFACT_INTERVAL_SECONDS)
    parser.add_argument("--runtime-interval-seconds", type=float, default=DEFAULT_RUNTIME_INTERVAL_SECONDS)
    parser.add_argument("--heavy-interval-seconds", type=float, default=DEFAULT_HEAVY_INTERVAL_SECONDS)
    parser.add_argument(
        "--dashboard-expected-refresh-seconds",
        type=float,
        default=DEFAULT_DASHBOARD_EXPECTED_REFRESH_SECONDS,
    )
    parser.add_argument("--loop-sleep-seconds", type=float, default=DEFAULT_LOOP_SLEEP_SECONDS)
    parser.add_argument("--once", action="store_true", help="Collect once and exit.")
    parser.add_argument("--json", action="store_true", help="With --once, print health snapshot as JSON.")
    parser.add_argument("--metrics", action="store_true", help="With --once, print Prometheus text output.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    exporter = build_exporter_from_args(args)

    if args.once:
        exporter.refresh(force=True)
        if args.metrics:
            print(exporter.get_metrics_text(), end="")
        else:
            print(json.dumps(exporter.get_health_payload(), indent=2 if args.json else None, sort_keys=True))
        return 0

    _ExporterHandler.exporter = exporter
    server = ThreadingHTTPServer((str(args.bind), int(args.port)), _ExporterHandler)
    exporter.start(loop_sleep_seconds=float(args.loop_sleep_seconds))
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        exporter.stop()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
