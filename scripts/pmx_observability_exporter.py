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

try:
    from scripts.production_gate_contract import (
        phase3_strict_ready as _phase3_strict_ready,
        phase3_strict_reason as _phase3_strict_reason,
    )
except Exception:  # pragma: no cover - script execution path fallback
    from production_gate_contract import (  # type: ignore
        phase3_strict_ready as _phase3_strict_ready,
        phase3_strict_reason as _phase3_strict_reason,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD_PATH = PROJECT_ROOT / "visualizations" / "dashboard_data.json"
DEFAULT_METRICS_SUMMARY_PATH = PROJECT_ROOT / "visualizations" / "performance" / "metrics_summary.json"
DEFAULT_PRODUCTION_GATE_PATH = PROJECT_ROOT / "logs" / "audit_gate" / "production_gate_latest.json"
DEFAULT_OPENCLAW_MAINTENANCE_PATH = PROJECT_ROOT / "logs" / "automation" / "openclaw_maintenance_latest.json"
DEFAULT_CRON_JOBS_PATH = Path.home() / ".openclaw" / "cron" / "jobs.json"
DEFAULT_REQUIRED_CRON_JOBS_PATH = PROJECT_ROOT / "config" / "observability_required_jobs.yml"
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
OPERATOR_CONSOLE_STATUS_CODES = {
    "PASS": 0,
    "OK": 0,
    "WARN": 1,
    "FAIL": 2,
    "ERROR": 3,
    "UNKNOWN": 4,
}
RECOVERY_MODE_CODES = {
    "steady_state": 0,
    "channels_status_timeout_softened": 1,
    "gateway_restart_recovered": 2,
    "whatsapp_handshake_recovered": 3,
    "gateway_detached_listener_conflict": 4,
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


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        parsed = float(value)
    except Exception:
        return None
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _latest_series_value(points: Any) -> Optional[float]:
    if not isinstance(points, list) or not points:
        return None
    for point in reversed(points):
        if not isinstance(point, dict):
            continue
        value = _safe_float(point.get("v"))
        if value is not None:
            return value
    return None


def _positions_summary(positions: Any) -> Dict[str, Optional[float]]:
    if not isinstance(positions, dict):
        return {
            "count": None,
            "long_count": None,
            "short_count": None,
            "gross_notional": None,
            "net_notional": None,
        }
    count = 0
    long_count = 0
    short_count = 0
    gross_notional = 0.0
    net_notional = 0.0
    for payload in positions.values():
        if not isinstance(payload, dict):
            continue
        shares = _safe_float(payload.get("shares"))
        if shares is None:
            continue
        count += 1
        if shares > 0:
            long_count += 1
        elif shares < 0:
            short_count += 1
        entry_price = _safe_float(payload.get("entry_price"))
        if entry_price is not None:
            notional = shares * entry_price
            gross_notional += abs(notional)
            net_notional += notional
    return {
        "count": float(count),
        "long_count": float(long_count),
        "short_count": float(short_count),
        "gross_notional": gross_notional,
        "net_notional": net_notional,
    }


def _latest_signal_summary(signals: Any) -> Dict[str, Optional[float]]:
    if not isinstance(signals, list) or not signals:
        return {
            "confidence": None,
            "expected_return": None,
            "shares": None,
            "mid_slippage_bp": None,
        }
    for payload in reversed(signals):
        if not isinstance(payload, dict):
            continue
        confidence = _safe_float(payload.get("effective_confidence"))
        if confidence is None:
            confidence = _safe_float(payload.get("signal_confidence"))
        return {
            "confidence": confidence,
            "expected_return": _safe_float(payload.get("expected_return")),
            "shares": _safe_float(payload.get("shares")),
            "mid_slippage_bp": _safe_float(payload.get("mid_slippage_bp")),
        }
    return {
        "confidence": None,
        "expected_return": None,
        "shares": None,
        "mid_slippage_bp": None,
    }


def _latest_trade_event_summary(events: Any) -> Dict[str, Optional[float]]:
    if not isinstance(events, list) or not events:
        return {
            "shares": None,
            "slippage_bp": None,
            "realized_pnl": None,
            "realized_pnl_pct": None,
        }
    for payload in reversed(events):
        if not isinstance(payload, dict):
            continue
        slippage_bp = _safe_float(payload.get("slippage_bp"))
        if slippage_bp is None:
            slippage = _safe_float(payload.get("slippage"))
            if slippage is not None:
                slippage_bp = slippage * 10000.0
        return {
            "shares": _safe_float(payload.get("shares")),
            "slippage_bp": slippage_bp,
            "realized_pnl": _safe_float(payload.get("realized_pnl")),
            "realized_pnl_pct": _safe_float(payload.get("realized_pnl_pct")),
        }
    return {
        "shares": None,
        "slippage_bp": None,
        "realized_pnl": None,
        "realized_pnl_pct": None,
    }


def _dashboard_quant_validation_payload(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    source = payload.get("quant_validation_health")
    if not isinstance(source, dict):
        source = payload.get("quant_validation") if isinstance(payload.get("quant_validation"), dict) else {}
    return {
        "total": _safe_float(source.get("total")),
        "pass_count": _safe_float(source.get("pass_count")),
        "fail_count": _safe_float(source.get("fail_count")),
        "fail_fraction": _safe_float(source.get("fail_fraction")),
        "negative_expected_profit_fraction": _safe_float(source.get("negative_expected_profit_fraction")),
        "max_fail_fraction": _safe_float(source.get("max_fail_fraction")),
        "max_negative_expected_profit_fraction": _safe_float(source.get("max_negative_expected_profit_fraction")),
    }


def _dashboard_summary(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    pnl = payload.get("pnl") if isinstance(payload.get("pnl"), dict) else {}
    latency = payload.get("latency") if isinstance(payload.get("latency"), dict) else {}
    routing = payload.get("routing") if isinstance(payload.get("routing"), dict) else {}
    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
    forecaster = payload.get("forecaster_health") if isinstance(payload.get("forecaster_health"), dict) else {}
    forecaster_metrics = forecaster.get("metrics") if isinstance(forecaster.get("metrics"), dict) else {}
    forecaster_rmse = forecaster_metrics.get("rmse") if isinstance(forecaster_metrics.get("rmse"), dict) else {}
    forecaster_thresholds = forecaster.get("thresholds") if isinstance(forecaster.get("thresholds"), dict) else {}
    forecaster_status = forecaster.get("status") if isinstance(forecaster.get("status"), dict) else {}
    operator_console = payload.get("operator_console") if isinstance(payload.get("operator_console"), dict) else {}
    operator_maintenance = operator_console.get("maintenance") if isinstance(operator_console.get("maintenance"), dict) else {}
    operator_activity = operator_console.get("activity") if isinstance(operator_console.get("activity"), dict) else {}
    operator_issues = operator_console.get("issues") if isinstance(operator_console.get("issues"), list) else []
    recovery_mode = str(operator_maintenance.get("recovery_mode") or "").strip()
    position_summary = _positions_summary(payload.get("positions"))
    latest_signal = _latest_signal_summary(payload.get("signals"))
    latest_trade_event = _latest_trade_event_summary(payload.get("trade_events"))
    quant_validation = _dashboard_quant_validation_payload(payload)

    failed_checks = 0
    for key in ("profit_factor_ok", "win_rate_ok", "rmse_ok"):
        value = forecaster_status.get(key)
        if value is False:
            failed_checks += 1

    return {
        "pnl_absolute": _safe_float(pnl.get("absolute")),
        "pnl_pct": _safe_float(pnl.get("pct")),
        "win_rate": _safe_float(payload.get("win_rate")),
        "trade_count": _safe_float(payload.get("trade_count")),
        "latency_ts_ms": _safe_float(latency.get("ts_ms")),
        "latency_llm_ms": _safe_float(latency.get("llm_ms")),
        "routing_ts_signals": _safe_float(routing.get("ts_signals")),
        "routing_llm_signals": _safe_float(routing.get("llm_signals")),
        "routing_fallback_used": _safe_float(routing.get("fallback_used")),
        "quality_average": _safe_float(quality.get("average")),
        "quality_minimum": _safe_float(quality.get("minimum")),
        "equity_last": _latest_series_value(payload.get("equity")),
        "equity_realized_last": _latest_series_value(payload.get("equity_realized")),
        "signal_count": float(len(payload.get("signals", []))) if isinstance(payload.get("signals"), list) else None,
        "trade_event_count": float(len(payload.get("trade_events", []))) if isinstance(payload.get("trade_events"), list) else None,
        "latest_signal_confidence": latest_signal["confidence"],
        "latest_signal_expected_return": latest_signal["expected_return"],
        "latest_signal_shares": latest_signal["shares"],
        "latest_signal_mid_slippage_bp": latest_signal["mid_slippage_bp"],
        "latest_trade_shares": latest_trade_event["shares"],
        "latest_trade_slippage_bp": latest_trade_event["slippage_bp"],
        "latest_trade_realized_pnl": latest_trade_event["realized_pnl"],
        "latest_trade_realized_pnl_pct": latest_trade_event["realized_pnl_pct"],
        "open_positions_count": position_summary["count"],
        "long_positions_count": position_summary["long_count"],
        "short_positions_count": position_summary["short_count"],
        "position_gross_notional": position_summary["gross_notional"],
        "position_net_notional": position_summary["net_notional"],
        "forecaster_profit_factor": _safe_float(forecaster_metrics.get("profit_factor")),
        "forecaster_win_rate": _safe_float(forecaster_metrics.get("win_rate")),
        "forecaster_rmse_ensemble": _safe_float(forecaster_rmse.get("ensemble")),
        "forecaster_rmse_baseline": _safe_float(forecaster_rmse.get("baseline")),
        "forecaster_rmse_ratio": _safe_float(forecaster_rmse.get("ratio")),
        "forecaster_profit_factor_min": _safe_float(forecaster_thresholds.get("profit_factor_min")),
        "forecaster_win_rate_min": _safe_float(forecaster_thresholds.get("win_rate_min")),
        "forecaster_rmse_ratio_max": _safe_float(forecaster_thresholds.get("rmse_ratio_max")),
        "forecaster_profit_factor_ok": 1.0 if forecaster_status.get("profit_factor_ok") is True else (0.0 if forecaster_status.get("profit_factor_ok") is False else None),
        "forecaster_win_rate_ok": 1.0 if forecaster_status.get("win_rate_ok") is True else (0.0 if forecaster_status.get("win_rate_ok") is False else None),
        "forecaster_rmse_ok": 1.0 if forecaster_status.get("rmse_ok") is True else (0.0 if forecaster_status.get("rmse_ok") is False else None),
        "forecaster_failed_checks": float(failed_checks) if forecaster_status else None,
        "quant_validation_total": quant_validation["total"],
        "quant_validation_pass_count": quant_validation["pass_count"],
        "quant_validation_fail_count": quant_validation["fail_count"],
        "quant_validation_fail_fraction": quant_validation["fail_fraction"],
        "quant_validation_negative_expected_profit_fraction": quant_validation["negative_expected_profit_fraction"],
        "quant_validation_max_fail_fraction": quant_validation["max_fail_fraction"],
        "quant_validation_max_negative_expected_profit_fraction": quant_validation["max_negative_expected_profit_fraction"],
        "operator_console_status_code": float(OPERATOR_CONSOLE_STATUS_CODES.get(str(operator_console.get("status") or "UNKNOWN").strip().upper(), 4))
        if operator_console
        else None,
        "operator_short_circuit_events": _safe_float(operator_activity.get("short_circuit_events")),
        "operator_tool_calls_recent": _safe_float(operator_activity.get("tool_calls")),
        "operator_issue_count": float(len(operator_issues)) if operator_console else None,
        "operator_reconnect_attempts": _safe_float(operator_maintenance.get("reconnect_attempts")),
        "operator_recovery_mode_code": float(RECOVERY_MODE_CODES.get(recovery_mode, 5)) if operator_console and recovery_mode else (0.0 if operator_console else None),
    }


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


def _load_required_cron_jobs_config(path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not path.exists():
        return {}, [f"missing:{path.name}"]
    try:
        import yaml  # type: ignore[import]
    except Exception as exc:
        return {}, [f"unavailable:{path.name}:{exc}"]
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {}, [f"invalid:{path.name}:{exc}"]
    jobs = raw.get("jobs")
    if not isinstance(jobs, dict):
        return {}, [f"invalid:{path.name}:jobs_not_mapping"]

    parsed: Dict[str, Dict[str, Any]] = {}
    for key, value in jobs.items():
        if not isinstance(value, dict):
            warnings.append(f"invalid:{path.name}:{key}:job_not_mapping")
            continue
        name = str(value.get("name") or key).strip()
        if not name:
            warnings.append(f"invalid:{path.name}:{key}:missing_name")
            continue
        expected_cadence_seconds = _safe_float(value.get("expected_cadence_seconds"))
        if expected_cadence_seconds is None or expected_cadence_seconds <= 0:
            warnings.append(f"invalid:{path.name}:{key}:expected_cadence_seconds")
            continue
        parsed[_slugify_job(name)] = {
            "name": name,
            "required_for_green": bool(value.get("required_for_green", False)),
            "severity": str(value.get("severity") or _severity_for_job(name) or "OTHER").strip().upper() or "OTHER",
            "expected_cadence_seconds": float(expected_cadence_seconds),
        }
    return parsed, warnings


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
        required_cron_jobs_path: Path = DEFAULT_REQUIRED_CRON_JOBS_PATH,
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
        self.required_cron_jobs_path = Path(required_cron_jobs_path)
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
            "shutdown_supported": True,
            "pid": os.getpid(),
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
        dashboard_summary = _dashboard_summary(dashboard_payload)

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
        phase3_ready = bool(_phase3_strict_ready(gate_payload))
        phase3_reason = (
            str(_phase3_strict_reason(gate_payload) or "").strip().upper()
            or str(gate_payload.get("status") or "").strip().upper()
            or ("READY" if phase3_ready else "FAIL")
        )
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
                **dashboard_summary,
            },
            "metrics_summary": {
                "generated_unixtime": _to_unixtime(metrics_ts),
                "status": metrics_status,
                "status_code": GENERIC_STATUS_CODES.get(metrics_status, 3),
            },
            "production_gate": {
                "generated_unixtime": _to_unixtime(gate_ts),
                "pass": 1.0 if phase3_ready else 0.0,
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

        required_jobs_config, config_warnings = _load_required_cron_jobs_config(self.required_cron_jobs_path)
        warnings.extend(config_warnings)
        if not required_jobs_config:
            return {"jobs": []}, warnings

        jobs = payload.get("jobs", [])
        if not isinstance(jobs, list):
            warnings.append("invalid:jobs.json:jobs_not_list")
            return {"jobs": []}, warnings

        job_rows: List[Dict[str, Any]] = []
        cache = self._persisted_state.setdefault("cron_last_success_ms", {})
        required_jobs = {
            slug: cfg for slug, cfg in required_jobs_config.items()
            if bool(cfg.get("required_for_green"))
        }
        seen_required_jobs: set[str] = set()
        now_ms = int(now.timestamp() * 1000)
        for job in jobs:
            if not isinstance(job, dict):
                continue
            state = job.get("state", {}) if isinstance(job.get("state"), dict) else {}
            schedule = job.get("schedule", {}) if isinstance(job.get("schedule"), dict) else {}
            name = str(job.get("name") or job.get("id") or "job").strip()
            job_label = _slugify_job(name)
            config = required_jobs_config.get(job_label)
            if config is None or not bool(config.get("required_for_green")):
                continue
            job_id = str(job.get("id") or job_label).strip()
            required_for_green = bool(config.get("required_for_green"))
            severity = str(config.get("severity") or _severity_for_job(name) or "OTHER").strip().upper() or "OTHER"
            enabled = 1.0 if bool(job.get("enabled", False)) and required_for_green else 0.0
            last_run_ms = int(state.get("lastRunAtMs") or 0)
            last_status = str(state.get("lastStatus") or state.get("lastRunStatus") or "unknown").strip().lower()
            if last_status == "success" and last_run_ms > 0:
                cache[job_id] = last_run_ms
                cache[job_label] = last_run_ms
            last_success_ms = int(cache.get(job_label) or cache.get(job_id) or 0)
            expected_interval_seconds = float(
                config.get("expected_cadence_seconds") or (_cron_interval_seconds(schedule) or 0.0)
            )
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
            if required_for_green:
                seen_required_jobs.add(job_label)

        for job_label, config in required_jobs.items():
            if job_label in seen_required_jobs:
                continue
            expected_interval_seconds = float(config.get("expected_cadence_seconds") or 0.0)
            last_success_ms = int(cache.get(job_label) or 0)
            freshness_lag_seconds = (
                max(0.0, (now_ms - last_success_ms) / 1000.0) if last_success_ms > 0 else 0.0
            )
            job_rows.append(
                {
                    "job": job_label,
                    "severity": str(config.get("severity") or "OTHER"),
                    "enabled": 0.0,
                    "last_run_unixtime": 0.0,
                    "last_success_unixtime": last_success_ms / 1000.0 if last_success_ms > 0 else 0.0,
                    "last_status_code": float(GENERIC_STATUS_CODES["MISSING"]),
                    "consecutive_errors": 0.0,
                    "expected_interval_seconds": expected_interval_seconds,
                    "stale_threshold_seconds": expected_interval_seconds * 2.0 if expected_interval_seconds > 0 else 0.0,
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
            dashboard_metric_specs = [
                ("pmx_dashboard_pnl_absolute", "Current PMX dashboard absolute PnL in account currency.", dashboard.get("pnl_absolute")),
                ("pmx_dashboard_pnl_pct", "Current PMX dashboard PnL as a fraction of portfolio equity.", dashboard.get("pnl_pct")),
                ("pmx_dashboard_win_rate", "Current PMX dashboard win rate.", dashboard.get("win_rate")),
                ("pmx_dashboard_trade_count", "Current PMX dashboard trade count.", dashboard.get("trade_count")),
                ("pmx_dashboard_signal_count", "Current PMX dashboard signal count.", dashboard.get("signal_count")),
                ("pmx_dashboard_trade_event_count", "Current PMX dashboard trade event count.", dashboard.get("trade_event_count")),
                ("pmx_dashboard_open_positions_count", "Current count of open positions in the PMX dashboard payload.", dashboard.get("open_positions_count")),
                ("pmx_dashboard_long_positions_count", "Current count of long positions in the PMX dashboard payload.", dashboard.get("long_positions_count")),
                ("pmx_dashboard_short_positions_count", "Current count of short positions in the PMX dashboard payload.", dashboard.get("short_positions_count")),
                ("pmx_dashboard_position_gross_notional", "Current gross notional exposure inferred from dashboard positions.", dashboard.get("position_gross_notional")),
                ("pmx_dashboard_position_net_notional", "Current net notional exposure inferred from dashboard positions.", dashboard.get("position_net_notional")),
                ("pmx_dashboard_latency_ts_ms", "Latest time-series signal latency from dashboard_data.json in milliseconds.", dashboard.get("latency_ts_ms")),
                ("pmx_dashboard_latency_llm_ms", "Latest LLM latency from dashboard_data.json in milliseconds.", dashboard.get("latency_llm_ms")),
                ("pmx_dashboard_quality_average", "Average dashboard quality score across rendered quality records.", dashboard.get("quality_average")),
                ("pmx_dashboard_quality_minimum", "Minimum dashboard quality score across rendered quality records.", dashboard.get("quality_minimum")),
                ("pmx_dashboard_equity_last", "Latest total equity value from dashboard_data.json.", dashboard.get("equity_last")),
                ("pmx_dashboard_equity_realized_last", "Latest realized equity value from dashboard_data.json.", dashboard.get("equity_realized_last")),
                ("pmx_dashboard_signal_count", "Latest PMX signal count rendered into dashboard_data.json.", dashboard.get("signal_count")),
                ("pmx_dashboard_trade_event_count", "Latest PMX trade-event count rendered into dashboard_data.json.", dashboard.get("trade_event_count")),
                ("pmx_dashboard_latest_signal_confidence", "Latest effective signal confidence rendered into dashboard_data.json.", dashboard.get("latest_signal_confidence")),
                ("pmx_dashboard_latest_signal_expected_return", "Latest expected return rendered into dashboard_data.json.", dashboard.get("latest_signal_expected_return")),
                ("pmx_dashboard_latest_signal_shares", "Latest signal share quantity rendered into dashboard_data.json.", dashboard.get("latest_signal_shares")),
                ("pmx_dashboard_latest_signal_mid_slippage_bp", "Latest signal mid-price slippage in basis points rendered into dashboard_data.json.", dashboard.get("latest_signal_mid_slippage_bp")),
                ("pmx_dashboard_latest_trade_shares", "Latest trade-event share quantity rendered into dashboard_data.json.", dashboard.get("latest_trade_shares")),
                ("pmx_dashboard_latest_trade_slippage_bp", "Latest trade-event slippage in basis points rendered into dashboard_data.json.", dashboard.get("latest_trade_slippage_bp")),
                ("pmx_dashboard_latest_trade_realized_pnl", "Latest realized trade PnL rendered into dashboard_data.json.", dashboard.get("latest_trade_realized_pnl")),
                ("pmx_dashboard_latest_trade_realized_pnl_pct", "Latest realized trade PnL percent rendered into dashboard_data.json.", dashboard.get("latest_trade_realized_pnl_pct")),
                ("pmx_routing_ts_signals", "Latest routed time-series signal count from dashboard_data.json.", dashboard.get("routing_ts_signals")),
                ("pmx_routing_llm_signals", "Latest routed LLM signal count from dashboard_data.json.", dashboard.get("routing_llm_signals")),
                ("pmx_routing_fallback_used", "Latest fallback routing count from dashboard_data.json.", dashboard.get("routing_fallback_used")),
                ("pmx_forecaster_profit_factor", "Latest PMX forecaster health profit factor metric.", dashboard.get("forecaster_profit_factor")),
                ("pmx_forecaster_win_rate", "Latest PMX forecaster health win rate metric.", dashboard.get("forecaster_win_rate")),
                ("pmx_forecaster_rmse_ensemble", "Latest PMX forecaster ensemble RMSE metric.", dashboard.get("forecaster_rmse_ensemble")),
                ("pmx_forecaster_rmse_baseline", "Latest PMX forecaster baseline RMSE metric.", dashboard.get("forecaster_rmse_baseline")),
                ("pmx_forecaster_rmse_ratio", "Latest PMX forecaster RMSE ratio metric.", dashboard.get("forecaster_rmse_ratio")),
                ("pmx_forecaster_profit_factor_min", "Configured PMX forecaster profit factor minimum threshold.", dashboard.get("forecaster_profit_factor_min")),
                ("pmx_forecaster_win_rate_min", "Configured PMX forecaster win rate diagnostic threshold (informational under barbell policy).", dashboard.get("forecaster_win_rate_min")),
                ("pmx_forecaster_rmse_ratio_max", "Configured PMX forecaster RMSE ratio maximum threshold.", dashboard.get("forecaster_rmse_ratio_max")),
                ("pmx_forecaster_profit_factor_ok", "Whether the PMX forecaster profit factor threshold is currently satisfied (1=yes,0=no).", dashboard.get("forecaster_profit_factor_ok")),
                ("pmx_forecaster_win_rate_ok", "Whether the PMX forecaster win-rate diagnostic threshold is currently satisfied (1=yes,0=no).", dashboard.get("forecaster_win_rate_ok")),
                ("pmx_forecaster_rmse_ok", "Whether the PMX forecaster RMSE threshold is currently satisfied (1=yes,0=no).", dashboard.get("forecaster_rmse_ok")),
                ("pmx_forecaster_failed_checks", "Count of failed PMX forecaster threshold checks in the latest dashboard payload.", dashboard.get("forecaster_failed_checks")),
                ("pmx_quant_validation_total", "Latest quant-validation total observation count from dashboard_data.json.", dashboard.get("quant_validation_total")),
                ("pmx_quant_validation_pass_count", "Latest quant-validation pass count from dashboard_data.json.", dashboard.get("quant_validation_pass_count")),
                ("pmx_quant_validation_fail_count", "Latest quant-validation fail count from dashboard_data.json.", dashboard.get("quant_validation_fail_count")),
                ("pmx_quant_validation_fail_fraction", "Latest quant-validation fail fraction from dashboard_data.json.", dashboard.get("quant_validation_fail_fraction")),
                ("pmx_quant_validation_negative_expected_profit_fraction", "Latest quant-validation negative expected-profit fraction from dashboard_data.json.", dashboard.get("quant_validation_negative_expected_profit_fraction")),
                ("pmx_quant_validation_max_fail_fraction", "Configured quant-validation maximum fail fraction threshold.", dashboard.get("quant_validation_max_fail_fraction")),
                ("pmx_quant_validation_max_negative_expected_profit_fraction", "Configured quant-validation maximum negative expected-profit fraction threshold.", dashboard.get("quant_validation_max_negative_expected_profit_fraction")),
                ("pmx_operator_console_status_code", "Operator console status code from dashboard_data.json (0=PASS/OK,1=WARN,2=FAIL,3=ERROR,4=UNKNOWN).", dashboard.get("operator_console_status_code")),
                ("pmx_operator_short_circuit_events", "Recent operator short-circuit or fast-path events surfaced through dashboard_data.json.", dashboard.get("operator_short_circuit_events")),
                ("pmx_operator_tool_calls_recent", "Recent operator tool-call count surfaced through dashboard_data.json.", dashboard.get("operator_tool_calls_recent")),
                ("pmx_operator_issue_count", "Current operator-console issue count surfaced through dashboard_data.json.", dashboard.get("operator_issue_count")),
                ("pmx_operator_reconnect_attempts", "Current primary-channel reconnect attempts surfaced through dashboard_data.json.", dashboard.get("operator_reconnect_attempts")),
                ("pmx_operator_recovery_mode_code", "Operator recovery mode code surfaced through dashboard_data.json.", dashboard.get("operator_recovery_mode_code")),
            ]
            for name, help_text, value in dashboard_metric_specs:
                if value is None:
                    continue
                registry.add(
                    name,
                    float(value),
                    help_text=help_text,
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
        required_cron_jobs_path=Path(args.required_cron_jobs_path),
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
    parser.add_argument("--required-cron-jobs-path", default=str(DEFAULT_REQUIRED_CRON_JOBS_PATH))
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
