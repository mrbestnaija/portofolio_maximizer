#!/usr/bin/env python3
"""
Prometheus alert exporter for Portfolio Maximizer.

This exporter deliberately reads the same canonical artifacts used by the
static evidence dashboard instead of inventing a parallel source of truth.
It is intended for alerting only; the human-facing run/evidence view remains
`visualizations/live_dashboard.html`.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from scripts import dashboard_db_bridge as bridge_mod
except Exception:  # pragma: no cover - standalone fallback
    bridge_mod = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD_JSON = ROOT / "visualizations" / "dashboard_data.json"
DEFAULT_PRODUCTION_GATE_JSON = ROOT / "logs" / "audit_gate" / "production_gate_latest.json"
DEFAULT_AUDIT_DB = ROOT / "data" / "dashboard_audit.db"

DEFAULT_DASHBOARD_STALE_SECONDS = 60
DEFAULT_PRODUCTION_GATE_STALE_SECONDS = 30 * 60
DEFAULT_AUDIT_STALE_SECONDS = 60 * 60


def _policy_seconds(env_name: str, fallback_minutes: float) -> float:
    try:
        return float(os.getenv(env_name, str(fallback_minutes))) * 60.0
    except Exception:
        return float(fallback_minutes) * 60.0


def _production_gate_stale_seconds() -> float:
    fallback_minutes = (
        float(getattr(bridge_mod, "DEFAULT_PRODUCTION_GATE_MAX_AGE_MINUTES"))
        if bridge_mod is not None
        else float(DEFAULT_PRODUCTION_GATE_STALE_SECONDS) / 60.0
    )
    return _policy_seconds("PMX_PRODUCTION_GATE_MAX_AGE_MINUTES", fallback_minutes)


def _audit_stale_seconds() -> float:
    fallback_minutes = (
        float(getattr(bridge_mod, "DEFAULT_AUDIT_SNAPSHOT_MAX_AGE_MINUTES"))
        if bridge_mod is not None
        else float(DEFAULT_AUDIT_STALE_SECONDS) / 60.0
    )
    return _policy_seconds("PMX_AUDIT_SNAPSHOT_MAX_AGE_MINUTES", fallback_minutes)


def _safe_load_json(path: Path) -> tuple[Dict[str, Any], Optional[str]]:
    if not path.exists():
        return {}, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, "unreadable"
    if not isinstance(payload, dict):
        return {}, "invalid"
    return payload, None


def _parse_utc_datetime(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _age_seconds(
    path: Path,
    *,
    payload: Optional[Dict[str, Any]] = None,
    candidate_keys: tuple[str, ...] = ("generated_utc", "timestamp_utc"),
    now: Optional[datetime] = None,
) -> Optional[float]:
    ref = None
    payload = payload or {}
    for key in candidate_keys:
        parsed = _parse_utc_datetime(payload.get(key))
        if parsed is not None:
            ref = parsed
            break
    if ref is None:
        try:
            ref = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return None
    now = now or datetime.now(timezone.utc)
    age = (now - ref).total_seconds()
    return age if age >= 0 else 0.0


def _audit_snapshot_summary(path: Path, *, now: Optional[datetime] = None) -> Dict[str, Any]:
    summary = {
        "present": path.exists(),
        "status": "MISSING",
        "snapshot_count": 0,
        "latest_created_at": None,
        "age_seconds": None,
        "error": None,
    }
    if not path.exists():
        return summary

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        count_row = conn.execute("SELECT COUNT(*) AS c FROM dashboard_snapshots").fetchone()
        latest_row = conn.execute(
            """
            SELECT created_at, run_id
            FROM dashboard_snapshots
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        summary["snapshot_count"] = int(count_row["c"] or 0) if count_row else 0
        if latest_row and latest_row["created_at"]:
            summary["latest_created_at"] = str(latest_row["created_at"])
            summary["age_seconds"] = _age_seconds(
                path,
                payload={"generated_utc": summary["latest_created_at"]},
                now=now,
            )
        summary["status"] = "OK" if summary["snapshot_count"] > 0 else "EMPTY"
        return summary
    except Exception as exc:
        summary["status"] = "ERROR"
        summary["error"] = str(exc)
        return summary
    finally:
        if conn is not None:
            conn.close()


def collect_metrics_snapshot(
    *,
    dashboard_json: Path = DEFAULT_DASHBOARD_JSON,
    production_gate_json: Path = DEFAULT_PRODUCTION_GATE_JSON,
    audit_db: Path = DEFAULT_AUDIT_DB,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    production_gate_stale_seconds = _production_gate_stale_seconds()
    audit_stale_seconds = _audit_stale_seconds()
    dashboard, dash_err = _safe_load_json(dashboard_json)
    gate_payload, gate_err = _safe_load_json(production_gate_json)
    audit = _audit_snapshot_summary(audit_db, now=now)

    dashboard_meta = dashboard.get("meta", {}) if isinstance(dashboard, dict) else {}
    dashboard_age = _age_seconds(
        dashboard_json,
        payload={"generated_utc": dashboard_meta.get("ts")},
        now=now,
    ) if dash_err is None else None
    trade_events = dashboard.get("trade_events", []) if isinstance(dashboard, dict) else []
    signals = dashboard.get("signals", []) if isinstance(dashboard, dict) else []
    price_series = dashboard.get("price_series", {}) if isinstance(dashboard, dict) else {}
    alerts = dashboard.get("alerts", []) if isinstance(dashboard, dict) else []
    evidence = dashboard.get("evidence", {}) if isinstance(dashboard, dict) else {}
    robustness = dashboard.get("robustness", {}) if isinstance(dashboard, dict) else {}
    gate_from_dashboard = evidence.get("production_gate", {}) if isinstance(evidence, dict) else {}
    gate_refresh = evidence.get("production_gate_refresh", {}) if isinstance(evidence, dict) else {}

    gate = gate_payload.get("production_profitability_gate", {}) if isinstance(gate_payload, dict) else {}
    proof = gate_payload.get("profitability_proof", {}) if isinstance(gate_payload, dict) else {}
    artifact_binding = gate_payload.get("artifact_binding", {}) if isinstance(gate_payload, dict) else {}
    gate_age = _age_seconds(
        production_gate_json,
        payload={
            "generated_utc": gate_payload.get("timestamp_utc")
            or (gate_payload.get("telemetry_contract", {}) or {}).get("generated_utc")
        } if isinstance(gate_payload, dict) else {},
        now=now,
    ) if gate_err is None else None

    origin = str(
        dashboard_meta.get("data_origin")
        or (dashboard_meta.get("provenance", {}) or {}).get("origin")
        or "unknown"
    ).strip().lower() or "unknown"
    robustness_status = str(
        robustness.get("overall_status") or robustness.get("status") or "UNKNOWN"
    ).upper()
    gate_status = str(
        gate.get("status")
        or gate_from_dashboard.get("status")
        or "UNKNOWN"
    ).upper()

    remaining_closed = int(
        (((proof.get("evidence_progress", {}) or {}).get("remaining_closed_trades")) or 0)
    ) if isinstance(proof, dict) else 0
    remaining_days = int(
        (((proof.get("evidence_progress", {}) or {}).get("remaining_trading_days")) or 0)
    ) if isinstance(proof, dict) else 0
    gate_generated_utc = (
        str(gate_from_dashboard.get("generated_utc") or gate_payload.get("timestamp_utc") or "").strip()
        or None
    )
    refresh_last_success_generated_utc = str(
        gate_refresh.get("last_success_generated_utc") or ""
    ).strip() or None
    refresh_last_success_age_seconds = _age_seconds(
        dashboard_json,
        payload={"generated_utc": gate_refresh.get("last_success_utc")},
        now=now,
    ) if isinstance(gate_refresh, dict) and gate_refresh.get("last_success_utc") else None
    refresh_status = str(
        gate_refresh.get("status")
        or ("DISABLED" if gate_refresh.get("enabled") is False else "UNKNOWN")
    ).upper()
    refresh_actor = str(gate_refresh.get("actor") or "unknown").strip().lower() or "unknown"
    refresh_last_success_actor = (
        str(gate_refresh.get("last_success_actor") or "unknown").strip().lower() or "unknown"
    )

    alert_flags = {
        "dashboard_payload_missing": 1 if dash_err is not None else 0,
        "dashboard_payload_stale": 1 if dashboard_age is not None and dashboard_age > DEFAULT_DASHBOARD_STALE_SECONDS else 0,
        "production_gate_missing": 1 if gate_err is not None else 0,
        "production_gate_stale": 1 if gate_age is not None and gate_age > production_gate_stale_seconds else 0,
        "production_gate_fail": 1 if gate_status != "PASS" else 0,
        "artifact_binding_fail": 1 if isinstance(artifact_binding, dict) and artifact_binding.get("pass") is False else 0,
        "proof_runway_incomplete": 1 if remaining_closed > 0 or remaining_days > 0 else 0,
        "dashboard_audit_missing": 1 if (not audit["present"] or str(audit.get("status") or "").upper() == "EMPTY") else 0,
        "dashboard_audit_error": 1 if str(audit.get("status") or "").upper() == "ERROR" else 0,
        "dashboard_audit_stale": 1 if audit["age_seconds"] is not None and audit["age_seconds"] > audit_stale_seconds else 0,
        "data_origin_not_live": 1 if origin != "live" else 0,
        "robustness_degraded": 1 if robustness_status not in {"OK", "PASS"} else 0,
    }

    return {
        "generated_utc": now.isoformat(),
        "dashboard": {
            "present": dash_err is None,
            "age_seconds": dashboard_age,
            "signals_total": len(signals) if isinstance(signals, list) else 0,
            "trade_events_total": len(trade_events) if isinstance(trade_events, list) else 0,
            "price_series_total": len(price_series) if isinstance(price_series, dict) else 0,
            "operator_alerts_total": len(alerts) if isinstance(alerts, list) else 0,
            "origin": origin,
            "robustness_status": robustness_status,
        },
        "gate": {
            "present": gate_err is None,
            "age_seconds": gate_age,
            "status": gate_status,
            "pass": 1 if gate_status == "PASS" else 0,
            "phase3_ready": 1 if bool(gate_payload.get("phase3_ready")) else 0,
            "artifact_binding_pass": 1 if isinstance(artifact_binding, dict) and bool(artifact_binding.get("pass")) else 0,
            "remaining_closed_trades": remaining_closed,
            "remaining_trading_days": remaining_days,
        },
        "gate_refresh": {
            "enabled": 1 if bool(gate_refresh.get("enabled")) else 0,
            "attempted": 1 if bool(gate_refresh.get("attempted")) else 0,
            "ok": 1 if gate_refresh.get("ok") is True else 0,
            "status": refresh_status,
            "actor": refresh_actor,
            "last_success_actor": refresh_last_success_actor,
            "last_success_age_seconds": refresh_last_success_age_seconds,
            "last_success_same_artifact": 1
            if (
                gate_generated_utc is not None
                and refresh_last_success_generated_utc is not None
                and gate_generated_utc == refresh_last_success_generated_utc
            )
            else 0,
        },
        "audit": audit,
        "policies": {
            "production_gate_stale_seconds": production_gate_stale_seconds,
            "audit_stale_seconds": audit_stale_seconds,
        },
        "alerts": alert_flags,
    }


def _metric_line(name: str, value: Any) -> str:
    if value is None:
        value = 0
    if isinstance(value, bool):
        value = 1 if value else 0
    return f"{name} {value}"


def _enum_metric_lines(
    name: str,
    active: str,
    choices: tuple[str, ...],
    *,
    label_name: str = "state",
) -> list[str]:
    active_normalized = str(active or "unknown").strip().lower() or "unknown"
    return [
        f'{name}{{{label_name}="{choice}"}} {1 if active_normalized == choice else 0}'
        for choice in choices
    ]


def render_metrics(snapshot: Dict[str, Any]) -> str:
    dashboard = snapshot.get("dashboard", {})
    gate = snapshot.get("gate", {})
    gate_refresh = snapshot.get("gate_refresh", {})
    audit = snapshot.get("audit", {})
    policies = snapshot.get("policies", {})
    alerts = snapshot.get("alerts", {})

    lines = [
        "# HELP pmx_dashboard_payload_present 1 when the canonical dashboard payload is present and parseable.",
        "# TYPE pmx_dashboard_payload_present gauge",
        _metric_line("pmx_dashboard_payload_present", dashboard.get("present", 0)),
        "# HELP pmx_dashboard_payload_age_seconds Age of visualizations/dashboard_data.json in seconds.",
        "# TYPE pmx_dashboard_payload_age_seconds gauge",
        _metric_line("pmx_dashboard_payload_age_seconds", dashboard.get("age_seconds", 0)),
        "# HELP pmx_dashboard_signals_total Number of signals in the canonical payload.",
        "# TYPE pmx_dashboard_signals_total gauge",
        _metric_line("pmx_dashboard_signals_total", dashboard.get("signals_total", 0)),
        "# HELP pmx_dashboard_trade_events_total Number of trade events in the canonical payload.",
        "# TYPE pmx_dashboard_trade_events_total gauge",
        _metric_line("pmx_dashboard_trade_events_total", dashboard.get("trade_events_total", 0)),
        "# HELP pmx_dashboard_price_series_total Number of price series included in the canonical payload.",
        "# TYPE pmx_dashboard_price_series_total gauge",
        _metric_line("pmx_dashboard_price_series_total", dashboard.get("price_series_total", 0)),
        "# HELP pmx_dashboard_operator_alerts_total Number of operator-facing evidence alerts in the payload.",
        "# TYPE pmx_dashboard_operator_alerts_total gauge",
        _metric_line("pmx_dashboard_operator_alerts_total", dashboard.get("operator_alerts_total", 0)),
        "# HELP pmx_production_gate_artifact_present 1 when the production gate artifact is present.",
        "# TYPE pmx_production_gate_artifact_present gauge",
        _metric_line("pmx_production_gate_artifact_present", gate.get("present", 0)),
        "# HELP pmx_production_gate_artifact_age_seconds Age of logs/audit_gate/production_gate_latest.json in seconds.",
        "# TYPE pmx_production_gate_artifact_age_seconds gauge",
        _metric_line("pmx_production_gate_artifact_age_seconds", gate.get("age_seconds", 0)),
        "# HELP pmx_production_gate_pass 1 when the production gate passes.",
        "# TYPE pmx_production_gate_pass gauge",
        _metric_line("pmx_production_gate_pass", gate.get("pass", 0)),
        "# HELP pmx_production_gate_phase3_ready 1 when phase3_ready is true.",
        "# TYPE pmx_production_gate_phase3_ready gauge",
        _metric_line("pmx_production_gate_phase3_ready", gate.get("phase3_ready", 0)),
        "# HELP pmx_production_gate_artifact_binding_pass 1 when artifact binding passes.",
        "# TYPE pmx_production_gate_artifact_binding_pass gauge",
        _metric_line("pmx_production_gate_artifact_binding_pass", gate.get("artifact_binding_pass", 0)),
        "# HELP pmx_proof_remaining_closed_trades Remaining closed trades required for proof runway.",
        "# TYPE pmx_proof_remaining_closed_trades gauge",
        _metric_line("pmx_proof_remaining_closed_trades", gate.get("remaining_closed_trades", 0)),
        "# HELP pmx_proof_remaining_trading_days Remaining trading days required for proof runway.",
        "# TYPE pmx_proof_remaining_trading_days gauge",
        _metric_line("pmx_proof_remaining_trading_days", gate.get("remaining_trading_days", 0)),
        "# HELP pmx_production_gate_refresh_enabled 1 when the canonical payload includes production gate refresh metadata.",
        "# TYPE pmx_production_gate_refresh_enabled gauge",
        _metric_line("pmx_production_gate_refresh_enabled", gate_refresh.get("enabled", 0)),
        "# HELP pmx_production_gate_refresh_attempted 1 when the current refresh state reflects an active attempt.",
        "# TYPE pmx_production_gate_refresh_attempted gauge",
        _metric_line("pmx_production_gate_refresh_attempted", gate_refresh.get("attempted", 0)),
        "# HELP pmx_production_gate_refresh_ok 1 when the current refresh state is a successful attempt.",
        "# TYPE pmx_production_gate_refresh_ok gauge",
        _metric_line("pmx_production_gate_refresh_ok", gate_refresh.get("ok", 0)),
        "# HELP pmx_production_gate_refresh_last_success_age_seconds Age of the last successful production gate refresh recorded in the canonical payload.",
        "# TYPE pmx_production_gate_refresh_last_success_age_seconds gauge",
        _metric_line("pmx_production_gate_refresh_last_success_age_seconds", gate_refresh.get("last_success_age_seconds", 0)),
        "# HELP pmx_production_gate_refresh_last_success_same_artifact 1 when the last successful refresh matches the current gate artifact timestamp.",
        "# TYPE pmx_production_gate_refresh_last_success_same_artifact gauge",
        _metric_line("pmx_production_gate_refresh_last_success_same_artifact", gate_refresh.get("last_success_same_artifact", 0)),
        "# HELP pmx_dashboard_audit_db_present 1 when data/dashboard_audit.db exists.",
        "# TYPE pmx_dashboard_audit_db_present gauge",
        _metric_line("pmx_dashboard_audit_db_present", audit.get("present", 0)),
        "# HELP pmx_dashboard_audit_snapshots_total Number of persisted dashboard snapshots.",
        "# TYPE pmx_dashboard_audit_snapshots_total gauge",
        _metric_line("pmx_dashboard_audit_snapshots_total", audit.get("snapshot_count", 0)),
        "# HELP pmx_dashboard_audit_snapshot_age_seconds Age of the latest persisted dashboard snapshot.",
        "# TYPE pmx_dashboard_audit_snapshot_age_seconds gauge",
        _metric_line("pmx_dashboard_audit_snapshot_age_seconds", audit.get("age_seconds", 0)),
        "# HELP pmx_alert_dashboard_payload_missing 1 when dashboard_data.json is missing or unreadable.",
        "# TYPE pmx_alert_dashboard_payload_missing gauge",
        _metric_line("pmx_alert_dashboard_payload_missing", alerts.get("dashboard_payload_missing", 0)),
        "# HELP pmx_alert_dashboard_payload_stale 1 when dashboard_data.json is stale.",
        "# TYPE pmx_alert_dashboard_payload_stale gauge",
        _metric_line("pmx_alert_dashboard_payload_stale", alerts.get("dashboard_payload_stale", 0)),
        "# HELP pmx_alert_production_gate_missing 1 when production_gate_latest.json is missing or unreadable.",
        "# TYPE pmx_alert_production_gate_missing gauge",
        _metric_line("pmx_alert_production_gate_missing", alerts.get("production_gate_missing", 0)),
        "# HELP pmx_alert_production_gate_stale 1 when the production gate artifact is older than the canonical freshness policy.",
        "# TYPE pmx_alert_production_gate_stale gauge",
        _metric_line("pmx_alert_production_gate_stale", alerts.get("production_gate_stale", 0)),
        "# HELP pmx_alert_production_gate_fail 1 when the production gate is not PASS.",
        "# TYPE pmx_alert_production_gate_fail gauge",
        _metric_line("pmx_alert_production_gate_fail", alerts.get("production_gate_fail", 0)),
        "# HELP pmx_alert_artifact_binding_fail 1 when artifact binding fails.",
        "# TYPE pmx_alert_artifact_binding_fail gauge",
        _metric_line("pmx_alert_artifact_binding_fail", alerts.get("artifact_binding_fail", 0)),
        "# HELP pmx_alert_proof_runway_incomplete 1 when proof runway still has remaining trades or days.",
        "# TYPE pmx_alert_proof_runway_incomplete gauge",
        _metric_line("pmx_alert_proof_runway_incomplete", alerts.get("proof_runway_incomplete", 0)),
        "# HELP pmx_alert_dashboard_audit_missing 1 when the audit DB is missing or has zero snapshots.",
        "# TYPE pmx_alert_dashboard_audit_missing gauge",
        _metric_line("pmx_alert_dashboard_audit_missing", alerts.get("dashboard_audit_missing", 0)),
        "# HELP pmx_alert_dashboard_audit_error 1 when the audit DB exists but cannot be queried cleanly.",
        "# TYPE pmx_alert_dashboard_audit_error gauge",
        _metric_line("pmx_alert_dashboard_audit_error", alerts.get("dashboard_audit_error", 0)),
        "# HELP pmx_alert_dashboard_audit_stale 1 when the latest audit snapshot is stale.",
        "# TYPE pmx_alert_dashboard_audit_stale gauge",
        _metric_line("pmx_alert_dashboard_audit_stale", alerts.get("dashboard_audit_stale", 0)),
        "# HELP pmx_alert_data_origin_not_live 1 when dashboard origin is not live.",
        "# TYPE pmx_alert_data_origin_not_live gauge",
        _metric_line("pmx_alert_data_origin_not_live", alerts.get("data_origin_not_live", 0)),
        "# HELP pmx_alert_robustness_degraded 1 when robustness is not OK/PASS.",
        "# TYPE pmx_alert_robustness_degraded gauge",
        _metric_line("pmx_alert_robustness_degraded", alerts.get("robustness_degraded", 0)),
        "# HELP pmx_policy_production_gate_stale_seconds Canonical freshness threshold used for the production gate artifact.",
        "# TYPE pmx_policy_production_gate_stale_seconds gauge",
        _metric_line("pmx_policy_production_gate_stale_seconds", policies.get("production_gate_stale_seconds", 0)),
        "# HELP pmx_policy_dashboard_audit_stale_seconds Canonical freshness threshold used for dashboard audit snapshots.",
        "# TYPE pmx_policy_dashboard_audit_stale_seconds gauge",
        _metric_line("pmx_policy_dashboard_audit_stale_seconds", policies.get("audit_stale_seconds", 0)),
    ]

    lines.extend(
        _enum_metric_lines(
            "pmx_dashboard_origin_state",
            str(dashboard.get("origin") or "unknown"),
            ("live", "synthetic", "mixed", "unknown"),
            label_name="origin",
        )
    )
    lines.extend(
        _enum_metric_lines(
            "pmx_dashboard_robustness_status",
            str(dashboard.get("robustness_status") or "UNKNOWN").lower(),
            ("ok", "pass", "warn", "stale", "missing", "unknown"),
            label_name="status",
        )
    )
    lines.extend(
        _enum_metric_lines(
            "pmx_production_gate_status",
            str(gate.get("status") or "UNKNOWN").lower(),
            ("pass", "fail", "inconclusive", "inconclusive_blocked", "unknown"),
            label_name="status",
        )
    )
    lines.extend(
        _enum_metric_lines(
            "pmx_production_gate_refresh_status",
            str(gate_refresh.get("status") or "UNKNOWN").lower(),
            ("ok", "error", "skipped", "disabled", "unknown"),
            label_name="status",
        )
    )
    lines.extend(
        _enum_metric_lines(
            "pmx_production_gate_refresh_actor",
            str(gate_refresh.get("actor") or "unknown").lower(),
            ("dashboard_launch", "dashboard_bridge", "unknown"),
            label_name="actor",
        )
    )
    lines.extend(
        _enum_metric_lines(
            "pmx_production_gate_refresh_last_success_actor",
            str(gate_refresh.get("last_success_actor") or "unknown").lower(),
            ("dashboard_launch", "dashboard_bridge", "unknown"),
            label_name="actor",
        )
    )
    return "\n".join(lines) + "\n"


def _build_handler(args: argparse.Namespace):
    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/metrics", "/"):
                self.send_response(404)
                self.end_headers()
                return
            snapshot = collect_metrics_snapshot(
                dashboard_json=Path(args.dashboard_json),
                production_gate_json=Path(args.production_gate_json),
                audit_db=Path(args.audit_db),
            )
            payload = render_metrics(snapshot).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return MetricsHandler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose PMX alert metrics for Prometheus.")
    parser.add_argument("--dashboard-json", default=str(DEFAULT_DASHBOARD_JSON), help="Canonical dashboard JSON path.")
    parser.add_argument("--production-gate-json", default=str(DEFAULT_PRODUCTION_GATE_JSON), help="Production gate artifact path.")
    parser.add_argument("--audit-db", default=str(DEFAULT_AUDIT_DB), help="Dashboard audit SQLite DB path.")
    parser.add_argument("--listen-host", default="127.0.0.1", help="Bind host (default: localhost only).")
    parser.add_argument("--port", type=int, default=9108, help="Exporter port.")
    parser.add_argument("--once", action="store_true", help="Print metrics once and exit.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.once:
        snapshot = collect_metrics_snapshot(
            dashboard_json=Path(args.dashboard_json),
            production_gate_json=Path(args.production_gate_json),
            audit_db=Path(args.audit_db),
        )
        print(render_metrics(snapshot), end="")
        return 0

    server = ThreadingHTTPServer((str(args.listen_host), int(args.port)), _build_handler(args))
    try:
        print(f"[PROMETHEUS] listening on http://{args.listen_host}:{args.port}/metrics")
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
