#!/usr/bin/env python3
"""
Local Alertmanager webhook bridge for PMX.

Receives Alertmanager webhook POSTs on localhost and reuses existing OpenClaw
and SMTP/email delivery paths without putting Prometheus on the trade path.
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import subprocess
import sys
import threading
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 9766
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "observability" / "alertmanager_webhooks.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_secret(name: str) -> str:
    try:
        from etl.secret_loader import load_secret

        value = load_secret(name)
        return str(value or "").strip()
    except Exception:
        return str(os.getenv(name) or "").strip()


def _bool_env(name: str, default: bool) -> bool:
    text = str(os.getenv(name) or "").strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _render_alertmanager_message(payload: Dict[str, Any]) -> str:
    alerts = payload.get("alerts", [])
    if not isinstance(alerts, list):
        alerts = []
    status = str(payload.get("status") or "unknown").strip().upper()
    lines = [f"[PMX Alertmanager] {status} ({len(alerts)} alert{'s' if len(alerts) != 1 else ''})"]
    for alert in alerts:
        if not isinstance(alert, dict):
            continue
        labels = alert.get("labels", {}) if isinstance(alert.get("labels"), dict) else {}
        annotations = alert.get("annotations", {}) if isinstance(alert.get("annotations"), dict) else {}
        name = str(labels.get("alertname") or "alert").strip()
        severity = str(labels.get("severity") or "info").strip()
        summary = str(annotations.get("summary") or annotations.get("description") or "").strip()
        job = str(labels.get("job") or "").strip()
        channel = str(labels.get("channel") or "").strip()
        line = f"- {name} [{severity}]"
        if job:
            line += f" job={job}"
        if channel:
            line += f" channel={channel}"
        if summary:
            line += f" :: {summary}"
        lines.append(line)
    return "\n".join(lines)


def _write_webhook_log(path: Path, payload: Dict[str, Any], *, delivery_mode: str, message: str) -> None:
    _ensure_dir(path)
    record = {
        "timestamp_utc": _utc_now_iso(),
        "delivery_mode": delivery_mode,
        "message_preview": message[:400],
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _send_openclaw(message: str, *, shadow_mode: bool) -> Tuple[bool, str]:
    shadow_targets = str(os.getenv("PMX_OBSERVABILITY_OPENCLAW_SHADOW_TARGETS") or "").strip()
    live_targets = str(os.getenv("OPENCLAW_TARGETS") or "").strip()
    live_to = str(os.getenv("OPENCLAW_TO") or "").strip()
    targets = shadow_targets if shadow_mode else live_targets
    args = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "openclaw_notify.py"),
        "--message",
        message,
    ]
    if targets:
        args.extend(["--targets", targets])
    elif live_to:
        args.extend(["--to", live_to])
    else:
        return False, "no_openclaw_targets"
    proc = subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ok = int(proc.returncode) == 0
    detail = (proc.stdout or proc.stderr or "").strip()[:200]
    return ok, detail or ("ok" if ok else "openclaw_send_failed")


def _send_email(message: str, *, shadow_mode: bool) -> Tuple[bool, str]:
    recipients_raw = (
        str(os.getenv("PMX_EMAIL_TEST_TO") or "").strip()
        if shadow_mode
        else str(os.getenv("PMX_EMAIL_TO") or os.getenv("PMX_EMAIL_RECIPIENTS") or "").strip()
    )
    recipients = [item.strip() for item in recipients_raw.split(",") if item.strip()]
    if not recipients:
        return False, "no_email_recipients"

    smtp_server = str(os.getenv("PMX_EMAIL_SMTP_SERVER") or "").strip()
    smtp_port = int(str(os.getenv("PMX_EMAIL_SMTP_PORT") or "587").strip() or "587")
    username = _load_secret("PMX_EMAIL_USERNAME")
    password = _load_secret("PMX_EMAIL_PASSWORD")
    from_address = _load_secret("PMX_EMAIL_FROM") or username
    if not smtp_server or not username or not password or not from_address:
        return False, "missing_email_config"

    subject = "[PMX Shadow Alert]" if shadow_mode else "[PMX Alertmanager]"
    msg = MIMEMultipart()
    msg["From"] = from_address
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(message, "plain", "utf-8"))

    timeout_seconds = float(str(os.getenv("PMX_EMAIL_TIMEOUT_SECONDS") or "10").strip() or "10")
    with smtplib.SMTP(smtp_server, smtp_port, timeout=timeout_seconds) as server:
        server.starttls()
        server.login(username, password)
        server.sendmail(from_address, recipients, msg.as_string())
    return True, "sent"


class AlertBridge:
    def __init__(self, *, log_path: Path = DEFAULT_LOG_PATH, shadow_mode: bool = True) -> None:
        self.log_path = Path(log_path)
        self.shadow_mode = bool(shadow_mode)

    def handle_alertmanager_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        message = _render_alertmanager_message(payload)
        delivery_mode = "shadow_log_only" if self.shadow_mode else "live"
        openclaw_result = {"attempted": False, "ok": False, "detail": "disabled"}
        email_result = {"attempted": False, "ok": False, "detail": "disabled"}

        if self.shadow_mode:
            if str(os.getenv("PMX_OBSERVABILITY_OPENCLAW_SHADOW_TARGETS") or "").strip():
                ok, detail = _send_openclaw(message, shadow_mode=True)
                openclaw_result = {"attempted": True, "ok": ok, "detail": detail}
                delivery_mode = "shadow_openclaw"
            if str(os.getenv("PMX_EMAIL_TEST_TO") or "").strip():
                ok, detail = _send_email(message, shadow_mode=True)
                email_result = {"attempted": True, "ok": ok, "detail": detail}
                delivery_mode = "shadow_email" if not openclaw_result["attempted"] else "shadow_multi"
        else:
            ok, detail = _send_openclaw(message, shadow_mode=False)
            openclaw_result = {"attempted": True, "ok": ok, "detail": detail}
            ok, detail = _send_email(message, shadow_mode=False)
            email_result = {"attempted": True, "ok": ok, "detail": detail}

        _write_webhook_log(self.log_path, payload, delivery_mode=delivery_mode, message=message)
        return {
            "timestamp_utc": _utc_now_iso(),
            "delivery_mode": delivery_mode,
            "shadow_mode": self.shadow_mode,
            "openclaw": openclaw_result,
            "email": email_result,
            "message_preview": message[:200],
        }


class _BridgeHandler(BaseHTTPRequestHandler):
    bridge: AlertBridge

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/healthz"):
            payload = json.dumps(
                {
                    "status": "ok",
                    "shadow_mode": self.bridge.shadow_mode,
                    "shutdown_supported": True,
                    "pid": os.getpid(),
                }
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

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
        if not self.path.startswith("/alertmanager"):
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return
        content_length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.end_headers()
            return
        result = self.bridge.handle_alertmanager_payload(payload if isinstance(payload, dict) else {})
        body = json.dumps(result, sort_keys=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PMX Alertmanager bridge")
    parser.add_argument("--bind", default=DEFAULT_BIND)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--shadow-mode", action="store_true", default=_bool_env("PMX_OBSERVABILITY_ALERT_SHADOW_MODE", True))
    parser.add_argument("--live-mode", dest="shadow_mode", action="store_false")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    bridge = AlertBridge(log_path=Path(args.log_path), shadow_mode=bool(args.shadow_mode))
    _BridgeHandler.bridge = bridge
    server = ThreadingHTTPServer((str(args.bind), int(args.port)), _BridgeHandler)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
