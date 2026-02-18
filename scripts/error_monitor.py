#!/usr/bin/env python3
"""
Enhanced Error Monitoring System
Monitors system errors and provides automated alerting
"""

import os
import sys
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class ErrorMonitor:
    """
    Enhanced error monitoring system with automated alerting
    """

    def __init__(self, config_file: str = "config/error_monitoring_config.yml"):
        self.config_file = config_file
        self.config = self._load_config()
        self.error_thresholds = {
            'max_errors_per_hour': 5,
            'max_errors_per_day': 20,
            'critical_error_types': ['TypeError', 'ValueError', 'ConnectionError'],
            'alert_cooldown_minutes': 30
        }
        self._apply_threshold_overrides()
        self.last_alert_time = None
        self.error_counts = {}
        self.alert_config = self._get_alert_config()

    def _load_config(self) -> Dict[str, Any]:
        path = Path(self.config_file)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning("Failed to load %s (%s); using defaults", path, exc)
            return {}

    def _apply_threshold_overrides(self) -> None:
        overrides = self.config.get("error_thresholds") if isinstance(self.config, dict) else None
        if not isinstance(overrides, dict):
            return

        for key in ("max_errors_per_hour", "max_errors_per_day", "alert_cooldown_minutes"):
            if key in overrides and overrides[key] is not None:
                try:
                    self.error_thresholds[key] = int(overrides[key])
                except Exception:
                    pass

        critical_types = overrides.get("critical_error_types")
        if isinstance(critical_types, list):
            self.error_thresholds["critical_error_types"] = [str(v) for v in critical_types if v]

    def _get_alert_config(self) -> Dict[str, Any]:
        alerts = self.config.get("alerts") if isinstance(self.config, dict) else None
        return alerts if isinstance(alerts, dict) else {}

    def monitor_errors(self) -> Dict[str, Any]:
        """
        Monitor current error status and generate alerts if needed
        """
        try:
            # Check error log
            error_log_path = "logs/errors/errors.log"
            if not os.path.exists(error_log_path):
                return {"status": "no_errors", "message": "No error log found"}

            # Analyze recent errors
            recent_errors = self._analyze_recent_errors(error_log_path)

            # Check error thresholds
            alert_needed = self._check_error_thresholds(recent_errors)

            # Generate alert if needed
            if alert_needed:
                self._send_alert(recent_errors)

            return {
                "status": "monitored",
                "recent_errors": recent_errors,
                "alert_sent": alert_needed,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error monitoring failed: {e}")
            return {"status": "error", "message": str(e)}

    def _analyze_recent_errors(self, error_log_path: str) -> Dict[str, Any]:
        """Analyze recent errors from log file"""
        try:
            with open(error_log_path, 'r') as f:
                lines = f.readlines()

            now = datetime.now()
            cutoff_24h = now - timedelta(hours=24)
            cutoff_1h = now - timedelta(hours=1)
            recent_errors = []
            errors_last_hour = 0

            for line in lines:
                if line.strip():
                    try:
                        # Parse timestamp from log line
                        timestamp_str = line.split(' - ')[0]
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                        if timestamp >= cutoff_24h:
                            recent_errors.append({
                                'timestamp': timestamp.isoformat(),
                                'line': line.strip()
                            })
                        if timestamp >= cutoff_1h:
                            errors_last_hour += 1
                    except:
                        # Skip lines that don't match expected format
                        continue

            # Categorize errors
            error_categories = {}
            for error in recent_errors:
                line = error['line']
                if 'ERROR' in line:
                    # Extract error type
                    if 'TypeError' in line:
                        error_categories['TypeError'] = error_categories.get('TypeError', 0) + 1
                    elif 'ValueError' in line:
                        error_categories['ValueError'] = error_categories.get('ValueError', 0) + 1
                    elif 'ConnectionError' in line:
                        error_categories['ConnectionError'] = error_categories.get('ConnectionError', 0) + 1
                    else:
                        error_categories['Other'] = error_categories.get('Other', 0) + 1

            return {
                'total_errors': len(recent_errors),
                'errors_last_hour': errors_last_hour,
                'error_categories': error_categories,
                'errors': recent_errors[-10:]  # Last 10 errors
            }

        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            return {'total_errors': 0, 'errors_last_hour': 0, 'error_categories': {}, 'errors': []}

    def _check_error_thresholds(self, recent_errors: Dict[str, Any]) -> bool:
        """Check if error thresholds are exceeded"""
        total_errors = recent_errors.get('total_errors', 0)
        errors_last_hour = recent_errors.get("errors_last_hour", 0)

        # Check hourly threshold
        if errors_last_hour > self.error_thresholds['max_errors_per_hour']:
            return True

        # Check daily threshold (24h window)
        if total_errors > self.error_thresholds["max_errors_per_day"]:
            return True

        # Check for critical error types
        error_categories = recent_errors.get('error_categories', {})
        for critical_type in self.error_thresholds['critical_error_types']:
            if error_categories.get(critical_type, 0) > 0:
                return True

        return False

    def _send_alert(self, recent_errors: Dict[str, Any]):
        """Send alert notification"""
        try:
            # Check cooldown period
            if self.last_alert_time:
                time_since_last = datetime.now() - self.last_alert_time
                if time_since_last.total_seconds() < self.error_thresholds['alert_cooldown_minutes'] * 60:
                    return  # Still in cooldown period

            # Create alert message
            alert_message = self._create_alert_message(recent_errors)

            # Log alert
            logger.warning(f"ERROR ALERT: {alert_message}")

            # Save alert to file
            self._save_alert_to_file(alert_message, recent_errors)

            # OpenClaw alert (optional)
            self._send_openclaw_alert(alert_message)

            # Email alert (optional, Gmail supported via SMTP)
            self._send_email_alert(alert_message)

            # Update last alert time
            self.last_alert_time = datetime.now()

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def _create_alert_message(self, recent_errors: Dict[str, Any]) -> str:
        """Create alert message"""
        total_errors = recent_errors.get('total_errors', 0)
        errors_last_hour = recent_errors.get("errors_last_hour", 0)
        error_categories = recent_errors.get('error_categories', {})

        message = f"""
[ALERT] ERROR ALERT - Portfolio Maximizer v45
============================================

Timestamp: {datetime.now().isoformat()}
Errors (1h): {errors_last_hour}
Total Errors (24h): {total_errors}

Error Breakdown:
"""

        for error_type, count in error_categories.items():
            message += f"  - {error_type}: {count}\n"

        if recent_errors.get('errors'):
            message += "\nRecent Errors:\n"
            for error in recent_errors['errors'][-3:]:  # Last 3 errors
                message += f"  - {error['timestamp']}: {error['line'][:100]}...\n"

        message += f"\nSystem Status: {'CRITICAL' if total_errors > 10 else 'WARNING'}"

        return message

    def _send_openclaw_alert(self, alert_message: str) -> None:
        """Optionally send the alert via OpenClaw CLI."""
        cfg = self.alert_config.get("openclaw")
        if not isinstance(cfg, dict):
            return
        if not bool(cfg.get("enabled", False)):
            return

        default_channel = (os.getenv("OPENCLAW_CHANNEL") or cfg.get("channel") or "").strip() or None
        env_targets = (os.getenv("OPENCLAW_TARGETS") or "").strip()
        env_to = (os.getenv("OPENCLAW_TO") or "").strip()

        try:
            from utils.openclaw_cli import resolve_openclaw_targets

            targets = resolve_openclaw_targets(
                env_targets=env_targets,
                env_to=env_to,
                cfg_to=cfg.get("to"),
                default_channel=default_channel,
            )
        except Exception:
            targets = []

        if not targets:
            logger.info(
                "OpenClaw alerts enabled but no targets configured (OPENCLAW_TARGETS / OPENCLAW_TO / alerts.openclaw.to)."
            )
            return

        command = (os.getenv("OPENCLAW_COMMAND") or cfg.get("command") or "openclaw").strip() or "openclaw"
        try:
            timeout_seconds = float(os.getenv("OPENCLAW_TIMEOUT_SECONDS") or cfg.get("timeout_seconds") or 20)
        except Exception:
            timeout_seconds = 20.0

        try:
            max_chars = int(cfg.get("max_message_chars") or 1500)
        except Exception:
            max_chars = 1500

        message = (alert_message or "").strip()
        if max_chars > 0 and len(message) > max_chars:
            message = message[: max(0, max_chars - 20)].rstrip() + "\n...(truncated)"

        message = self._redact_outbound_text(message)

        try:
            from utils.openclaw_cli import send_message_multi

            results = send_message_multi(
                targets=targets,
                message=message,
                command=command,
                cwd=project_root,
                timeout_seconds=timeout_seconds,
            )
            for result in results:
                if result.ok:
                    continue
                logger.warning(
                    "OpenClaw alert failed (exit=%s): %s",
                    result.returncode,
                    (result.stderr or result.stdout or "").strip()[:200],
                )
        except Exception as exc:
            logger.warning("OpenClaw alert failed: %s", exc)

    def _redact_outbound_text(self, text: str) -> str:
        """
        Best-effort redaction before sending text to external surfaces (OpenClaw, email).

        We redact any environment variable values whose names look like secrets.
        This prevents accidental leakage if an upstream error line included a key/token.
        """
        payload = text or ""

        secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
        for name, value in os.environ.items():
            if not value or len(value) < 8:
                continue
            upper = name.upper()
            if any(marker in upper for marker in secret_markers):
                payload = payload.replace(value, "[REDACTED]")

        # Common Authorization patterns
        payload = re.sub(r"(Bearer\\s+)[A-Za-z0-9\\-\\._~\\+/]+=*", r"\\1[REDACTED]", payload)
        return payload

    def _send_email_alert(self, alert_message: str) -> None:
        """Optionally send the alert via SMTP (Gmail supported via STARTTLS)."""
        cfg = self.alert_config.get("email")
        if not isinstance(cfg, dict):
            return
        if not bool(cfg.get("enabled", False)):
            return

        def _truthy(value: str) -> bool:
            return (value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

        def _env_int(name: str, fallback: int) -> int:
            raw = (os.getenv(name) or "").strip()
            if not raw:
                return fallback
            try:
                return int(raw)
            except Exception:
                return fallback

        def _env_float(name: str, fallback: float) -> float:
            raw = (os.getenv(name) or "").strip()
            if not raw:
                return fallback
            try:
                return float(raw)
            except Exception:
                return fallback

        env_to = (
            os.getenv("PMX_EMAIL_TO")
            or os.getenv("PMX_EMAIL_RECIPIENTS")
            or os.getenv("MAIN_EMAIL_GMAIL")
            or ""
        ).strip()
        recipients: List[str] = []
        if env_to:
            recipients = [p.strip() for p in env_to.replace(";", ",").split(",") if p.strip()]
        else:
            raw_recipients = cfg.get("recipients")
            if isinstance(raw_recipients, list):
                recipients = [str(v).strip() for v in raw_recipients if str(v).strip()]
            elif isinstance(raw_recipients, str):
                recipients = [p.strip() for p in raw_recipients.replace(";", ",").split(",") if p.strip()]

        if not recipients:
            logger.info("Email alerts enabled but no recipients configured (PMX_EMAIL_TO / alerts.email.recipients).")
            return

        smtp_server = (os.getenv("PMX_EMAIL_SMTP_SERVER") or cfg.get("smtp_server") or "").strip()
        if not smtp_server:
            logger.info(
                "Email alerts enabled but smtp_server missing (PMX_EMAIL_SMTP_SERVER / alerts.email.smtp_server)."
            )
            return

        smtp_port = _env_int("PMX_EMAIL_SMTP_PORT", int(cfg.get("smtp_port") or 587))
        timeout_seconds = _env_float("PMX_EMAIL_TIMEOUT_SECONDS", float(cfg.get("timeout_seconds") or 10.0))

        use_tls_env = os.getenv("PMX_EMAIL_USE_TLS")
        if use_tls_env is None:
            use_tls = bool(cfg.get("use_tls", True))
        else:
            use_tls = _truthy(use_tls_env)

        try:
            from etl.secret_loader import load_secret

            username = (load_secret("PMX_EMAIL_USERNAME") or cfg.get("username") or "").strip()
            password = (load_secret("PMX_EMAIL_PASSWORD") or cfg.get("password") or "").strip()
            from_address = (load_secret("PMX_EMAIL_FROM") or os.getenv("PMX_EMAIL_FROM") or cfg.get("from_address") or "").strip()
        except Exception:
            username = ((os.getenv("PMX_EMAIL_USERNAME") or cfg.get("username") or "")).strip()
            password = ((os.getenv("PMX_EMAIL_PASSWORD") or cfg.get("password") or "")).strip()
            from_address = ((os.getenv("PMX_EMAIL_FROM") or cfg.get("from_address") or "")).strip()

        if not username or not password:
            logger.info(
                "Email alerts enabled but missing credentials (PMX_EMAIL_USERNAME/PMX_EMAIL_PASSWORD or alerts.email.username/password)."
            )
            return

        if not from_address:
            from_address = username

        subject = "Portfolio Maximizer Error Alert"
        try:
            templates = self.config.get("templates") if isinstance(self.config, dict) else None
            if isinstance(templates, dict):
                subj = templates.get("error_alert", {}).get("subject")
                if subj:
                    subject = str(subj)
        except Exception:
            pass

        try:
            max_body_chars = _env_int("PMX_EMAIL_MAX_BODY_CHARS", int(cfg.get("max_body_chars") or 5000))
        except Exception:
            max_body_chars = 5000

        body = (alert_message or "").strip() + "\n\n(Generated by scripts/error_monitor.py)"
        if max_body_chars > 0 and len(body) > max_body_chars:
            body = body[: max(0, max_body_chars - 20)].rstrip() + "\n...(truncated)"

        body = self._redact_outbound_text(body)

        msg = MIMEMultipart()
        msg["From"] = from_address
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=timeout_seconds) as server:
                server.ehlo()
                if use_tls:
                    server.starttls()
                    server.ehlo()
                server.login(username, password)
                server.sendmail(from_address, recipients, msg.as_string())
        except Exception as exc:
            logger.warning("Email alert failed: %s", exc)

    def _save_alert_to_file(self, alert_message: str, recent_errors: Dict[str, Any]):
        """Save alert to file"""
        alert_file = f"logs/alerts/error_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs(os.path.dirname(alert_file), exist_ok=True)

        with open(alert_file, 'w') as f:
            f.write(alert_message)
            f.write(f"\n\nDetailed Error Data:\n")
            f.write(json.dumps(recent_errors, indent=2))

    def generate_error_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        try:
            error_log_path = "logs/errors/errors.log"
            if not os.path.exists(error_log_path):
                return {"error": "No error log found"}

            with open(error_log_path, 'r') as f:
                lines = f.readlines()

            # Analyze errors over specified period
            cutoff_time = datetime.now() - timedelta(days=days)
            errors_by_day = {}
            error_types = {}

            for line in lines:
                if line.strip():
                    try:
                        timestamp_str = line.split(' - ')[0]
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                        if timestamp >= cutoff_time:
                            day_key = timestamp.strftime('%Y-%m-%d')
                            errors_by_day[day_key] = errors_by_day.get(day_key, 0) + 1

                            # Categorize error types
                            if 'TypeError' in line:
                                error_types['TypeError'] = error_types.get('TypeError', 0) + 1
                            elif 'ValueError' in line:
                                error_types['ValueError'] = error_types.get('ValueError', 0) + 1
                            elif 'ConnectionError' in line:
                                error_types['ConnectionError'] = error_types.get('ConnectionError', 0) + 1
                            else:
                                error_types['Other'] = error_types.get('Other', 0) + 1
                    except:
                        continue

            return {
                "period_days": days,
                "total_errors": sum(errors_by_day.values()),
                "errors_by_day": errors_by_day,
                "error_types": error_types,
                "average_errors_per_day": sum(errors_by_day.values()) / max(len(errors_by_day), 1),
                "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else "None",
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": str(e)}


def main():
    """Main error monitoring function"""
    # Avoid UnicodeEncodeError on Windows consoles when printing emoji/status markers.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Load `.env` safely (best-effort) without printing or overwriting existing env vars.
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    monitor = ErrorMonitor()

    print("[INFO] Starting enhanced error monitoring...")

    # Monitor current errors
    result = monitor.monitor_errors()
    print(f"Monitoring Result: {result['status']}")

    if result.get('alert_sent'):
        print("[WARN] Alert sent due to error threshold exceeded")

    # Generate error report
    print("\n[INFO] Generating error report...")
    report = monitor.generate_error_report(7)

    if 'error' in report:
        print(f"[ERROR] Report generation failed: {report['error']}")
    else:
        print("[OK] Error report generated:")
        print(f"  - Total Errors (7 days): {report['total_errors']}")
        print(f"  - Average per day: {report['average_errors_per_day']:.1f}")
        print(f"  - Most common error: {report['most_common_error']}")

    print("\n[OK] Error monitoring complete")


if __name__ == "__main__":
    main()
