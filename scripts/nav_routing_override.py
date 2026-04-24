#!/usr/bin/env python3
"""Validate and manage the NAV routing override approval artifact.

The override is intentionally narrow:
- schema_version 1 only
- scope must remain routing_only
- active approvals must have issued/expires timestamps and a 14-day cap
- inactive template files are allowed so the repo can ship a committed schema,
  but they must carry template=True so they cannot be mistaken for a live
  approval artifact.

The active JSON is written atomically and any previous active approval is
appended to the JSONL audit log before overwrite.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OVERRIDE_PATH = ROOT / "config" / "operator_approvals" / "nav_routing_override.json"
DEFAULT_AUDIT_LOG_PATH = ROOT / "config" / "operator_approvals" / "nav_routing_override.jsonl"
SCHEMA_VERSION = 1
MAX_EXPIRY_DAYS = 14
RUNTIME_SCOPE = "routing_only"


def _parse_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        try:
            import os

            os.fsync(handle.fileno())
        except Exception:
            pass


def template_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "inactive",
        "scope": RUNTIME_SCOPE,
        "template": True,
        "reason": None,
        "approved_by": None,
        "issued_utc": None,
        "expires_utc": None,
    }


def load_routing_override(path: Path) -> dict[str, Any]:
    return _load_json(path)


def validate_routing_override(payload: dict[str, Any], *, now: Optional[datetime] = None) -> dict[str, Any]:
    now = now or _utc_now()
    errors: list[str] = []
    warnings: list[str] = []
    schema_version_raw = payload.get("schema_version") if isinstance(payload, dict) else None
    try:
        schema_version = int(schema_version_raw) if schema_version_raw is not None else 0
    except Exception:
        schema_version = 0

    status = str(payload.get("status") or "inactive").strip().lower()
    scope = str(payload.get("scope") or "").strip().lower()
    template = bool(payload.get("template"))
    reason = payload.get("reason")
    approved_by = payload.get("approved_by")
    issued_utc = _parse_utc(payload.get("issued_utc"))
    expires_utc = _parse_utc(payload.get("expires_utc"))

    if schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version:{schema_version}")
    if status not in {"inactive", "active"}:
        errors.append(f"status:{status or 'missing'}")
    if scope != RUNTIME_SCOPE:
        errors.append(f"scope:{scope or 'missing'}")

    is_active = status == "active"
    if is_active:
        if template:
            errors.append("template_marker_present_for_active_override")
        if not str(reason or "").strip():
            errors.append("missing_reason")
        if not str(approved_by or "").strip():
            errors.append("missing_approved_by")
        if issued_utc is None:
            errors.append("missing_issued_utc")
        if expires_utc is None:
            errors.append("missing_expires_utc")
        if issued_utc is not None and expires_utc is not None:
            if expires_utc <= issued_utc:
                errors.append("expires_before_issue")
            if expires_utc - issued_utc > timedelta(days=MAX_EXPIRY_DAYS):
                errors.append("expiry_exceeds_14_days")
            if expires_utc <= now:
                errors.append("approval_expired")
    else:
        if not template:
            errors.append("missing_template_marker")
        if any(field is not None for field in (reason, approved_by, issued_utc, expires_utc)):
            warnings.append("inactive_template_has_optional_fields")

    return {
        "ok": not errors,
        "active": bool(is_active and not errors),
        "schema_version": schema_version,
        "status": status,
        "scope": scope,
        "errors": errors,
        "warnings": warnings,
    }


def write_routing_override(
    path: Path,
    payload: dict[str, Any],
    *,
    audit_log_path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    report = validate_routing_override(payload, now=now)
    if not report["ok"]:
        raise ValueError("; ".join(report["errors"]))

    current = load_routing_override(path)
    if audit_log_path is not None and current and str(current.get("status") or "").strip().lower() == "active":
        _append_jsonl(audit_log_path, current)

    _atomic_write_json(path, payload)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT_OVERRIDE_PATH), help="Active routing override JSON path.")
    parser.add_argument("--audit-log-path", default=str(DEFAULT_AUDIT_LOG_PATH), help="Append-only audit log path.")
    parser.add_argument("--json", action="store_true", help="Emit validation JSON.")
    args = parser.parse_args(argv)

    path = Path(args.path)
    payload = load_routing_override(path)
    report = validate_routing_override(payload if payload else template_payload())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        if report["active"]:
            print(f"[PASS] active routing override valid at {path}")
        elif report["ok"]:
            print(f"[PASS] routing override template valid at {path}")
        else:
            print(f"[FAIL] routing override invalid at {path}: {'; '.join(report['errors'])}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
