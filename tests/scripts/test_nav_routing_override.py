from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import nav_routing_override as mod


def test_repo_template_override_file_validates_as_inactive() -> None:
    payload = json.loads(
        (Path(__file__).resolve().parents[2] / "config" / "operator_approvals" / "nav_routing_override.json").read_text(
            encoding="utf-8"
        )
    )
    report = mod.validate_routing_override(payload)

    assert report["ok"] is True
    assert report["active"] is False
    assert report["schema_version"] == 1
    assert report["scope"] == "routing_only"
    assert payload["template"] is True


def test_active_override_round_trips_and_audit_log_appends_prior_state(tmp_path: Path) -> None:
    active_path = tmp_path / "nav_routing_override.json"
    audit_path = tmp_path / "nav_routing_override.jsonl"
    current = {
        "schema_version": 1,
        "status": "active",
        "scope": "routing_only",
        "reason": "pre-flat approval",
        "approved_by": "Ops Lead",
        "issued_utc": "2026-04-20T12:00:00Z",
        "expires_utc": "2026-04-25T12:00:00Z",
    }
    active_path.write_text(json.dumps(current), encoding="utf-8")

    next_payload = {
        "schema_version": 1,
        "status": "active",
        "scope": "routing_only",
        "reason": "renewed approval",
        "approved_by": "Ops Lead",
        "issued_utc": "2026-04-21T12:00:00Z",
        "expires_utc": "2026-04-26T12:00:00Z",
    }

    report = mod.write_routing_override(active_path, next_payload, audit_log_path=audit_path, now=datetime(2026, 4, 21, 13, 0, tzinfo=timezone.utc))

    assert report["ok"] is True
    assert json.loads(active_path.read_text(encoding="utf-8"))["reason"] == "renewed approval"
    audit_rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(audit_rows) == 1
    assert audit_rows[0]["reason"] == "pre-flat approval"
    assert audit_rows[0]["status"] == "active"


def test_active_override_with_template_marker_is_rejected() -> None:
    payload = {
        "schema_version": 1,
        "status": "active",
        "template": True,
        "scope": "routing_only",
        "reason": "should not pass",
        "approved_by": "Ops Lead",
        "issued_utc": "2026-04-21T12:00:00Z",
        "expires_utc": "2026-04-26T12:00:00Z",
    }

    report = mod.validate_routing_override(payload, now=datetime(2026, 4, 21, 13, 0, tzinfo=timezone.utc))

    assert report["ok"] is False
    assert "template_marker_present_for_active_override" in report["errors"]


def test_write_routing_override_appends_before_overwrite(tmp_path: Path, monkeypatch) -> None:
    active_path = tmp_path / "nav_routing_override.json"
    audit_path = tmp_path / "nav_routing_override.jsonl"
    active_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "active",
                "template": False,
                "scope": "routing_only",
                "reason": "current approval",
                "approved_by": "Ops Lead",
                "issued_utc": "2026-04-20T12:00:00Z",
                "expires_utc": "2026-04-25T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    call_order: list[str] = []

    def _record_append(path: Path, payload: dict) -> None:
        call_order.append("append")
        assert path == audit_path
        assert payload["reason"] == "current approval"

    def _record_write(path: Path, payload: dict) -> None:
        call_order.append("write")
        assert path == active_path
        assert payload["reason"] == "renewed approval"

    monkeypatch.setattr(mod, "_append_jsonl", _record_append)
    monkeypatch.setattr(mod, "_atomic_write_json", _record_write)

    report = mod.write_routing_override(
        active_path,
        {
            "schema_version": 1,
            "status": "active",
            "template": False,
            "scope": "routing_only",
            "reason": "renewed approval",
            "approved_by": "Ops Lead",
            "issued_utc": "2026-04-21T12:00:00Z",
            "expires_utc": "2026-04-26T12:00:00Z",
        },
        audit_log_path=audit_path,
        now=datetime(2026, 4, 21, 13, 0, tzinfo=timezone.utc),
    )

    assert report["ok"] is True
    assert call_order == ["append", "write"]


@pytest.mark.parametrize(
    "payload, expected_error",
    [
        (
            {
                "schema_version": 1,
                "status": "active",
                "scope": "routing_only",
                "reason": "too long",
                "approved_by": "Ops Lead",
                "issued_utc": "2026-04-01T12:00:00Z",
                "expires_utc": "2026-04-20T12:00:01Z",
            },
            "expiry_exceeds_14_days",
        ),
        (
            {
                "schema_version": 1,
                "status": "active",
                "scope": "routing_only",
                "reason": "expired",
                "approved_by": "Ops Lead",
                "issued_utc": "2026-04-01T12:00:00Z",
                "expires_utc": "2026-04-10T12:00:00Z",
            },
            "approval_expired",
        ),
    ],
)
def test_active_override_rejects_invalid_expiry(payload: dict, expected_error: str) -> None:
    report = mod.validate_routing_override(payload, now=datetime(2026, 4, 21, 13, 0, tzinfo=timezone.utc))

    assert report["ok"] is False
    assert expected_error in report["errors"]
