from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import scripts.institutional_unattended_gate as mod


class _Proc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_gate_returns_structured_findings() -> None:
    findings = mod.run_gate()
    assert isinstance(findings, list)
    assert findings, "Gate must produce at least one finding."
    assert all(isinstance(f, mod.Finding) for f in findings)


def test_main_json_exit_code_fails_on_blocking_findings(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mod,
        "run_gate",
        lambda: [mod.Finding("P0", "example", "FAIL", "blocking")],
    )
    rc = mod.main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 1
    assert payload[0]["status"] == "FAIL"


def test_main_json_exit_code_zero_when_no_failures(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mod,
        "run_gate",
        lambda: [mod.Finding("P0", "example", "PASS", "ok")],
    )
    rc = mod.main(["--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert payload[0]["status"] == "PASS"


def test_phase_p2_fails_on_invalid_json(monkeypatch) -> None:
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: _Proc(0, stdout="{not-json}"))
    findings = mod._phase_p2_platt_data()
    assert findings
    assert findings[0].status == "FAIL"
    assert "Unable to parse" in findings[0].detail


def test_phase_p2_fails_on_empty_findings(monkeypatch) -> None:
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: _Proc(0, stdout="[]"))
    findings = mod._phase_p2_platt_data()
    assert findings
    assert findings[0].status == "FAIL"
    assert "empty findings list" in findings[0].detail


def test_phase_p4_missing_artifact_warns_in_ci(monkeypatch, tmp_path) -> None:
    # Missing artifact = CI/fresh environment: WARN (not FAIL) so CI is not blocked.
    # Stale artifact (exists but old) = production concern: FAIL.
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    findings = mod._phase_p4_prior_gate_verification()
    assert findings
    assert findings[0].status == "WARN"
    assert "not found" in findings[0].detail


def test_phase_p4_stale_artifact_fails_closed(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    artifact = tmp_path / "logs" / "gate_status_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    artifact.write_text(
        json.dumps({"overall_passed": True, "timestamp_utc": stale_ts}),
        encoding="utf-8",
    )
    findings = mod._phase_p4_prior_gate_verification()
    assert findings
    assert findings[0].status == "FAIL"
    assert "stale" in findings[0].detail.lower()


def test_phase_p4_reports_current_run_warmup_covered_blocker_semantically(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    artifact = tmp_path / "logs" / "gate_status_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "overall_passed": False,
                "status_stage": "pre_institutional",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "phase3_posture": "WARMUP_COVERED_PASS",
                "phase3_reason": "READY,GATE_SEMANTICS_INCONCLUSIVE_ALLOWED",
                "readiness_components": {
                    "gates_pass": True,
                    "linkage_pass": True,
                    "evidence_hygiene_pass": True,
                    "integrity_pass": True,
                },
            }
        ),
        encoding="utf-8",
    )

    findings = mod._phase_p4_prior_gate_verification()
    assert findings
    assert findings[0].status == "FAIL"
    assert "Current run pre-institutional snapshot" in findings[0].detail
    assert "WARMUP_COVERED_PASS" in findings[0].detail
    assert "Subgates are green" in findings[0].detail


def test_phase_p4_reports_latest_artifact_warmup_covered_blocker_semantically(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    artifact = tmp_path / "logs" / "gate_status_latest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "overall_passed": False,
                "status_stage": "final",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "phase3_posture": "WARMUP_COVERED_PASS",
                "phase3_reason": "READY,GATE_SEMANTICS_INCONCLUSIVE_ALLOWED",
                "readiness_components": {
                    "gates_pass": True,
                    "linkage_pass": True,
                    "evidence_hygiene_pass": True,
                    "integrity_pass": True,
                },
            }
        ),
        encoding="utf-8",
    )

    findings = mod._phase_p4_prior_gate_verification()
    assert findings
    assert findings[0].status == "FAIL"
    assert "Latest gate artifact" in findings[0].detail
    assert "WARMUP_COVERED_PASS" in findings[0].detail


def test_phase_p5_emission_error_fails_before_schema_gate(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    snapshot = tmp_path / "logs" / "canonical_snapshot_latest.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "emission_error": "canonical_source_registry.yml not found at /tmp/config/canonical_source_registry.yml",
                "gate": {
                    "freshness_status": {"status": "fresh"},
                    "warmup_state": {"posture": "expired", "deadline_utc": "2026-04-24T20:00:00Z", "matched_needed": 0},
                    "trajectory_alarm": {"active": False},
                },
                "summary": {
                    "evidence_health": "clean",
                    "roi_ann_pct": 9.86,
                    "objective_valid": True,
                },
                "source_contract": {
                    "status": "clean",
                    "canonical_sources": [],
                    "allowlisted_readers": [],
                    "violations_found": [],
                    "scan_timestamp_utc": "2026-04-18T12:00:00Z",
                },
                "alpha_objective": {"objective_valid": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", snapshot)

    findings = mod._phase_p5_canonical_snapshot_contract()
    assert findings
    assert findings[0].status == "FAIL"
    assert "emission_error=" in findings[0].detail
    assert "schema_version=0" not in findings[0].detail


def test_phase_p5_coverage_ratio_alarm_is_advisory_only(monkeypatch, tmp_path) -> None:
    """coverage_ratio_alarm is advisory-only and must NOT cause P5 to FAIL.

    Before this change, coverage_ratio_alarm duplicated the matched<10 gate blocker.
    Now it is purely informational — the unattended gate already fails on matched<10.
    """
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    snapshot = tmp_path / "logs" / "canonical_snapshot_latest.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "gate": {
                    "freshness_status": {"status": "fresh"},
                    "warmup_state": {"posture": "genuine_pass", "deadline_utc": "2026-04-24T20:00:00Z", "matched_needed": 0},
                    "trajectory_alarm": {"active": False},
                    "coverage_ratio_alarm": {
                        "active": True,
                        "severity": "critical",
                        "ratio": 0.074,
                        "warn_threshold": 1.0,
                        "critical_threshold": 0.25,
                        "expected_closes_remaining": 0.662,
                        "matched_needed": 9,
                        "shortfall": 8.338,
                    },
                    "post_deadline_time_to_10_estimate": {"status": "inactive", "estimated_days": None},
                },
                "summary": {
                    "evidence_health": "clean",
                    "roi_ann_pct": 9.86,
                    "objective_valid": True,
                    "unattended_gate": "PASS",
                },
                "source_contract": {
                    "status": "clean",
                    "canonical_sources": [],
                    "allowlisted_readers": [],
                    "violations_found": [],
                    "scan_timestamp_utc": "2026-04-18T12:00:00Z",
                },
                "alpha_objective": {"objective_valid": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "DEFAULT_CANONICAL_SNAPSHOT_PATH", snapshot)

    findings = mod._phase_p5_canonical_snapshot_contract()
    assert findings
    # coverage_ratio_alarm active=True must NOT cause FAIL — it is advisory-only
    assert findings[0].status == "PASS"
    assert "coverage_ratio_alarm_active" not in findings[0].detail
