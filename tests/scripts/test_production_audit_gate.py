from __future__ import annotations

from pathlib import Path

import pytest


def test_extract_lift_output_metrics_inconclusive() -> None:
    import scripts.production_audit_gate as mod

    output = "\n".join(
        [
            "Effective audits with RMSE: 7",
            "Violations (ensemble worse than baseline beyond tolerance): 2",
            "Violation rate: 28.57% (max allowed 35.00%)",
            "RMSE gate inconclusive: effective_audits=7 < required_audits=20.",
        ]
    )
    metrics = mod._extract_lift_output_metrics(output)
    assert metrics["effective_audits"] == 7
    assert metrics["violation_count"] == 2
    assert metrics["violation_rate"] == pytest.approx(0.2857, abs=1e-6)
    assert metrics["max_violation_rate"] == pytest.approx(0.35, abs=1e-6)
    assert metrics["decision"] is None
    assert metrics["decision_reason"] is None


def test_extract_lift_output_metrics_decision_and_lift_fraction() -> None:
    import scripts.production_audit_gate as mod

    output = "\n".join(
        [
            "Ensemble lift fraction: 28.57% (required >= 25.00%)",
            "Decision: KEEP (lift demonstrated during holding period)",
        ]
    )
    metrics = mod._extract_lift_output_metrics(output)
    assert metrics["lift_fraction"] == pytest.approx(0.2857, abs=1e-6)
    assert metrics["min_lift_fraction"] == pytest.approx(0.25, abs=1e-6)
    assert metrics["decision"] == "KEEP"
    assert metrics["decision_reason"] == "lift demonstrated during holding period"


def test_summary_matches_invocation_max_files_match() -> None:
    import scripts.production_audit_gate as mod

    audit_dir = Path.cwd() / "logs" / "forecast_audits"
    summary = {"audit_dir": str(audit_dir), "max_files": 500}
    assert mod._summary_matches_invocation(summary, audit_dir=audit_dir, max_files=500)


def test_summary_matches_invocation_max_files_mismatch() -> None:
    import scripts.production_audit_gate as mod

    audit_dir = Path.cwd() / "logs" / "forecast_audits"
    summary = {"audit_dir": str(audit_dir), "max_files": 500}
    assert not mod._summary_matches_invocation(summary, audit_dir=audit_dir, max_files=50)
