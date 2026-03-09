from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _active_audit_payload() -> dict:
    return {
        "artifacts": {
            "residual_experiment": {
                "experiment_id": "EXP-R5-001",
                "anchor_model_id": "mssa_rl",
                "residual_status": "active",
                "residual_active": True,
            }
        }
    }


def test_truth_detects_active_audits_but_summary_skip(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_json(
        tmp_path / "logs" / "forecast_audits" / "forecast_audit_20260308_000001.json",
        _active_audit_payload(),
    )
    _write_json(
        tmp_path / "visualizations" / "performance" / "residual_experiment_summary.json",
        {
            "status": "SKIP",
            "reason_code": "RESIDUAL_EXPERIMENT_NOT_FITTED",
            "n_windows_with_residual_metrics": 0,
            "n_windows_with_realized_residual_metrics": 0,
            "n_windows_structural_only_metrics": 0,
            "m2_review_ready": False,
        },
    )
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "forecasting_config.yml").write_text(
        "forecasting:\n  residual_experiment:\n    enabled: true\n",
        encoding="utf-8",
    )

    from scripts.residual_experiment_truth import build_truth_snapshot, main

    snapshot = build_truth_snapshot()
    assert snapshot["canonical_summary_path"].replace("\\", "/") == (
        "visualizations/performance/residual_experiment_summary.json"
    )
    assert "ACTIVE_AUDITS_BUT_SUMMARY_SKIP" in snapshot["contradictions"]
    assert "ACTIVE_AUDITS_BUT_ZERO_MEASURED_WINDOWS" in snapshot["contradictions"]
    assert snapshot["n_windows_with_realized_residual_metrics"] == 0
    assert snapshot["n_windows_structural_only_metrics"] == 0
    assert snapshot["m2_review_ready"] is False
    assert main(["--json"]) == 1


def test_truth_detects_active_audits_but_zero_measured_windows(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_json(
        tmp_path / "logs" / "forecast_audits" / "forecast_audit_20260308_000001.json",
        _active_audit_payload(),
    )
    _write_json(
        tmp_path / "visualizations" / "performance" / "residual_experiment_summary.json",
        {
            "status": "PASS",
            "reason_code": "RESIDUAL_EXPERIMENT_AVAILABLE",
            "n_windows_with_residual_metrics": 0,
        },
    )

    from scripts.residual_experiment_truth import build_truth_snapshot

    snapshot = build_truth_snapshot()
    assert "ACTIVE_AUDITS_BUT_SUMMARY_SKIP" not in snapshot["contradictions"]
    assert "ACTIVE_AUDITS_BUT_ZERO_MEASURED_WINDOWS" in snapshot["contradictions"]


def test_truth_passes_when_summary_and_audits_are_consistent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_json(
        tmp_path / "logs" / "forecast_audits" / "forecast_audit_20260308_000001.json",
        _active_audit_payload(),
    )
    _write_json(
        tmp_path / "visualizations" / "performance" / "residual_experiment_summary.json",
        {
            "status": "PASS",
            "reason_code": "RESIDUAL_EXPERIMENT_AVAILABLE",
            "n_windows_with_residual_metrics": 3,
        },
    )

    from scripts.residual_experiment_truth import build_truth_snapshot, main

    snapshot = build_truth_snapshot()
    assert snapshot["ok"] is True
    assert snapshot["contradictions"] == []
    assert main(["--json"]) == 0


def test_default_residual_summary_path_is_canonical():
    from scripts.run_quality_pipeline import DEFAULT_RESIDUAL_EXPERIMENT_OUT

    assert str(DEFAULT_RESIDUAL_EXPERIMENT_OUT).replace("\\", "/").endswith(
        "visualizations/performance/residual_experiment_summary.json"
    )
