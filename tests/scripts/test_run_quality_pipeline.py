"""
Tests for scripts/run_quality_pipeline.py
"""
from __future__ import annotations

import json

from pathlib import Path
from unittest.mock import patch


def test_all_steps_pass(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline
    db = tmp_path / "db.db"
    db.write_bytes(b"")

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
    ), patch(
        "scripts.run_quality_pipeline.compute_context_quality",
        return_value={"warnings": [], "partial_data": False},
    ), patch(
        "scripts.run_quality_pipeline.apply_eligibility_gates",
        return_value={
            "warnings": [],
            "errors": [],
            "lab_only_tickers": [],
            "gate_written": True,
        },
    ), patch(
        "scripts.run_quality_pipeline.run_data_sufficiency",
        return_value={"status": "SUFFICIENT", "sufficient": True, "recommendations": []},
    ), patch(
        "scripts.run_quality_pipeline.generate_performance_artifacts",
        return_value={"warnings": [], "errors": [], "metrics_path": str(tmp_path / "m.json")},
    ) as mock_gen:
        result = run_quality_pipeline(
            db_path=db,
            audit_dir=tmp_path / "audits",
            eligibility_out=tmp_path / "elig.json",
            eligibility_gates_out=tmp_path / "elig_gates.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
        )
    assert mock_gen.call_count == 1
    kwargs = mock_gen.call_args.kwargs
    assert kwargs["strict_mode"] is True
    assert kwargs["sufficiency"]["status"] == "SUFFICIENT"
    assert result["status"] == "PASS"
    assert "pipeline_version" in result
    assert "started_at" in result
    assert "finished_at" in result
    assert "duration_seconds" in result
    assert len(result["steps"]) == 5
    gate_step = next(step for step in result["steps"] if step["name"] == "apply_ticker_eligibility_gates")
    assert gate_step["status"] == "PASS"
    assert gate_step["gate_written"] is True


def test_warn_when_partial_data(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline
    db = tmp_path / "db.db"
    db.write_bytes(b"")

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 0, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 0},
    ), patch(
        "scripts.run_quality_pipeline.compute_context_quality",
        return_value={"warnings": ["missing_optional_confidence_columns"], "partial_data": True},
    ), patch(
        "scripts.run_quality_pipeline.apply_eligibility_gates",
        return_value={
            "warnings": [],
            "errors": [],
            "lab_only_tickers": ["AAPL"],
            "gate_written": True,
        },
    ), patch(
        "scripts.run_quality_pipeline.run_data_sufficiency",
        return_value={"status": "INSUFFICIENT", "sufficient": False, "recommendations": ["TRADE_COUNT"]},
    ), patch(
        "scripts.run_quality_pipeline.generate_performance_artifacts",
        return_value={"warnings": ["sufficiency_not_green"], "errors": [], "metrics_path": str(tmp_path / "m.json")},
    ):
        result = run_quality_pipeline(
            db_path=db,
            audit_dir=tmp_path / "audits",
            eligibility_out=tmp_path / "elig.json",
            eligibility_gates_out=tmp_path / "elig_gates.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
        )
    assert result["status"] == "WARN"
    assert "zero_healthy_tickers" in result["warnings"]


def test_error_when_data_error(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 0, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 0},
    ), patch(
        "scripts.run_quality_pipeline.compute_context_quality",
        return_value={"warnings": [], "partial_data": False},
    ), patch(
        "scripts.run_quality_pipeline.apply_eligibility_gates",
        return_value={
            "warnings": [],
            "errors": [],
            "lab_only_tickers": [],
            "gate_written": True,
        },
    ), patch(
        "scripts.run_quality_pipeline.generate_performance_artifacts",
        return_value={"warnings": [], "errors": [], "metrics_path": str(tmp_path / "m.json")},
    ):
        result = run_quality_pipeline(
            db_path=tmp_path / "missing.db",
            audit_dir=tmp_path / "audits",
            eligibility_out=tmp_path / "elig.json",
            eligibility_gates_out=tmp_path / "elig_gates.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
        )
    assert result["status"] == "ERROR"


def test_error_when_chart_stage_reports_missing_artifacts(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    db.write_bytes(b"")

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "errors": [], "n_tickers": 1},
    ), patch(
        "scripts.run_quality_pipeline.compute_context_quality",
        return_value={"warnings": [], "partial_data": False},
    ), patch(
        "scripts.run_quality_pipeline.apply_eligibility_gates",
        return_value={
            "warnings": [],
            "errors": [],
            "lab_only_tickers": [],
            "gate_written": True,
        },
    ), patch(
        "scripts.run_quality_pipeline.run_data_sufficiency",
        return_value={"status": "SUFFICIENT", "sufficient": True, "recommendations": []},
    ), patch(
        "scripts.run_quality_pipeline.generate_performance_artifacts",
        return_value={
            "warnings": ["chart_missing:per_ticker_wr_pf"],
            "errors": ["chart_missing:per_ticker_wr_pf"],
            "metrics_path": str(tmp_path / "m.json"),
        },
    ):
        result = run_quality_pipeline(
            db_path=db,
            audit_dir=tmp_path / "audits",
            eligibility_out=tmp_path / "elig.json",
            eligibility_gates_out=tmp_path / "elig_gates.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
        )

    assert result["status"] == "ERROR"
    assert "chart_missing:per_ticker_wr_pf" in result["errors"]
    chart_step = next(step for step in result["steps"] if step["name"] == "generate_performance_charts")
    assert chart_step["status"] == "ERROR"


def test_cli_json_schema(tmp_path, capsys):
    from scripts.run_quality_pipeline import main

    with patch(
        "scripts.run_quality_pipeline.run_quality_pipeline",
        return_value={
            "status": "PASS",
            "steps": [],
            "artifacts": {},
            "warnings": [],
            "errors": [],
            "pipeline_version": "x",
            "started_at": "a",
            "finished_at": "b",
            "duration_seconds": 0.1,
        },
    ):
        rc = main(["--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["pipeline_version"] == "x"


def test_warn_when_eligibility_gate_reports_warning(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    db.write_bytes(b"")

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
    ), patch(
        "scripts.run_quality_pipeline.apply_eligibility_gates",
        return_value={
            "warnings": ["eligibility_missing"],
            "errors": [],
            "lab_only_tickers": [],
            "gate_written": True,
        },
    ), patch(
        "scripts.run_quality_pipeline.compute_context_quality",
        return_value={"warnings": [], "partial_data": False},
    ), patch(
        "scripts.run_quality_pipeline.run_data_sufficiency",
        return_value={"status": "SUFFICIENT", "sufficient": True, "recommendations": []},
    ), patch(
        "scripts.run_quality_pipeline.generate_performance_artifacts",
        return_value={"warnings": [], "errors": [], "metrics_path": str(tmp_path / "m.json")},
    ):
        result = run_quality_pipeline(
            db_path=db,
            audit_dir=tmp_path / "audits",
            eligibility_out=tmp_path / "elig.json",
            eligibility_gates_out=tmp_path / "elig_gates.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
        )

    gate_step = next(step for step in result["steps"] if step["name"] == "apply_ticker_eligibility_gates")
    assert gate_step["status"] == "WARN"
    assert "eligibility_missing" in result["warnings"]
