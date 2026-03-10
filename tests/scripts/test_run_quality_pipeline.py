"""
Tests for scripts/run_quality_pipeline.py
"""
from __future__ import annotations

import json
import sqlite3

from pathlib import Path
from unittest.mock import patch


def _seed_ohlcv(db_path: Path, ticker: str, rows: list[tuple[str, float]]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                close REAL
            )
            """
        )
        conn.executemany(
            "INSERT INTO ohlcv_data (ticker, date, close) VALUES (?, ?, ?)",
            [(ticker, d, c) for d, c in rows],
        )
        conn.commit()
    finally:
        conn.close()


def _create_db_file(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"")


def test_all_steps_pass(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline
    db = tmp_path / "db.db"
    _create_db_file(db)

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
    ), patch(
        "scripts.run_quality_pipeline.compute_context_quality",
        return_value={"warnings": [], "partial_data": False},
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


def test_warn_when_partial_data(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline
    db = tmp_path / "db.db"
    _create_db_file(db)

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 0, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 0},
    ), patch(
        "scripts.run_quality_pipeline.compute_context_quality",
        return_value={"warnings": ["missing_optional_confidence_columns"], "partial_data": True},
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
        "scripts.run_quality_pipeline.generate_performance_artifacts",
        return_value={"warnings": [], "errors": [], "metrics_path": str(tmp_path / "m.json")},
    ):
        result = run_quality_pipeline(
            db_path=tmp_path / "missing.db",
            audit_dir=tmp_path / "audits",
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
        )
    assert result["status"] == "ERROR"


def test_error_when_chart_stage_reports_missing_artifacts(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "errors": [], "n_tickers": 1},
    ), patch(
        "scripts.run_quality_pipeline.compute_context_quality",
        return_value={"warnings": [], "partial_data": False},
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


def test_residual_experiment_flag_off_keeps_default_step_count(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=False,
        )

    assert len(result["steps"]) == 5
    assert result["residual_experiment_enabled"] is False
    assert "residual_experiment" not in result["artifacts"]


def test_residual_experiment_flag_on_writes_summary(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "forecast_audit_20260101_000000.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "residual_experiment": {
                        "experiment_id": "EXP-R5-001",
                        "anchor_model_id": "mssa_rl",
                        "residual_status": "active",
                        "residual_active": True,
                        "y_hat_anchor": [1.0, 2.0, 3.0],
                        "y_hat_residual_ensemble": [1.2, 2.1, 3.1],
                        "rmse_anchor": 2.0,
                        "rmse_residual_ensemble": 1.8,
                        "da_anchor": 0.51,
                        "da_residual_ensemble": 0.57,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    residual_out = tmp_path / "residual_experiment_summary.json"

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )

    assert result["residual_experiment_enabled"] is True
    assert result["artifacts"]["residual_experiment"] == str(residual_out)
    residual_step = next(step for step in result["steps"] if step["name"] == "residual_experiment_audit")
    assert residual_step["status"] == "PASS"
    summary = json.loads(residual_out.read_text(encoding="utf-8"))
    assert summary["n_windows_with_residual_metrics"] == 1
    assert summary["n_windows_with_realized_residual_metrics"] == 1
    assert summary["n_windows_structural_only_metrics"] == 0
    assert summary["m2_review_ready"] is False
    assert summary["rmse_ratio_mean"] == 0.9
    assert summary["corr_anchor_residual_mean"] is None
    assert summary["contract_version"] == "exp-r5-001.v1"
    assert summary["contract_probe"]["ok"] is True
    assert summary["n_non_active_signal_windows"] == 0
    assert summary["n_active_without_metrics_windows"] == 0


def test_residual_experiment_flag_on_without_metrics_warns_skip(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "forecast_audit_20260101_000000.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "evaluation_metrics": {
                        "mssa_rl": {"rmse": 2.0, "directional_accuracy": 0.55},
                        "ensemble": {"rmse": 2.1, "directional_accuracy": 0.5},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    residual_out = tmp_path / "residual_experiment_summary.json"

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )

    residual_step = next(step for step in result["steps"] if step["name"] == "residual_experiment_audit")
    assert residual_step["status"] == "WARN"
    assert "residual_experiment_not_available" in residual_step["warnings"]
    summary = json.loads(residual_out.read_text(encoding="utf-8"))
    assert summary["status"] == "SKIP"
    assert summary["reason_code"] == "RESIDUAL_EXPERIMENT_NOT_AVAILABLE"
    assert summary["n_windows_with_residual_metrics"] == 0
    assert summary["n_windows_with_realized_residual_metrics"] == 0
    assert summary["m2_review_ready"] is False


def test_residual_experiment_fallback_metrics_without_active_markers_is_not_counted(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "forecast_audit_20260101_000000.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "evaluation_metrics": {
                        "mssa_rl": {"rmse": 2.0, "directional_accuracy": 0.55},
                        "residual_ensemble": {"rmse": 1.8, "directional_accuracy": 0.6},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    residual_out = tmp_path / "residual_experiment_summary.json"

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )

    residual_step = next(step for step in result["steps"] if step["name"] == "residual_experiment_audit")
    assert residual_step["status"] == "WARN"
    assert "residual_experiment_non_active_signal_windows:1" in residual_step["warnings"]
    summary = json.loads(residual_out.read_text(encoding="utf-8"))
    assert summary["status"] == "SKIP"
    assert summary["reason_code"] == "RESIDUAL_EXPERIMENT_NOT_AVAILABLE"
    assert summary["n_non_active_signal_windows"] == 1
    assert summary["n_windows_with_residual_metrics"] == 0
    assert summary["rmse_ratio_mean"] is None


def test_residual_experiment_flag_on_not_fitted_is_skip_not_error(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "forecast_audit_20260101_000000.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "residual_experiment": {
                        "experiment_id": "EXP-R5-001",
                        "anchor_model_id": "mssa_rl",
                        "phase": 1,
                        "residual_status": "inactive",
                        "residual_active": False,
                        "reason": "residual_model_not_fitted (Phase 2 pending)",
                        "y_hat_anchor": None,
                        "y_hat_residual_ensemble": None,
                        "rmse_anchor": None,
                        "rmse_residual_ensemble": None,
                        "rmse_ratio": None,
                        "da_anchor": None,
                        "da_residual_ensemble": None,
                        "corr_anchor_residual": None,
                        "residual_mean": None,
                        "residual_std": None,
                        "n_corrected": 0,
                        "promotion_contract": {
                            "rmse_ratio_threshold": 0.98,
                            "corr_threshold": 0.9,
                            "min_effective_audits": 20,
                            "note": "advisory only - not enforced at runtime",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    residual_out = tmp_path / "residual_experiment_summary.json"

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )

    residual_step = next(step for step in result["steps"] if step["name"] == "residual_experiment_audit")
    assert residual_step["status"] == "WARN"
    assert "residual_experiment_not_fitted" in residual_step["warnings"]
    summary = json.loads(residual_out.read_text(encoding="utf-8"))
    assert summary["status"] == "SKIP"
    assert summary["reason_code"] == "RESIDUAL_EXPERIMENT_NOT_FITTED"
    assert summary["n_not_fitted_windows"] == 1
    assert summary["n_windows_with_residual_metrics"] == 0
    assert summary["rmse_ratio_mean"] is None
    assert summary["contract_probe"]["ok"] is True
    assert summary["contract_probe"]["expected_not_fitted"] is True


def test_residual_experiment_flag_on_uses_agent_a_fixture(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "forecast_audit_agent_a_residual_real.json"
    )
    (audit_dir / "forecast_audit_20260307_000001.json").write_text(
        fixture.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    residual_out = tmp_path / "residual_experiment_summary.json"

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )

    residual_step = next(step for step in result["steps"] if step["name"] == "residual_experiment_audit")
    assert residual_step["status"] == "PASS"
    summary = json.loads(residual_out.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["reason_code"] == "RESIDUAL_EXPERIMENT_AVAILABLE"
    assert summary["n_windows_with_residual_metrics"] == 1
    assert summary["n_windows_with_realized_residual_metrics"] == 1
    assert summary["corr_anchor_residual_mean"] is None
    assert summary["rmse_ratio_mean"] is not None


def test_residual_experiment_malformed_payload_is_error(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "forecast_audit_20260101_000000.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "residual_experiment": {
                        "y_hat_anchor": "invalid-series",
                        "y_hat_residual_ensemble": [1.0, 2.0, 3.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    residual_out = tmp_path / "residual_experiment_summary.json"

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )

    assert result["status"] == "ERROR"
    residual_step = next(step for step in result["steps"] if step["name"] == "residual_experiment_audit")
    assert residual_step["status"] == "ERROR"
    assert "residual_experiment_payload_malformed" in result["errors"]
    summary = json.loads(residual_out.read_text(encoding="utf-8"))
    assert summary["status"] == "ERROR"
    assert summary["reason_code"] == "RESIDUAL_EXPERIMENT_PAYLOAD_MALFORMED"


# Structural-active windows can be present before realized residual scalars are populated.
def test_residual_experiment_structural_only_window_is_reported_explicitly(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "forecast_audit_20260101_000000.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "residual_experiment": {
                        "experiment_id": "EXP-R5-001",
                        "anchor_model_id": "mssa_rl",
                        "residual_status": "active",
                        "residual_active": True,
                        "y_hat_anchor": [1.0, 2.0, 3.0],
                        "y_hat_residual_ensemble": [1.2, 2.2, 3.1],
                        "rmse_anchor": 2.0,
                        "rmse_residual_ensemble": None,
                        "rmse_ratio": None,
                        "da_anchor": 0.5,
                        "da_residual_ensemble": None,
                        "corr_anchor_residual": 0.8,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    residual_out = tmp_path / "residual_experiment_summary.json"

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )

    residual_step = next(step for step in result["steps"] if step["name"] == "residual_experiment_audit")
    assert residual_step["status"] == "WARN"
    assert "residual_experiment_realized_metrics_unavailable" in residual_step["warnings"]
    assert "residual_experiment_missing_realized_metrics_windows:1" in residual_step["warnings"]
    summary = json.loads(residual_out.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["n_windows_with_residual_metrics"] == 1
    assert summary["n_windows_with_realized_residual_metrics"] == 0
    assert summary["n_windows_structural_only_metrics"] == 1
    assert summary["n_active_windows_missing_realized_metrics"] == 1
    assert summary["m2_review_ready"] is False
    assert summary["rmse_residual_ensemble_mean"] is None
    assert summary["rmse_ratio_mean"] is None
    assert summary["da_residual_ensemble_mean"] is None


def test_residual_experiment_observability_fields_are_preserved(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "forecast_audit_20260101_000000.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "residual_experiment": {
                        "experiment_id": "EXP-R5-001",
                        "anchor_model_id": "mssa_rl",
                        "residual_status": "active",
                        "residual_active": True,
                        "y_hat_anchor": [1.0, 2.0, 3.0],
                        "y_hat_residual_ensemble": [1.1, 2.1, 3.1],
                        "rmse_anchor": 2.0,
                        "rmse_residual_ensemble": 1.9,
                        "rmse_ratio": 0.95,
                        "da_anchor": 0.5,
                        "da_residual_ensemble": 0.55,
                        "corr_anchor_residual": 0.2,
                        "phi_hat": 0.42,
                        "intercept_hat": 0.01,
                        "n_train_residuals": 120,
                        "oos_n_used": 30,
                        "skip_reason": None,
                        "residual_signal_valid": True,
                        "correction_applied": True,
                        "reason_code": "OK",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    residual_out = tmp_path / "residual_experiment_summary.json"

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
        run_quality_pipeline(
            db_path=db,
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )

    summary = json.loads(residual_out.read_text(encoding="utf-8"))
    assert summary["windows"][0]["phi_hat"] == 0.42
    assert summary["windows"][0]["intercept_hat"] == 0.01
    assert summary["windows"][0]["n_train_residuals"] == 120
    assert summary["windows"][0]["oos_n_used"] == 30
    assert summary["windows"][0]["skip_reason"] is None
    assert summary["windows"][0]["residual_signal_valid"] is True
    assert summary["windows"][0]["correction_applied"] is True
    assert summary["windows"][0]["reason_code"] == "OK"


def test_residual_experiment_backfills_realized_metrics_from_ohlcv(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _seed_ohlcv(
        db,
        "AAPL",
        [
            ("2023-11-21", 101.0),
            ("2023-11-22", 102.0),
            ("2023-11-23", 103.0),
        ],
    )
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "forecast_audit_20260101_000000.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "ticker": "AAPL",
                    "end": "2023-11-20 00:00:00",
                    "forecast_horizon": 3,
                },
                "artifacts": {
                    "residual_experiment": {
                        "experiment_id": "EXP-R5-001",
                        "anchor_model_id": "mssa_rl",
                        "residual_status": "active",
                        "residual_active": True,
                        "y_hat_anchor": [100.5, 101.5, 102.5],
                        "y_hat_residual_ensemble": [101.0, 102.0, 103.0],
                        "rmse_anchor": None,
                        "rmse_residual_ensemble": None,
                        "rmse_ratio": None,
                        "da_anchor": None,
                        "da_residual_ensemble": None,
                        "corr_anchor_residual": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    residual_out = tmp_path / "residual_experiment_summary.json"

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )

    residual_step = next(step for step in result["steps"] if step["name"] == "residual_experiment_audit")
    assert residual_step["status"] == "PASS"
    summary = json.loads(residual_out.read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["n_windows_with_realized_residual_metrics"] == 1
    assert summary["n_windows_structural_only_metrics"] == 0
    assert summary["n_windows_with_db_backfilled_realized_metrics"] == 1
    assert summary["rmse_residual_ensemble_mean"] is not None
    assert summary["rmse_ratio_mean"] is not None
    assert "residual_experiment_realized_metrics_unavailable" not in summary["warnings"]


def test_residual_experiment_backfill_without_enough_ohlcv_rows_stays_structural(tmp_path):
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _seed_ohlcv(
        db,
        "AAPL",
        [
            ("2023-11-21", 101.0),
        ],
    )
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "forecast_audit_20260101_000000.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "ticker": "AAPL",
                    "end": "2023-11-20 00:00:00",
                    "forecast_horizon": 3,
                },
                "artifacts": {
                    "residual_experiment": {
                        "experiment_id": "EXP-R5-001",
                        "anchor_model_id": "mssa_rl",
                        "residual_status": "active",
                        "residual_active": True,
                        "y_hat_anchor": [100.5, 101.5, 102.5],
                        "y_hat_residual_ensemble": [101.0, 102.0, 103.0],
                        "rmse_anchor": None,
                        "rmse_residual_ensemble": None,
                        "rmse_ratio": None,
                        "da_anchor": None,
                        "da_residual_ensemble": None,
                        "corr_anchor_residual": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    residual_out = tmp_path / "residual_experiment_summary.json"

    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )

    residual_step = next(step for step in result["steps"] if step["name"] == "residual_experiment_audit")
    assert residual_step["status"] == "WARN"
    assert "residual_experiment_realized_metrics_unavailable" in residual_step["warnings"]
    summary = json.loads(residual_out.read_text(encoding="utf-8"))
    assert summary["n_windows_with_realized_residual_metrics"] == 0
    assert summary["n_windows_structural_only_metrics"] == 1
    assert summary["n_windows_with_db_backfilled_realized_metrics"] == 0
    assert summary["rmse_ratio_mean"] is None


# ---------------------------------------------------------------------------
# Early abort guard tests (EXP-R5-001, Agent C item 5, 2026-03-08)
# ---------------------------------------------------------------------------

def _make_phase3_audit(rmse_ratio, seed: int = 0) -> dict:
    """Minimal Phase 3 active audit with given rmse_ratio.
    seed makes each window's dataset fingerprint unique so dedup doesn't collapse them.
    The dedup fingerprint keys on dataset.ticker+start+end+length+horizon.
    """
    resid_val = rmse_ratio if rmse_ratio is None else float(rmse_ratio)
    anchor_val = 1.0
    return {
        "dataset": {
            "ticker": "AAPL",
            "start": f"2020-01-0{seed % 9 + 1}",  # unique start per seed
            "end": f"2023-0{seed % 9 + 1}-01",
            "length": 100 + seed,
            "forecast_horizon": 30,
        },
        "artifacts": {
            "residual_experiment": {
                "experiment_id": "EXP-R5-001",
                "anchor_model_id": "mssa_rl",
                "residual_status": "active",
                "residual_active": True,
                "y_hat_anchor": [1.0, 2.0, 3.0],
                "y_hat_residual_ensemble": [1.1, 2.1, 3.1],
                "rmse_anchor": anchor_val,
                "rmse_residual_ensemble": resid_val,
                "rmse_ratio": resid_val,
                "da_anchor": 0.5,
                "da_residual_ensemble": 0.5,
                "corr_anchor_residual": 0.1,
            }
        },
    }


def _run_residual_pipeline(tmp_path, audit_payloads):
    """Write audit files and run quality pipeline; return summary JSON."""
    from scripts.run_quality_pipeline import run_quality_pipeline

    db = tmp_path / "db.db"
    _create_db_file(db)
    audit_dir = tmp_path / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in audit_payloads.items():
        (audit_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    residual_out = tmp_path / "residual_experiment_summary.json"
    with patch(
        "scripts.run_quality_pipeline.compute_eligibility",
        return_value={"summary": {"HEALTHY": 1, "WEAK": 0, "LAB_ONLY": 0}, "warnings": [], "n_tickers": 1},
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
        run_quality_pipeline(
            db_path=db,
            audit_dir=audit_dir,
            eligibility_out=tmp_path / "elig.json",
            context_out=tmp_path / "ctx.json",
            charts_out_dir=tmp_path / "charts",
            metrics_out=tmp_path / "m.json",
            enable_residual_experiment=True,
            residual_experiment_out=residual_out,
        )
    return json.loads(residual_out.read_text(encoding="utf-8"))


def test_early_abort_not_triggered_below_threshold(tmp_path):
    """4 consecutive bad windows (rmse_ratio > 1.02) does NOT trigger abort."""
    audits = {
        f"forecast_audit_2026010{i}_000000.json": _make_phase3_audit(1.05, seed=i)
        for i in range(1, 5)  # 4 unique windows
    }
    summary = _run_residual_pipeline(tmp_path, audits)
    assert summary["early_abort_signal"] is False
    assert summary["early_abort_consecutive_rmse_above_threshold"] == 4


def test_early_abort_triggered_at_five_consecutive(tmp_path):
    """5 consecutive windows with rmse_ratio > 1.02 triggers early abort."""
    audits = {
        f"forecast_audit_2026010{i}_000000.json": _make_phase3_audit(1.05, seed=i)
        for i in range(1, 6)  # 5 unique windows
    }
    summary = _run_residual_pipeline(tmp_path, audits)
    assert summary["early_abort_signal"] is True
    assert summary["early_abort_consecutive_rmse_above_threshold"] == 5
    assert any("EARLY_ABORT_SIGNAL" in w for w in summary["warnings"])


def test_early_abort_resets_on_good_window(tmp_path):
    """A good window (rmse_ratio <= 1.02) in the middle resets the streak."""
    # 3 bad, 1 good, 3 bad â€” max streak = 3, no abort
    audits = {
        "forecast_audit_20260101_000000.json": _make_phase3_audit(1.05, seed=1),
        "forecast_audit_20260102_000000.json": _make_phase3_audit(1.05, seed=2),
        "forecast_audit_20260103_000000.json": _make_phase3_audit(1.05, seed=3),
        "forecast_audit_20260104_000000.json": _make_phase3_audit(0.99, seed=4),  # good
        "forecast_audit_20260105_000000.json": _make_phase3_audit(1.05, seed=5),
        "forecast_audit_20260106_000000.json": _make_phase3_audit(1.05, seed=6),
        "forecast_audit_20260107_000000.json": _make_phase3_audit(1.05, seed=7),
    }
    summary = _run_residual_pipeline(tmp_path, audits)
    assert summary["early_abort_signal"] is False
    assert summary["early_abort_consecutive_rmse_above_threshold"] == 3


def test_early_abort_not_triggered_when_rmse_ratio_null(tmp_path):
    """Windows where rmse_ratio=null (Phase 2) do not contribute to abort streak."""
    # Phase 2 windows are retained as structural-only evidence (forecast vectors present),
    # but rmse_ratio=None does not exceed the 1.02 threshold so streak never advances.
    audits = {
        f"forecast_audit_2026010{i}_000000.json": _make_phase3_audit(
            rmse_ratio=None, seed=i  # type: ignore[arg-type]
        )
        for i in range(1, 8)  # 7 windows with rmse_ratio=null
    }
    summary = _run_residual_pipeline(tmp_path, audits)
    assert summary["early_abort_signal"] is False
    assert summary["early_abort_consecutive_rmse_above_threshold"] == 0


