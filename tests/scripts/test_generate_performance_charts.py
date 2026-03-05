"""
Smoke tests for scripts/generate_performance_charts.py
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest


class TestImportability:
    def test_script_is_importable(self):
        import importlib

        mod = importlib.import_module("scripts.generate_performance_charts")
        assert hasattr(mod, "main")
        assert hasattr(mod, "generate_performance_artifacts")
        assert hasattr(mod, "chart_per_ticker_wr_pf")

    def test_threshold_constants_match_shared_source(self):
        from scripts.generate_performance_charts import R3_WIN_RATE
        from scripts.robustness_thresholds import MIN_LIFT_FRACTION, R3_MIN_WIN_RATE

        assert R3_WIN_RATE == R3_MIN_WIN_RATE
        assert MIN_LIFT_FRACTION >= 0.0


class TestCharts:
    def test_chart_functions_skip_gracefully(self, tmp_path):
        pytest.importorskip("matplotlib")
        from scripts.generate_performance_charts import (
            chart_context_quality_heatmap,
            chart_global_wr_over_time,
            chart_per_ticker_wr_pf,
            chart_ticker_eligibility_grid,
        )

        chart_per_ticker_wr_pf([], tmp_path / "per_ticker.png")
        chart_global_wr_over_time([], tmp_path / "wr.png")
        chart_ticker_eligibility_grid({}, tmp_path / "elig.png")
        chart_context_quality_heatmap({}, tmp_path / "ctx.png")
        assert not (tmp_path / "per_ticker.png").exists()

    def test_generate_artifacts_writes_default_metrics_json(self, tmp_path):
        from scripts.generate_performance_charts import generate_performance_artifacts

        out_dir = tmp_path / "charts"
        metrics_path = out_dir / "metrics_summary.json"
        result = generate_performance_artifacts(
            db_path=tmp_path / "missing.db",
            audit_dir=tmp_path / "missing_audits",
            out_dir=out_dir,
            eligibility_path=tmp_path / "missing_elig.json",
            context_quality_path=tmp_path / "missing_ctx.json",
            json_metrics_path=metrics_path,
        )
        assert metrics_path.exists()
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert "thresholds" in payload
        assert "source_hashes" in payload["thresholds"]
        assert "eligibility_counts" in payload
        assert "context_summary" in payload
        assert "sufficiency_status" in payload
        assert "warnings" in payload
        assert "errors" in payload
        assert result["metrics_path"] == str(metrics_path)
        assert result["status"] == "ERROR"
        assert result["errors"]

    def test_lift_axis_bounds_preserve_negative_ci(self):
        from scripts.generate_performance_charts import _lift_axis_bounds

        ymin, ymax = _lift_axis_bounds(
            lift_global_pct=1.5,
            lift_recent_pct=10.0,
            threshold_pct=25.0,
            ci_low_pct=-7.2,
            ci_high_pct=-2.8,
        )
        assert ymin < -7.2
        assert ymax > 25.0

    def test_missing_artifacts_fail_closed_in_strict_mode(self, tmp_path):
        from scripts.generate_performance_charts import generate_performance_artifacts

        with patch("scripts.generate_performance_charts.chart_per_ticker_wr_pf"), patch(
            "scripts.generate_performance_charts.chart_global_wr_over_time"
        ), patch("scripts.generate_performance_charts.chart_lift_global_vs_recent"), patch(
            "scripts.generate_performance_charts.chart_ticker_eligibility_grid"
        ), patch("scripts.generate_performance_charts.chart_context_quality_heatmap"):
            result = generate_performance_artifacts(
                db_path=tmp_path / "missing.db",
                audit_dir=tmp_path / "missing_audits",
                out_dir=tmp_path / "charts",
                eligibility_path=tmp_path / "missing_elig.json",
                context_quality_path=tmp_path / "missing_ctx.json",
                json_metrics_path=tmp_path / "charts" / "metrics_summary.json",
                sufficiency={"status": "SUFFICIENT", "sufficient": True, "recommendations": []},
                strict_mode=True,
            )

        assert result["status"] == "ERROR"
        assert any(item.startswith("chart_missing:") for item in result["errors"])

    def test_missing_artifacts_are_warn_only_when_not_strict(self, tmp_path):
        from scripts.generate_performance_charts import generate_performance_artifacts

        with patch("scripts.generate_performance_charts.chart_per_ticker_wr_pf"), patch(
            "scripts.generate_performance_charts.chart_global_wr_over_time"
        ), patch("scripts.generate_performance_charts.chart_lift_global_vs_recent"), patch(
            "scripts.generate_performance_charts.chart_ticker_eligibility_grid"
        ), patch("scripts.generate_performance_charts.chart_context_quality_heatmap"):
            result = generate_performance_artifacts(
                db_path=tmp_path / "missing.db",
                audit_dir=tmp_path / "missing_audits",
                out_dir=tmp_path / "charts",
                eligibility_path=tmp_path / "missing_elig.json",
                context_quality_path=tmp_path / "missing_ctx.json",
                json_metrics_path=tmp_path / "charts" / "metrics_summary.json",
                sufficiency={"status": "SUFFICIENT", "sufficient": True, "recommendations": []},
                strict_mode=False,
            )

        assert result["status"] == "WARN"
        assert result["errors"] == []
        assert any(item.startswith("chart_missing:") for item in result["warnings"])


class TestMain:
    def test_main_returns_1_with_missing_paths_in_strict_mode(self, tmp_path):
        from scripts.generate_performance_charts import main

        out_dir = tmp_path / "charts"
        rc = main(
            [
                "--db",
                str(tmp_path / "missing.db"),
                "--audit-dir",
                str(tmp_path / "missing_audits"),
                "--out-dir",
                str(out_dir),
            ]
        )
        assert rc == 1
        assert (out_dir / "metrics_summary.json").exists()
