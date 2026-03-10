"""
tests/scripts/test_hygiene_wiring.py

Tests for Phase 7.44 evidence-hygiene wiring fixes:
 - context_type="TRADE" explicitly set in _attach_signal_context_to_forecast_audit
 - Dataset horizon used (not signal horizon) to prevent HORIZON_MISMATCH
 - Ticker backfilled into dataset section when missing from audit JSON
 - ETL pipeline routes to research audit subdir (audit_log_dir key in ensemble_kwargs)
 - Forecaster defaults to production/ subdir when it exists
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_auto_trader import _attach_signal_context_to_forecast_audit  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_audit_payload(
    ticker: str = "AAPL",
    dataset_horizon: int = 30,
    signal_horizon: int = 30,
    ts_signal_id: str = "ts_AAPL_20260101_0001",
    run_id: str = "20260101_000000",
    entry_ts: str = "2026-01-01T00:00:00+00:00",
) -> Dict[str, Any]:
    """Return a minimal forecast audit payload."""
    return {
        "dataset": {
            "ticker": ticker,
            "start": "2024-01-01",
            "end": "2026-01-01",
            "length": 500,
            "forecast_horizon": dataset_horizon,
        },
        "signal_context": {},
        "results": {},
    }


def _make_execution_report(
    ts_signal_id: str = "ts_AAPL_20260101_0001",
    run_id: str = "20260101_000000",
    signal_timestamp: str = "2026-01-01T00:00:00+00:00",
) -> Dict[str, Any]:
    return {
        "ts_signal_id": ts_signal_id,
        "signal_timestamp": signal_timestamp,
        "action": "BUY",
    }


# ---------------------------------------------------------------------------
# Tests: context_type="TRADE" explicitly set
# ---------------------------------------------------------------------------

class TestContextTypeTradeWiring:
    """_attach_signal_context_to_forecast_audit must set context_type='TRADE'."""

    def test_context_type_trade_written_to_audit(self, tmp_path: Path) -> None:
        """After patching, signal_context.context_type must be 'TRADE'."""
        audit_file = tmp_path / "forecast_audit_20260101_000000_aaa.json"
        payload = _make_audit_payload()
        audit_file.write_text(json.dumps(payload), encoding="utf-8")

        forecast_bundle: Dict[str, Any] = {
            "horizon": 30,
            "forecast_audit_path": str(audit_file),
        }
        execution_report = _make_execution_report()

        with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
            _attach_signal_context_to_forecast_audit(
                forecast_bundle=forecast_bundle,
                execution_report=execution_report,
                ticker="AAPL",
                run_id="20260101_000000",
            )

        result = json.loads(audit_file.read_text())
        sc = result.get("signal_context", {})
        assert sc.get("context_type") == "TRADE", (
            f"Expected context_type='TRADE', got {sc.get('context_type')!r}"
        )

    def test_context_type_trade_overwrites_existing_forecast_only(self, tmp_path: Path) -> None:
        """Reverted Agent B code may have written FORECAST_ONLY — must be overwritten."""
        audit_file = tmp_path / "forecast_audit_20260306_000000_bbb.json"
        payload = _make_audit_payload()
        payload["signal_context"] = {"context_type": "FORECAST_ONLY", "ts_signal_id": None}
        audit_file.write_text(json.dumps(payload), encoding="utf-8")

        forecast_bundle = {"horizon": 30, "forecast_audit_path": str(audit_file)}
        execution_report = _make_execution_report()

        with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
            _attach_signal_context_to_forecast_audit(
                forecast_bundle=forecast_bundle,
                execution_report=execution_report,
                ticker="AAPL",
                run_id="20260101_000000",
            )

        result = json.loads(audit_file.read_text())
        assert result["signal_context"]["context_type"] == "TRADE"


# ---------------------------------------------------------------------------
# Tests: HORIZON_MISMATCH prevention via dataset horizon
# ---------------------------------------------------------------------------

class TestHorizonMismatchPrevention:
    """signal_context.forecast_horizon must come from dataset, not forecast_bundle."""

    def test_dataset_horizon_used_not_signal_horizon(self, tmp_path: Path) -> None:
        """When dataset.forecast_horizon=30 and forecast_bundle.horizon=2, use 30."""
        audit_file = tmp_path / "forecast_audit_20260309_000000_ccc.json"
        payload = _make_audit_payload(dataset_horizon=30)
        audit_file.write_text(json.dumps(payload), encoding="utf-8")

        forecast_bundle = {"horizon": 2, "forecast_audit_path": str(audit_file)}
        execution_report = _make_execution_report()

        with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
            _attach_signal_context_to_forecast_audit(
                forecast_bundle=forecast_bundle,
                execution_report=execution_report,
                ticker="AAPL",
                run_id="20260101_000000",
            )

        result = json.loads(audit_file.read_text())
        sc = result.get("signal_context", {})
        assert sc.get("forecast_horizon") == 30, (
            f"Expected 30 (dataset horizon), got {sc.get('forecast_horizon')!r}"
        )

    def test_fallback_to_signal_horizon_when_no_dataset_horizon(self, tmp_path: Path) -> None:
        """When dataset has no forecast_horizon, fall back to forecast_bundle.horizon."""
        audit_file = tmp_path / "forecast_audit_20260309_000000_ddd.json"
        payload = {"dataset": {"ticker": "AAPL"}, "results": {}}
        audit_file.write_text(json.dumps(payload), encoding="utf-8")

        forecast_bundle = {"horizon": 6, "forecast_audit_path": str(audit_file)}
        execution_report = _make_execution_report()

        with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
            _attach_signal_context_to_forecast_audit(
                forecast_bundle=forecast_bundle,
                execution_report=execution_report,
                ticker="AAPL",
                run_id="20260101_000000",
            )

        result = json.loads(audit_file.read_text())
        sc = result.get("signal_context", {})
        assert sc.get("forecast_horizon") == 6


# ---------------------------------------------------------------------------
# Tests: Ticker backfilled into dataset section
# ---------------------------------------------------------------------------

class TestTickerBackfill:
    """Ticker from signal_context must be backfilled into dataset when absent."""

    def test_ticker_backfilled_when_dataset_missing_ticker(self, tmp_path: Path) -> None:
        """If dataset has no ticker but signal_context does, backfill it."""
        audit_file = tmp_path / "forecast_audit_20260306_000000_eee.json"
        payload = {
            "dataset": {"length": 500, "forecast_horizon": 30},
            "results": {},
        }
        audit_file.write_text(json.dumps(payload), encoding="utf-8")

        forecast_bundle = {"horizon": 30, "forecast_audit_path": str(audit_file)}
        execution_report = _make_execution_report()

        with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
            _attach_signal_context_to_forecast_audit(
                forecast_bundle=forecast_bundle,
                execution_report=execution_report,
                ticker="NVDA",
                run_id="20260101_000000",
            )

        result = json.loads(audit_file.read_text())
        assert result["dataset"].get("ticker") == "NVDA", (
            "Ticker was not backfilled into dataset section"
        )

    def test_existing_dataset_ticker_not_overwritten(self, tmp_path: Path) -> None:
        """Existing ticker in dataset must not be overwritten."""
        audit_file = tmp_path / "forecast_audit_20260306_000000_fff.json"
        payload = _make_audit_payload(ticker="AAPL")
        audit_file.write_text(json.dumps(payload), encoding="utf-8")

        forecast_bundle = {"horizon": 30, "forecast_audit_path": str(audit_file)}
        execution_report = _make_execution_report()

        with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
            _attach_signal_context_to_forecast_audit(
                forecast_bundle=forecast_bundle,
                execution_report=execution_report,
                ticker="GOOG",  # Different ticker in signal vs audit
                run_id="20260101_000000",
            )

        result = json.loads(audit_file.read_text())
        # Existing dataset ticker should be preserved (no clobber)
        assert result["dataset"].get("ticker") == "AAPL"


# ---------------------------------------------------------------------------
# Tests: ETL pipeline routes to research audit dir
# ---------------------------------------------------------------------------

class TestEtlAuditDirRouting:
    """run_etl_pipeline._build_model_config must set audit_log_dir=research."""

    def test_ensemble_kwargs_has_research_audit_dir(self) -> None:
        """The ETL ensemble_kwargs must contain audit_log_dir pointing to research/."""
        # Import run_etl_pipeline and check the build config
        # We just inspect that audit_log_dir is in ensemble_kwargs when built
        # by parsing a minimal version of the _build_model_config closure.
        # This test validates the change we made to run_etl_pipeline.py line ~1889.
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "run_etl_pipeline_inspect",
                str(Path(__file__).parent.parent.parent / "scripts" / "run_etl_pipeline.py"),
            )
            # Rather than importing the whole module (which has side effects),
            # check the source text for the audit_log_dir wiring.
            src = (
                Path(__file__).parent.parent.parent / "scripts" / "run_etl_pipeline.py"
            ).read_text(encoding="utf-8")
            assert "audit_log_dir" in src, "ETL pipeline must wire audit_log_dir"
            assert "forecast_audits/research" in src, (
                "ETL pipeline must route to logs/forecast_audits/research"
            )
        except Exception as exc:
            pytest.fail(f"ETL pipeline audit_log_dir check failed: {exc}")


# ---------------------------------------------------------------------------
# Tests: Forecaster defaults to production/ when it exists
# ---------------------------------------------------------------------------

class TestForecasterAuditDirDefault:
    """Forecaster._audit_dir defaults to production/ subdir when it exists."""

    def test_forecaster_uses_production_subdir_when_exists(self, tmp_path: Path) -> None:
        """When logs/forecast_audits/production/ exists, default to it."""
        prod_dir = tmp_path / "logs" / "forecast_audits" / "production"
        prod_dir.mkdir(parents=True)

        # Monkeypatch os.getcwd() so Path("logs/...") resolves under tmp_path
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            # Remove cached env var so it uses code default
            env_backup = os.environ.pop("TS_FORECAST_AUDIT_DIR", None)
            try:
                from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
                fc = TimeSeriesForecaster(TimeSeriesForecasterConfig(forecast_horizon=5))
                assert fc._audit_dir is not None
                assert "production" in str(fc._audit_dir), (
                    f"Expected production/ in audit dir, got {fc._audit_dir}"
                )
            finally:
                if env_backup is not None:
                    os.environ["TS_FORECAST_AUDIT_DIR"] = env_backup
        finally:
            os.chdir(original_cwd)

    def test_forecaster_falls_back_when_production_missing(self, tmp_path: Path) -> None:
        """When production/ does not exist, fall back to logs/forecast_audits/."""
        root_dir = tmp_path / "logs" / "forecast_audits"
        root_dir.mkdir(parents=True)
        # production/ NOT created

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            env_backup = os.environ.pop("TS_FORECAST_AUDIT_DIR", None)
            try:
                from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
                fc = TimeSeriesForecaster(TimeSeriesForecasterConfig(forecast_horizon=5))
                assert fc._audit_dir is not None
                assert str(fc._audit_dir).replace("\\", "/").endswith("logs/forecast_audits"), (
                    f"Expected fallback to logs/forecast_audits, got {fc._audit_dir}"
                )
            finally:
                if env_backup is not None:
                    os.environ["TS_FORECAST_AUDIT_DIR"] = env_backup
        finally:
            os.chdir(original_cwd)
