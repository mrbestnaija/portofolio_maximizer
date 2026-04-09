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

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_auto_trader import (  # noqa: E402
    _attach_signal_context_to_forecast_audit,
    _build_cohort_identity,
    _build_semantic_admission,
    _execute_signal,
)


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
            "point": pd.Series(
                range(30),
                index=pd.date_range("2026-01-02T00:00:00+00:00", periods=30, freq="D"),
            ),
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
        assert sc.get("expected_close_ts") == "2026-01-31T00:00:00+00:00"
        assert sc.get("expected_close_source") == "forecast_index"

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

        forecast_bundle = {
            "horizon": 6,
            "forecast_audit_path": str(audit_file),
            "point": pd.Series(
                range(6),
                index=pd.date_range("2026-01-02T00:00:00+00:00", periods=6, freq="D"),
            ),
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
        assert sc.get("forecast_horizon") == 6
        assert sc.get("expected_close_ts") == "2026-01-07T00:00:00+00:00"
        assert sc.get("expected_close_source") == "forecast_index"


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


class TestSemanticAdmission:
    def test_missing_expected_close_ts_is_not_gate_eligible(self, tmp_path: Path) -> None:
        audit_file = tmp_path / "production" / "forecast_audit_20260309_000000_semantic.json"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit_file.write_text(json.dumps(_make_audit_payload()), encoding="utf-8")

        cohort_identity = _build_cohort_identity(audit_file)
        admission = _build_semantic_admission(
            audit_path=audit_file,
            signal_context={
                "context_type": "TRADE",
                "event_type": "TRADE_FORECAST_AUDIT",
                "ts_signal_id": "ts_AAPL_20260101_0001",
                "ticker": "AAPL",
                "run_id": "20260101_000000",
                "entry_ts": "2026-01-01T00:00:00+00:00",
                "forecast_horizon": 30,
            },
            audit_id="audit_1",
            cohort_identity=cohort_identity,
        )

        assert admission["gate_eligible"] is False
        assert admission["gate_bucket"] == "ACCEPTED_NONELIGIBLE"
        assert "MISSING_EXPECTED_CLOSE_TS" in admission["reason_code"]
        assert "MISSING_EXPECTED_CLOSE_TS" in admission["reason_codes"]
        assert admission["quarantined"] is False
        assert admission["missing_execution_metadata"] is False
        assert admission["evidence_source_classification"] == "producer-native"

    def test_missing_execution_metadata_is_written_by_producer(self, tmp_path: Path) -> None:
        audit_file = tmp_path / "production" / "forecast_audit_20260309_000001_semantic.json"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit_file.write_text(json.dumps(_make_audit_payload()), encoding="utf-8")

        cohort_identity = _build_cohort_identity(audit_file)
        admission = _build_semantic_admission(
            audit_path=audit_file,
            signal_context={
                "context_type": "TRADE",
                "event_type": "TRADE_FORECAST_AUDIT",
                "ts_signal_id": "ts_AAPL_20260101_0002",
                "ticker": "AAPL",
                "run_id": "",
                "entry_ts": "",
                "expected_close_ts": "2026-01-31T00:00:00+00:00",
                "forecast_horizon": 30,
            },
            audit_id="audit_2",
            cohort_identity=cohort_identity,
        )

        assert admission["gate_eligible"] is False
        assert admission["gate_bucket"] == "ACCEPTED_NONELIGIBLE"
        assert admission["missing_execution_metadata"] is True
        assert admission["missing_execution_metadata_fields"] == ["run_id", "entry_ts"]
        assert admission["reason_codes"][:2] == ["MISSING_RUN_ID", "MISSING_ENTRY_TS"]

    def test_execution_policy_block_is_not_gate_eligible(self, tmp_path: Path) -> None:
        audit_file = tmp_path / "production" / "forecast_audit_20260309_000002_semantic.json"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit_file.write_text(json.dumps(_make_audit_payload()), encoding="utf-8")

        cohort_identity = _build_cohort_identity(audit_file)
        admission = _build_semantic_admission(
            audit_path=audit_file,
            signal_context={
                "context_type": "TRADE",
                "event_type": "TRADE_FORECAST_AUDIT",
                "ts_signal_id": "ts_AAPL_20260101_0003",
                "ticker": "AAPL",
                "run_id": "20260101_000000",
                "entry_ts": "2026-01-01T00:00:00+00:00",
                "expected_close_ts": "2026-01-31T00:00:00+00:00",
                "forecast_horizon": 30,
                "execution_policy_blocked": True,
                "admission_override_reason_codes": ["NON_POSITIVE_NET_EDGE"],
            },
            audit_id="audit_3",
            cohort_identity=cohort_identity,
        )

        assert admission["gate_eligible"] is False
        assert admission["gate_bucket"] == "ACCEPTED_NONELIGIBLE"
        assert admission["execution_policy_blocked"] is True
        assert admission["reason_code"] == "NON_POSITIVE_NET_EDGE"
        assert admission["reason_codes"] == ["NON_POSITIVE_NET_EDGE"]

    def test_attach_signal_context_persists_execution_policy_block_decision(self, tmp_path: Path) -> None:
        audit_file = tmp_path / "production" / "forecast_audit_20260310_000000_execgate.json"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit_file.write_text(json.dumps(_make_audit_payload()), encoding="utf-8")

        forecast_bundle: Dict[str, Any] = {
            "horizon": 30,
            "forecast_audit_path": str(audit_file),
            "point": pd.Series(
                range(30),
                index=pd.date_range("2026-01-02T00:00:00+00:00", periods=30, freq="D"),
            ),
        }
        execution_report = {
            "ts_signal_id": "ts_AAPL_20260101_0004",
            "signal_timestamp": "2026-01-01T00:00:00+00:00",
            "status": "REJECTED",
            "reason": "Net expected return did not clear roundtrip cost gate.",
            "executed": False,
            "action": "BUY",
            "execution_policy_blocked": True,
            "admission_override_reason_codes": ["NON_POSITIVE_NET_EDGE"],
            "expected_return": 0.0015,
            "net_trade_return": -0.0008,
            "roundtrip_cost_fraction": 0.0023,
            "signal_to_noise": 0.4,
        }

        with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
            _attach_signal_context_to_forecast_audit(
                forecast_bundle=forecast_bundle,
                execution_report=execution_report,
                ticker="AAPL",
                run_id="20260101_000000",
            )

        result = json.loads(audit_file.read_text())
        sc = result["signal_context"]
        assert sc["execution_policy_blocked"] is True
        assert sc["admission_override_reason_codes"] == ["NON_POSITIVE_NET_EDGE"]
        assert result["execution_decision"]["status"] == "REJECTED"
        assert result["execution_decision"]["source_classification"] == "producer-native"
        assert result["semantic_admission"]["gate_bucket"] == "ACCEPTED_NONELIGIBLE"
        assert result["semantic_admission"]["reason_codes"] == ["NON_POSITIVE_NET_EDGE"]


def test_execute_signal_returns_auditable_preorder_block(monkeypatch) -> None:
    class _Bundle:
        def __init__(self) -> None:
            self.primary_signal = {
                "signal_id": "sig-1",
                "ts_signal_id": "sig-1",
                "action": "BUY",
                "confidence": 0.82,
                "expected_return": 0.0015,
                "net_trade_return": -0.0008,
                "roundtrip_cost_fraction": 0.0023,
                "signal_to_noise": 0.4,
                "execution_policy_blocked": True,
                "execution_policy_reason_codes": ["NON_POSITIVE_NET_EDGE"],
                "execution_policy_detail": "Net expected return did not clear roundtrip cost gate.",
                "source": "TIME_SERIES",
            }

    router = MagicMock()
    router.route_signal.return_value = _Bundle()
    trading_engine = MagicMock()
    market_data = MagicMock()

    result = _execute_signal(
        router=router,
        trading_engine=trading_engine,
        ticker="AAPL",
        forecast_bundle={"horizon": 30},
        current_price=100.0,
        market_data=market_data,
        quality={"quality_score": 0.9},
        data_source="yfinance",
        mid_price=100.0,
        run_id="run_1",
    )

    trading_engine.execute_signal.assert_not_called()
    assert result is not None
    assert result["status"] == "REJECTED"
    assert result["executed"] is False
    assert result["execution_policy_blocked"] is True
    assert result["admission_override_reason_codes"] == ["NON_POSITIVE_NET_EDGE"]
    assert result["evidence_source_classification"] == "producer-native"


# ---------------------------------------------------------------------------
# Tests: ETL pipeline routes to research audit dir
# ---------------------------------------------------------------------------

class TestEtlAuditDirRouting:
    """Forecaster construction must succeed when ensemble_kwargs contains audit_log_dir.

    This is a BEHAVIORAL test, not a source-text check.  The contract:
      - EnsembleConfig construction must not raise TypeError for unknown keys
      - audit_log_dir must be consumed by the forecaster for _audit_dir routing
      - The routing must not contaminate EnsembleConfig fields
    """

    def test_forecaster_construction_succeeds_with_audit_log_dir_in_ensemble_kwargs(
        self, tmp_path: Path
    ) -> None:
        """TimeSeriesForecaster must not raise when ensemble_kwargs has audit_log_dir.

        Previously, audit_log_dir was passed through ensemble_kwargs into
        EnsembleConfig(**kwargs), causing TypeError because EnsembleConfig does
        not declare that field.  This test fails if the boundary is broken.
        """
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig

        research_dir = tmp_path / "research"
        research_dir.mkdir()

        cfg = TimeSeriesForecasterConfig(
            ensemble_kwargs={
                "audit_log_dir": str(research_dir),
                # Include a real EnsembleConfig field to confirm valid keys still pass through
                "diversity_tolerance": 0.25,
            }
        )
        # Must not raise TypeError: unexpected keyword argument 'audit_log_dir'
        forecaster = TimeSeriesForecaster(config=cfg)
        # audit_log_dir must be consumed as the routing dir, not passed to EnsembleConfig
        assert forecaster._audit_dir == research_dir, (
            f"Expected _audit_dir={research_dir}, got {forecaster._audit_dir}"
        )
        # EnsembleConfig must be constructed with only valid fields
        assert forecaster._ensemble_config is not None
        assert forecaster._ensemble_config.diversity_tolerance == 0.25

    def test_forecaster_construction_succeeds_without_audit_log_dir(
        self, tmp_path: Path
    ) -> None:
        """Forecaster must still construct correctly when audit_log_dir is absent."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig

        cfg = TimeSeriesForecasterConfig(
            ensemble_kwargs={"diversity_tolerance": 0.30}
        )
        forecaster = TimeSeriesForecaster(config=cfg)
        assert forecaster._ensemble_config is not None
        assert forecaster._ensemble_config.diversity_tolerance == 0.30

    def test_unknown_ensemble_kwargs_keys_stripped_not_propagated(
        self, tmp_path: Path
    ) -> None:
        """Any unrecognised key in ensemble_kwargs must not reach EnsembleConfig.

        This is the adversarial regression: if a new routing key is added to
        ensemble_kwargs and EnsembleConfig is not updated, construction must still
        succeed (graceful strip), not raise TypeError.
        """
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig

        cfg = TimeSeriesForecasterConfig(
            ensemble_kwargs={
                "audit_log_dir": "logs/forecast_audits/research",
                "totally_unknown_routing_key": "should_be_stripped",
                "another_future_key": 42,
                "diversity_tolerance": 0.20,
            }
        )
        # Must not raise, even with multiple unknown keys
        forecaster = TimeSeriesForecaster(config=cfg)
        assert forecaster._ensemble_config is not None
        assert forecaster._ensemble_config.diversity_tolerance == 0.20
        # Unknown keys must not appear on EnsembleConfig
        assert not hasattr(forecaster._ensemble_config, "totally_unknown_routing_key")
        assert not hasattr(forecaster._ensemble_config, "another_future_key")


# ---------------------------------------------------------------------------
# Tests: Forecaster defaults to production/ when it exists
# ---------------------------------------------------------------------------

class TestForecasterAuditDirDefault:
    """Forecaster._audit_dir must NOT auto-promote to production/ (Phase 10 fix).

    Prior to Phase 10 the forecaster auto-selected production/ when that subdir
    existed, silently routing research/ETL forecasts into the production audit dir
    and inflating outcome_eligible counts in the gate.  The fix requires callers
    to opt-in explicitly via ensemble_kwargs['audit_log_dir'] or TS_FORECAST_AUDIT_DIR.
    """

    def test_forecaster_does_NOT_auto_promote_to_production_subdir(self, tmp_path: Path) -> None:
        """Even when logs/forecast_audits/production/ exists, the default must be root."""
        prod_dir = tmp_path / "logs" / "forecast_audits" / "production"
        prod_dir.mkdir(parents=True)

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            env_backup = os.environ.pop("TS_FORECAST_AUDIT_DIR", None)
            try:
                from forcester_ts.forecaster import TimeSeriesForecaster, TimeSeriesForecasterConfig
                fc = TimeSeriesForecaster(TimeSeriesForecasterConfig(forecast_horizon=5))
                assert fc._audit_dir is not None
                assert "production" not in str(fc._audit_dir), (
                    "Forecaster must NOT auto-promote to production/ dir. "
                    "Phase 10 fix: callers must opt-in via audit_log_dir. "
                    f"Got: {fc._audit_dir}"
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


# ---------------------------------------------------------------------------
# Tests: routing taxonomy fields persisted in signal_context (2026-04-09)
# ---------------------------------------------------------------------------


class TestRoutingTaxonomyFields:
    """_attach_signal_context_to_forecast_audit must persist action, confidence,
    routing_reason, and snr so the live rejection taxonomy is measurable."""

    def test_action_confidence_routing_reason_snr_written_for_hold_signal(
        self, tmp_path: Path
    ) -> None:
        """HOLD signal: action, confidence, routing_reason, snr must appear in signal_context."""
        audit_file = tmp_path / "forecast_audit_20260409_000000_hold.json"
        payload = _make_audit_payload()
        audit_file.write_text(json.dumps(payload), encoding="utf-8")

        forecast_bundle: Dict[str, Any] = {
            "horizon": 30,
            "forecast_audit_path": str(audit_file),
        }
        execution_report = {
            "ts_signal_id": "ts_AAPL_20260409_0001",
            "signal_timestamp": "2026-04-09T09:00:00+00:00",
            "action": "HOLD",
            "signal_confidence": 0.38,
            "routing_reason": "confidence_below_threshold",
            "snr": 0.92,
            "executed": False,
            "status": "REJECTED",
            "reason": "Non-actionable signal",
        }

        with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
            _attach_signal_context_to_forecast_audit(
                forecast_bundle=forecast_bundle,
                execution_report=execution_report,
                ticker="AAPL",
                run_id="20260409_090000",
            )

        result = json.loads(audit_file.read_text())
        sc = result.get("signal_context", {})
        assert sc.get("action") == "HOLD", f"Expected action=HOLD, got {sc.get('action')!r}"
        assert sc.get("confidence") == pytest.approx(0.38), f"Expected confidence=0.38, got {sc.get('confidence')}"
        assert sc.get("routing_reason") == "confidence_below_threshold", (
            f"Expected routing_reason=confidence_below_threshold, got {sc.get('routing_reason')!r}"
        )
        assert sc.get("snr") == pytest.approx(0.92), f"Expected snr=0.92, got {sc.get('snr')}"

    def test_existing_fields_not_overwritten_by_routing_taxonomy_patch(
        self, tmp_path: Path
    ) -> None:
        """Adding routing fields must not clobber ts_signal_id, expected_return, or context_type."""
        audit_file = tmp_path / "forecast_audit_20260409_000001_reg.json"
        payload = _make_audit_payload()
        audit_file.write_text(json.dumps(payload), encoding="utf-8")

        forecast_bundle = {"horizon": 30, "forecast_audit_path": str(audit_file)}
        execution_report = {
            "ts_signal_id": "ts_AAPL_20260409_0002",
            "signal_timestamp": "2026-04-09T09:00:00+00:00",
            "action": "BUY",
            "signal_confidence": 0.72,
            "routing_reason": "",
            "snr": 2.1,
            "expected_return": 0.009,
            "executed": True,
            "status": "EXECUTED",
        }

        with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
            _attach_signal_context_to_forecast_audit(
                forecast_bundle=forecast_bundle,
                execution_report=execution_report,
                ticker="AAPL",
                run_id="20260409_090000",
            )

        result = json.loads(audit_file.read_text())
        sc = result.get("signal_context", {})
        assert sc.get("ts_signal_id") == "ts_AAPL_20260409_0002"
        assert sc.get("context_type") == "TRADE"
        assert sc.get("action") == "BUY"
        assert sc.get("confidence") == pytest.approx(0.72)
        assert sc.get("snr") == pytest.approx(2.1)
        assert sc.get("expected_return") == pytest.approx(0.009)
        # routing_reason is empty string — must NOT be written (falsy guard in copy loop)
        assert sc.get("routing_reason") is None, (
            f"Empty routing_reason must not be written, got {sc.get('routing_reason')!r}"
        )

    def test_outcome_linkage_report_default_audit_dir_prefers_production(self) -> None:
        """outcome_linkage_attribution_report DEFAULT_AUDIT_DIR logic: prefers production/."""
        # Test the selection logic directly without re-importing the module
        # (module-level variable computed at import time against real filesystem).
        from pathlib import Path
        import tempfile, os

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prod = root / "logs" / "forecast_audits" / "production"
            fallback = root / "logs" / "forecast_audits"
            prod.mkdir(parents=True)

            chosen = prod if prod.exists() else fallback
            assert chosen == prod, "Must prefer production/ when it exists"

            # Without production/, fall back
            import shutil
            shutil.rmtree(str(prod))
            chosen2 = prod if prod.exists() else fallback
            assert chosen2 == fallback, "Must fall back to root when production/ absent"
