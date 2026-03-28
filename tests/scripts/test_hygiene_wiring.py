from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_auto_trader import (  # noqa: E402
    _audit_payload_matches_ticker,
    _attach_signal_context_to_forecast_audit,
    _build_cohort_identity,
    _build_semantic_admission,
    _raw_staging_path_matches_ticker,
    _resolve_forecast_audit_path,
    _stabilize_forecast_audit_artifact,
)


def _make_audit_payload(ticker: str = "AAPL", dataset_horizon: int = 30) -> Dict[str, Any]:
    return {
        "dataset": {
            "ticker": ticker,
            "forecast_horizon": dataset_horizon,
        },
        "signal_context": {},
        "results": {},
    }


def _make_nested_component_audit_payload(ticker: str = "AAPL") -> Dict[str, Any]:
    return {
        "dataset": {
            "forecast_horizon": 30,
        },
        "artifacts": {
            "component_summaries": {
                "events": [
                    {"model": "samossa", "ticker": ticker},
                ]
            }
        },
    }


def _make_execution_report(**overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ts_signal_id": "ts_AAPL_20260101_0001",
        "signal_id": "ts_AAPL_20260101_0001",
        "signal_timestamp": "2026-01-01T00:00:00+00:00",
        "executed": True,
        "status": "EXECUTED",
        "event_type": "TRADE_ENTRY",
        "forecast_id": 123,
        "order_id": "order_123",
        "position_id": "position_123",
        "entry_trade_id": 456,
        "close_trade_id": None,
    }
    payload.update(overrides)
    return payload


def test_attach_signal_context_writes_producer_native_provenance_and_lineage(tmp_path: Path) -> None:
    audit_file = tmp_path / "production" / "forecast_audit_20260101_000000.json"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(_make_audit_payload()), encoding="utf-8")

    forecast_bundle = {
        "horizon": 2,
        "forecast_audit_path": str(audit_file),
        "forecast_id": 321,
        "forecast_record_ids": {"COMBINED": 321},
    }
    execution_report = _make_execution_report()

    with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
        _attach_signal_context_to_forecast_audit(
            forecast_bundle=forecast_bundle,
            execution_report=execution_report,
            ticker="AAPL",
            run_id="20260101_000000",
        )

    result = json.loads(audit_file.read_text(encoding="utf-8"))
    assert result["evidence_source"] == "producer"
    assert result["evidence_source_classification"] == "producer-native"
    assert result["semantic_admission"]["source"] == "producer"
    assert result["semantic_admission"]["gate_eligible"] is True
    assert result["signal_context"]["context_type"] == "TRADE"
    assert result["signal_context"]["producer_event_type"] == "TRADE_ENTRY"
    assert result["lineage_v2"]["forecast_id"] == 321
    assert result["lineage_v2"]["position_id"] == "position_123"
    assert result["lineage_v2"]["entry_trade_id"] == 456
    assert result["lineage_v2"]["close_trade_id"] is None
    assert result["lineage_v2"]["order_id"] == "order_123"
    assert result["lineage_v2"]["producer_event_type"] == "TRADE_ENTRY"
    manifest_path = audit_file.parent / "forecast_audit_manifest.jsonl"
    assert manifest_path.exists()
    manifest_entry = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert manifest_entry["evidence_source"] == "producer"
    assert manifest_entry["evidence_source_classification"] == "producer-native"


def test_missing_close_linkage_fields_are_noneligible(tmp_path: Path) -> None:
    audit_file = tmp_path / "production" / "forecast_audit_close.json"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(_make_audit_payload()), encoding="utf-8")
    cohort_identity = _build_cohort_identity(audit_file)

    execution_report = _make_execution_report(
        event_type="TRADE_CLOSE",
        entry_trade_id=None,
        close_trade_id=None,
    )
    admission = _build_semantic_admission(
        audit_path=audit_file,
        signal_context={
            "context_type": "TRADE",
            "event_type": "TRADE_FORECAST_AUDIT",
            "producer_event_type": "TRADE_CLOSE",
            "ts_signal_id": "ts_AAPL_20260101_0001",
            "ticker": "AAPL",
            "run_id": "20260101_000000",
            "entry_ts": "2026-01-01T00:00:00+00:00",
            "expected_close_ts": "2026-01-31T00:00:00+00:00",
            "forecast_horizon": 30,
            "executed": True,
        },
        execution_report=execution_report,
        audit_id="audit_close",
        cohort_identity=cohort_identity,
    )

    assert admission["gate_eligible"] is False
    assert admission["gate_bucket"] == "ACCEPTED_NONELIGIBLE"
    assert admission["missing_execution_metadata"] is True
    assert "MISSING_ENTRY_TRADE_ID" in admission["reason_codes"]
    assert "MISSING_CLOSE_TRADE_ID" in admission["reason_codes"]
    assert "entry_trade_id" in admission["missing_execution_metadata_fields"]
    assert "close_trade_id" in admission["missing_execution_metadata_fields"]


def test_nonexecuted_report_never_becomes_eligible(tmp_path: Path) -> None:
    audit_file = tmp_path / "production" / "forecast_audit_rejected.json"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(_make_audit_payload()), encoding="utf-8")
    cohort_identity = _build_cohort_identity(audit_file)

    execution_report = _make_execution_report(
        executed=False,
        status="REJECTED",
        event_type="TRADE_SIGNAL_REJECTED",
        order_id=None,
        position_id=None,
        entry_trade_id=None,
        forecast_id=None,
    )
    admission = _build_semantic_admission(
        audit_path=audit_file,
        signal_context={
            "context_type": "TRADE",
            "event_type": "TRADE_FORECAST_AUDIT",
            "producer_event_type": "TRADE_SIGNAL_REJECTED",
            "ts_signal_id": "ts_AAPL_20260101_0001",
            "ticker": "AAPL",
            "run_id": "20260101_000000",
            "entry_ts": "2026-01-01T00:00:00+00:00",
            "expected_close_ts": "2026-01-31T00:00:00+00:00",
            "forecast_horizon": 30,
            "executed": False,
        },
        execution_report=execution_report,
        audit_id="audit_rejected",
        cohort_identity=cohort_identity,
    )

    assert admission["gate_eligible"] is False
    assert admission["gate_bucket"] == "ACCEPTED_NONELIGIBLE"
    assert "NOT_EXECUTED" in admission["reason_codes"]
    assert "MISSING_FORECAST_ID" not in admission["reason_codes"]


def test_missing_audit_candidate_writes_explicit_failure_artifact(tmp_path: Path) -> None:
    production_dir = tmp_path / "logs" / "forecast_audits" / "production"
    production_dir.mkdir(parents=True, exist_ok=True)
    forecast_bundle = {
        "horizon": 5,
        "forecast_audit_path": str(production_dir / "missing_forecast_audit.json"),
        "forecast_id": 123,
    }
    execution_report = _make_execution_report(ticker="AAPL")

    with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
        result = _attach_signal_context_to_forecast_audit(
            forecast_bundle=forecast_bundle,
            execution_report=execution_report,
            ticker="AAPL",
            run_id="20260101_000000",
        )

    assert result["patched"] is False
    assert result["explicit_failure"] is not None
    failure_path = Path(result["explicit_failure"]["artifact_path"])
    assert failure_path.exists()
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "AUDIT_PATCH_FAILURE"
    assert payload["reconciliation_bucket"] == "EXPLICIT_FAILED"
    assert payload["evidence_source"] == "producer"
    assert payload["semantic_admission"]["gate_bucket"] == "ACCEPTED_NONELIGIBLE"
    manifest_path = production_dir / "audit_failure_manifest.jsonl"
    assert manifest_path.exists()
    manifest_entry = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    assert manifest_entry["evidence_source"] == "producer"
    assert manifest_entry["evidence_source_classification"] == "producer-native"


def test_missing_audit_candidate_uses_forecast_audit_dir_for_explicit_failure(tmp_path: Path) -> None:
    production_dir = tmp_path / "logs" / "forecast_audits" / "cohorts" / "demo" / "production"
    production_dir.mkdir(parents=True, exist_ok=True)
    forecast_bundle = {
        "horizon": 5,
        "forecast_audit_dir": str(production_dir),
        "forecast_id": 123,
    }
    execution_report = _make_execution_report(ticker="AAPL")

    with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
        result = _attach_signal_context_to_forecast_audit(
            forecast_bundle=forecast_bundle,
            execution_report=execution_report,
            ticker="AAPL",
            run_id="20260101_000000",
        )

    assert result["patched"] is False
    assert result["explicit_failure"] is not None
    failure_path = Path(result["explicit_failure"]["artifact_path"])
    assert failure_path.parent == production_dir
    assert (production_dir / "audit_failure_manifest.jsonl").exists()


def test_collision_hardened_audit_copy_preserves_symbol_specific_attachment(tmp_path: Path) -> None:
    production_dir = tmp_path / "production"
    production_dir.mkdir(parents=True, exist_ok=True)
    source_path = production_dir / "forecast_audit_20260101_000000.json"
    source_path.write_text(json.dumps(_make_audit_payload("AAPL")), encoding="utf-8")
    forecast_bundle = {
        "horizon": 5,
        "forecast_audit_path": str(source_path),
        "forecast_id": 123,
    }
    execution_report = _make_execution_report()

    stabilized = _stabilize_forecast_audit_artifact(forecast_bundle, ticker="AAPL")
    assert stabilized is not None
    copied_path = Path(stabilized["forecast_audit_path"])
    assert copied_path.exists()
    assert copied_path != source_path
    assert stabilized["forecast_audit_collision_hardened"] is True

    source_path.write_text(json.dumps(_make_audit_payload("MSFT")), encoding="utf-8")

    with patch("scripts.run_auto_trader.ROOT_PATH", tmp_path):
        result = _attach_signal_context_to_forecast_audit(
            forecast_bundle=stabilized,
            execution_report=execution_report,
            ticker="AAPL",
            run_id="20260101_000000",
        )

    assert result["patched"] is True
    copied_payload = json.loads(copied_path.read_text(encoding="utf-8"))
    assert copied_payload["dataset"]["ticker"] == "AAPL"
    assert copied_payload["semantic_admission"]["gate_eligible"] is True
    original_payload = json.loads(source_path.read_text(encoding="utf-8"))
    assert original_payload["dataset"]["ticker"] == "MSFT"


def test_collision_hardened_copy_promotes_raw_staging_artifact_into_canonical_dir(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "production"
    raw_dir = canonical_dir / "_raw_forecaster" / "AAPL"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_path = raw_dir / "forecast_audit_20260101_000000.json"
    source_path.write_text(json.dumps(_make_audit_payload("AAPL")), encoding="utf-8")

    stabilized = _stabilize_forecast_audit_artifact(
        {
            "forecast_audit_path": str(source_path),
            "forecast_audit_dir": str(canonical_dir),
        },
        ticker="AAPL",
    )

    assert stabilized is not None
    copied_path = Path(stabilized["forecast_audit_path"])
    assert copied_path.exists()
    assert copied_path.parent == canonical_dir
    assert copied_path.parent != raw_dir
    copied_payload = json.loads(copied_path.read_text(encoding="utf-8"))
    assert copied_payload["dataset"]["ticker"] == "AAPL"
    assert copied_payload["forecast_audit_source_path"] == str(source_path)
    assert copied_payload["forecast_audit_copy_reason"] == "collision_hardened"


def test_audit_payload_matches_ticker_from_nested_component_events() -> None:
    payload = _make_nested_component_audit_payload("AAPL")

    assert _audit_payload_matches_ticker(payload, "AAPL") is True
    assert _audit_payload_matches_ticker(payload, "MSFT") is False


def test_raw_staging_path_matches_ticker_without_dataset_ticker(tmp_path: Path) -> None:
    raw_path = tmp_path / "production" / "_raw_forecaster" / "NVDA" / "forecast_audit_20260101_000000.json"

    assert _raw_staging_path_matches_ticker(raw_path, "NVDA") is True
    assert _raw_staging_path_matches_ticker(raw_path, "MSFT") is False
    assert _audit_payload_matches_ticker({"dataset": {}}, "NVDA", artifact_path=raw_path) is True


def test_resolve_forecast_audit_path_prefers_ticker_matched_content(tmp_path: Path) -> None:
    audit_dir = tmp_path / "production"
    audit_dir.mkdir(parents=True, exist_ok=True)
    latest_path = audit_dir / "forecast_audit_20260101_000001.json"
    matched_path = audit_dir / "forecast_audit_20260101_000000.json"
    matched_path.write_text(json.dumps(_make_audit_payload("MSFT")), encoding="utf-8")
    latest_path.write_text(json.dumps(_make_audit_payload("AAPL")), encoding="utf-8")

    forecast_bundle = {"forecast_audit_path": str(latest_path)}
    resolved = _resolve_forecast_audit_path(
        forecast_bundle,
        audit_dir=audit_dir,
        ticker="MSFT",
    )

    assert resolved is not None
    assert Path(resolved["forecast_audit_path"]) == matched_path
    assert resolved["forecast_audit_resolution"] == "ticker_matched_content"


def test_resolve_forecast_audit_path_prefers_raw_staging_match_when_payload_is_anonymous(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "production"
    raw_dir = canonical_dir / "_raw_forecaster" / "MSFT"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "forecast_audit_20260101_000000.json"
    raw_path.write_text(json.dumps({"dataset": {"forecast_horizon": 30}}), encoding="utf-8")

    forecast_bundle = {
        "forecast_audit_dir": str(canonical_dir),
        "forecast_audit_raw_dir": str(raw_dir),
    }
    resolved = _resolve_forecast_audit_path(
        forecast_bundle,
        audit_dir=canonical_dir,
        ticker="MSFT",
    )

    assert resolved is not None
    assert Path(resolved["forecast_audit_path"]) == raw_path
    assert resolved["forecast_audit_resolution"] == "ticker_matched_raw_staging"


def test_stabilize_forecast_audit_artifact_uses_raw_staging_dir_when_direct_path_missing(tmp_path: Path) -> None:
    canonical_dir = tmp_path / "production"
    raw_dir = canonical_dir / "_raw_forecaster" / "MSFT"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "forecast_audit_20260101_000000.json"
    raw_path.write_text(json.dumps({"dataset": {"forecast_horizon": 30}}), encoding="utf-8")

    stabilized = _stabilize_forecast_audit_artifact(
        {
            "forecast_audit_dir": str(canonical_dir),
            "forecast_audit_raw_dir": str(raw_dir),
        },
        ticker="MSFT",
    )

    assert stabilized is not None
    copied_path = Path(stabilized["forecast_audit_path"])
    assert copied_path.exists()
    assert copied_path.parent == canonical_dir
    copied_payload = json.loads(copied_path.read_text(encoding="utf-8"))
    assert copied_payload["forecast_audit_source_path"] == str(raw_path)
    assert copied_payload["dataset"]["ticker"] == "MSFT"
    assert copied_payload["forecast_audit_candidate_ticker"] == "MSFT"
