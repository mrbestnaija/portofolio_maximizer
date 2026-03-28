#!/usr/bin/env python3
"""
Deterministic replay harness for the producer-native evidence contract.

This script exercises the stamped -> validate -> manifest -> latest sequence
without requiring live trading. It verifies writer-side provenance, admission,
and a small set of injected failure modes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from utils.evidence_io import atomic_write_json, build_manifest_entry, quarantine_file, upsert_jsonl_record


SCENARIOS = {
    "happy_path",
    "executed_round_trip",
    "missing_entry_link",
    "invalid_context",
    "missing_metadata",
    "duplicate_conflict",
    "misrouted_production",
    "manifest_registration_failure",
    "interrupted_promotion",
}


def _infer_cohort_id(output_dir: Path, explicit_cohort_id: Optional[str]) -> str:
    token = str(explicit_cohort_id or "").strip()
    if token:
        return token
    parts = [part.strip() for part in output_dir.parts]
    for index, part in enumerate(parts):
        if part.lower() == "cohorts" and index + 1 < len(parts):
            candidate = parts[index + 1].strip()
            if candidate:
                return candidate
    return "replay_shadow_v2"


def _base_payload(*, audit_id: str, cohort_id: str) -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_contract_version": 2,
        "audit_id": audit_id,
        "cohort_id": cohort_id,
        "event_type": "TRADE_FORECAST_AUDIT",
        "evidence_source": "producer",
        "evidence_source_classification": "producer-native",
        "dataset": {
            "ticker": "AAPL",
            "start": "2025-12-01T00:00:00+00:00",
            "end": "2026-01-02T00:00:00+00:00",
            "length": 23,
            "forecast_horizon": 30,
        },
        "artifacts": {
            "evaluation_metrics": {
                "ensemble": {"rmse": 0.92, "smape": 0.031, "tracking_error": 0.92},
                "samossa": {"rmse": 1.00, "smape": 0.035, "tracking_error": 1.00},
                "garch": {"rmse": 1.08, "smape": 0.041, "tracking_error": 1.08},
            }
        },
        "signal_context": {
            "context_type": "TRADE",
            "event_type": "TRADE_FORECAST_AUDIT",
            "producer_event_type": "TRADE_ENTRY",
            "ts_signal_id": "ts_AAPL_20260314_000001",
            "ticker": "AAPL",
            "run_id": "replay_20260314",
            "entry_ts": "2026-01-02T09:30:00+00:00",
            "forecast_horizon": 30,
            "expected_close_ts": "2026-02-03T09:30:00+00:00",
            "executed": True,
            "execution_status": "EXECUTED",
        },
        "lineage_v2": {
            "run_id": "replay_20260314",
            "forecast_id": 501,
            "signal_id": "ts_AAPL_20260314_000001",
            "order_id": "order_replay_001",
            "position_id": "position_replay_001",
            "entry_trade_id": 101,
            "close_trade_id": None,
            "audit_id": audit_id,
            "context_type": "TRADE",
            "event_type": "TRADE_FORECAST_AUDIT",
            "producer_event_type": "TRADE_ENTRY",
            "ts_signal_id": "ts_AAPL_20260314_000001",
            "entry_ts": "2026-01-02T09:30:00+00:00",
            "expected_close_ts": "2026-02-03T09:30:00+00:00",
            "executed": True,
            "execution_status": "EXECUTED",
        },
        "semantic_admission": {
            "admission_contract_version": 1,
            "accepted_for_audit_history": True,
            "admissible_for_readiness": True,
            "gate_eligible": True,
            "gate_bucket": "ELIGIBLE",
            "reason_code": "READY",
            "reason_codes": [],
            "production_labeled": True,
            "not_quarantined": True,
            "quarantined": False,
            "superseded": False,
            "duplicate_conflict": False,
            "missing_execution_metadata": False,
            "missing_execution_metadata_fields": [],
            "manifest_registered": None,
            "source": "producer",
            "source_classification": "producer-native",
        },
}


def _prepare_payload(scenario: str, *, cohort_id: str) -> Dict[str, Any]:
    payload = _base_payload(audit_id=f"replay_{scenario}", cohort_id=cohort_id)
    if scenario == "executed_round_trip":
        payload["audit_id"] = "forecast_audit_replay_executed_round_trip"
        payload["lineage_v2"]["audit_id"] = payload["audit_id"]
        payload["lineage_v2"]["close_trade_id"] = 202
        payload["semantic_admission"]["reason_code"] = "READY"
    if scenario == "missing_entry_link":
        payload["lineage_v2"]["producer_event_type"] = "TRADE_CLOSE"
        payload["lineage_v2"]["entry_trade_id"] = None
        payload["lineage_v2"]["close_trade_id"] = 202
        payload["signal_context"]["producer_event_type"] = "TRADE_CLOSE"
        payload["semantic_admission"].update(
            {
                "admissible_for_readiness": False,
                "gate_eligible": False,
                "gate_bucket": "ACCEPTED_NONELIGIBLE",
                "reason_code": "MISSING_ENTRY_TRADE_ID",
                "reason_codes": ["MISSING_ENTRY_TRADE_ID"],
                "missing_execution_metadata": True,
                "missing_execution_metadata_fields": ["entry_trade_id"],
            }
        )
    elif scenario == "invalid_context":
        payload["signal_context"]["context_type"] = "FORECAST_ONLY"
        payload["semantic_admission"].update(
            {
                "admissible_for_readiness": False,
                "gate_eligible": False,
                "gate_bucket": "ACCEPTED_NONELIGIBLE",
                "reason_code": "NON_TRADE_CONTEXT",
                "reason_codes": ["NON_TRADE_CONTEXT"],
            }
        )
    elif scenario == "missing_metadata":
        payload["signal_context"]["run_id"] = None
        payload["lineage_v2"]["run_id"] = None
        payload["semantic_admission"].update(
            {
                "admissible_for_readiness": False,
                "gate_eligible": False,
                "gate_bucket": "ACCEPTED_NONELIGIBLE",
                "reason_code": "MISSING_RUN_ID",
                "reason_codes": ["MISSING_RUN_ID"],
                "missing_execution_metadata": True,
                "missing_execution_metadata_fields": ["run_id"],
            }
        )
    elif scenario == "misrouted_production":
        payload["semantic_admission"].update(
            {
                "admissible_for_readiness": False,
                "gate_eligible": False,
                "gate_bucket": "ACCEPTED_NONELIGIBLE",
                "production_labeled": False,
                "reason_code": "ROUTING_AMBIGUOUS",
                "reason_codes": ["ROUTING_AMBIGUOUS"],
            }
        )
    payload["lineage_v2"]["audit_id"] = payload["audit_id"]
    payload["cohort_id"] = cohort_id
    return payload


def _canonical_replay_artifact_name(scenario: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    scenario_token = scenario.strip().lower()
    if scenario_token == "executed_round_trip":
        return f"forecast_audit_{timestamp}_AAPL_replay_executed_round_trip.json"
    return f"forecast_audit_{timestamp}_{scenario_token}.json"


def _write_proof_db(
    *,
    db_path: Path,
    ts_signal_id: str,
    ticker: str,
    run_id: str,
    entry_trade_id: int,
    close_trade_id: int,
    entry_ts: str,
    close_ts: str,
) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            DROP VIEW IF EXISTS production_closed_trades;
            DROP TABLE IF EXISTS trade_executions;
            CREATE TABLE trade_executions (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                trade_date TEXT,
                action TEXT,
                shares REAL,
                price REAL,
                total_value REAL,
                run_id TEXT,
                realized_pnl REAL,
                realized_pnl_pct REAL,
                holding_period_days INTEGER,
                is_close INTEGER,
                bar_timestamp TEXT,
                exit_reason TEXT,
                is_diagnostic INTEGER DEFAULT 0,
                is_synthetic INTEGER DEFAULT 0,
                entry_trade_id INTEGER,
                ts_signal_id TEXT
            );
            CREATE VIEW production_closed_trades AS
            SELECT *
            FROM trade_executions
            WHERE is_close = 1
              AND COALESCE(is_diagnostic, 0) = 0
              AND COALESCE(is_synthetic, 0) = 0;
            """
        )
        conn.execute(
            """
            INSERT INTO trade_executions (
                id, ticker, trade_date, action, shares, price, total_value,
                run_id, realized_pnl, realized_pnl_pct, holding_period_days,
                is_close, bar_timestamp, exit_reason, is_diagnostic, is_synthetic,
                entry_trade_id, ts_signal_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_trade_id,
                ticker,
                entry_ts[:10],
                "BUY",
                10.0,
                100.0,
                1000.0,
                run_id,
                None,
                None,
                None,
                0,
                entry_ts,
                None,
                0,
                0,
                None,
                ts_signal_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO trade_executions (
                id, ticker, trade_date, action, shares, price, total_value,
                run_id, realized_pnl, realized_pnl_pct, holding_period_days,
                is_close, bar_timestamp, exit_reason, is_diagnostic, is_synthetic,
                entry_trade_id, ts_signal_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                close_trade_id,
                ticker,
                close_ts[:10],
                "SELL",
                10.0,
                112.5,
                1125.0,
                run_id,
                125.0,
                0.125,
                32,
                1,
                close_ts,
                "TAKE_PROFIT",
                0,
                0,
                entry_trade_id,
                ts_signal_id,
            ),
        )
        conn.commit()
        return db_path
    finally:
        conn.close()


def _validate_payload(payload: Dict[str, Any]) -> tuple[bool, str]:
    semantic = payload.get("semantic_admission") if isinstance(payload.get("semantic_admission"), dict) else {}
    lineage = payload.get("lineage_v2") if isinstance(payload.get("lineage_v2"), dict) else {}
    producer_event_type = str(lineage.get("producer_event_type") or "").upper()
    if producer_event_type == "TRADE_CLOSE" and not lineage.get("entry_trade_id"):
        return False, "MISSING_ENTRY_TRADE_ID"
    if not bool(semantic.get("gate_eligible")):
        return False, str(semantic.get("reason_code") or "GATE_INELIGIBLE")
    return True, "READY"


def run_replay(*, output_dir: Path, scenario: str, cohort_id: Optional[str] = None) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamped_dir = output_dir / "stamped"
    quarantine_dir = output_dir / "quarantine"
    manifest_path = output_dir / "forecast_audit_manifest.jsonl"
    latest_path = output_dir / "latest.json"
    resolved_cohort_id = _infer_cohort_id(output_dir, cohort_id)
    payload = _prepare_payload(scenario, cohort_id=resolved_cohort_id)
    stamped_path = stamped_dir / f"{payload['audit_id']}.json"
    canonical_artifact_path: Optional[Path] = None
    proof_db_path: Optional[Path] = None

    def _write_and_register(candidate_payload: Dict[str, Any], candidate_path: Path) -> tuple[bool, str]:
        atomic_write_json(candidate_path, candidate_payload)
        valid, reason = _validate_payload(candidate_payload)
        if not valid:
            quarantine_file(
                candidate_path,
                quarantine_dir=quarantine_dir,
                reason=reason,
                metadata={"scenario": scenario},
            )
            return False, reason
        if scenario == "manifest_registration_failure":
            quarantine_file(
                candidate_path,
                quarantine_dir=quarantine_dir,
                reason="MANIFEST_REGISTRATION_FAILED",
                metadata={"scenario": scenario},
            )
            return False, "MANIFEST_REGISTRATION_FAILED"
        entry = build_manifest_entry(
            candidate_path,
            source="scripts.replay_trade_evidence_chain",
            extra={
                "audit_id": candidate_payload.get("audit_id"),
                "event_type": candidate_payload.get("event_type"),
                "cohort_id": candidate_payload.get("cohort_id"),
                "evidence_source": candidate_payload.get("evidence_source"),
                "evidence_source_classification": candidate_payload.get("evidence_source_classification"),
            },
        )
        if not entry:
            quarantine_file(
                candidate_path,
                quarantine_dir=quarantine_dir,
                reason="HASH_UNAVAILABLE",
                metadata={"scenario": scenario},
            )
            return False, "HASH_UNAVAILABLE"
        upsert_jsonl_record(manifest_path, entry, key_field="file")
        return True, "READY"

    ok, reason = _write_and_register(payload, stamped_path)
    if not ok:
        return {
            "scenario": scenario,
            "cohort_id": resolved_cohort_id,
            "status": "FAIL",
            "reason_code": reason,
            "latest_exists": latest_path.exists(),
            "quarantine_count": len(list(quarantine_dir.glob("*.json"))),
        }

    if scenario == "executed_round_trip":
        canonical_artifact_path = output_dir / _canonical_replay_artifact_name(scenario)
        atomic_write_json(canonical_artifact_path, payload)
        entry = build_manifest_entry(
            canonical_artifact_path,
            source="scripts.replay_trade_evidence_chain",
            extra={
                "audit_id": payload.get("audit_id"),
                "event_type": payload.get("event_type"),
                "cohort_id": payload.get("cohort_id"),
                "evidence_source": payload.get("evidence_source"),
                "evidence_source_classification": payload.get("evidence_source_classification"),
            },
        )
        if entry:
            upsert_jsonl_record(manifest_path, entry, key_field="file")
        signal_context = payload.get("signal_context") if isinstance(payload.get("signal_context"), dict) else {}
        lineage = payload.get("lineage_v2") if isinstance(payload.get("lineage_v2"), dict) else {}
        proof_db_path = _write_proof_db(
            db_path=output_dir / "proof.sqlite3",
            ts_signal_id=str(signal_context.get("ts_signal_id") or lineage.get("ts_signal_id") or ""),
            ticker=str(signal_context.get("ticker") or "AAPL"),
            run_id=str(signal_context.get("run_id") or lineage.get("run_id") or "replay_20260314"),
            entry_trade_id=int(lineage.get("entry_trade_id") or 101),
            close_trade_id=int(lineage.get("close_trade_id") or 202),
            entry_ts=str(signal_context.get("entry_ts") or lineage.get("entry_ts") or "2026-01-02T09:30:00+00:00"),
            close_ts=str(signal_context.get("expected_close_ts") or lineage.get("expected_close_ts") or "2026-02-03T09:30:00+00:00"),
        )

    if scenario == "interrupted_promotion":
        return {
            "scenario": scenario,
            "cohort_id": resolved_cohort_id,
            "status": "FAIL",
            "reason_code": "INTERRUPTED_BEFORE_LATEST_PROMOTION",
            "latest_exists": latest_path.exists(),
            "quarantine_count": len(list(quarantine_dir.glob("*.json"))),
        }

    if scenario == "duplicate_conflict":
        conflict_payload = deepcopy(payload)
        conflict_payload["signal_context"]["ts_signal_id"] = "ts_AAPL_20260314_conflict"
        conflict_payload["lineage_v2"]["ts_signal_id"] = "ts_AAPL_20260314_conflict"
        conflict_payload["semantic_admission"]["duplicate_conflict"] = True
        conflict_payload["semantic_admission"]["gate_eligible"] = False
        conflict_payload["semantic_admission"]["admissible_for_readiness"] = False
        conflict_payload["semantic_admission"]["gate_bucket"] = "QUARANTINED"
        conflict_payload["semantic_admission"]["reason_code"] = "DUPLICATE_CONFLICT"
        conflict_payload["semantic_admission"]["reason_codes"] = ["DUPLICATE_CONFLICT"]
        conflict_payload["semantic_admission"]["quarantined"] = True
        conflict_payload["semantic_admission"]["not_quarantined"] = False
        conflict_path = stamped_dir / f"{payload['audit_id']}_conflict.json"
        ok, reason = _write_and_register(conflict_payload, conflict_path)
        if ok:
            return {
                "scenario": scenario,
                "cohort_id": resolved_cohort_id,
                "status": "FAIL",
                "reason_code": "EXPECTED_DUPLICATE_CONFLICT",
                "latest_exists": latest_path.exists(),
                "quarantine_count": len(list(quarantine_dir.glob("*.json"))),
            }

    atomic_write_json(latest_path, payload)
    return {
        "scenario": scenario,
        "cohort_id": resolved_cohort_id,
        "status": "PASS",
        "reason_code": "READY",
        "latest_exists": latest_path.exists(),
        "quarantine_count": len(list(quarantine_dir.glob("*.json"))),
        "proof_db_path": str(proof_db_path) if proof_db_path is not None else None,
        "canonical_artifact_path": str(canonical_artifact_path) if canonical_artifact_path is not None else None,
    }


def _scenario_output_dir(output_dir: Path, scenario: str) -> Path:
    name = output_dir.name.strip().lower()
    if name == scenario.strip().lower():
        return output_dir
    if name in {"production", "research"}:
        return output_dir
    return output_dir / scenario


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="happy_path", choices=sorted(SCENARIOS))
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "evidence_replay")
    parser.add_argument("--cohort-id", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    summary = run_replay(
        output_dir=_scenario_output_dir(args.output_dir, args.scenario),
        scenario=args.scenario,
        cohort_id=args.cohort_id,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"scenario={summary['scenario']} status={summary['status']} reason={summary['reason_code']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
