#!/usr/bin/env python3
"""
Deterministic replay harness for the momentum-safe evidence contract.

This script exercises the stamped -> validate -> manifest -> latest sequence
without requiring live trading. It is intentionally narrow: it verifies the
evidence lifecycle contract and a small set of injected failure modes.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from utils.evidence_io import atomic_write_json, build_manifest_entry, quarantine_file, upsert_jsonl_record


SCENARIOS = {
    "happy_path",
    "missing_entry_link",
    "invalid_context",
    "missing_metadata",
    "duplicate_conflict",
    "misrouted_production",
    "manifest_registration_failure",
    "interrupted_promotion",
}


def _base_payload(*, audit_id: str) -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_contract_version": 2,
        "audit_id": audit_id,
        "cohort_id": "replay_shadow_v2",
        "event_type": "TRADE_FORECAST_AUDIT",
        "signal_context": {
            "context_type": "TRADE",
            "event_type": "TRADE_FORECAST_AUDIT",
            "ts_signal_id": "ts_AAPL_20260314_000001",
            "ticker": "AAPL",
            "run_id": "replay_20260314",
            "entry_ts": "2026-03-14T09:30:00+00:00",
            "forecast_horizon": 5,
        },
        "lineage_v2": {
            "run_id": "replay_20260314",
            "forecast_id": None,
            "signal_id": "ts_AAPL_20260314_000001",
            "order_id": "order_replay_001",
            "position_id": "position_replay_001",
            "entry_trade_id": 101,
            "close_trade_id": None,
            "audit_id": audit_id,
            "context_type": "TRADE",
            "event_type": "TRADE_FORECAST_AUDIT",
            "ts_signal_id": "ts_AAPL_20260314_000001",
        },
        "semantic_admission": {
            "accepted_for_audit_history": True,
            "admissible_for_readiness": True,
            "gate_eligible": True,
            "gate_bucket": "ELIGIBLE",
            "reason_code": "READY",
            "production_labeled": True,
            "not_quarantined": True,
            "superseded": False,
            "duplicate_conflict": False,
            "manifest_registered": None,
        },
    }


def _prepare_payload(scenario: str) -> Dict[str, Any]:
    payload = _base_payload(audit_id=f"replay_{scenario}")
    if scenario == "missing_entry_link":
        payload["event_type"] = "TRADE_CLOSE"
        payload["lineage_v2"]["event_type"] = "TRADE_CLOSE"
        payload["lineage_v2"]["entry_trade_id"] = None
        payload["lineage_v2"]["close_trade_id"] = 202
        payload["semantic_admission"].update(
            {
                "admissible_for_readiness": False,
                "gate_eligible": False,
                "gate_bucket": "ACCEPTED_NONELIGIBLE",
                "reason_code": "MISSING_ENTRY_TRADE_ID",
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
            }
        )
    return payload


def _validate_payload(payload: Dict[str, Any]) -> tuple[bool, str]:
    semantic = payload.get("semantic_admission") if isinstance(payload.get("semantic_admission"), dict) else {}
    if str(payload.get("event_type") or "").upper() == "TRADE_CLOSE" and not payload.get("lineage_v2", {}).get("entry_trade_id"):
        return False, "MISSING_ENTRY_TRADE_ID"
    if not bool(semantic.get("gate_eligible")):
        return False, str(semantic.get("reason_code") or "GATE_INELIGIBLE")
    return True, "READY"


def run_replay(*, output_dir: Path, scenario: str) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamped_dir = output_dir / "stamped"
    quarantine_dir = output_dir / "quarantine"
    manifest_path = output_dir / "forecast_audit_manifest.jsonl"
    latest_path = output_dir / "latest.json"
    payload = _prepare_payload(scenario)
    stamped_path = stamped_dir / f"{payload['audit_id']}.json"

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
            "status": "FAIL",
            "reason_code": reason,
            "latest_exists": latest_path.exists(),
            "quarantine_count": len(list(quarantine_dir.glob("*.json"))),
        }

    if scenario == "interrupted_promotion":
        return {
            "scenario": scenario,
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
        conflict_path = stamped_dir / f"{payload['audit_id']}_conflict.json"
        ok, reason = _write_and_register(conflict_payload, conflict_path)
        if ok:
            return {
                "scenario": scenario,
                "status": "FAIL",
                "reason_code": "EXPECTED_DUPLICATE_CONFLICT",
                "latest_exists": latest_path.exists(),
                "quarantine_count": len(list(quarantine_dir.glob("*.json"))),
            }

    atomic_write_json(latest_path, payload)
    return {
        "scenario": scenario,
        "status": "PASS",
        "reason_code": "READY",
        "latest_exists": latest_path.exists(),
        "quarantine_count": len(list(quarantine_dir.glob("*.json"))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="happy_path", choices=sorted(SCENARIOS))
    parser.add_argument("--output-dir", type=Path, default=Path("logs") / "evidence_replay")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = run_replay(output_dir=args.output_dir, scenario=args.scenario)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"scenario={summary['scenario']} status={summary['status']} reason={summary['reason_code']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
