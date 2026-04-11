#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evidence_io import atomic_write_json, atomic_write_jsonl, build_manifest_entry, quarantine_file

DEFAULT_PRODUCTION_AUDIT_DIR = ROOT / "logs" / "forecast_audits" / "production"
DEFAULT_PRODUCTION_EVAL_AUDIT_DIR = ROOT / "logs" / "forecast_audits" / "production_eval"
DEFAULT_PRODUCTION_MANIFEST_PATH = DEFAULT_PRODUCTION_AUDIT_DIR / "forecast_audit_manifest.jsonl"
DEFAULT_PRODUCTION_EVAL_MANIFEST_PATH = (
    DEFAULT_PRODUCTION_EVAL_AUDIT_DIR / "forecast_audit_manifest.jsonl"
)


def _parse_utc_datetime(raw: Any) -> Optional[datetime]:
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _audit_event_type(payload: Dict[str, Any]) -> str:
    signal_context = payload.get("signal_context") if isinstance(payload.get("signal_context"), dict) else {}
    raw = payload.get("event_type") or signal_context.get("event_type") or ""
    return str(raw).strip().upper()


def _audit_evidence_context(payload: Dict[str, Any]) -> Optional[str]:
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    raw = payload.get("evidence_context")
    if raw in (None, ""):
        raw = dataset.get("evidence_context")
    text = str(raw or "").strip().upper()
    return text or None


def _has_trade_metadata(payload: Dict[str, Any]) -> bool:
    signal_context = payload.get("signal_context") if isinstance(payload.get("signal_context"), dict) else {}
    execution_decision = (
        payload.get("execution_decision") if isinstance(payload.get("execution_decision"), dict) else {}
    )
    for raw in (
        payload.get("run_id"),
        signal_context.get("run_id"),
        signal_context.get("entry_ts"),
        signal_context.get("ts_signal_id"),
        signal_context.get("expected_close_ts"),
        execution_decision.get("signal_executed"),
    ):
        if raw not in (None, "", [], {}):
            return True
    return False


def _has_rmse_only_markers(payload: Dict[str, Any]) -> bool:
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    if isinstance(artifacts.get("evaluation_metrics"), dict) and artifacts.get("evaluation_metrics"):
        return True
    if isinstance(payload.get("benchmark_summary"), dict) and payload.get("benchmark_summary"):
        return True
    if isinstance(payload.get("model_benchmarks"), list) and payload.get("model_benchmarks"):
        return True
    return False


def classify_rmse_only_relocation(path: Path, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    event_type = _audit_event_type(payload)
    evidence_context = _audit_evidence_context(payload)
    if _has_trade_metadata(payload) or not _has_rmse_only_markers(payload):
        return None
    if event_type not in ("", "FORECAST_AUDIT") and evidence_context != "RMSE_ONLY":
        return None
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    return {
        "file": path.name,
        "reason": (
            "EXPLICIT_RMSE_ONLY_PRODUCTION_ARTIFACT"
            if evidence_context == "RMSE_ONLY"
            else "LEGACY_RMSE_ONLY_PRODUCTION_ARTIFACT"
        ),
        "event_type": event_type or None,
        "evidence_context": evidence_context,
        "ticker": dataset.get("ticker"),
        "dataset_end": dataset.get("end"),
        "forecast_horizon": dataset.get("forecast_horizon"),
    }


def _stamp_rmse_only_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    stamped = copy.deepcopy(payload) if isinstance(payload, dict) else {}
    dataset = stamped.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
        stamped["dataset"] = dataset
    stamped["event_type"] = "FORECAST_AUDIT"
    stamped["evidence_context"] = "RMSE_ONLY"
    dataset["evidence_context"] = "RMSE_ONLY"
    return stamped


def _relocate_rmse_only_artifact(
    *,
    path: Path,
    eval_audit_dir: Path,
    payload: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    eval_audit_dir.mkdir(parents=True, exist_ok=True)
    target = eval_audit_dir / path.name
    overwritten = target.exists()
    atomic_write_json(target, _stamp_rmse_only_payload(payload))
    path.unlink()
    return {
        "source_path": str(path),
        "target_path": str(target),
        "reason": reason,
        "overwrote_existing": overwritten,
    }


def classify_audit(
    *,
    path: Path,
    payload: Dict[str, Any],
    max_positive_gap_days: float,
    max_negative_gap_days: float,
    require_missing_expected_close_source: bool,
) -> Dict[str, Any]:
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    signal_context = (
        payload.get("signal_context") if isinstance(payload.get("signal_context"), dict) else {}
    )
    event_type = str(
        payload.get("event_type")
        or signal_context.get("event_type")
        or "TRADE_FORECAST_AUDIT"
    ).strip().upper()
    context_type = str(signal_context.get("context_type") or "TRADE").strip().upper() or "TRADE"
    dataset_end = _parse_utc_datetime(dataset.get("end"))
    entry_ts = _parse_utc_datetime(signal_context.get("entry_ts"))
    expected_close_source = str(signal_context.get("expected_close_source") or "").strip() or None
    gap_days = None
    if dataset_end is not None and entry_ts is not None:
        gap_days = (entry_ts - dataset_end).total_seconds() / 86400.0

    reason_codes: List[str] = []
    if context_type == "TRADE" and event_type == "TRADE_FORECAST_AUDIT" and gap_days is not None:
        if gap_days > max_positive_gap_days:
            reason_codes.append("ENTRY_AFTER_DATASET_END_EXCESSIVE")
        if gap_days < (-1.0 * max_negative_gap_days):
            reason_codes.append("ENTRY_BEFORE_DATASET_END")
    if require_missing_expected_close_source and reason_codes and not expected_close_source:
        reason_codes.append("MISSING_EXPECTED_CLOSE_SOURCE")

    suspect = bool(reason_codes) and (
        not require_missing_expected_close_source or not expected_close_source
    )
    return {
        "file": path.name,
        "suspect": suspect,
        "reason_codes": reason_codes,
        "gap_days": gap_days,
        "ticker": dataset.get("ticker"),
        "dataset_end": dataset.get("end"),
        "entry_ts": signal_context.get("entry_ts"),
        "expected_close_source": expected_close_source,
    }


def _rebuild_manifest(audit_dir: Path, manifest_path: Path) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for audit_path in sorted(audit_dir.glob("forecast_audit_*.json")):
        entry = build_manifest_entry(
            audit_path,
            source="scripts.sanitize_production_forecast_audits.rebuild_manifest",
        )
        if entry:
            records.append(entry)
    atomic_write_jsonl(manifest_path, records)
    return {
        "manifest_path": str(manifest_path),
        "record_count": len(records),
    }


def sanitize_production_forecast_audits(
    *,
    audit_dir: Path,
    eval_audit_dir: Path,
    quarantine_dir: Path,
    manifest_path: Path,
    eval_manifest_path: Path,
    max_positive_gap_days: float = 7.0,
    max_negative_gap_days: float = 1.0,
    require_missing_expected_close_source: bool = True,
    apply: bool = False,
) -> Dict[str, Any]:
    scanned_count = 0
    rows: List[Dict[str, Any]] = []
    suspects: List[Dict[str, Any]] = []
    rmse_only_candidates: List[Dict[str, Any]] = []
    for path in sorted(audit_dir.glob("forecast_audit_*.json")):
        payload = _load_json(path)
        if not payload:
            continue
        scanned_count += 1
        rmse_only_row = classify_rmse_only_relocation(path, payload)
        if rmse_only_row:
            rmse_only_candidates.append(rmse_only_row)
            continue
        row = classify_audit(
            path=path,
            payload=payload,
            max_positive_gap_days=max_positive_gap_days,
            max_negative_gap_days=max_negative_gap_days,
            require_missing_expected_close_source=require_missing_expected_close_source,
        )
        rows.append(row)
        if row["suspect"]:
            suspects.append(row)

    quarantine_results: List[Dict[str, Any]] = []
    relocation_results: List[Dict[str, Any]] = []
    manifest_result: Optional[Dict[str, Any]] = None
    eval_manifest_result: Optional[Dict[str, Any]] = None
    if apply and rmse_only_candidates:
        for row in rmse_only_candidates:
            source_path = audit_dir / row["file"]
            payload = _load_json(source_path)
            if not payload:
                continue
            relocation_results.append(
                _relocate_rmse_only_artifact(
                    path=source_path,
                    eval_audit_dir=eval_audit_dir,
                    payload=payload,
                    reason=str(row["reason"] or "RMSE_ONLY_PRODUCTION_ARTIFACT"),
                )
            )
    if apply and suspects:
        for row in suspects:
            quarantine_results.append(
                quarantine_file(
                    audit_dir / row["file"],
                    quarantine_dir=quarantine_dir,
                    reason="SUSPECT_TIME_INDEXED_PRODUCTION_AUDIT",
                    metadata={
                        "reason_codes": row["reason_codes"],
                        "gap_days": row["gap_days"],
                        "ticker": row["ticker"],
                        "dataset_end": row["dataset_end"],
                        "entry_ts": row["entry_ts"],
                        "expected_close_source": row["expected_close_source"],
                    },
                )
            )
    if apply and (suspects or relocation_results):
        manifest_result = _rebuild_manifest(audit_dir, manifest_path)
        eval_manifest_result = _rebuild_manifest(eval_audit_dir, eval_manifest_path)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "audit_dir": str(audit_dir),
        "eval_audit_dir": str(eval_audit_dir),
        "quarantine_dir": str(quarantine_dir),
        "manifest_path": str(manifest_path),
        "eval_manifest_path": str(eval_manifest_path),
        "apply": bool(apply),
        "thresholds": {
            "max_positive_gap_days": float(max_positive_gap_days),
            "max_negative_gap_days": float(max_negative_gap_days),
            "require_missing_expected_close_source": bool(require_missing_expected_close_source),
        },
        "totals": {
            "audits_scanned": scanned_count,
            "suspects": len(suspects),
            "rmse_only_candidates": len(rmse_only_candidates),
            "quarantined": len(quarantine_results),
            "relocated": len(relocation_results),
        },
        "reason_code_counts": Counter(
            code for row in suspects for code in (row.get("reason_codes") or [])
        ),
        "rmse_only_reason_counts": Counter(str(row.get("reason") or "") for row in rmse_only_candidates),
        "suspect_examples": suspects[:25],
        "rmse_only_examples": rmse_only_candidates[:25],
        "quarantine_results": quarantine_results[:25],
        "relocation_results": relocation_results[:25],
        "manifest_rebuild": manifest_result,
        "eval_manifest_rebuild": eval_manifest_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=DEFAULT_PRODUCTION_AUDIT_DIR,
    )
    parser.add_argument(
        "--eval-audit-dir",
        type=Path,
        default=DEFAULT_PRODUCTION_EVAL_AUDIT_DIR,
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=DEFAULT_PRODUCTION_AUDIT_DIR / "quarantine",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_PRODUCTION_MANIFEST_PATH,
    )
    parser.add_argument(
        "--eval-manifest-path",
        type=Path,
        default=DEFAULT_PRODUCTION_EVAL_MANIFEST_PATH,
    )
    parser.add_argument("--max-positive-gap-days", type=float, default=7.0)
    parser.add_argument("--max-negative-gap-days", type=float, default=1.0)
    parser.add_argument("--allow-missing-source", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = sanitize_production_forecast_audits(
        audit_dir=args.audit_dir,
        eval_audit_dir=args.eval_audit_dir,
        quarantine_dir=args.quarantine_dir,
        manifest_path=args.manifest_path,
        eval_manifest_path=args.eval_manifest_path,
        max_positive_gap_days=args.max_positive_gap_days,
        max_negative_gap_days=args.max_negative_gap_days,
        require_missing_expected_close_source=not bool(args.allow_missing_source),
        apply=bool(args.apply),
    )
    output_path = ROOT / "logs" / "audit_gate" / "production_audit_sanitization_latest.json"
    atomic_write_json(output_path, summary)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        totals = summary["totals"]
        print(
            "production_audit_sanitization "
            f"scanned={totals['audits_scanned']} suspects={totals['suspects']} "
            f"rmse_only_candidates={totals['rmse_only_candidates']} "
            f"quarantined={totals['quarantined']} relocated={totals['relocated']} "
            f"output={output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
