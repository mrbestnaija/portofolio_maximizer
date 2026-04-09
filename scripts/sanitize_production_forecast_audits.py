#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    quarantine_dir: Path,
    manifest_path: Path,
    max_positive_gap_days: float = 7.0,
    max_negative_gap_days: float = 1.0,
    require_missing_expected_close_source: bool = True,
    apply: bool = False,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    suspects: List[Dict[str, Any]] = []
    for path in sorted(audit_dir.glob("forecast_audit_*.json")):
        payload = _load_json(path)
        if not payload:
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
    manifest_result: Optional[Dict[str, Any]] = None
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
        manifest_result = _rebuild_manifest(audit_dir, manifest_path)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "audit_dir": str(audit_dir),
        "quarantine_dir": str(quarantine_dir),
        "manifest_path": str(manifest_path),
        "apply": bool(apply),
        "thresholds": {
            "max_positive_gap_days": float(max_positive_gap_days),
            "max_negative_gap_days": float(max_negative_gap_days),
            "require_missing_expected_close_source": bool(require_missing_expected_close_source),
        },
        "totals": {
            "audits_scanned": len(rows),
            "suspects": len(suspects),
            "quarantined": len(quarantine_results),
        },
        "reason_code_counts": Counter(
            code for row in suspects for code in (row.get("reason_codes") or [])
        ),
        "suspect_examples": suspects[:25],
        "quarantine_results": quarantine_results[:25],
        "manifest_rebuild": manifest_result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=ROOT / "logs" / "forecast_audits" / "production",
    )
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=ROOT / "logs" / "forecast_audits" / "production" / "quarantine",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT / "logs" / "forecast_audits" / "production" / "forecast_audit_manifest.jsonl",
    )
    parser.add_argument("--max-positive-gap-days", type=float, default=7.0)
    parser.add_argument("--max-negative-gap-days", type=float, default=1.0)
    parser.add_argument("--allow-missing-source", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = sanitize_production_forecast_audits(
        audit_dir=args.audit_dir,
        quarantine_dir=args.quarantine_dir,
        manifest_path=args.manifest_path,
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
            f"quarantined={totals['quarantined']} output={output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
