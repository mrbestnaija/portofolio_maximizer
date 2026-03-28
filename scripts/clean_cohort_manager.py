#!/usr/bin/env python3
"""
Freeze and operate a clean evidence cohort without mutating legacy production evidence.

This helper is intentionally narrow:
- freeze an immutable cohort identity
- emit a PowerShell activation script for the cohort
- run a cohort-scoped proof loop with explicit audit/output paths
- summarize producer-native versus legacy-derived provenance in the cohort
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from utils.evidence_io import atomic_write_json, load_json_file


CONTRACT_VERSION = 2
DEFAULT_COHORT_ROOT = ROOT_PATH / "logs" / "forecast_audits" / "cohorts"
DEFAULT_REPLAY_ROOT = ROOT_PATH / "logs" / "evidence_replay"
OUTCOME_ELIGIBILITY_BUFFER = timedelta(minutes=5)


def _stable_fingerprint(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fingerprint_files(paths: list[Path]) -> Optional[str]:
    pieces: list[str] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            rel = path.relative_to(ROOT_PATH)
        except ValueError:
            rel = path
        pieces.append(str(rel))
        pieces.append(hashlib.sha256(path.read_bytes()).hexdigest())
    if not pieces:
        return None
    return hashlib.sha256("|".join(pieces).encode("utf-8")).hexdigest()


def _default_build_fingerprint() -> str:
    for name in ("PMX_BUILD_FINGERPRINT", "GIT_COMMIT"):
        candidate = str(os.environ.get(name) or "").strip()
        if candidate:
            return candidate
    return "workspace_uncommitted"


def _cohort_paths(cohort_id: str, cohort_root: Path) -> Dict[str, Path]:
    root = cohort_root / cohort_id
    return {
        "root": root,
        "production_dir": root / "production",
        "research_dir": root / "research",
        "identity_path": root / "cohort_identity.json",
        "activation_path": root / "activate_clean_cohort.ps1",
        "proof_output_path": root / "proof_loop_latest.json",
        "gate_output_path": root / "production_gate_latest.json",
    }


def build_cohort_identity(
    *,
    cohort_id: str,
    build_fingerprint: Optional[str] = None,
    routing_mode: str = "explicit_env",
    contract_version: int = CONTRACT_VERSION,
    config_paths: Optional[list[Path]] = None,
) -> Dict[str, Any]:
    if config_paths is None:
        config_paths = [
            ROOT_PATH / "config" / "forecasting_config.yml",
            ROOT_PATH / "config" / "signal_routing_config.yml",
        ]
    identity = {
        "cohort_id": cohort_id,
        "build_fingerprint": str(build_fingerprint or _default_build_fingerprint()).strip(),
        "contract_version": int(contract_version),
        "routing_mode": routing_mode,
        "strategy_config_fingerprint": _fingerprint_files(config_paths),
    }
    identity["contract_fingerprint"] = _stable_fingerprint(identity)
    return identity


def _parse_utc_datetime(raw: Any) -> Optional[datetime]:
    if raw in (None, ""):
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_closed_trade_counts(db_path: Optional[Path]) -> Dict[str, int]:
    if db_path is None or not db_path.exists():
        return {}
    counts: Dict[str, int] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            """
            SELECT ts_signal_id, COUNT(*) AS n
            FROM production_closed_trades
            WHERE ts_signal_id IS NOT NULL AND TRIM(ts_signal_id) <> ''
            GROUP BY ts_signal_id
            """
        )
        for ts_signal_id, count in cur.fetchall():
            key = str(ts_signal_id or "").strip()
            if key:
                counts[key] = int(count or 0)
    finally:
        conn.close()
    return counts


def _increment_counter(counter: Dict[str, int], key: Optional[str]) -> None:
    token = str(key or "").strip()
    if not token:
        return
    counter[token] = int(counter.get(token, 0)) + 1


def _manifest_source_counts(audit_dir: Path) -> Dict[str, int]:
    manifest_path = audit_dir / "forecast_audit_manifest.jsonl"
    counts = {
        "manifest_records": 0,
        "manifest_producer_native_records": 0,
        "manifest_legacy_derived_records": 0,
        "manifest_unknown_records": 0,
    }
    if not manifest_path.exists():
        return counts
    for raw_line in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        counts["manifest_records"] += 1
        source_classification = str(entry.get("evidence_source_classification") or "").strip().lower()
        if source_classification == "producer-native":
            counts["manifest_producer_native_records"] += 1
        elif source_classification == "legacy-derived":
            counts["manifest_legacy_derived_records"] += 1
        else:
            counts["manifest_unknown_records"] += 1
    return counts


def summarize_evidence_provenance(audit_dir: Path) -> Dict[str, Any]:
    summary = {
        "audit_dir": str(audit_dir),
        "total_records": 0,
        "producer_native_records": 0,
        "legacy_derived_records": 0,
        "unknown_records": 0,
        "eligible_records": 0,
        "eligible_producer_native_records": 0,
        "explicit_failed_records": 0,
    }
    summary.update(_manifest_source_counts(audit_dir))
    if not audit_dir.exists():
        return summary

    for path in sorted(audit_dir.glob("forecast_audit_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        summary["total_records"] += 1
        source_classification = str(
            payload.get("evidence_source_classification")
            or ((payload.get("semantic_admission") or {}).get("source_classification"))
            or ""
        ).strip().lower()
        if source_classification == "producer-native":
            summary["producer_native_records"] += 1
        elif source_classification == "legacy-derived":
            summary["legacy_derived_records"] += 1
        else:
            summary["unknown_records"] += 1

        semantic = payload.get("semantic_admission")
        if isinstance(semantic, dict) and bool(semantic.get("gate_eligible")):
            summary["eligible_records"] += 1
            if source_classification == "producer-native":
                summary["eligible_producer_native_records"] += 1
    summary["explicit_failed_records"] = len(list(audit_dir.glob("audit_failure_*.json")))
    return summary


def summarize_cohort_funnel(
    audit_dir: Path,
    *,
    db_path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    closed_trade_counts = _load_closed_trade_counts(db_path)
    summary: Dict[str, Any] = {
        "audit_dir": str(audit_dir),
        "db_path": str(db_path) if db_path is not None else None,
        "processed_records": 0,
        "accepted_records": 0,
        "eligible_records": 0,
        "accepted_noneligible_records": 0,
        "quarantined_records": 0,
        "explicit_failed_records": 0,
        "producer_native_records": 0,
        "legacy_derived_records": 0,
        "unknown_records": 0,
        "matched_records": 0,
        "outcome_missing_records": 0,
        "not_due_records": 0,
        "invalid_context_records": 0,
        "non_trade_context_records": 0,
        "ambiguous_match_records": 0,
        "missing_execution_metadata_records": 0,
        "outcome_join_loaded": bool(closed_trade_counts),
        "closed_trade_signal_count": len(closed_trade_counts),
        "bucket_exclusivity_ok": True,
        "reason_code_counts": {},
        "missing_execution_metadata_field_counts": {},
        "producer_native_funnel": {
            "accepted": 0,
            "eligible": 0,
            "matched": 0,
            "invalid": 0,
            "not_due": 0,
        },
    }
    summary.update(_manifest_source_counts(audit_dir))
    if not audit_dir.exists():
        return summary

    audit_paths = sorted(audit_dir.glob("forecast_audit_*.json"))
    quarantine_dir = audit_dir / "quarantine"
    quarantine_paths = [path for path in quarantine_dir.glob("*.json") if not path.name.endswith(".meta.json")]
    explicit_failed_paths = sorted(audit_dir.glob("audit_failure_*.json"))
    summary["quarantined_records"] = len(quarantine_paths)
    summary["explicit_failed_records"] = len(explicit_failed_paths)

    for path in audit_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        summary["processed_records"] += 1
        summary["accepted_records"] += 1
        semantic = payload.get("semantic_admission") if isinstance(payload.get("semantic_admission"), dict) else {}
        signal_context = payload.get("signal_context") if isinstance(payload.get("signal_context"), dict) else {}
        lineage = payload.get("lineage_v2") if isinstance(payload.get("lineage_v2"), dict) else {}
        source_classification = str(
            payload.get("evidence_source_classification")
            or semantic.get("source_classification")
            or ""
        ).strip().lower()
        if source_classification == "producer-native":
            summary["producer_native_records"] += 1
            summary["producer_native_funnel"]["accepted"] += 1
        elif source_classification == "legacy-derived":
            summary["legacy_derived_records"] += 1
        else:
            summary["unknown_records"] += 1

        gate_eligible = bool(semantic.get("gate_eligible"))
        gate_bucket = str(semantic.get("gate_bucket") or "").strip().upper()
        quarantined = bool(semantic.get("quarantined"))
        if gate_eligible:
            summary["eligible_records"] += 1
            if source_classification == "producer-native":
                summary["producer_native_funnel"]["eligible"] += 1
        elif gate_bucket == "ACCEPTED_NONELIGIBLE" or not quarantined:
            summary["accepted_noneligible_records"] += 1

        bucket_count = int(gate_eligible) + int(gate_bucket == "ACCEPTED_NONELIGIBLE") + int(quarantined or gate_bucket == "QUARANTINED")
        if bucket_count != 1:
            summary["bucket_exclusivity_ok"] = False

        reason_codes = semantic.get("reason_codes")
        if isinstance(reason_codes, list):
            for code in reason_codes:
                _increment_counter(summary["reason_code_counts"], str(code or "").strip().upper())
        else:
            _increment_counter(summary["reason_code_counts"], str(semantic.get("reason_code") or "").strip().upper())

        if bool(semantic.get("missing_execution_metadata")):
            summary["missing_execution_metadata_records"] += 1
            missing_fields = semantic.get("missing_execution_metadata_fields")
            if isinstance(missing_fields, list):
                for field in missing_fields:
                    _increment_counter(summary["missing_execution_metadata_field_counts"], str(field or "").strip())

        if not gate_eligible:
            continue

        context_type = str(signal_context.get("context_type") or lineage.get("context_type") or "").strip().upper()
        if context_type != "TRADE":
            summary["non_trade_context_records"] += 1
            if source_classification == "producer-native":
                summary["producer_native_funnel"]["invalid"] += 1
            continue

        ts_signal_id = str(signal_context.get("ts_signal_id") or lineage.get("ts_signal_id") or "").strip()
        run_id = str(signal_context.get("run_id") or lineage.get("run_id") or "").strip()
        entry_ts = _parse_utc_datetime(signal_context.get("entry_ts") or lineage.get("entry_ts"))
        expected_close_ts = _parse_utc_datetime(
            signal_context.get("expected_close_ts") or lineage.get("expected_close_ts")
        )
        if not run_id or not ts_signal_id or entry_ts is None or expected_close_ts is None or expected_close_ts < entry_ts:
            summary["invalid_context_records"] += 1
            if source_classification == "producer-native":
                summary["producer_native_funnel"]["invalid"] += 1
            continue

        if (expected_close_ts + OUTCOME_ELIGIBILITY_BUFFER) > current_time:
            summary["not_due_records"] += 1
            if source_classification == "producer-native":
                summary["producer_native_funnel"]["not_due"] += 1
            continue

        match_count = int(closed_trade_counts.get(ts_signal_id, 0))
        if match_count == 1:
            summary["matched_records"] += 1
            if source_classification == "producer-native":
                summary["producer_native_funnel"]["matched"] += 1
        elif match_count == 0:
            summary["outcome_missing_records"] += 1
        else:
            summary["invalid_context_records"] += 1
            summary["ambiguous_match_records"] += 1
            if source_classification == "producer-native":
                summary["producer_native_funnel"]["invalid"] += 1

    summary["processed_records"] += summary["quarantined_records"] + summary["explicit_failed_records"]
    summary["producer_share"] = (
        summary["producer_native_records"] / summary["accepted_records"]
        if summary["accepted_records"]
        else 0.0
    )
    summary["eligible_to_matched_ratio"] = (
        summary["matched_records"] / summary["eligible_records"]
        if summary["eligible_records"]
        else 0.0
    )
    producer_native_accepted = int(summary["producer_native_funnel"]["accepted"])
    summary["producer_native_funnel"]["producer_share"] = (
        producer_native_accepted / summary["accepted_records"]
        if summary["accepted_records"]
        else 0.0
    )
    return summary


def freeze_clean_cohort(
    *,
    cohort_id: str,
    cohort_root: Path = DEFAULT_COHORT_ROOT,
    build_fingerprint: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    paths = _cohort_paths(cohort_id, cohort_root)
    identity = build_cohort_identity(cohort_id=cohort_id, build_fingerprint=build_fingerprint)

    existing = load_json_file(paths["identity_path"]) if paths["identity_path"].exists() else None
    if isinstance(existing, dict):
        existing_identity = existing.get("cohort_identity")
        if isinstance(existing_identity, dict):
            if build_fingerprint is None and not force:
                identity = existing_identity
            elif existing_identity.get("contract_fingerprint") != identity.get("contract_fingerprint"):
                if not force:
                    raise ValueError(
                        "existing cohort fingerprint differs; refusing to mutate frozen cohort identity"
                    )
            else:
                identity = existing_identity

    for key in ("root", "production_dir", "research_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_identity": identity,
        "paths": {
            "root": str(paths["root"]),
            "production_dir": str(paths["production_dir"]),
            "research_dir": str(paths["research_dir"]),
        },
        "env": {
            "PMX_EVIDENCE_COHORT_ID": cohort_id,
            "PMX_BUILD_FINGERPRINT": identity.get("build_fingerprint"),
            "TS_FORECAST_AUDIT_DIR": str(paths["production_dir"]),
        },
    }
    atomic_write_json(paths["identity_path"], payload)

    activation_script = "\n".join(
        [
            f"$env:PMX_EVIDENCE_COHORT_ID='{cohort_id}'",
            f"$env:PMX_BUILD_FINGERPRINT='{identity.get('build_fingerprint')}'",
            f"$env:TS_FORECAST_AUDIT_DIR='{paths['production_dir']}'",
            "",
        ]
    )
    paths["activation_path"].write_text(activation_script, encoding="utf-8")
    return payload


def _run_command(
    cmd: list[str],
    *,
    env: Dict[str, str],
    cwd: Path = ROOT_PATH,
) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "cmd": cmd,
        "exit_code": int(proc.returncode),
        "passed": int(proc.returncode) == 0,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
    }


def run_clean_cohort_proof_loop(
    *,
    cohort_id: str,
    cohort_root: Path = DEFAULT_COHORT_ROOT,
    replay_root: Path = DEFAULT_REPLAY_ROOT,
    include_global_gates: bool = False,
    build_fingerprint: Optional[str] = None,
    replay_scenario: str = "happy_path",
) -> Dict[str, Any]:
    frozen = freeze_clean_cohort(
        cohort_id=cohort_id,
        cohort_root=cohort_root,
        build_fingerprint=build_fingerprint,
    )
    paths = _cohort_paths(cohort_id, cohort_root)
    identity = frozen["cohort_identity"]

    env = os.environ.copy()
    env.update(
        {
            "PMX_EVIDENCE_COHORT_ID": cohort_id,
            "PMX_BUILD_FINGERPRINT": str(identity.get("build_fingerprint") or ""),
            "TS_FORECAST_AUDIT_DIR": str(paths["production_dir"]),
        }
    )

    replay_dir = paths["production_dir"]
    replay_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    results: Dict[str, Dict[str, Any]] = {}
    results["replay_trade_evidence_chain"] = _run_command(
        [
            python,
            "scripts/replay_trade_evidence_chain.py",
            "--scenario",
            replay_scenario,
            "--output-dir",
            str(replay_dir),
            "--cohort-id",
            cohort_id,
            "--json",
        ],
        env=env,
    )
    results["pnl_integrity_enforcer"] = _run_command(
        [python, "-m", "integrity.pnl_integrity_enforcer"],
        env=env,
    )
    results["production_audit_gate_clean_cohort"] = _run_command(
        [
            python,
            "scripts/production_audit_gate.py",
            "--audit-dir",
            str(paths["production_dir"]),
            "--output",
            str(paths["gate_output_path"]),
            "--unattended-profile",
        ],
        env=env,
    )
    if include_global_gates:
        results["run_all_gates_global"] = _run_command(
            [python, "scripts/run_all_gates.py", "--json"],
            env=env,
        )

    provenance_summary = summarize_evidence_provenance(paths["production_dir"])
    funnel_summary = summarize_cohort_funnel(
        paths["production_dir"],
        db_path=paths["production_dir"] / "proof.sqlite3",
    )
    overall_passed = all(bool(result.get("passed")) for result in results.values())
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_identity": identity,
        "paths": {
            "cohort_root": str(paths["root"]),
            "production_dir": str(paths["production_dir"]),
            "production_gate_output": str(paths["gate_output_path"]),
            "proof_output": str(paths["proof_output_path"]),
            "replay_dir": str(replay_dir),
        },
        "include_global_gates": bool(include_global_gates),
        "replay_scenario": replay_scenario,
        "overall_passed": overall_passed,
        "provenance_summary": provenance_summary,
        "funnel_summary": funnel_summary,
        "steps": results,
    }
    atomic_write_json(paths["proof_output_path"], summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze", help="Freeze a clean cohort identity.")
    freeze_parser.add_argument("--cohort-id", required=True)
    freeze_parser.add_argument("--cohort-root", default=str(DEFAULT_COHORT_ROOT))
    freeze_parser.add_argument("--build-fingerprint", default=None)
    freeze_parser.add_argument("--force", action="store_true")
    freeze_parser.add_argument("--json", action="store_true", dest="emit_json")

    proof_parser = subparsers.add_parser("proof-loop", help="Run the clean cohort proof loop.")
    proof_parser.add_argument("--cohort-id", required=True)
    proof_parser.add_argument("--cohort-root", default=str(DEFAULT_COHORT_ROOT))
    proof_parser.add_argument("--replay-root", default=str(DEFAULT_REPLAY_ROOT))
    proof_parser.add_argument("--include-global-gates", action="store_true")
    proof_parser.add_argument("--build-fingerprint", default=None)
    proof_parser.add_argument("--replay-scenario", default="happy_path")
    proof_parser.add_argument("--json", action="store_true", dest="emit_json")

    funnel_parser = subparsers.add_parser("funnel", help="Summarize clean cohort funnel metrics.")
    funnel_parser.add_argument("--audit-dir", required=True)
    funnel_parser.add_argument("--db", default=None)
    funnel_parser.add_argument("--json", action="store_true", dest="emit_json")

    args = parser.parse_args()

    if args.command == "freeze":
        payload = freeze_clean_cohort(
            cohort_id=args.cohort_id,
            cohort_root=Path(args.cohort_root),
            build_fingerprint=args.build_fingerprint,
            force=bool(args.force),
        )
        if args.emit_json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Frozen cohort   : {args.cohort_id}")
            print(f"Identity path   : {payload['paths']['root']}")
            print(f"Production dir  : {payload['paths']['production_dir']}")
        return 0

    if args.command == "proof-loop":
        summary = run_clean_cohort_proof_loop(
            cohort_id=args.cohort_id,
            cohort_root=Path(args.cohort_root),
            replay_root=Path(args.replay_root),
            include_global_gates=bool(args.include_global_gates),
            build_fingerprint=args.build_fingerprint,
            replay_scenario=str(args.replay_scenario),
        )
        if args.emit_json:
            print(json.dumps(summary, indent=2))
        else:
            print(f"Clean cohort    : {args.cohort_id}")
            print(f"Overall passed  : {int(bool(summary['overall_passed']))}")
            print(f"Proof output    : {summary['paths']['proof_output']}")
        return 0 if bool(summary["overall_passed"]) else 1

    funnel_summary = summarize_cohort_funnel(
        Path(args.audit_dir),
        db_path=Path(args.db) if args.db else None,
    )
    if args.emit_json:
        print(json.dumps(funnel_summary, indent=2))
    else:
        print(f"Audit dir       : {funnel_summary['audit_dir']}")
        print(f"Accepted        : {funnel_summary['accepted_records']}")
        print(f"Eligible        : {funnel_summary['eligible_records']}")
        print(f"Matched         : {funnel_summary['matched_records']}")
        print(f"Producer share  : {funnel_summary['producer_share']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
