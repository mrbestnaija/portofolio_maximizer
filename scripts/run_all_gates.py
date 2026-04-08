#!/usr/bin/env python3
"""
run_all_gates.py - Master gate orchestrator.

Runs production gates in strict priority order and returns a JSON summary:

  1. ci_integrity_gate.py (PnL structural invariants)
  2. check_quant_validation_health.py (quant validation fail-rate health)
  3. production_audit_gate.py (forecast lift + profitability proof)
  4. institutional_unattended_gate.py (unattended-run hardening contracts)

Exit codes:
    0  all blocking gates passed
    1  one or more blocking gates failed
    2  invocation / subprocess error

Usage:
    python scripts/run_all_gates.py
    python scripts/run_all_gates.py --skip-forecast-gate
    python scripts/run_all_gates.py --skip-profitability-gate
    python scripts/run_all_gates.py --skip-institutional-gate
    python scripts/run_all_gates.py --strict
    python scripts/run_all_gates.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.production_gate_contract import (
        gate_semantics_status as _gate_semantics_status,
        legacy_phase3_ready as _legacy_phase3_ready,
        legacy_phase3_reason as _legacy_phase3_reason,
        phase3_genuine_ready as _phase3_genuine_ready,
        phase3_posture as _phase3_posture,
        phase3_strict_ready as _phase3_strict_ready,
        phase3_strict_reason as _phase3_strict_reason,
    )
except Exception:  # pragma: no cover - script execution path fallback
    from production_gate_contract import (  # type: ignore
        gate_semantics_status as _gate_semantics_status,
        legacy_phase3_ready as _legacy_phase3_ready,
        legacy_phase3_reason as _legacy_phase3_reason,
        phase3_genuine_ready as _phase3_genuine_ready,
        phase3_posture as _phase3_posture,
        phase3_strict_ready as _phase3_strict_ready,
        phase3_strict_reason as _phase3_strict_reason,
    )

# BYP-01 fix: cap how many optional gates can be skipped before we force overall_passed=False.
# With 3 optional gates, allowing only 1 skip means at least 2/3 must actually run.
MAX_SKIPPED_OPTIONAL_GATES = 1

REQUIRED_PRODUCTION_PAYLOAD_KEYS = {
    "pass_semantics_version",
    "lift_inconclusive_allowed",
    "proof_profitable_required",
    "warmup_expired",
    "phase3_ready",
    "phase3_reason",
    "phase3_strict_ready",
    "phase3_strict_reason",
    "posture",
    "lift_gate",
    "profitability_proof",
    "production_profitability_gate",
    "readiness",
}
REQUIRED_READINESS_KEYS = {
    "gates_pass",
    "linkage_pass",
    "evidence_hygiene_pass",
    "integrity_pass",
    "phase3_ready",
    "phase3_reason",
    "phase3_strict_ready",
    "phase3_strict_reason",
    "posture",
}
REQUIRED_GATE_BLOCK_KEYS = {
    "status",
    "pass",
    "strict_pass",
    "gate_semantics_status",
    "posture",
}

# Artifact written after every run so downstream tools (e.g. institutional gate) can
# verify gates ran recently without re-running them (BYP-03 fix).
GATE_STATUS_ARTIFACT = Path(__file__).resolve().parents[1] / "logs" / "gate_status_latest.json"
PRODUCTION_GATE_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "logs" / "audit_gate" / "production_gate_latest.json"
)

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], label: str) -> dict[str, Any]:
    """Run a subprocess and return a result dict."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )
        return {
            "label": label,
            "exit_code": int(proc.returncode),
            "passed": int(proc.returncode) == 0,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-2000:],
        }
    except Exception as exc:
        return {
            "label": label,
            "exit_code": -1,
            "passed": False,
            "stdout": "",
            "stderr": str(exc),
        }


def _validate_production_gate_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """Fail-closed schema guard to prevent mismatched wiring or threshold dodge."""
    if not isinstance(payload, dict) or not payload:
        return False, ["payload_missing"]

    warnings: list[str] = []
    for key in sorted(REQUIRED_PRODUCTION_PAYLOAD_KEYS):
        if key not in payload:
            warnings.append(f"missing:{key}")

    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    for key in sorted(REQUIRED_READINESS_KEYS):
        if key not in readiness:
            warnings.append(f"missing:readiness.{key}")

    gate_block = (
        payload.get("production_profitability_gate")
        if isinstance(payload.get("production_profitability_gate"), dict)
        else {}
    )
    for key in sorted(REQUIRED_GATE_BLOCK_KEYS):
        if key not in gate_block:
            warnings.append(f"missing:production_gate.{key}")

    return len(warnings) == 0, warnings


def _print_gate_result(result: dict[str, Any], *, emit_json: bool) -> None:
    if emit_json:
        return
    label = str(result.get("label"))
    exit_code = int(result.get("exit_code", -1))
    if bool(result.get("passed")):
        print(f"[PASS] {label}")
        return
    print(f"[FAIL] {label} (exit {exit_code})")
    stdout = str(result.get("stdout") or "").strip()
    if stdout:
        print(stdout)


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_summary(
    *,
    overall_pass: bool,
    skipped_optional: int,
    results: list[dict[str, Any]],
    production_gate_payload: dict[str, Any],
    stage: str,
    production_gate_schema_ok: bool,
    production_gate_schema_warnings: list[str],
    skipped_gate_labels: list[str],
) -> dict[str, Any]:
    readiness = (
        production_gate_payload.get("readiness", {})
        if isinstance(production_gate_payload.get("readiness"), dict)
        else {}
    )
    gate_payload_block = (
        production_gate_payload.get("production_profitability_gate", {})
        if isinstance(production_gate_payload.get("production_profitability_gate"), dict)
        else {}
    )
    strict_phase3_ready = _phase3_strict_ready(production_gate_payload)
    strict_phase3_reason = _phase3_strict_reason(production_gate_payload)
    phase3_posture = _phase3_posture(production_gate_payload)
    genuine_phase3_ready = _phase3_genuine_ready(production_gate_payload)
    legacy_phase3_ready = _legacy_phase3_ready(production_gate_payload)
    legacy_phase3_reason = _legacy_phase3_reason(production_gate_payload)
    return {
        "phase": "institutional_unattended_hardening",
        "status_stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_passed": bool(overall_pass),
        "skipped_optional_gates": skipped_optional,
        "max_skipped_optional_gates": MAX_SKIPPED_OPTIONAL_GATES,
        "gates": results,
        "pass_semantics_version": production_gate_payload.get("pass_semantics_version"),
        "lift_inconclusive_allowed": production_gate_payload.get("lift_inconclusive_allowed"),
        "proof_profitable_required": production_gate_payload.get("proof_profitable_required"),
        "warmup_expired": production_gate_payload.get("warmup_expired"),
        "phase3_posture": phase3_posture,
        "gate_semantics_status": _gate_semantics_status(production_gate_payload)
        or gate_payload_block.get("gate_semantics_status"),
        "phase3_ready": bool(genuine_phase3_ready),
        "phase3_reason": strict_phase3_reason,
        "phase3_strict_ready": bool(strict_phase3_ready),
        "phase3_strict_reason": strict_phase3_reason,
        "phase3_legacy_ready": bool(legacy_phase3_ready),
        "phase3_legacy_reason": legacy_phase3_reason,
        "production_gate_schema_ok": production_gate_schema_ok,
        "production_gate_schema_warnings": production_gate_schema_warnings,
        "readiness_components": {
            "gates_pass": readiness.get("gates_pass"),
            "linkage_pass": readiness.get("linkage_pass"),
            "evidence_hygiene_pass": readiness.get("evidence_hygiene_pass"),
            "integrity_pass": readiness.get("integrity_pass"),
        },
        "skipped_gate_labels": skipped_gate_labels,
    }


def _write_gate_status_artifact(summary: dict[str, Any]) -> None:
    try:
        GATE_STATUS_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        GATE_STATUS_ARTIFACT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:
        print(
            f"[WARN] Failed to write gate status artifact: {GATE_STATUS_ARTIFACT} ({exc})",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-forecast-gate",
        action="store_true",
        help="Skip check_quant_validation_health.py gate.",
    )
    parser.add_argument(
        "--skip-profitability-gate",
        action="store_true",
        help="Skip production_audit_gate.py gate.",
    )
    parser.add_argument(
        "--skip-institutional-gate",
        action="store_true",
        help="Skip institutional_unattended_gate.py gate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Pass --strict to ci_integrity_gate (fail on MEDIUM violations too).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="emit_json",
        help="Emit JSON summary to stdout.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Override DB path passed to gates that accept --db.",
    )
    args = parser.parse_args()

    python = sys.executable
    results: list[dict[str, Any]] = []
    overall_pass = True

    # Gate 1: PnL integrity
    cmd1 = [python, "scripts/ci_integrity_gate.py"]
    if args.strict:
        cmd1.append("--strict")
    if args.db:
        cmd1 += ["--db", args.db]
    r1 = _run(cmd1, "ci_integrity_gate")
    results.append(r1)
    overall_pass = overall_pass and bool(r1.get("passed"))
    _print_gate_result(r1, emit_json=args.emit_json)

    # Gate 2: quant validation health
    if not args.skip_forecast_gate:
        r2 = _run([python, "scripts/check_quant_validation_health.py"], "check_quant_validation_health")
        results.append(r2)
        overall_pass = overall_pass and bool(r2.get("passed"))
        _print_gate_result(r2, emit_json=args.emit_json)
    else:
        skipped = {"label": "check_quant_validation_health", "skipped": True, "passed": True}
        results.append(skipped)
        if not args.emit_json:
            print("[SKIP] check_quant_validation_health (--skip-forecast-gate)")

    # Gate 3: production audit gate
    if not args.skip_profitability_gate:
        cmd3 = [
            python,
            "scripts/production_audit_gate.py",
            "--reconcile",  # dry-run: surface unlinked closes without applying; overnight job applies
            "--unattended-profile",
            "--output-json",
            str(PRODUCTION_GATE_ARTIFACT),
        ]
        if args.db:
            cmd3 += ["--db", args.db]
        r3 = _run(cmd3, "production_audit_gate")
        results.append(r3)
        overall_pass = overall_pass and bool(r3.get("passed"))
        _print_gate_result(r3, emit_json=args.emit_json)
    else:
        skipped = {"label": "production_audit_gate", "skipped": True, "passed": True}
        results.append(skipped)
        if not args.emit_json:
            print("[SKIP] production_audit_gate (--skip-profitability-gate)")

    production_gate_ran = any(
        str(r.get("label")) == "production_audit_gate" and not bool(r.get("skipped"))
        for r in results
    )
    production_gate_payload = _load_json_file(PRODUCTION_GATE_ARTIFACT) if production_gate_ran else {}
    schema_ok, schema_warnings = (True, [])
    if production_gate_ran:
        schema_ok, schema_warnings = _validate_production_gate_payload(production_gate_payload)
        schema_result = {
            "label": "production_gate_schema",
            "exit_code": 0 if schema_ok else 1,
            "passed": bool(schema_ok),
            "warnings": schema_warnings,
        }
        results.append(schema_result)
        overall_pass = overall_pass and schema_ok
        overall_pass = overall_pass and _phase3_genuine_ready(production_gate_payload)

    # Publish a pre-institutional snapshot so institutional P4 validates current run
    # state instead of stale prior-run artifacts.
    pre_skipped_optional = sum(1 for r in results if r.get("skipped", False))
    skipped_labels_pre = [str(r.get("label")) for r in results if r.get("skipped", False)]
    pre_summary = _build_summary(
        overall_pass=overall_pass,
        skipped_optional=pre_skipped_optional,
        results=results,
        production_gate_payload=production_gate_payload,
        stage="pre_institutional",
        production_gate_schema_ok=schema_ok,
        production_gate_schema_warnings=schema_warnings,
        skipped_gate_labels=skipped_labels_pre,
    )
    _write_gate_status_artifact(pre_summary)

    # Gate 4: unattended-run hardening contracts
    if not args.skip_institutional_gate:
        r4 = _run([python, "scripts/institutional_unattended_gate.py"], "institutional_unattended_gate")
        results.append(r4)
        overall_pass = overall_pass and bool(r4.get("passed"))
        _print_gate_result(r4, emit_json=args.emit_json)
    else:
        skipped = {"label": "institutional_unattended_gate", "skipped": True, "passed": True}
        results.append(skipped)
        if not args.emit_json:
            print("[SKIP] institutional_unattended_gate (--skip-institutional-gate)")

    # BYP-01 fix: enforce minimum active gate count.
    # Skipped optional gates are counted; if too many were skipped, force FAIL.
    skipped_optional = sum(1 for r in results if r.get("skipped", False))
    if skipped_optional > MAX_SKIPPED_OPTIONAL_GATES:
        overall_pass = False
        skip_fail_msg = (
            f"[FAIL] Too many optional gates skipped ({skipped_optional}/3). "
            f"Requires at most {MAX_SKIPPED_OPTIONAL_GATES} skip(s) for overall_passed=True."
        )
        if not args.emit_json:
            print(skip_fail_msg)

    summary = _build_summary(
        overall_pass=overall_pass,
        skipped_optional=skipped_optional,
        results=results,
        production_gate_payload=production_gate_payload,
        stage="final",
        production_gate_schema_ok=schema_ok,
        production_gate_schema_warnings=schema_warnings,
        skipped_gate_labels=[str(r.get("label")) for r in results if r.get("skipped", False)],
    )
    _write_gate_status_artifact(summary)

    if args.emit_json:
        print(json.dumps(summary, indent=2))
    else:
        status = "[PASS]" if overall_pass else "[FAIL]"
        passed_count = sum(1 for r in results if bool(r.get("passed")))
        print(f"\n{status} run_all_gates: {passed_count}/{len(results)} gates passed")
        if summary.get("phase3_reason") is not None:
            print(
                "Phase3 ready  : "
                f"{int(bool(summary.get('phase3_ready')))} "
                f"(reason={summary.get('phase3_reason')})"
            )

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
