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

# BYP-01 fix: cap how many optional gates can be skipped before we force overall_passed=False.
# With 3 optional gates, allowing only 1 skip means at least 2/3 must actually run.
MAX_SKIPPED_OPTIONAL_GATES = 1

# Artifact written after every run so downstream tools (e.g. institutional gate) can
# verify gates ran recently without re-running them (BYP-03 fix).
GATE_STATUS_ARTIFACT = Path(__file__).resolve().parents[1] / "logs" / "gate_status_latest.json"

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
        # BYP-05 fix: --allow-inconclusive-lift is time-bounded by max_warmup_days in
        # forecaster_monitoring.yml (regression_metrics.max_warmup_days = 30).
        # After max_warmup_days from first audit, INCONCLUSIVE must be treated as FAIL.
        cmd3 = [python, "scripts/production_audit_gate.py",
                "--reconcile",              # dry-run: surface unlinked closes without applying; overnight job applies
                "--allow-inconclusive-lift",  # Phase 7.19: INCONCLUSIVE during warmup = ok (ensemble is DISABLE_DEFAULT)
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

    summary = {
        "phase": "institutional_unattended_hardening",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_passed": bool(overall_pass),
        "skipped_optional_gates": skipped_optional,
        "max_skipped_optional_gates": MAX_SKIPPED_OPTIONAL_GATES,
        "gates": results,
    }

    # BYP-03 fix: write a status artifact so downstream tools can verify gates ran recently.
    try:
        GATE_STATUS_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        GATE_STATUS_ARTIFACT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass  # Non-fatal: artifact is best-effort

    if args.emit_json:
        print(json.dumps(summary, indent=2))
    else:
        status = "[PASS]" if overall_pass else "[FAIL]"
        passed_count = sum(1 for r in results if bool(r.get("passed")))
        print(f"\n{status} run_all_gates: {passed_count}/{len(results)} gates passed")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()

