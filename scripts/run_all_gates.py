#!/usr/bin/env python3
"""
run_all_gates.py — Master gate orchestrator (Phase 7.13-C3).

Runs all production gates in strict priority order and returns a JSON summary:

  1. ci_integrity_gate.py    (PnL structural invariants — CRITICAL/HIGH)
  2. check_quant_validation_health.py (FAIL-rate / GREEN-YELLOW-RED)
  3. production_audit_gate.py (forecast lift + profitability proof)

Exit codes:
    0  all blocking gates passed
    1  one or more blocking gates failed
    2  invocation / subprocess error

Usage:
    python scripts/run_all_gates.py
    python scripts/run_all_gates.py --skip-forecast-gate
    python scripts/run_all_gates.py --skip-profitability-gate
    python scripts/run_all_gates.py --strict    # MEDIUM violations also fail
    python scripts/run_all_gates.py --json      # emit JSON summary to stdout
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list, label: str) -> dict:
    """Run a subprocess and return a result dict."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        return {
            "label": label,
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "stdout": proc.stdout[-4000:] if proc.stdout else "",
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
        }
    except Exception as exc:
        return {
            "label": label,
            "exit_code": -1,
            "passed": False,
            "stdout": "",
            "stderr": str(exc),
        }


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
    results = []
    overall_pass = True

    # ── Gate 1: PnL integrity (always runs) ──────────────────────────────
    cmd1 = [python, "scripts/ci_integrity_gate.py"]
    if args.strict:
        cmd1.append("--strict")
    if args.db:
        cmd1 += ["--db", args.db]
    r1 = _run(cmd1, "ci_integrity_gate")
    results.append(r1)
    if not r1["passed"]:
        overall_pass = False
        if not args.emit_json:
            print(f"[FAIL] {r1['label']} (exit {r1['exit_code']})")
            if r1["stdout"]:
                print(r1["stdout"])
    else:
        if not args.emit_json:
            print(f"[PASS] {r1['label']}")

    # ── Gate 2: Quant validation health ──────────────────────────────────
    if not args.skip_forecast_gate:
        r2 = _run([python, "scripts/check_quant_validation_health.py"], "check_quant_validation_health")
        results.append(r2)
        if not r2["passed"]:
            overall_pass = False
            if not args.emit_json:
                print(f"[FAIL] {r2['label']} (exit {r2['exit_code']})")
                if r2["stdout"]:
                    print(r2["stdout"])
        else:
            if not args.emit_json:
                print(f"[PASS] {r2['label']}")
    else:
        results.append({"label": "check_quant_validation_health", "skipped": True, "passed": True})
        if not args.emit_json:
            print("[SKIP] check_quant_validation_health (--skip-forecast-gate)")

    # ── Gate 3: Production audit gate (forecast lift + profitability) ─────
    if not args.skip_profitability_gate:
        cmd3 = [python, "scripts/production_audit_gate.py"]
        if args.db:
            cmd3 += ["--db", args.db]
        r3 = _run(cmd3, "production_audit_gate")
        results.append(r3)
        if not r3["passed"]:
            overall_pass = False
            if not args.emit_json:
                print(f"[FAIL] {r3['label']} (exit {r3['exit_code']})")
                if r3["stdout"]:
                    print(r3["stdout"])
        else:
            if not args.emit_json:
                print(f"[PASS] {r3['label']}")
    else:
        results.append({"label": "production_audit_gate", "skipped": True, "passed": True})
        if not args.emit_json:
            print("[SKIP] production_audit_gate (--skip-profitability-gate)")

    # ── Summary ──────────────────────────────────────────────────────────
    summary = {
        "phase": "7.13",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_passed": overall_pass,
        "gates": results,
    }

    if args.emit_json:
        print(json.dumps(summary, indent=2))
    else:
        status = "[PASS]" if overall_pass else "[FAIL]"
        passed_count = sum(1 for r in results if r.get("passed"))
        print(f"\n{status} run_all_gates: {passed_count}/{len(results)} gates passed")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
