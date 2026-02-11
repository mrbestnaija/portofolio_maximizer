#!/usr/bin/env python3
"""CI gate: PnL integrity checks.

Runs the PnL integrity enforcer audit and exits non-zero if any
CRITICAL or HIGH severity violations are found.

Usage:
    python scripts/ci_integrity_gate.py
    python scripts/ci_integrity_gate.py --db path/to/db
    python scripts/ci_integrity_gate.py --strict  # fail on MEDIUM too

Exit codes:
    0 -- all checks passed
    1 -- CRITICAL or HIGH violations found
    2 -- database not found or schema error
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer


def main():
    import argparse

    DEFAULT_DB = os.path.join(ROOT, "data", "portfolio_maximizer.db")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite database")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail on MEDIUM severity violations",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"[SKIP] Database not found: {args.db}")
        print("[SKIP] CI gate skipped (no DB to check).")
        sys.exit(0)

    try:
        with PnLIntegrityEnforcer(args.db) as enforcer:
            violations = enforcer.run_full_integrity_audit()
            metrics = enforcer.get_canonical_metrics()
    except Exception as exc:
        print(f"[ERROR] Failed to run integrity audit: {exc}")
        sys.exit(2)

    # Report
    print("=== PnL Integrity CI Gate ===")
    print(f"  Round-trips:    {metrics.total_round_trips}")
    print(f"  Total PnL:      ${metrics.total_realized_pnl:+,.2f}")
    print(f"  Win rate:        {metrics.win_rate:.1%}")
    print(f"  Double-count:    {metrics.opening_legs_with_pnl} (must be 0)")

    if not violations:
        print("\n[PASS] All integrity checks passed.")
        sys.exit(0)

    # Categorize violations
    fail_severities = {"CRITICAL", "HIGH"}
    if args.strict:
        fail_severities.add("MEDIUM")

    blocking = [v for v in violations if v.severity in fail_severities]
    warnings = [v for v in violations if v.severity not in fail_severities]

    for v in violations:
        status = "FAIL" if v.severity in fail_severities else "WARN"
        print(f"\n  [{status}] [{v.severity}] {v.check_name}")
        print(f"         {v.description}")

    if blocking:
        print(f"\n[FAIL] {len(blocking)} blocking violation(s) found.")
        sys.exit(1)
    else:
        print(f"\n[PASS] {len(warnings)} non-blocking warning(s).")
        sys.exit(0)


if __name__ == "__main__":
    main()
