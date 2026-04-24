#!/usr/bin/env python3
"""Read the family calibration JSONL and gate any downstream analysis.

This is the reader-side companion to ``scripts/family_calibration_writer.py``.
It intentionally refuses to emit an actionable recommendation unless the latest
row already passed the measurement gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from scripts.family_calibration_writer import DEFAULT_OUTPUT, load_family_calibration_rows


def load_latest_family_calibration_row(path: Path) -> dict[str, Any]:
    rows = load_family_calibration_rows(path)
    return rows[-1] if rows else {}


def evaluate_family_calibration_analysis_gate(path: Path) -> dict[str, Any]:
    latest = load_latest_family_calibration_row(path)
    passed = bool(latest.get("analysis_gate_passed")) if latest else False
    reasons = latest.get("analysis_gate_reasons") if isinstance(latest.get("analysis_gate_reasons"), list) else []
    return {
        "analysis_gate_passed": passed,
        "analysis_gate_reasons": reasons,
        "latest_schema_version": latest.get("schema_version"),
        "latest_window_cycles": latest.get("window_cycles"),
        "row_present": bool(latest),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the family calibration analysis gate.")
    parser.add_argument("--input", default=str(DEFAULT_OUTPUT), help="Family calibration JSONL path.")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    args = parser.parse_args(argv)

    report = evaluate_family_calibration_analysis_gate(Path(args.input))
    if not report["analysis_gate_passed"]:
        reasons = ", ".join(str(item) for item in report["analysis_gate_reasons"] if item)
        print(f"[FAIL] family calibration analysis gate blocked at {args.input}: {reasons or 'no actionable row'}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"[PASS] latest family calibration row is actionable at {args.input}")
    return 0 if report["analysis_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
