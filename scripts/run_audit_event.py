#!/usr/bin/env python3
"""
run_audit_event.py
------------------

Append structured JSONL audit events for Windows batch entrypoints.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: str, max_len: int = 4000) -> str:
    value = str(value or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(description="Append one structured run audit event to JSONL.")
    parser.add_argument("--audit-file", required=True, help="Path to JSONL audit file.")
    parser.add_argument("--run-id", required=True, help="Unique run ID.")
    parser.add_argument("--parent-run-id", default="", help="Optional parent run ID.")
    parser.add_argument("--script-name", required=True, help="Calling script name.")
    parser.add_argument("--event", required=True, help="Event type (RUN_START, STEP_START, STEP_END, RUN_END).")
    parser.add_argument("--status", default="", help="Event status (STARTED, RUNNING, SUCCESS, FAIL, SKIPPED).")
    parser.add_argument("--step", default="", help="Logical step name.")
    parser.add_argument("--subprocess-id", default="", help="Unique subprocess ID for this step.")
    parser.add_argument("--exit-code", type=int, default=0, help="Exit code for the event context.")
    parser.add_argument("--message", default="", help="Human-readable message.")
    parser.add_argument("--command", default="", help="Command summary (not full secrets).")
    parser.add_argument("--log-file", default="", help="Primary text log file path.")
    args = parser.parse_args()

    out_path = Path(args.audit_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "timestamp_utc": _utc_now(),
        "host_pid": os.getpid(),
        "run_id": _clean(args.run_id, 200),
        "parent_run_id": _clean(args.parent_run_id, 200),
        "script_name": _clean(args.script_name, 260),
        "event": _clean(args.event, 80),
        "status": _clean(args.status, 80),
        "step": _clean(args.step, 160),
        "subprocess_id": _clean(args.subprocess_id, 240),
        "exit_code": int(args.exit_code),
        "message": _clean(args.message, 2000),
        "command": _clean(args.command, 2000),
        "log_file": _clean(args.log_file, 600),
    }

    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

