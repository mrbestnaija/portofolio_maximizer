#!/usr/bin/env python3
"""
run_gate_lift_replay.py
-----------------------

Optional historical replay helper for profitability gate evidence accumulation.
Uses real market data via run_auto_trader.py --as-of-date and emits an audit
artifact for independent review.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "audit_gate"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _append_log(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except Exception:
        pass


def _build_business_dates(days: int, start_offset_days: int) -> List[date]:
    out: List[date] = []
    cursor = datetime.now(UTC).date() - timedelta(days=max(0, int(start_offset_days)))
    while len(out) < days:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor -= timedelta(days=1)
    out.reverse()  # oldest -> newest for chronological replay continuity
    return out


def _emit_audit_event(
    *,
    python_bin: Path,
    audit_file: Path | None,
    run_id: str,
    parent_run_id: str,
    step: str,
    subprocess_id: str,
    status: str,
    exit_code: int,
    message: str,
    log_file: Path | None,
) -> None:
    if audit_file is None:
        return
    audit_script = Path(__file__).resolve().with_name("run_audit_event.py")
    if not audit_script.exists():
        return

    cmd = [
        str(python_bin),
        str(audit_script),
        "--audit-file",
        str(audit_file),
        "--run-id",
        run_id,
        "--parent-run-id",
        parent_run_id,
        "--script-name",
        "run_gate_lift_replay.py",
        "--event",
        "STEP_END",
        "--status",
        status,
        "--step",
        step,
        "--subprocess-id",
        subprocess_id,
        "--exit-code",
        str(int(exit_code)),
        "--message",
        message,
        "--log-file",
        str(log_file or ""),
    ]
    try:
        subprocess.run(cmd, check=False)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run historical as-of-date replays to accumulate gate evidence."
    )
    parser.add_argument("--python-bin", required=True, help="Python interpreter for run_auto_trader.")
    parser.add_argument("--auto-trader-script", required=True, help="Path to scripts/run_auto_trader.py.")
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker list.")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--initial-capital", type=float, default=25000.0)
    parser.add_argument("--days", type=int, default=5, help="Number of business dates to replay.")
    parser.add_argument("--start-offset-days", type=int, default=1, help="Start N days before today.")
    parser.add_argument("--yfinance-interval", default="1d", help="Market interval for replay run.")
    parser.add_argument("--proof-mode", action="store_true", help="Enable proof-mode exits during replay.")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume portfolio state between replay runs (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start replay without loading prior portfolio state.",
    )
    parser.add_argument("--strict", action="store_true", default=False, help="Return non-zero if any replay run fails.")
    parser.add_argument("--run-id", default="", help="Parent run id for audit correlation.")
    parser.add_argument("--parent-run-id", default="", help="Optional parent run id.")
    parser.add_argument("--audit-file", default="", help="Optional JSONL audit file path.")
    parser.add_argument("--log-file", default="", help="Optional shared text log path.")
    parser.add_argument("--output-json", default="", help="Optional replay summary artifact path.")
    args = parser.parse_args()

    python_bin = Path(args.python_bin).expanduser().resolve()
    auto_trader_script = Path(args.auto_trader_script).expanduser().resolve()
    if not python_bin.exists():
        raise SystemExit(f"[ERROR] python bin not found: {python_bin}")
    if not auto_trader_script.exists():
        raise SystemExit(f"[ERROR] auto trader script not found: {auto_trader_script}")

    days = max(0, int(args.days))
    if days == 0:
        print("[INFO] replay skipped (days=0)")
        return 0

    replay_dates = _build_business_dates(days=days, start_offset_days=args.start_offset_days)
    if not replay_dates:
        print("[INFO] replay skipped (no business dates resolved)")
        return 0

    audit_file = Path(args.audit_file).expanduser().resolve() if args.audit_file else None
    log_file = Path(args.log_file).expanduser().resolve() if args.log_file else None
    root_run_id = args.run_id.strip() or f"gate_lift_replay_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    parent_run_id = args.parent_run_id.strip()

    _append_log(log_file, f"[REPLAY] start run_id={root_run_id} dates={len(replay_dates)}")
    results: List[Dict[str, Any]] = []
    failures = 0

    for idx, replay_date in enumerate(replay_dates, start=1):
        subproc_id = f"{root_run_id}_R{idx}"
        as_of_date = replay_date.isoformat()
        step_name = f"gate_lift_replay_{as_of_date}"
        cmd = [
            str(python_bin),
            str(auto_trader_script),
            "--tickers",
            args.tickers,
            "--lookback-days",
            str(int(args.lookback_days)),
            "--initial-capital",
            str(float(args.initial_capital)),
            "--cycles",
            "1",
            "--sleep-seconds",
            "0",
            "--yfinance-interval",
            str(args.yfinance_interval),
            "--as-of-date",
            as_of_date,
        ]
        cmd.append("--resume" if args.resume else "--no-resume")
        if args.proof_mode:
            cmd.append("--proof-mode")

        _append_log(
            log_file,
            (
                f"[REPLAY] [{idx}/{len(replay_dates)}] as_of={as_of_date} "
                f"cmd={Path(auto_trader_script).name} --as-of-date {as_of_date}"
            ),
        )
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            failures += 1

        stdout_tail = "\n".join((run.stdout or "").splitlines()[-40:])
        stderr_tail = "\n".join((run.stderr or "").splitlines()[-40:])
        _append_log(log_file, f"[REPLAY] as_of={as_of_date} rc={run.returncode}")
        if stdout_tail:
            _append_log(log_file, f"[REPLAY][STDOUT]\n{stdout_tail}")
        if stderr_tail:
            _append_log(log_file, f"[REPLAY][STDERR]\n{stderr_tail}")

        status = "SUCCESS" if run.returncode == 0 else "FAIL"
        _emit_audit_event(
            python_bin=python_bin,
            audit_file=audit_file,
            run_id=root_run_id,
            parent_run_id=parent_run_id,
            step=step_name,
            subprocess_id=subproc_id,
            status=status,
            exit_code=run.returncode,
            message=f"Replay as-of-date {as_of_date} completed",
            log_file=log_file,
        )

        results.append(
            {
                "as_of_date": as_of_date,
                "subprocess_id": subproc_id,
                "exit_code": int(run.returncode),
                "status": status,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
        )

    artifact_path: Path
    if args.output_json:
        artifact_path = Path(args.output_json).expanduser().resolve()
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        artifact_path = DEFAULT_OUTPUT_DIR / f"gate_lift_replay_{root_run_id}.json"

    payload = {
        "timestamp_utc": _utc_now(),
        "run_id": root_run_id,
        "parent_run_id": parent_run_id,
        "inputs": {
            "tickers": args.tickers,
            "lookback_days": int(args.lookback_days),
            "initial_capital": float(args.initial_capital),
            "days": days,
            "start_offset_days": int(args.start_offset_days),
            "yfinance_interval": args.yfinance_interval,
            "proof_mode": bool(args.proof_mode),
            "resume": bool(args.resume),
        },
        "summary": {
            "total_runs": len(results),
            "failed_runs": failures,
            "passed_runs": len(results) - failures,
        },
        "results": results,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _append_log(log_file, f"[REPLAY] artifact={artifact_path}")

    print(f"[REPLAY] {artifact_path}")
    if failures and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
