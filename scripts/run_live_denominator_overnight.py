#!/usr/bin/env python3
"""
Run frozen live auto-trader cycles overnight and record fresh denominator signals.

This runner is intentionally narrower than run_overnight_refresh.py:
- live execution mode only
- no proof-mode/bootstrap logic
- no gate or linkage semantics changes
- appends only the three fresh TRADE-cohort signals after each cycle
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON = REPO_ROOT / "simpleTrader_env" / "Scripts" / "python.exe"
PYTHON = str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else sys.executable
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "overnight_denominator"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_trade_day_rows(summary: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    latest_day: str | None = None
    latest_rows: list[dict[str, Any]] = []
    for row in summary.get("dataset_windows", []):
        if str(row.get("context_type") or "").strip().upper() != "TRADE":
            continue
        file_name = str(row.get("file") or "")
        if not file_name.startswith("forecast_audit_") or len(file_name) < 23:
            continue
        day = file_name[len("forecast_audit_") : len("forecast_audit_") + 8]
        if not day.isdigit():
            continue
        if latest_day is None or day > latest_day:
            latest_day = day
            latest_rows = [row]
        elif day == latest_day:
            latest_rows.append(row)
    return latest_day, latest_rows


def extract_fresh_trade_signals(
    summary_path: Path,
    db_path: Path,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    latest_day, raw_trade_rows = _latest_trade_day_rows(summary)
    fresh_rows = [
        row
        for row in raw_trade_rows
        if str(row.get("outcome_status") or "").strip().upper() != "NON_TRADE_CONTEXT"
    ]
    non_trade_diagnostics = len(raw_trade_rows) - len(fresh_rows)
    status_counts = Counter(
        str(row.get("outcome_status") or "").strip().upper() for row in fresh_rows
    )
    reason_counts = Counter(
        str(row.get("outcome_reason") or "").strip().upper() for row in fresh_rows
    )

    production_valid_matched = 0
    production_valid_rows = 0
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            for row in fresh_rows:
                ts_signal_id = str(row.get("ts_signal_id") or "").strip()
                if not ts_signal_id:
                    continue
                exists = cur.execute(
                    "SELECT 1 FROM production_closed_trades WHERE ts_signal_id = ? LIMIT 1",
                    (ts_signal_id,),
                ).fetchone()
                if exists:
                    production_valid_rows += 1
                    if str(row.get("outcome_status") or "").strip().upper() == "MATCHED":
                        production_valid_matched += 1
        finally:
            conn.close()

    return {
        "generated_utc": summary.get("generated_utc"),
        "latest_day": latest_day,
        "fresh_trade_context_rows_raw": len(raw_trade_rows),
        "fresh_trade_rows": len(fresh_rows),
        "fresh_trade_exclusions": {
            "non_trade_context": 0,
            "invalid_context": status_counts.get("INVALID_CONTEXT", 0),
            "missing_execution_metadata": reason_counts.get("MISSING_EXECUTION_METADATA", 0),
        },
        "fresh_trade_diagnostics": {
            "non_trade_context_rows": non_trade_diagnostics,
        },
        "fresh_linkage_included": sum(
            1 for row in fresh_rows if bool(row.get("counts_toward_linkage_denominator"))
        ),
        "fresh_production_valid_rows": production_valid_rows,
        "fresh_production_valid_matched": production_valid_matched,
    }


def _run_command(cmd: list[str], log_path: Path) -> int:
    with subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utc_now().isoformat()}] RUN {' '.join(cmd)}\n")
            for line in proc.stdout or []:
                sys.stdout.write(line)
                handle.write(line)
        return proc.wait()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _seconds_until_next_weekday(now: datetime) -> int:
    weekday = now.weekday()
    if weekday < 5:
        return 0
    return (7 - weekday) * 86400


def _production_audit_dir(audit_dir: Path) -> Path:
    audit_dir = Path(audit_dir)
    if audit_dir.name == "production":
        return audit_dir
    return audit_dir / "production"


def _forecast_audit_sanitization_cmd(audit_dir: Path) -> list[str]:
    production_audit_dir = _production_audit_dir(audit_dir)
    eval_audit_dir = production_audit_dir.parent / "production_eval"
    quarantine_dir = production_audit_dir / "quarantine"
    return [
        PYTHON,
        "scripts/sanitize_production_forecast_audits.py",
        "--audit-dir",
        str(production_audit_dir),
        "--eval-audit-dir",
        str(eval_audit_dir),
        "--quarantine-dir",
        str(quarantine_dir),
        "--manifest-path",
        str(production_audit_dir / "forecast_audit_manifest.jsonl"),
        "--eval-manifest-path",
        str(eval_audit_dir / "forecast_audit_manifest.jsonl"),
        "--apply",
    ]


def _sanitize_forecast_audits(audit_dir: Path, log_path: Path) -> int:
    return _run_command(_forecast_audit_sanitization_cmd(audit_dir), log_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run live frozen-strategy cycles overnight and record fresh denominator signals."
    )
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker list.")
    parser.add_argument("--cycles", type=int, default=12, help="Number of one-cycle live runs.")
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=86400,
        help="Delay between one-cycle live runs. Default: 86400.",
    )
    parser.add_argument(
        "--audit-dir",
        default="logs/forecast_audits",
        help="Forecast audit directory passed to check_forecast_audits.py.",
    )
    parser.add_argument(
        "--db",
        default="data/portfolio_maximizer.db",
        help="SQLite DB path for production_closed_trades.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=500,
        help="Max files passed to check_forecast_audits.py.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing portfolio state.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start from a fresh portfolio state.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip run_auto_trader and refresh only the audit summary/signals.",
    )
    parser.add_argument(
        "--stop-on-first-match",
        action="store_true",
        help="Stop early once a fresh production-valid matched row appears.",
    )
    parser.add_argument(
        "--progress-linkage-threshold",
        type=int,
        default=2,
        help="Treat fresh_linkage_included at or above this threshold as progress. Default: 2.",
    )
    parser.add_argument(
        "--stop-on-progress",
        action="store_true",
        help=(
            "Stop early once fresh_linkage_included reaches the configured threshold "
            "or a fresh production-valid matched row appears."
        ),
    )
    parser.add_argument(
        "--weekdays-only",
        dest="weekdays_only",
        action="store_true",
        default=True,
        help="Run only on weekdays; weekend loops sleep through to Monday. Default: on.",
    )
    parser.add_argument(
        "--allow-weekends",
        dest="weekdays_only",
        action="store_false",
        help="Allow Saturday/Sunday cycles.",
    )
    args = parser.parse_args()

    if args.cycles < 1:
        raise SystemExit("--cycles must be >= 1")
    if args.sleep_seconds < 0:
        raise SystemExit("--sleep-seconds must be >= 0")
    if args.progress_linkage_threshold < 1:
        raise SystemExit("--progress-linkage-threshold must be >= 1")

    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = utc_now().strftime("%Y%m%d_%H%M%S")
    log_path = DEFAULT_LOG_DIR / f"live_denominator_{run_id}.log"
    jsonl_path = DEFAULT_LOG_DIR / f"live_denominator_{run_id}.jsonl"
    latest_path = DEFAULT_LOG_DIR / "live_denominator_latest.json"
    summary_path = REPO_ROOT / "logs" / "forecast_audits_cache" / "latest_summary.json"
    db_path = (REPO_ROOT / args.db) if not Path(args.db).is_absolute() else Path(args.db)

    run_meta = {
        "run_id": run_id,
        "started_utc": utc_now().isoformat(),
        "tickers": [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()],
        "cycles": args.cycles,
        "sleep_seconds": args.sleep_seconds,
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "audit_dir": args.audit_dir,
        "db": str(db_path),
        "max_files": args.max_files,
        "progress_linkage_threshold": args.progress_linkage_threshold,
        "stop_on_progress": bool(args.stop_on_progress),
        "weekdays_only": bool(args.weekdays_only),
    }
    latest_path.write_text(json.dumps({"run_meta": run_meta, "cycles": []}, indent=2), encoding="utf-8")

    cycle_records: list[dict[str, Any]] = []
    cycle = 0
    while cycle < args.cycles:
        now = utc_now()
        if args.weekdays_only and not args.dry_run:
            weekend_sleep = _seconds_until_next_weekday(now)
            if weekend_sleep > 0:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"[{now.isoformat()}] WEEKEND_SKIP sleeping_seconds={weekend_sleep}\n"
                    )
                time.sleep(weekend_sleep)
                continue

        cycle += 1
        cycle_record: dict[str, Any] = {
            "run_id": run_id,
            "cycle": cycle,
            "started_utc": now.isoformat(),
        }

        trader_rc = 0
        if not args.dry_run:
            trader_cmd = [
                PYTHON,
                "scripts/run_auto_trader.py",
                "--tickers",
                args.tickers,
                "--cycles",
                "1",
                "--sleep-seconds",
                "0",
                "--execution-mode",
                "live",
                "--resume" if args.resume else "--no-resume",
            ]
            trader_rc = _run_command(trader_cmd, log_path)
        cycle_record["run_auto_trader_rc"] = trader_rc

        audit_sanitization_rc = None
        if not args.dry_run:
            audit_sanitization_rc = _sanitize_forecast_audits(Path(args.audit_dir), log_path)
        cycle_record["forecast_audit_sanitization_rc"] = audit_sanitization_rc

        audit_cmd = [
            PYTHON,
            "scripts/check_forecast_audits.py",
            "--audit-dir",
            args.audit_dir,
            "--db",
            str(db_path),
            "--max-files",
            str(args.max_files),
        ]
        audit_rc = _run_command(audit_cmd, log_path)
        cycle_record["check_forecast_audits_rc"] = audit_rc

        signals = extract_fresh_trade_signals(summary_path, db_path)
        cycle_record.update(signals)
        progress_triggered = (
            int(signals["fresh_linkage_included"]) >= args.progress_linkage_threshold
            or int(signals["fresh_production_valid_matched"]) > 0
        )
        progress_reasons: list[str] = []
        if int(signals["fresh_linkage_included"]) >= args.progress_linkage_threshold:
            progress_reasons.append("LINKAGE_THRESHOLD_REACHED")
        if int(signals["fresh_production_valid_matched"]) > 0:
            progress_reasons.append("PRODUCTION_VALID_MATCHED")
        cycle_record["progress_triggered"] = progress_triggered
        cycle_record["progress_reasons"] = progress_reasons
        cycle_record["completed_utc"] = utc_now().isoformat()
        cycle_records.append(cycle_record)
        _append_jsonl(jsonl_path, cycle_record)
        latest_path.write_text(
            json.dumps({"run_meta": run_meta, "cycles": cycle_records}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(cycle_record, indent=2))

        if args.stop_on_first_match and int(signals["fresh_production_valid_matched"]) > 0:
            break
        if args.stop_on_progress and progress_triggered:
            break
        if cycle < args.cycles and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
