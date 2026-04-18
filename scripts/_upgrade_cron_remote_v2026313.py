"""
One-shot upgrade:

- add Telegram fallback delivery to WhatsApp jobs
- backfill missing sessionTarget on agentTurn jobs
- quarantine obviously malformed cron records
- reset error counts for delivery failures

Safe to re-run (idempotent).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.openclaw_cron_contract import (
    DEFAULT_SESSION_TARGET,
    load_cron_jobs_payload,
    sanitize_cron_jobs_payload,
)
from utils.repo_python import resolve_repo_python


PROJECT_ROOT = Path(__file__).resolve().parents[1]

JOBS_PATH = Path.home() / ".openclaw" / "cron" / "jobs.json"
QUARANTINE_DIR = JOBS_PATH.parent / "quarantine"

# Telegram target — same number as WhatsApp
TELEGRAM_FALLBACK_TO = "+2348061573767"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    data, err = load_cron_jobs_payload(JOBS_PATH)
    if err:
        raise SystemExit(err)
    if not isinstance(data.get("jobs"), list):
        raise SystemExit(f"cron jobs payload missing jobs list: {JOBS_PATH}")

    sanitized, report = sanitize_cron_jobs_payload(
        data,
        default_session_target=DEFAULT_SESSION_TARGET,
        python_executable=resolve_repo_python(PROJECT_ROOT),
    )

    changed = 0
    for job in sanitized["jobs"]:
        delivery = job.get("delivery", {}) if isinstance(job.get("delivery"), dict) else {}
        state = job.get("state", {}) if isinstance(job.get("state"), dict) else {}
        consecutive_errors = state.get("consecutiveErrors", 0)

        # Add Telegram fallback to every job that delivers via WhatsApp.
        if delivery.get("channel") == "whatsapp" and "fallback" not in delivery:
            job["delivery"] = dict(delivery)
            job["delivery"]["fallback"] = {
                "channel": "telegram",
                "to": TELEGRAM_FALLBACK_TO,
            }
            changed += 1
            print(f"  [+fallback] {job.get('name', 'job')}")

        # Reset consecutive error counters for delivery failures
        # (gateway unreachable / WhatsApp disconnect - not logic errors).
        last_err = str(state.get("lastError", "") or "")
        if consecutive_errors > 0 and "delivery failed" in last_err.lower():
            if not isinstance(job.get("state"), dict):
                job["state"] = {}
            job["state"]["consecutiveErrors"] = 0
            job["state"]["lastError"] = ""
            changed += 1
            print(f"  [+reset]    {job.get('name', 'job')} ({consecutive_errors} errors cleared)")

    if report.get("quarantined_count", 0) > 0:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        quarantine_path = QUARANTINE_DIR / (
            f"cron_quarantine_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        _write_json(
            quarantine_path,
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "source_path": str(JOBS_PATH),
                "quarantined_count": report["quarantined_count"],
                "quarantined_jobs": report["quarantined_jobs"],
            },
        )
        print(f"  [+quarantine] {report['quarantined_count']} jobs -> {quarantine_path.name}")

    if report.get("backfilled_count", 0) > 0:
        for item in report["backfilled_jobs"]:
            print(
                f"  [+backfill]  {item.get('name') or item.get('id') or 'job'} "
                f"sessionTarget={item.get('sessionTarget')}"
            )

    if report.get("rewritten_count", 0) > 0:
        for item in report["rewritten_jobs"]:
            print(
                f"  [+rewrite]   {item.get('name') or item.get('id') or 'job'} "
                f"python={item.get('replacement')}"
            )

    if report.get("changed") or changed > 0 or report.get("quarantined_count", 0) > 0 or report.get("backfilled_count", 0) > 0:
        _write_json(JOBS_PATH, sanitized)

    print(
        "\nDone: "
        f"{changed} delivery updates, "
        f"{report.get('backfilled_count', 0)} backfills, "
        f"{report.get('rewritten_count', 0)} rewrites, "
        f"{report.get('quarantined_count', 0)} quarantined, "
        f"{len(sanitized['jobs'])} active jobs."
    )


if __name__ == "__main__":
    main()
