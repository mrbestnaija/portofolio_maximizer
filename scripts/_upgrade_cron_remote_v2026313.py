"""
One-shot upgrade: add Telegram fallback delivery + reset error counts for all
failing cron jobs as part of OpenClaw v2026.3.13 remote-workflow upgrade.

Safe to re-run (idempotent).
"""
import json
from pathlib import Path

JOBS_PATH = Path.home() / ".openclaw" / "cron" / "jobs.json"

# Telegram target — same number as WhatsApp
TELEGRAM_FALLBACK_TO = "+2348061573767"


def main() -> None:
    with open(JOBS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    changed = 0
    for job in data["jobs"]:
        delivery = job.get("delivery", {})
        state = job.get("state", {})
        consecutive_errors = state.get("consecutiveErrors", 0)

        # Add Telegram fallback to every job that delivers via WhatsApp
        if delivery.get("channel") == "whatsapp" and "fallback" not in delivery:
            job["delivery"]["fallback"] = {
                "channel": "telegram",
                "to": TELEGRAM_FALLBACK_TO,
            }
            changed += 1
            print(f"  [+fallback] {job['name']}")

        # Reset consecutive error counters for delivery failures
        # (gateway unreachable / WhatsApp disconnect — not logic errors)
        last_err = state.get("lastError", "")
        if consecutive_errors > 0 and "delivery failed" in last_err.lower():
            job["state"]["consecutiveErrors"] = 0
            job["state"]["lastError"] = ""
            print(f"  [+reset]    {job['name']} ({consecutive_errors} errors cleared)")

    with open(JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone: {changed} jobs updated, {len(data['jobs'])} total jobs.")


if __name__ == "__main__":
    main()
