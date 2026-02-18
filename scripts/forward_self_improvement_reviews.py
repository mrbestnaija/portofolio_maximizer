#!/usr/bin/env python3
"""
Forward pending self-improvement proposals to human reviewers via OpenClaw.

Safety defaults:
- Sends only compact summaries (file + short description), never full diffs.
- Redacts obvious secret-like tokens in generated message text.
- Persists sent proposal ids to avoid repeated spam across restarts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from etl.secret_loader import bootstrap_dotenv

DEFAULT_PROPOSALS_DIR = PROJECT_ROOT / "logs" / "llm_activity" / "proposals"
DEFAULT_STATE_FILE = PROJECT_ROOT / "logs" / "automation" / "self_improve_review_forward_state.json"
DEFAULT_REPORT_FILE = PROJECT_ROOT / "logs" / "automation" / "self_improve_review_forward_latest.json"

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9\-\._~\+/=]{16,}\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b"),
    re.compile(r"\b(token|secret|password|api[_-]?key)\s*[:=]\s*[^,\s]+", re.IGNORECASE),
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _redact(text: str) -> str:
    out = str(text or "")
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


def _truncate(text: str, max_chars: int) -> str:
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _parse_ts(text: str) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _load_proposals(
    proposals_dir: Path,
    *,
    max_age_days: int,
) -> list[dict[str, Any]]:
    if not proposals_dir.exists():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(max_age_days)))
    out: list[dict[str, Any]] = []
    for path in sorted(proposals_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        proposed_at = _parse_ts(payload.get("proposed_at", "")) or datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if proposed_at < cutoff:
            continue

        status = str(payload.get("status", "proposed")).strip().lower()
        approved = bool(payload.get("approved", False))
        applied = bool(payload.get("applied", False))
        if status in {"approved", "applied", "closed", "done"} or approved or applied:
            continue

        out.append(
            {
                "id": path.stem,
                "path": str(path),
                "proposed_at": proposed_at.isoformat(),
                "target_file": str(payload.get("target_file", "")),
                "description": str(payload.get("description", "")),
                "status": status or "proposed",
            }
        )
    out.sort(key=lambda x: str(x.get("proposed_at", "")))
    return out


def _build_message(
    proposals: list[dict[str, Any]],
    *,
    max_items: int,
) -> str:
    rows = proposals[: max(1, int(max_items))]
    lines = [
        f"[PMX] Self-improvement review queue: {len(proposals)} pending proposal(s).",
        "Human review requested for project improvements:",
    ]
    for idx, row in enumerate(rows, start=1):
        target = _truncate(_redact(str(row.get("target_file") or "")), 70)
        desc = _truncate(_redact(str(row.get("description") or "")), 120)
        lines.append(f"{idx}. {target} :: {desc}")
    if len(proposals) > len(rows):
        lines.append(f"... plus {len(proposals) - len(rows)} more pending proposal(s).")
    lines.append("Review source: logs/llm_activity/proposals")
    lines.append("Action: approve/reject and apply through normal reviewed workflow.")
    return "\n".join(lines)


def _send_openclaw_message(
    *,
    message: str,
    targets: str,
    to: str,
    channel: str,
    timeout_seconds: float,
    dry_run: bool,
) -> tuple[int, str, str, list[str]]:
    notify_script = PROJECT_ROOT / "scripts" / "openclaw_notify.py"
    cmd = [
        sys.executable,
        str(notify_script),
        "--message",
        message,
        "--timeout-seconds",
        str(float(timeout_seconds)),
        "--no-infer-to",
    ]
    if targets.strip():
        cmd.extend(["--targets", targets.strip()])
    elif to.strip():
        cmd.extend(["--to", to.strip()])
    if channel.strip():
        cmd.extend(["--channel", channel.strip()])
    if dry_run:
        cmd.append("--dry-run")

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=max(10.0, float(timeout_seconds) + 10.0),
        check=False,
    )
    return int(proc.returncode), (proc.stdout or ""), (proc.stderr or ""), cmd


def main(argv: list[str]) -> int:
    bootstrap_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals-dir", default=str(DEFAULT_PROPOSALS_DIR), help="Directory containing proposal JSON files.")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE), help="JSON state file for already-forwarded proposal ids.")
    parser.add_argument("--report-file", default=str(DEFAULT_REPORT_FILE), help="JSON run report output.")
    parser.add_argument("--max-items", type=int, default=8, help="Maximum proposal items to include in one message.")
    parser.add_argument("--max-age-days", type=int, default=30, help="Only consider proposals newer than this age.")
    parser.add_argument("--resend-pending", action="store_true", help="Resend all pending proposals, not just unseen ones.")
    parser.add_argument("--min-interval-minutes", type=int, default=30, help="Minimum minutes between sends.")
    parser.add_argument("--targets", default=os.getenv("OPENCLAW_TARGETS", ""), help="OpenClaw targets override (comma list).")
    parser.add_argument("--to", default=os.getenv("OPENCLAW_TO", ""), help="OpenClaw single target override.")
    parser.add_argument("--channel", default=os.getenv("OPENCLAW_CHANNEL", ""), help="OpenClaw channel override.")
    parser.add_argument("--timeout-seconds", type=float, default=20.0, help="OpenClaw notify timeout seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Do not send; only print and report.")
    parser.add_argument("--force", action="store_true", help="Ignore min-interval guard.")
    args = parser.parse_args(argv)

    proposals_dir = Path(args.proposals_dir).expanduser()
    state_file = Path(args.state_file).expanduser()
    report_file = Path(args.report_file).expanduser()

    state = _safe_load_json(state_file)
    sent_ids = set(state.get("sent_ids", [])) if isinstance(state.get("sent_ids"), list) else set()
    last_sent_at = _parse_ts(state.get("last_sent_at", ""))

    pending = _load_proposals(proposals_dir, max_age_days=int(args.max_age_days))
    if not args.resend_pending:
        pending = [p for p in pending if p.get("id") not in sent_ids]

    report: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "proposals_dir": str(proposals_dir),
        "state_file": str(state_file),
        "pending_count": len(pending),
        "resend_pending": bool(args.resend_pending),
        "dry_run": bool(args.dry_run),
        "status": "NOOP",
    }

    if not pending:
        report["reason"] = "No pending proposals to forward."
        _safe_write_json(report_file, report)
        print("[self_improve_forward] No pending proposals to forward.")
        return 0

    if not args.force and last_sent_at is not None:
        since = datetime.now(timezone.utc) - last_sent_at
        min_interval = timedelta(minutes=max(0, int(args.min_interval_minutes)))
        if since < min_interval:
            report["reason"] = f"Rate-limited by min interval ({args.min_interval_minutes} minutes)."
            report["status"] = "RATE_LIMITED"
            _safe_write_json(report_file, report)
            print("[self_improve_forward] Rate-limited; skipping send.")
            return 0

    message = _build_message(pending, max_items=int(args.max_items))
    report["message_preview"] = _truncate(message, 500)
    report["proposal_ids"] = [str(p.get("id")) for p in pending]

    effective_targets = str(args.targets or "").strip()
    effective_to = str(args.to or "").strip()
    if not effective_targets and not effective_to:
        report["status"] = "NOOP"
        report["reason"] = "OPENCLAW_TARGETS/OPENCLAW_TO not configured for review forwarding."
        _safe_write_json(report_file, report)
        print("[self_improve_forward] No OpenClaw targets configured; skipping send.")
        return 0

    try:
        rc, out, err, cmd = _send_openclaw_message(
            message=message,
            targets=effective_targets,
            to=effective_to,
            channel=str(args.channel or ""),
            timeout_seconds=float(args.timeout_seconds),
            dry_run=bool(args.dry_run),
        )
    except subprocess.TimeoutExpired:
        report["status"] = "FAIL"
        report["reason"] = "openclaw_notify timeout"
        _safe_write_json(report_file, report)
        print("[self_improve_forward] FAILED: openclaw_notify timeout", file=sys.stderr)
        return 1
    except Exception as exc:
        report["status"] = "FAIL"
        report["reason"] = f"openclaw_notify exception: {exc}"
        _safe_write_json(report_file, report)
        print(f"[self_improve_forward] FAILED: {exc}", file=sys.stderr)
        return 1

    report["notify_command"] = cmd
    report["notify_exit_code"] = rc
    report["notify_stdout_tail"] = "\n".join(out.splitlines()[-20:])
    report["notify_stderr_tail"] = "\n".join(err.splitlines()[-20:])

    if rc != 0:
        report["status"] = "FAIL"
        report["reason"] = "openclaw_notify failed"
        _safe_write_json(report_file, report)
        print("[self_improve_forward] FAILED: openclaw_notify returned non-zero", file=sys.stderr)
        return 1

    if not args.dry_run:
        merged = list(sent_ids.union({str(p.get("id")) for p in pending}))
        if len(merged) > 5000:
            merged = merged[-5000:]
        next_state = {
            "last_sent_at": _utc_now_iso(),
            "sent_ids": merged,
        }
        _safe_write_json(state_file, next_state)

    report["status"] = "SENT" if not args.dry_run else "DRY_RUN"
    _safe_write_json(report_file, report)
    print(f"[self_improve_forward] {report['status']} pending={len(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
