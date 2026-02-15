#!/usr/bin/env python3
"""
Inbox workflows (Gmail + Proton Mail Bridge) for Portfolio Maximizer.

This script is intentionally conservative:
- Scans are read-only by default (no message state changes).
- Sending is disabled by default; enable explicitly via config or PMX_INBOX_ALLOW_SEND=1.
- Credentials are loaded locally via etl/secret_loader.py (supports *_FILE).

OpenClaw integration:
- If OPENCLAW_TARGETS/OPENCLAW_TO is configured, a scan summary can be sent via OpenClaw.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.email_inbox import (  # noqa: E402
    InboxScanResult,
    fetch_message_eml,
    load_inbox_config,
    scan_account,
    scan_result_to_dict,
    send_email,
)

def _read_stdin() -> str:
    try:
        return sys.stdin.read()
    except Exception:
        return ""


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_scan_summary(results: Sequence[InboxScanResult], *, max_lines: int = 40) -> str:
    lines: List[str] = []
    lines.append("[INBOX] Scan summary")
    for res in results:
        if not res.mailbox and not res.search_criteria:
            lines.append(f"- {res.account}: disabled")
            continue
        lines.append(f"- {res.account} ({res.label}): matched={res.total_matched}, fetched={res.fetched}")
        for msg in res.messages[:10]:
            uid = msg.uid or msg.seq
            subj = (msg.subject or "").strip() or "(no subject)"
            frm = (msg.from_address or "").strip() or "(no from)"
            lines.append(f"  uid={uid} | {frm} | {subj}")

    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["...(truncated)"]
    return "\n".join(lines).strip()


def _maybe_notify_openclaw(summary: str, *, config: Dict[str, Any]) -> None:
    notifications = config.get("notifications") if isinstance(config, dict) else None
    notifications = notifications if isinstance(notifications, dict) else {}
    oc_cfg = notifications.get("openclaw") if isinstance(notifications, dict) else None
    oc_cfg = oc_cfg if isinstance(oc_cfg, dict) else {}

    if not bool(oc_cfg.get("enabled", False)):
        return

    # Unified repo knob to disable OpenClaw notifications.
    if (os.getenv("PMX_NOTIFY_OPENCLAW") or "").strip() == "0":
        return

    default_channel = (os.getenv("OPENCLAW_CHANNEL") or "").strip() or None
    env_targets = (os.getenv("OPENCLAW_TARGETS") or "").strip()
    env_to = (os.getenv("OPENCLAW_TO") or "").strip()
    try:
        from utils.openclaw_cli import resolve_openclaw_targets

        targets = resolve_openclaw_targets(
            env_targets=env_targets,
            env_to=env_to,
            cfg_to=oc_cfg.get("to"),
            default_channel=default_channel,
        )
    except Exception:
        targets = []

    if not targets:
        return

    try:
        max_chars = int(oc_cfg.get("max_message_chars") or 1500)
    except Exception:
        max_chars = 1500

    message = (summary or "").strip()
    if max_chars > 0 and len(message) > max_chars:
        message = message[: max(0, max_chars - 20)].rstrip() + "\n...(truncated)"

    command = (os.getenv("OPENCLAW_COMMAND") or "openclaw").strip() or "openclaw"
    try:
        timeout_seconds = float(os.getenv("OPENCLAW_TIMEOUT_SECONDS") or 20)
    except Exception:
        timeout_seconds = 20.0

    try:
        from utils.openclaw_cli import send_message_multi

        send_message_multi(
            targets=targets,
            message=message,
            command=command,
            cwd=PROJECT_ROOT,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        return


def cmd_list_accounts(config: Dict[str, Any]) -> int:
    accounts = config.get("accounts") if isinstance(config, dict) else None
    if not isinstance(accounts, dict):
        print("[inbox_workflow] No accounts configured.")
        return 2

    print("Accounts (from config/inbox_workflows.yml)")
    for name, cfg in accounts.items():
        if not isinstance(cfg, dict):
            continue
        enabled = bool(cfg.get("enabled", False))
        label = str(cfg.get("label") or name)
        print(f"- {name}: enabled={enabled} label={label}")
    return 0


def cmd_scan(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    accounts = config.get("accounts") if isinstance(config, dict) else None
    if not isinstance(accounts, dict):
        print("[inbox_workflow] Invalid config: missing accounts.", file=sys.stderr)
        return 2

    selected: List[str] = []
    if args.account:
        selected = [args.account]
    else:
        selected = [k for k, v in accounts.items() if isinstance(v, dict) and bool(v.get("enabled", False))]

    if not selected:
        print("[inbox_workflow] No enabled accounts to scan.", file=sys.stderr)
        return 2

    results: List[InboxScanResult] = []
    for account in selected:
        try:
            results.append(scan_account(project_root=PROJECT_ROOT, config=config, account_name=account))
        except Exception as exc:
            print(f"[inbox_workflow] Scan failed for {account}: {exc}", file=sys.stderr)
            continue

    payload = {
        "timestamp_utc": results[0].timestamp_utc if results else "",
        "results": [scan_result_to_dict(r) for r in results],
    }

    if not args.no_write:
        stamp = (payload.get("timestamp_utc") or "").replace(":", "").replace("-", "")
        stamp = stamp.replace("T", "_").replace("Z", "")
        for res in results:
            out = PROJECT_ROOT / "logs" / "inbox" / res.account / f"scan_{stamp}.json"
            _write_json(out, scan_result_to_dict(res))

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = _format_scan_summary(results)
        print(summary)
        notify_openclaw = args.notify_openclaw
        if notify_openclaw is None:
            # Auto-notify if a target is configured, unless explicitly disabled.
            notify_openclaw = (os.getenv("PMX_INBOX_NOTIFY_OPENCLAW") or "").strip() != "0" and bool(
                (os.getenv("OPENCLAW_TO") or "").strip()
            )
        if notify_openclaw:
            _maybe_notify_openclaw(summary, config=config)

    return 0


def cmd_send(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    to = (args.to or "").strip()
    if not to:
        print("[inbox_workflow] Missing --to", file=sys.stderr)
        return 2

    recipients = [p.strip() for p in to.replace(";", ",").split(",") if p.strip()]
    if not recipients:
        print("[inbox_workflow] Missing recipients", file=sys.stderr)
        return 2

    subject = (args.subject or "").strip()
    body = args.body if args.body is not None else ""
    if args.body_file:
        body = Path(args.body_file).read_text(encoding="utf-8")
    if not body:
        body = _read_stdin()
    body = (body or "").rstrip()
    if not body:
        print("[inbox_workflow] Missing body (use --body/--body-file or pipe stdin)", file=sys.stderr)
        return 2

    try:
        send_email(
            project_root=PROJECT_ROOT,
            config=config,
            account_name=args.account,
            to_addresses=recipients,
            subject=subject,
            body=body,
            timeout_seconds=float(args.timeout_seconds),
            in_reply_to=(args.in_reply_to or "").strip(),
            references=(args.references or "").strip(),
        )
    except PermissionError as exc:
        print(f"[inbox_workflow] Send blocked: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[inbox_workflow] Send failed: {exc}", file=sys.stderr)
        return 1

    print("[inbox_workflow] OK (sent)")
    return 0


def cmd_fetch(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    account = (args.account or "").strip()
    uid = (args.uid or "").strip()
    seq = (args.seq or "").strip()
    mailbox = (args.mailbox or "").strip()
    if not mailbox:
        mailbox = "INBOX"

    if not uid and not seq:
        print("[inbox_workflow] Provide --uid or --seq", file=sys.stderr)
        return 2

    try:
        eml = fetch_message_eml(
            project_root=PROJECT_ROOT,
            config=config,
            account_name=account,
            uid=uid,
            seq=seq,
            mailbox=mailbox,
            timeout_seconds=float(args.timeout_seconds),
        )
    except Exception as exc:
        print(f"[inbox_workflow] Fetch failed: {exc}", file=sys.stderr)
        return 1

    if args.stdout:
        try:
            sys.stdout.buffer.write(eml)
            return 0
        except Exception as exc:
            print(f"[inbox_workflow] Failed to write to stdout: {exc}", file=sys.stderr)
            return 1

    out_path = Path(args.output) if args.output else None
    if out_path is None:
        safe_id = uid or f"seq_{seq}"
        out_path = PROJECT_ROOT / "logs" / "inbox" / account / f"message_{safe_id}.eml"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(eml)
    print(str(out_path))
    return 0


def main(argv: Sequence[str]) -> int:
    # Load `.env` safely (best-effort) without printing or overwriting existing env vars.
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Inbox workflows (IMAP read, SMTP send).")
    parser.add_argument(
        "--config",
        default="config/inbox_workflows.yml",
        help="Path to inbox workflows config (default: config/inbox_workflows.yml).",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-accounts", help="List configured accounts.")
    p_list.set_defaults(_handler="list_accounts")

    p_scan = sub.add_parser("scan", help="Scan inbox for matching messages (default: UNSEEN).")
    p_scan.add_argument("--account", default="", help="Scan only a specific account (default: all enabled).")
    p_scan.add_argument("--json", action="store_true", help="Print JSON output.")
    p_scan.add_argument("--no-write", action="store_true", help="Do not write scan artifacts under logs/inbox/.")
    p_scan.add_argument(
        "--notify-openclaw",
        dest="notify_openclaw",
        action="store_true",
        help="Force sending scan summary via OpenClaw.",
    )
    p_scan.add_argument(
        "--no-notify-openclaw",
        dest="notify_openclaw",
        action="store_false",
        help="Disable OpenClaw notification for this run.",
    )
    p_scan.set_defaults(notify_openclaw=None)
    p_scan.set_defaults(_handler="scan")

    p_send = sub.add_parser("send", help="Send an email (disabled by default).")
    p_send.add_argument("--account", default="gmail", help="Account to send from (default: gmail).")
    p_send.add_argument("--to", default="", help="Comma-separated recipients.")
    p_send.add_argument("--subject", default="", help="Subject.")
    p_send.add_argument("--body", default=None, help="Body. If omitted, reads from --body-file or stdin.")
    p_send.add_argument("--body-file", default="", help="Path to a UTF-8 text file used as the body.")
    p_send.add_argument("--timeout-seconds", default="20", help="SMTP timeout seconds (default: 20).")
    p_send.add_argument("--in-reply-to", default="", help="Optional In-Reply-To header value.")
    p_send.add_argument("--references", default="", help="Optional References header value.")
    p_send.set_defaults(_handler="send")

    p_fetch = sub.add_parser("fetch", help="Fetch a full message as .eml (read-only).")
    p_fetch.add_argument("--account", default="gmail", help="Account to fetch from (default: gmail).")
    p_fetch.add_argument("--uid", default="", help="IMAP UID to fetch (preferred).")
    p_fetch.add_argument("--seq", default="", help="IMAP sequence number to fetch.")
    p_fetch.add_argument("--mailbox", default="", help="Mailbox name (default: INBOX).")
    p_fetch.add_argument("--timeout-seconds", default="20", help="IMAP timeout seconds (default: 20).")
    p_fetch.add_argument("--output", default="", help="Write to this path (default: logs/inbox/<account>/...).")
    p_fetch.add_argument("--stdout", action="store_true", help="Write raw EML bytes to stdout.")
    p_fetch.set_defaults(_handler="fetch")

    args = parser.parse_args(list(argv))
    config = load_inbox_config(project_root=PROJECT_ROOT, config_path=args.config)

    if args._handler == "list_accounts":
        return cmd_list_accounts(config)
    if args._handler == "scan":
        return cmd_scan(args, config)
    if args._handler == "send":
        return cmd_send(args, config)
    if args._handler == "fetch":
        return cmd_fetch(args, config)

    print("[inbox_workflow] Unknown command.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
