#!/usr/bin/env python3
"""
OpenClaw WhatsApp search-routing end-to-end validator.

Purpose:
- Validate robust web-search routing via PMX bridge over WhatsApp.
- Confirm provider fallback search works.
- Confirm bridge fast-path events are emitted in llm activity logs.

This script is designed for operational verification, not unit test mocking.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_LOG_DIR = PROJECT_ROOT / "logs" / "llm_activity"

FAST_PATH_WEB_START = "bridge_web_fast_path_start"
FAST_PATH_WEB_COMPLETE = "bridge_web_fast_path_complete"
FAST_PATH_STATUS_START = "bridge_status_fast_path_start"
FAST_PATH_STATUS_COMPLETE = "bridge_status_fast_path_complete"


@dataclass(frozen=True)
class _CmdResult:
    ok: bool
    returncode: int
    command: list[str]
    stdout: str
    stderr: str
    duration_seconds: float


def _split_command(command: str) -> list[str]:
    raw = str(command or "").strip()
    if not raw:
        return ["openclaw"]
    try:
        parts = shlex.split(raw, posix=(os.name != "nt"))
    except Exception:
        parts = raw.split()
    if os.name == "nt" and len(parts) == 1:
        return ["cmd", "/d", "/s", "/c", parts[0]]
    return parts


def _run(cmd: list[str], *, timeout_seconds: float) -> _CmdResult:
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(3.0, float(timeout_seconds)),
            check=False,
        )
    except FileNotFoundError as exc:
        return _CmdResult(False, 127, cmd, "", str(exc), round(time.monotonic() - started, 3))
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        return _CmdResult(False, 124, cmd, out, err or "timeout", round(time.monotonic() - started, 3))

    return _CmdResult(
        ok=int(proc.returncode) == 0,
        returncode=int(proc.returncode),
        command=cmd,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        duration_seconds=round(time.monotonic() - started, 3),
    )


def _parse_json_best_effort(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty output")
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("invalid json output")


def _parse_iso8601(ts: str) -> Optional[datetime]:
    text = str(ts or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _channel_ready(payload: dict[str, Any], channel: str) -> tuple[bool, str]:
    channels = payload.get("channels") if isinstance(payload.get("channels"), dict) else {}
    row = channels.get(channel) if isinstance(channels, dict) else None
    if not isinstance(row, dict):
        return False, "channel_missing"
    if not bool(row.get("configured")):
        return False, "channel_not_configured"
    if channel == "whatsapp" and not bool(row.get("linked")):
        return False, "whatsapp_not_linked"
    if not bool(row.get("running")):
        return False, "channel_not_running"
    if not bool(row.get("connected", True)):
        return False, "channel_not_connected"

    accounts = payload.get("channelAccounts") if isinstance(payload.get("channelAccounts"), dict) else {}
    rows = accounts.get(channel) if isinstance(accounts, dict) else None
    if isinstance(rows, list):
        enabled = [r for r in rows if isinstance(r, dict) and bool(r.get("enabled", True))]
        if enabled and not any(bool(r.get("running")) and bool(r.get("connected", True)) for r in enabled):
            return False, "enabled_account_not_running"

    return True, "ok"


def _extract_whatsapp_reply_target(payload: dict[str, Any]) -> Optional[str]:
    channels = payload.get("channels") if isinstance(payload.get("channels"), dict) else {}
    wa = channels.get("whatsapp") if isinstance(channels, dict) else None
    if isinstance(wa, dict):
        self_row = wa.get("self")
        if isinstance(self_row, dict):
            e164 = str(self_row.get("e164") or "").strip()
            if e164:
                return e164

    accounts = payload.get("channelAccounts") if isinstance(payload.get("channelAccounts"), dict) else {}
    wa_accounts = accounts.get("whatsapp") if isinstance(accounts, dict) else None
    if isinstance(wa_accounts, list):
        for row in wa_accounts:
            if not isinstance(row, dict):
                continue
            allow_from = row.get("allowFrom")
            if isinstance(allow_from, list):
                for target in allow_from:
                    v = str(target or "").strip()
                    if v.startswith("+"):
                        return v
    return None


def _bridge_output_passed(raw_stdout: str) -> bool:
    low = str(raw_stdout or "").lower()
    return "web search: pass" in low


def _load_openclaw_events_since(*, since_ts: datetime, channel: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not ACTIVITY_LOG_DIR.exists():
        return rows

    for path in sorted(ACTIVITY_LOG_DIR.glob("*.jsonl"))[-4:]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") != "openclaw_event":
                continue
            if str(item.get("channel") or "").strip().lower() != channel:
                continue
            ts = _parse_iso8601(str(item.get("timestamp") or ""))
            if ts is None:
                continue
            if ts < since_ts:
                continue
            rows.append(item)
    return rows


def _fast_path_event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "web_start": 0,
        "web_complete": 0,
        "status_start": 0,
        "status_complete": 0,
    }
    for item in events:
        event_type = str(item.get("event_type") or "")
        if event_type == FAST_PATH_WEB_START:
            counts["web_start"] += 1
        elif event_type == FAST_PATH_WEB_COMPLETE:
            counts["web_complete"] += 1
        elif event_type == FAST_PATH_STATUS_START:
            counts["status_start"] += 1
        elif event_type == FAST_PATH_STATUS_COMPLETE:
            counts["status_complete"] += 1
    return counts


def run_e2e(
    *,
    openclaw_command: str,
    python_bin: str,
    channel: str,
    reply_to: Optional[str],
    probe_query: str,
    probe_providers: str,
    bridge_prompt: str,
    timeout_seconds: float,
    wait_seconds: float,
    restart_gateway: bool,
    require_web_events: bool,
    allow_legacy_status_events: bool,
) -> tuple[bool, dict[str, Any]]:
    report: dict[str, Any] = {
        "status": "FAIL",
        "channel": channel,
        "steps": {},
        "errors": [],
        "warnings": [],
    }

    oc_cmd = _split_command(openclaw_command)
    if restart_gateway:
        restart_res = _run([*oc_cmd, "gateway", "restart"], timeout_seconds=timeout_seconds)
        report["steps"]["gateway_restart"] = {
            "ok": restart_res.ok,
            "returncode": restart_res.returncode,
            "duration_seconds": restart_res.duration_seconds,
            "stdout_tail": "\n".join(restart_res.stdout.splitlines()[-5:]),
            "stderr_tail": "\n".join(restart_res.stderr.splitlines()[-5:]),
        }
        if not restart_res.ok:
            report["errors"].append("gateway_restart_failed")
            return False, report
        if wait_seconds > 0:
            time.sleep(max(0.0, wait_seconds))

    status_res = _run(
        [*oc_cmd, "--no-color", "channels", "status", "--probe", "--json"],
        timeout_seconds=timeout_seconds,
    )
    report["steps"]["channels_probe"] = {
        "ok": status_res.ok,
        "returncode": status_res.returncode,
        "duration_seconds": status_res.duration_seconds,
    }
    if not status_res.ok:
        report["errors"].append("channels_probe_failed")
        report["steps"]["channels_probe"]["stderr_tail"] = "\n".join(status_res.stderr.splitlines()[-12:])
        return False, report

    try:
        status_payload = _parse_json_best_effort(status_res.stdout)
    except Exception:
        report["errors"].append("channels_probe_invalid_json")
        report["steps"]["channels_probe"]["stdout_tail"] = "\n".join(status_res.stdout.splitlines()[-12:])
        return False, report
    if not isinstance(status_payload, dict):
        report["errors"].append("channels_probe_invalid_payload")
        return False, report

    ready, ready_reason = _channel_ready(status_payload, channel)
    report["steps"]["primary_channel"] = {"ok": ready, "reason": ready_reason}
    if not ready:
        report["errors"].append(f"primary_channel_not_ready:{ready_reason}")
        return False, report

    probe_res = _run(
        [
            str(python_bin),
            str(PROJECT_ROOT / "scripts" / "tavily_search.py"),
            "--query",
            str(probe_query),
            "--providers",
            str(probe_providers),
            "--json",
        ],
        timeout_seconds=max(timeout_seconds, 30.0),
    )
    report["steps"]["provider_probe"] = {
        "ok": probe_res.ok,
        "returncode": probe_res.returncode,
        "duration_seconds": probe_res.duration_seconds,
    }
    if not probe_res.ok:
        report["errors"].append("provider_probe_failed")
        report["steps"]["provider_probe"]["stderr_tail"] = "\n".join(probe_res.stderr.splitlines()[-12:])
        return False, report

    try:
        probe_payload = _parse_json_best_effort(probe_res.stdout)
    except Exception:
        report["errors"].append("provider_probe_invalid_json")
        report["steps"]["provider_probe"]["stdout_tail"] = "\n".join(probe_res.stdout.splitlines()[-12:])
        return False, report
    if not isinstance(probe_payload, dict):
        report["errors"].append("provider_probe_invalid_payload")
        return False, report
    report["steps"]["provider_probe"]["provider"] = str(probe_payload.get("provider") or "none")
    report["steps"]["provider_probe"]["probe_ok"] = bool(probe_payload.get("ok"))
    if not bool(probe_payload.get("ok")):
        report["errors"].append(f"provider_probe_not_ok:{probe_payload.get('error') or 'unknown'}")
        return False, report

    resolved_reply_to = str(reply_to or "").strip() or _extract_whatsapp_reply_target(status_payload)
    if not resolved_reply_to:
        report["errors"].append("missing_reply_target")
        return False, report
    report["steps"]["reply_target"] = {"value": resolved_reply_to}

    bridge_start_ts = datetime.now(timezone.utc)
    bridge_res = _run(
        [
            str(python_bin),
            str(PROJECT_ROOT / "scripts" / "llm_multi_model_orchestrator.py"),
            "openclaw-bridge",
            "--channel",
            str(channel),
            "--reply-to",
            str(resolved_reply_to),
            "--message",
            str(bridge_prompt),
        ],
        timeout_seconds=max(timeout_seconds, 60.0),
    )
    report["steps"]["bridge_call"] = {
        "ok": bridge_res.ok,
        "returncode": bridge_res.returncode,
        "duration_seconds": bridge_res.duration_seconds,
        "stdout_tail": "\n".join(bridge_res.stdout.splitlines()[-12:]),
        "stderr_tail": "\n".join(bridge_res.stderr.splitlines()[-12:]),
    }
    if not bridge_res.ok:
        report["errors"].append("bridge_call_failed")
        return False, report

    bridge_passed = _bridge_output_passed(bridge_res.stdout)
    report["steps"]["bridge_call"]["bridge_output_passed"] = bridge_passed
    if not bridge_passed:
        report["errors"].append("bridge_output_not_pass")
        return False, report

    events = _load_openclaw_events_since(since_ts=bridge_start_ts, channel=channel)
    counts = _fast_path_event_counts(events)
    report["steps"]["activity_events"] = {
        "events_after_bridge_start": len(events),
        "fast_path_counts": counts,
    }

    web_events_ok = counts["web_start"] > 0 and counts["web_complete"] > 0
    legacy_events_ok = counts["status_start"] > 0 and counts["status_complete"] > 0

    if require_web_events and not web_events_ok:
        if allow_legacy_status_events and legacy_events_ok:
            report["warnings"].append("web_fast_path_events_missing_using_legacy_status_events")
        else:
            report["errors"].append("web_fast_path_events_missing")
            return False, report
    elif not web_events_ok and legacy_events_ok:
        report["warnings"].append("using_legacy_status_fast_path_events")
    elif not web_events_ok and not legacy_events_ok:
        report["warnings"].append("no_fast_path_events_found_post_bridge_call")

    report["status"] = "PASS" if not report["warnings"] else "WARN"
    return True, report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        default=os.getenv("OPENCLAW_COMMAND", "openclaw"),
        help="OpenClaw command (default OPENCLAW_COMMAND or openclaw).",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter for PMX scripts.",
    )
    parser.add_argument(
        "--channel",
        default=os.getenv("OPENCLAW_CHANNEL", "whatsapp"),
        help="Channel for bridge validation (default: whatsapp).",
    )
    parser.add_argument(
        "--reply-to",
        default=(os.getenv("OPENCLAW_REPLY_TO") or "").strip(),
        help="Explicit reply target; when omitted for WhatsApp, target is auto-derived.",
    )
    parser.add_argument(
        "--probe-query",
        default="OpenAI API docs",
        help="Query for provider fallback probe.",
    )
    parser.add_argument(
        "--probe-providers",
        default="duckduckgo,wikipedia",
        help="Provider order for probe query.",
    )
    parser.add_argument(
        "--bridge-prompt",
        default="web search latest OpenClaw docs",
        help="Prompt sent to openclaw bridge for fast-path validation.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Timeout budget per command.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=5.0,
        help="Wait after gateway restart.",
    )
    parser.add_argument(
        "--no-restart-gateway",
        action="store_true",
        help="Skip gateway restart step.",
    )
    parser.add_argument(
        "--require-web-events",
        action="store_true",
        help="Require bridge_web_fast_path_start/complete events after the bridge call.",
    )
    parser.add_argument(
        "--allow-legacy-status-events",
        action="store_true",
        help="Accept legacy bridge_status_fast_path events when web events are absent.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report.",
    )
    args = parser.parse_args(argv)

    ok, report = run_e2e(
        openclaw_command=str(args.command),
        python_bin=str(args.python_bin),
        channel=str(args.channel).strip().lower() or "whatsapp",
        reply_to=str(args.reply_to or "").strip() or None,
        probe_query=str(args.probe_query),
        probe_providers=str(args.probe_providers),
        bridge_prompt=str(args.bridge_prompt),
        timeout_seconds=float(args.timeout_seconds),
        wait_seconds=float(args.wait_seconds),
        restart_gateway=not bool(args.no_restart_gateway),
        require_web_events=bool(args.require_web_events),
        allow_legacy_status_events=bool(args.allow_legacy_status_events),
    )

    if bool(args.json):
        print(json.dumps(report, indent=2))
    else:
        print(
            f"[openclaw_search_routing_e2e] status={report.get('status')} "
            f"errors={len(report.get('errors', []))} warnings={len(report.get('warnings', []))}"
        )
        for row in report.get("errors", []):
            print(f"[openclaw_search_routing_e2e] error: {row}")
        for row in report.get("warnings", []):
            print(f"[openclaw_search_routing_e2e] warning: {row}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
