#!/usr/bin/env python3
"""
Send a notification via OpenClaw CLI.

This is a small wrapper around:
- `openclaw message send ...` (plain notifications)
- `openclaw agent ...` (prompt an agent; optionally deliver the reply)

If `OPENCLAW_TARGETS`/`OPENCLAW_TO` is not set, the script can (optionally)
infer a WhatsApp "message yourself" target from `openclaw status --json`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.openclaw_cli import send_message, send_message_multi  # noqa: E402


def _read_stdin() -> str:
    try:
        if sys.stdin is None or sys.stdin.closed or sys.stdin.isatty():
            return ""
        return sys.stdin.read()
    except Exception:
        return ""


def _mask_target(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"[+][0-9]{6,15}", text):
        # Keep country prefix-ish + first digits and last 4 for debugging.
        if len(text) <= 8:
            return text
        return text[:5] + "***" + text[-4:]
    if ":" in text:
        prefix = text.split(":", 1)[0]
        return prefix + ":***"
    return "***"


def _redact_text(text: str) -> str:
    payload = text or ""
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    for name, value in os.environ.items():
        if not value or len(value) < 8:
            continue
        upper = name.upper()
        if any(marker in upper for marker in secret_markers):
            payload = payload.replace(value, "[REDACTED]")
    payload = re.sub(r"(Bearer\\s+)[A-Za-z0-9\\-\\._~\\+/]+=*", r"\\1[REDACTED]", payload)
    return payload


def _write_run_log(*, mode: str, to: str, channel: str | None, message_len: int, result) -> None:
    """Best-effort run logging (never blocks the main action)."""
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = (PROJECT_ROOT / "logs" / "openclaw_notify").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_mode = re.sub(r"[^a-zA-Z0-9_-]+", "_", mode or "run").strip("_") or "run"
        status = "ok" if bool(getattr(result, "ok", False)) else "fail"
        path = out_dir / f"openclaw_notify_{stamp}_{safe_mode}_{status}.json"

        raw_cmd = list(getattr(result, "command", []) or [])
        cmd = list(raw_cmd)
        # Prevent logging full targets/messages in command args.
        redact_next_as_target = {"--target", "-t", "--to", "--reply-to"}
        redact_next_as_message = {"--message", "-m"}
        redact_next_as_media = {"--media"}
        for i, arg in enumerate(list(cmd)):
            if i + 1 >= len(cmd):
                continue
            if arg in redact_next_as_target:
                cmd[i + 1] = _mask_target(cmd[i + 1])
            elif arg in redact_next_as_message:
                cmd[i + 1] = f"<redacted len={int(message_len)}>"
            elif arg in redact_next_as_media:
                leaf = Path(cmd[i + 1]).name if cmd[i + 1] else ""
                cmd[i + 1] = f"<redacted media={leaf or 'file'}>"

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "to_masked": _mask_target(to),
            "channel": channel or "",
            "message_len": int(message_len),
            "ok": bool(getattr(result, "ok", False)),
            "returncode": int(getattr(result, "returncode", -1)),
            "command": cmd,
            "stdout": _redact_text(str(getattr(result, "stdout", "") or "")),
            "stderr": _redact_text(str(getattr(result, "stderr", "") or "")),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


def main() -> int:
    # Load `.env` safely (best-effort) without printing or overwriting existing env vars.
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        pass

    # Sentinel used to detect `--flag` provided without a value. This happens
    # commonly in PowerShell for targets like `@mychat` (splatting) unless quoted.
    _MISSING_VALUE = "__PMX_MISSING_VALUE__"

    def _env_float(name: str, fallback: float) -> float:
        try:
            return float(os.getenv(name, str(fallback)))
        except Exception:
            return fallback

    parser = argparse.ArgumentParser(description="Send a notification via OpenClaw.")
    parser.add_argument(
        "--to",
        nargs="?",
        const=_MISSING_VALUE,
        default=os.getenv("OPENCLAW_TO", ""),
        help=(
            "OpenClaw target (e.g., +15551234567, discord:..., slack:...). "
            "In PowerShell, quote targets like '@mychat' (or use 'telegram:@mychat'). "
            "Can also be set via OPENCLAW_TO."
        ),
    )
    parser.add_argument(
        "--targets",
        default=os.getenv("OPENCLAW_TARGETS", ""),
        help=(
            "(Send mode) Comma-separated target list. Items may be 'channel:target' "
            '(e.g. "whatsapp:+1555..., telegram:@mychat"). Can also be set via OPENCLAW_TARGETS.'
        ),
    )
    parser.add_argument(
        "--channel",
        default=os.getenv("OPENCLAW_CHANNEL", ""),
        help="OpenClaw channel (e.g., whatsapp, telegram, discord). Can also be set via OPENCLAW_CHANNEL.",
    )
    parser.add_argument(
        "--message",
        default="",
        help="Message text. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--media",
        default="",
        help="(Send mode) Optional media attachment (path or URL).",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Run an OpenClaw agent turn instead of sending a plain notification (uses `openclaw agent`).",
    )
    parser.add_argument(
        "--deliver",
        dest="deliver",
        action="store_true",
        default=None,
        help="(Prompt mode) Deliver the agent's reply back to the selected channel/target.",
    )
    parser.add_argument(
        "--no-deliver",
        dest="deliver",
        action="store_false",
        help="(Prompt mode) Do not deliver; only run the agent turn.",
    )
    parser.add_argument(
        "--agent-id",
        default="",
        help="(Prompt mode) Optional OpenClaw agent id to route to (maps to `openclaw agent --agent`).",
    )
    parser.add_argument(
        "--thinking",
        default="",
        help="(Prompt mode) Thinking level: off|minimal|low|medium|high (maps to `openclaw agent --thinking`).",
    )
    parser.add_argument(
        "--local-agent",
        action="store_true",
        help="(Prompt mode) Run embedded agent locally (`openclaw agent --local`). Requires model provider keys.",
    )
    parser.add_argument(
        "--reply-channel",
        default=os.getenv("OPENCLAW_REPLY_CHANNEL", ""),
        help="(Prompt mode) Reply channel override (maps to `openclaw agent --reply-channel`).",
    )
    parser.add_argument(
        "--reply-to",
        nargs="?",
        const=_MISSING_VALUE,
        default=os.getenv("OPENCLAW_REPLY_TO", ""),
        help=(
            "(Prompt mode) Reply target override (maps to `openclaw agent --reply-to`). "
            "In PowerShell, quote targets like '@mychat'."
        ),
    )
    parser.add_argument(
        "--reply-account",
        default=os.getenv("OPENCLAW_REPLY_ACCOUNT", ""),
        help="(Prompt mode) Reply account override (maps to `openclaw agent --reply-account`).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="(Send mode) Print payload and skip sending (passes `--dry-run` to OpenClaw CLI).",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="(Send mode) Send silently (Telegram + Discord).",
    )
    parser.add_argument(
        "--command",
        default=os.getenv("OPENCLAW_COMMAND", "openclaw"),
        help='OpenClaw command (default: "openclaw"). Use "wsl openclaw" on Windows if needed.',
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=_env_float("OPENCLAW_TIMEOUT_SECONDS", 20.0),
        help="(Send mode) Command timeout in seconds (default: 20).",
    )
    parser.add_argument(
        "--agent-timeout-seconds",
        type=float,
        default=_env_float("OPENCLAW_AGENT_TIMEOUT_SECONDS", 600.0),
        help="(Prompt mode) Command timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--infer-to",
        dest="infer_to",
        action="store_true",
        default=True,
        help="If --to/OPENCLAW_TO is missing, try to infer a WhatsApp self-target from `openclaw status --json`.",
    )
    parser.add_argument(
        "--no-infer-to",
        dest="infer_to",
        action="store_false",
        help="Disable inferring a default target from OpenClaw status.",
    )
    parser.add_argument(
        "--prompt-to",
        dest="prompt_to",
        action="store_true",
        default=None,
        help="If target is missing, prompt interactively (only when stdin is a TTY).",
    )
    parser.add_argument(
        "--no-prompt-to",
        dest="prompt_to",
        action="store_false",
        help="Never prompt interactively for a missing target.",
    )
    args = parser.parse_args()

    if args.to == _MISSING_VALUE:
        print(
            "[openclaw_notify] --to was provided without a value.\n"
            "[openclaw_notify] PowerShell treats @name as splatting unless quoted.\n"
            "[openclaw_notify] Fix: --to '@mychat'  (or: --to telegram:@mychat)",
            file=sys.stderr,
        )
        return 2
    if args.reply_to == _MISSING_VALUE:
        print(
            "[openclaw_notify] --reply-to was provided without a value.\n"
            "[openclaw_notify] PowerShell treats @name as splatting unless quoted.\n"
            "[openclaw_notify] Fix: --reply-to '@mychat'  (or: --reply-to telegram:@mychat)",
            file=sys.stderr,
        )
        return 2

    media = (args.media or "").strip() or None

    message = args.message if args.message else _read_stdin()
    message = (message or "").strip()
    if not message and not media:
        print("[openclaw_notify] Missing --message/--media (and stdin was empty).", file=sys.stderr)
        return 2

    # On Windows, `echo` is a cmd.exe builtin, not an executable. Support the
    # common "dry-run" pattern `--command echo ...` by wrapping it.
    command = args.command
    if os.name == "nt" and (command or "").strip().lower() == "echo":
        command = "cmd /c echo"

    channel = (args.channel or "").strip() or None

    targets: list[tuple[str | None, str]] = []

    raw_targets = (args.targets or "").strip()
    if raw_targets and not bool(args.prompt):
        try:
            from utils.openclaw_cli import parse_openclaw_targets

            targets = parse_openclaw_targets(raw_targets, default_channel=channel)
        except Exception:
            targets = []

    to = (args.to or "").strip()
    if not targets and not bool(args.prompt):
        # Allow OPENCLAW_TO / --to to be a comma-separated list as well.
        if any(sep in to for sep in [",", ";", "\n"]):
            try:
                from utils.openclaw_cli import parse_openclaw_targets

                targets = parse_openclaw_targets(to, default_channel=channel)
                to = ""
            except Exception:
                targets = []

    if not targets:
        if not to and bool(args.infer_to):
            try:
                from utils.openclaw_cli import infer_linked_whatsapp_target

                inferred = infer_linked_whatsapp_target(command=command, cwd=PROJECT_ROOT, timeout_seconds=10.0)
                if inferred:
                    to = inferred
            except Exception:
                pass

        prompt_to = args.prompt_to if args.prompt_to is not None else bool(sys.stdin.isatty())
        if not to and prompt_to:
            try:
                to = input("OpenClaw target (e.g. +15551234567): ").strip()
            except Exception:
                to = ""

        if not to:
            hint = (
                "[openclaw_notify] Missing target. "
                "Set OPENCLAW_TARGETS/OPENCLAW_TO in .env, pass --targets/--to, or enable inference/prompting."
            )
            print(hint, file=sys.stderr)
            return 2

    # Default to WhatsApp when the target looks like an E.164 phone number.
    # (OpenClaw can route this without an explicit --channel, but being explicit
    # makes behavior less surprising when multiple channels are configured.)
    if channel is None and re.fullmatch(r"[+][0-9]{6,15}", to or ""):
        channel = "whatsapp"

    if bool(args.prompt) and bool(args.dry_run):
        print("[openclaw_notify] --dry-run is only supported in send mode (omit --prompt).", file=sys.stderr)
        return 2

    if bool(args.prompt) and media:
        print("[openclaw_notify] --media is only supported in send mode (omit --prompt).", file=sys.stderr)
        return 2

    if bool(args.prompt):
        try:
            from utils.openclaw_cli import run_agent_turn

            deliver = bool(args.deliver) if args.deliver is not None else True
            agent_id = (args.agent_id or "").strip() or None
            thinking = (args.thinking or "").strip() or None
            reply_channel = (args.reply_channel or "").strip() or None
            reply_to = (args.reply_to or "").strip() or None
            reply_account = (args.reply_account or "").strip() or None
            result = run_agent_turn(
                to=to,
                message=message,
                command=command,
                cwd=PROJECT_ROOT,
                timeout_seconds=float(args.agent_timeout_seconds),
                deliver=deliver,
                channel=channel,
                reply_channel=reply_channel,
                reply_to=reply_to,
                reply_account=reply_account,
                agent_id=agent_id,
                thinking=thinking,
                local=bool(args.local_agent),
            )
        except Exception as exc:
            print(f"[openclaw_notify] FAILED (agent mode): {exc}", file=sys.stderr)
            return 1
    else:
        extra_args = ["--dry-run"] if bool(args.dry_run) else None
        if not targets:
            targets = [(channel, to)]

        results = send_message_multi(
            targets=targets,
            message=message,
            media=media,
            command=command,
            cwd=PROJECT_ROOT,
            timeout_seconds=args.timeout_seconds,
            extra_args=extra_args,
            silent=bool(args.silent),
        )
        for (ch, tgt), result in zip(targets, results):
            _write_run_log(
                mode="message",
                to=tgt,
                channel=ch,
                message_len=len(message),
                result=result,
            )

        if all(r.ok for r in results):
            print("[openclaw_notify] OK")
            return 0

        # Report first failing result for readability.
        result = next((r for r in results if not r.ok), results[0])
        combined = ((result.stderr or "") + "\n" + (result.stdout or "")).lower()
        gateway_unreachable = (
            "econnrefused" in combined
            or "connect refused" in combined
            or "connection refused" in combined
            or "gateway closed" in combined
            or "connect econnrefused" in combined
        )
        missing_whatsapp_listener = "no active whatsapp web listener" in combined
        whatsapp_dns_failure = ("enotfound" in combined or "getaddrinfo" in combined) and "web.whatsapp.com" in combined
        if gateway_unreachable:
            print(
                "[openclaw_notify] Hint: OpenClaw Gateway may not be running/reachable. "
                "Try `openclaw gateway restart` then retry.",
                file=sys.stderr,
            )
        if missing_whatsapp_listener:
            print(
                "[openclaw_notify] Hint: WhatsApp listener is not active. "
                "Run `python scripts/openclaw_maintenance.py --apply --primary-channel whatsapp` "
                "or relink with `openclaw channels login --channel whatsapp --account default --verbose`.",
                file=sys.stderr,
            )
        if whatsapp_dns_failure:
            print(
                "[openclaw_notify] Hint: DNS resolution failed for web.whatsapp.com. "
                "Check network/firewall/proxy settings, then restart gateway.",
                file=sys.stderr,
            )

        if ("telegram" in combined and "not configured" in combined) or "telegram token not configured" in combined:
            print(
                "[openclaw_notify] Hint: Telegram channel isn't configured. "
                "Set TELEGRAM_BOT_TOKEN in .env, then run "
                "`python scripts/openclaw_env.py channels add --channel telegram --use-env` "
                "and `openclaw gateway restart`.",
                file=sys.stderr,
            )
        if ("discord" in combined and "not configured" in combined) or "discord token not configured" in combined:
            has_discord_app = all(
                bool((os.getenv(k) or "").strip())
                for k in (
                    "DISCORD_APP_NAME",
                    "DISCORD_APPLICATION_ID",
                    "DISCORD_PUBLIC_KEY",
                    "DISCORD_APP_INSTALL_LINK",
                )
            )
            has_interactions_key = bool((os.getenv("INTERACTIONS_API_KEY") or "").strip())
            if has_discord_app and has_interactions_key:
                print(
                    "[openclaw_notify] Hint: Discord app credentials are present for interaction-mode flows, "
                    "but OpenClaw channel messaging still requires DISCORD_BOT_TOKEN. "
                    "Set DISCORD_BOT_TOKEN for channel send, or keep using interactions endpoints.",
                    file=sys.stderr,
                )
            print(
                "[openclaw_notify] Hint: Discord channel isn't configured. "
                "Set DISCORD_BOT_TOKEN in .env, then run "
                "`python scripts/openclaw_env.py channels add --channel discord --use-env` "
                "and `openclaw gateway restart`.",
                file=sys.stderr,
            )

        print(f"[openclaw_notify] FAILED (exit={result.returncode})", file=sys.stderr)
        stderr_tail = "\n".join((result.stderr or "").splitlines()[-20:])
        stdout_tail = "\n".join((result.stdout or "").splitlines()[-20:])
        if stderr_tail:
            print("[openclaw_notify] stderr (tail):", file=sys.stderr)
            print(stderr_tail, file=sys.stderr)
        if stdout_tail:
            print("[openclaw_notify] stdout (tail):", file=sys.stderr)
            print(stdout_tail, file=sys.stderr)
        return 1

    _write_run_log(
        mode="agent",
        to=to,
        channel=channel,
        message_len=len(message),
        result=result,
    )

    if result.ok:
        print("[openclaw_notify] OK")
        return 0

    # Keep errors readable (OpenClaw can emit a lot of output).
    stderr_tail = "\n".join((result.stderr or "").splitlines()[-20:])
    stdout_tail = "\n".join((result.stdout or "").splitlines()[-20:])
    print(f"[openclaw_notify] FAILED (exit={result.returncode})", file=sys.stderr)
    if bool(args.prompt):
        combined = ((result.stderr or "") + "\n" + (result.stdout or "")).lower()
        missing_provider = "no api key found for provider" in combined
        gateway_unreachable = (
            "econnrefused" in combined
            or "connect refused" in combined
            or "connection refused" in combined
            or "gateway closed" in combined
            or "connect econnrefused" in combined
        )
        missing_whatsapp_listener = "no active whatsapp web listener" in combined
        whatsapp_dns_failure = ("enotfound" in combined or "getaddrinfo" in combined) and "web.whatsapp.com" in combined
        if gateway_unreachable:
            print(
                "[openclaw_notify] Hint: OpenClaw Gateway may not be running/reachable. "
                "Try `openclaw gateway restart` (or run `openclaw onboard`) then retry.",
                file=sys.stderr,
            )
        if missing_whatsapp_listener:
            print(
                "[openclaw_notify] Hint: WhatsApp listener is not active. "
                "Run `python scripts/openclaw_maintenance.py --apply --primary-channel whatsapp` "
                "or relink with `openclaw channels login --channel whatsapp --account default --verbose`.",
                file=sys.stderr,
            )
        if whatsapp_dns_failure:
            print(
                "[openclaw_notify] Hint: DNS resolution failed for web.whatsapp.com. "
                "Check network/firewall/proxy settings, then restart gateway.",
                file=sys.stderr,
            )
        if missing_provider:
            print(
                "[openclaw_notify] Hint: OpenClaw agent could not find model provider credentials. "
                "Run `openclaw configure` (or `openclaw onboard`) to set up providers, then retry. "
                "If you keep keys in repo .env, run `python scripts/openclaw_models.py sync-auth --restart-gateway`.",
                file=sys.stderr,
            )
    if stderr_tail:
        print("[openclaw_notify] stderr (tail):", file=sys.stderr)
        print(stderr_tail, file=sys.stderr)
    if stdout_tail:
        print("[openclaw_notify] stdout (tail):", file=sys.stderr)
        print(stdout_tail, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
