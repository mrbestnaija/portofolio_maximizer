#!/usr/bin/env python3
"""
Send a notification via OpenClaw CLI.

This is a small wrapper around `openclaw message send ...` so Portfolio Maximizer
automation can optionally deliver alerts through a user's OpenClaw gateway.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.openclaw_cli import send_message  # noqa: E402


def _read_stdin() -> str:
    try:
        return sys.stdin.read()
    except Exception:
        return ""


def main() -> int:
    # Load `.env` safely (best-effort) without printing or overwriting existing env vars.
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        pass

    def _env_float(name: str, fallback: float) -> float:
        try:
            return float(os.getenv(name, str(fallback)))
        except Exception:
            return fallback

    parser = argparse.ArgumentParser(description="Send a notification via OpenClaw.")
    parser.add_argument(
        "--to",
        default=os.getenv("OPENCLAW_TO", ""),
        help="OpenClaw target (e.g., +15551234567, discord:..., slack:...). Can also be set via OPENCLAW_TO.",
    )
    parser.add_argument(
        "--message",
        default="",
        help="Message text. If omitted, reads from stdin.",
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
        help="Command timeout in seconds (default: 20).",
    )
    args = parser.parse_args()

    to = (args.to or "").strip()
    if not to:
        print("[openclaw_notify] Missing --to (or OPENCLAW_TO).", file=sys.stderr)
        return 2

    message = args.message if args.message else _read_stdin()
    message = (message or "").strip()
    if not message:
        print("[openclaw_notify] Missing --message (and stdin was empty).", file=sys.stderr)
        return 2

    # On Windows, `echo` is a cmd.exe builtin, not an executable. Support the
    # common "dry-run" pattern `--command echo ...` by wrapping it.
    command = args.command
    if os.name == "nt" and (command or "").strip().lower() == "echo":
        command = "cmd /c echo"

    result = send_message(
        to=to,
        message=message,
        command=command,
        cwd=PROJECT_ROOT,
        timeout_seconds=args.timeout_seconds,
    )

    if result.ok:
        print("[openclaw_notify] OK")
        return 0

    # Keep errors readable (OpenClaw can emit a lot of output).
    stderr_tail = "\n".join((result.stderr or "").splitlines()[-20:])
    stdout_tail = "\n".join((result.stdout or "").splitlines()[-20:])
    print(f"[openclaw_notify] FAILED (exit={result.returncode})", file=sys.stderr)
    if stderr_tail:
        print("[openclaw_notify] stderr (tail):", file=sys.stderr)
        print(stderr_tail, file=sys.stderr)
    if stdout_tail:
        print("[openclaw_notify] stdout (tail):", file=sys.stderr)
        print(stdout_tail, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
