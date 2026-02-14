#!/usr/bin/env python3
"""
Run the OpenClaw CLI with repo `.env` loaded.

Why:
- OpenClaw itself does not automatically read this repo's `.env`.
- Many PMX scripts already call `etl.secret_loader.bootstrap_dotenv()`.
- This helper gives the OpenClaw CLI the same `.env` context when you run it via Python.

Security:
- Never prints `.env` values.
- Does not overwrite already-set environment variables.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str]) -> int:
    # Load `.env` safely (best-effort) without printing or overwriting existing env vars.
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Run the OpenClaw CLI with repo `.env` loaded.",
        add_help=True,
    )
    parser.add_argument(
        "--command",
        default=os.getenv("OPENCLAW_COMMAND", "openclaw"),
        help='OpenClaw command (default: "openclaw"). Use "wsl openclaw" on Windows if needed.',
    )
    parser.add_argument(
        "openclaw_args",
        nargs=argparse.REMAINDER,
        help='Arguments passed to OpenClaw, e.g. `status --json` or `message send --target ... --message ...`.',
    )
    args = parser.parse_args(argv)

    raw = list(args.openclaw_args or [])
    if raw and raw[0] == "--":
        raw = raw[1:]

    if not raw:
        parser.print_help(sys.stderr)
        return 2

    use_no_color = "--json" in raw and "--no-color" not in raw

    try:
        from utils.openclaw_cli import _split_command  # type: ignore

        base = _split_command(args.command)
    except Exception:
        base = [args.command]

    cmd = [*base, *(["--no-color"] if use_no_color else []), *raw]

    env = dict(os.environ)
    env.setdefault("NODE_NO_WARNINGS", "1")

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
