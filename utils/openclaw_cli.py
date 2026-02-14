"""
OpenClaw CLI integration helpers.

This module is intentionally tiny: it shells out to the `openclaw` CLI (or a
user-supplied wrapper like `wsl openclaw`) and returns structured results.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class OpenClawResult:
    ok: bool
    returncode: int
    command: list[str]
    stdout: str
    stderr: str


def _split_command(command: str) -> list[str]:
    command = (command or "").strip()
    if not command:
        return ["openclaw"]
    try:
        # On Windows, posix=False handles quoted paths more predictably.
        return shlex.split(command, posix=(os.name != "nt"))
    except ValueError:
        return command.split()


def build_message_send_command(*, command: str, to: str, message: str) -> list[str]:
    base = _split_command(command)
    # `to` is an OpenClaw target string (phone number, channel id, etc).
    return [*base, "message", "send", "--to", str(to), "--message", str(message)]


def send_message(
    *,
    to: str,
    message: str,
    command: str = "openclaw",
    cwd: Optional[Path] = None,
    timeout_seconds: float = 20.0,
    extra_args: Optional[Sequence[str]] = None,
) -> OpenClawResult:
    cmd = build_message_send_command(command=command, to=to, message=message)
    if extra_args:
        cmd.extend([str(arg) for arg in extra_args])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
        return OpenClawResult(
            ok=proc.returncode == 0,
            returncode=int(proc.returncode),
            command=cmd,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )
    except FileNotFoundError as exc:
        return OpenClawResult(
            ok=False,
            returncode=127,
            command=cmd,
            stdout="",
            stderr=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return OpenClawResult(
            ok=False,
            returncode=124,
            command=cmd,
            stdout=stdout,
            stderr=stderr or f"OpenClaw command timed out after {timeout_seconds}s",
        )

