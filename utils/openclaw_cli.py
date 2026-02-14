"""
OpenClaw CLI integration helpers.

This module is intentionally tiny: it shells out to the `openclaw` CLI (or a
user-supplied wrapper like `wsl openclaw`) and returns structured results.
"""

from __future__ import annotations

import os
import shlex
import shutil
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
        parts = ["openclaw"]
        return _wrap_windows_command(parts)
    try:
        # On Windows, posix=False handles quoted paths more predictably.
        parts = shlex.split(command, posix=(os.name != "nt"))
    except ValueError:
        parts = command.split()
    return _wrap_windows_command(parts)


def _wrap_windows_command(parts: list[str]) -> list[str]:
    """
    On Windows, npm-installed CLIs are frequently `.cmd` shims which cannot be
    executed directly via CreateProcess. Wrap in `cmd /c` when needed.
    """
    if os.name != "nt":
        return parts

    if not parts:
        return ["cmd", "/d", "/s", "/c", "openclaw"]

    prog = (parts[0] or "").strip()
    if not prog:
        return ["cmd", "/d", "/s", "/c", "openclaw"]

    prog_lower = prog.lower()
    if prog_lower in {"cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        return parts

    # If the entrypoint is a .cmd/.bat shim (typical for npm global bins),
    # CreateProcess cannot execute it directly; `cmd /c` can.
    needs_cmd = prog_lower.endswith((".cmd", ".bat"))
    if not needs_cmd:
        resolved = shutil.which(prog)
        if resolved and resolved.lower().endswith((".cmd", ".bat")):
            needs_cmd = True

    if needs_cmd:
        return ["cmd", "/d", "/s", "/c", *parts]

    return parts


def build_message_send_command(*, command: str, to: str, message: str) -> list[str]:
    base = _split_command(command)
    # `to` is an OpenClaw target string (phone number, channel id, etc).
    # Newer OpenClaw CLIs use `--target`. We'll fall back to `--to` in send_message
    # when targeting older CLIs.
    return [*base, "message", "send", "--target", str(to), "--message", str(message)]


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

    def _is_unknown_option(output: str, flag: str) -> bool:
        text = (output or "").lower()
        needle = (flag or "").strip().lower()
        if not needle:
            return False
        return "unknown option" in text and needle in text

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
        )
        if proc.returncode != 0 and _is_unknown_option((proc.stderr or proc.stdout), "--target"):
            legacy = list(cmd)
            try:
                idx = legacy.index("--target")
                legacy[idx] = "--to"
            except ValueError:
                legacy = [arg if arg != "--target" else "--to" for arg in legacy]

            proc = subprocess.run(
                legacy,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
            )
            cmd = legacy
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
