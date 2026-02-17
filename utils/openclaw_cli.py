"""
OpenClaw CLI integration helpers.

This module is intentionally tiny: it shells out to the `openclaw` CLI (or a
user-supplied wrapper like `wsl openclaw`) and returns structured results.

Includes:
- Retry with exponential backoff for transient failures
- Rate limiting (token bucket) to prevent message flooding
- Message deduplication to prevent auto-reply loops
- Session error detection for prekey bundle issues
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class OpenClawResult:
    ok: bool
    returncode: int
    command: list[str]
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# Rate limiter (token bucket)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Thread-safe token-bucket rate limiter for outbound messages."""

    def __init__(self, max_per_minute: int = 10, burst: int = 3):
        self._max_per_minute = max(1, max_per_minute)
        self._burst = max(1, burst)
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 5.0) -> bool:
        """Try to acquire a send token. Returns False if rate-limited."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._burst,
                    self._tokens + elapsed * (self._max_per_minute / 60.0),
                )
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.2)


# Global rate limiter: 10 messages/minute, burst of 3
_rate_limiter = _RateLimiter(
    max_per_minute=int(os.getenv("OPENCLAW_RATE_LIMIT_PER_MINUTE", "10")),
    burst=int(os.getenv("OPENCLAW_RATE_LIMIT_BURST", "3")),
)


# ---------------------------------------------------------------------------
# Message deduplication (prevents auto-reply loops)
# ---------------------------------------------------------------------------

class _MessageDeduplicator:
    """Tracks recently sent messages to prevent loops and duplicate sends."""

    def __init__(self, window_seconds: float = 30.0, max_entries: int = 100):
        self._window = window_seconds
        self._max = max_entries
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def _fingerprint(self, to: str, message: str) -> str:
        raw = f"{to}:{message[:200]}".encode("utf-8", errors="replace")
        return hashlib.md5(raw).hexdigest()

    def is_duplicate(self, to: str, message: str) -> bool:
        fp = self._fingerprint(to, message)
        now = time.monotonic()
        with self._lock:
            # Prune expired entries
            expired = [k for k, t in self._seen.items() if now - t > self._window]
            for k in expired:
                del self._seen[k]
            # Prune if too large
            if len(self._seen) > self._max:
                oldest = sorted(self._seen.items(), key=lambda x: x[1])
                for k, _ in oldest[: len(self._seen) - self._max // 2]:
                    del self._seen[k]
            if fp in self._seen:
                return True
            self._seen[fp] = now
            return False


_deduplicator = _MessageDeduplicator(
    window_seconds=float(os.getenv("OPENCLAW_DEDUP_WINDOW_SECONDS", "30")),
)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _is_retryable_error(result: OpenClawResult) -> bool:
    """Check if the error is transient and worth retrying."""
    combined = ((result.stderr or "") + " " + (result.stdout or "")).lower()
    retryable_patterns = [
        "econnrefused",
        "connection refused",
        "gateway closed",
        "timeout",
        "timed out",
        "bad mac",
        "session error",
        "socket hang up",
        "econnreset",
        "network error",
    ]
    # Do NOT retry: auth errors, bad requests, missing config
    non_retryable = [
        "does not support tools",
        "api key",
        "not configured",
        "unknown option",
        "missing required",
    ]
    if any(p in combined for p in non_retryable):
        return False
    return any(p in combined for p in retryable_patterns)


def _is_session_error(result: OpenClawResult) -> bool:
    """Detect WhatsApp session/prekey bundle errors."""
    combined = ((result.stderr or "") + " " + (result.stdout or "")).lower()
    return any(p in combined for p in [
        "prekey bundle",
        "closed session",
        "bad mac",
        "session error",
    ])


# E.164 style phone number, e.g. +15551234567
_E164_IN_TEXT_RE = re.compile(r"([+][0-9]{6,15})")
_E164_EXACT_RE = re.compile(r"[+][0-9]{6,15}$")

# Allowed channel prefixes for `channel:target` parsing.
# Keep this in sync (loosely) with `openclaw agent --help` channel list.
_KNOWN_CHANNEL_PREFIXES = {
    "last",
    "telegram",
    "whatsapp",
    "discord",
    "irc",
    "googlechat",
    "slack",
    "signal",
    "imessage",
    "feishu",
    "nostr",
    "msteams",
    "mattermost",
    "nextcloud-talk",
    "matrix",
    "bluebubbles",
    "line",
    "zalo",
    "zalouser",
    "tlon",
}


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


def build_message_send_command(
    *,
    command: str,
    to: str,
    message: Optional[str] = None,
    media: Optional[str] = None,
    channel: Optional[str] = None,
    silent: bool = False,
) -> list[str]:
    base = _split_command(command)
    # `to` is an OpenClaw target string (phone number, channel id, etc).
    # Newer OpenClaw CLIs use `--target`. We'll fall back to `--to` in send_message
    # when targeting older CLIs.
    cmd = [*base, "message", "send"]
    if channel:
        cmd.extend(["--channel", str(channel)])
    if silent:
        cmd.append("--silent")
    cmd.extend(["--target", str(to)])
    msg = (message or "").strip()
    if msg:
        cmd.extend(["--message", str(message)])
    if media:
        cmd.extend(["--media", str(media)])
    return cmd


def infer_linked_whatsapp_target(
    *,
    command: str = "openclaw",
    cwd: Optional[Path] = None,
    timeout_seconds: float = 10.0,
) -> Optional[str]:
    """
    Infer a reasonable default OpenClaw target for WhatsApp, if possible.

    This uses `openclaw status --json` and extracts the linked WhatsApp account
    E.164 number. That enables a "message yourself" default without requiring
    OPENCLAW_TO to be configured.
    """

    base = _split_command(command)
    cmd = [*base, "--no-color", "status", "--json"]
    try:
        env = dict(os.environ)
        env.setdefault("NODE_NO_WARNINGS", "1")
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            env=env,
        )
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    raw = (proc.stdout or "").strip()
    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except Exception:
        # Sometimes CLIs can accidentally interleave logs with JSON; best-effort
        # to salvage by extracting the first JSON object.
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(raw[start : end + 1])
            except Exception:
                return None
        else:
            return None

    if not isinstance(payload, dict):
        return None

    link = payload.get("linkChannel")
    whatsapp_linked = bool(isinstance(link, dict) and str(link.get("id") or "").lower() == "whatsapp" and link.get("linked"))

    channel_summary = payload.get("channelSummary")
    if isinstance(channel_summary, list):
        for entry in channel_summary:
            if not isinstance(entry, str):
                continue
            text = entry.strip()
            low = text.lower()
            if "whatsapp" not in low or "linked" not in low:
                continue
            match = _E164_IN_TEXT_RE.search(text)
            if match:
                return match.group(1)

    # If WhatsApp is linked but the number wasn't in channelSummary, do a narrow
    # scan over the JSON string.
    if whatsapp_linked:
        try:
            blob = json.dumps(payload, ensure_ascii=True)
        except Exception:
            blob = raw
        match = _E164_IN_TEXT_RE.search(blob or "")
        if match:
            return match.group(1)

    return None


def send_message(
    *,
    to: str,
    message: str,
    media: Optional[str] = None,
    command: str = "openclaw",
    cwd: Optional[Path] = None,
    timeout_seconds: float = 20.0,
    extra_args: Optional[Sequence[str]] = None,
    channel: Optional[str] = None,
    silent: bool = False,
    max_retries: int = 2,
    skip_dedup: bool = False,
    skip_rate_limit: bool = False,
) -> OpenClawResult:
    cmd = build_message_send_command(command=command, to=to, message=message, media=media, channel=channel, silent=silent)
    if extra_args:
        cmd.extend([str(arg) for arg in extra_args])

    # --- Deduplication guard ---
    if not skip_dedup and not any(a == "--dry-run" for a in (extra_args or [])):
        if _deduplicator.is_duplicate(to, message or ""):
            return OpenClawResult(
                ok=True,
                returncode=0,
                command=cmd,
                stdout="[dedup] Duplicate message suppressed (same target+content within window)",
                stderr="",
            )

    # --- Rate limiting guard ---
    if not skip_rate_limit and not any(a == "--dry-run" for a in (extra_args or [])):
        if not _rate_limiter.acquire(timeout=3.0):
            return OpenClawResult(
                ok=False,
                returncode=429,
                command=cmd,
                stdout="",
                stderr="Rate limited: too many messages in short period. Wait and retry.",
            )

    def _is_unknown_option(output: str, flag: str) -> bool:
        text = (output or "").lower()
        needle = (flag or "").strip().lower()
        if not needle:
            return False
        return "unknown option" in text and needle in text

    def _try_send(send_cmd: list[str]) -> OpenClawResult:
        try:
            if not (message or "").strip() and not (media or "").strip():
                return OpenClawResult(
                    ok=False,
                    returncode=2,
                    command=send_cmd,
                    stdout="",
                    stderr="Missing message/media (OpenClaw requires --message unless --media is set).",
                )
            env = dict(os.environ)
            env.setdefault("NODE_NO_WARNINGS", "1")
            proc = subprocess.run(
                send_cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
                env=env,
            )
            if proc.returncode != 0 and _is_unknown_option((proc.stderr or proc.stdout), "--target"):
                legacy = list(send_cmd)
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
                    env=env,
                )
                send_cmd = legacy
            return OpenClawResult(
                ok=proc.returncode == 0,
                returncode=int(proc.returncode),
                command=send_cmd,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
            )
        except FileNotFoundError as exc:
            return OpenClawResult(
                ok=False,
                returncode=127,
                command=send_cmd,
                stdout="",
                stderr=str(exc),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return OpenClawResult(
                ok=False,
                returncode=124,
                command=send_cmd,
                stdout=stdout,
                stderr=stderr or f"OpenClaw command timed out after {timeout_seconds}s",
            )

    # --- Retry loop with exponential backoff ---
    last_result = _try_send(cmd)
    if last_result.ok:
        return last_result

    for attempt in range(max_retries):
        if not _is_retryable_error(last_result):
            break
        delay = min(2.0 * (2 ** attempt), 10.0)  # 2s, 4s, capped at 10s
        time.sleep(delay)
        last_result = _try_send(cmd)
        if last_result.ok:
            return last_result

    # --- Session error detection ---
    if _is_session_error(last_result):
        last_result = OpenClawResult(
            ok=last_result.ok,
            returncode=last_result.returncode,
            command=last_result.command,
            stdout=last_result.stdout,
            stderr=(
                last_result.stderr
                + "\n[PMX] Session/prekey error detected. "
                "This is typically caused by WhatsApp Web re-keying. "
                "Try: openclaw gateway restart"
            ),
        )

    return last_result


def build_agent_turn_command(
    *,
    command: str,
    to: str,
    message: str,
    deliver: bool = False,
    channel: Optional[str] = None,
    reply_channel: Optional[str] = None,
    reply_to: Optional[str] = None,
    reply_account: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    thinking: Optional[str] = None,
    local: bool = False,
    json_output: bool = False,
) -> list[str]:
    base = _split_command(command)
    cmd: list[str] = [*base, "agent"]
    if channel:
        cmd.extend(["--channel", str(channel)])
    if deliver:
        cmd.append("--deliver")
    if local:
        cmd.append("--local")
    if agent_id:
        cmd.extend(["--agent", str(agent_id)])
    if session_id:
        cmd.extend(["--session-id", str(session_id)])
    cmd.extend(["--to", str(to), "--message", str(message)])
    if reply_account:
        cmd.extend(["--reply-account", str(reply_account)])
    if reply_channel:
        cmd.extend(["--reply-channel", str(reply_channel)])
    if reply_to:
        cmd.extend(["--reply-to", str(reply_to)])
    if thinking:
        cmd.extend(["--thinking", str(thinking)])
    if json_output:
        cmd.append("--json")
    return cmd


def run_agent_turn(
    *,
    to: str,
    message: str,
    command: str = "openclaw",
    cwd: Optional[Path] = None,
    timeout_seconds: float = 600.0,
    deliver: bool = False,
    channel: Optional[str] = None,
    reply_channel: Optional[str] = None,
    reply_to: Optional[str] = None,
    reply_account: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    thinking: Optional[str] = None,
    local: bool = False,
    json_output: bool = False,
    max_retries: int = 1,
) -> OpenClawResult:
    cmd = build_agent_turn_command(
        command=command,
        to=to,
        message=message,
        deliver=deliver,
        channel=channel,
        reply_channel=reply_channel,
        reply_to=reply_to,
        reply_account=reply_account,
        agent_id=agent_id,
        session_id=session_id,
        thinking=thinking,
        local=local,
        json_output=json_output,
    )

    def _try_run() -> OpenClawResult:
        try:
            env = dict(os.environ)
            env.setdefault("NODE_NO_WARNINGS", "1")
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
                env=env,
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

    last_result = _try_run()
    if last_result.ok:
        return last_result

    for attempt in range(max_retries):
        if not _is_retryable_error(last_result):
            break
        delay = min(3.0 * (2 ** attempt), 15.0)
        time.sleep(delay)
        last_result = _try_run()
        if last_result.ok:
            return last_result

    return last_result


def parse_openclaw_targets(raw: str, *, default_channel: Optional[str] = None) -> list[tuple[Optional[str], str]]:
    """
    Parse OpenClaw targets from a comma/semicolon/newline-separated string.

    Supported forms:
    - +15551234567                 (E.164; implies whatsapp)
    - whatsapp:+15551234567        (explicit channel)
    - telegram:@my_channel         (explicit channel)
    - discord:channel:1234567890   (explicit channel; target may contain colons)

    If a target has no explicit channel prefix:
    - E.164 implies channel="whatsapp"
    - otherwise, uses default_channel when provided (else leaves channel=None)
    """

    text = (raw or "").strip()
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\n", ",").replace(";", ",")
    parts = [p.strip() for p in normalized.split(",") if p and p.strip()]

    out: list[tuple[Optional[str], str]] = []
    for part in parts:
        channel: Optional[str] = None
        target = part

        if ":" in part:
            prefix, rest = part.split(":", 1)
            prefix_norm = prefix.strip().lower()
            if prefix_norm in _KNOWN_CHANNEL_PREFIXES:
                channel = prefix_norm
                target = rest.strip()

        if not target:
            continue

        if channel is None and _E164_EXACT_RE.fullmatch(target):
            channel = "whatsapp"
        if channel is None:
            dc = (default_channel or "").strip()
            channel = dc or None

        out.append((channel, target))

    return out


def _coerce_targets(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        # Support YAML configs that choose `to: [ ... ]`.
        items = [str(x).strip() for x in value if str(x).strip()]
        return ", ".join(items)
    return str(value)


def resolve_openclaw_targets(
    *,
    env_targets: Optional[str] = None,
    env_to: Optional[str] = None,
    cfg_to=None,
    default_channel: Optional[str] = None,
) -> list[tuple[Optional[str], str]]:
    """
    Resolve target specs from env/config, returning a parsed list.

    Precedence:
    - env_targets (OPENCLAW_TARGETS)
    - env_to (OPENCLAW_TO)
    - cfg_to (alerts.openclaw.to)
    """

    raw = (env_targets or "").strip() or (env_to or "").strip() or _coerce_targets(cfg_to).strip()
    return parse_openclaw_targets(raw, default_channel=default_channel)


def send_message_multi(
    *,
    targets: Iterable[tuple[Optional[str], str]],
    message: str,
    media: Optional[str] = None,
    command: str = "openclaw",
    cwd: Optional[Path] = None,
    timeout_seconds: float = 20.0,
    extra_args: Optional[Sequence[str]] = None,
    silent: bool = False,
) -> list[OpenClawResult]:
    results: list[OpenClawResult] = []
    for channel, to in targets:
        results.append(
            send_message(
                to=to,
                message=message,
                media=media,
                command=command,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                extra_args=extra_args,
                channel=channel,
                silent=silent,
            )
        )
    return results
