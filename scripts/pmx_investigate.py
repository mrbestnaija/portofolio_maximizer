#!/usr/bin/env python3
"""
Repo investigation helper (result-oriented).

Use cases:
- Quickly search the codebase for a symbol/error and get a compact summary.
- Optionally notify yourself via OpenClaw (WhatsApp/Telegram/Discord/etc).

Security:
- Never prints `.env` values.
- Redacts likely secret env var values before sending text to external surfaces.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _read_stdin() -> str:
    try:
        if sys.stdin is None or sys.stdin.closed or sys.stdin.isatty():
            return ""
        return sys.stdin.read()
    except Exception:
        return ""


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


def _clamp(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    t = text or ""
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 10)] + "\n...[truncated]"


def _run_rg(
    *,
    query: str,
    root: Path,
    max_hits: int,
    glob: Optional[str],
    fixed: bool,
    ignore_case: bool,
    context: int,
) -> tuple[int, list[str], str]:
    """
    Returns: (exit_code, hit_lines, raw_stderr_tail)
    - exit_code 0 => hits found
    - exit_code 1 => no hits
    - exit_code 2 => error
    """

    exe = shutil.which("rg")
    if not exe:
        return 2, [], "rg not found on PATH"

    cmd: list[str] = [exe, "--line-number", "--column", "--with-filename", "--no-heading"]
    if ignore_case:
        cmd.append("-i")
    if fixed:
        cmd.append("-F")
    if context and int(context) > 0:
        cmd.extend(["-C", str(int(context))])
    if glob:
        cmd.extend(["--glob", str(glob)])
    cmd.extend(["--", query, str(root)])

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

    # rg: 0 = matches, 1 = no matches, 2 = error
    raw = (proc.stdout or "").splitlines()
    hits: list[str] = []
    for line in raw:
        hits.append(line.rstrip("\n"))
        if max_hits > 0 and len(hits) >= max_hits:
            break
    stderr_tail = "\n".join((proc.stderr or "").splitlines()[-20:])
    return int(proc.returncode), hits, stderr_tail


def _render_summary(*, query: str, root: Path, hits: list[str], max_chars: int) -> str:
    header = f"[pmx_investigate] query={query!r} root={str(root)}"
    if not hits:
        return header + "\n(no matches)"

    lines = [header, f"matches (showing up to {len(hits)}):"]
    for h in hits:
        # Avoid huge lines in notifications.
        lines.append(_clamp(h, 400))
    return _clamp("\n".join(lines), max_chars)


def _resolve_targets_from_args(args) -> list[tuple[Optional[str], str]]:
    default_channel = (os.getenv("OPENCLAW_CHANNEL") or "").strip() or None

    try:
        from utils.openclaw_cli import infer_linked_whatsapp_target, parse_openclaw_targets, resolve_openclaw_targets
    except Exception:
        return []

    # Highest precedence: explicit flags
    if (args.targets or "").strip():
        return parse_openclaw_targets((args.targets or "").strip(), default_channel=default_channel)
    if (args.to or "").strip():
        return parse_openclaw_targets((args.to or "").strip(), default_channel=default_channel)

    env_targets = (os.getenv("OPENCLAW_TARGETS") or "").strip()
    env_to = (os.getenv("OPENCLAW_TO") or "").strip()
    targets = resolve_openclaw_targets(env_targets=env_targets, env_to=env_to, cfg_to=None, default_channel=default_channel)
    if targets:
        return targets

    if bool(args.infer_to):
        inferred = infer_linked_whatsapp_target(command=str(args.command or "openclaw"), cwd=PROJECT_ROOT, timeout_seconds=10.0)
        if inferred:
            return [(None, inferred)]

    return []


def main(argv: list[str]) -> int:
    # Load `.env` safely (best-effort) without printing or overwriting existing env vars.
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Search the repo (rg) and optionally notify results via OpenClaw.")
    p.add_argument("--query", default="", help="Search query (regex by default). If omitted, reads from stdin.")
    p.add_argument("--root", default=str(PROJECT_ROOT), help="Root directory to search (default: repo root).")
    p.add_argument("--glob", default="", help="Optional rg --glob pattern (e.g. '*.py').")
    p.add_argument("--fixed", action="store_true", help="Treat query as a fixed string (-F).")
    p.add_argument("--ignore-case", action="store_true", help="Case-insensitive search (-i).")
    p.add_argument("--context", type=int, default=0, help="Context lines (-C).")
    p.add_argument("--max-hits", type=int, default=50, help="Max lines to include in output (default: 50).")
    p.add_argument("--max-notify-chars", type=int, default=3500, help="Max characters to send via OpenClaw.")
    p.add_argument("--notify", action="store_true", help="Send summary via OpenClaw after search.")

    p.add_argument("--to", nargs="?", default="", help="OpenClaw target or comma-separated targets (optional).")
    p.add_argument("--targets", default="", help="OpenClaw multi-target string (optional).")
    p.add_argument("--infer-to", dest="infer_to", action="store_true", default=True, help="Infer WhatsApp self target if none configured.")
    p.add_argument("--no-infer-to", dest="infer_to", action="store_false", help="Disable inference fallback.")
    p.add_argument("--command", default=os.getenv("OPENCLAW_COMMAND", "openclaw"), help='OpenClaw command (default: "openclaw").')
    p.add_argument("--timeout-seconds", type=float, default=float(os.getenv("OPENCLAW_TIMEOUT_SECONDS", "20") or 20), help="OpenClaw send timeout.")

    args = p.parse_args(argv)

    query = (args.query or "").strip() or (_read_stdin().strip())
    if not query:
        print("[pmx_investigate] Missing --query (and stdin was empty).", file=sys.stderr)
        return 2

    root = Path(args.root).expanduser().resolve()
    glob = (args.glob or "").strip() or None

    code, hits, err_tail = _run_rg(
        query=query,
        root=root,
        max_hits=int(args.max_hits),
        glob=glob,
        fixed=bool(args.fixed),
        ignore_case=bool(args.ignore_case),
        context=int(args.context),
    )

    summary = _render_summary(query=query, root=root, hits=hits, max_chars=int(args.max_notify_chars))
    print(summary)
    if code == 2 and err_tail:
        print("[pmx_investigate] rg error (tail):", file=sys.stderr)
        print(err_tail, file=sys.stderr)

    if not bool(args.notify):
        return 0 if code in {0, 1} else 1

    targets = _resolve_targets_from_args(args)
    if not targets:
        print("[pmx_investigate] No OpenClaw targets configured (set OPENCLAW_TARGETS/OPENCLAW_TO or pass --targets/--to).", file=sys.stderr)
        return 2

    try:
        from utils.openclaw_cli import send_message_multi

        results = send_message_multi(
            targets=targets,
            message=_redact_text(summary),
            command=str(args.command or "openclaw"),
            cwd=PROJECT_ROOT,
            timeout_seconds=float(args.timeout_seconds),
        )
        ok = all(r.ok for r in results)
        if ok:
            print("[pmx_investigate] OpenClaw notify OK")
            return 0 if code in {0, 1} else 1
        bad = next((r for r in results if not r.ok), results[0])
        print(f"[pmx_investigate] OpenClaw notify FAILED (exit={bad.returncode})", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[pmx_investigate] OpenClaw notify FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

