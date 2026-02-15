"""
Secrets / credential leak guard.

Primary goals:
- Prevent accidentally committing or pushing credentials (PATs, API keys, passwords, private keys).
- Never print secret values. Output is metadata only (rule + file + location).
- Provide a safe wrapper to run commands with stdout/stderr redaction for agent workflows.

Typical usage:
  python tools/secrets_guard.py scan --staged
  python tools/secrets_guard.py scan --tracked --strict
  python tools/secrets_guard.py run -- git remote -v
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Finding:
    severity: str  # "ERROR" | "WARN"
    check: str  # "diff" | "tracked" | "remote" | "paths"
    rule: str
    path: Optional[str]
    message: str


# High-confidence secret patterns. Keep this list conservative to avoid noise.
_SECRET_REGEX_RULES: List[Tuple[str, str, re.Pattern[str]]] = [
    # GitHub PATs (classic + fine-grained). Avoid committing even "fake-looking" strings; they trigger scanners.
    ("github_pat_classic", "ERROR", re.compile(r"gh[pousr]_[A-Za-z0-9]{36}")),
    ("github_pat_fine", "ERROR", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    # Anthropic API keys.
    ("anthropic_key", "ERROR", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    # Slack tokens.
    ("slack_token", "ERROR", re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}")),
    # Google API key.
    ("google_api_key", "ERROR", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
    # AWS access key id.
    ("aws_access_key_id", "ERROR", re.compile(r"AKIA[0-9A-Z]{16}")),
    # Private key blocks (PEM/OpenSSH/PGP).
    # Anchor to whole line to avoid flagging source code/docs that mention the marker.
    ("private_key_block", "ERROR", re.compile(r"^\s*-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----\s*$", re.MULTILINE)),
    ("pgp_private_key_block", "ERROR", re.compile(r"^\s*-----BEGIN PGP PRIVATE KEY BLOCK-----\s*$", re.MULTILINE)),
]

_SENSITIVE_ENV_NAME = re.compile(r"(?:^|_)(?:KEY|TOKEN|SECRET|PASSWORD)(?:$|_)", re.IGNORECASE)

# Files we should never see staged/tracked (with a few explicit exceptions).
_SECRET_PATH_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)\.env\.[^/]+$"),
    re.compile(r"(^|/)secrets/"),
    re.compile(r"(^|/)config/secrets/"),
    re.compile(r"(^|/)credentials/"),
    re.compile(r"(^|/)keys/"),
    re.compile(r"(^|/)\.secrets/"),
    re.compile(r"\.(?:key|pem|p12)$", re.IGNORECASE),
]

_SECRET_PATH_ALLOWLIST: List[re.Pattern[str]] = [
    re.compile(r"(^|/)\.env\.template$"),
]

_SKIP_TRACKED_SCAN_EXTS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".zst",
    ".bin",
    ".exe",
    ".dll",
}


def _git(args: Sequence[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _looks_like_placeholder(value: str) -> bool:
    v = value.strip().strip('"').strip("'").strip()
    if not v:
        return True
    lowered = v.lower()
    if lowered in {"null", "none"}:
        return True
    if v.startswith("${") and v.endswith("}"):
        return True
    placeholder_markers = (
        "your_",
        "your-",
        "example",
        "changeme",
        "replace_me",
        "redacted",
        "***",
        "<",
        ">",
        "dummy",
        "placeholder",
        "tbd",
        "todo",
        "test",
    )
    return any(m in lowered for m in placeholder_markers)


def _mask_url_userinfo(url: str) -> str:
    """
    Mask any https(s) userinfo (token@ or user:token@) without changing host/path.
    """
    m = re.match(r"^(https?://)([^/]+)(/.*)?$", url.strip())
    if not m:
        return url.strip()
    scheme, authority, rest = m.group(1), m.group(2), m.group(3) or ""
    if "@" not in authority:
        return url.strip()
    host = authority.split("@", 1)[1]
    return f"{scheme}***@{host}{rest}"


def _is_secret_path(path: str) -> bool:
    p = path.replace("\\", "/").lstrip("./")
    if any(rx.search(p) for rx in _SECRET_PATH_ALLOWLIST):
        return False
    return any(rx.search(p) for rx in _SECRET_PATH_PATTERNS)


def _parse_added_lines_from_diff(diff_text: str) -> List[Tuple[Optional[str], str]]:
    """
    Returns list of (path, added_line_without_plus).
    Path is best-effort from +++ b/<path> headers.
    """
    out: List[Tuple[Optional[str], str]] = []
    current_path: Optional[str] = None
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            # Example: +++ b/foo/bar.py
            if raw.startswith("+++ b/"):
                current_path = raw[len("+++ b/") :].strip()
            else:
                current_path = None
            continue

        if not raw.startswith("+") or raw.startswith("+++"):
            continue

        out.append((current_path, raw[1:]))
    return out


def _scan_line_for_secrets(path: Optional[str], line: str) -> List[Finding]:
    findings: List[Finding] = []
    for rule_id, severity, rx in _SECRET_REGEX_RULES:
        if rx.search(line):
            findings.append(
                Finding(
                    severity=severity,
                    check="diff",
                    rule=rule_id,
                    path=path,
                    message="Matched high-confidence secret pattern in added content.",
                )
            )

    # Heuristic: env/config assignment of sensitive variables to a non-placeholder value.
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
    if m:
        key = m.group(1)
        value = m.group(2)
        # Restrict to env-style variables (ALL_CAPS) to avoid flagging normal code like `key = ...`.
        if key.isupper() and _SENSITIVE_ENV_NAME.search(key) and not _looks_like_placeholder(value):
            sev = "ERROR" if (path or "").replace("\\", "/").endswith(".env.template") else "WARN"
            findings.append(
                Finding(
                    severity=sev,
                    check="diff",
                    rule="suspicious_assignment",
                    path=path,
                    message=f"Non-placeholder assignment to sensitive variable '{key}'.",
                )
            )

    return findings


def scan_staged_diff() -> List[Finding]:
    proc = _git(["diff", "--cached", "--no-color", "--no-ext-diff", "-U0"])
    if proc.returncode != 0:
        return [
            Finding(
                severity="WARN",
                check="diff",
                rule="git_diff_failed",
                path=None,
                message="Failed to read staged diff; skipping diff-based scanning.",
            )
        ]

    findings: List[Finding] = []

    # Block secret paths being staged at all.
    staged_files = _git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"]).stdout.splitlines()
    for p in staged_files:
        if _is_secret_path(p):
            findings.append(
                Finding(
                    severity="ERROR",
                    check="paths",
                    rule="secret_path_staged",
                    path=p,
                    message="Secret-like file path is staged (should not be committed).",
                )
            )

    for path, added in _parse_added_lines_from_diff(proc.stdout):
        findings.extend(_scan_line_for_secrets(path, added))

    return findings


def scan_worktree_diff() -> List[Finding]:
    proc = _git(["diff", "--no-color", "--no-ext-diff", "-U0"])
    if proc.returncode != 0:
        return [
            Finding(
                severity="WARN",
                check="diff",
                rule="git_diff_failed",
                path=None,
                message="Failed to read worktree diff; skipping diff-based scanning.",
            )
        ]

    findings: List[Finding] = []
    for path, added in _parse_added_lines_from_diff(proc.stdout):
        findings.extend(_scan_line_for_secrets(path, added))
    return findings


def scan_remotes() -> List[Finding]:
    proc = _git(["remote", "-v"])
    if proc.returncode != 0:
        return [
            Finding(
                severity="WARN",
                check="remote",
                rule="git_remote_failed",
                path=None,
                message="Failed to read git remotes; skipping remote URL scanning.",
            )
        ]

    findings: List[Finding] = []
    for raw in proc.stdout.splitlines():
        parts = raw.strip().split()
        if len(parts) < 2:
            continue
        name, url = parts[0], parts[1]

        # Only flag embedded credentials in https(s) remotes; ssh userinfo is normal.
        if re.search(r"^https?://[^/]*@", url):
            findings.append(
                Finding(
                    severity="ERROR",
                    check="remote",
                    rule="remote_url_contains_userinfo",
                    path=name,
                    message=f"Remote URL contains userinfo; sanitize remote (e.g. {_mask_url_userinfo(url)}).",
                )
            )

        # Also scan URL string for high-confidence tokens.
        for rule_id, severity, rx in _SECRET_REGEX_RULES:
            if rx.search(url):
                findings.append(
                    Finding(
                        severity="ERROR",
                        check="remote",
                        rule=f"remote_{rule_id}",
                        path=name,
                        message="Remote URL matched a high-confidence secret pattern; sanitize remote.",
                    )
                )

    return findings


def scan_tracked_files() -> List[Finding]:
    proc = _git(["ls-files"])
    if proc.returncode != 0:
        return [
            Finding(
                severity="WARN",
                check="tracked",
                rule="git_ls_files_failed",
                path=None,
                message="Failed to list tracked files; skipping tracked scan.",
            )
        ]

    findings: List[Finding] = []
    for rel in proc.stdout.splitlines():
        if not rel:
            continue
        rel_norm = rel.replace("\\", "/")

        # Hard fail if obvious secret paths are tracked.
        if _is_secret_path(rel_norm):
            findings.append(
                Finding(
                    severity="ERROR",
                    check="tracked",
                    rule="secret_path_tracked",
                    path=rel_norm,
                    message="Secret-like file path is tracked in git.",
                )
            )
            continue

        p = REPO_ROOT / rel_norm
        if not p.exists() or not p.is_file():
            continue
        if p.suffix.lower() in _SKIP_TRACKED_SCAN_EXTS:
            continue
        try:
            raw = p.read_bytes()
        except Exception:
            continue
        if b"\x00" in raw[:4096]:
            # Likely binary.
            continue

        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            continue

        for rule_id, severity, rx in _SECRET_REGEX_RULES:
            if rx.search(text):
                findings.append(
                    Finding(
                        severity=severity,
                        check="tracked",
                        rule=rule_id,
                        path=rel_norm,
                        message="Tracked file contains a high-confidence secret pattern.",
                    )
                )

    return findings


def _build_redactor() -> Callable[[str], str]:
    literal_secrets: List[str] = []
    for k, v in os.environ.items():
        if not v:
            continue
        if not _SENSITIVE_ENV_NAME.search(k):
            continue
        if _looks_like_placeholder(v):
            continue
        if len(v.strip()) < 6:
            continue
        literal_secrets.append(v.strip())
    # Replace longest secrets first to avoid partial masking issues.
    literal_secrets.sort(key=len, reverse=True)

    regexes = [rx for _, _, rx in _SECRET_REGEX_RULES]

    def redact(text: str) -> str:
        out = text
        for rx in regexes:
            out = rx.sub("***REDACTED***", out)
        for secret in literal_secrets:
            out = out.replace(secret, "***REDACTED***")
        return out

    return redact


def _print_findings(findings: List[Finding], *, as_json: bool) -> None:
    if as_json:
        payload = [
            {
                "severity": f.severity,
                "check": f.check,
                "rule": f.rule,
                "path": f.path,
                "message": f.message,
            }
            for f in findings
        ]
        print(json.dumps({"findings": payload}, indent=2, sort_keys=True))
        return

    for f in findings:
        loc = f.path or "-"
        print(f"[{f.severity}] {f.check}:{f.rule} {loc} :: {f.message}")


def _exit_code(findings: List[Finding], *, strict: bool) -> int:
    has_error = any(f.severity == "ERROR" for f in findings)
    has_warn = any(f.severity == "WARN" for f in findings)
    if has_error:
        return 1
    if strict and has_warn:
        return 1
    return 0


def _cmd_scan(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="secrets_guard scan", add_help=False)
    parser.add_argument("--staged", action="store_true", help="Scan staged (cached) diff.")
    parser.add_argument("--worktree", action="store_true", help="Scan unstaged worktree diff.")
    parser.add_argument("--tracked", action="store_true", help="Scan tracked file contents (high-confidence patterns).")
    parser.add_argument("--no-remote", action="store_true", help="Skip git remote URL checks.")
    parser.add_argument("--strict", action="store_true", help="Fail on WARN as well as ERROR.")
    parser.add_argument("--json", action="store_true", help="JSON output.")
    args = parser.parse_args(list(argv))

    # Default behavior: staged scan only (best signal).
    if not args.staged and not args.worktree and not args.tracked:
        args.staged = True

    findings: List[Finding] = []
    if args.staged:
        findings.extend(scan_staged_diff())
    if args.worktree:
        findings.extend(scan_worktree_diff())
    if args.tracked:
        findings.extend(scan_tracked_files())
    if not args.no_remote:
        findings.extend(scan_remotes())

    _print_findings(findings, as_json=args.json)
    return _exit_code(findings, strict=args.strict)


def _cmd_run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="secrets_guard run", add_help=False)
    parser.add_argument("--", dest="cmd", nargs=argparse.REMAINDER)
    # argparse can't reliably separate "--"; accept remainder directly.
    cmd = list(argv)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("Usage: python tools/secrets_guard.py run -- <command ...>", file=sys.stderr)
        return 2

    redact = _build_redactor()

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(redact(line))
    return int(proc.wait())


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage:")
        print("  python tools/secrets_guard.py scan [--staged] [--worktree] [--tracked] [--strict] [--json] [--no-remote]")
        print("  python tools/secrets_guard.py run -- <command ...>")
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd == "scan":
        return _cmd_scan(rest)
    if cmd == "run":
        return _cmd_run(rest)

    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
