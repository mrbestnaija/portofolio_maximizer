#!/usr/bin/env python3
"""
Check local git repository and GitHub remote status.

This is designed for OpenClaw/agent exec usage when a user asks:
"check this project repository on GitHub".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


GITHUB_API_BASE = "https://api.github.com"


@dataclass
class CmdResult:
    ok: bool
    code: int
    out: str
    err: str


def _run(cmd: list[str], *, cwd: Path, timeout: float = 15.0) -> CmdResult:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout)),
            check=False,
        )
        return CmdResult(ok=p.returncode == 0, code=int(p.returncode), out=(p.stdout or "").strip(), err=(p.stderr or "").strip())
    except FileNotFoundError as exc:
        return CmdResult(ok=False, code=127, out="", err=str(exc))
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        return CmdResult(ok=False, code=124, out=(out or "").strip(), err=(err or "command timed out").strip())


def _parse_github_slug(remote_url: str) -> Optional[str]:
    raw = (remote_url or "").strip()
    if not raw:
        return None

    patterns = [
        r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
        r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
        r"^http://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
        r"^ssh://git@github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$",
    ]
    for pattern in patterns:
        m = re.match(pattern, raw, flags=re.IGNORECASE)
        if not m:
            continue
        owner = (m.group("owner") or "").strip()
        repo = (m.group("repo") or "").strip()
        if owner and repo:
            return f"{owner}/{repo}"
    return None


def _gh_json(path: str, *, token: Optional[str], timeout: float = 12.0) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[int]]:
    url = f"{GITHUB_API_BASE.rstrip('/')}/{path.lstrip('/')}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "pmx-github-repo-check/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, float(timeout))) as resp:
            status = int(getattr(resp, "status", 200))
            text = resp.read().decode("utf-8", errors="replace")
            data = json.loads(text) if text else {}
            if isinstance(data, dict):
                return data, None, status
            return None, "unexpected_github_response", status
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict) and parsed.get("message"):
                return None, str(parsed.get("message")), int(exc.code)
        except Exception:
            pass
        return None, f"http_{exc.code}", int(exc.code)
    except Exception as exc:
        return None, str(exc), None


def check_repo(cwd: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "PASS",
        "cwd": str(cwd),
        "checks": {},
        "issues": [],
    }

    git_probe = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=cwd)
    if not git_probe.ok or git_probe.out.lower() != "true":
        return {
            "status": "FAIL",
            "cwd": str(cwd),
            "error": "not_a_git_repository",
            "details": git_probe.err or git_probe.out,
        }

    origin = _run(["git", "remote", "get-url", "origin"], cwd=cwd)
    origin_url = origin.out if origin.ok else ""
    payload["checks"]["origin_url"] = origin_url or None

    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    head = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    short_head = _run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd)
    dirty = _run(["git", "status", "--porcelain"], cwd=cwd)
    last_commit = _run(["git", "log", "-1", "--format=%cI|%an|%s"], cwd=cwd)

    payload["checks"]["local"] = {
        "branch": branch.out if branch.ok else None,
        "head": head.out if head.ok else None,
        "head_short": short_head.out if short_head.ok else None,
        "dirty": bool((dirty.out or "").strip()),
    }
    if last_commit.ok and "|" in last_commit.out:
        committed_at, author, subject = last_commit.out.split("|", 2)
        payload["checks"]["local"]["last_commit"] = {
            "committed_at": committed_at.strip(),
            "author": author.strip(),
            "subject": subject.strip(),
        }

    upstream = _run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"], cwd=cwd)
    if upstream.ok:
        ahead_behind = _run(["git", "rev-list", "--left-right", "--count", "@{upstream}...HEAD"], cwd=cwd)
        ahead = behind = None
        if ahead_behind.ok:
            parts = (ahead_behind.out or "").split()
            if len(parts) == 2:
                behind = int(parts[0])
                ahead = int(parts[1])
        payload["checks"]["tracking"] = {
            "upstream": upstream.out,
            "ahead": ahead,
            "behind": behind,
        }
    else:
        payload["checks"]["tracking"] = {"upstream": None}
        payload["issues"].append("no_upstream_tracking_branch")

    slug = _parse_github_slug(origin_url)
    payload["checks"]["github_slug"] = slug
    if not slug:
        payload["status"] = "WARN"
        payload["issues"].append("origin_is_not_a_github_remote")
        return payload

    token = (os.getenv("GITHUB_TOKEN") or "").strip() or None
    repo_info, repo_err, repo_status = _gh_json(f"repos/{slug}", token=token)
    default_branch = None
    if repo_info:
        default_branch = str(repo_info.get("default_branch") or "").strip() or None
        payload["checks"]["github_repo"] = {
            "private": bool(repo_info.get("private", False)),
            "default_branch": default_branch,
            "open_issues_count": repo_info.get("open_issues_count"),
            "stargazers_count": repo_info.get("stargazers_count"),
            "pushed_at": repo_info.get("pushed_at"),
        }
    else:
        payload["status"] = "WARN"
        payload["issues"].append(f"github_repo_api_unavailable:{repo_err or 'unknown'}")
        payload["checks"]["github_repo"] = {"status_code": repo_status}

    if default_branch:
        commit_info, commit_err, commit_status = _gh_json(f"repos/{slug}/commits/{default_branch}", token=token)
        if commit_info and isinstance(commit_info, dict):
            sha = str(commit_info.get("sha") or "").strip() or None
            payload["checks"]["github_default_branch_head"] = {
                "branch": default_branch,
                "sha": sha,
                "html_url": commit_info.get("html_url"),
            }
            local_head = payload["checks"]["local"].get("head")
            if sha and local_head and sha != local_head:
                payload["status"] = "WARN"
                payload["issues"].append("local_head_differs_from_github_default_head")
        else:
            payload["status"] = "WARN"
            payload["issues"].append(f"github_commit_api_unavailable:{commit_err or 'unknown'}")
            payload["checks"]["github_default_branch_head"] = {"status_code": commit_status}

    if payload["checks"]["local"].get("dirty"):
        payload["status"] = "WARN"
        payload["issues"].append("local_worktree_has_uncommitted_changes")

    return payload


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check local repository vs GitHub remote status.")
    parser.add_argument("--cwd", default=".", help="Repository path (default: current directory).")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    out = check_repo(cwd)
    if args.pretty:
        print(json.dumps(out, indent=2, ensure_ascii=True))
    else:
        print(json.dumps(out, ensure_ascii=True))

    status = str(out.get("status") or "").upper()
    if status == "FAIL":
        return 2
    if status == "WARN":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

