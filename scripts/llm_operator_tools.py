"""
Helper utilities for LLM/OpenClaw operator and reviewer workflows.

These functions are intentionally read-only. They help the orchestrator review
current repo changes, summarize pytest failures, and triage production gate
artifacts without mutating project state or relaxing gate policy.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE_ARTIFACT = PROJECT_ROOT / "logs" / "audit_gate" / "production_gate_latest.json"
DEFAULT_GATE_SUMMARY_CACHE = PROJECT_ROOT / "logs" / "forecast_audits_cache" / "latest_summary.json"
DEFAULT_GATE_DECOMP_JSON = PROJECT_ROOT / "logs" / "audit_gate" / "production_gate_decomposition_latest.json"
DEFAULT_GATE_DECOMP_MD = PROJECT_ROOT / "logs" / "audit_gate" / "production_gate_decomposition_latest.md"

_TEXT_PREVIEW_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".out",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_rel_path(path_text: str) -> str:
    text = str(path_text or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _safe_truncate(text: str, max_chars: int) -> str:
    raw = str(text or "")
    if len(raw) <= max_chars:
        return raw
    clipped = max(0, int(max_chars) - 3)
    return raw[:clipped] + "..."


def _git_executable() -> str:
    git_bin = shutil.which("git")
    if not git_bin:
        raise RuntimeError("git executable not available")
    return git_bin


def _run_git(project_root: Path, args: list[str], *, timeout_seconds: float = 15.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_git_executable(), *args],
        cwd=str(project_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(5.0, float(timeout_seconds)),
        check=False,
    )


def _parse_git_status_porcelain(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip("\r\n")
        if not line:
            continue
        if line.startswith("?? "):
            index_status = "?"
            worktree_status = "?"
            path_part = line[3:]
        else:
            if len(line) < 4:
                continue
            index_status = line[0]
            worktree_status = line[1]
            path_part = line[3:]
        old_path = ""
        path = path_part
        if " -> " in path_part:
            old_path, path = path_part.split(" -> ", 1)
        rows.append(
            {
                "index_status": index_status,
                "worktree_status": worktree_status,
                "path": _normalize_rel_path(path),
                "old_path": _normalize_rel_path(old_path) if old_path else "",
            }
        )
    return rows


def _change_tags(path_text: str) -> list[str]:
    path_lower = str(path_text or "").lower()
    tags: list[str] = []
    if path_lower.startswith("config/"):
        tags.append("config")
    if path_lower.startswith("tests/"):
        tags.append("tests")
    if "openclaw" in path_lower:
        tags.append("openclaw")
    if "gate" in path_lower or "audit" in path_lower:
        tags.append("gate")
    if "llm" in path_lower or "orchestrator" in path_lower:
        tags.append("llm")
    if "production" in path_lower:
        tags.append("production")
    if path_lower.startswith("scripts/"):
        tags.append("script")
    return tags


def _read_untracked_preview(project_root: Path, rel_path: str, *, max_chars: int) -> dict[str, Any] | None:
    path = (project_root / rel_path).resolve()
    try:
        if not path.exists() or not path.is_file():
            return None
        if path.suffix.lower() not in _TEXT_PREVIEW_SUFFIXES:
            return {
                "path": rel_path,
                "status": "untracked",
                "preview": "",
                "note": "preview_skipped_non_text_suffix",
            }
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    return {
        "path": rel_path,
        "status": "untracked",
        "preview": _safe_truncate(text, max_chars),
        "note": "preview_from_worktree",
    }


def review_changed_files(
    *,
    project_root: Path = PROJECT_ROOT,
    paths: Optional[list[str]] = None,
    base_ref: str = "",
    max_files: int = 12,
    max_diff_chars: int = 16000,
    diff_context: int = 1,
    include_untracked: bool = True,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    selected_paths = {
        _normalize_rel_path(row)
        for row in (paths or [])
        if _normalize_rel_path(row)
    }

    status_proc = _run_git(project_root, ["status", "--porcelain=v1"], timeout_seconds=timeout_seconds)
    if int(status_proc.returncode) != 0:
        return {
            "action": "review_changed_files",
            "status": "FAIL",
            "error": (status_proc.stderr or status_proc.stdout or "git status failed").strip(),
        }

    changed_rows = _parse_git_status_porcelain(status_proc.stdout)
    if selected_paths:
        changed_rows = [row for row in changed_rows if row.get("path") in selected_paths]

    changed_rows = changed_rows[: max(1, int(max_files))]
    tracked_paths = [str(row["path"]) for row in changed_rows if str(row.get("index_status")) != "?" or str(row.get("worktree_status")) != "?"]
    untracked_paths = [str(row["path"]) for row in changed_rows if str(row.get("index_status")) == "?" and str(row.get("worktree_status")) == "?"]

    diff_preview = ""
    diff_target = (base_ref or "").strip() or "HEAD"
    if tracked_paths:
        diff_args = ["diff", f"--unified={max(0, int(diff_context))}", "--no-ext-diff", diff_target, "--", *tracked_paths]
        diff_proc = _run_git(project_root, diff_args, timeout_seconds=timeout_seconds)
        diff_preview = _safe_truncate(diff_proc.stdout or diff_proc.stderr or "", max(800, int(max_diff_chars)))

    untracked_previews: list[dict[str, Any]] = []
    if include_untracked:
        remaining = max(600, int(max_diff_chars) // 3)
        for rel_path in untracked_paths:
            preview = _read_untracked_preview(project_root, rel_path, max_chars=remaining)
            if preview:
                untracked_previews.append(preview)

    change_types = Counter()
    tag_counts = Counter()
    for row in changed_rows:
        change_types[f"{row.get('index_status','?')}{row.get('worktree_status','?')}"] += 1
        for tag in _change_tags(str(row.get("path") or "")):
            tag_counts[tag] += 1

    return {
        "action": "review_changed_files",
        "status": "PASS",
        "worktree_state": "dirty" if changed_rows else "clean",
        "project_root": str(project_root.resolve()),
        "base_ref": diff_target if base_ref else "",
        "selected_paths": sorted(selected_paths),
        "total_changed_files": len(changed_rows),
        "change_types": dict(change_types),
        "tag_counts": dict(tag_counts),
        "changed_files": [
            {
                **row,
                "tags": _change_tags(str(row.get("path") or "")),
            }
            for row in changed_rows
        ],
        "diff_preview": diff_preview,
        "untracked_previews": untracked_previews,
    }


def _load_test_failure_text(*, log_path: str = "", output_text: str = "") -> tuple[str, str]:
    direct_text = str(output_text or "")
    if direct_text.strip():
        return direct_text, "inline_text"

    raw_path = str(log_path or "").strip()
    if not raw_path:
        return "", ""

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "", str(path)
    return text, str(path)


def summarize_test_failure(
    *,
    log_path: str = "",
    output_text: str = "",
    max_failures: int = 10,
    max_suspect_paths: int = 8,
    max_error_types: int = 8,
) -> dict[str, Any]:
    text, source = _load_test_failure_text(log_path=log_path, output_text=output_text)
    if not text.strip():
        return {
            "action": "summarize_test_failure",
            "status": "FAIL",
            "error": "no test output provided",
            "source": source,
        }

    failures: list[dict[str, Any]] = []
    for match in re.finditer(r"^(FAILED|ERROR)\s+([^\s]+)\s+-\s+(.+)$", text, flags=re.MULTILINE):
        kind = match.group(1).strip().upper()
        nodeid = match.group(2).strip()
        summary = match.group(3).strip()
        failures.append({"kind": kind, "nodeid": nodeid, "summary": _safe_truncate(summary, 220)})
        if len(failures) >= max(1, int(max_failures)):
            break

    error_counter: Counter[str] = Counter()
    for match in re.finditer(r"^E\s+([A-Za-z_][A-Za-z0-9_\.]*)(?::|\b)", text, flags=re.MULTILINE):
        error_counter[match.group(1).strip()] += 1

    suspect_paths = Counter()
    for match in re.finditer(r"^([A-Za-z]:)?[^:\n]+\.py:\d+(?::\s+in\s+.+)?$", text, flags=re.MULTILINE):
        value = match.group(0).strip()
        suspect_paths[value] += 1

    summary_line = ""
    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        if "failed" in line.lower() and any(token in line.lower() for token in ("passed", "error", "warning", "skipped", "failed")):
            summary_line = line
            break

    if not failures and not error_counter and "failed" not in text.lower():
        return {
            "action": "summarize_test_failure",
            "status": "FAIL",
            "error": "no pytest failure markers found",
            "source": source,
        }

    return {
        "action": "summarize_test_failure",
        "status": "PASS",
        "source": source,
        "summary_line": summary_line,
        "total_failures": len(failures),
        "total_errors": sum(1 for row in failures if row.get("kind") == "ERROR"),
        "failures": failures,
        "error_types": [
            {"type": key, "count": int(value)}
            for key, value in error_counter.most_common(max(1, int(max_error_types)))
        ],
        "suspect_paths": [
            {"path": key, "count": int(value)}
            for key, value in suspect_paths.most_common(max(1, int(max_suspect_paths)))
        ],
    }


def _resolved_path(value: str, default_path: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return default_path
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def triage_gate_failure(
    *,
    gate_artifact: str = "",
    summary_cache: str = "",
    output_json_path: str = "",
    output_md_path: str = "",
    force_refresh: bool = False,
) -> dict[str, Any]:
    artifact_path = _resolved_path(gate_artifact, DEFAULT_GATE_ARTIFACT)
    summary_cache_path = _resolved_path(summary_cache, DEFAULT_GATE_SUMMARY_CACHE)
    decomp_json_path = _resolved_path(output_json_path, DEFAULT_GATE_DECOMP_JSON)
    decomp_md_path = _resolved_path(output_md_path, DEFAULT_GATE_DECOMP_MD)

    if not artifact_path.exists():
        return {
            "action": "triage_gate_failure",
            "status": "FAIL",
            "error": f"gate artifact not found: {artifact_path}",
            "artifact_path": str(artifact_path),
        }

    try:
        from scripts import gate_failure_decomposition as decomp

        report, refresh = decomp.refresh_decomposition_report(
            artifact_path=artifact_path,
            output_json_path=decomp_json_path,
            summary_cache_path=summary_cache_path,
            output_md_path=decomp_md_path,
            force=bool(force_refresh),
        )
    except Exception as exc:
        return {
            "action": "triage_gate_failure",
            "status": "FAIL",
            "error": str(exc),
            "artifact_path": str(artifact_path),
        }

    components = report.get("components") if isinstance(report.get("components"), dict) else {}
    failed_components = [
        key for key, value in components.items()
        if isinstance(value, dict) and not bool(value.get("pass"))
    ]
    reason_breakdown = report.get("reason_breakdown") if isinstance(report.get("reason_breakdown"), dict) else {}
    phase3_ready = bool(report.get("phase3_strict_ready", report.get("phase3_ready")))
    phase3_reason = str(report.get("phase3_strict_reason") or report.get("phase3_reason") or "")

    return {
        "action": "triage_gate_failure",
        "status": "PASS",
        "gate_status": "PASS" if phase3_ready else "FAIL",
        "phase3_ready": phase3_ready,
        "phase3_reason": phase3_reason,
        "artifact_path": str(artifact_path.resolve()),
        "output_json_path": str(decomp_json_path.resolve()),
        "output_md_path": str(decomp_md_path.resolve()),
        "refresh": refresh,
        "failed_components": failed_components,
        "top_invalid_context_reasons": list(reason_breakdown.get("invalid_context_top_reasons") or [])[:5],
        "top_non_trade_context_reasons": list(reason_breakdown.get("non_trade_context_top_reasons") or [])[:5],
        "report": report,
    }


__all__ = [
    "review_changed_files",
    "summarize_test_failure",
    "triage_gate_failure",
]
