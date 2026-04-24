#!/usr/bin/env python3
"""Build-time source-contract guard for canonical-source and metrics-summary reads.

This scan is intentionally conservative:
- Python files are checked via AST so docstrings/comments do not trigger.
- Shell files under ``bash/`` are checked line-by-line, ignoring comment lines.

The guard fails if:
- a non-allowlisted code path references ``metrics_summary.json`` as a path
  input/output
- a code path outside ``scripts/robustness_thresholds.py`` directly embeds one
  of the floored config keys as a string literal
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent

PYTHON_SCAN_ROOTS = (ROOT / "scripts", ROOT / "models", ROOT / "etl")
SHELL_SCAN_ROOTS = (ROOT / "bash",)
PYTHON_ALLOWLIST = {
    "scripts/build_automation_dashboard.py",
    "scripts/dashboard_db_bridge.py",
    "scripts/emit_canonical_snapshot.py",
    "scripts/generate_performance_charts.py",
    "scripts/pmx_observability_exporter.py",
    "scripts/source_contract_guard.py",
}
THRESHOLD_HELPER_REL = "scripts/robustness_thresholds.py"
SOURCE_CONTRACT_GUARD_REL = "scripts/source_contract_guard.py"
FORBIDDEN_THRESHOLD_KEYS = {
    "linkage_min_matched",
    "linkage_min_ratio",
    "min_signal_to_noise",
}
METRICS_SUMMARY_TOKEN = "metrics_summary.json"
FORBIDDEN_BYPASS_PATTERNS = (
    re.compile(r"PMX_DISABLE_GATE_FLOORS", re.IGNORECASE),
    re.compile(r"FORCE_PASS", re.IGNORECASE),
    re.compile(r"DISABLE.*FLOOR", re.IGNORECASE),
)
_SHELL_SOURCE_RE = re.compile(r"^\s*(?:source|\.)\s+(.+)$")


@dataclass
class Violation:
    kind: str
    file: str
    line: int
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "file": self.file,
            "line": self.line,
            "detail": self.detail,
        }


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def _contains_forbidden_bypass_token(text: str) -> bool:
    return any(pattern.search(text) for pattern in FORBIDDEN_BYPASS_PATTERNS)


def _docstring_line_numbers(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = getattr(first, "value", None)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if getattr(value, "lineno", None) is not None:
                lines.add(int(value.lineno))
    return lines


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _iter_python_files(root: Path) -> Iterable[Path]:
    for scan_root in PYTHON_SCAN_ROOTS:
        if not scan_root.exists():
            continue
        yield from scan_root.rglob("*.py")


def _iter_shell_files(root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    queue: deque[Path] = deque()
    for scan_root in SHELL_SCAN_ROOTS:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*.sh"):
            queue.append(path.resolve())

    while queue:
        path = queue.popleft()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        yield path
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = _SHELL_SOURCE_RE.match(stripped)
            if not match:
                continue
            token = match.group(1).strip().split("#", 1)[0].strip()
            if not token:
                continue
            if token[0] in {'"', "'"} and token[-1:] == token[0]:
                token = token[1:-1]
            candidate = (path.parent / token).resolve()
            if candidate.exists() and candidate.suffix == ".sh" and candidate not in seen:
                queue.append(candidate)


def _scan_python_file(path: Path, root: Path) -> list[Violation]:
    rel = _relative_path(path, root)
    if rel in PYTHON_ALLOWLIST:
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as exc:
        return [Violation("python_read_error", rel, 0, str(exc))]
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Violation("python_syntax_error", rel, int(exc.lineno or 0), str(exc))]

    docstring_lines = _docstring_line_numbers(tree)
    parents = _parent_map(tree)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        lineno = int(getattr(node, "lineno", 0) or 0)
        if lineno in docstring_lines:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Dict) and any(key is node for key in parent.keys):
            continue
        value = node.value
        if METRICS_SUMMARY_TOKEN in value:
            violations.append(
                Violation(
                    "metrics_summary_reference",
                    rel,
                    lineno,
                    f"metrics_summary.json reference outside allowlist",
                )
            )
        if rel != SOURCE_CONTRACT_GUARD_REL and _contains_forbidden_bypass_token(value):
            violations.append(
                Violation(
                    "forbidden_bypass_token",
                    rel,
                    lineno,
                    "forbidden bypass / force-pass token in production code path",
                )
            )
        if rel != THRESHOLD_HELPER_REL and value in FORBIDDEN_THRESHOLD_KEYS:
            violations.append(
                Violation(
                    "threshold_key_reference",
                    rel,
                    lineno,
                    f"direct reference to forbidden threshold key: {value}",
                )
            )
    return violations


def _scan_shell_file(path: Path, root: Path) -> list[Violation]:
    rel = _relative_path(path, root)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [Violation("shell_read_error", rel, 0, str(exc))]

    violations: list[Violation] = []
    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if METRICS_SUMMARY_TOKEN in raw_line:
            violations.append(
                Violation(
                    "metrics_summary_reference",
                    rel,
                    lineno,
                    "metrics_summary.json reference in shell code path",
                )
            )
        if rel != SOURCE_CONTRACT_GUARD_REL and _contains_forbidden_bypass_token(raw_line):
            violations.append(
                Violation(
                    "forbidden_bypass_token",
                    rel,
                    lineno,
                    "forbidden bypass / force-pass token in shell code path",
                )
            )
        for key in sorted(FORBIDDEN_THRESHOLD_KEYS):
            if key in raw_line and rel != THRESHOLD_HELPER_REL:
                violations.append(
                    Violation(
                        "threshold_key_reference",
                        rel,
                        lineno,
                        f"direct reference to forbidden threshold key: {key}",
                    )
                )
    return violations


def run_source_contract_guard(root: Path | str = ROOT) -> dict[str, Any]:
    repo_root = Path(root)
    violations: list[Violation] = []

    for path in _iter_python_files(repo_root):
        violations.extend(_scan_python_file(path, repo_root))
    for path in _iter_shell_files(repo_root):
        violations.extend(_scan_shell_file(path, repo_root))

    return {
        "ok": not violations,
        "repo_root": str(repo_root),
        "violations": [v.as_dict() for v in violations],
        "scanned_python_roots": [str(p) for p in PYTHON_SCAN_ROOTS],
        "scanned_shell_roots": [str(p) for p in SHELL_SCAN_ROOTS],
        "allowlisted_python_paths": sorted(PYTHON_ALLOWLIST),
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    report = run_source_contract_guard(ROOT)
    if report["ok"]:
        print(json.dumps(report, indent=2))
        return 0
    print(json.dumps(report, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
