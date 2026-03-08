#!/usr/bin/env python3
"""Validate repo-local documentation contract for temporary/generated docs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = (
    "Doc Type:",
    "Authority:",
    "Owner:",
    "Last Verified:",
    "Verification Commands:",
    "Artifacts:",
    "Supersedes:",
    "Expires When:",
)

CONTRACT_DOCS = (
    "Documentation/DOC_SOURCES_OF_TRUTH.md",
    "Documentation/DOC_METADATA_CONTRACT.md",
    "Documentation/AGENT_B_DASHBOARD_RUNTIME_TRUTH_HANDOFF_2026-03-08.md",
    "Documentation/AGENT_C_EXPERIMENT_BRIEF_EXP-R5-001_2026-03-08.md",
    "Documentation/AGENT_C_PHASED_IMPLEMENTATION_PLAN_2026-03-08.md",
    "Documentation/AGENT_C_PERSISTENCE_MANAGER_INTEGRATION_2026-03-08.md",
    "Documentation/AGENT_C_READINESS_BLOCKER_MATRIX_2026-03-08.md",
    "Documentation/AGENT_C_RESUME_PACK_2026-03-08_AM.md",
    "Documentation/GENERATED_RUNTIME_STATUS_SNAPSHOT.md",
)


@dataclass
class Issue:
    path: str
    message: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _validate_header(path: Path, text: str, *, root: Path) -> list[Issue]:
    issues: list[Issue] = []
    head = text.splitlines()[:30]
    for field in REQUIRED_FIELDS:
        if not any(line.startswith(field) for line in head):
            issues.append(Issue(_display_path(path, root), f"missing header field '{field}'"))
    return issues


def validate_docs(root: Path = ROOT, docs: Iterable[str] = CONTRACT_DOCS) -> list[Issue]:
    issues: list[Issue] = []
    for rel in docs:
        path = root / rel
        if not path.exists():
            issues.append(Issue(rel, "missing required contract doc"))
            continue
        text = _read_text(path)
        issues.extend(_validate_header(path, text, root=root))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate documentation metadata contract.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any issues are found")
    args = parser.parse_args()

    issues = validate_docs()
    payload = {
        "ok": not issues,
        "issue_count": len(issues),
        "issues": [{"path": issue.path, "message": issue.message} for issue in issues],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if issues:
            for issue in issues:
                print(f"[FAIL] {issue.path}: {issue.message}")
        else:
            print("[OK] documentation contract clean")

    return 1 if args.strict and issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
