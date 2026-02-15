#!/usr/bin/env python3
"""
Compare two production gate artifacts and diff their recorded repo_state.

Use this to make worktree claims provable going forward: capture a baseline gate
artifact at time T0, then compare a later artifact at T1.

This script is read-only (no git writes, no DB writes).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GateArtifact:
    path: Path
    raw: Dict[str, Any]
    timestamp_utc: str
    repo_state: Dict[str, Any]


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_gate_artifact(path: Path) -> GateArtifact:
    raw = _load_json(path)
    ts = str(raw.get("timestamp_utc") or "").strip()
    repo_state = raw.get("repo_state") if isinstance(raw.get("repo_state"), dict) else {}
    return GateArtifact(path=path, raw=raw, timestamp_utc=ts, repo_state=repo_state)


def _entries(repo_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    status = repo_state.get("status") if isinstance(repo_state.get("status"), dict) else {}
    entries = status.get("entries") if isinstance(status.get("entries"), list) else []
    return [e for e in entries if isinstance(e, dict)]


def _file_meta(repo_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    files = repo_state.get("files") if isinstance(repo_state.get("files"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in files:
        if not isinstance(row, dict):
            continue
        path = str(row.get("path") or "").strip()
        if not path:
            continue
        out[path] = row
    return out


def _entry_map(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        path = str(e.get("path") or "").strip()
        if not path:
            continue
        out.setdefault(path, e)
    return out


def _diff_dict(a: Dict[str, Any], b: Dict[str, Any], keys: List[str]) -> Dict[str, Tuple[Any, Any]]:
    out: Dict[str, Tuple[Any, Any]] = {}
    for k in keys:
        if a.get(k) != b.get(k):
            out[k] = (a.get(k), b.get(k))
    return out


def diff_repo_state(artifact_a: GateArtifact, artifact_b: GateArtifact) -> Dict[str, Any]:
    rs_a = artifact_a.repo_state or {}
    rs_b = artifact_b.repo_state or {}

    entries_a = _entries(rs_a)
    entries_b = _entries(rs_b)
    map_a = _entry_map(entries_a)
    map_b = _entry_map(entries_b)

    meta_a = _file_meta(rs_a)
    meta_b = _file_meta(rs_b)

    paths_a = set(map_a)
    paths_b = set(map_b)

    added = sorted(paths_b - paths_a)
    removed = sorted(paths_a - paths_b)
    common = sorted(paths_a & paths_b)

    changed: List[Dict[str, Any]] = []
    for path in common:
        e_a = map_a.get(path) or {}
        e_b = map_b.get(path) or {}
        m_a = meta_a.get(path) or {}
        m_b = meta_b.get(path) or {}

        entry_delta = _diff_dict(e_a, e_b, keys=["kind", "code", "path_orig"])
        meta_delta = _diff_dict(
            m_a,
            m_b,
            keys=[
                "exists",
                "bytes",
                "mtime_utc",
                "ctime_utc",
                "sha256",
                "sha256_skip_reason",
                "last_commit",
                "last_committed_at",
            ],
        )
        if entry_delta or meta_delta:
            changed.append(
                {
                    "path": path,
                    "entry_delta": entry_delta,
                    "meta_delta": meta_delta,
                }
            )

    def _status(repo_state: Dict[str, Any]) -> Dict[str, Any]:
        s = repo_state.get("status") if isinstance(repo_state.get("status"), dict) else {}
        return {
            "tracked_changed": s.get("tracked_changed"),
            "untracked": s.get("untracked"),
            "staged": s.get("staged"),
            "unstaged": s.get("unstaged"),
        }

    return {
        "a": {
            "path": str(artifact_a.path),
            "timestamp_utc": artifact_a.timestamp_utc,
            "branch": rs_a.get("branch"),
            "head": rs_a.get("head"),
            "upstream": rs_a.get("upstream"),
            "ahead": rs_a.get("ahead"),
            "behind": rs_a.get("behind"),
            "status": _status(rs_a),
        },
        "b": {
            "path": str(artifact_b.path),
            "timestamp_utc": artifact_b.timestamp_utc,
            "branch": rs_b.get("branch"),
            "head": rs_b.get("head"),
            "upstream": rs_b.get("upstream"),
            "ahead": rs_b.get("ahead"),
            "behind": rs_b.get("behind"),
            "status": _status(rs_b),
        },
        "diff": {
            "added_paths": added,
            "removed_paths": removed,
            "changed_paths": changed,
        },
    }


def render_text(diff: Dict[str, Any]) -> str:
    a = diff.get("a") if isinstance(diff.get("a"), dict) else {}
    b = diff.get("b") if isinstance(diff.get("b"), dict) else {}
    d = diff.get("diff") if isinstance(diff.get("diff"), dict) else {}

    a_status = a.get("status") if isinstance(a.get("status"), dict) else {}
    b_status = b.get("status") if isinstance(b.get("status"), dict) else {}

    lines: List[str] = []
    lines.append("=== Gate Artifact Repo-State Diff ===")
    lines.append(f"A: {a.get('path')}  (timestamp_utc={a.get('timestamp_utc')})")
    lines.append(f"   branch={a.get('branch')} head={a.get('head')} upstream={a.get('upstream')}")
    lines.append(
        "   worktree="
        f"tracked_changed:{a_status.get('tracked_changed')} untracked:{a_status.get('untracked')} "
        f"staged:{a_status.get('staged')} unstaged:{a_status.get('unstaged')} "
        f"ahead:{a.get('ahead')} behind:{a.get('behind')}"
    )
    lines.append(f"B: {b.get('path')}  (timestamp_utc={b.get('timestamp_utc')})")
    lines.append(f"   branch={b.get('branch')} head={b.get('head')} upstream={b.get('upstream')}")
    lines.append(
        "   worktree="
        f"tracked_changed:{b_status.get('tracked_changed')} untracked:{b_status.get('untracked')} "
        f"staged:{b_status.get('staged')} unstaged:{b_status.get('unstaged')} "
        f"ahead:{b.get('ahead')} behind:{b.get('behind')}"
    )

    added = d.get("added_paths") if isinstance(d.get("added_paths"), list) else []
    removed = d.get("removed_paths") if isinstance(d.get("removed_paths"), list) else []
    changed = d.get("changed_paths") if isinstance(d.get("changed_paths"), list) else []

    lines.append("")
    lines.append("Paths:")
    lines.append(f"- added: {len(added)}")
    for p in added[:50]:
        lines.append(f"  {p}")
    if len(added) > 50:
        lines.append("  ...")
    lines.append(f"- removed: {len(removed)}")
    for p in removed[:50]:
        lines.append(f"  {p}")
    if len(removed) > 50:
        lines.append("  ...")
    lines.append(f"- changed: {len(changed)}")
    for row in changed[:50]:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        entry_delta = row.get("entry_delta") if isinstance(row.get("entry_delta"), dict) else {}
        meta_delta = row.get("meta_delta") if isinstance(row.get("meta_delta"), dict) else {}
        bits = []
        if entry_delta:
            bits.append("entry")
        if meta_delta:
            bits.append("meta")
        marker = ",".join(bits) if bits else "changed"
        lines.append(f"  {path} ({marker})")
    if len(changed) > 50:
        lines.append("  ...")

    lines.append("")
    lines.append("Notes:")
    lines.append("- This diff shows what changed between the two capture times, not who/what process caused it.")
    lines.append("- Untracked files are intentionally not hashed to avoid accidental secret fingerprints.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Gate artifact A JSON path.")
    parser.add_argument("--b", type=Path, required=True, help="Gate artifact B JSON path.")
    parser.add_argument("--json", dest="emit_json", action="store_true", help="Emit JSON diff instead of text.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path.")

    args = parser.parse_args(argv)

    path_a = args.a.expanduser().resolve()
    path_b = args.b.expanduser().resolve()
    if not path_a.exists():
        raise SystemExit(f"[ERROR] Missing artifact A: {path_a}")
    if not path_b.exists():
        raise SystemExit(f"[ERROR] Missing artifact B: {path_b}")

    art_a = _load_gate_artifact(path_a)
    art_b = _load_gate_artifact(path_b)

    if not art_a.repo_state or not isinstance(art_a.repo_state, dict):
        raise SystemExit(f"[ERROR] Artifact A missing repo_state: {path_a}")
    if not art_b.repo_state or not isinstance(art_b.repo_state, dict):
        raise SystemExit(f"[ERROR] Artifact B missing repo_state: {path_b}")

    diff = diff_repo_state(art_a, art_b)
    if args.emit_json:
        out = json.dumps(diff, indent=2, sort_keys=True, default=str) + "\n"
    else:
        out = render_text(diff)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out, encoding="utf-8")
    else:
        print(out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

