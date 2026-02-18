#!/usr/bin/env python3
"""
OpenClaw Self-Improvement Agent for Portfolio Maximizer.

Gives OpenClaw read access to the PMX source code for autonomous capability
acquisition, with full audit logging of all proposed and applied changes.

Architecture:
  - OpenClaw (via local LLMs) can READ project files to understand capabilities
  - It can PROPOSE changes (logged to logs/llm_activity/ as self_improvement events)
  - Changes require human approval before being applied (unless auto_apply=True)
  - All actions are logged with timestamps, diffs, and approval status

Usage:
  python scripts/openclaw_self_improve.py index        # Build source code index
  python scripts/openclaw_self_improve.py capabilities  # List current capabilities
  python scripts/openclaw_self_improve.py audit         # Show self-improvement audit log
  python scripts/openclaw_self_improve.py propose --file <path> --description "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_llm.llm_activity_logger import get_logger

# ---------------------------------------------------------------------------
# Source code index (what OpenClaw can see)
# ---------------------------------------------------------------------------

# Directories OpenClaw has READ access to
READABLE_DIRS = [
    "ai_llm",
    "config",
    "etl",
    "execution",
    "forcester_ts",
    "integrity",
    "models",
    "risk",
    "scripts",
    "tests/utils",
]

# File patterns OpenClaw can read
READABLE_PATTERNS = ["*.py", "*.yml", "*.yaml", "*.json", "*.md", "*.toml"]

# Files OpenClaw must NEVER read (security)
BLOCKED_FILES = [".env", ".env.local", "credentials.json", "auth-profiles.json"]

INDEX_FILE = PROJECT_ROOT / "logs" / "llm_activity" / "source_index.json"


def build_source_index() -> dict:
    """Build an index of readable source files with metadata."""
    index: dict[str, Any] = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "files": [],
        "capabilities": {},
    }

    for dir_name in READABLE_DIRS:
        dir_path = PROJECT_ROOT / dir_name
        if not dir_path.exists():
            continue
        for pattern in READABLE_PATTERNS:
            for f in dir_path.rglob(pattern):
                rel = str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
                if any(blocked in rel for blocked in BLOCKED_FILES):
                    continue
                try:
                    stat = f.stat()
                    lines = f.read_text(encoding="utf-8", errors="replace").count("\n")
                except OSError:
                    continue
                index["files"].append({
                    "path": rel,
                    "size_bytes": stat.st_size,
                    "lines": lines,
                    "modified": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })

    # Extract capability summary
    index["capabilities"] = _extract_capabilities(index["files"])
    index["total_files"] = len(index["files"])
    index["total_lines"] = sum(f["lines"] for f in index["files"])

    # Persist index
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def _extract_capabilities(files: list[dict]) -> dict:
    """Extract high-level capability categories from file list."""
    caps: dict[str, list[str]] = {
        "data_extraction": [],
        "forecasting": [],
        "trading_execution": [],
        "risk_management": [],
        "llm_integration": [],
        "integrity": [],
        "orchestration": [],
        "social_media": [],
    }

    for f in files:
        p = f["path"]
        if "etl/" in p:
            caps["data_extraction"].append(p)
        elif "forcester_ts/" in p or "models/" in p:
            caps["forecasting"].append(p)
        elif "execution/" in p:
            caps["trading_execution"].append(p)
        elif "risk/" in p:
            caps["risk_management"].append(p)
        elif "ai_llm/" in p:
            caps["llm_integration"].append(p)
        elif "integrity/" in p:
            caps["integrity"].append(p)
        elif "orchestrat" in p.lower():
            caps["orchestration"].append(p)
        elif "openclaw" in p.lower() or "social" in p.lower() or "telegram" in p.lower():
            caps["social_media"].append(p)

    return {k: {"count": len(v), "files": v[:5]} for k, v in caps.items() if v}


def read_source_file(rel_path: str) -> Optional[str]:
    """Read a source file (with security checks and audit logging)."""
    # Security: block sensitive files
    if any(blocked in rel_path for blocked in BLOCKED_FILES):
        get_logger().log_self_improvement(
            action="read_blocked",
            target_file=rel_path,
            description=f"Blocked read attempt on sensitive file: {rel_path}",
        )
        return None

    # Security: must be in readable dirs
    if not any(rel_path.startswith(d) for d in READABLE_DIRS):
        get_logger().log_self_improvement(
            action="read_blocked",
            target_file=rel_path,
            description=f"File outside readable dirs: {rel_path}",
        )
        return None

    full_path = PROJECT_ROOT / rel_path
    if not full_path.exists():
        return None

    content = full_path.read_text(encoding="utf-8", errors="replace")

    get_logger().log_self_improvement(
        action="read_source",
        target_file=rel_path,
        description=f"Read {len(content)} chars from {rel_path}",
    )
    return content


def propose_change(
    target_file: str,
    description: str,
    diff_preview: str = "",
    auto_apply: bool = False,
) -> dict:
    """Propose a self-improvement change (logged for human review)."""
    proposal = {
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "target_file": target_file,
        "description": description,
        "diff_preview": diff_preview[:1000],
        "auto_apply": auto_apply,
        "status": "proposed",
    }

    get_logger().log_self_improvement(
        action="propose_change",
        target_file=target_file,
        description=description,
        diff_preview=diff_preview,
        approved=False,
        applied=False,
    )

    # Save proposal to dedicated file for human review
    proposals_dir = PROJECT_ROOT / "logs" / "llm_activity" / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = target_file.replace("/", "_").replace("\\", "_")
    proposal_file = proposals_dir / f"{ts}_{safe_name}.json"
    proposal_file.write_text(json.dumps(proposal, indent=2), encoding="utf-8")

    return proposal


def get_audit_log(days: int = 7) -> list[dict]:
    """Get self-improvement audit entries from the last N days."""
    entries = get_logger().get_recent(hours=days * 24)
    return [e for e in entries if e.get("type") == "self_improvement"]


# ---------------------------------------------------------------------------
# OpenClaw workspace integration
# ---------------------------------------------------------------------------

def update_openclaw_tools_md() -> None:
    """Update OpenClaw's TOOLS.md with PMX source code access info."""
    tools_md = Path.home() / ".openclaw" / "workspace" / "TOOLS.md"
    if not tools_md.exists():
        return

    content = tools_md.read_text(encoding="utf-8")

    pmx_section = """

## Portfolio Maximizer (PMX) Source Access

PMX project root: `C:\\Users\\Bestman\\personal_projects\\portfolio_maximizer_v45\\portfolio_maximizer_v45`

### Readable Directories
- `ai_llm/` - LLM integration (ollama_client, activity logger, signal generator)
- `config/` - YAML configs (pipeline, forecasting, LLM, signal routing)
- `etl/` - Data extraction, validation, preprocessing
- `execution/` - Paper trading engine, order management
- `forcester_ts/` - Time series forecasting models
- `integrity/` - PnL integrity enforcement
- `models/` - Signal generation, routing, regime detection
- `risk/` - Risk management (barbell policy)
- `scripts/` - Orchestration, migrations, utilities

### Self-Improvement Protocol
1. Read source via: `python scripts/openclaw_self_improve.py index`
2. Propose changes via: `python scripts/openclaw_self_improve.py propose --file <path> --description "..."`
3. All proposals logged to: `logs/llm_activity/proposals/`
4. Human reviews and approves before application
5. Audit trail: `python scripts/openclaw_self_improve.py audit`

### Tool Preference
- For production gate/reconciliation operations, prefer orchestrator tool
  `run_production_audit_gate` over generic shell/exec command strings.
- Forward human review queue via `python scripts/forward_self_improvement_reviews.py`
  (typically scheduled through `bash/production_cron.sh self_improvement_review_forward`).

### Blocked Files (security)
- `.env`, `.env.local`, `credentials.json`, `auth-profiles.json`
"""

    if "Portfolio Maximizer (PMX) Source Access" not in content:
        content += pmx_section
        tools_md.write_text(content, encoding="utf-8")
        print("[OK] Updated OpenClaw TOOLS.md with PMX source access")
    else:
        print("[SKIP] OpenClaw TOOLS.md already has PMX section")


def update_openclaw_memory(note: str) -> None:
    """Append a note to today's OpenClaw memory file."""
    memory_dir = Path.home() / ".openclaw" / "workspace" / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    memory_file = memory_dir / f"{today}.md"

    timestamp = datetime.now(timezone.utc).strftime("%H:%M UTC")
    entry = f"\n- [{timestamp}] {note}\n"

    with open(memory_file, "a", encoding="utf-8") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_index(_args) -> int:
    idx = build_source_index()
    print(f"[self-improve] Source index built: {idx['total_files']} files, {idx['total_lines']} lines")
    print(f"  Index saved to: {INDEX_FILE}")
    for cap, info in idx["capabilities"].items():
        print(f"  {cap}: {info['count']} files")
    return 0


def cmd_capabilities(_args) -> int:
    if INDEX_FILE.exists():
        idx = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    else:
        idx = build_source_index()

    print("[self-improve] PMX Capabilities:")
    for cap, info in idx.get("capabilities", {}).items():
        print(f"  {cap} ({info['count']} files):")
        for f in info.get("files", []):
            print(f"    - {f}")
    return 0


def cmd_audit(args) -> int:
    entries = get_audit_log(days=args.days)
    print(f"[self-improve] Audit log ({len(entries)} entries, last {args.days} days):")
    for e in entries[-20:]:
        ts = e.get("timestamp", "?")[:19]
        action = e.get("action", "?")
        target = e.get("target_file", "?")
        desc = e.get("description", "")[:80]
        approved = "[APPROVED]" if e.get("approved") else "[PENDING]"
        print(f"  {ts} {action} {target} {approved} {desc}")
    if not entries:
        print("  (no self-improvement activity recorded)")
    return 0


def cmd_propose(args) -> int:
    result = propose_change(
        target_file=args.file,
        description=args.description,
        auto_apply=False,
    )
    print(f"[self-improve] Proposal created: {result['target_file']}")
    print(f"  Description: {result['description']}")
    print(f"  Status: {result['status']} (awaiting human review)")
    return 0


def cmd_setup(_args) -> int:
    """Set up OpenClaw workspace integration."""
    build_source_index()
    update_openclaw_tools_md()
    update_openclaw_memory(
        "PMX self-improvement agent configured. "
        "Source index built. Read access to ai_llm/, config/, etl/, "
        "execution/, forcester_ts/, integrity/, models/, risk/, scripts/. "
        "All changes require human approval via logs/llm_activity/proposals/."
    )
    print("[OK] OpenClaw self-improvement setup complete")
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="OpenClaw self-improvement agent for PMX")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("index", help="Build source code index")
    sub.add_parser("capabilities", help="List PMX capabilities")
    sub.add_parser("setup", help="Set up OpenClaw workspace integration")

    pa = sub.add_parser("audit", help="Show self-improvement audit log")
    pa.add_argument("--days", type=int, default=7, help="Days to look back")

    pp = sub.add_parser("propose", help="Propose a change")
    pp.add_argument("--file", required=True, help="Target file (relative path)")
    pp.add_argument("--description", required=True, help="Change description")

    args = p.parse_args(argv)

    if args.cmd == "index":
        return cmd_index(args)
    elif args.cmd == "capabilities":
        return cmd_capabilities(args)
    elif args.cmd == "audit":
        return cmd_audit(args)
    elif args.cmd == "propose":
        return cmd_propose(args)
    elif args.cmd == "setup":
        return cmd_setup(args)
    else:
        p.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
