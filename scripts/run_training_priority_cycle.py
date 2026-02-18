#!/usr/bin/env python3
"""
Run prioritized forecaster/LLM training and fine-tuning workflows.

Config-driven runner for local cron and CI schedules.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "training_priority.yml"


@dataclass
class Task:
    task_id: str
    description: str
    profiles: list[str]
    targets: list[str]
    priority: int
    command: list[str]
    required_paths: list[str]
    timeout_seconds: float
    critical: bool
    enabled: bool
    env: dict[str, str]


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _safe_load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid config payload in {path}")
    return payload


def _parse_command(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return shlex.split(raw, posix=(os.name != "nt"))
    return []


def _as_env_dict(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        key = str(k or "").strip()
        if not key:
            continue
        out[key] = str(v if v is not None else "")
    return out


def _normalize_tasks(config_payload: dict[str, Any]) -> tuple[list[Task], dict[str, Any], dict[str, str]]:
    root = config_payload.get("training_priority")
    if not isinstance(root, dict):
        raise SystemExit("Config must contain top-level training_priority object")

    defaults = root.get("defaults") if isinstance(root.get("defaults"), dict) else {}
    default_env = _as_env_dict(defaults.get("env") if isinstance(defaults, dict) else {})
    raw_tasks = root.get("tasks")
    if not isinstance(raw_tasks, list):
        raise SystemExit("training_priority.tasks must be a list")

    tasks: list[Task] = []
    for idx, raw in enumerate(raw_tasks, start=1):
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("id") or f"task_{idx}").strip()
        command = _parse_command(raw.get("command"))
        if not command:
            continue
        tasks.append(
            Task(
                task_id=task_id,
                description=str(raw.get("description") or "").strip(),
                profiles=_as_list(raw.get("profiles")) or ["all"],
                targets=_as_list(raw.get("targets")) or ["local_cron"],
                priority=max(1, int(raw.get("priority", 100))),
                command=command,
                required_paths=_as_list(raw.get("required_paths")),
                timeout_seconds=max(10.0, float(raw.get("timeout_seconds", 900))),
                critical=bool(raw.get("critical", False)),
                enabled=bool(raw.get("enabled", True)),
                env=_as_env_dict(raw.get("env")),
            )
        )

    tasks.sort(key=lambda t: (t.priority, t.task_id))
    return tasks, defaults, default_env


def _resolve_python_command(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    first = str(tokens[0]).strip()
    lower = first.lower()
    if lower in {"python", "python3"}:
        return [sys.executable, *tokens[1:]]
    rel = Path(first)
    if rel.suffix.lower() == ".py":
        script_path = (PROJECT_ROOT / rel).resolve()
        return [sys.executable, str(script_path), *tokens[1:]]
    return tokens


def _path_exists(rel_or_abs: str) -> bool:
    p = Path(rel_or_abs).expanduser()
    if p.is_absolute():
        return p.exists()
    return (PROJECT_ROOT / p).exists()


def _tail(text: str, max_lines: int) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _run_task(
    task: Task,
    *,
    dry_run: bool,
    skip_missing_prereqs: bool,
    output_tail_lines: int,
    base_env: dict[str, str],
) -> dict[str, Any]:
    missing = [p for p in task.required_paths if not _path_exists(p)]
    if missing and skip_missing_prereqs:
        return {
            "task_id": task.task_id,
            "status": "SKIPPED_MISSING_PREREQS",
            "priority": task.priority,
            "critical": task.critical,
            "missing_prereqs": missing,
            "command": task.command,
            "env_keys": sorted(set(base_env.keys()).union(task.env.keys())),
        }

    cmd = _resolve_python_command(task.command)
    if dry_run:
        return {
            "task_id": task.task_id,
            "status": "DRY_RUN",
            "priority": task.priority,
            "critical": task.critical,
            "missing_prereqs": missing,
            "command": cmd,
            "env_keys": sorted(set(base_env.keys()).union(task.env.keys())),
        }

    started = datetime.now(timezone.utc).isoformat()
    proc_env = dict(os.environ)
    # Apply config defaults only when caller has not already provided a value.
    for k, v in base_env.items():
        proc_env.setdefault(k, v)
    proc_env.update(task.env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env=proc_env,
            timeout=float(task.timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "task_id": task.task_id,
            "status": "FAIL_TIMEOUT",
            "priority": task.priority,
            "critical": task.critical,
            "started_at_utc": started,
            "timeout_seconds": float(task.timeout_seconds),
            "command": cmd,
            "missing_prereqs": missing,
        }
    except Exception as exc:
        return {
            "task_id": task.task_id,
            "status": "FAIL_EXCEPTION",
            "priority": task.priority,
            "critical": task.critical,
            "started_at_utc": started,
            "command": cmd,
            "missing_prereqs": missing,
            "error": str(exc),
        }

    status = "PASS" if int(proc.returncode) == 0 else "FAIL"
    return {
        "task_id": task.task_id,
        "status": status,
        "priority": task.priority,
        "critical": task.critical,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": int(proc.returncode),
        "command": cmd,
        "missing_prereqs": missing,
        "env_keys": sorted(set(base_env.keys()).union(task.env.keys())),
        "stdout_tail": _tail(proc.stdout or "", output_tail_lines),
        "stderr_tail": _tail(proc.stderr or "", output_tail_lines),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to training priority YAML config.")
    parser.add_argument("--profile", choices=["all", "forecasters", "llm"], default="all", help="Task profile to run.")
    parser.add_argument("--target", default="local_cron", help="Execution target tag (e.g., local_cron, ci).")
    parser.add_argument("--max-priority", type=int, default=0, help="Optional max priority cutoff (inclusive).")
    parser.add_argument("--dry-run", action="store_true", help="Plan tasks without executing commands.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep executing tasks after failures.")
    parser.add_argument("--skip-missing-prereqs", action="store_true", help="Skip tasks with missing required paths.")
    parser.add_argument("--no-skip-missing-prereqs", dest="skip_missing_prereqs", action="store_false", help="Fail tasks when prereqs are missing.")
    parser.add_argument("--output-json", default="logs/automation/training_priority/training_priority_latest.json", help="Path to write summary JSON.")
    parser.set_defaults(skip_missing_prereqs=True)
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser()
    config_payload = _safe_load_config(config_path)
    tasks, defaults, default_env = _normalize_tasks(config_payload)
    output_tail_lines = max(10, int((defaults or {}).get("max_output_tail_lines", 40)))

    selected: list[Task] = []
    for task in tasks:
        if not task.enabled:
            continue
        if args.profile != "all" and args.profile not in task.profiles:
            continue
        if args.target not in task.targets:
            continue
        if int(args.max_priority or 0) > 0 and task.priority > int(args.max_priority):
            continue
        selected.append(task)

    results: list[dict[str, Any]] = []
    for task in selected:
        row = _run_task(
            task,
            dry_run=bool(args.dry_run),
            skip_missing_prereqs=bool(args.skip_missing_prereqs),
            output_tail_lines=output_tail_lines,
            base_env=default_env,
        )
        results.append(row)
        status = str(row.get("status") or "")
        print(f"[training_cycle] p{task.priority:03d} {task.task_id}: {status}")

        failed = status.startswith("FAIL")
        if failed and task.critical and not args.continue_on_error:
            print("[training_cycle] critical failure; stopping remaining tasks.")
            break
        if failed and not args.continue_on_error:
            print("[training_cycle] failure encountered; stopping (use --continue-on-error to continue).")
            break

    failures = [r for r in results if str(r.get("status") or "").startswith("FAIL")]
    skipped = [r for r in results if str(r.get("status") or "").startswith("SKIPPED")]
    passed = [r for r in results if str(r.get("status") or "") == "PASS"]
    dry = [r for r in results if str(r.get("status") or "") == "DRY_RUN"]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "profile": args.profile,
        "target": args.target,
        "dry_run": bool(args.dry_run),
        "selected_tasks": len(selected),
        "results": {
            "pass": len(passed),
            "fail": len(failures),
            "skipped": len(skipped),
            "dry_run": len(dry),
        },
        "tasks": results,
    }

    output_path = Path(args.output_json).expanduser()
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[training_cycle] summary={output_path}")

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
