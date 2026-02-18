#!/usr/bin/env python3
"""
Prepare a local LLM fine-tuning dataset from PMX activity logs.

This script is intentionally conservative:
- it only uses logged prompt/response previews,
- redacts obvious secret-like patterns,
- can run an internal default trainer command when no explicit command is provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTIVITY_DIR = PROJECT_ROOT / "logs" / "llm_activity"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "training" / "llm_finetune" / "latest_dataset.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "logs" / "automation" / "training_priority" / "llm_finetune_dataset_summary.json"
DEFAULT_INTERNAL_FINETUNE_COMMAND = (
    "python scripts/train_local_instruction_lm.py "
    "--dataset {dataset_path} "
    "--output-dir models/llm_finetune/latest "
    "--epochs 2 "
    "--batch-size 8 "
    "--min-samples 8"
)

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9\-\._~\+/=]{16,}\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b"),
)


def _redact_text(text: str) -> str:
    out = str(text or "")
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out.strip()


def _safe_parse_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                raw = (line or "").strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    yield parsed
    except Exception:
        return


def _iter_recent_activity_files(activity_dir: Path, days: int) -> list[Path]:
    if not activity_dir.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
    out: list[Path] = []
    for path in sorted(activity_dir.glob("*.jsonl")):
        stem = path.stem
        try:
            day = datetime.strptime(stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if day >= cutoff:
            out.append(path)
    return out


def _extract_record(entry: dict[str, Any]) -> dict[str, Any] | None:
    etype = str(entry.get("type") or "").strip().lower()
    if etype == "llm_request":
        instruction = _redact_text(entry.get("prompt_preview") or "")
        output = _redact_text(entry.get("response_preview") or "")
        if not instruction or not output:
            return None
        return {
            "instruction": instruction,
            "output": output,
            "source": "llm_request",
            "model": str(entry.get("model") or ""),
            "task_type": str(entry.get("task_type") or ""),
        }
    if etype == "orchestration":
        instruction = _redact_text(entry.get("prompt_preview") or "")
        output = _redact_text(entry.get("response_preview") or "")
        if not instruction or not output:
            return None
        return {
            "instruction": instruction,
            "output": output,
            "source": "orchestration",
            "model": str(entry.get("orchestrator") or "qwen3:8b"),
            "task_type": "orchestration",
        }
    return None


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in records:
        key_raw = f"{row.get('instruction','')}|{row.get('output','')}"
        key = hashlib.sha256(key_raw.encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _run_trainer(command: str, *, dataset_path: Path, timeout_seconds: float) -> dict[str, Any]:
    raw = str(command or "").strip()
    if not raw:
        return {
            "status": "SKIPPED",
            "reason": "PMX_LLM_FINETUNE_COMMAND not configured",
        }

    expanded = (
        raw.replace("{dataset}", str(dataset_path))
        .replace("{dataset_path}", str(dataset_path))
        .replace("{project_root}", str(PROJECT_ROOT))
        .replace("{python_bin}", str(sys.executable))
    )
    parts = shlex.split(expanded, posix=(os.name != "nt"))
    if not parts:
        return {"status": "SKIPPED", "reason": "Trainer command empty after expansion"}
    if parts and str(parts[0]).strip().lower() in {"python", "python3"}:
        parts[0] = str(sys.executable)

    try:
        proc = subprocess.run(
            parts,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=max(30.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "FAIL",
            "reason": f"Trainer timeout after {timeout_seconds:.1f}s",
            "command": parts,
        }
    except Exception as exc:
        return {
            "status": "FAIL",
            "reason": f"Trainer failed to start: {exc}",
            "command": parts,
        }

    return {
        "status": "PASS" if int(proc.returncode) == 0 else "FAIL",
        "command": parts,
        "exit_code": int(proc.returncode),
        "stdout_tail": "\n".join((proc.stdout or "").splitlines()[-30:]),
        "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-30:]),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-dir", default=str(DEFAULT_ACTIVITY_DIR), help="Directory containing LLM activity JSONL logs.")
    parser.add_argument("--days", type=int, default=14, help="Lookback window in days.")
    parser.add_argument("--max-records", type=int, default=5000, help="Maximum records to keep in output dataset.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSONL dataset path.")
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY), help="Summary JSON output path.")
    parser.add_argument("--run-trainer", action="store_true", help="Run external trainer command after dataset creation.")
    parser.add_argument(
        "--trainer-command",
        default="",
        help="Optional trainer command; falls back to PMX_LLM_FINETUNE_COMMAND, then an internal default.",
    )
    parser.add_argument("--trainer-timeout-seconds", type=float, default=7200.0, help="Trainer timeout in seconds.")
    args = parser.parse_args(argv)

    activity_dir = Path(args.activity_dir).expanduser()
    output_path = Path(args.output).expanduser()
    summary_path = Path(args.summary_json).expanduser()

    files = _iter_recent_activity_files(activity_dir, days=args.days)
    records: list[dict[str, Any]] = []
    total_entries = 0
    for path in files:
        for entry in _safe_parse_jsonl(path):
            total_entries += 1
            row = _extract_record(entry)
            if row:
                records.append(row)

    deduped = _dedupe_records(records)
    if len(deduped) > max(1, int(args.max_records)):
        deduped = deduped[-int(args.max_records):]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in deduped:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    trainer_result = {"status": "SKIPPED", "reason": "not requested"}
    trainer_command_source = "none"
    if args.run_trainer:
        env_cmd = os.getenv("PMX_LLM_FINETUNE_COMMAND", "")
        if args.trainer_command:
            trainer_command = args.trainer_command
            trainer_command_source = "cli"
        elif env_cmd:
            trainer_command = env_cmd
            trainer_command_source = "env:PMX_LLM_FINETUNE_COMMAND"
        else:
            trainer_command = DEFAULT_INTERNAL_FINETUNE_COMMAND
            trainer_command_source = "internal_default"
        trainer_result = _run_trainer(
            trainer_command,
            dataset_path=output_path,
            timeout_seconds=float(args.trainer_timeout_seconds),
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "activity_dir": str(activity_dir),
        "days": int(args.days),
        "source_files": [str(p) for p in files],
        "source_entries_scanned": int(total_entries),
        "records_before_dedupe": int(len(records)),
        "records_written": int(len(deduped)),
        "dataset_path": str(output_path),
        "trainer": trainer_result,
        "trainer_command_source": trainer_command_source,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[llm_finetune] records_written={len(deduped)} dataset={output_path}")
    print(f"[llm_finetune] summary={summary_path}")
    if args.run_trainer:
        print(f"[llm_finetune] trainer_status={trainer_result.get('status')}")

    trainer_status = str((trainer_result or {}).get("status") or "").upper()
    if args.run_trainer and trainer_status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
