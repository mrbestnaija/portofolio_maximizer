#!/usr/bin/env python3
"""
Sanitize log artifacts by removing corrupt JSONL entries, stale PID files,
and empty directories under a logs root.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                check=False,
            )
            return str(pid) in (result.stdout or "")
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    return True


def _iter_jsonl_files(root: Path, include_archive: bool) -> Iterable[Path]:
    for path in root.rglob("*.jsonl"):
        if not include_archive and "archive" in path.parts:
            continue
        yield path


def _sanitize_jsonl(
    path: Path, archive_root: Path, dry_run: bool
) -> Tuple[int, int, int]:
    total = 0
    empty = 0
    invalid = 0
    valid_lines: list[str] = []
    bad_lines: list[str] = []

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="replace")

    for line in raw.splitlines():
        total += 1
        stripped = line.strip()
        if not stripped:
            empty += 1
            bad_lines.append(line)
            continue
        try:
            json.loads(stripped)
            valid_lines.append(stripped)
        except Exception:
            invalid += 1
            bad_lines.append(line)

    if (empty or invalid) and not dry_run:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        rel = path.relative_to(path.parents[1]) if len(path.parents) > 1 else path.name
        archive_dir = archive_root / stamp / rel.parent
        archive_dir.mkdir(parents=True, exist_ok=True)
        if bad_lines:
            archive_path = archive_dir / f"{path.stem}.corrupt.jsonl"
            archive_path.write_text("\n".join(bad_lines) + "\n", encoding="utf-8")
        # Rewrite with valid lines only
        new_payload = "\n".join(valid_lines) + ("\n" if valid_lines else "")
        path.write_text(new_payload, encoding="utf-8")

    return total, empty, invalid


def _remove_empty_dirs(root: Path, dry_run: bool) -> list[Path]:
    removed: list[Path] = []
    for directory in sorted(root.rglob("*"), reverse=True):
        if not directory.is_dir():
            continue
        try:
            if any(directory.iterdir()):
                continue
        except Exception:
            continue
        if not dry_run:
            try:
                directory.rmdir()
            except Exception:
                continue
        removed.append(directory)
    return removed


def _remove_stale_pid_files(root: Path, include_archive: bool, dry_run: bool) -> list[Path]:
    removed: list[Path] = []
    for path in root.rglob("*.pid"):
        if not include_archive and "archive" in path.parts:
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
            pid = int(content)
        except Exception:
            pid = -1
        if pid <= 0 or not _pid_exists(pid):
            if not dry_run:
                try:
                    path.unlink()
                except Exception:
                    continue
            removed.append(path)
    return removed


def _remove_zero_byte_files(root: Path, include_archive: bool, dry_run: bool) -> list[Path]:
    removed: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not include_archive and "archive" in path.parts:
            continue
        try:
            if path.stat().st_size != 0:
                continue
        except Exception:
            continue
        if not dry_run:
            try:
                path.unlink()
            except Exception:
                continue
        removed.append(path)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanitize logs directory.")
    parser.add_argument("--logs-root", default="logs", help="Logs root directory.")
    parser.add_argument(
        "--archive-root",
        default="logs/archive/sanitize",
        help="Archive root for corrupt entries.",
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Include logs/archive in sanitation scope.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report findings without modifying files.",
    )
    parser.add_argument(
        "--keep-zero-byte",
        action="store_true",
        help="Do not delete zero-byte files.",
    )
    args = parser.parse_args()

    logs_root = Path(args.logs_root)
    archive_root = Path(args.archive_root)

    if not logs_root.exists():
        print(f"[SANITIZE] Logs root not found: {logs_root}")
        return 1

    print(f"[SANITIZE] Logs root: {logs_root}")
    print(f"[SANITIZE] Archive root: {archive_root}")
    print(f"[SANITIZE] Include archive: {args.include_archive}")
    print(f"[SANITIZE] Dry run: {args.dry_run}")

    total_files = 0
    total_invalid = 0
    total_empty = 0

    for jsonl_path in _iter_jsonl_files(logs_root, args.include_archive):
        total, empty, invalid = _sanitize_jsonl(jsonl_path, archive_root, args.dry_run)
        if empty or invalid:
            print(
                f"[JSONL] {jsonl_path} -> total={total} empty={empty} invalid={invalid}"
            )
        total_files += 1
        total_invalid += invalid
        total_empty += empty

    removed_pids = _remove_stale_pid_files(logs_root, args.include_archive, args.dry_run)
    if removed_pids:
        for path in removed_pids:
            print(f"[PID] Removed stale pid file: {path}")

    removed_zero = []
    if not args.keep_zero_byte:
        removed_zero = _remove_zero_byte_files(logs_root, args.include_archive, args.dry_run)
        if removed_zero:
            for path in removed_zero:
                print(f"[ZERO] Removed empty file: {path}")

    removed_dirs = _remove_empty_dirs(logs_root, args.dry_run)
    if removed_dirs:
        for path in removed_dirs:
            print(f"[DIR] Removed empty dir: {path}")

    print(
        f"[SUMMARY] JSONL files scanned={total_files} "
        f"empty_lines={total_empty} invalid_lines={total_invalid}"
    )
    print(
        f"[SUMMARY] Stale pid files removed={len(removed_pids)}; "
        f"zero-byte files removed={len(removed_zero)}; "
        f"empty dirs removed={len(removed_dirs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
