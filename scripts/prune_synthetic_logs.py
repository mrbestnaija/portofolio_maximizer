#!/usr/bin/env python
"""Prune synthetic run logs and artifacts to enforce retention."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune synthetic logs/manifests older than retention window.")
    parser.add_argument("--log-dir", default="logs/automation", help="Directory containing synthetic log files.")
    parser.add_argument("--retention-days", type=int, default=14, help="Retention window in days.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _parse_ts(ts: str) -> datetime | None:
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def prune_jsonl(log_path: Path, cutoff: datetime) -> None:
    lines = log_path.read_text().splitlines()
    kept = []
    for line in lines:
        try:
            obj = json.loads(line)
            ts = obj.get("timestamp")
            dt = _parse_ts(ts) if isinstance(ts, str) else None
            if dt and dt >= cutoff:
                kept.append(line)
        except Exception:
            # Keep unparsable lines to avoid data loss
            kept.append(line)
    log_path.write_text("\n".join(kept) + ("\n" if kept else ""))


def prune_files(files: Iterable[Path], cutoff_ts: float) -> None:
    for f in files:
        try:
            if f.stat().st_mtime < cutoff_ts:
                f.unlink()
                logging.info("Pruned %s", f)
        except FileNotFoundError:
            continue
        except Exception as exc:
            logging.debug("Skip pruning %s: %s", f, exc)


def main() -> None:
    args = parse_args()
    configure_logging()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        return

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=args.retention_days)
    cutoff_ts = cutoff_dt.timestamp()

    # Prune JSONL runs log
    runs_log = log_dir / "synthetic_runs.log"
    if runs_log.exists():
        prune_jsonl(runs_log, cutoff_dt)

    # Prune per-run logs/reports/manifests older than cutoff
    patterns = [
        "synthetic_validation_*.json",
        "syn_*.log",
        "syn_*.json",
        "syn_*.txt",
    ]
    files = []
    for pattern in patterns:
        files.extend(log_dir.glob(pattern))
    prune_files(files, cutoff_ts)


if __name__ == "__main__":
    main()
