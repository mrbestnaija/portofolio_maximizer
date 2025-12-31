#!/usr/bin/env python
"""Sanitize cache/log artifacts older than a retention window.

This helper removes stale cached data and log artifacts to keep disk/memory
footprint bounded. Defaults to a 14-day retention window and targets common
cache/log locations.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _iter_files(paths: Iterable[Path], patterns: Sequence[str]) -> Iterable[Path]:
    for base in paths:
        if not base.exists():
            continue
        for pat in patterns:
            yield from base.rglob(pat)


def _prune_files(files: Iterable[Path], cutoff_ts: float, exclude: Sequence[str]) -> int:
    count = 0
    for f in files:
        if any(str(f).startswith(ex) for ex in exclude):
            continue
        try:
            if f.stat().st_mtime < cutoff_ts:
                f.unlink(missing_ok=True)
                count += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Skip pruning %s: %s", f, exc)
    return count


def sanitize_cache_and_logs(
    retention_days: int = 14,
    data_dirs: Sequence[str] | None = None,
    log_dirs: Sequence[str] | None = None,
    patterns: Sequence[str] | None = None,
    exclude_prefixes: Sequence[str] | None = None,
) -> int:
    """Remove cached data/log artifacts older than retention_days.

    Args:
        retention_days: age threshold in days.
        data_dirs: directories to scan for cached data.
        log_dirs: directories to scan for logs/reports.
        patterns: glob patterns to match.
        exclude_prefixes: path prefixes to skip (e.g., dvc stores).
    Returns:
        Count of deleted files.
    """
    data_dirs = data_dirs or [
        "data/raw",
        "data/processed",
        "data/training",
        "data/validation",
        "data/testing",
    ]
    log_dirs = log_dirs or [
        "logs",
        "logs/automation",
        "logs/forecast_audits",
        "visualizations",
    ]
    patterns = patterns or ["*.parquet", "*.json", "*.jsonl", "*.log", "*.png", "*.html", "*.csv", "*.pdf", "*.svg"]
    exclude_prefixes = exclude_prefixes or ["data/dvcstore", ".dvc"]

    paths = [Path(p) for p in (*data_dirs, *log_dirs)]
    cutoff = datetime.now() - timedelta(days=retention_days)
    files = list(_iter_files(paths, patterns))
    deleted = _prune_files(files, cutoff.timestamp(), exclude_prefixes)
    if deleted:
        logger.info("Pruned %d stale files older than %d days", deleted, retention_days)
    else:
        logger.info("No stale files found for pruning (retention=%d days)", retention_days)
    return deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune cache/log artifacts older than retention window.")
    parser.add_argument("--retention-days", type=int, default=14, help="Retention window in days (default: 14).")
    parser.add_argument(
        "--data-dirs",
        default=None,
        help="Comma-separated list of data directories to prune (default targets raw/processed/training/validation/testing).",
    )
    parser.add_argument(
        "--log-dirs",
        default=None,
        help="Comma-separated list of log/report dirs to prune (default targets logs/, logs/automation, logs/forecast_audits).",
    )
    parser.add_argument(
        "--patterns",
        default=None,
        help="Comma-separated glob patterns to prune (default includes parquet,json,jsonl,log,png,html,csv,pdf,svg).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dirs = args.data_dirs.split(",") if args.data_dirs else None
    log_dirs = args.log_dirs.split(",") if args.log_dirs else None
    patterns = args.patterns.split(",") if args.patterns else None
    sanitize_cache_and_logs(
        retention_days=args.retention_days,
        data_dirs=data_dirs,
        log_dirs=log_dirs,
        patterns=patterns,
    )


if __name__ == "__main__":
    main()
