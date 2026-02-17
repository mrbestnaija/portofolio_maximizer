"""
Centralized logging setup for Portfolio Maximizer.

Configures:
- UTC timestamps on all log formatters
- Size-based rotation (10MB, 5 backups) for general logs
- Time-based rotation (daily, 14 days) for automation logs
- Automatic cleanup of logs older than 30 days
- Structured JSON formatting option

Usage:
    from utils.logging_setup import configure_logging
    configure_logging()  # Call once at startup
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_ROOT = PROJECT_ROOT / "logs"


class _UTCFormatter(logging.Formatter):
    """Formatter that always uses UTC timestamps."""
    converter = time.gmtime


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_rotation: bool = True,
    enable_console: bool = True,
) -> None:
    """Configure centralized logging with UTC timestamps and rotation."""
    log_root = log_dir or LOGS_ROOT
    log_root.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to prevent duplicates on re-init
    if root_logger.handlers:
        for h in list(root_logger.handlers):
            if isinstance(h, (logging.handlers.RotatingFileHandler,
                              logging.handlers.TimedRotatingFileHandler)):
                root_logger.removeHandler(h)

    fmt = _UTCFormatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    # Console handler (if not already present)
    if enable_console and not any(isinstance(h, logging.StreamHandler) and
                                   not isinstance(h, logging.FileHandler)
                                   for h in root_logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root_logger.addHandler(ch)

    if enable_rotation:
        # Size-based rotation for main log
        main_log = log_root / "pipeline.log"
        rh = logging.handlers.RotatingFileHandler(
            str(main_log),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        rh.setFormatter(fmt)
        root_logger.addHandler(rh)

        # Time-based rotation for automation logs
        auto_dir = log_root / "automation"
        auto_dir.mkdir(parents=True, exist_ok=True)
        auto_log = auto_dir / "execution.log"
        th = logging.handlers.TimedRotatingFileHandler(
            str(auto_log),
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
            utc=True,
        )
        th.setFormatter(fmt)
        logging.getLogger("execution").addHandler(th)
        logging.getLogger("scripts").addHandler(th)


def cleanup_old_logs(
    *,
    log_dir: Optional[Path] = None,
    max_age_days: int = 30,
    exempt_dirs: Optional[list[str]] = None,
) -> int:
    """Remove log files older than max_age_days. Returns count of deleted files."""
    log_root = log_dir or LOGS_ROOT
    if not log_root.exists():
        return 0

    exempt = set(exempt_dirs or ["audit_sprint", "llm_activity"])
    cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)
    deleted = 0

    for f in log_root.rglob("*"):
        if not f.is_file():
            continue
        # Skip exempt directories
        if any(ex in f.parts for ex in exempt):
            continue
        # Skip non-log files
        if f.suffix not in {".log", ".jsonl", ".json", ".tmp"}:
            continue
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
        except OSError:
            continue

    return deleted
