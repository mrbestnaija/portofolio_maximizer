"""
Utility helpers for capturing warnings and writing them to the project logs.

The brutal-run documentation calls for recording suppressed warnings so they can
be audited later instead of silently ignoring them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable
import warnings

_WARNING_LOGGER: logging.Logger | None = None


def _build_warning_logger() -> logging.Logger:
    log_dir = Path("logs/warnings")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("warning_recorder")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid attaching duplicate handlers when multiple modules import this helper.
    handler_exists = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "_pm_warning_handler", False)
        for handler in logger.handlers
    )

    if not handler_exists:
        handler = logging.FileHandler(log_dir / "warning_events.log")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        handler._pm_warning_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    return logger


def _get_warning_logger() -> logging.Logger:
    global _WARNING_LOGGER
    if _WARNING_LOGGER is None:
        _WARNING_LOGGER = _build_warning_logger()
    return _WARNING_LOGGER


def log_warning(message: str, context: str) -> None:
    """Write a single warning message to the warning log."""
    logger = _get_warning_logger()
    logger.warning("[%s] %s", context, message)


def log_warning_records(
    records: Iterable[warnings.WarningMessage], context: str
) -> None:
    """Persist warnings captured via warnings.catch_warnings(record=True)."""
    records = list(records)
    if not records:
        return
    logger = _get_warning_logger()
    for record in records:
        # Filter out expected statsmodels startup diagnostics that are
        # explicitly treated as non-fatal in the forecasting stack.
        if issubclass(record.category, UserWarning):
            message_text = str(record.message)
            if (
                "Non-stationary starting autoregressive parameters found" in message_text
                or "Non-invertible starting MA parameters found" in message_text
            ):
                continue
        logger.warning(
            "[%s] %s: %s (filename=%s lineno=%s)",
            context,
            record.category.__name__,
            record.message,
            record.filename,
            record.lineno,
        )
