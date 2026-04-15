"""Canonical environment-flag parser for Portfolio Maximizer.

All env-flag reads that affect trade classification (synthetic vs real) and
execution routing MUST go through this module. Using raw os.getenv() on a
boolean flag is a silent failure mode: os.getenv("SYNTHETIC_ONLY") returns
the string "0" when the variable is set to 0, and bool("0") is True in Python.

Single source of truth:
  - parse_env_bool(name) — returns bool (not Optional[bool])
  - is_synthetic_mode()  — returns bool, checks SYNTHETIC_ONLY + DATA_SOURCE
"""

import os
from typing import Optional

__all__ = ["parse_env_bool", "is_synthetic_mode"]

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off", ""})


def parse_env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable correctly.

    Accepted truthy values  : "1", "true", "yes", "on"  (case-insensitive)
    Accepted falsy values   : "0", "false", "no", "off", "" (empty string)
    Unset (None)            : returns *default*
    Any other value         : logs a warning and returns *default*

    Unlike ``bool(os.getenv(name))``, this function never treats the string
    "0" or "false" as True.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        Parsed boolean value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    normalised = raw.strip().lower()
    if normalised in _TRUTHY:
        return True
    if normalised in _FALSY:
        return False
    # Unexpected value — warn and fall back to default.
    import logging
    logging.getLogger(__name__).warning(
        "Unexpected value for env flag %s=%r; expected one of %s — using default=%s",
        name, raw, sorted(_TRUTHY | _FALSY), default,
    )
    return default


def is_synthetic_mode(
    *,
    execution_mode: Optional[str] = None,
) -> bool:
    """Return True when the runtime should be treated as synthetic.

    Decision hierarchy (first match wins):
      1. SYNTHETIC_ONLY=1/true/yes → synthetic
      2. DATA_SOURCE=synthetic or PMX_PREFERRED_DATA_SOURCE=synthetic → synthetic
      3. execution_mode arg == "synthetic" → synthetic
      4. EXECUTION_MODE=synthetic env var → synthetic
      5. Everything else → real

    This is the single authoritative check for whether trades should be
    classified as is_synthetic=1 at the execution layer.  Call it once and
    pass the result down; never re-read SYNTHETIC_ONLY with os.getenv().

    Args:
        execution_mode: Optional override from caller context (e.g. CLI flag).

    Returns:
        True  → treat as synthetic (is_synthetic=1 on trade records).
        False → treat as real     (is_synthetic=0 on trade records).
    """
    if parse_env_bool("SYNTHETIC_ONLY"):
        return True
    for env_name in ("DATA_SOURCE", "PMX_PREFERRED_DATA_SOURCE"):
        src = (os.getenv(env_name) or "").strip().lower()
        if src == "synthetic":
            return True
    if execution_mode is not None and str(execution_mode).lower() == "synthetic":
        return True
    env_exec = (os.getenv("EXECUTION_MODE") or os.getenv("PMX_EXECUTION_MODE") or "").strip().lower()
    if env_exec == "synthetic":
        return True
    return False
