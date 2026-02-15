#!/usr/bin/env python3
"""Verify Anthropic API key works (no secrets printed).

This script performs a minimal authenticated request (models.list) to confirm
the configured key is valid.

Env vars (checked in order):
- ANTHROPIC_API_KEY (preferred)
- CLAUDE_API_KEY (fallback)
Both support Docker secrets via *_FILE using etl/secret_loader.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.secret_loader import load_secret


def _load_key() -> tuple[Optional[str], Optional[str]]:
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        val = load_secret(name)
        if val:
            return val, name
    return None, None


def main() -> int:
    key, used = _load_key()
    if not key:
        print("FAIL: Missing Anthropic API key (set ANTHROPIC_API_KEY or CLAUDE_API_KEY)")
        return 2

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=key)
        client.models.list(limit=1, timeout=10.0)
        print(f"PASS: Anthropic auth OK via {used}")
        return 0
    except Exception as e:
        status = getattr(e, "status_code", None)
        extra = f" status={status}" if status else ""
        print(f"FAIL: Anthropic API call failed ({type(e).__name__}){extra}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

