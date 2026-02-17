"""
Ephemeral Git HTTPS credentials helper.

This file is intentionally small and prints ONLY the requested credential value
to stdout for Git (username or PAT). It must never log or print anything else.

Note: Do not commit this file long-term. Prefer `gh auth login` / credential
manager where possible. This helper exists to unblock non-interactive pushes in
automation contexts where prompts are impossible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict


def _parse_env_file(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return mapping

    for raw_line in lines:
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue

        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue

        key = key.strip()
        if not key or key in mapping:
            continue

        cleaned = value.strip().strip('"').strip("'").strip()
        if not cleaned:
            continue
        mapping[key] = cleaned

    return mapping


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    env_map = _parse_env_file(env_path) if env_path.exists() else {}

    prompt = argv[1] if len(argv) > 1 else ""
    want_user = "username" in prompt.lower()

    username = (env_map.get("GitHub_Username") or os.getenv("GitHub_Username") or "").strip()
    token = (env_map.get("GitHub_TOKEN") or os.getenv("GitHub_TOKEN") or "").strip()

    sys.stdout.write((username if want_user else token) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))

