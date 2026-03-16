from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_REPO_PYTHON = REPO_ROOT / "simpleTrader_env" / "Scripts" / "python.exe"
POSIX_REPO_PYTHON = REPO_ROOT / "simpleTrader_env" / "bin" / "python"


def candidate_repo_python_paths(repo_root: Path | None = None) -> list[Path]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    windows_path = root / "simpleTrader_env" / "Scripts" / "python.exe"
    posix_path = root / "simpleTrader_env" / "bin" / "python"
    if os.name == "nt":
        return [windows_path, posix_path]
    return [posix_path, windows_path]


def resolve_repo_python(repo_root: Path | None = None) -> str:
    candidates = candidate_repo_python_paths(repo_root)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Required repo interpreter not found. Expected simpleTrader_env at one of: "
        f"{rendered}"
    )
