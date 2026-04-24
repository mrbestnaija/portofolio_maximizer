from __future__ import annotations

import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_REPO_PYTHON_WIN = REPO_ROOT / "simpleTrader_env_win" / "Scripts" / "python.exe"
WINDOWS_REPO_PYTHON = REPO_ROOT / "simpleTrader_env" / "Scripts" / "python.exe"
POSIX_REPO_PYTHON = REPO_ROOT / "simpleTrader_env" / "bin" / "python"


def _env_python_candidate() -> Path | None:
    raw = str(os.getenv("PMX_PYTHON_BIN", "")).strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return candidate
    return None


def candidate_repo_python_paths(repo_root: Path | None = None) -> list[Path]:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    windows_win_path = root / "simpleTrader_env_win" / "Scripts" / "python.exe"
    windows_path = root / "simpleTrader_env" / "Scripts" / "python.exe"
    posix_path = root / "simpleTrader_env" / "bin" / "python"
    env_candidate = _env_python_candidate()
    candidates: list[Path] = []
    if env_candidate is not None:
        candidates.append(env_candidate)
    candidates.extend([windows_win_path, windows_path, posix_path])
    if os.name != "nt":
        candidates = [posix_path, windows_win_path, windows_path]
        if env_candidate is not None:
            candidates.insert(0, env_candidate)
    return candidates


def resolve_repo_python(repo_root: Path | None = None) -> str:
    candidates = candidate_repo_python_paths(repo_root)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if shutil.which("python"):
        return "python"
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Required repo interpreter not found. Expected simpleTrader_env_win or simpleTrader_env at one of: "
        f"{rendered}"
    )
