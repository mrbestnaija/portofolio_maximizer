from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(root: Path) -> Optional[str]:
    if not (root / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def build_config_provenance(paths: Iterable[Path]) -> Dict[str, Any]:
    entries: list[Dict[str, Any]] = []
    bundle_lines: list[str] = []
    for raw_path in paths:
        resolved = Path(raw_path).expanduser().resolve()
        entry: Dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
        if resolved.exists() and resolved.is_file():
            file_hash = sha256_file(resolved)
            entry["sha256"] = file_hash
            bundle_lines.append(f"{resolved.as_posix()}:{file_hash}")
        entries.append(entry)
    return {
        "config_paths": [entry["path"] for entry in entries],
        "config_hash": sha256_text("\n".join(sorted(bundle_lines))) if bundle_lines else None,
        "config_files": entries,
    }


def build_dataset_hash(
    *,
    db_path: Optional[Path],
    tickers: Iterable[str],
    start_date: Optional[str],
    end_date: Optional[str],
    db_max_ohlcv_date: Optional[str],
) -> str:
    resolved_db_path = Path(db_path).expanduser().resolve() if db_path else None
    db_file_hash = (
        sha256_file(resolved_db_path)
        if resolved_db_path is not None and resolved_db_path.exists()
        else None
    )
    payload = {
        "db_path": str(resolved_db_path) if resolved_db_path is not None else None,
        "db_file_hash": db_file_hash,
        "tickers": sorted(str(t).upper() for t in tickers),
        "start_date": start_date,
        "end_date": end_date,
        "db_max_ohlcv_date": db_max_ohlcv_date,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))
