from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_text(relative_path: str) -> str:
    return (_repo_root() / relative_path).read_text(encoding="utf-8")


def test_windows_wrapper_invokes_core_cron_entrypoint() -> None:
    text = _read_text("run_core_auto_trader_once.bat")

    assert "where wsl >nul 2>&1" in text
    assert 'wsl --cd "%SCRIPT_DIR%" bash -lc "bash/production_cron.sh auto_trader_core"' in text
    assert "OpenClaw reporting stays in the" in text
