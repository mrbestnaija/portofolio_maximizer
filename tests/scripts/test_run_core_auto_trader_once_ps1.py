from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_text(relative_path: str) -> str:
    return (_repo_root() / relative_path).read_text(encoding="utf-8")


def test_ps1_wrapper_defers_to_bat_and_market_hours_guard() -> None:
    text = _read_text("scripts/run_core_auto_trader_once.ps1")

    assert '$wrapper = Join-Path $repoRoot "run_core_auto_trader_once.bat"' in text
    assert "Test-MarketHours" in text
    assert "[DayOfWeek]::Saturday" in text
    assert "[DayOfWeek]::Sunday" in text
    assert "& $wrapper" in text


def test_docs_reference_the_task_scheduler_friendly_entrypoint() -> None:
    cron_docs = _read_text("Documentation/CRON_AUTOMATION.md")
    bash_readme = _read_text("bash/BASH_README.md")
    core_docs = _read_text("Documentation/CORE_PROJECT_DOCUMENTATION.md")

    assert "run_core_auto_trader_once.ps1" in cron_docs
    assert "run_core_auto_trader_once.ps1" in bash_readme
    assert "run_core_auto_trader_once.ps1" in core_docs
