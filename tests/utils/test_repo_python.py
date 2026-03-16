from __future__ import annotations

from pathlib import Path

import pytest

from utils import repo_python as mod


def test_resolve_repo_python_prefers_windows_layout_on_windows(monkeypatch, tmp_path: Path) -> None:
    windows_python = tmp_path / "simpleTrader_env" / "Scripts" / "python.exe"
    posix_python = tmp_path / "simpleTrader_env" / "bin" / "python"
    windows_python.parent.mkdir(parents=True)
    posix_python.parent.mkdir(parents=True)
    windows_python.write_text("", encoding="utf-8")
    posix_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(mod.os, "name", "nt")

    assert mod.resolve_repo_python(tmp_path) == str(windows_python)


def test_resolve_repo_python_raises_when_simpletrader_env_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Required repo interpreter not found"):
        mod.resolve_repo_python(tmp_path)
