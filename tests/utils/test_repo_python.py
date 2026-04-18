from __future__ import annotations

from pathlib import Path

from utils.repo_python import resolve_repo_python


def test_resolve_repo_python_prefers_pmx_python_bin(monkeypatch, tmp_path: Path) -> None:
    env_python = tmp_path / "custom" / "python.exe"
    env_python.parent.mkdir(parents=True, exist_ok=True)
    env_python.write_text("", encoding="utf-8")

    windows_win_python = tmp_path / "simpleTrader_env_win" / "Scripts" / "python.exe"
    windows_win_python.parent.mkdir(parents=True, exist_ok=True)
    windows_win_python.write_text("", encoding="utf-8")

    monkeypatch.setenv("PMX_PYTHON_BIN", str(env_python))

    assert resolve_repo_python(tmp_path) == str(env_python)


def test_resolve_repo_python_prefers_simpletrader_env_win_before_legacy_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PMX_PYTHON_BIN", raising=False)

    windows_win_python = tmp_path / "simpleTrader_env_win" / "Scripts" / "python.exe"
    windows_win_python.parent.mkdir(parents=True, exist_ok=True)
    windows_win_python.write_text("", encoding="utf-8")

    legacy_python = tmp_path / "simpleTrader_env" / "Scripts" / "python.exe"
    legacy_python.parent.mkdir(parents=True, exist_ok=True)
    legacy_python.write_text("", encoding="utf-8")

    assert resolve_repo_python(tmp_path) == str(windows_win_python)


def test_resolve_repo_python_falls_back_to_legacy_env_when_win_variant_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PMX_PYTHON_BIN", raising=False)

    legacy_python = tmp_path / "simpleTrader_env" / "Scripts" / "python.exe"
    legacy_python.parent.mkdir(parents=True, exist_ok=True)
    legacy_python.write_text("", encoding="utf-8")

    assert resolve_repo_python(tmp_path) == str(legacy_python)
