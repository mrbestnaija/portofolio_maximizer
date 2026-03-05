from __future__ import annotations

from pathlib import Path


def _script_text() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "run_openclaw_maintenance.ps1"
    return path.read_text(encoding="utf-8")


def test_wsl_path_runs_exec_environment_enforcement_before_maintenance() -> None:
    text = _script_text()
    enforce_decl = "$enforceCmd = \"cd '$repoWsl' && python scripts/enforce_openclaw_exec_environment.py\""
    enforce_run = "& wsl bash -lc $enforceCmd"
    maint_decl = "$cmd = \"cd '$repoWsl' && \" + ($envParts -join \" \") + \" bash/production_cron.sh openclaw_maintenance\""

    assert enforce_decl in text
    assert enforce_run in text
    assert maint_decl in text
    assert text.index(enforce_run) < text.index(maint_decl)
    assert '$enforceCmd += " --dry-run"' in text


def test_windows_path_still_enforces_exec_environment() -> None:
    text = _script_text()
    assert '$execEnvArgs = @("scripts/enforce_openclaw_exec_environment.py")' in text
    assert "& $pythonExe @execEnvArgs" in text
