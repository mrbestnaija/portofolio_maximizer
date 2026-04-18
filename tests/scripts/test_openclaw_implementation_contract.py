"""
Repo-wide OpenClaw implementation contract tests.

These are anti-regression guards for hard policy invariants. They enforce
cross-file wiring consistency and fail if any required OpenClaw safety path
drifts.
"""

from __future__ import annotations

from pathlib import Path

from scripts import enforce_openclaw_exec_environment as enforce_mod
from scripts import project_runtime_status as runtime_mod
from scripts import verify_openclaw_config as verify_mod


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_repo_file(relative_path: str) -> str:
    path = _repo_root() / relative_path
    return path.read_text(encoding="utf-8")


def test_exec_host_allowlist_is_consistent_repo_wide() -> None:
    expected = {"sandbox", "gateway", "node"}
    assert enforce_mod.VALID_EXEC_HOSTS == expected
    assert runtime_mod.VALID_EXEC_HOSTS == expected
    assert verify_mod.VALID_EXEC_HOSTS == expected


def test_sandbox_host_modes_are_consistent_repo_wide() -> None:
    expected = {"non-main", "all"}
    assert enforce_mod.VALID_SANDBOX_MODES_FOR_SANDBOX_HOST == expected
    assert runtime_mod.VALID_SANDBOX_MODES_FOR_SANDBOX_HOST == expected
    assert verify_mod.VALID_SANDBOX_MODES_FOR_SANDBOX_HOST == expected


def test_maintenance_wrapper_enforces_exec_env_before_maintenance() -> None:
    text = _read_repo_file("scripts/run_openclaw_maintenance.ps1")
    assert "simpleTrader_env_win\\Scripts\\python.exe" in text
    assert '$enforceCmd = "cd \'$repoWsl\' && python scripts/enforce_openclaw_exec_environment.py"' in text
    assert "& wsl bash -lc $enforceCmd" in text
    assert '$execEnvArgs = @("scripts/enforce_openclaw_exec_environment.py")' in text
    assert "& $pythonExe @execEnvArgs" in text
    assert text.index("& wsl bash -lc $enforceCmd") < text.index("bash/production_cron.sh openclaw_maintenance")


def test_guardian_enforces_exec_env_before_watch_launch() -> None:
    text = _read_repo_file("scripts/start_openclaw_guardian.ps1")
    assert "simpleTrader_env_win\\Scripts\\python.exe" in text
    assert '$remoteWorkflowScript = Join-Path $repoRoot "scripts\\openclaw_remote_workflow.py"' in text
    assert '$execEnvScript = Join-Path $repoRoot "scripts\\enforce_openclaw_exec_environment.py"' in text
    assert '$execEnvArgs = @($execEnvScript)' in text
    assert "& $pythonExe @execEnvArgs" in text
    assert '[switch]$EnsureFunctionalState' in text
    assert '[switch]$Quiet' in text
    assert 'Invoke-FunctionalRecovery' in text
    assert '$args = @($remoteWorkflowScript) + $Arguments' in text
    assert 'health", "--json' in text
    assert 'gateway-restart", "--json' in text
    assert '[bool]$DisableBrokenChannels = $true' in text
    assert '$args += "--disable-broken-channels"' in text
    assert 'OPENCLAW_FAST_SUPERVISOR_FAILURE_THRESHOLD' in text
    assert 'OPENCLAW_FAST_SUPERVISOR_RESTART_COOLDOWN_SECONDS' in text
    assert 'OPENCLAW_PRIMARY_RESTART_ATTEMPTS' in text
    assert "$proc = Start-Process" in text
    assert text.index("& $pythonExe @execEnvArgs") < text.index("$proc = Start-Process")


def test_runtime_status_exposes_required_exec_env_signals() -> None:
    source = _read_repo_file("scripts/project_runtime_status.py")
    for token in (
        "invalid_exec_host",
        "invalid_sandbox_mode",
        "sandbox_runtime_unavailable",
        "missing_acp_default_agent",
        "exec_env_valid",
    ):
        assert token in source


def test_policy_doc_exists_and_mentions_required_contracts() -> None:
    path = _repo_root() / "Documentation" / "OPENCLAW_IMPLEMENTATION_POLICY.md"
    assert path.exists(), "Policy doc missing: Documentation/OPENCLAW_IMPLEMENTATION_POLICY.md"
    text = path.read_text(encoding="utf-8").lower()
    required = (
        "tools.exec.host",
        "sandbox",
        "acp.defaultagent",
        "enforce_openclaw_exec_environment.py",
        "project_runtime_status.py",
        "utf-8 bom",
        "coding",
        "apply_patch",
        "invalid_exec_host",
        "invalid_sandbox_mode",
        "sandbox_runtime_unavailable",
        "missing_acp_default_agent",
        "exec_env_valid",
        "openclaw_storm_guard_enabled",
        "openclaw_storm_base_cooldown_seconds",
        "openclaw_storm_max_cooldown_seconds",
        "openclaw_storm_backoff_multiplier",
        "openclaw_storm_reset_window_seconds",
    )
    for token in required:
        assert token in text, f"Policy doc missing contract token: {token}"


def test_openclaw_cli_implements_persistent_storm_guard_contract() -> None:
    source = _read_repo_file("utils/openclaw_cli.py").lower()
    required = (
        "openclaw_storm_guard_enabled",
        "openclaw_storm_base_cooldown_seconds",
        "openclaw_storm_max_cooldown_seconds",
        "openclaw_storm_backoff_multiplier",
        "openclaw_storm_reset_window_seconds",
        "storm_failures",
        "suppressed notification storm",
        "record_delivery_result",
        "_classify_notification_storm_error",
    )
    for token in required:
        assert token in source, f"OpenClaw storm guard contract token missing: {token}"
