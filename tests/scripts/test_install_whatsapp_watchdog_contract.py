from __future__ import annotations

from pathlib import Path


def _script_text() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "install_whatsapp_watchdog.ps1"
    return path.read_text(encoding="utf-8")


def test_watchdog_installer_declares_expected_tasks() -> None:
    text = _script_text()
    assert '$logonTask = "PMX-OpenClaw-Guardian-Logon"' in text
    assert '$keepAliveTask = "PMX-OpenClaw-Guardian-KeepAlive"' in text
    assert '$legacyTask = "PMX-OpenClaw-Maintenance"' in text


def test_watchdog_installer_targets_guardian_launcher_and_supports_uninstall() -> None:
    text = _script_text()
    assert 'start_openclaw_guardian.ps1' in text
    assert '[switch]$Uninstall' in text
    assert 'Remove-TaskIfPresent -TaskName $logonTask' in text
    assert 'Remove-TaskIfPresent -TaskName $keepAliveTask' in text


def test_watchdog_installer_disables_legacy_maintenance_by_default() -> None:
    text = _script_text()
    assert '[switch]$KeepLegacyMaintenanceTask' in text
    assert '"/Change", "/TN", $legacyTask, "/DISABLE"' in text
    assert 'Disable-LegacyMaintenanceTask' in text
