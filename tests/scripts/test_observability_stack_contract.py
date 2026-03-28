from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (_repo_root() / path).read_text(encoding="utf-8")


def test_observability_doc_declares_low_cardinality_contract_and_shadow_mode() -> None:
    text = _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md").lower()
    for token in (
        "metric prefix: `pmx_`",
        "severity",
        "component",
        "channel",
        "job",
        "run_id",
        "message_id",
        "shadow mode",
        "grafana is ops/sre-only",
    ):
        assert token in text


def test_prometheus_and_alertmanager_configs_stay_localhost_only() -> None:
    prometheus = _read("observability/prometheus/prometheus.yml")
    rules = _read("observability/prometheus/rules/pmx_alerts.yml")
    alertmanager = _read("observability/alertmanager/alertmanager.yml")
    assert "127.0.0.1:9765" in prometheus
    assert "127.0.0.1:9093" in prometheus
    assert "127.0.0.1:9766/alertmanager" in alertmanager
    for token in (
        "pmx_openclaw_gateway_up",
        "pmx_cron_job_last_success_unixtime",
        "pmx_dashboard_snapshot_age_seconds",
        "pmx_openclaw_channels_status_latency_ms",
        "pmx_production_gate_pass",
    ):
        assert token in rules


def test_grafana_provisioning_disables_alerting_and_loads_four_dashboards() -> None:
    grafana_ini = _read("observability/grafana/grafana.ini").lower()
    providers = _read("observability/grafana/provisioning/dashboards/dashboard_providers.yml")
    datasource = _read("observability/grafana/provisioning/datasources/prometheus.yml")
    assert "[alerting]" in grafana_ini
    assert "enabled = false" in grafana_ini
    assert "pmx_grafana_dashboards_path" in providers.lower()
    assert "uid: prometheus" in datasource

    dashboard_dir = _repo_root() / "observability" / "grafana" / "dashboards"
    dashboard_files = sorted(dashboard_dir.glob("*.json"))
    assert len(dashboard_files) == 4
    for path in dashboard_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["refresh"] == "30s"
        assert payload["uid"].startswith("pmx-")


def test_windows_startup_scripts_and_installer_reference_repo_owned_launchers() -> None:
    install_text = _read("scripts/install_observability_stack.ps1")
    stack_text = _read("scripts/start_observability_stack.ps1")
    stop_text = _read("scripts/stop_observability_stack.ps1")
    prometheus_text = _read("scripts/start_prometheus.ps1")
    alertmanager_text = _read("scripts/start_alertmanager.ps1")
    grafana_text = _read("scripts/start_grafana.ps1")
    exporter_text = _read("scripts/start_pmx_observability_exporter.ps1")
    bridge_text = _read("scripts/start_pmx_alertmanager_bridge.ps1")
    helper_text = _read("scripts/observability_process_helpers.ps1")

    assert 'PMX-Observability-Stack.cmd' in install_text
    assert '[switch]$DownloadOfficialBinaries' in install_text
    assert '$DefaultPrometheusZipUrl' in install_text
    assert '$DefaultAlertmanagerZipUrl' in install_text
    assert '$DefaultGrafanaZipUrl' in install_text
    assert 'curl.exe -L --fail --retry 5 --retry-delay 5 --retry-all-errors -C -' in install_text
    assert 'start_observability_stack.ps1' in install_text
    assert 'start_pmx_observability_exporter.ps1' in stack_text
    assert 'start_pmx_alertmanager_bridge.ps1' in stack_text
    assert 'start_prometheus.ps1' in stack_text
    assert 'start_alertmanager.ps1' in stack_text
    assert 'start_grafana.ps1' in stack_text
    assert '[switch]$Foreground' in stack_text
    assert 'stop_observability_stack.ps1' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert 'Stop-ObservedProcesses' in stop_text
    assert 'Request-LocalShutdown' in stop_text
    assert 'Wait-PortClosed' in stop_text
    assert 'http://127.0.0.1:9765/shutdown' in stop_text
    assert 'http://127.0.0.1:9766/shutdown' in stop_text
    assert 'Get-ListeningProcessIdsByPort -Port 9765' in stop_text
    assert 'Get-ListeningProcessIdsByPort -Port 9766' in stop_text
    assert 'function Normalize-PathCandidates' in helper_text
    assert 'function Test-HttpHealthy' in helper_text
    assert 'function Ensure-RepoService' in helper_text
    assert 'function Stop-ObservedProcesses' in helper_text
    assert 'function Get-ListeningProcessIdsByPort' in helper_text
    assert 'function Wait-PortClosed' in helper_text
    assert 'function Request-LocalShutdown' in helper_text
    assert 'Normalize-PathCandidates @(' in prometheus_text
    assert 'Normalize-PathCandidates @(' in alertmanager_text
    assert 'Normalize-PathCandidates @(' in grafana_text
    assert 'Get-ListeningProcessIdsByPort -Port $Port' in exporter_text
    assert 'Get-ListeningProcessIdsByPort -Port $Port' in bridge_text
    assert 'Get-ListeningProcessIdsByPort -Port 9090' in prometheus_text
    assert 'Get-ListeningProcessIdsByPort -Port 9093' in alertmanager_text
    assert 'Get-ListeningProcessIdsByPort -Port 3000' in grafana_text
    assert 'Ensure-RepoService' in exporter_text
    assert 'Ensure-RepoService' in bridge_text
    assert 'Ensure-RepoService' in prometheus_text
    assert 'Ensure-RepoService' in alertmanager_text
    assert 'Ensure-RepoService' in grafana_text
    assert '--storage.tsdb.retention.time=14d' in prometheus_text
    assert '--web.listen-address=127.0.0.1:9090' in prometheus_text
    assert '--web.listen-address=127.0.0.1:9093' in alertmanager_text
    assert 'PMX_GRAFANA_PROVISIONING_PATH' in grafana_text
    assert '--bind", $Bind, "--port", "$Port"' in exporter_text
    assert '--bind", $Bind, "--port", "$Port"' in bridge_text
