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
        "loki",
        "alloy",
        "severity",
        "component",
        "channel",
        "job",
        "run_id",
        "message_id",
        "shadow mode",
        "grafana is ops/sre-only",
        "observability_required_jobs.yml",
        "run_auto_trader_latest.json",
        "project_runtime_status.py --strict --pretty",
        "72-hour clean shadow soak",
    ):
        assert token in text


def test_prometheus_alertmanager_and_loki_configs_stay_localhost_only() -> None:
    prometheus = _read("observability/prometheus/prometheus.yml")
    rules = _read("observability/prometheus/rules/pmx_alerts.yml")
    alertmanager = _read("observability/alertmanager/alertmanager.yml")
    loki = _read("observability/loki/loki.yml")
    alloy = _read("observability/alloy/logs.alloy")
    assert "127.0.0.1:9765" in prometheus
    assert "127.0.0.1:9093" in prometheus
    assert "127.0.0.1:9766/alertmanager" in alertmanager
    assert "127.0.0.1" in loki
    assert "3100" in loki
    assert "PMX_LOKI_URL" in alloy
    assert "pmx_observability" in alloy
    assert "pmx_openclaw" in alloy
    assert "logs\\\\run_audit\\\\*.txt" in alloy
    assert "\\\\.openclaw\\\\logs\\\\*.jsonl" in alloy
    assert 'loki.source.file "pmx_run_audit"' in alloy
    assert 'loki.source.file "pmx_llm_activity"' in alloy
    assert 'loki.source.file "pmx_openclaw_json"' in alloy
    assert alloy.count("tail_from_end = false") >= 3
    assert alloy.count('stage.truncate {') >= 2
    assert 'limit       = "240KiB"' in alloy
    assert 'suffix      = " ...[truncated]"' in alloy
    assert 'stage.labels {' in alloy
    assert 'stage.label_drop {' in alloy
    assert 'values = ["filename"]' in alloy
    assert 'channel = "channel"' in alloy
    for token in (
        "pmx_openclaw_gateway_up",
        "pmx_cron_job_last_success_unixtime",
        "pmx_dashboard_snapshot_age_seconds",
        "pmx_openclaw_channels_status_latency_ms",
        "pmx_production_gate_pass",
    ):
        assert token in rules


def test_grafana_provisioning_disables_alerting_and_loads_loki_and_six_dashboards() -> None:
    grafana_ini = _read("observability/grafana/grafana.ini").lower()
    providers = _read("observability/grafana/provisioning/dashboards/dashboard_providers.yml")
    datasource = _read("observability/grafana/provisioning/datasources/prometheus.yml")
    loki_datasource = _read("observability/grafana/provisioning/datasources/loki.yml")
    assert "[alerting]" in grafana_ini
    assert "enabled = false" in grafana_ini
    assert "pmx_grafana_dashboards_path" in providers.lower()
    assert "uid: prometheus" in datasource
    assert "uid: loki" in loki_datasource

    dashboard_dir = _repo_root() / "observability" / "grafana" / "dashboards"
    dashboard_files = sorted(dashboard_dir.glob("*.json"))
    assert len(dashboard_files) == 6
    unified_found = False
    logs_found = False
    for path in dashboard_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["refresh"] == "30s"
        assert payload["uid"].startswith("pmx-")
        if payload["uid"] == "pmx-unified-ops":
            unified_found = True
            assert payload["title"] == "PMX Unified Ops"
            targets = [
                target["expr"]
                for panel in payload["panels"]
                for target in panel.get("targets", [])
                if isinstance(target, dict) and "expr" in target
            ]
            for expr in (
                "pmx_dashboard_pnl_absolute",
                "pmx_dashboard_win_rate",
                "pmx_dashboard_open_positions_count",
                "pmx_dashboard_equity_last",
                "pmx_dashboard_signal_count",
                "pmx_dashboard_latest_signal_confidence",
                "pmx_dashboard_latest_trade_slippage_bp",
                "pmx_forecaster_rmse_ratio",
                "pmx_forecaster_profit_factor",
                "pmx_quant_validation_fail_fraction",
                "pmx_operator_console_status_code",
                "pmx_proof_runway_remaining_days",
                "sum by (job) (count_over_time({job=~\"pmx_.*\"}[15m]))",
                "sum by (channel) (count_over_time({job=\"pmx_llm_activity\",channel=~\".+\"}[15m]))",
                "{job=~\"pmx_(observability|run_audit|openclaw)\"} |~ \"(?i)(error|warn|fail|critical)\"",
                "{job=\"pmx_run_audit\"} |~ \"(?i)(error|warn|fail|critical|gate|integrity|orphaned|contamination)\"",
            ):
                assert expr in targets
        if payload["uid"] == "pmx-logs":
            logs_found = True
            assert payload["title"] == "PMX Logs"
            targets = [
                target["expr"]
                for panel in payload["panels"]
                for target in panel.get("targets", [])
                if isinstance(target, dict) and "expr" in target
            ]
            assert "{job=~\"pmx_.*\"}" in targets
            assert "{job=\"pmx_observability\"} |~ \"(?i)(error|warn|fail|critical)\"" in targets
            assert "{job=\"pmx_openclaw\"}" in targets
            assert "sum by (job) (count_over_time({job=~\"pmx_.*\"}[15m]))" in targets
            assert "sum by (channel) (count_over_time({job=\"pmx_llm_activity\",channel=~\".+\"}[15m]))" in targets
            assert "{job=\"pmx_llm_activity\",channel=~\".+\"}" in targets
            assert "{job=\"pmx_run_audit\"} |~ \"(?i)(error|warn|fail|critical|gate|integrity|orphaned|contamination)\"" in targets
    assert unified_found
    assert logs_found


def test_windows_startup_scripts_and_installer_reference_repo_owned_launchers() -> None:
    install_text = _read("scripts/install_observability_stack.ps1")
    stack_text = _read("scripts/start_observability_stack.ps1")
    stop_text = _read("scripts/stop_observability_stack.ps1")
    status_text = _read("scripts/status_observability_stack.ps1")
    loki_text = _read("scripts/start_loki.ps1")
    alloy_text = _read("scripts/start_alloy.ps1")
    prometheus_text = _read("scripts/start_prometheus.ps1")
    alertmanager_text = _read("scripts/start_alertmanager.ps1")
    grafana_text = _read("scripts/start_grafana.ps1")
    exporter_text = _read("scripts/start_pmx_observability_exporter.ps1")
    bridge_text = _read("scripts/start_pmx_alertmanager_bridge.ps1")
    helper_text = _read("scripts/observability_process_helpers.ps1")
    exporter_py = _read("scripts/pmx_observability_exporter.py")
    bridge_py = _read("scripts/pmx_alertmanager_bridge.py")
    alloy_config = _read("observability/alloy/logs.alloy")

    assert 'PMX-Observability-Stack.cmd' in install_text
    assert '[switch]$DownloadOfficialBinaries' in install_text
    assert '[string]$LokiZipPath = ""' in install_text
    assert '[string]$AlloyZipPath = ""' in install_text
    assert '$DefaultPrometheusZipUrl' in install_text
    assert '$DefaultAlertmanagerZipUrl' in install_text
    assert '$DefaultGrafanaZipUrl' in install_text
    assert '$DefaultLokiZipUrl' in install_text
    assert '$DefaultAlloyZipUrl' in install_text
    assert 'curl.exe -L --fail --retry 5 --retry-delay 5 --retry-all-errors -C -' in install_text
    assert 'function Install-LocalZip' in install_text
    assert 'function Install-ExtractedPackage' in install_text
    assert 'using_local_zip loki' in install_text
    assert 'using_local_zip alloy' in install_text
    assert 'start_observability_stack.ps1' in install_text
    assert 'start_pmx_observability_exporter.ps1' in stack_text
    assert 'start_pmx_alertmanager_bridge.ps1' in stack_text
    assert 'start_loki.ps1' in stack_text
    assert 'start_alloy.ps1' in stack_text
    assert 'optional_component_missing loki' in stack_text
    assert 'optional_component_missing alloy' in stack_text
    assert 'start_prometheus.ps1' in stack_text
    assert 'start_alertmanager.ps1' in stack_text
    assert 'start_grafana.ps1' in stack_text
    assert '[switch]$Foreground' in stack_text
    assert 'stop_observability_stack.ps1' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert 'status_observability_stack.ps1' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert 'Stop-ObservedProcesses' in stop_text
    assert 'Request-LocalShutdown' in stop_text
    assert 'Wait-PortClosed' in stop_text
    assert 'http://127.0.0.1:9765/shutdown' in stop_text
    assert 'http://127.0.0.1:9766/shutdown' in stop_text
    assert 'Get-ListeningProcessIdsByPort -Port 9765' in stop_text
    assert 'Get-ListeningProcessIdsByPort -Port 9766' in stop_text
    assert 'Get-ListeningProcessIdsByPort -Port 3100' in stop_text
    assert 'Get-ListeningProcessIdsByPort -Port 12345' in stop_text
    assert 'function Normalize-PathCandidates' in helper_text
    assert 'function Test-HttpHealthy' in helper_text
    assert 'function Get-HttpJson' in helper_text
    assert 'function Ensure-RepoService' in helper_text
    assert 'simpleTrader_env_win\\Scripts\\python.exe' in helper_text
    assert 'PMX_PYTHON_BIN' in helper_text
    assert 'function Stop-ObservedProcesses' in helper_text
    assert 'function Get-ListeningProcessIdsByPort' in helper_text
    assert 'function Wait-PortClosed' in helper_text
    assert 'function Request-LocalShutdown' in helper_text
    assert 'Get-ServiceStatusRow' in status_text
    assert '[switch]$Json' in status_text
    assert '[switch]$RequireCurrent' in status_text
    assert 'ConvertTo-Json -Depth 6' in status_text
    assert 'status = $(' in status_text
    assert '"legacy"' in status_text
    assert '"partial"' in status_text
    assert 'legacy_sidecar_count' in status_text
    assert 'optional_missing_count' in status_text
    assert 'optional=not_installed' in status_text
    assert 'exit 2' in status_text
    assert 'already_healthy_legacy' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert '-RequireCurrent' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert 'mode = "legacy"' in status_text
    assert '"loki"' in status_text
    assert '"alloy"' in status_text
    assert 'Normalize-PathCandidates @(' in loki_text
    assert 'Normalize-PathCandidates @(' in alloy_text
    assert 'GrafanaLabs\\Alloy\\alloy.exe' in alloy_text
    assert 'Normalize-PathCandidates @(' in prometheus_text
    assert 'Normalize-PathCandidates @(' in alertmanager_text
    assert 'Normalize-PathCandidates @(' in grafana_text
    assert 'Get-ListeningProcessIdsByPort -Port $Port' in exporter_text
    assert 'Get-ListeningProcessIdsByPort -Port $Port' in bridge_text
    assert 'Get-ListeningProcessIdsByPort -Port 3100' in loki_text
    assert 'Get-ListeningProcessIdsByPort -Port 12345' in alloy_text
    assert 'GrafanaLabs\\Alloy\\alloy.exe' in stop_text
    assert 'Get-ListeningProcessIdsByPort -Port 9090' in prometheus_text
    assert 'Get-ListeningProcessIdsByPort -Port 9093' in alertmanager_text
    assert 'Get-ListeningProcessIdsByPort -Port 3000' in grafana_text
    assert 'Ensure-RepoService' in exporter_text
    assert 'Ensure-RepoService' in bridge_text
    assert 'Ensure-RepoService' in loki_text
    assert 'Ensure-RepoService' in alloy_text
    assert '-RequireShutdownSupport' in exporter_text
    assert '-RequireShutdownSupport' in bridge_text
    assert 'Ensure-RepoService' in prometheus_text
    assert 'Ensure-RepoService' in alertmanager_text
    assert 'Ensure-RepoService' in grafana_text
    assert '-config.expand-env=true' in loki_text
    assert '--server.http.listen-addr=127.0.0.1:12345' in alloy_text
    assert 'PMX_REPO_ROOT' in alloy_text
    assert 'PMX_LOKI_URL' in alloy_text
    assert '\\\\.openclaw\\\\logs\\\\*.jsonl' in alloy_config
    assert 'logs\\\\run_audit\\\\*.txt' in alloy_config
    assert 'stage.label_drop {' in alloy_config
    assert 'values = ["filename"]' in alloy_config
    assert alloy_config.count('tail_from_end = false') >= 3
    assert alloy_config.count('stage.truncate {') >= 2
    assert '--storage.tsdb.retention.time=14d' in prometheus_text
    assert '--web.listen-address=127.0.0.1:9090' in prometheus_text
    assert '--web.listen-address=127.0.0.1:9093' in alertmanager_text
    assert 'PMX_GRAFANA_PROVISIONING_PATH' in grafana_text
    assert 'PMX_LOKI_URL' in grafana_text
    assert '--bind", $Bind, "--port", "$Port"' in exporter_text
    assert '--bind", $Bind, "--port", "$Port"' in bridge_text
    assert '"shutdown_supported": True' in exporter_py
    assert '"shutdown_supported": True' in bridge_py
    assert '"pid": os.getpid()' in exporter_py
    assert '"pid": os.getpid()' in bridge_py
    assert 'portable Alloy install' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert 'Manual Loki ZIP import' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert '-LokiZipPath "C:\\path\\to\\loki-windows-amd64.exe.zip"' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert 'transient path labels such as `filename` are dropped' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert 'interactive channel activity counts from shipped `llm_activity` logs' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert 'decision-critical feeds backfill from the start on first boot' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
    assert 'oversized JSONL lines are truncated before write so Loki does not reject whole OpenClaw/LLM batches on Windows' in _read("Documentation/OBSERVABILITY_PROMETHEUS_GRAFANA.md")
