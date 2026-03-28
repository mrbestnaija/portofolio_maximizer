param(
    [switch]$Foreground,
    [string]$GrafanaExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "observability_process_helpers.ps1")

$repoRoot = Resolve-RepoRoot
$exeCandidates = Normalize-PathCandidates @(
    $GrafanaExe,
    $env:PMX_GRAFANA_EXE,
    (Join-Path $repoRoot "tools\observability\grafana\bin\grafana-server.exe")
)
$exe = Resolve-RequiredPath -Label "grafana-server.exe" -Candidates $exeCandidates
$configPath = Join-Path $repoRoot "observability\grafana\grafana.ini"
$provisioningPath = Join-Path $repoRoot "observability\grafana\provisioning"
$dashboardsPath = Join-Path $repoRoot "observability\grafana\dashboards"
$dataPath = Join-Path $repoRoot "data\grafana"
$grafanaLogPath = Join-Path $repoRoot "logs\observability\grafana"
$stdoutPath = Join-Path $repoRoot "logs\observability\grafana.stdout.log"
$stderrPath = Join-Path $repoRoot "logs\observability\grafana.stderr.log"
$homePath = Split-Path -Parent (Split-Path -Parent $exe)
New-Item -ItemType Directory -Force -Path $dataPath, $grafanaLogPath | Out-Null

$env:PMX_GRAFANA_PROVISIONING_PATH = $provisioningPath
$env:PMX_GRAFANA_DASHBOARDS_PATH = $dashboardsPath
$env:PMX_GRAFANA_DATA_PATH = $dataPath
$env:PMX_GRAFANA_LOG_PATH = $grafanaLogPath
$env:PMX_PROMETHEUS_URL = "http://127.0.0.1:9090"

$args = @("--config", $configPath, "--homepath", $homePath)
$existingProcessIds = @(
    (Get-ObservedProcessIds -ExecutablePathLike (Join-Path $repoRoot "tools\observability\grafana\bin\*.exe"))
    (Get-ListeningProcessIdsByPort -Port 3000)
) | Sort-Object -Unique

Ensure-RepoService `
    -Label "grafana" `
    -FilePath $exe `
    -ArgumentList $args `
    -WorkingDirectory $repoRoot `
    -StdOutPath $stdoutPath `
    -StdErrPath $stderrPath `
    -HealthUrl "http://127.0.0.1:3000/api/health" `
    -ExistingProcessIds $existingProcessIds `
    -StartupTimeoutSec 30 `
    -Foreground:$Foreground
