param(
    [switch]$Foreground
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

& (Join-Path $repoRoot "scripts\start_pmx_observability_exporter.ps1") -Foreground:$Foreground
& (Join-Path $repoRoot "scripts\start_pmx_alertmanager_bridge.ps1") -Foreground:$Foreground
& (Join-Path $repoRoot "scripts\start_prometheus.ps1") -Foreground:$Foreground
& (Join-Path $repoRoot "scripts\start_alertmanager.ps1") -Foreground:$Foreground
& (Join-Path $repoRoot "scripts\start_grafana.ps1") -Foreground:$Foreground
