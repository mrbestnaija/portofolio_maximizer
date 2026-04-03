param(
    [switch]$Json,
    [switch]$RequireCurrent
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "observability_process_helpers.ps1")

function Get-ServiceStatusRow {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][int]$Port,
        [Parameter(Mandatory = $true)][string]$HealthUrl,
        [switch]$RequireShutdownSupport,
        [AllowEmptyCollection()][string[]]$InstallCandidates = @(),
        [switch]$Optional
    )

    $processIds = @(Get-ListeningProcessIdsByPort -Port $Port | Sort-Object -Unique)
    $healthy = Test-HttpHealthy -Url $HealthUrl -TimeoutSec 5
    $healthPayload = Get-HttpJson -Url $HealthUrl -TimeoutSec 5
    $installed = $true
    if ($InstallCandidates.Count -gt 0) {
        $installed = $false
        foreach ($candidate in (Normalize-PathCandidates $InstallCandidates)) {
            if (Test-Path $candidate) {
                $installed = $true
                break
            }
        }
    }

    $shutdownSupported = $null
    $reportedPid = $null
    if ($null -ne $healthPayload) {
        $propertyNames = @($healthPayload.PSObject.Properties | Select-Object -ExpandProperty Name)
        if ($propertyNames -contains "shutdown_supported") {
            $shutdownSupported = [bool]$healthPayload.shutdown_supported
        }
        if ($propertyNames -contains "pid") {
            $reportedPid = [int]$healthPayload.pid
        }
    }

    $mode = "n/a"
    if ($RequireShutdownSupport) {
        if ($healthy -and $shutdownSupported -eq $true) {
            $mode = "current"
        }
        elseif ($healthy) {
            $mode = "legacy"
        }
        else {
            $mode = "down"
        }
    }

    $state = "down"
    if ($Optional -and -not $installed -and -not $healthy -and $processIds.Count -eq 0) {
        $state = "not_installed"
    }
    elseif ($healthy) {
        $state = "healthy"
    }
    elseif ($processIds.Count -gt 0) {
        $state = "listening_unhealthy"
    }

    return [ordered]@{
        label = $Label
        port = $Port
        state = $state
        healthy = $healthy
        installed = $installed
        optional = [bool]$Optional
        mode = $mode
        shutdown_supported = $shutdownSupported
        listener_pids = @($processIds)
        reported_pid = $reportedPid
        health_url = $HealthUrl
    }
}

$services = @(
    (Get-ServiceStatusRow -Label "pmx_observability_exporter" -Port 9765 -HealthUrl "http://127.0.0.1:9765/healthz" -RequireShutdownSupport),
    (Get-ServiceStatusRow -Label "pmx_alertmanager_bridge" -Port 9766 -HealthUrl "http://127.0.0.1:9766/healthz" -RequireShutdownSupport),
    (Get-ServiceStatusRow -Label "loki" -Port 3100 -HealthUrl "http://127.0.0.1:3100/metrics" -InstallCandidates @((Join-Path $PSScriptRoot "..\\tools\\observability\\loki\\loki*.exe")) -Optional),
    (Get-ServiceStatusRow -Label "alloy" -Port 12345 -HealthUrl "http://127.0.0.1:12345/metrics" -InstallCandidates @(
        (Join-Path $env:ProgramFiles "GrafanaLabs\\Alloy\\alloy.exe"),
        (Join-Path $PSScriptRoot "..\\tools\\observability\\alloy\\alloy*.exe")
    ) -Optional),
    (Get-ServiceStatusRow -Label "prometheus" -Port 9090 -HealthUrl "http://127.0.0.1:9090/-/healthy"),
    (Get-ServiceStatusRow -Label "alertmanager" -Port 9093 -HealthUrl "http://127.0.0.1:9093/-/healthy"),
    (Get-ServiceStatusRow -Label "grafana" -Port 3000 -HealthUrl "http://127.0.0.1:3000/api/health")
)

$allHealthy = @($services | Where-Object { -not $_.optional -and -not $_.healthy }).Count -eq 0
$legacySidecars = @($services | Where-Object { $_.mode -eq "legacy" }).Count
$optionalMissing = @($services | Where-Object { $_.optional -and -not $_.installed }).Count
$summary = [ordered]@{
    status = $(
        if (-not $allHealthy) {
            "degraded"
        }
        elseif ($legacySidecars -gt 0) {
            "legacy"
        }
        elseif ($optionalMissing -gt 0) {
            "partial"
        }
        else {
            "ok"
        }
    )
    all_healthy = $allHealthy
    legacy_sidecar_count = $legacySidecars
    optional_missing_count = $optionalMissing
    require_current = [bool]$RequireCurrent
    services = $services
}

if ($Json) {
    $summary | ConvertTo-Json -Depth 6
}
else {
    foreach ($service in $services) {
        $parts = @(
            $service.label,
            ("state=" + $service.state),
            ("port=" + $service.port)
        )
    if ($service.mode -ne "n/a") {
        $parts += ("mode=" + $service.mode)
    }
    if ($service.optional -and -not $service.installed) {
        $parts += "optional=not_installed"
    }
        if ($service.listener_pids.Count -gt 0) {
            $parts += ("pid=" + ($service.listener_pids -join ","))
        }
        Write-Status ($parts -join " ")
    }

    $summaryLine = "status=" + $summary.status
    if ($legacySidecars -gt 0) {
        $summaryLine += " legacy_sidecar_count=" + $legacySidecars
    }
    Write-Status $summaryLine
}

if (-not $allHealthy) {
    exit 1
}

if ($RequireCurrent -and $legacySidecars -gt 0) {
    exit 2
}

exit 0
