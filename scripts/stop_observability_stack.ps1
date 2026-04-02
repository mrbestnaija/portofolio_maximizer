Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "observability_process_helpers.ps1")

$repoRoot = Resolve-RepoRoot

$gracefulTargets = @(
    @{
        Label = "pmx_alertmanager_bridge"
        Port = 9766
        ShutdownUrl = "http://127.0.0.1:9766/shutdown"
    },
    @{
        Label = "pmx_observability_exporter"
        Port = 9765
        ShutdownUrl = "http://127.0.0.1:9765/shutdown"
    }
)

$forceTargets = @(
    @{
        Label = "grafana"
        ProcessIds = @(
            (Get-ObservedProcessIds -ExecutablePathLike (Join-Path $repoRoot "tools\observability\grafana\bin\*.exe"))
            (Get-ListeningProcessIdsByPort -Port 3000)
        ) | Sort-Object -Unique
        Port = 3000
    },
    @{
        Label = "alloy"
        ProcessIds = @(
            (Get-ObservedProcessIds -ExecutablePathLike (Join-Path $env:ProgramFiles "GrafanaLabs\Alloy\alloy.exe"))
            (Get-ObservedProcessIds -ExecutablePathLike (Join-Path $repoRoot "tools\observability\alloy\*.exe"))
            (Get-ListeningProcessIdsByPort -Port 12345)
        ) | Sort-Object -Unique
        Port = 12345
    },
    @{
        Label = "loki"
        ProcessIds = @(
            (Get-ObservedProcessIds -ExecutablePathLike (Join-Path $repoRoot "tools\observability\loki\*.exe"))
            (Get-ListeningProcessIdsByPort -Port 3100)
        ) | Sort-Object -Unique
        Port = 3100
    },
    @{
        Label = "alertmanager"
        ProcessIds = @(
            (Get-ObservedProcessIds -ExecutablePathLike (Join-Path $repoRoot "tools\observability\alertmanager\*.exe"))
            (Get-ListeningProcessIdsByPort -Port 9093)
        ) | Sort-Object -Unique
        Port = 9093
    },
    @{
        Label = "prometheus"
        ProcessIds = @(
            (Get-ObservedProcessIds -ExecutablePathLike (Join-Path $repoRoot "tools\observability\prometheus\*.exe"))
            (Get-ListeningProcessIdsByPort -Port 9090)
        ) | Sort-Object -Unique
        Port = 9090
    }
)

$fallbackTargets = @(
    @{
        Label = "pmx_alertmanager_bridge"
        ProcessIds = @(Get-ListeningProcessIdsByPort -Port 9766)
        Port = 9766
    },
    @{
        Label = "pmx_observability_exporter"
        ProcessIds = @(Get-ListeningProcessIdsByPort -Port 9765)
        Port = 9765
    }
)

$stoppedGracefully = @{}
foreach ($target in $gracefulTargets) {
    $processIds = @(Get-ListeningProcessIdsByPort -Port $target.Port)
    if ($processIds.Count -eq 0) {
        Write-Status ("already_stopped " + $target.Label)
        $stoppedGracefully[$target.Label] = $true
        continue
    }

    $requested = Request-LocalShutdown -Label $target.Label -Url $target.ShutdownUrl
    if ($requested -and (Wait-PortClosed -Port $target.Port -TimeoutSec 10)) {
        Write-Status ("stopped " + $target.Label + " via_shutdown")
        $stoppedGracefully[$target.Label] = $true
        continue
    }

    $stoppedGracefully[$target.Label] = $false
}

$targets = @(
    $forceTargets
    $fallbackTargets
)

foreach ($target in $targets) {
    if ($stoppedGracefully.ContainsKey($target.Label) -and $stoppedGracefully[$target.Label]) {
        continue
    }

    Stop-ObservedProcesses -Label $target.Label -ProcessIds $target.ProcessIds
    if ($target.Port -and -not (Wait-PortClosed -Port $target.Port -TimeoutSec 10)) {
        Write-Status ("stop_unresolved " + $target.Label + " port=" + $target.Port)
    }
}
