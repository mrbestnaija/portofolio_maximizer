Param(
    [int]$WatchIntervalSeconds = 120,
    [int]$EnsureIntervalMinutes = 5,
    [string]$PrimaryChannel = "whatsapp",
    [switch]$Uninstall,
    [switch]$KeepLegacyMaintenanceTask
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$guardianScript = Join-Path $repoRoot "scripts\start_openclaw_guardian.ps1"
$logonTask = "PMX-OpenClaw-Guardian-Logon"
$startupTask = "PMX-OpenClaw-Guardian-Startup"
$wakeTask = "PMX-OpenClaw-Guardian-Wake"
$keepAliveTask = "PMX-OpenClaw-Guardian-KeepAlive"
$legacyTask = "PMX-OpenClaw-Maintenance"
$startupFolder = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$startupFallback = Join-Path $startupFolder "PMX-OpenClaw-Guardian-Startup.cmd"

if (-not (Test-Path $guardianScript)) {
    throw "Missing guardian launcher: $guardianScript"
}

if ($WatchIntervalSeconds -lt 30) {
    $WatchIntervalSeconds = 30
}
if ($EnsureIntervalMinutes -lt 1) {
    $EnsureIntervalMinutes = 1
}

function Invoke-Schtasks {
    Param([string[]]$Arguments)
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & schtasks.exe @Arguments 2>&1
        $rc = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    return [pscustomobject]@{
        ReturnCode = [int]$rc
        Output = (($output | ForEach-Object { "$_" }) -join [Environment]::NewLine)
    }
}

function Remove-TaskIfPresent {
    Param([string]$TaskName)
    $query = Invoke-Schtasks @("/Query", "/TN", $TaskName)
    if ($query.ReturnCode -ne 0) {
        Write-Host "[watchdog] task_missing $TaskName"
        return
    }
    $delete = Invoke-Schtasks @("/Delete", "/TN", $TaskName, "/F")
    if ($delete.ReturnCode -ne 0) {
        throw "Failed to delete task $TaskName`n$($delete.Output)"
    }
    Write-Host "[watchdog] task_removed $TaskName"
}

function Register-GuardianTask {
    Param(
        [string]$TaskName,
        [string[]]$ScheduleArgs
    )

    $taskCommand = (
        'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "' + $guardianScript + '"' +
        ' -WatchIntervalSeconds ' + $WatchIntervalSeconds +
        ' -PrimaryChannel "' + $PrimaryChannel + '"' +
        ' -EnsureFunctionalState'
    )

    $args = @(
        "/Create",
        "/TN", $TaskName,
        "/TR", $taskCommand,
        "/RL", "LIMITED",
        "/F"
    ) + $ScheduleArgs

    $result = Invoke-Schtasks -Arguments $args
    if ($result.ReturnCode -ne 0) {
        Write-Warning "[watchdog] task_failed $TaskName :: $($result.Output)"
        return $false
    }
    Write-Host "[watchdog] task_ready $TaskName"
    return $true
}

function Disable-LegacyMaintenanceTask {
    if ($KeepLegacyMaintenanceTask) {
        Write-Host "[watchdog] legacy_task_kept $legacyTask"
        return $true
    }

    $query = Invoke-Schtasks @("/Query", "/TN", $legacyTask)
    if ($query.ReturnCode -ne 0) {
        Write-Host "[watchdog] legacy_task_missing $legacyTask"
        return $true
    }

    $disable = Invoke-Schtasks @("/Change", "/TN", $legacyTask, "/DISABLE")
    if ($disable.ReturnCode -ne 0) {
        Write-Warning "[watchdog] legacy_task_disable_failed $legacyTask :: $($disable.Output)"
        return $false
    }
    Write-Host "[watchdog] legacy_task_disabled $legacyTask"
    return $true
}

function Install-StartupFolderFallback {
    $content = @(
        "@echo off",
        "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$guardianScript`" -WatchIntervalSeconds $WatchIntervalSeconds -PrimaryChannel `"$PrimaryChannel`" -EnsureFunctionalState"
    ) -join [Environment]::NewLine
    New-Item -ItemType Directory -Path $startupFolder -Force | Out-Null
    Set-Content -Path $startupFallback -Value $content -Encoding ASCII
    Write-Host "[watchdog] startup_fallback_ready $startupFallback"
    return $true
}

function Remove-StartupFolderFallback {
    if (Test-Path $startupFallback) {
        Remove-Item -Path $startupFallback -Force
        Write-Host "[watchdog] startup_fallback_removed $startupFallback"
    } else {
        Write-Host "[watchdog] startup_fallback_missing $startupFallback"
    }
}

if ($Uninstall) {
    Remove-TaskIfPresent -TaskName $logonTask
    Remove-TaskIfPresent -TaskName $startupTask
    Remove-TaskIfPresent -TaskName $wakeTask
    Remove-TaskIfPresent -TaskName $keepAliveTask
    Remove-StartupFolderFallback
    exit 0
}

$logonReady = Register-GuardianTask -TaskName $logonTask -ScheduleArgs @("/SC", "ONLOGON")
$startupReady = Register-GuardianTask -TaskName $startupTask -ScheduleArgs @("/SC", "ONSTART")
$wakeReady = Register-GuardianTask -TaskName $wakeTask -ScheduleArgs @(
    "/SC", "ONEVENT",
    "/EC", "System",
    "/MO", "*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]"
)
$keepAliveReady = Register-GuardianTask -TaskName $keepAliveTask -ScheduleArgs @("/SC", "MINUTE", "/MO", "$EnsureIntervalMinutes")
$legacyReady = Disable-LegacyMaintenanceTask
$startupFallbackReady = $true
if (-not ($logonReady -and $startupReady)) {
    $startupFallbackReady = Install-StartupFolderFallback
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $guardianScript -WatchIntervalSeconds $WatchIntervalSeconds -PrimaryChannel $PrimaryChannel -EnsureFunctionalState
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[watchdog] install_complete startup=$startupTask wake=$wakeTask logon=$logonTask keepalive=$keepAliveTask interval_minutes=$EnsureIntervalMinutes"
if (-not ((($logonReady -and $startupReady) -or $startupFallbackReady) -and $wakeReady -and $keepAliveReady -and $legacyReady)) {
    Write-Warning "[watchdog] partial_install guardian_started=true scheduler_tasks_may_require_elevation"
    exit 1
}
