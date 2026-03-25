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
$keepAliveTask = "PMX-OpenClaw-Guardian-KeepAlive"
$legacyTask = "PMX-OpenClaw-Maintenance"

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
        ' -PrimaryChannel "' + $PrimaryChannel + '"'
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

if ($Uninstall) {
    Remove-TaskIfPresent -TaskName $logonTask
    Remove-TaskIfPresent -TaskName $keepAliveTask
    exit 0
}

$logonReady = Register-GuardianTask -TaskName $logonTask -ScheduleArgs @("/SC", "ONLOGON")
$keepAliveReady = Register-GuardianTask -TaskName $keepAliveTask -ScheduleArgs @("/SC", "MINUTE", "/MO", "$EnsureIntervalMinutes")
$legacyReady = Disable-LegacyMaintenanceTask

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $guardianScript -WatchIntervalSeconds $WatchIntervalSeconds -PrimaryChannel $PrimaryChannel
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[watchdog] install_complete logon=$logonTask keepalive=$keepAliveTask interval_minutes=$EnsureIntervalMinutes"
if (-not ($logonReady -and $keepAliveReady -and $legacyReady)) {
    Write-Warning "[watchdog] partial_install guardian_started=true scheduler_tasks_may_require_elevation"
    exit 1
}
