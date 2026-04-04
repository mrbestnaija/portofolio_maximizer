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
$legacyStartupFallback = Join-Path $startupFolder "PMX-OpenClaw-Guardian.cmd"
$launcherDir = Join-Path $env:LOCALAPPDATA "PMX\OpenClawGuardian"
$taskLauncher = Join-Path $launcherDir "launch_openclaw_guardian.cmd"

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

function Test-TaskPresent {
    Param([string]$TaskName)
    $query = Invoke-Schtasks @("/Query", "/TN", $TaskName)
    return ($query.ReturnCode -eq 0)
}

function Write-GuardianLauncherStub {
    $content = @(
        "@echo off",
        "powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$guardianScript`" -WatchIntervalSeconds $WatchIntervalSeconds -PrimaryChannel `"$PrimaryChannel`" -EnsureFunctionalState -Quiet",
        "exit /b %ERRORLEVEL%"
    ) -join [Environment]::NewLine
    New-Item -ItemType Directory -Path $launcherDir -Force | Out-Null
    Set-Content -Path $taskLauncher -Value $content -Encoding ASCII
    Write-Host "[watchdog] launcher_stub_ready $taskLauncher"
    return $taskLauncher
}

function Remove-GuardianLauncherStub {
    if (Test-Path $taskLauncher) {
        Remove-Item -Path $taskLauncher -Force
        Write-Host "[watchdog] launcher_stub_removed $taskLauncher"
    } else {
        Write-Host "[watchdog] launcher_stub_missing $taskLauncher"
    }
}

function Register-GuardianTask {
    Param(
        [string]$TaskName,
        [string[]]$ScheduleArgs
    )

    $taskCommand = 'cmd.exe /d /s /c ""' + $taskLauncher + '""'
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
        "call `"$taskLauncher`"",
        "exit /b %ERRORLEVEL%"
    ) -join [Environment]::NewLine
    New-Item -ItemType Directory -Path $startupFolder -Force | Out-Null
    Set-Content -Path $startupFallback -Value $content -Encoding ASCII
    Write-Host "[watchdog] startup_fallback_ready $startupFallback"
    return $true
}

function Remove-StartupFolderFallback {
    foreach ($path in @($startupFallback, $legacyStartupFallback)) {
        if (Test-Path $path) {
            Remove-Item -Path $path -Force
            Write-Host "[watchdog] startup_fallback_removed $path"
        } else {
            Write-Host "[watchdog] startup_fallback_missing $path"
        }
    }
}

if ($Uninstall) {
    Remove-TaskIfPresent -TaskName $logonTask
    Remove-TaskIfPresent -TaskName $startupTask
    Remove-TaskIfPresent -TaskName $wakeTask
    Remove-TaskIfPresent -TaskName $keepAliveTask
    Remove-StartupFolderFallback
    Remove-GuardianLauncherStub
    exit 0
}

$null = Write-GuardianLauncherStub

$logonTaskPresent = Test-TaskPresent -TaskName $logonTask
$logonReady = Register-GuardianTask -TaskName $logonTask -ScheduleArgs @("/SC", "ONLOGON")
$startupReady = Register-GuardianTask -TaskName $startupTask -ScheduleArgs @("/SC", "ONSTART")
$wakeReady = Register-GuardianTask -TaskName $wakeTask -ScheduleArgs @(
    "/SC", "ONEVENT",
    "/EC", "System",
    "/MO", "*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]"
)
$keepAliveReady = Register-GuardianTask -TaskName $keepAliveTask -ScheduleArgs @("/SC", "MINUTE", "/MO", "$EnsureIntervalMinutes")
$legacyReady = Disable-LegacyMaintenanceTask
$startupFallbackReady = $false
if (-not ($logonReady -or $logonTaskPresent)) {
    $startupFallbackReady = Install-StartupFolderFallback
} else {
    Remove-StartupFolderFallback
}

& cmd.exe /d /s /c "`"$taskLauncher`""
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "[watchdog] install_complete startup=$startupTask wake=$wakeTask logon=$logonTask keepalive=$keepAliveTask interval_minutes=$EnsureIntervalMinutes"
if (-not (($logonReady -or $logonTaskPresent -or $startupFallbackReady) -and $wakeReady -and $keepAliveReady -and $legacyReady)) {
    Write-Warning "[watchdog] partial_install guardian_started=true scheduler_tasks_may_require_elevation"
    exit 1
}
