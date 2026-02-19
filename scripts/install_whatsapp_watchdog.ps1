<#
.SYNOPSIS
    Installs persistent OpenClaw guardian tasks for gateway + WhatsApp stability.

.DESCRIPTION
    Registers two idempotent Scheduled Tasks:
    1. PMX-OpenClaw-Guardian-Logon
       Starts the guardian at user logon.
    2. PMX-OpenClaw-Guardian-KeepAlive
       Re-runs every N minutes. The guardian starter is PID-aware and only
       launches a new watcher when one is not already running.

    The guardian script runs:
      scripts/openclaw_maintenance.py --watch --apply ...
    which keeps OpenClaw gateway and WhatsApp connectivity healthy over time.

.PARAMETER Uninstall
    Removes watchdog tasks and stops the running guardian process (best-effort).

.PARAMETER WatchIntervalSeconds
    Guardian maintenance watch loop interval in seconds (minimum: 30).

.PARAMETER EnsureIntervalMinutes
    KeepAlive task interval in minutes (minimum: 1).

.PARAMETER PrimaryChannel
    Primary channel to keep healthy (default: whatsapp).

.PARAMETER NoApply
    Start guardian in dry-run mode (no healing mutations).

.PARAMETER DryRun
    Preview actions without changing scheduled tasks.

.PARAMETER KeepLegacyMaintenanceTask
    Keep legacy one-shot maintenance schedulers enabled (not recommended alongside guardian watch mode).
#>

Param(
    [switch]$Uninstall,
    [int]$WatchIntervalSeconds = 120,
    [int]$EnsureIntervalMinutes = 5,
    [string]$PrimaryChannel = "whatsapp",
    [string]$IntegrityUnlinkedCloseWhitelistIds = "",
    [switch]$NoApply,
    [switch]$KeepLegacyMaintenanceTask,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$TaskNameLogon = "PMX-OpenClaw-Guardian-Logon"
$TaskNameKeepAlive = "PMX-OpenClaw-Guardian-KeepAlive"
$GatewayTaskName = "OpenClaw Gateway"
$ConflictingTaskNames = @("PMX-OpenClaw-Maintenance")

if ($WatchIntervalSeconds -lt 30) { $WatchIntervalSeconds = 30 }
if ($EnsureIntervalMinutes -lt 1) { $EnsureIntervalMinutes = 1 }
if (-not $IntegrityUnlinkedCloseWhitelistIds) {
    $IntegrityUnlinkedCloseWhitelistIds = if ($env:INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS) {
        $env:INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS
    } else {
        "66"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$guardianScript = Join-Path $repoRoot "scripts\start_openclaw_guardian.ps1"
$pidFile = Join-Path $repoRoot "logs\automation\openclaw_guardian.pid.json"

if (-not (Test-Path $guardianScript)) {
    Write-Error "Guardian starter missing: $guardianScript"
    exit 1
}

$powerShellExe = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
if (-not $powerShellExe) {
    $powerShellExe = "powershell.exe"
}

$guardianArgsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$guardianScript`"",
    "-WatchIntervalSeconds", "$WatchIntervalSeconds",
    "-PrimaryChannel", "$PrimaryChannel",
    "-IntegrityUnlinkedCloseWhitelistIds", "$IntegrityUnlinkedCloseWhitelistIds"
)
if ($NoApply) {
    $guardianArgsList += "-NoApply"
}
$guardianArgs = ($guardianArgsList -join " ")

function Remove-TaskIfExists {
    Param([string]$TaskName)
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $existing) {
        Write-Host "[SKIP] Task not found: $TaskName"
        return
    }

    if ($DryRun) {
        Write-Host "[DRY-RUN] Would remove task: $TaskName"
        return
    }

    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[OK] Removed task: $TaskName"
}

function Stop-GuardianIfRunning {
    if (-not (Test-Path $pidFile)) {
        return
    }

    try {
        $raw = Get-Content -Path $pidFile -Raw | ConvertFrom-Json
        $guardianPid = [int]($raw.pid)
    } catch {
        $guardianPid = 0
    }

    if ($guardianPid -le 0) {
        return
    }

    try {
        $proc = Get-Process -Id $guardianPid -ErrorAction Stop
        if ($DryRun) {
            Write-Host "[DRY-RUN] Would stop guardian process pid=$guardianPid"
            return
        }
        Stop-Process -Id $guardianPid -Force -ErrorAction SilentlyContinue
        Write-Host "[OK] Stopped guardian process pid=$guardianPid"
    } catch {
        Write-Host "[SKIP] Guardian pid $guardianPid not running"
    }
}

function Disable-ConflictingTasks {
    if ($KeepLegacyMaintenanceTask) {
        Write-Host "[SKIP] Keeping legacy maintenance task(s) by request."
        return
    }

    foreach ($taskName in $ConflictingTaskNames) {
        $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if (-not $existing) {
            continue
        }
        if ($existing.State -eq "Disabled") {
            Write-Host "[OK] Conflicting task already disabled: $taskName"
            continue
        }

        if ($DryRun) {
            Write-Host "[DRY-RUN] Would disable conflicting task: $taskName"
            continue
        }

        try {
            Disable-ScheduledTask -TaskName $taskName -ErrorAction Stop | Out-Null
            Write-Host "[OK] Disabled conflicting task: $taskName"
        } catch {
            Write-Warning "Failed to disable conflicting task '$taskName': $($_.Exception.Message)"
        }
    }
}

function Harden-GatewayTaskPolicy {
    Param(
        [string]$TaskName = $GatewayTaskName,
        [int]$RestartCount = 5,
        [int]$RestartIntervalMinutes = 1
    )

    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $existing) {
        Write-Host "[SKIP] Gateway task not found: $TaskName"
        return $false
    }

    if ($RestartCount -lt 1) { $RestartCount = 1 }
    if ($RestartIntervalMinutes -lt 1) { $RestartIntervalMinutes = 1 }

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
        -RestartCount $RestartCount `
        -RestartInterval (New-TimeSpan -Minutes $RestartIntervalMinutes) `
        -MultipleInstances IgnoreNew

    if ($DryRun) {
        Write-Host "[DRY-RUN] Would harden gateway task policy: $TaskName"
        Write-Host "  RestartCount: $RestartCount"
        Write-Host "  RestartIntervalMinutes: $RestartIntervalMinutes"
        Write-Host "  ExecutionTimeLimit: Infinite"
        return $true
    }

    try {
        Set-ScheduledTask `
            -TaskName $TaskName `
            -Settings $settings `
            -ErrorAction Stop | Out-Null
        Write-Host "[OK] Hardened gateway task policy: $TaskName"
        return $true
    } catch {
        Write-Warning "Failed to harden gateway task policy '$TaskName': $($_.Exception.Message)"
        return $false
    }
}

function Upsert-Task {
    Param(
        [string]$TaskName,
        [Microsoft.Management.Infrastructure.CimInstance[]]$Triggers,
        [string]$Description,
        [int]$ExecutionLimitMinutes = 3
    )

    $action = New-ScheduledTaskAction `
        -Execute $powerShellExe `
        -Argument $guardianArgs `
        -WorkingDirectory $repoRoot

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $ExecutionLimitMinutes) `
        -MultipleInstances IgnoreNew

    if ($DryRun) {
        Write-Host "[DRY-RUN] Would create/update task: $TaskName"
        Write-Host "  Execute: $powerShellExe"
        Write-Host "  Args: $guardianArgs"
        return $true
    }

    try {
        $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($existing) {
            Set-ScheduledTask `
                -TaskName $TaskName `
                -Action $action `
                -Trigger $Triggers `
                -Settings $settings `
                -ErrorAction Stop | Out-Null
            Write-Host "[OK] Updated task: $TaskName"
        } else {
            Register-ScheduledTask `
                -TaskName $TaskName `
                -Action $action `
                -Trigger $Triggers `
                -Settings $settings `
                -Description $Description `
                -ErrorAction Stop | Out-Null
            Write-Host "[OK] Created task: $TaskName"
        }
        return $true
    } catch {
        Write-Warning "Failed to create/update task '$TaskName': $($_.Exception.Message)"
        return $false
    }
}

if ($Uninstall) {
    Write-Host ""
    Write-Host "=== Uninstall OpenClaw WhatsApp Watchdog ==="
    Remove-TaskIfExists -TaskName $TaskNameLogon
    Remove-TaskIfExists -TaskName $TaskNameKeepAlive
    Stop-GuardianIfRunning
    if (-not $DryRun) {
        Write-Host "[OK] Uninstall complete"
    }
    exit 0
}

Write-Host ""
Write-Host "=== Install OpenClaw WhatsApp Watchdog ==="
Write-Host "repo: $repoRoot"
Write-Host "watch_interval_seconds: $WatchIntervalSeconds"
Write-Host "ensure_interval_minutes: $EnsureIntervalMinutes"
Write-Host "primary_channel: $PrimaryChannel"
Write-Host "integrity_unlinked_close_whitelist_ids: $IntegrityUnlinkedCloseWhitelistIds"
Write-Host "apply_mode: $([bool](-not $NoApply))"

$logonTrigger = New-ScheduledTaskTrigger -AtLogOn
$keepAliveTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $EnsureIntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$logonTaskOk = Upsert-Task `
    -TaskName $TaskNameLogon `
    -Triggers @($logonTrigger) `
    -Description "PMX: Start OpenClaw guardian at logon to keep gateway and WhatsApp healthy."

$keepAliveTaskOk = Upsert-Task `
    -TaskName $TaskNameKeepAlive `
    -Triggers @($keepAliveTrigger) `
    -Description "PMX: KeepAlive launcher for OpenClaw guardian (idempotent)."

function Verify-GuardianTasks {
    Param([switch]$Repair)

    $missing = @()
    foreach ($taskName in @($TaskNameLogon, $TaskNameKeepAlive)) {
        $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if (-not $existing) {
            $missing += $taskName
        }
    }

    if ($missing.Count -eq 0) {
        Write-Host "[OK] Guardian task verification passed (both tasks present)."
        return $true
    }

    Write-Warning "Guardian task verification failed. Missing: $($missing -join ', ')"
    if (-not $Repair) {
        return $false
    }

    Write-Host "[INFO] Attempting one-shot guardian task auto-repair..."
    foreach ($taskName in $missing) {
        if ($taskName -eq $TaskNameLogon) {
            $null = Upsert-Task `
                -TaskName $TaskNameLogon `
                -Triggers @($logonTrigger) `
                -Description "PMX: Start OpenClaw guardian at logon to keep gateway and WhatsApp healthy."
            continue
        }
        if ($taskName -eq $TaskNameKeepAlive) {
            $null = Upsert-Task `
                -TaskName $TaskNameKeepAlive `
                -Triggers @($keepAliveTrigger) `
                -Description "PMX: KeepAlive launcher for OpenClaw guardian (idempotent)."
            continue
        }
    }

    $missingAfterRepair = @()
    foreach ($taskName in @($TaskNameLogon, $TaskNameKeepAlive)) {
        $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        if (-not $existing) {
            $missingAfterRepair += $taskName
        }
    }
    if ($missingAfterRepair.Count -gt 0) {
        Write-Warning "Guardian task auto-repair incomplete. Still missing: $($missingAfterRepair -join ', ')"
        return $false
    }

    Write-Host "[OK] Guardian task auto-repair complete."
    return $true
}

if (-not $DryRun) {
    Disable-ConflictingTasks
    $gatewayPolicyOk = Harden-GatewayTaskPolicy
    if (-not $gatewayPolicyOk) {
        Write-Warning "Gateway restart policy hardening not applied (task missing or permission-limited shell)."
    }

    $verified = Verify-GuardianTasks -Repair
    if (-not $verified) {
        Write-Warning "Watchdog installation incomplete: required guardian tasks were not fully registered."
        exit 2
    }

    $startParams = @{
        WatchIntervalSeconds = $WatchIntervalSeconds
        PrimaryChannel = $PrimaryChannel
        IntegrityUnlinkedCloseWhitelistIds = $IntegrityUnlinkedCloseWhitelistIds
    }
    if ($NoApply) {
        $startParams["NoApply"] = $true
    }

    & $guardianScript @startParams
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Guardian start script returned exit code $LASTEXITCODE"
    }

    $tasks = Get-ScheduledTask -TaskName $TaskNameLogon, $TaskNameKeepAlive -ErrorAction SilentlyContinue |
        Select-Object TaskName, State

    Write-Host ""
    Write-Host "=== Installed Tasks ==="
    if ($tasks) {
        $tasks | Format-Table -AutoSize
    } else {
        Write-Host "[WARN] Tasks not found after installation."
    }

    Write-Host ""
    Write-Host "Guardian logs:"
    Write-Host "  logs/automation/openclaw_guardian_stdout.log"
    Write-Host "  logs/automation/openclaw_guardian_stderr.log"

    if (-not $keepAliveTaskOk) {
        Write-Warning "KeepAlive task could not be created. Run this script from an elevated PowerShell session."
        exit 2
    }
    if (-not $logonTaskOk) {
        Write-Warning "Logon task was not created (permission-limited shell). KeepAlive task is active and will maintain guardian persistence."
    }
} else {
    Disable-ConflictingTasks
    $null = Harden-GatewayTaskPolicy
    Write-Host "[DRY-RUN] Skipping task verification/repair and guardian start."
}

Write-Host ""
Write-Host "Done."
exit 0
