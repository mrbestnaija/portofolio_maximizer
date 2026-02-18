<#
.SYNOPSIS
    Installs a Windows Scheduled Task that runs OpenClaw WhatsApp maintenance
    every 15 minutes, ensuring persistent connectivity.

.DESCRIPTION
    Creates two scheduled tasks:
    1. PMX-OpenClaw-Maintenance: Runs openclaw_maintenance.py every 15 minutes
       to detect and heal broken WhatsApp connections (DNS failures, session
       drops, gateway crashes).
    2. PMX-OpenClaw-GatewayStart: Runs at user logon to ensure the OpenClaw
       gateway is started and WhatsApp listener is connected.

    Both tasks run under the current user context and do NOT require
    Administrator privileges (they use the current user's session).

.PARAMETER Uninstall
    Remove the scheduled tasks instead of installing them.

.PARAMETER IntervalMinutes
    How often to run the maintenance task (default: 15).

.PARAMETER DryRun
    Show what would be done without creating tasks.

.EXAMPLE
    .\install_whatsapp_watchdog.ps1
    .\install_whatsapp_watchdog.ps1 -IntervalMinutes 10
    .\install_whatsapp_watchdog.ps1 -Uninstall
#>

Param(
    [switch]$Uninstall,
    [int]$IntervalMinutes = 15,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$TaskNameMaintenance = "PMX-OpenClaw-Maintenance"
$TaskNameGateway     = "PMX-OpenClaw-GatewayStart"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# --- Resolve Python interpreter ---
$pythonExe = $null
$venvPython = Join-Path $repoRoot "simpleTrader_env\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} elseif ($env:PMX_PYTHON_BIN -and (Test-Path $env:PMX_PYTHON_BIN)) {
    $pythonExe = $env:PMX_PYTHON_BIN
} else {
    $pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $pythonExe -or -not (Test-Path $pythonExe)) {
    Write-Error "Cannot find Python interpreter. Activate venv or set PMX_PYTHON_BIN."
    exit 1
}

# --- Uninstall ---
if ($Uninstall) {
    foreach ($name in @($TaskNameMaintenance, $TaskNameGateway)) {
        $existing = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if ($existing) {
            if ($DryRun) {
                Write-Host "[DRY-RUN] Would remove scheduled task: $name"
            } else {
                Unregister-ScheduledTask -TaskName $name -Confirm:$false
                Write-Host "[OK] Removed scheduled task: $name"
            }
        } else {
            Write-Host "[SKIP] Task not found: $name"
        }
    }
    exit 0
}

# --- Build maintenance command ---
$maintScript = Join-Path $repoRoot "scripts\openclaw_maintenance.py"
if (-not (Test-Path $maintScript)) {
    Write-Error "Maintenance script not found: $maintScript"
    exit 1
}

$maintArgs = @(
    "`"$maintScript`"",
    "--apply",
    "--restart-gateway-on-rpc-failure",
    "--attempt-primary-reenable",
    "--primary-channel", "whatsapp",
    "--primary-restart-attempts", "2",
    "--recheck-delay-seconds", "8",
    "--report-file", "`"$(Join-Path $repoRoot 'logs\automation\openclaw_maintenance_latest.json')`""
)

# --- Task 1: Periodic maintenance (every N minutes) ---
Write-Host ""
Write-Host "=== Task 1: $TaskNameMaintenance (every $IntervalMinutes min) ==="

$maintAction = New-ScheduledTaskAction `
    -Execute $pythonExe `
    -Argument ($maintArgs -join " ") `
    -WorkingDirectory $repoRoot

$maintTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 9999)

$maintSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

if ($DryRun) {
    Write-Host "[DRY-RUN] Would create: $TaskNameMaintenance"
    Write-Host "  Execute: $pythonExe"
    Write-Host "  Args: $($maintArgs -join ' ')"
    Write-Host "  Interval: $IntervalMinutes minutes"
} else {
    $existing = Get-ScheduledTask -TaskName $TaskNameMaintenance -ErrorAction SilentlyContinue
    if ($existing) {
        Set-ScheduledTask -TaskName $TaskNameMaintenance `
            -Action $maintAction -Trigger $maintTrigger -Settings $maintSettings | Out-Null
        Write-Host "[OK] Updated existing task: $TaskNameMaintenance"
    } else {
        Register-ScheduledTask -TaskName $TaskNameMaintenance `
            -Action $maintAction -Trigger $maintTrigger -Settings $maintSettings `
            -Description "PMX: OpenClaw WhatsApp maintenance guard (session heal, gateway restart)" | Out-Null
        Write-Host "[OK] Created task: $TaskNameMaintenance"
    }
}

# --- Task 2: Gateway startup at logon ---
Write-Host ""
Write-Host "=== Task 2: $TaskNameGateway (at logon) ==="

$gatewayAction = New-ScheduledTaskAction `
    -Execute "openclaw" `
    -Argument "gateway start" `
    -WorkingDirectory $repoRoot

$gatewayTrigger = New-ScheduledTaskTrigger -AtLogOn

$gatewaySettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -MultipleInstances IgnoreNew

if ($DryRun) {
    Write-Host "[DRY-RUN] Would create: $TaskNameGateway"
    Write-Host "  Execute: openclaw gateway start"
} else {
    $existing = Get-ScheduledTask -TaskName $TaskNameGateway -ErrorAction SilentlyContinue
    if ($existing) {
        Set-ScheduledTask -TaskName $TaskNameGateway `
            -Action $gatewayAction -Trigger $gatewayTrigger -Settings $gatewaySettings | Out-Null
        Write-Host "[OK] Updated existing task: $TaskNameGateway"
    } else {
        Register-ScheduledTask -TaskName $TaskNameGateway `
            -Action $gatewayAction -Trigger $gatewayTrigger -Settings $gatewaySettings `
            -Description "PMX: Start OpenClaw gateway at logon for WhatsApp connectivity" | Out-Null
        Write-Host "[OK] Created task: $TaskNameGateway"
    }
}

# --- Summary ---
Write-Host ""
Write-Host "=== WhatsApp Watchdog Installation Complete ==="
Write-Host "  Maintenance runs every $IntervalMinutes minutes"
Write-Host "  Gateway starts at logon"
Write-Host "  Reports: logs/automation/openclaw_maintenance_latest.json"
Write-Host ""
Write-Host "To verify:"
Write-Host "  Get-ScheduledTask -TaskName 'PMX-*' | Format-Table TaskName, State"
Write-Host ""
Write-Host "To uninstall:"
Write-Host "  .\install_whatsapp_watchdog.ps1 -Uninstall"
