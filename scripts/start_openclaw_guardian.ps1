Param(
    [int]$WatchIntervalSeconds = 120,
    [int]$FastSupervisorIntervalSeconds = 5,
    [string]$PrimaryChannel = "whatsapp",
    [switch]$NoApply,
    [string]$OpenClawCommand = "",
    [string]$IntegrityUnlinkedCloseWhitelistIds = "",
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = if ($env:PMX_PYTHON_BIN -and (Test-Path $env:PMX_PYTHON_BIN)) {
    $env:PMX_PYTHON_BIN
} else {
    $venv = Join-Path $repoRoot "simpleTrader_env\Scripts\python.exe"
    if (Test-Path $venv) { $venv } else { "python" }
}

$logDir = Join-Path $repoRoot "logs\automation"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$pidFile = Join-Path $logDir "openclaw_guardian.pid.json"
$stdoutLog = Join-Path $logDir "openclaw_guardian_stdout.log"
$stderrLog = Join-Path $logDir "openclaw_guardian_stderr.log"

if (-not $IntegrityUnlinkedCloseWhitelistIds) {
    $IntegrityUnlinkedCloseWhitelistIds = if ($env:INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS) {
        $env:INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS
    } else {
        "66"
    }
}

if (-not $OpenClawCommand) {
    $OpenClawCommand = if ($env:OPENCLAW_COMMAND) { $env:OPENCLAW_COMMAND } else { "openclaw" }
}
if ($FastSupervisorIntervalSeconds -lt 1) { $FastSupervisorIntervalSeconds = 1 }

function Test-GuardianPid {
    Param([int]$ProcessId)
    if ($ProcessId -le 0) { return $false }
    try {
        $proc = Get-Process -Id $ProcessId -ErrorAction Stop
        return ($null -ne $proc)
    } catch {
        return $false
    }
}

function Get-OpenClawWatchProcesses {
    $rows = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -match "^python" -and
        $_.CommandLine -and
        $_.CommandLine -match "openclaw_maintenance\.py" -and
        $_.CommandLine -match "--watch"
    }
    return @($rows)
}

function Write-GuardianPidFile {
    Param(
        [int]$ProcessId,
        [string]$PythonPath
    )
    $payload = @{
        pid = $ProcessId
        started_at = (Get-Date).ToString("o")
        watch_interval_seconds = [Math]::Max(30, $WatchIntervalSeconds)
        fast_supervisor_interval_seconds = [Math]::Max(1, $FastSupervisorIntervalSeconds)
        primary_channel = $PrimaryChannel
        apply = [bool](-not $NoApply)
        integrity_unlinked_close_whitelist_ids = $IntegrityUnlinkedCloseWhitelistIds
        python = $PythonPath
    }
    $payload | ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8
}

$existingPid = 0
if (Test-Path $pidFile) {
    try {
        $raw = Get-Content -Path $pidFile -Raw | ConvertFrom-Json
        $existingPid = [int]($raw.pid)
    } catch {
        $existingPid = 0
    }
}

if ($existingPid -gt 0 -and (Test-GuardianPid -ProcessId $existingPid)) {
    if (-not $ForceRestart) {
        Write-Host "[openclaw_guardian] already_running pid=$existingPid"
        exit 0
    }
    try {
        Stop-Process -Id $existingPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    } catch {}
}

$watchProcs = Get-OpenClawWatchProcesses
if ($watchProcs.Count -gt 0 -and -not $ForceRestart) {
    $live = $watchProcs | Sort-Object ProcessId | Select-Object -First 1
    $livePid = [int]($live.ProcessId)
    Write-GuardianPidFile -ProcessId $livePid -PythonPath ([string]($live.ExecutablePath))
    Write-Host "[openclaw_guardian] already_running pid=$livePid (discovered by process scan)"
    exit 0
}

if ($watchProcs.Count -gt 0 -and $ForceRestart) {
    foreach ($proc in $watchProcs) {
        $pid = [int]($proc.ProcessId)
        if ($pid -le 0) { continue }
        try {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        } catch {}
    }
    Start-Sleep -Seconds 1
}

$scriptPath = Join-Path $repoRoot "scripts\openclaw_maintenance.py"
if (-not (Test-Path $scriptPath)) {
    Write-Error "Missing script: $scriptPath"
    exit 1
}

$args = @(
    "-u",
    "`"$scriptPath`"",
    "--watch",
    "--watch-interval", "$([Math]::Max(30, $WatchIntervalSeconds))",
    "--fast-supervisor",
    "--fast-supervisor-interval-seconds", "$([Math]::Max(1, $FastSupervisorIntervalSeconds))",
    "--fast-supervisor-failure-threshold", "2",
    "--fast-supervisor-restart-cooldown-seconds", "20",
    "--fast-supervisor-probe-timeout-seconds", "8",
    "--fast-supervisor-post-restart-recheck-seconds", "4",
    "--primary-channel", "$PrimaryChannel",
    "--command", "$OpenClawCommand",
    "--restart-gateway-on-rpc-failure",
    "--attempt-primary-reenable",
    "--primary-restart-attempts", "2",
    "--recheck-delay-seconds", "8",
    "--report-file", "`"$(Join-Path $repoRoot 'logs\automation\openclaw_maintenance_latest.json')`""
)
if (-not $NoApply) {
    $args += "--apply"
}

$env:INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS = $IntegrityUnlinkedCloseWhitelistIds

$proc = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList ($args -join " ") `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog

Write-GuardianPidFile -ProcessId ([int]$proc.Id) -PythonPath $pythonExe

Write-Host "[openclaw_guardian] started pid=$($proc.Id)"
Write-Host "[openclaw_guardian] pid_file=$pidFile"
Write-Host "[openclaw_guardian] stdout_log=$stdoutLog"
Write-Host "[openclaw_guardian] stderr_log=$stderrLog"
