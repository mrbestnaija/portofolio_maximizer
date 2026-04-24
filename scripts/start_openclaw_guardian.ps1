Param(
    [int]$WatchIntervalSeconds = 120,
    [int]$FastSupervisorIntervalSeconds = 60,
    [string]$PrimaryChannel = "whatsapp",
    [bool]$DisableBrokenChannels = $true,
    [switch]$EnsureFunctionalState,
    [switch]$Quiet,
    [switch]$NoApply,
    [string]$OpenClawCommand = "",
    [string]$IntegrityUnlinkedCloseWhitelistIds = "",
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"
if (-not $PSBoundParameters.ContainsKey("Quiet")) {
    $Quiet = $true
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepoPython {
    param([Parameter(Mandatory = $true)][string]$RepoRoot)

    if ($env:PMX_PYTHON_BIN -and (Test-Path -LiteralPath $env:PMX_PYTHON_BIN)) {
        return $env:PMX_PYTHON_BIN
    }

    $candidates = @(
        (Join-Path $RepoRoot "simpleTrader_env_win\Scripts\python.exe"),
        (Join-Path $RepoRoot "simpleTrader_env\Scripts\python.exe"),
        (Join-Path $RepoRoot "simpleTrader_env\bin\python")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return "python"
}

$pythonExe = Resolve-RepoPython -RepoRoot $repoRoot
$remoteWorkflowScript = Join-Path $repoRoot "scripts\openclaw_remote_workflow.py"
$execEnvScript = Join-Path $repoRoot "scripts\enforce_openclaw_exec_environment.py"

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

function Write-GuardianInfo {
    Param([string]$Message)
    if (-not $Quiet) {
        Write-Host $Message
    }
}

function Get-EnvInt {
    Param(
        [string]$Name,
        [int]$DefaultValue,
        [int]$Minimum = 0
    )
    $raw = [Environment]::GetEnvironmentVariable($Name)
    $value = $DefaultValue
    if (-not [string]::IsNullOrWhiteSpace($raw)) {
        $parsed = 0
        if ([int]::TryParse($raw, [ref]$parsed)) {
            $value = $parsed
        }
    }
    if ($value -lt $Minimum) { return $Minimum }
    return $value
}

$FastSupervisorFailureThreshold = Get-EnvInt -Name "OPENCLAW_FAST_SUPERVISOR_FAILURE_THRESHOLD" -DefaultValue 3 -Minimum 1
$FastSupervisorRestartCooldownSeconds = Get-EnvInt -Name "OPENCLAW_FAST_SUPERVISOR_RESTART_COOLDOWN_SECONDS" -DefaultValue 300 -Minimum 0
$FastSupervisorProbeTimeoutSeconds = Get-EnvInt -Name "OPENCLAW_FAST_SUPERVISOR_PROBE_TIMEOUT_SECONDS" -DefaultValue 12 -Minimum 3
$FastSupervisorPostRestartRecheckSeconds = Get-EnvInt -Name "OPENCLAW_FAST_SUPERVISOR_POST_RESTART_RECHECK_SECONDS" -DefaultValue 12 -Minimum 0
$PrimaryRestartAttempts = Get-EnvInt -Name "OPENCLAW_PRIMARY_RESTART_ATTEMPTS" -DefaultValue 1 -Minimum 1

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

function Invoke-RemoteWorkflowCommand {
    Param(
        [string[]]$Arguments,
        [int]$SuccessCode = 0,
        [int[]]$AcceptCodes = @()
    )
    $args = @($remoteWorkflowScript) + $Arguments
    & $pythonExe @args | Out-Null
    $rc = [int]$LASTEXITCODE
    if ($AcceptCodes -contains $rc) {
        return [pscustomobject]@{
            ReturnCode = $rc
            Success = $true
        }
    }
    return [pscustomobject]@{
        ReturnCode = $rc
        Success = ($rc -eq $SuccessCode)
    }
}

function Test-FunctionalState {
    $result = Invoke-RemoteWorkflowCommand -Arguments @("health", "--json") -AcceptCodes @(0, 1)
    return $result
}

function Invoke-FunctionalRecovery {
    $health = Test-FunctionalState
    if ($health.Success) {
        return [pscustomobject]@{
            Healthy = $true
            ReturnCode = $health.ReturnCode
            RestartedGateway = $false
        }
    }

    Write-GuardianInfo "[openclaw_guardian] health_check_failed rc=$($health.ReturnCode); attempting gateway recovery"
    $restart = Invoke-RemoteWorkflowCommand -Arguments @("gateway-restart", "--json")
    Start-Sleep -Seconds 5
    $postHealth = Test-FunctionalState
    return [pscustomobject]@{
        Healthy = [bool]$postHealth.Success
        ReturnCode = $postHealth.ReturnCode
        RestartedGateway = [bool]$restart.Success
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
        fast_supervisor_failure_threshold = $FastSupervisorFailureThreshold
        fast_supervisor_restart_cooldown_seconds = $FastSupervisorRestartCooldownSeconds
        fast_supervisor_probe_timeout_seconds = $FastSupervisorProbeTimeoutSeconds
        fast_supervisor_post_restart_recheck_seconds = $FastSupervisorPostRestartRecheckSeconds
        primary_restart_attempts = $PrimaryRestartAttempts
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
    if ($EnsureFunctionalState -and -not $ForceRestart) {
        $recovery = Invoke-FunctionalRecovery
        if ($recovery.Healthy) {
            Write-GuardianInfo "[openclaw_guardian] already_running pid=$existingPid functional_state=ok"
            exit 0
        }
        Write-GuardianInfo "[openclaw_guardian] forcing restart for existing pid=$existingPid functional_state=fail rc=$($recovery.ReturnCode)"
        $ForceRestart = $true
    }
    if (-not $ForceRestart) {
        Write-GuardianInfo "[openclaw_guardian] already_running pid=$existingPid"
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
    if ($EnsureFunctionalState) {
        $recovery = Invoke-FunctionalRecovery
        if (-not $recovery.Healthy) {
            Write-GuardianInfo "[openclaw_guardian] forcing restart for discovered pid=$livePid functional_state=fail rc=$($recovery.ReturnCode)"
            $ForceRestart = $true
        }
    }
    if (-not $ForceRestart) {
        Write-GuardianPidFile -ProcessId $livePid -PythonPath ([string]($live.ExecutablePath))
        Write-GuardianInfo "[openclaw_guardian] already_running pid=$livePid (discovered by process scan)"
        exit 0
    }
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
    "--fast-supervisor-failure-threshold", "$FastSupervisorFailureThreshold",
    "--fast-supervisor-restart-cooldown-seconds", "$FastSupervisorRestartCooldownSeconds",
    "--fast-supervisor-probe-timeout-seconds", "$FastSupervisorProbeTimeoutSeconds",
    "--fast-supervisor-post-restart-recheck-seconds", "$FastSupervisorPostRestartRecheckSeconds",
    "--primary-channel", "$PrimaryChannel",
    "--command", "$OpenClawCommand",
    "--restart-gateway-on-rpc-failure",
    "--attempt-primary-reenable",
    "--primary-restart-attempts", "$PrimaryRestartAttempts",
    "--recheck-delay-seconds", "8",
    "--report-file", "`"$(Join-Path $repoRoot 'logs\automation\openclaw_maintenance_latest.json')`""
)
if (-not $NoApply) {
    $args += "--apply"
}
if ($DisableBrokenChannels) {
    $args += "--disable-broken-channels"
}

$env:INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS = $IntegrityUnlinkedCloseWhitelistIds

$execEnvArgs = @($execEnvScript)
$execEnvOutput = & $pythonExe @execEnvArgs 2>&1
$execEnvRc = [int]$LASTEXITCODE
if ($execEnvRc -ne 0) {
    $hadOutput = $false
    foreach ($line in @($execEnvOutput)) {
        $text = [string]$line
        if ([string]::IsNullOrWhiteSpace($text)) { continue }
        $hadOutput = $true
        Write-Error $text
    }
    if (-not $hadOutput) {
        Write-Error "[enforce_openclaw_exec_environment] failed rc=$execEnvRc"
    }
    exit $execEnvRc
}
if (-not $Quiet) {
    foreach ($line in @($execEnvOutput)) {
        $text = [string]$line
        if ([string]::IsNullOrWhiteSpace($text)) { continue }
        Write-Host $text
    }
}

$proc = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList ($args -join " ") `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog

Write-GuardianPidFile -ProcessId ([int]$proc.Id) -PythonPath $pythonExe

Write-GuardianInfo "[openclaw_guardian] started pid=$($proc.Id)"
Write-GuardianInfo "[openclaw_guardian] pid_file=$pidFile"
Write-GuardianInfo "[openclaw_guardian] stdout_log=$stdoutLog"
Write-GuardianInfo "[openclaw_guardian] stderr_log=$stderrLog"

if ($EnsureFunctionalState) {
    Start-Sleep -Seconds 8
    $recovery = Invoke-FunctionalRecovery
    if (-not $recovery.Healthy) {
        Write-Error "[openclaw_guardian] functional_state_check_failed rc=$($recovery.ReturnCode)"
        exit 1
    }
    Write-GuardianInfo "[openclaw_guardian] functional_state=ok"
}
