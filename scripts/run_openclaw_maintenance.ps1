Param(
    [bool]$Apply = $true,
    [bool]$DisableBrokenChannels = $true,
    [bool]$RestartGatewayOnFailure = $true,
    [string]$PrimaryChannel = "whatsapp"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = if ($env:PMX_PYTHON_BIN -and (Test-Path $env:PMX_PYTHON_BIN)) {
    $env:PMX_PYTHON_BIN
} else {
    "python"
}

function Test-WslReady {
    try {
        $null = Get-Command wsl -ErrorAction Stop
        $distros = & wsl -l -q 2>$null
        return [bool]($distros -and $distros.Trim())
    } catch {
        return $false
    }
}

if (Test-WslReady) {
    $repoWsl = (& wsl wslpath -a "$repoRoot").Trim()
    if (-not $repoWsl) {
        throw "Failed to resolve WSL path for repo root."
    }

    $envParts = @(
        "CRON_OPENCLAW_MAINTENANCE_APPLY=$([int]$Apply)",
        "CRON_OPENCLAW_DISABLE_BROKEN_CHANNELS=$([int]$DisableBrokenChannels)",
        "CRON_OPENCLAW_RESTART_GATEWAY_ON_FAILURE=$([int]$RestartGatewayOnFailure)",
        "CRON_OPENCLAW_PRIMARY_CHANNEL='$PrimaryChannel'"
    )
    $cmd = "cd '$repoWsl' && " + ($envParts -join " ") + " bash/production_cron.sh openclaw_maintenance"
    & wsl bash -lc $cmd
    exit $LASTEXITCODE
}

Set-Location $repoRoot
$args = @(
    "scripts/openclaw_maintenance.py",
    "--primary-channel", $PrimaryChannel
)
if ($Apply) {
    $args += "--apply"
}
if ($DisableBrokenChannels) {
    $args += "--disable-broken-channels"
}
if ($RestartGatewayOnFailure) {
    $args += "--restart-gateway-on-rpc-failure"
}

& $pythonExe @args
exit $LASTEXITCODE

