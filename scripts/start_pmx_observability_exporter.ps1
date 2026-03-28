param(
    [switch]$Foreground,
    [string]$Bind = "127.0.0.1",
    [int]$Port = 9765
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "observability_process_helpers.ps1")

$repoRoot = Resolve-RepoRoot
$pythonExe = Resolve-PreferredPython -RepoRoot $repoRoot
$scriptPath = Join-Path $repoRoot "scripts\pmx_observability_exporter.py"
$logDir = Join-Path $repoRoot "logs\observability"
$stdoutPath = Join-Path $logDir "pmx_observability_exporter.stdout.log"
$stderrPath = Join-Path $logDir "pmx_observability_exporter.stderr.log"
$statePath = Join-Path $logDir "exporter_state.json"
$args = @($scriptPath, "--bind", $Bind, "--port", "$Port", "--state-path", $statePath)
$existingProcessIds = @(Get-ListeningProcessIdsByPort -Port $Port)

Ensure-RepoService `
    -Label "pmx_observability_exporter" `
    -FilePath $pythonExe `
    -ArgumentList $args `
    -WorkingDirectory $repoRoot `
    -StdOutPath $stdoutPath `
    -StdErrPath $stderrPath `
    -HealthUrl ("http://" + $Bind + ":" + $Port + "/healthz") `
    -ExistingProcessIds $existingProcessIds `
    -StartupTimeoutSec 20 `
    -Foreground:$Foreground
