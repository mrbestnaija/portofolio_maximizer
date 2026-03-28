param(
    [switch]$Foreground,
    [string]$Bind = "127.0.0.1",
    [int]$Port = 9766
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "observability_process_helpers.ps1")

$repoRoot = Resolve-RepoRoot
$pythonExe = Resolve-PreferredPython -RepoRoot $repoRoot
$scriptPath = Join-Path $repoRoot "scripts\pmx_alertmanager_bridge.py"
$logDir = Join-Path $repoRoot "logs\observability"
$stdoutPath = Join-Path $logDir "pmx_alertmanager_bridge.stdout.log"
$stderrPath = Join-Path $logDir "pmx_alertmanager_bridge.stderr.log"
$args = @($scriptPath, "--bind", $Bind, "--port", "$Port")

Start-RepoProcess `
    -FilePath $pythonExe `
    -ArgumentList $args `
    -WorkingDirectory $repoRoot `
    -StdOutPath $stdoutPath `
    -StdErrPath $stderrPath `
    -Foreground:$Foreground
