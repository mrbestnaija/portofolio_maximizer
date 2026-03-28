param(
    [switch]$Foreground,
    [string]$AlertmanagerExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "observability_process_helpers.ps1")

$repoRoot = Resolve-RepoRoot
$exe = Resolve-RequiredPath -Label "alertmanager.exe" -Candidates @(
    $AlertmanagerExe,
    $env:PMX_ALERTMANAGER_EXE,
    (Join-Path $repoRoot "tools\observability\alertmanager\alertmanager.exe")
)
$configPath = Join-Path $repoRoot "observability\alertmanager\alertmanager.yml"
$storagePath = Join-Path $repoRoot "data\alertmanager"
$logDir = Join-Path $repoRoot "logs\observability"
$stdoutPath = Join-Path $logDir "alertmanager.stdout.log"
$stderrPath = Join-Path $logDir "alertmanager.stderr.log"
New-Item -ItemType Directory -Force -Path $storagePath | Out-Null
$args = @(
    "--config.file=$configPath",
    "--storage.path=$storagePath",
    "--web.listen-address=127.0.0.1:9093"
)

Start-RepoProcess `
    -FilePath $exe `
    -ArgumentList $args `
    -WorkingDirectory $repoRoot `
    -StdOutPath $stdoutPath `
    -StdErrPath $stderrPath `
    -Foreground:$Foreground
