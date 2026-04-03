param(
    [switch]$Foreground,
    [string]$LokiExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "observability_process_helpers.ps1")

$repoRoot = Resolve-RepoRoot
$exeCandidates = Normalize-PathCandidates @(
    $LokiExe,
    $env:PMX_LOKI_EXE,
    (Join-Path $repoRoot "tools\observability\loki\loki*.exe")
)
$exe = Resolve-RequiredPath -Label "loki.exe" -Candidates $exeCandidates
$configPath = Join-Path $repoRoot "observability\loki\loki.yml"
$dataPath = Join-Path $repoRoot "data\loki"
$logDir = Join-Path $repoRoot "logs\observability"
$stdoutPath = Join-Path $logDir "loki.stdout.log"
$stderrPath = Join-Path $logDir "loki.stderr.log"
New-Item -ItemType Directory -Force -Path $dataPath | Out-Null

$env:PMX_LOKI_DATA_PATH = $dataPath
$args = @(
    "-config.file=$configPath",
    "-config.expand-env=true"
)
$existingProcessIds = @(
    (Get-ObservedProcessIds -ExecutablePathLike (Join-Path $repoRoot "tools\observability\loki\*.exe"))
    (Get-ListeningProcessIdsByPort -Port 3100)
) | Sort-Object -Unique

Ensure-RepoService `
    -Label "loki" `
    -FilePath $exe `
    -ArgumentList $args `
    -WorkingDirectory $repoRoot `
    -StdOutPath $stdoutPath `
    -StdErrPath $stderrPath `
    -HealthUrl "http://127.0.0.1:3100/metrics" `
    -ExistingProcessIds $existingProcessIds `
    -StartupTimeoutSec 20 `
    -Foreground:$Foreground
