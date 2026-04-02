param(
    [switch]$Foreground,
    [string]$AlloyExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "observability_process_helpers.ps1")

$repoRoot = Resolve-RepoRoot
$exeCandidates = Normalize-PathCandidates @(
    $AlloyExe,
    $env:PMX_ALLOY_EXE,
    (Join-Path $env:ProgramFiles "GrafanaLabs\Alloy\alloy.exe"),
    (Join-Path $repoRoot "tools\observability\alloy\alloy*.exe")
)
$exe = Resolve-RequiredPath -Label "alloy.exe" -Candidates $exeCandidates
$configPath = Join-Path $repoRoot "observability\alloy\logs.alloy"
$storagePath = Join-Path $repoRoot "data\alloy"
$logDir = Join-Path $repoRoot "logs\observability"
$stdoutPath = Join-Path $logDir "alloy.stdout.log"
$stderrPath = Join-Path $logDir "alloy.stderr.log"
New-Item -ItemType Directory -Force -Path $storagePath | Out-Null

$env:PMX_REPO_ROOT = $repoRoot
$env:PMX_LOKI_URL = "http://127.0.0.1:3100"
$args = @(
    "run",
    $configPath,
    "--storage.path=$storagePath",
    "--server.http.listen-addr=127.0.0.1:12345"
)
$existingProcessIds = @(
    (Get-ObservedProcessIds -ExecutablePathLike (Join-Path $repoRoot "tools\observability\alloy\*.exe"))
    (Get-ListeningProcessIdsByPort -Port 12345)
) | Sort-Object -Unique

Ensure-RepoService `
    -Label "alloy" `
    -FilePath $exe `
    -ArgumentList $args `
    -WorkingDirectory $repoRoot `
    -StdOutPath $stdoutPath `
    -StdErrPath $stderrPath `
    -HealthUrl "http://127.0.0.1:12345/metrics" `
    -ExistingProcessIds $existingProcessIds `
    -StartupTimeoutSec 20 `
    -Foreground:$Foreground
