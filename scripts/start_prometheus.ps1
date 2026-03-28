param(
    [switch]$Foreground,
    [string]$PrometheusExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "observability_process_helpers.ps1")

$repoRoot = Resolve-RepoRoot
$exe = Resolve-RequiredPath -Label "prometheus.exe" -Candidates @(
    $PrometheusExe,
    $env:PMX_PROMETHEUS_EXE,
    (Join-Path $repoRoot "tools\observability\prometheus\prometheus.exe")
)
$configPath = Join-Path $repoRoot "observability\prometheus\prometheus.yml"
$storagePath = Join-Path $repoRoot "data\prometheus"
$logDir = Join-Path $repoRoot "logs\observability"
$stdoutPath = Join-Path $logDir "prometheus.stdout.log"
$stderrPath = Join-Path $logDir "prometheus.stderr.log"
New-Item -ItemType Directory -Force -Path $storagePath | Out-Null
$args = @(
    "--config.file=$configPath",
    "--storage.tsdb.path=$storagePath",
    "--storage.tsdb.retention.time=14d",
    "--web.listen-address=127.0.0.1:9090"
)

Start-RepoProcess `
    -FilePath $exe `
    -ArgumentList $args `
    -WorkingDirectory $repoRoot `
    -StdOutPath $stdoutPath `
    -StdErrPath $stderrPath `
    -Foreground:$Foreground
