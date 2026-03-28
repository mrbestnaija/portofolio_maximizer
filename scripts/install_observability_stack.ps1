param(
    [switch]$Uninstall,
    [switch]$DownloadOfficialBinaries,
    [string]$PrometheusZipUrl = "",
    [string]$AlertmanagerZipUrl = "",
    [string]$GrafanaZipUrl = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$startupCmd = Join-Path $startupDir "PMX-Observability-Stack.cmd"
$toolsRoot = Join-Path $repoRoot "tools\observability"

function Write-Status {
    param([string]$Message)
    Write-Output ("[observability] " + $Message)
}

function Remove-StartupShortcut {
    if (Test-Path $startupCmd) {
        Remove-Item -Force $startupCmd
        Write-Status "startup_shortcut_removed $startupCmd"
    }
}

function Install-StartupShortcut {
    New-Item -ItemType Directory -Force -Path $startupDir | Out-Null
    $stackScript = Join-Path $repoRoot "scripts\start_observability_stack.ps1"
    $content = "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"$stackScript`"`r`n"
    Set-Content -Path $startupCmd -Value $content -Encoding ASCII
    Write-Status "startup_shortcut_ready $startupCmd"
}

function Expand-OfficialZip {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$DestinationRoot
    )

    if (-not $Url) {
        throw "Missing download URL for $Name."
    }

    New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null
    $tmpZip = Join-Path $env:TEMP ("pmx_" + $Name + ".zip")
    $tmpDir = Join-Path $env:TEMP ("pmx_" + $Name + "_extract")
    if (Test-Path $tmpZip) { Remove-Item -Force $tmpZip }
    if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir }
    Invoke-WebRequest -Uri $Url -OutFile $tmpZip
    Expand-Archive -LiteralPath $tmpZip -DestinationPath $tmpDir -Force
    $root = Get-ChildItem -Path $tmpDir | Select-Object -First 1
    if (-not $root) {
        throw "Failed to extract $Name from $Url"
    }
    $dest = Join-Path $DestinationRoot $Name
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    Copy-Item -Recurse -Force -Path $root.FullName -Destination $dest
    Write-Status ("downloaded " + $Name + " -> " + $dest)
}

if ($Uninstall) {
    Remove-StartupShortcut
    Write-Status "uninstall_complete"
    exit 0
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $repoRoot "data\prometheus"), `
    (Join-Path $repoRoot "data\alertmanager"), `
    (Join-Path $repoRoot "data\grafana"), `
    (Join-Path $repoRoot "logs\observability"), `
    $toolsRoot | Out-Null

if ($DownloadOfficialBinaries) {
    if ($PrometheusZipUrl) {
        Expand-OfficialZip -Url $PrometheusZipUrl -Name "prometheus" -DestinationRoot $toolsRoot
    }
    if ($AlertmanagerZipUrl) {
        Expand-OfficialZip -Url $AlertmanagerZipUrl -Name "alertmanager" -DestinationRoot $toolsRoot
    }
    if ($GrafanaZipUrl) {
        Expand-OfficialZip -Url $GrafanaZipUrl -Name "grafana" -DestinationRoot $toolsRoot
    }
}

Install-StartupShortcut
Write-Status "install_complete"
