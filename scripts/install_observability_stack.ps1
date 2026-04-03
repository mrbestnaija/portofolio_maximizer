param(
    [switch]$Uninstall,
    [switch]$DownloadOfficialBinaries,
    [string]$PrometheusZipUrl = "",
    [string]$AlertmanagerZipUrl = "",
    [string]$GrafanaZipUrl = "",
    [string]$LokiZipUrl = "",
    [string]$AlloyZipUrl = "",
    [string]$LokiZipPath = "",
    [string]$AlloyZipPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$startupCmd = Join-Path $startupDir "PMX-Observability-Stack.cmd"
$toolsRoot = Join-Path $repoRoot "tools\observability"
$DefaultPrometheusZipUrl = "https://github.com/prometheus/prometheus/releases/download/v3.10.0/prometheus-3.10.0.windows-amd64.zip"
$DefaultAlertmanagerZipUrl = "https://github.com/prometheus/alertmanager/releases/download/v0.31.1/alertmanager-0.31.1.windows-amd64.zip"
$DefaultGrafanaZipUrl = "https://dl.grafana.com/oss/release/grafana-12.4.2.windows-amd64.zip"
$DefaultLokiZipUrl = "https://github.com/grafana/loki/releases/download/v3.6.3/loki-windows-amd64.exe.zip"
$DefaultAlloyZipUrl = "https://github.com/grafana/alloy/releases/download/v1.14.0/alloy-windows-amd64.exe.zip"

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
    if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir }
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        Write-Status ("downloading " + $Name + " via curl resume -> " + $tmpZip)
        & curl.exe -L --fail --retry 5 --retry-delay 5 --retry-all-errors -C - $Url -o $tmpZip
        if ($LASTEXITCODE -ne 0) {
            throw "curl download failed for $Name from $Url (exit $LASTEXITCODE)"
        }
    }
    else {
        Write-Status ("downloading " + $Name + " via Invoke-WebRequest -> " + $tmpZip)
        Invoke-WebRequest -Uri $Url -OutFile $tmpZip -TimeoutSec 0
    }
    Expand-Archive -LiteralPath $tmpZip -DestinationPath $tmpDir -Force
    Install-ExtractedPackage -ExtractedRoot $tmpDir -Name $Name -DestinationRoot $DestinationRoot -SourceDescription $Url
}

function Install-ExtractedPackage {
    param(
        [Parameter(Mandatory = $true)][string]$ExtractedRoot,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$DestinationRoot,
        [Parameter(Mandatory = $true)][string]$SourceDescription
    )

    $children = @(Get-ChildItem -Path $ExtractedRoot -Force)
    if ($children.Count -eq 0) {
        throw "Failed to extract $Name from $SourceDescription"
    }
    $dest = Join-Path $DestinationRoot $Name
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    if ($children.Count -eq 1 -and $children[0].PSIsContainer) {
        Copy-Item -Recurse -Force -Path $children[0].FullName -Destination $dest
    }
    else {
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        foreach ($child in $children) {
            Copy-Item -Recurse -Force -Path $child.FullName -Destination $dest
        }
    }
    Write-Status ("installed " + $Name + " -> " + $dest)
}

function Install-LocalZip {
    param(
        [Parameter(Mandatory = $true)][string]$ZipPath,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$DestinationRoot
    )

    if (-not (Test-Path -LiteralPath $ZipPath)) {
        throw "Local zip not found for ${Name}: $ZipPath"
    }

    $tmpDir = Join-Path $env:TEMP ("pmx_" + $Name + "_extract")
    if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir }
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $tmpDir -Force
    Install-ExtractedPackage -ExtractedRoot $tmpDir -Name $Name -DestinationRoot $DestinationRoot -SourceDescription $ZipPath
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
    (Join-Path $repoRoot "data\loki"), `
    (Join-Path $repoRoot "data\alloy"), `
    (Join-Path $repoRoot "logs\observability"), `
    $toolsRoot | Out-Null

if ($DownloadOfficialBinaries -or $LokiZipPath -or $AlloyZipPath) {
    if ($DownloadOfficialBinaries) {
    if (-not $PrometheusZipUrl) {
        $PrometheusZipUrl = $DefaultPrometheusZipUrl
        Write-Status ("using_default_url prometheus " + $PrometheusZipUrl)
    }
    if (-not $AlertmanagerZipUrl) {
        $AlertmanagerZipUrl = $DefaultAlertmanagerZipUrl
        Write-Status ("using_default_url alertmanager " + $AlertmanagerZipUrl)
    }
    if (-not $GrafanaZipUrl) {
        $GrafanaZipUrl = $DefaultGrafanaZipUrl
        Write-Status ("using_default_url grafana " + $GrafanaZipUrl)
    }
    if (-not $LokiZipUrl) {
        $LokiZipUrl = $DefaultLokiZipUrl
        Write-Status ("using_default_url loki " + $LokiZipUrl)
    }
    if (-not $AlloyZipUrl) {
        $AlloyZipUrl = $DefaultAlloyZipUrl
        Write-Status ("using_default_url alloy " + $AlloyZipUrl)
    }

        Expand-OfficialZip -Url $PrometheusZipUrl -Name "prometheus" -DestinationRoot $toolsRoot
        Expand-OfficialZip -Url $AlertmanagerZipUrl -Name "alertmanager" -DestinationRoot $toolsRoot
        Expand-OfficialZip -Url $GrafanaZipUrl -Name "grafana" -DestinationRoot $toolsRoot
    }
    if ($LokiZipPath) {
        Write-Status ("using_local_zip loki " + $LokiZipPath)
        Install-LocalZip -ZipPath $LokiZipPath -Name "loki" -DestinationRoot $toolsRoot
    }
    elseif ($DownloadOfficialBinaries) {
        Expand-OfficialZip -Url $LokiZipUrl -Name "loki" -DestinationRoot $toolsRoot
    }
    if ($AlloyZipPath) {
        Write-Status ("using_local_zip alloy " + $AlloyZipPath)
        Install-LocalZip -ZipPath $AlloyZipPath -Name "alloy" -DestinationRoot $toolsRoot
    }
    elseif ($DownloadOfficialBinaries) {
        Expand-OfficialZip -Url $AlloyZipUrl -Name "alloy" -DestinationRoot $toolsRoot
    }
}

Install-StartupShortcut
Write-Status "install_complete"
