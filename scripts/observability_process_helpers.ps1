Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    return (Split-Path -Parent $PSScriptRoot)
}

function Resolve-PreferredPython {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot
    )

    $candidates = @()
    if ($env:PMX_OBSERVABILITY_PYTHON) { $candidates += $env:PMX_OBSERVABILITY_PYTHON }
    $candidates += (Join-Path $RepoRoot "simpleTrader_env\Scripts\python.exe")
    $candidates += "python"

    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        if (Test-Path $candidate) { return $candidate }
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }

    throw "Unable to locate a usable Python interpreter for observability startup."
}

function Resolve-RequiredPath {
    param(
        [Parameter(Mandatory = $true)][string[]]$Candidates,
        [Parameter(Mandatory = $true)][string]$Label
    )

    foreach ($candidate in $Candidates) {
        if (-not $candidate) { continue }
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }

    throw "Unable to locate $Label. Checked: $($Candidates -join ', ')"
}

function Start-RepoProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StdOutPath,
        [Parameter(Mandatory = $true)][string]$StdErrPath,
        [switch]$Foreground
    )

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StdOutPath) | Out-Null

    if ($Foreground) {
        & $FilePath @ArgumentList
        return $LASTEXITCODE
    }

    $proc = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdOutPath `
        -RedirectStandardError $StdErrPath `
        -PassThru
    return $proc.Id
}
