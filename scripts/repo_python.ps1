[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $repoRoot "simpleTrader_env\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Error "Required repo interpreter not found: $pythonExe"
    exit 1
}

$existingPythonPath = [string]$env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($existingPythonPath)) {
    $env:PYTHONPATH = $repoRoot
} elseif (-not ($existingPythonPath.Split([IO.Path]::PathSeparator) -contains $repoRoot)) {
    $env:PYTHONPATH = "$repoRoot$([IO.Path]::PathSeparator)$existingPythonPath"
}

& $pythonExe @PythonArgs
exit $LASTEXITCODE
