Param()

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = if ($env:PMX_PYTHON_BIN -and (Test-Path $env:PMX_PYTHON_BIN)) {
    $env:PMX_PYTHON_BIN
} else {
    "python"
}

function Test-WslReady {
    try {
        $null = Get-Command wsl -ErrorAction Stop
        $distros = & wsl -l -q 2>$null
        return [bool]($distros -and $distros.Trim())
    } catch {
        return $false
    }
}

if (Test-WslReady) {
    $repoWsl = (& wsl wslpath -a "$repoRoot").Trim()
    if (-not $repoWsl) {
        throw "Failed to resolve WSL path for repo root."
    }
    $cmd = "cd '$repoWsl' && bash/production_cron.sh self_improvement_review_forward"
    & wsl bash -lc $cmd
    exit $LASTEXITCODE
}

Set-Location $repoRoot
& $pythonExe "scripts/forward_self_improvement_reviews.py"
exit $LASTEXITCODE
