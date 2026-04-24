Param()

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Resolve-RepoPython {
    param([Parameter(Mandatory = $true)][string]$RepoRoot)

    if ($env:PMX_PYTHON_BIN -and (Test-Path -LiteralPath $env:PMX_PYTHON_BIN)) {
        return $env:PMX_PYTHON_BIN
    }

    $candidates = @(
        (Join-Path $RepoRoot "simpleTrader_env_win\Scripts\python.exe"),
        (Join-Path $RepoRoot "simpleTrader_env\Scripts\python.exe"),
        (Join-Path $RepoRoot "simpleTrader_env\bin\python")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return "python"
}

$pythonExe = Resolve-RepoPython -RepoRoot $repoRoot

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
