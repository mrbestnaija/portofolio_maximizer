Param(
    [int]$StartHour = 7,
    [int]$EndHour = 20,
    [switch]$SkipOutsideMarketHours = $true
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wrapper = Join-Path $repoRoot "run_core_auto_trader_once.bat"

function Test-MarketHours {
    param(
        [Parameter(Mandatory = $true)][datetime]$Now,
        [Parameter(Mandatory = $true)][int]$WindowStartHour,
        [Parameter(Mandatory = $true)][int]$WindowEndHour
    )

    if ($Now.DayOfWeek -eq [DayOfWeek]::Saturday -or $Now.DayOfWeek -eq [DayOfWeek]::Sunday) {
        return $false
    }
    return ($Now.Hour -ge $WindowStartHour -and $Now.Hour -lt $WindowEndHour)
}

if (-not (Test-Path -LiteralPath $wrapper)) {
    throw "Core auto-trader wrapper missing: $wrapper"
}

$now = Get-Date
if ($SkipOutsideMarketHours -and -not (Test-MarketHours -Now $now -WindowStartHour $StartHour -WindowEndHour $EndHour)) {
    Write-Host ("[CORE] Skipping outside market hours: {0:yyyy-MM-dd HH:mm:ss}" -f $now)
    exit 0
}

Write-Host "[CORE] Launching core auto-trader wrapper"
& $wrapper
exit $LASTEXITCODE
