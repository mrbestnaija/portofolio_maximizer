# PowerShell Audit Sprint - Windows-native equivalent of run_20_audit_sprint.sh
# Usage: powershell -ExecutionPolicy Bypass -File bash\run_audit_sprint.ps1
# Optional: -AuditRuns 5 -Tickers "AAPL,MSFT,NVDA" -AsOfStartDate "2025-06-01"

param(
    [int]$AuditRuns = 20,
    [string]$Tickers = "AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,JPM,GS,V",
    [int]$LookbackDays = 365,
    [int]$InitialCapital = 25000,
    [int]$Cycles = 1,
    [string]$IntradayInterval = "1h",
    [int]$IntradayHorizon = 6,
    [int]$IntradayLookback = 30,
    [int]$IntradayCycles = 3,
    [int]$WaitBetweenRunsSeconds = 0,
    [switch]$NoProofMode,
    [string]$AsOfStartDate = "",
    [int]$AsOfStepDays = 1,
    [string]$RiskMode = "research_production"
)

$ErrorActionPreference = "Continue"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PythonBin = Join-Path $RootDir "simpleTrader_env\Scripts\python.exe"

if (-not (Test-Path $PythonBin)) {
    Write-Host "[ERROR] Python not found at $PythonBin" -ForegroundColor Red
    exit 1
}

$RunTag = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $RootDir "logs\audit_sprint\$RunTag"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$LogFile = Join-Path $LogDir "run.log"

function Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

# Environment
$env:EXECUTION_MODE = "live"
$env:RISK_MODE = $RiskMode
$env:ENABLE_DATA_CACHE = "0"
$env:ENABLE_CACHE_DELTAS = "0"
$env:TS_FORECAST_AUDIT_DIR = Join-Path $RootDir "logs\forecast_audits"
$env:TS_FORECAST_MONITOR_CONFIG = Join-Path $RootDir "config\forecaster_monitoring.yml"

New-Item -ItemType Directory -Force -Path $env:TS_FORECAST_AUDIT_DIR | Out-Null

$ProofArgs = @()
if (-not $NoProofMode) {
    $ProofArgs += "--proof-mode"
}

$AutoTrader = Join-Path $RootDir "scripts\run_auto_trader.py"
$CheckForecasts = Join-Path $RootDir "scripts\check_forecast_audits.py"
$CheckQuant = Join-Path $RootDir "scripts\check_quant_validation_health.py"
$AuditDashboard = Join-Path $RootDir "scripts\audit_dashboard_payload_sources.py"

Log "[RUNBOOK] Audit sprint started: $RunTag"
Log "[RUNBOOK] Python: $PythonBin"
Log "[RUNBOOK] Logs: $LogDir"
Log "[RUNBOOK] RISK_MODE=$RiskMode"
Log "[RUNBOOK] PROOF_MODE=$(-not $NoProofMode)"
Log "[RUNBOOK] TICKERS=$Tickers"
Log "[RUNBOOK] AUDIT_RUNS=$AuditRuns"
if ($AsOfStartDate) { Log "[RUNBOOK] AS_OF_START_DATE=$AsOfStartDate STEP=$AsOfStepDays" }
Log ""

$GateFailures = 0

for ($run = 1; $run -le $AuditRuns; $run++) {
    Log ""
    Log "============================================================"
    Log "[AUDIT] Run $run/$AuditRuns - $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')"
    Log "============================================================"

    $AsOfArgs = @()
    if ($AsOfStartDate) {
        $offsetDays = ($run - 1) * $AsOfStepDays
        $asOfDate = (Get-Date $AsOfStartDate).AddDays($offsetDays).ToString("yyyy-MM-dd")
        $AsOfArgs += "--as-of-date"
        $AsOfArgs += $asOfDate
        Log "[AUDIT] as_of_date=$asOfDate"
    }

    # Daily run
    $dailyLog = Join-Path $LogDir "audit_${run}_daily.log"
    Log "[STEP] audit_${run}_daily"
    $dailyArgs = @(
        $AutoTrader,
        "--tickers", $Tickers,
        "--lookback-days", $LookbackDays,
        "--initial-capital", $InitialCapital,
        "--cycles", $Cycles,
        "--sleep-seconds", "10",
        "--resume",
        "--bar-aware",
        "--persist-bar-state"
    ) + $AsOfArgs + $ProofArgs

    & $PythonBin @dailyArgs 2>&1 | Tee-Object -FilePath $dailyLog -Append | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        Log "[ERROR] Daily run $run failed (exit=$LASTEXITCODE)"
    }

    # Intraday run
    $intradayLog = Join-Path $LogDir "audit_${run}_intraday.log"
    Log "[STEP] audit_${run}_intraday"
    $intradayArgs = @(
        $AutoTrader,
        "--tickers", $Tickers,
        "--yfinance-interval", $IntradayInterval,
        "--lookback-days", $IntradayLookback,
        "--forecast-horizon", $IntradayHorizon,
        "--initial-capital", $InitialCapital,
        "--cycles", $IntradayCycles,
        "--sleep-seconds", "10",
        "--resume",
        "--bar-aware",
        "--persist-bar-state"
    ) + $AsOfArgs + $ProofArgs

    & $PythonBin @intradayArgs 2>&1 | Tee-Object -FilePath $intradayLog -Append | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        Log "[ERROR] Intraday run $run failed (exit=$LASTEXITCODE)"
    }

    # Forecast audit gate (non-fatal for holdout)
    $gateLog = Join-Path $LogDir "gate_${run}_forecast.log"
    Log "[STEP] gate_${run}_forecast_audits"
    & $PythonBin $CheckForecasts --config-path $env:TS_FORECAST_MONITOR_CONFIG --max-files 500 2>&1 | Tee-Object -FilePath $gateLog -Append | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        $GateFailures++
        Log "[WARN] Forecast gate failed (non-fatal for audit, count=$GateFailures)"
    }

    # Quant health gate
    $quantLog = Join-Path $LogDir "gate_${run}_quant.log"
    Log "[STEP] gate_${run}_quant_health"
    & $PythonBin $CheckQuant 2>&1 | Tee-Object -FilePath $quantLog -Append | Tee-Object -FilePath $LogFile -Append

    # Dashboard audit
    $dashLog = Join-Path $LogDir "gate_${run}_dashboard.log"
    Log "[STEP] gate_${run}_dashboard_audit"
    & $PythonBin $AuditDashboard 2>&1 | Tee-Object -FilePath $dashLog -Append | Tee-Object -FilePath $LogFile -Append

    # Quick-fail check at run level: prune clearly losing strategy
    $quickFailResult = & $PythonBin -c @"
import sqlite3, os, json
db = os.path.join(r'$RootDir', 'data', 'portfolio_maximizer.db')
result = {'stop': False, 'reason': ''}
if os.path.exists(db):
    conn = sqlite3.connect(db)
    try:
        row = conn.execute('''
            SELECT COUNT(*) as n,
                   SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
                   COALESCE(SUM(realized_pnl), 0) as total_pnl
            FROM trade_executions WHERE realized_pnl IS NOT NULL
        ''').fetchone()
        n, wins, losses, total_pnl = row
        if n >= 10:
            wr = wins / n if n > 0 else 0
            if wr < 0.15:
                result = {'stop': True, 'reason': f'Win rate {wr:.0%} < 15% after {n} trades'}
            elif total_pnl < -2500:
                result = {'stop': True, 'reason': f'Total PnL ${total_pnl:.0f} < -$2500 after {n} trades'}
    except Exception:
        pass
    conn.close()
print(json.dumps(result))
"@ 2>$null

    try {
        $qf = $quickFailResult | ConvertFrom-Json
        if ($qf.stop) {
            Log "[QUICK-FAIL] $($qf.reason) -- stopping audit sprint early at run $run/$AuditRuns"
            break
        }
    } catch {
        # Ignore parse errors, continue sprint
    }

    # Wait between runs
    if ($WaitBetweenRunsSeconds -gt 0 -and $run -lt $AuditRuns) {
        Log "[WAIT] Sleeping ${WaitBetweenRunsSeconds}s before next run"
        Start-Sleep -Seconds $WaitBetweenRunsSeconds
    }
}

Log ""
Log "============================================================"
Log "[RUNBOOK] Sprint complete: $AuditRuns runs, $GateFailures gate failures"
Log "============================================================"

# Summary
Log "[SUMMARY] Checking trade and audit counts..."
& $PythonBin -c @"
import sqlite3, os, glob
db = os.path.join(r'$RootDir', 'data', 'portfolio_maximizer.db')
if os.path.exists(db):
    conn = sqlite3.connect(db)
    try:
        total = conn.execute('SELECT COUNT(*) FROM trade_executions').fetchone()[0]
        closed = conn.execute('SELECT COUNT(*) FROM trade_executions WHERE realized_pnl IS NOT NULL').fetchone()[0]
        print(f'[TRADES] {total} total, {closed} closed (with realized PnL)')
    except Exception as e:
        print(f'[TRADES] Error: {e}')
    conn.close()
audit_dir = os.path.join(r'$RootDir', 'logs', 'forecast_audits')
if os.path.isdir(audit_dir):
    audits = glob.glob(os.path.join(audit_dir, '*.json'))
    print(f'[AUDITS] {len(audits)} forecast audit files')
"@
