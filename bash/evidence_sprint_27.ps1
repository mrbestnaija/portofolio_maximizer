# Evidence Sprint: 27 historical --as-of-date runs -> production_eval/ OOS audits
#
# Purpose: Grow RMSE effective_n from ~28 to ~55, targeting violation_rate <= 35%
#          before warmup expires 2026-04-24 (Priority 1 gate blocker).
#
# How it works:
#   Each run calls run_auto_trader.py with a historical --as-of-date, which:
#     1. Fetches AAPL market data up to that date (yfinance cache)
#     2. Fits ensemble forecaster and runs _run_oos_evaluation_audit()
#     3. Writes one audit file to logs/forecast_audits/production_eval/ with
#        evaluation_metrics populated (RMSE_ONLY context)
#   check_forecast_audits.py automatically includes production_eval/ when
#   audit_dir=production/ via _resolve_rmse_audit_roots().
#
# NOTE on Priority 2 (THIN_LINKAGE):
#   --as-of-date runs tag all trades as is_synthetic=1, excluded from
#   production_closed_trades view. THIN_LINKAGE (need matched >= 10) requires
#   LIVE market-hours runs -- run manually during NYSE hours (09:30-16:00 ET):
#     .\simpleTrader_env\Scripts\python.exe scripts\run_auto_trader.py `
#         --tickers AAPL,MSFT,NVDA --cycles 3 --resume
#
# Usage:
#   cd C:\Users\Bestman\personal_projects\portfolio_maximizer_v45\portfolio_maximizer_v45
#   .\bash\evidence_sprint_27.ps1
#
# PRE-REQUISITE (WSL migration): Windows venv was replaced by WSL venvs.
#   If no venv exists, create one first:
#     C:\Users\Bestman\AppData\Local\Programs\Python\Python312\python.exe -m venv simpleTrader_env_win
#     .\simpleTrader_env_win\Scripts\pip install -r requirements.txt
#   Then re-run this script.

$ErrorActionPreference = "Continue"

# Auto-detect Python executable (WSL migration: Scripts\python.exe no longer exists)
$_candidates = @(
    ".\simpleTrader_env_win\Scripts\python.exe",
    ".\simpleTrader_env\Scripts\python.exe",
    ".\simpleTrader_env.windows-backup-20260414003046\Scripts\python.exe",
    "C:\Users\Bestman\AppData\Local\Programs\Python\Python312\python.exe",
    "C:\Python314\python.exe",
    "python"
)
$PYTHON = $null
foreach ($_c in $_candidates) {
    try {
        $null = & $_c --version 2>&1
        if ($LASTEXITCODE -eq 0) { $PYTHON = $_c; break }
    } catch {}
}
if (-not $PYTHON) {
    Write-Error "ERROR: No working Python found. Create a Windows venv first:"
    Write-Error "  C:\Users\Bestman\AppData\Local\Programs\Python\Python312\python.exe -m venv simpleTrader_env_win"
    Write-Error "  .\simpleTrader_env_win\Scripts\pip install -r requirements.txt"
    exit 1
}
Write-Host "Using Python: $PYTHON" -ForegroundColor Cyan
# Verify project packages are available
& $PYTHON -c "import pandas, yfinance" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Python at '$PYTHON' is missing project packages."
    Write-Error "Run:  & '$PYTHON' -m pip install -r requirements.txt"
    exit 1
}
$SCRIPT   = "scripts\run_auto_trader.py"
$TICKERS  = "AAPL"
$LOOKBACK = 365
$LOG_DIR  = "logs\sprint_evidence"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

# 27 dates spanning 2021-2024, one per distinct market regime window.
# Dedup key = (start, end, length, horizon) -- no ticker -- so each date is
# a unique entry in effective_n regardless of ticker.
$dates = @(
    # 2021: COVID recovery / reflation / meme-stock era / low-vol bull
    "2021-02-22",   # post-meme-peak; moderate vol, reflation trade
    "2021-04-05",   # Q1 earnings, steady bull
    "2021-05-17",   # crypto/tech rotation; AAPL pullback
    "2021-07-06",   # recovery after June dip
    "2021-08-16",   # summer bull, low vol
    "2021-10-04",   # Sept correction end
    "2021-11-15",   # pre-Omicron peak bull

    # 2022: Fed hawkish pivot / Ukraine / peak inflation / bear market
    "2022-01-10",   # Fed hawkish pivot shock
    "2022-02-22",   # Ukraine crisis onset
    "2022-04-04",   # bear market onset, rate hike regime
    "2022-05-16",   # peak inflation sell-off
    "2022-07-05",   # bear market rally
    "2022-08-15",   # Jackson Hole pre-shock
    "2022-10-03",   # bear market low
    "2022-11-14",   # relief rally (CPI positive surprise)

    # 2023: banking crisis / AI rally / rate peak / disinflation
    "2023-01-09",   # post-holiday recovery
    "2023-02-21",   # re-rate-hike fear
    "2023-04-03",   # banking crisis stabilized
    "2023-05-15",   # AI rally onset
    "2023-07-10",   # disinflation narrative
    "2023-08-14",   # Fitch downgrade aftermath
    "2023-10-02",   # rate-high-for-longer selloff
    "2023-11-13",   # CPI relief rally
    "2023-12-11",   # Fed pivot expectations

    # 2024: AI bull / tech outperformance
    "2024-01-22",   # AI bull ramp start
    "2024-03-04",   # AAPL Vision Pro era
    "2024-05-06"    # post-earnings consolidation
)

$total         = $dates.Count
$pass_count    = 0
$fail_count    = 0
$skipped_count = 0

Write-Host "=== Evidence Sprint: $total dates ===" -ForegroundColor Cyan
Write-Host "Target: effective_n ~28 -> ~$($total + 28), violation_rate <= 35%" -ForegroundColor Cyan
Write-Host "Logs: $LOG_DIR\" -ForegroundColor Cyan
Write-Host ""

for ($i = 0; $i -lt $dates.Count; $i++) {
    $d   = $dates[$i]
    $idx = $i + 1
    $log = "$LOG_DIR\sprint_${d}.log"

    Write-Host "[$idx/$total] as-of-date=$d ..." -NoNewline

    # Clear env vars that could interfere with OOS eval audit routing.
    # CRITICAL: DataSourceManager reads SYNTHETIC_ONLY via raw os.getenv(), NOT _env_flag().
    # Setting SYNTHETIC_ONLY="0" is a truthy string in Python → forces synthetic-only mode
    # and skips yfinance entirely. Remove it from env entirely instead of setting to "0".
    $env:SKIP_OOS_EVAL_AUDIT = "0"   # _env_flag() is used here, so "0" → False (safe)
    Remove-Item Env:\SYNTHETIC_ONLY  -ErrorAction SilentlyContinue
    Remove-Item Env:\EXECUTION_MODE  -ErrorAction SilentlyContinue

    $t0 = Get-Date
    & $PYTHON $SCRIPT `
        --tickers        $TICKERS `
        --as-of-date     $d `
        --lookback-days  $LOOKBACK `
        --cycles         1 `
        --no-resume `
        --execution-mode auto `
        2>&1 | Tee-Object -FilePath $log | Out-Null
    $exit    = $LASTEXITCODE
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds

    # Detect whether a production_eval audit was written during this run
    $eval_dir  = "logs\forecast_audits\production_eval"
    $new_files = Get-ChildItem $eval_dir -Filter "forecast_audit_*.json" `
                 -ErrorAction SilentlyContinue |
                 Where-Object { $_.LastWriteTime -gt $t0 }

    if ($new_files) {
        $pass_count++
        Write-Host "  OK ($($new_files.Count) eval audit(s), ${elapsed}s)" -ForegroundColor Green
    } elseif ($exit -ne 0) {
        $fail_count++
        Write-Host "  FAIL exit=$exit (${elapsed}s) -- see $log" -ForegroundColor Red
    } else {
        $skipped_count++
        Write-Host "  SKIP no eval audit written (${elapsed}s)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "=== Sprint complete ===" -ForegroundColor Cyan
Write-Host "Runs: $total  |  Eval audits written: $pass_count  |  Failed: $fail_count  |  Skipped: $skipped_count"
Write-Host ""

# Post-sprint: count effective_n in production_eval
Write-Host "--- Counting effective OOS audits ---" -ForegroundColor Cyan
& $PYTHON -c @"
import json, pathlib
eval_dir = pathlib.Path('logs/forecast_audits/production_eval')
if not eval_dir.exists():
    print('production_eval/ not found')
else:
    files = list(eval_dir.glob('forecast_audit_*.json'))
    with_eval = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
            em = (d.get('artifacts') or d).get('evaluation_metrics')
            if em and isinstance(em, dict) and em.get('ensemble_rmse') is not None:
                with_eval.append(f.name)
        except Exception:
            pass
    print(f'production_eval/ files total:    {len(files)}')
    print(f'  with evaluation_metrics+RMSE:  {len(with_eval)}')
"@

Write-Host ""
Write-Host "--- RMSE gate check ---" -ForegroundColor Cyan
& $PYTHON "scripts\check_forecast_audits.py" `
    --audit-dir   "logs\forecast_audits\production" `
    --config-path "config\forecaster_monitoring.yml"

Write-Host ""
Write-Host "--- Full gate state ---" -ForegroundColor Cyan
& $PYTHON "scripts\run_all_gates.py" --json

Write-Host ""
Write-Host "Priority 2 (THIN_LINKAGE) reminder:" -ForegroundColor Yellow
Write-Host "  Run live auto_trader during NYSE hours (09:30-16:00 ET) to accumulate matched >= 10:"
Write-Host "  & '$PYTHON' scripts\run_auto_trader.py --tickers AAPL,MSFT,NVDA --cycles 3 --resume"
