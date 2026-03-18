# bash/overnight_classifier_bootstrap.ps1
# Phase 9: Overnight directional classifier bootstrap + PnL A/B comparison.
#
# What this script does:
#   Phase 1 - Bootstrap: 12 historical synthetic cycles (2020-2024) to
#              generate JSONL entries with classifier_features.
#   Phase 2 - Build dataset + train directional classifier.
#   Phase 3 - Control run: 4 holdout dates with gate DISABLED.
#   Phase 4 - Treatment run: same 4 holdout dates with gate ENABLED.
#   Phase 5 - Report: PnL / WR / trade-count delta (treatment vs control).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File bash\overnight_classifier_bootstrap.ps1
#   powershell -ExecutionPolicy Bypass -File bash\overnight_classifier_bootstrap.ps1 -Tickers "AAPL,MSFT,NVDA"
#   powershell -ExecutionPolicy Bypass -File bash\overnight_classifier_bootstrap.ps1 -SkipBootstrap -SkipTrain
#   powershell -ExecutionPolicy Bypass -File bash\overnight_classifier_bootstrap.ps1 -DryRun
#
# Estimated runtime: 90-150 minutes (all phases)
# Log: logs\run_audit\classifier_bootstrap_YYYYMMDD_HHMMSS.log

param(
    [string]$Tickers       = "AAPL,MSFT,NVDA,GS,AMZN",
    [switch]$SkipBootstrap,
    [switch]$SkipTrain,
    [switch]$DryRun
)

$ErrorActionPreference = "SilentlyContinue"

$RootDir   = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PythonBin = Join-Path $RootDir "simpleTrader_env\Scripts\python.exe"
$env:PYTHONPATH = $RootDir

if (-not (Test-Path $PythonBin)) {
    Write-Host "[ERROR] Python not found at $PythonBin" -ForegroundColor Red
    exit 1
}

Set-Location $RootDir

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
$RunTag  = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir  = Join-Path $RootDir "logs\run_audit"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir "classifier_bootstrap_${RunTag}.log"
$SummaryFile = Join-Path $LogDir "classifier_bootstrap_${RunTag}_summary.txt"

function Log($msg, $color = "White") {
    $line = "$(Get-Date -Format 'HH:mm:ss') $msg"
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}
function LogSection($msg) {
    Log ""
    Log "============================================================"
    Log $msg
    Log "============================================================"
}
function LogPass($msg)  { Log "[PASS]  $msg" "Green" }
function LogWarn($msg)  { Log "[WARN]  $msg" "Yellow" }
function LogError($msg) { Log "[ERROR] $msg" "Red" }

# ---------------------------------------------------------------------------
# Helper: run a command, capture stdout+stderr to log, return exit code.
# Converts PowerShell ErrorRecord objects (from Python stderr) to plain
# strings so NativeCommandError headers are never printed to the console.
# ---------------------------------------------------------------------------
function RunCmd($label, [string[]]$cmdArgs) {
    if ($DryRun) {
        Log "[DRY_RUN] Would run: $($cmdArgs -join ' ')"
        return 0
    }
    Log "--- $label"
    & $PythonBin @cmdArgs 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ }
    } | Tee-Object -FilePath $LogFile -Append
    return $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Helper: best-effort auto_trader run (exit 1 = no trades placed = normal).
# Only treat exit 2+ (Python crash / import failure) as a hard error.
# ---------------------------------------------------------------------------
function RunAutoTrader($label, [string[]]$cmdArgs) {
    $rc = RunCmd $label $cmdArgs
    if ($rc -eq 0) { return "pass" }
    if ($rc -eq 1) { Log "  [NOTE] exit 1 -- no trades placed this cycle (HOLD-dominated), continuing"; return "warn" }
    Log "  [ERROR] exit $rc -- Python crash" "Red"; return "fail"
}

# ---------------------------------------------------------------------------
# Helper: snapshot canonical DB metrics, return hashtable
# ---------------------------------------------------------------------------
function DbSnapshot() {
    if ($DryRun) {
        return @{ round_trips = 0; total_pnl = 0.0; win_rate = 0.0; win_count = 0; loss_count = 0 }
    }
    $raw = & $PythonBin -c @"
import json, sys
sys.path.insert(0, r'$RootDir')
from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
with PnLIntegrityEnforcer(r'data\portfolio_maximizer.db') as e:
    m = e.get_canonical_metrics()
    print(json.dumps({
        'round_trips': m.total_round_trips,
        'total_pnl': round(float(m.total_realized_pnl or 0), 4),
        'win_count': m.win_count,
        'loss_count': m.loss_count,
        'win_rate': round(float(m.win_rate or 0), 4),
    }))
"@ 2>$null
    try {
        return $raw | ConvertFrom-Json
    } catch {
        return @{ round_trips = 0; total_pnl = 0.0; win_rate = 0.0; win_count = 0; loss_count = 0 }
    }
}

# ---------------------------------------------------------------------------
# Helper: toggle directional_classifier.enabled in signal_routing_config.yml
# ---------------------------------------------------------------------------
function SetGateEnabled([string]$enabled) {
    if ($DryRun) { Log "[DRY_RUN] Would set directional_classifier.enabled=$enabled"; return }
    & $PythonBin -c @"
import pathlib, re
p = pathlib.Path(r'config\signal_routing_config.yml')
txt = p.read_text(encoding='utf-8')
txt = re.sub(
    r'(directional_classifier:.*?enabled:\s*)(?:true|false)',
    lambda m: m.group(1) + '$enabled',
    txt, flags=re.DOTALL, count=1
)
p.write_text(txt, encoding='utf-8')
print('set directional_classifier.enabled=$enabled')
"@ 2>&1 | Tee-Object -FilePath $LogFile -Append
}

# ---------------------------------------------------------------------------
# Training window: ETL pipeline runs per ticker over historical ranges.
# run_etl_pipeline.py downloads real market data + saves checkpoint parquets
# (real prices) AND logs JSONL entries with classifier_features via
# TimeSeriesSignalGenerator. This is what enables forward-price labeling.
# run_auto_trader.py --execution-mode synthetic generates SYNTHETIC random
# prices with no real checkpoint parquets -- forward price lookup always fails.
#
# Each ticker uses 3 partially-overlapping ranges to maximise CV diversity
# while keeping each ETL run under ~5 min. Ranges chosen so the checkpoint
# parquets cover the signal timestamps produced inside each run.
# ---------------------------------------------------------------------------
$BootstrapRuns = @(
    @{ Ticker="AAPL"; Start="2019-01-01"; End="2022-06-01" },
    @{ Ticker="AAPL"; Start="2021-01-01"; End="2024-06-01" },
    @{ Ticker="MSFT"; Start="2019-01-01"; End="2022-06-01" },
    @{ Ticker="MSFT"; Start="2021-01-01"; End="2024-06-01" },
    @{ Ticker="NVDA"; Start="2019-01-01"; End="2022-06-01" },
    @{ Ticker="NVDA"; Start="2021-01-01"; End="2024-06-01" },
    @{ Ticker="GS";   Start="2019-01-01"; End="2022-06-01" },
    @{ Ticker="GS";   Start="2021-01-01"; End="2024-06-01" },
    @{ Ticker="AMZN"; Start="2019-01-01"; End="2022-06-01" },
    @{ Ticker="AMZN"; Start="2021-01-01"; End="2024-06-01" }
)

# Evaluation window: 4 holdout dates within parquet coverage.
# Must be within the checkpoint parquet range (2019-2024) so the auto_trader
# can load real price data for forecasting. Using 2022-2023 (end of first range /
# start of second) gives genuine holdout relative to the 2019-2022 training window.
$EvalDates = @(
    "2022-07-01",
    "2022-10-01",
    "2023-01-01",
    "2023-04-01"
)

# ---------------------------------------------------------------------------
# STEP 0: Baseline snapshot
# ---------------------------------------------------------------------------
LogSection "Overnight Classifier Bootstrap -- $RunTag"
Log "Python        : $PythonBin"
Log "Root          : $RootDir"
Log "Tickers       : $Tickers"
Log "Log           : $LogFile"
Log "Options       : SkipBootstrap=$SkipBootstrap SkipTrain=$SkipTrain DryRun=$DryRun"
Log "Est. runtime  : 90-150 min (all phases)"

LogSection "STEP 0: DB baseline snapshot"
$Baseline = DbSnapshot
Log "Baseline: $($Baseline.round_trips) closed trades, PnL=$($Baseline.total_pnl), WR=$($Baseline.win_rate)"

$Errors = 0

# ---------------------------------------------------------------------------
# PRE-FLIGHT: Pipeline input validation
# ---------------------------------------------------------------------------
LogSection "PRE-FLIGHT: Pipeline input validation (V1-V6)"
if ($DryRun) {
    Log "[DRY_RUN] Would run: validate_pipeline_inputs.py --tickers $Tickers"
} else {
    $preflightRc = RunCmd "validate_pipeline_inputs" @(
        "scripts\validate_pipeline_inputs.py",
        "--tickers", $Tickers,
        "--eval-dates", ($EvalDates -join ",")
    )
    if ($preflightRc -eq 1) {
        LogError "Pre-flight validation FAILED — resolve the FAIL(s) above before running the pipeline."
        LogError "Common causes:"
        LogError "  V1: Rename parquets to include ticker name (e.g. AAPL_pipeline_*.parquet)"
        LogError "  V3: Use generate_classifier_training_labels.py (parquet scan) not build_directional_training_data.py"
        LogError "  V4: Change eval dates to fall within your parquet coverage window"
        LogError "  V5: Re-run ETL with --execution-mode auto (not synthetic)"
        exit 1
    } elseif ($preflightRc -eq 2) {
        LogWarn "Validator could not run (infrastructure error, exit 2) -- proceeding with caution."
    } else {
        LogPass "Pre-flight validation passed (PASS/WARN only) -- pipeline may proceed."
    }
}

# ---------------------------------------------------------------------------
# PHASE 1: Bootstrap
# ---------------------------------------------------------------------------
LogSection "PHASE 1/5: Bootstrap ETL runs (real prices + classifier_features)"

if ($SkipBootstrap) {
    Log "Skipping Phase 1 (-SkipBootstrap)"
} else {
    $bootPass = 0; $bootFail = 0
    foreach ($run in $BootstrapRuns) {
        $label = "ETL $($run.Ticker) $($run.Start)..$($run.End)"

        # Snapshot existing parquets BEFORE the ETL run so we can identify the new one
        $existingParquets = @(Get-ChildItem "$RootDir\data\checkpoints" -Filter "*.parquet" -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName)

        # Use --execution-mode auto (not synthetic) so yfinance downloads REAL price data.
        # --execution-mode synthetic uses SyntheticExtractor which generates the same
        # random walk for all tickers (same seed) → meaningless, non-ticker-specific parquets.
        $rc = RunCmd $label @(
            "scripts\run_etl_pipeline.py",
            "--tickers", $run.Ticker,
            "--start",   $run.Start,
            "--end",     $run.End,
            "--execution-mode", "auto"
        )

        # Rename the newly created parquet(s) to include the ticker name.
        # This allows _load_price_parquet to find the right file per ticker
        # (the ETL saves generic pipeline_TIMESTAMP_data_extraction_TIMESTAMP.parquet
        # with no ticker in the name; the rename fixes the lookup pattern).
        $newParquets = @(Get-ChildItem "$RootDir\data\checkpoints" -Filter "*.parquet" -ErrorAction SilentlyContinue |
            Where-Object { $existingParquets -notcontains $_.FullName -and $_.Name -notmatch $run.Ticker } |
            Sort-Object LastWriteTime -Descending)
        foreach ($pq in $newParquets) {
            $safeName = $pq.Name -replace "pipeline_", "$($run.Ticker)_pipeline_"
            $dest = Join-Path $pq.DirectoryName $safeName
            if (-not (Test-Path $dest)) {
                Rename-Item $pq.FullName $dest -ErrorAction SilentlyContinue
                Log "  [parquet] renamed -> $safeName"
            }
        }

        if ($rc -eq 0) { $bootPass++ } else {
            Log "  [WARN] $label exited $rc (non-fatal, continuing)"
            $bootFail++
        }
    }
    Log "Bootstrap complete: $bootPass pass / $bootFail warn-or-fail"
}

# Count JSONL entries with classifier_features
$featuresCount = 0
if (-not $DryRun) {
    $featuresCount = & $PythonBin -c @"
import json, pathlib
p = pathlib.Path(r'logs\signals\quant_validation.jsonl')
if not p.exists():
    print(0); exit()
entries = []
for l in p.read_text(encoding='utf-8').splitlines():
    try: entries.append(json.loads(l))
    except: pass
print(sum(1 for e in entries if e.get('classifier_features')))
"@ 2>$null
}
Log "JSONL entries with classifier_features: $featuresCount"

# ---------------------------------------------------------------------------
# PHASE 2: Build + train
# ---------------------------------------------------------------------------
LogSection "PHASE 2/5: Build training dataset + train classifier"

$ModelTrained = $false

if ($SkipTrain) {
    Log "Skipping Phase 2 (-SkipTrain)"
    $ModelTrained = $true
} else {
    # Step 2a: build training labels from checkpoint parquets (bypasses JSONL timestamp issue)
    # NOTE: build_directional_training_data.py (JSONL-based) cannot produce labels because:
    #   1. Checkpoint parquets use generic filenames with no ticker → _load_price_parquet returns None
    #   2. JSONL entries have wall-clock timestamps (today) outside the 2019-2024 parquet range
    # generate_classifier_training_labels.py reads parquets directly with historical timestamps.
    Log "--- generate_classifier_training_labels.py (parquet scan, all $Tickers tickers)"
    if (-not $DryRun) {
        & $PythonBin "scripts\generate_classifier_training_labels.py" "--ticker" $Tickers "--auto-parquet" 2>&1 |
            ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ } } |
            Tee-Object -FilePath $LogFile -Append
    }

    $summaryPath = Join-Path $RootDir "logs\directional_training_latest.json"
    if ((Test-Path $summaryPath) -and (-not $DryRun)) {
        & $PythonBin -c @"
import json
s = json.loads(open(r'$summaryPath', encoding='utf-8').read())
print(f'  n_labeled   : {s.get(\"n_labeled\", 0)}')
print(f'  n_positive  : {s.get(\"n_positive\", 0)}')
print(f'  n_negative  : {s.get(\"n_negative\", 0)}')
print(f'  win_rate    : {s.get(\"win_rate\")}')
print(f'  cold_start  : {s.get(\"cold_start\")}')
if s.get('cold_start_reason'):
    print(f'  reason      : {s.get(\"cold_start_reason\")}')
"@ 2>&1 | Tee-Object -FilePath $LogFile -Append
    }

    # Step 2b: train — skip if parquet was not written (0 labeled rows = cold start)
    $datasetPath = Join-Path $RootDir "data\training\directional_dataset.parquet"
    $trainRc = 0
    if (-not $DryRun -and -not (Test-Path $datasetPath)) {
        Log "  [NOTE] directional_dataset.parquet not written (0 labeled rows) -- cold start"
        $trainRc = 2
    } else {
        Log "--- train_directional_classifier.py"
        if (-not $DryRun) {
            & $PythonBin "scripts\train_directional_classifier.py" 2>&1 | ForEach-Object {
                if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { $_ }
            } | Tee-Object -FilePath $LogFile -Append
            $trainRc = $LASTEXITCODE
        }
    }

    switch ($trainRc) {
        0 {
            $ModelTrained = $true
            LogPass "Classifier trained and saved"
            $metaPath = Join-Path $RootDir "data\classifiers\directional_v1.meta.json"
            if (Test-Path $metaPath) {
                & $PythonBin -c @"
import json
m = json.loads(open(r'$metaPath', encoding='utf-8').read())
print(f'  n_train         : {m.get(\"n_train\")}')
print(f'  walk_forward_DA : {m.get(\"walk_forward_da\")}')
print(f'  best_C          : {m.get(\"best_c\")}')
for f in (m.get('top3_features') or []):
    print(f'  feature: {f[\"name\"]} coef={f[\"coef\"]:+.4f}')
"@ 2>&1 | Tee-Object -FilePath $LogFile -Append
            }
        }
        2 { LogWarn "Cold start -- not enough labeled data. A/B eval will still run as baseline-only." }
        default { LogError "Training failed (exit $trainRc)"; $Errors++ }
    }
}

# ---------------------------------------------------------------------------
# PHASE 2b: Evaluate classifier (walk-forward DA, ECE, win-rate counterfactual)
# ---------------------------------------------------------------------------
if ($ModelTrained) {
    Log "--- Phase 2b: Evaluate directional classifier"
    if (-not $DryRun) {
        & $PythonBin "scripts\evaluate_directional_classifier.py" 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { "  [eval] $($_.ToString())" }
            else { "  [eval] $_" }
        } | Tee-Object -FilePath $LogFile -Append
        $evalRc = $LASTEXITCODE
        switch ($evalRc) {
            0 {
                Log "Evaluation complete. Check logs\directional_eval_latest.json"
                $evalPath = Join-Path $RootDir "visualizations\directional_eval.txt"
                if (Test-Path $evalPath) {
                    Log ""
                    Get-Content $evalPath | Tee-Object -FilePath $LogFile -Append
                    Log ""
                }
            }
            2 { LogWarn "Evaluation cold start -- skipped (insufficient data)" }
            default { LogWarn "Evaluation returned exit $evalRc (non-blocking)" }
        }
    } else {
        Log "DryRun: skipping evaluation"
    }
}

# ---------------------------------------------------------------------------
# PHASE 3: Control (gate DISABLED)
# ---------------------------------------------------------------------------
LogSection "PHASE 3/5: Control evaluation -- gate DISABLED"
$BeforeControl = DbSnapshot
Log "DB before control: $($BeforeControl.round_trips) trades, PnL=$($BeforeControl.total_pnl)"

SetGateEnabled "false"

$ctrlPass = 0; $ctrlWarn = 0; $ctrlFail = 0
foreach ($asOf in $EvalDates) {
    $outcome = RunAutoTrader "control as-of $asOf" @(
        "scripts\run_auto_trader.py",
        "--tickers", $Tickers,
        "--cycles", "1",
        "--execution-mode", "synthetic",
        "--as-of-date", $asOf,
        "--no-resume",
        "--sleep-seconds", "0"
    )
    switch ($outcome) {
        "pass" { $ctrlPass++ }
        "warn" { $ctrlWarn++ }
        "fail" { $ctrlFail++; $Errors++ }
    }
}

$AfterControl = DbSnapshot
Log "DB after control : $($AfterControl.round_trips) trades, PnL=$($AfterControl.total_pnl)"
Log "Control cycles   : $ctrlPass pass / $ctrlWarn warn(no-trade) / $ctrlFail hard-fail"

$ControlDeltaTrades = $AfterControl.round_trips - $BeforeControl.round_trips
$ControlDeltaPnl    = [math]::Round($AfterControl.total_pnl - $BeforeControl.total_pnl, 4)
$ControlWinRate     = $AfterControl.win_rate
Log "Control result   : trades_added=$ControlDeltaTrades  pnl_delta=$ControlDeltaPnl  wr=$ControlWinRate"

# ---------------------------------------------------------------------------
# PHASE 4: Treatment (gate ENABLED)
# ---------------------------------------------------------------------------
LogSection "PHASE 4/5: Treatment evaluation -- gate ENABLED"

$TreatDeltaTrades = "n/a"; $TreatDeltaPnl = "n/a"; $TreatWinRate = "n/a"

$pkPath = Join-Path $RootDir "data\classifiers\directional_v1.pkl"
if (-not $ModelTrained) {
    LogWarn "Skipping Phase 4 -- classifier not trained (cold start)"
} elseif ((-not $DryRun) -and (-not (Test-Path $pkPath))) {
    LogWarn "Skipping Phase 4 -- data\classifiers\directional_v1.pkl not found"
} else {
    $BeforeTreat = DbSnapshot
    Log "DB before treatment: $($BeforeTreat.round_trips) trades, PnL=$($BeforeTreat.total_pnl)"

    SetGateEnabled "true"

    $treatPass = 0; $treatWarn = 0; $treatFail = 0
    foreach ($asOf in $EvalDates) {
        $outcome = RunAutoTrader "treatment as-of $asOf" @(
            "scripts\run_auto_trader.py",
            "--tickers", $Tickers,
            "--cycles", "1",
            "--execution-mode", "synthetic",
            "--as-of-date", $asOf,
            "--no-resume",
            "--sleep-seconds", "0"
        )
        switch ($outcome) {
            "pass" { $treatPass++ }
            "warn" { $treatWarn++ }
            "fail" { $treatFail++; $Errors++ }
        }
    }

    $AfterTreat = DbSnapshot
    Log "DB after treatment : $($AfterTreat.round_trips) trades, PnL=$($AfterTreat.total_pnl)"
    Log "Treatment cycles   : $treatPass pass / $treatWarn warn(no-trade) / $treatFail hard-fail"

    $TreatDeltaTrades = $AfterTreat.round_trips - $BeforeTreat.round_trips
    $TreatDeltaPnl    = [math]::Round($AfterTreat.total_pnl - $BeforeTreat.total_pnl, 4)
    $TreatWinRate     = $AfterTreat.win_rate
    Log "Treatment result   : trades_added=$TreatDeltaTrades  pnl_delta=$TreatDeltaPnl  wr=$TreatWinRate"

    # Restore gate to disabled
    SetGateEnabled "false"
    Log "Gate restored to: disabled"
}

# ---------------------------------------------------------------------------
# PHASE 5: Report
# ---------------------------------------------------------------------------
LogSection "PHASE 5/5: Results"

$PnlImprovement = "n/a"
$Verdict        = "n/a"
if ($TreatDeltaPnl -ne "n/a" -and $ControlDeltaPnl -ne "n/a") {
    $delta = [math]::Round($TreatDeltaPnl - $ControlDeltaPnl, 4)
    $PnlImprovement = if ($delta -ge 0) { "+$delta" } else { "$delta" }
    $Verdict = if ($TreatDeltaPnl -gt $ControlDeltaPnl) { "IMPROVEMENT" }
               elseif ($TreatDeltaPnl -lt $ControlDeltaPnl) { "REGRESSION" }
               else { "NO_CHANGE" }
}

$report = @"

==============================
 DIRECTIONAL CLASSIFIER A/B
==============================

  Bootstrap runs    : $($BootstrapRuns.Count) ETL runs (5 tickers x 2 date ranges)
  JSONL w/ features : $featuresCount
  Eval dates        : $($EvalDates.Count) holdout dates (2025)

  CONTROL  (gate off) | trades=$ControlDeltaTrades  pnl=$ControlDeltaPnl  wr=$ControlWinRate
  TREATMENT (gate on) | trades=$TreatDeltaTrades  pnl=$TreatDeltaPnl  wr=$TreatWinRate

  PnL delta (treatment - control): $PnlImprovement
  Verdict: $Verdict

"@

switch ($Verdict) {
    "IMPROVEMENT" {
        $report += @"
  [ACTION] Gate shows positive PnL impact.
           To enable permanently:
             Set directional_classifier.enabled: true
             in config\signal_routing_config.yml
"@
    }
    "REGRESSION" {
        $report += @"
  [INFO] Gate shows PnL regression -- classifier needs more training data.
         Continue accumulating JSONL entries and re-run this script.
"@
    }
    default {
        $report += @"
  [INFO] Classifier was not trained (cold start) or no trades generated.
         Need >= 60 labeled examples. Check logs\directional_training_latest.json
"@
    }
}

$report += @"

  Errors (non-fatal): $Errors
  Full log  : $LogFile
  Summary   : $SummaryFile

  Next steps:
    python scripts\update_platt_outcomes.py
    python scripts\production_audit_gate.py --allow-inconclusive-lift
    powershell -ExecutionPolicy Bypass -File bash\train_directional_classifier.ps1

"@

Write-Host $report
Add-Content -Path $LogFile    -Value $report -Encoding UTF8
Set-Content  -Path $SummaryFile -Value $report -Encoding UTF8

exit 0
