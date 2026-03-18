# bash/train_directional_classifier.ps1
# Phase 9: Build directional training dataset and train the classifier.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File bash\train_directional_classifier.ps1
#   powershell -ExecutionPolicy Bypass -File bash\train_directional_classifier.ps1 -FallbackToPnlLabel
#   powershell -ExecutionPolicy Bypass -File bash\train_directional_classifier.ps1 -SkipBuild
#
# Exit codes: 0 = model trained, 2 = cold start, 1 = hard error

param(
    [switch]$FallbackToPnlLabel,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Continue"

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PythonBin = Join-Path $RootDir "simpleTrader_env\Scripts\python.exe"
$env:PYTHONPATH = $RootDir

if (-not (Test-Path $PythonBin)) {
    Write-Host "[ERROR] Virtual environment Python not found at $PythonBin" -ForegroundColor Red
    Write-Host "        Run: simpleTrader_env\Scripts\activate" -ForegroundColor Red
    exit 1
}

function Log($msg) { Write-Host "$(Get-Date -Format 'HH:mm:ss') $msg" }

Set-Location $RootDir

Log "=== Phase 9: Directional Classifier Training ==="
Log "    Root  : $RootDir"
Log "    Python: $PythonBin"
Log ""

# ---------------------------------------------------------------------------
# Step 1: Build dataset
# ---------------------------------------------------------------------------
if ($SkipBuild) {
    Log "--- Step 1: Skipped (-SkipBuild)"
} else {
    Log "--- Step 1: Build training dataset"
    $buildArgs = @("scripts\build_directional_training_data.py")
    if ($FallbackToPnlLabel) { $buildArgs += "--fallback-to-pnl-label" }

    & $PythonBin @buildArgs
    $buildRc = $LASTEXITCODE

    if ($buildRc -eq 1) {
        Log "[ERROR] Dataset build failed."
        Log "        Check that logs\signals\quant_validation.jsonl exists."
        exit 1
    }

    # Print summary
    $summaryPath = Join-Path $RootDir "logs\directional_training_latest.json"
    if (Test-Path $summaryPath) {
        Log ""
        Log "  Dataset summary:"
        & $PythonBin -c @"
import json
s = json.loads(open(r'$summaryPath', encoding='utf-8').read())
print(f'    n_labeled   : {s.get(\"n_labeled\", 0)}')
print(f'    n_positive  : {s.get(\"n_positive\", 0)}')
print(f'    n_negative  : {s.get(\"n_negative\", 0)}')
print(f'    win_rate    : {s.get(\"win_rate\")}')
print(f'    cold_start  : {s.get(\"cold_start\")}')
if s.get('cold_start_reason'):
    print(f'    reason      : {s.get(\"cold_start_reason\")}')
"@
        # Check cold start
        $coldStart = & $PythonBin -c @"
import json
s = json.loads(open(r'$summaryPath', encoding='utf-8').read())
print('1' if s.get('cold_start') else '0')
"@
        if ($coldStart.Trim() -eq "1") {
            Log ""
            Log "[COLD_START] Dataset cold start -- skipping training." -ForegroundColor Yellow
            Log ""
            Log "  Need >= 60 labeled examples with class balance >= 10 per class."
            Log "  Current count: logs\directional_training_latest.json"
            Log ""
            Log "  Next steps:"
            Log "    - Run the pipeline to accumulate signals with classifier_features in JSONL"
            Log "    - Re-run this script once sufficient data is available"
            Log "    - Use -FallbackToPnlLabel to use PnL win/loss labels as a stopgap"
            exit 2
        }
    }
    Log ""
}

# ---------------------------------------------------------------------------
# Step 2: Train
# ---------------------------------------------------------------------------
Log "--- Step 2: Train directional classifier"
& $PythonBin "scripts\train_directional_classifier.py"
$trainRc = $LASTEXITCODE
Log ""

switch ($trainRc) {
    0 {
        Log "[OK] Classifier trained and saved." -ForegroundColor Green
        Log ""
        $metaPath = Join-Path $RootDir "data\classifiers\directional_v1.meta.json"
        if (Test-Path $metaPath) {
            & $PythonBin -c @"
import json
m = json.loads(open(r'$metaPath', encoding='utf-8').read())
print(f'  n_train         : {m.get(\"n_train\")}')
print(f'  walk_forward_DA : {m.get(\"walk_forward_da\")}')
print(f'  best_C          : {m.get(\"best_c\")}')
print('  top features:')
for f in (m.get('top3_features') or []):
    print(f'    {f[\"name\"]}: {f[\"coef\"]:+.4f}')
da = m.get('walk_forward_da', 0)
n  = m.get('n_train', 0)
print()
ready = da > 0.50 and n >= 60
print('[GATE] Activation ready:', 'YES' if ready else 'NO (DA <= 50% or n < 60)')
print('       To activate: set directional_classifier.enabled: true')
print('       in config/signal_routing_config.yml')
"@
        }

        # Step 3: Evaluate (walk-forward DA, ECE, counterfactual)
        Log ""
        Log "--- Step 3: Evaluate directional classifier"
        & $PythonBin "scripts\evaluate_directional_classifier.py"
        $evalRc = $LASTEXITCODE
        switch ($evalRc) {
            0 {
                Log ""
                Log "[OK] Evaluation complete. Report: visualizations\directional_eval.txt" -ForegroundColor Green
            }
            2 { Log "[INFO] Evaluation skipped (cold start -- too few labeled examples)" -ForegroundColor Yellow }
            default { Log "[WARN] Evaluation returned exit $evalRc (non-blocking)" -ForegroundColor Yellow }
        }
        Log ""
        exit 0
    }
    2 {
        Log "[COLD_START] Not enough labeled examples -- model NOT saved." -ForegroundColor Yellow
        Log ""
        Log "  Need >= 60 labeled examples with class balance >= 10 per class."
        Log "  Current count: logs\directional_training_latest.json"
        Log ""
        Log "  Next steps:"
        Log "    - Run the pipeline to accumulate signals with classifier_features in JSONL"
        Log "    - Re-run this script once sufficient data is available"
        Log "    - Use -FallbackToPnlLabel to use PnL win/loss labels as a stopgap"
        exit 2
    }
    default {
        Log "[ERROR] Training failed (exit $trainRc)." -ForegroundColor Red
        exit 1
    }
}
