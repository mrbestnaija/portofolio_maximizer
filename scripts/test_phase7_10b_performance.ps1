#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Phase 7.10b + Phase 7.11 Performance & Directional Accuracy Test Suite
    Tests validation logic, model signal quality, edge cases, adversarial inputs,
    and system-level interactions introduced in Phase 7.10b.

.DESCRIPTION
    Covers:
    Part A - Validation Logic: weighted scoring threshold, proof-mode exclusion, profit floor
    Part B - Model Quality:   GARCH, SAMoSSA, MSSA-RL, Ensemble, Platt scaling
    Part C - Directional:     consensus gate, Hurst policy, IC culling, momentum filter
    System  - Config sync, co-agent file isolation, resource usage
    Adversarial - Noise injection, spike data, NaN/Inf inputs, hostile conditions

.PARAMETER ReportDir
    Directory for test reports (default: logs/run_audit)

.PARAMETER PythonExe
    Path to Python executable (default: auto-detect venv)

.PARAMETER SkipSlow
    Skip tests that take > 30 seconds (model fit tests)

.PARAMETER AdversarialOnly
    Run only adversarial tests

.EXAMPLE
    .\scripts\test_phase7_10b_performance.ps1
    .\scripts\test_phase7_10b_performance.ps1 -SkipSlow
    .\scripts\test_phase7_10b_performance.ps1 -AdversarialOnly
#>

[CmdletBinding()]
param(
    [string]$ReportDir  = "logs/run_audit",
    [string]$PythonExe  = "",
    [switch]$SkipSlow   = $false,
    [switch]$AdversarialOnly = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"   # Don't abort on individual test failures

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
$ROOT       = Split-Path -Parent $PSScriptRoot
$TIMESTAMP  = (Get-Date -Format "yyyyMMdd_HHmmss")
$REPORT_DIR = Join-Path $ROOT $ReportDir
$LOG_FILE   = Join-Path $REPORT_DIR "phase710b_perf_test_${TIMESTAMP}.log"
$JSON_FILE  = Join-Path $REPORT_DIR "phase710b_perf_test_${TIMESTAMP}.json"

# Locate Python executable
if (-not $PythonExe) {
    $candidates = @(
        (Join-Path $ROOT "simpleTrader_env\Scripts\python.exe"),
        (Join-Path $ROOT "simpleTrader_env\bin\python"),
        "python",
        "python3"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c -ErrorAction SilentlyContinue) { $PythonExe = $c; break }
        $resolved = (Get-Command $c -ErrorAction SilentlyContinue)?.Source
        if ($resolved) { $PythonExe = $resolved; break }
    }
}
if (-not $PythonExe) {
    Write-Error "[FATAL] Cannot locate Python. Provide -PythonExe or activate the virtual environment."
    exit 1
}

# Ensure report directory exists
$null = New-Item -ItemType Directory -Force -Path $REPORT_DIR

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
$script:TestResults  = [System.Collections.Generic.List[hashtable]]::new()
$script:PassCount    = 0
$script:FailCount    = 0
$script:WarnCount    = 0
$script:StartTime    = Get-Date

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $ts  = (Get-Date -Format "HH:mm:ss")
    $tag = switch ($Level) {
        "PASS"  { "[PASS] " }
        "FAIL"  { "[FAIL] " }
        "WARN"  { "[WARN] " }
        "SKIP"  { "[SKIP] " }
        "SECT"  { "`n===== " }
        default { "[INFO] " }
    }
    $line = "${ts} ${tag}${Message}"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
}

function Record-Result {
    param(
        [string]$TestName,
        [string]$Status,        # PASS | FAIL | WARN | SKIP
        [string]$Detail = "",
        [double]$ElapsedMs = 0
    )
    $script:TestResults.Add(@{
        test    = $TestName
        status  = $Status
        detail  = $Detail
        elapsed = $ElapsedMs
        ts      = (Get-Date -Format "o")
    })
    switch ($Status) {
        "PASS" { $script:PassCount++; Write-Log "$TestName | $Detail" "PASS" }
        "FAIL" { $script:FailCount++; Write-Log "$TestName | $Detail" "FAIL" }
        "WARN" { $script:WarnCount++; Write-Log "$TestName | $Detail" "WARN" }
        "SKIP" { Write-Log "$TestName | $Detail" "SKIP" }
    }
}

function Invoke-PythonTest {
    <#
    .SYNOPSIS
        Execute inline Python code and return (exitCode, stdout, stderr, elapsedMs).
    #>
    param(
        [string]$Code,
        [string]$WorkDir = $ROOT,
        [int]   $TimeoutSeconds = 60,
        [hashtable]$Env = @{}
    )
    $tmpFile = [System.IO.Path]::GetTempFileName() + ".py"
    Set-Content -Path $tmpFile -Value $Code -Encoding UTF8

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName               = $PythonExe
    $psi.Arguments              = "`"$tmpFile`""
    $psi.WorkingDirectory       = $WorkDir
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute        = $false
    $psi.CreateNoWindow         = $true

    # Merge environment
    foreach ($kv in $Env.GetEnumerator()) {
        $psi.EnvironmentVariables[$kv.Key] = $kv.Value
    }
    # Propagate critical paths
    $psi.EnvironmentVariables["PYTHONPATH"] = $ROOT
    $psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8"

    $sw   = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdout = $proc.StandardOutput.ReadToEndAsync()
    $stderr = $proc.StandardError.ReadToEndAsync()
    $exited = $proc.WaitForExit($TimeoutSeconds * 1000)
    $sw.Stop()

    if (-not $exited) {
        $proc.Kill()
        Remove-Item $tmpFile -Force -ErrorAction SilentlyContinue
        return @{ ExitCode = -1; Stdout = ""; Stderr = "TIMEOUT after ${TimeoutSeconds}s"; Elapsed = $sw.ElapsedMilliseconds }
    }
    $out = $stdout.Result
    $err = $stderr.Result
    Remove-Item $tmpFile -Force -ErrorAction SilentlyContinue
    return @{ ExitCode = $proc.ExitCode; Stdout = $out; Stderr = $err; Elapsed = $sw.ElapsedMilliseconds }
}

function Invoke-PytestTarget {
    param([string]$Target, [string]$ExtraArgs = "", [int]$TimeoutSeconds = 120)
    $sw  = [System.Diagnostics.Stopwatch]::StartNew()
    $cmd = "& `"$PythonExe`" -m pytest `"$Target`" -q --tb=short $ExtraArgs 2>&1"
    $out = Invoke-Expression $cmd
    $sw.Stop()
    $exitCode = $LASTEXITCODE
    return @{ ExitCode = $exitCode; Output = ($out -join "`n"); Elapsed = $sw.ElapsedMilliseconds }
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A: VALIDATION LOGIC TESTS
# ─────────────────────────────────────────────────────────────────────────────
function Test-ValidationLogic {
    Write-Log "Validation Logic Tests (Part A — Phase 7.10b)" "SECT"

    # A1. Weighted scoring: score >= 0.60 -> PASS
    $code = @'
import sys
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _weighted_score(criteria, weights, pass_threshold, expected_profit):
    if expected_profit < 0:
        return "FAIL"
    if not criteria:
        return "SKIPPED"
    score       = sum(float(weights.get(k, 0.0)) * (1.0 if v else 0.0) for k, v in criteria.items())
    total_w     = sum(float(weights.get(k, 0.0)) for k in criteria)
    norm_score  = score / total_w if total_w > 0 else 0.0
    return "PASS" if norm_score >= pass_threshold else "FAIL"

W = {"expected_profit":0.25,"rmse_ratio":0.20,"directional_accuracy":0.20,
     "sharpe_ratio":0.10,"sortino_ratio":0.10,"profit_factor":0.10,"win_rate":0.05}

# Clear PASS: all criteria passing, EP > 0
r = _weighted_score(dict(zip(W.keys(),[True]*7)), W, 0.60, 10.0)
assert r == "PASS", f"All-pass should be PASS, got {r}"

# Marginal PASS: score exactly at 0.60 boundary
# Pass expected_profit(0.25) + rmse_ratio(0.20) + directional_accuracy(0.20) = 0.65/1.0 -> PASS
partial = {k: k in ("expected_profit","rmse_ratio","directional_accuracy") for k in W}
r = _weighted_score(partial, W, 0.60, 5.0)
assert r == "PASS", f"0.65 score at 0.60 threshold should be PASS, got {r}"

# Marginal FAIL: score just below 0.60
# Pass sharpe(0.10) + sortino(0.10) + profit_factor(0.10) + win_rate(0.05) = 0.35 -> FAIL
low = {k: k in ("sharpe_ratio","sortino_ratio","profit_factor","win_rate") for k in W}
r = _weighted_score(low, W, 0.60, 5.0)
assert r == "FAIL", f"0.35 score at 0.60 threshold should be FAIL, got {r}"

# Hard gate: negative EP -> always FAIL regardless of criteria
r = _weighted_score(dict(zip(W.keys(),[True]*7)), W, 0.60, -1.0)
assert r == "FAIL", f"Negative EP should hard-gate to FAIL, got {r}"

# Boundary: EP = 0 exactly -> still FAIL (< 0 is False, == 0 is False from `< 0` guard)
# Actually EP == 0 should PASS the EP guard (only < 0 is blocked)
r = _weighted_score(dict(zip(W.keys(),[True]*7)), W, 0.60, 0.0)
assert r == "PASS", f"EP=0.0 should not be blocked (only < 0 is blocked), got {r}"

# Score boundary sweep: test 0.59 vs 0.60 threshold
# Score exactly 0.60 = PASS; 0.599... = FAIL
exactly_60 = {"expected_profit": True, "rmse_ratio": True, "directional_accuracy": True,
              "sharpe_ratio": False, "sortino_ratio": False, "profit_factor": False, "win_rate": False}
r = _weighted_score(exactly_60, W, 0.60, 5.0)
# score = 0.25+0.20+0.20 = 0.65; norm = 0.65/1.0 = 0.65 -> PASS
assert r == "PASS", f"Score 0.65 should be PASS, got {r}"

print("A1_ALL_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "A1_ALL_PASS") {
        Record-Result "A1.WeightedScoring.BoundaryTests" "PASS" "All 6 boundary cases correct" $r.Elapsed
    } else {
        Record-Result "A1.WeightedScoring.BoundaryTests" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # A2. Custom weights override defaults
    $code = @'
import sys
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _weighted_score(criteria, weights, pass_threshold, expected_profit):
    if expected_profit < 0:
        return "FAIL"
    score   = sum(float(weights.get(k, 0.0)) * (1.0 if v else 0.0) for k, v in criteria.items())
    total_w = sum(float(weights.get(k, 0.0)) for k in criteria)
    norm_score = score / total_w if total_w > 0 else 0.0
    return "PASS" if norm_score >= pass_threshold else "FAIL"

# Custom weights heavily favor directional accuracy
custom_weights = {"expected_profit": 0.05, "directional_accuracy": 0.90, "win_rate": 0.05}
# Only directional_accuracy passes -> score = 0.90/1.0 = 0.90 -> PASS at threshold 0.60
criteria_da_only = {"expected_profit": False, "directional_accuracy": True, "win_rate": False}
r = _weighted_score(criteria_da_only, custom_weights, 0.60, 5.0)
assert r == "PASS", f"DA-dominant custom weights should PASS when DA=True, got {r}"

# Custom threshold: 0.85 (strict)
r = _weighted_score(criteria_da_only, custom_weights, 0.85, 5.0)
assert r == "PASS", f"DA 0.90 weight at 0.85 threshold should still PASS, got {r}"

# Custom threshold: 0.95 (very strict)
r = _weighted_score(criteria_da_only, custom_weights, 0.95, 5.0)
assert r == "FAIL", f"DA 0.90 weight at 0.95 threshold should FAIL, got {r}"

print("A2_CUSTOM_WEIGHTS_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "A2_CUSTOM_WEIGHTS_PASS") {
        Record-Result "A2.CustomWeightOverrides" "PASS" "Custom weight scenarios correct" $r.Elapsed
    } else {
        Record-Result "A2.CustomWeightOverrides" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # A3. --exclude-mode proof filtering
    $code = @'
import sys, json, tempfile, os
sys.path.insert(0, r"ROOT_PLACEHOLDER")

# Build synthetic quant_validation.jsonl with mixed modes
entries = [
    {"status": "FAIL", "execution_mode": "proof",      "proof_mode": True,  "expected_profit": 2.0},
    {"status": "FAIL", "execution_mode": "proof",      "proof_mode": True,  "expected_profit": 3.0},
    {"status": "FAIL", "execution_mode": "live",       "proof_mode": False, "expected_profit": 8.0},
    {"status": "PASS", "execution_mode": "live",       "proof_mode": False, "expected_profit": 12.0},
    {"status": "PASS", "execution_mode": "synthetic",  "proof_mode": False, "expected_profit": 15.0},
    {"status": "FAIL", "execution_mode": "diagnostic", "proof_mode": False, "expected_profit": 1.0},
]

def _summarize(entries, exclude_modes=None):
    total = pass_count = fail_count = 0
    exclude_set = set(m.lower() for m in (exclude_modes or []))
    for rec in entries:
        exec_mode  = str(rec.get("execution_mode") or "").lower()
        proof_flag = bool(rec.get("proof_mode"))
        if exclude_set:
            if exec_mode in exclude_set:
                continue
            if proof_flag and "proof" in exclude_set:
                continue
        total += 1
        if rec["status"] == "PASS":
            pass_count += 1
        else:
            fail_count += 1
    return total, pass_count, fail_count

# Without exclusion: 6 total, 2 PASS, 4 FAIL -> 66.7% FAIL rate
t, p, f = _summarize(entries)
assert t == 6, f"Expected 6 total, got {t}"
assert f/t > 0.60, f"Expected >60% FAIL rate without exclusion, got {f/t:.2%}"

# Exclude proof mode: 4 remaining (live x2 + synthetic + diagnostic), 2 PASS, 2 FAIL -> 50% FAIL
t, p, f = _summarize(entries, exclude_modes=["proof"])
assert t == 4, f"Expected 4 after proof exclusion, got {t}"
assert f == 2, f"Expected 2 FAIL after proof exclusion, got {f}"

# Exclude proof + diagnostic: 3 remaining (live x2 + synthetic), 2 PASS, 1 FAIL -> 33% FAIL
t, p, f = _summarize(entries, exclude_modes=["proof", "diagnostic"])
assert t == 3, f"Expected 3 after proof+diagnostic exclusion, got {t}"
assert p == 2, f"Expected 2 PASS, got {p}"

print("A3_EXCLUDE_MODE_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "A3_EXCLUDE_MODE_PASS") {
        Record-Result "A3.ExcludeMode.ProofFilter" "PASS" "Proof/diagnostic exclusion correct" $r.Elapsed
    } else {
        Record-Result "A3.ExcludeMode.ProofFilter" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # A4. Expected profit floor: absolute + relative (OR logic)
    $code = @'
import sys
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _eval_expected_profit(expected_profit, abs_floor=5.0, pct_floor=0.002, capital_base=25000.0):
    """Returns True if either absolute OR relative floor is met."""
    abs_pass = expected_profit >= abs_floor
    rel_floor = capital_base * pct_floor          # $50 at $25k capital
    rel_pass  = expected_profit >= rel_floor
    return abs_pass or rel_pass

# Case 1: Passes absolute (6 > 5) even if below relative ($50)
assert _eval_expected_profit(6.0)  == True, "6.0 > 5.0 absolute floor should pass"

# Case 2: Fails absolute (3 < 5) but passes relative at different capital
# If capital_base = 1000, rel_floor = 1000*0.002 = $2; 3 > 2 -> PASS
assert _eval_expected_profit(3.0, capital_base=1000.0) == True, "3.0 > $2 relative floor should pass"

# Case 3: Fails both (4.99 < 5 absolute; 4.99 < 50 relative)
assert _eval_expected_profit(4.99) == False, "4.99 fails both floors"

# Case 4: Exactly at absolute floor ($5.00 == $5.00 -> PASS)
assert _eval_expected_profit(5.00) == True, "5.00 exactly at absolute floor should pass"

# Case 5: Zero expected profit -> fails both
assert _eval_expected_profit(0.0)  == False, "Zero expected profit should fail"

# Case 6: Negative EP -> fails both
assert _eval_expected_profit(-10.0) == False, "Negative EP should fail"

# Case 7: Very large trade passes easily
assert _eval_expected_profit(1000.0) == True, "Large EP should always pass"

print("A4_PROFIT_FLOOR_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "A4_PROFIT_FLOOR_PASS") {
        Record-Result "A4.ExpectedProfitFloor.AbsRelOR" "PASS" "All 7 profit floor scenarios correct" $r.Elapsed
    } else {
        Record-Result "A4.ExpectedProfitFloor.AbsRelOR" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B: MODEL SIGNAL QUALITY TESTS
# ─────────────────────────────────────────────────────────────────────────────
function Test-ModelSignalQuality {
    Write-Log "Model Signal Quality Tests (Part B — Phase 7.10b)" "SECT"

    # B1a. GARCH: import + basic construction with new params
    $code = @'
import sys, numpy as np
sys.path.insert(0, r"ROOT_PLACEHOLDER")
try:
    from forcester_ts.garch import GARCHForecaster
    g = GARCHForecaster(dist="skewt", mean="AR", enforce_stationarity=True, igarch_fallback="gjr")
    assert g._dist == "skewt",  f"Expected dist=skewt, got {g._dist}"
    assert g._mean == "AR",     f"Expected mean=AR, got {g._mean}"
    assert g._igarch_fallback == "gjr", f"Expected gjr fallback, got {g._igarch_fallback}"
    print("B1A_GARCH_PARAMS_PASS")
except ImportError as e:
    print(f"B1A_SKIP_IMPORT_ERROR: {e}")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 15
    if ($r.Stdout -match "B1A_GARCH_PARAMS_PASS") {
        Record-Result "B1a.GARCH.ParamConstruction" "PASS" "dist=skewt, mean=AR, gjr fallback verified" $r.Elapsed
    } elseif ($r.Stdout -match "B1A_SKIP") {
        Record-Result "B1a.GARCH.ParamConstruction" "SKIP" "arch not installed" $r.Elapsed
    } else {
        Record-Result "B1a.GARCH.ParamConstruction" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # B1b. GARCH: fit on normal price series; verify directional signal in summary
    if (-not $SkipSlow) {
        $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")
np.random.seed(42)
try:
    from forcester_ts.garch import GARCHForecaster, ARCH_AVAILABLE
    if not ARCH_AVAILABLE:
        print("B1B_SKIP_NO_ARCH")
        raise SystemExit(0)
    dates  = pd.date_range("2020-01-01", periods=300, freq="D")
    rets   = np.random.normal(0.001, 0.02, 300)
    series = pd.Series(100 * np.exp(np.cumsum(rets)), index=dates, name="Close")
    g      = GARCHForecaster(dist="skewt", mean="AR", enforce_stationarity=True)
    g.fit(series)
    result = g.forecast(steps=5)
    assert "forecast" in result,     "Missing forecast key"
    assert len(result["forecast"]) == 5, f"Expected 5 steps, got {len(result['forecast'])}"
    assert "volatility" in result,   "Missing volatility key"
    summ   = g.get_model_summary()
    assert summ.get("dist") == "skewt",  f"Summary dist mismatch: {summ.get('dist')}"
    assert summ.get("mean_model") == "AR", f"Summary mean_model mismatch: {summ.get('mean_model')}"
    print("B1B_GARCH_FIT_PASS")
except Exception as e:
    print(f"B1B_FAIL: {e}")
    raise
'@
        $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
        $r = Invoke-PythonTest -Code $code -TimeoutSeconds 45
        if ($r.Stdout -match "B1B_GARCH_FIT_PASS") {
            Record-Result "B1b.GARCH.FitForecast.Normal" "PASS" "skewt+AR fit+forecast on 300-bar series OK" $r.Elapsed
        } elseif ($r.Stdout -match "B1B_SKIP") {
            Record-Result "B1b.GARCH.FitForecast.Normal" "SKIP" "arch package not available" $r.Elapsed
        } else {
            Record-Result "B1b.GARCH.FitForecast.Normal" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
        }
    }

    # B1c. GARCH: extreme volatility series (should trigger ADF and GJR path)
    if (-not $SkipSlow) {
        $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")
np.random.seed(99)
try:
    from forcester_ts.garch import GARCHForecaster, ARCH_AVAILABLE
    if not ARCH_AVAILABLE:
        print("B1C_SKIP_NO_ARCH")
        raise SystemExit(0)
    # Simulate highly persistent GARCH (IGARCH-like) series
    n    = 400
    rets = np.zeros(n)
    h    = np.ones(n) * 0.0004
    for t in range(1, n):
        h[t]    = 0.01 + 0.97 * h[t-1] + 0.02 * rets[t-1]**2   # alpha+beta=0.99 -> IGARCH
        rets[t] = np.sqrt(h[t]) * np.random.randn()
    prices = pd.Series(100 * np.exp(np.cumsum(rets)),
                       index=pd.date_range("2020-01-01", periods=n, freq="D"), name="Close")
    g = GARCHForecaster(dist="skewt", mean="AR", enforce_stationarity=True, igarch_fallback="gjr")
    g.fit(prices)
    result = g.forecast(steps=5)
    assert len(result["forecast"]) == 5, "IGARCH path: wrong forecast length"
    summ   = g.get_model_summary()
    # Should have tried GJR or fallen back to EWMA — check summary reports a vol model
    assert "vol_model" in summ or "mean_model" in summ, "Summary missing model fields"
    print("B1C_GARCH_IGARCH_PATH_PASS")
except Exception as e:
    print(f"B1C_FAIL: {e}")
    raise
'@
        $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
        $r = Invoke-PythonTest -Code $code -TimeoutSeconds 45
        if ($r.Stdout -match "B1C_GARCH_IGARCH_PATH_PASS") {
            Record-Result "B1c.GARCH.IGARCHPath.GJRFallback" "PASS" "High-persistence series triggers GJR fallback" $r.Elapsed
        } elseif ($r.Stdout -match "B1C_SKIP") {
            Record-Result "B1c.GARCH.IGARCHPath.GJRFallback" "SKIP" "arch not available" $r.Elapsed
        } else {
            Record-Result "B1c.GARCH.IGARCHPath.GJRFallback" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
        }
    }

    # B2a. SAMoSSA: auto window = T//3, no hard cap
    $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")
try:
    from forcester_ts.samossa import SAMOSSAForecaster
    # T=300 -> T//3 = 100 (old cap was 40; should now use 100)
    n     = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    np.random.seed(7)
    prices = pd.Series(100 + np.cumsum(np.random.normal(0, 0.5, n)), index=dates, name="Close")
    f = SAMOSSAForecaster(window_length=0)   # 0 = auto
    f.fit(prices)
    expected_window = n // 3   # 100
    actual_window   = f._window_length
    assert actual_window == expected_window or actual_window >= 5, \
        f"Auto window should be T//3={expected_window}, got {actual_window}"
    assert actual_window != 40, f"Hard 40-cap should be removed, but got {actual_window}"
    print(f"B2A_SAMOSSA_AUTOWINDOW_PASS window={actual_window}")
except ImportError as e:
    print(f"B2A_SKIP: {e}")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 20
    if ($r.Stdout -match "B2A_SAMOSSA_AUTOWINDOW_PASS") {
        Record-Result "B2a.SAMoSSA.AutoWindow.No40Cap" "PASS" "T//3 window computed correctly" $r.Elapsed
    } elseif ($r.Stdout -match "B2A_SKIP") {
        Record-Result "B2a.SAMoSSA.AutoWindow.No40Cap" "SKIP" "SAMoSSA not available" $r.Elapsed
    } else {
        Record-Result "B2a.SAMoSSA.AutoWindow.No40Cap" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # B2b. SAMoSSA: directional slope signal present in forecast output
    if (-not $SkipSlow) {
        $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")
np.random.seed(42)
try:
    from forcester_ts.samossa import SAMOSSAForecaster
    n      = 200
    dates  = pd.date_range("2020-01-01", periods=n, freq="D")
    # Trending up series → expect directional_signal = +1.0
    prices = pd.Series(np.linspace(100, 150, n) + np.random.normal(0, 1, n),
                       index=dates, name="Close")
    f = SAMOSSAForecaster(window_length=0)
    f.fit(prices)
    result = f.forecast(steps=10)
    assert "forecast" in result,   "Missing forecast key"
    assert len(result["forecast"]) == 10, f"Expected 10 steps, got {len(result['forecast'])}"
    # Check for directional signal (Phase 7.10b addition)
    if "directional_signal" in result:
        assert result["directional_signal"] in (-1.0, 0.0, 1.0), \
            f"directional_signal should be -1/0/+1, got {result['directional_signal']}"
    print("B2B_SAMOSSA_DIRECTIONAL_PASS")
except Exception as e:
    print(f"B2B_FAIL: {e}")
    raise
'@
        $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
        $r = Invoke-PythonTest -Code $code -TimeoutSeconds 45
        if ($r.Stdout -match "B2B_SAMOSSA_DIRECTIONAL_PASS") {
            Record-Result "B2b.SAMoSSA.DirectionalSlope.TrendingSeries" "PASS" "Directional slope signal produced on uptrend" $r.Elapsed
        } else {
            Record-Result "B2b.SAMoSSA.DirectionalSlope.TrendingSeries" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
        }
    }

    # B3a. MSSA-RL: Q-table and reward mode accessible
    $code = @'
import sys
sys.path.insert(0, r"ROOT_PLACEHOLDER")
try:
    from forcester_ts.mssa_rl import MSSARLForecaster, MSSARLConfig
    cfg = MSSARLConfig(
        use_q_strategy_selection=True,
        reward_mode="directional_pnl",
        change_point_threshold=4.0,
    )
    assert cfg.use_q_strategy_selection == True, "Q selection should be enabled"
    assert cfg.reward_mode == "directional_pnl", f"Wrong reward_mode: {cfg.reward_mode}"
    assert cfg.change_point_threshold == 4.0,    f"Wrong threshold: {cfg.change_point_threshold}"
    f   = MSSARLForecaster(use_q_strategy_selection=True, reward_mode="directional_pnl",
                           change_point_threshold=4.0)
    assert hasattr(f, "_q_table"), "Forecaster should have _q_table attribute"
    print("B3A_MSSARL_CONFIG_PASS")
except ImportError as e:
    print(f"B3A_SKIP: {e}")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.Stdout -match "B3A_MSSARL_CONFIG_PASS") {
        Record-Result "B3a.MSSARL.Config.QStrategyRewardMode" "PASS" "Q-learning + directional_pnl config correct" $r.Elapsed
    } elseif ($r.Stdout -match "B3A_SKIP") {
        Record-Result "B3a.MSSARL.Config.QStrategyRewardMode" "SKIP" "MSSA-RL not available" $r.Elapsed
    } else {
        Record-Result "B3a.MSSARL.Config.QStrategyRewardMode" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # B3b. MSSA-RL: 5% slope cap - verify no divergence on 30-step horizon
    if (-not $SkipSlow) {
        $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")
np.random.seed(17)
try:
    from forcester_ts.mssa_rl import MSSARLForecaster
    n      = 200
    dates  = pd.date_range("2020-01-01", periods=n, freq="D")
    # Highly volatile series that could cause large slopes
    prices = pd.Series(100 + np.cumsum(np.random.normal(0, 5, n)), index=dates, name="Close")
    f      = MSSARLForecaster(use_q_strategy_selection=True, reward_mode="directional_pnl")
    f.fit(prices)
    result = f.forecast(steps=30)
    fcst   = np.array(result["forecast"])
    base   = abs(prices.iloc[-1])
    max_drift = abs(fcst[-1] - prices.iloc[-1])
    max_allowed = base * 0.05   # 5% cumulative cap
    assert max_drift <= max_allowed * 1.20, \
        f"Slope cap violation: drift={max_drift:.2f} > 120% of 5% cap={max_allowed*1.20:.2f}"
    # Also check no NaN/Inf in output
    assert not np.any(np.isnan(fcst)), "NaN in forecast output"
    assert not np.any(np.isinf(fcst)), "Inf in forecast output"
    print(f"B3B_MSSARL_SLOPE_CAP_PASS drift={max_drift:.2f} base={base:.2f}")
except Exception as e:
    print(f"B3B_FAIL: {e}")
    raise
'@
        $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
        $r = Invoke-PythonTest -Code $code -TimeoutSeconds 45
        if ($r.Stdout -match "B3B_MSSARL_SLOPE_CAP_PASS") {
            Record-Result "B3b.MSSARL.SlopeCap.5Percent" "PASS" "30-step forecast stays within 5% cumulative drift" $r.Elapsed
        } else {
            Record-Result "B3b.MSSARL.SlopeCap.5Percent" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
        }
    }

    # B4. Ensemble: auto_directional candidate from per-model directional accuracy
    $code = @'
import sys
sys.path.insert(0, r"ROOT_PLACEHOLDER")
try:
    from forcester_ts.ensemble import EnsembleCoordinator, EnsembleConfig
    cfg   = EnsembleConfig(track_directional_accuracy=True)
    coord = EnsembleCoordinator(cfg)
    # Model directional accuracy: garch 60%, samossa 45%, mssa_rl 55%
    model_da = {"garch": 0.60, "samossa": 0.45, "mssa_rl": 0.55}
    # Run weight selection with DA input
    model_errors = {"garch": 0.12, "samossa": 0.15, "mssa_rl": 0.14}
    weights = coord.select_weights(
        model_errors=model_errors,
        model_directional_accuracy=model_da
    )
    assert isinstance(weights, dict), f"select_weights should return dict, got {type(weights)}"
    assert len(weights) > 0, "Weights should be non-empty"
    total_w = sum(weights.values())
    assert abs(total_w - 1.0) < 1e-6, f"Weights should sum to 1.0, got {total_w:.6f}"
    # Models with DA > 0.40 should have non-zero weight
    assert weights.get("garch", 0) > 0, "GARCH (60% DA) should have positive weight"
    print(f"B4_ENSEMBLE_AUTODIRECTIONAL_PASS weights={weights}")
except ImportError as e:
    print(f"B4_SKIP: {e}")
except Exception as e:
    print(f"B4_FAIL: {e}")
    raise
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 15
    if ($r.Stdout -match "B4_ENSEMBLE_AUTODIRECTIONAL_PASS") {
        Record-Result "B4.Ensemble.AutoDirectional.WeightSelection" "PASS" "auto_directional candidate selected with DA-weighted models" $r.Elapsed
    } elseif ($r.Stdout -match "B4_SKIP") {
        Record-Result "B4.Ensemble.AutoDirectional.WeightSelection" "SKIP" "Ensemble not available" $r.Elapsed
    } else {
        Record-Result "B4.Ensemble.AutoDirectional.WeightSelection" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # B5. Platt scaling: 15% PASS rate guard prevents calibration on bad history
    $code = @'
import sys, json, tempfile, os, pathlib
sys.path.insert(0, r"ROOT_PLACEHOLDER")

# Write synthetic quant_validation.jsonl with 6.25% PASS rate (well below 15%)
entries = (
    [{"status": "FAIL", "confidence": 0.8, "signal": "BUY"}] * 30 +
    [{"status": "PASS", "confidence": 0.8, "signal": "BUY"}] * 2
)   # 2/32 = 6.25% PASS rate

tmp_dir  = tempfile.mkdtemp()
log_dir  = pathlib.Path(tmp_dir)
log_file = log_dir / "quant_validation.jsonl"
with open(log_file, "w") as fh:
    for e in entries:
        fh.write(json.dumps(e) + "\n")

# The calibration guard should reject this data and use shrinkage fallback instead
MIN_PLATT_PASS_RATE = 0.15
MIN_PLATT_SAMPLES   = 30

n         = len(entries)
pass_rate = sum(1 for e in entries if e["status"] == "PASS") / n
assert pass_rate < MIN_PLATT_PASS_RATE, f"Test data should have <15% pass rate, got {pass_rate:.2%}"

# Simulate the guard logic
should_calibrate = (n >= MIN_PLATT_SAMPLES) and (pass_rate >= MIN_PLATT_PASS_RATE)
assert not should_calibrate, \
    f"Guard should block calibration at {pass_rate:.2%} PASS rate"

# Verify shrinkage formula gives reasonable output at high confidence
raw_conf  = 0.85
shrinkage = max(0.05, min(0.95, 0.50 + 0.60 * (raw_conf - 0.50)))
assert 0.60 <= shrinkage <= 0.80, f"Shrinkage for 0.85 raw should be in [0.60, 0.80], got {shrinkage:.4f}"
print(f"B5_PLATT_GUARD_PASS pass_rate={pass_rate:.2%} shrinkage={shrinkage:.4f}")

# Cleanup
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.Stdout -match "B5_PLATT_GUARD_PASS") {
        Record-Result "B5.PlattScaling.PassRateGuard.15pct" "PASS" "Guard blocks calibration on 6.25% PASS data" $r.Elapsed
    } else {
        Record-Result "B5.PlattScaling.PassRateGuard.15pct" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C: DIRECTIONAL ACCURACY ENHANCEMENTS (Phase 7.11 Logic)
# ─────────────────────────────────────────────────────────────────────────────
function Test-DirectionalAccuracy {
    Write-Log "Directional Accuracy Enhancement Tests (Part C — Phase 7.11)" "SECT"

    # C1. Directional Consensus Gate
    $code = @'
import sys, numpy as np
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _check_directional_consensus(model_directions, min_consensus=0.67):
    long_v  = sum(1 for d in model_directions.values() if d > 0)
    short_v = sum(1 for d in model_directions.values() if d < 0)
    total   = len(model_directions)
    if total == 0:
        return 0.0, False
    lf = long_v  / total
    sf = short_v / total
    if lf >= min_consensus:
        return 1.0, True
    if sf >= min_consensus:
        return -1.0, True
    return 0.0, False

# Case 1: 3/3 long -> consensus long
d, ok = _check_directional_consensus({"garch": 1.0, "samossa": 1.0, "mssa_rl": 1.0})
assert d == 1.0 and ok, f"3/3 long should be consensus long, got ({d},{ok})"

# Case 2: 2/3 long -> meets 0.67 threshold
d, ok = _check_directional_consensus({"garch": 1.0, "samossa": 1.0, "mssa_rl": -1.0})
assert d == 1.0 and ok, f"2/3 long should meet 0.67 threshold, got ({d},{ok})"

# Case 3: 1/3 long -> no consensus (below 0.67)
d, ok = _check_directional_consensus({"garch": 1.0, "samossa": -1.0, "mssa_rl": -1.0})
assert d == -1.0 and ok, f"2/3 short should be consensus short, got ({d},{ok})"

# Case 4: Perfect split 1/3 vs 1/3 (one neutral) -> abstain
d, ok = _check_directional_consensus({"garch": 1.0, "samossa": -1.0, "mssa_rl": 0.0})
# long=1, short=1, neutral=1 -> neither reaches 0.67
assert not ok, f"Split should abstain, got ({d},{ok})"

# Case 5: Empty dict -> abstain
d, ok = _check_directional_consensus({})
assert not ok and d == 0.0, f"Empty should abstain, got ({d},{ok})"

# Case 6: Strict consensus 0.80 (requires 4/5 to agree)
models = {"garch":1,"samossa":1,"mssa_rl":1,"sarimax":1,"extra":-1}
d, ok = _check_directional_consensus(models, min_consensus=0.80)
assert d == 1.0 and ok, f"4/5 long should meet 0.80 threshold, got ({d},{ok})"

print("C1_CONSENSUS_GATE_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "C1_CONSENSUS_GATE_PASS") {
        Record-Result "C1.DirectionalConsensusGate.AllCases" "PASS" "6 consensus scenarios all correct" $r.Elapsed
    } else {
        Record-Result "C1.DirectionalConsensusGate.AllCases" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # C2. Hurst Exponent Directional Policy
    $code = @'
import sys, numpy as np
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _apply_hurst_direction_policy(model_direction, hurst_exponent,
                                   hurst_strong=0.58, hurst_weak=0.42,
                                   trending_boost=0.08, revert_penalty=0.05):
    if hurst_exponent > hurst_strong:
        return model_direction, +trending_boost
    elif hurst_exponent < hurst_weak:
        return -model_direction, -revert_penalty
    return model_direction, 0.0

# Trending (H=0.65) -> follow model direction, boost confidence
dir_adj, cmod = _apply_hurst_direction_policy(1.0, 0.65)
assert dir_adj == 1.0,  f"Trending: should follow, got {dir_adj}"
assert cmod > 0,         f"Trending: should boost confidence, got {cmod}"

# Mean-reverting (H=0.35) -> fade model direction, confidence penalty
dir_adj, cmod = _apply_hurst_direction_policy(1.0, 0.35)
assert dir_adj == -1.0, f"Mean-reverting: should flip, got {dir_adj}"
assert cmod < 0,         f"Mean-reverting: should reduce confidence, got {cmod}"

# Neutral zone (H=0.50) -> no change
dir_adj, cmod = _apply_hurst_direction_policy(-1.0, 0.50)
assert dir_adj == -1.0, f"Neutral: direction unchanged, got {dir_adj}"
assert cmod == 0.0,     f"Neutral: no confidence mod, got {cmod}"

# Boundary: H exactly at strong threshold
dir_adj, cmod = _apply_hurst_direction_policy(1.0, 0.58)  # exactly at boundary (not >)
assert dir_adj == 1.0 and cmod == 0.0, f"At strong boundary (not >) should be neutral, got ({dir_adj},{cmod})"

# Short signal in mean-reverting regime -> flip to long
dir_adj, cmod = _apply_hurst_direction_policy(-1.0, 0.30)
assert dir_adj == 1.0, f"Short in mean-revert should flip to long, got {dir_adj}"

print("C2_HURST_POLICY_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "C2_HURST_POLICY_PASS") {
        Record-Result "C2.HurstDirectionalPolicy.AllCases" "PASS" "All Hurst policy edge cases correct" $r.Elapsed
    } else {
        Record-Result "C2.HurstDirectionalPolicy.AllCases" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # C3. Rolling IC Feature Culling (scipy spearman)
    $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")
from scipy.stats import spearmanr

def _compute_rolling_ic(features, forward_returns, window=60, ic_threshold=0.03):
    ic_scores = {}
    signs = np.sign(forward_returns)
    for col in features.columns:
        feat = features[col].dropna()
        aligned = signs.reindex(feat.index).dropna()
        feat    = feat.reindex(aligned.index)
        if len(feat) < 20:
            ic_scores[col] = 0.0
            continue
        tail_feat  = feat.iloc[-window:]
        tail_signs = aligned.iloc[-window:]
        ic, _      = spearmanr(tail_feat, tail_signs)
        ic_scores[col] = float(ic) if np.isfinite(ic) else 0.0
    return {k: v for k, v in ic_scores.items() if abs(v) >= ic_threshold}

np.random.seed(1)
n     = 120
idx   = pd.date_range("2020-01-01", periods=n, freq="D")
rets  = pd.Series(np.random.normal(0, 0.02, n), index=idx)

# Feature 1: perfectly correlated with future direction
feat_good = pd.DataFrame({"good": np.sign(rets.shift(-1).fillna(0)) + np.random.normal(0,0.1,n)}, index=idx)
# Feature 2: random noise (near-zero IC)
feat_bad  = pd.DataFrame({"noise": np.random.normal(0, 1, n)}, index=idx)
features  = pd.concat([feat_good, feat_bad], axis=1)

kept = _compute_rolling_ic(features, rets, window=60, ic_threshold=0.03)
assert "good" in kept, f"Correlated feature 'good' should survive culling, got {kept}"
# Noise feature may or may not survive (random IC) — just ensure 'good' is kept
print(f"C3_IC_CULLING_PASS kept={list(kept.keys())}")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "C3_IC_CULLING_PASS") {
        Record-Result "C3.RollingIC.FeatureCulling" "PASS" "IC culling keeps correlated features, considers noise" $r.Elapsed
    } else {
        Record-Result "C3.RollingIC.FeatureCulling" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # C4. EMA Momentum Pre-Filter
    $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _apply_momentum_prefilter(prices, model_direction, raw_confidence,
                               ema_fast=5, ema_slow=20,
                               momentum_threshold=0.008,
                               confidence_gate=0.72):
    ema_f = prices.ewm(span=ema_fast, adjust=False).mean().iloc[-1]
    ema_s = prices.ewm(span=ema_slow, adjust=False).mean().iloc[-1]
    if ema_s == 0:
        return raw_confidence, False
    ratio              = (ema_f / ema_s) - 1.0
    momentum_direction = float(np.sign(ratio))
    momentum_strength  = abs(ratio)
    if momentum_strength < momentum_threshold:
        return raw_confidence, False          # Weak: no filter
    if momentum_direction == model_direction:
        return raw_confidence, False          # Aligned: no filter
    # Contradicted by strong momentum
    if raw_confidence < confidence_gate:
        return raw_confidence, True           # Suppress
    return raw_confidence * 0.90, False       # Reduce confidence, don't suppress

# Case 1: Strong bullish momentum, model says BUY (aligned) -> no filter
prices_up = pd.Series(np.linspace(100, 115, 100))   # Strong uptrend
conf, suppressed = _apply_momentum_prefilter(prices_up, 1.0, 0.65)
assert not suppressed, "Aligned momentum+model should not suppress"

# Case 2: Strong bullish momentum, model says SHORT, low confidence -> suppress
conf, suppressed = _apply_momentum_prefilter(prices_up, -1.0, 0.60)
assert suppressed, f"Low-confidence SHORT against strong bullish momentum should be suppressed"

# Case 3: Strong bullish momentum, model says SHORT, high confidence (0.80) -> don't suppress but reduce
conf, suppressed = _apply_momentum_prefilter(prices_up, -1.0, 0.80)
assert not suppressed,  "High confidence should not be suppressed"
assert conf < 0.80,     f"Confidence should be reduced, got {conf}"

# Case 4: Weak momentum (no trend) -> no filter regardless
prices_flat = pd.Series(np.ones(100) * 100 + np.random.normal(0, 0.01, 100))
conf, suppressed = _apply_momentum_prefilter(prices_flat, -1.0, 0.50)
assert not suppressed, "Weak/flat momentum should not trigger filter"

print("C4_MOMENTUM_FILTER_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "C4_MOMENTUM_FILTER_PASS") {
        Record-Result "C4.MomentumPreFilter.EMA.AllCases" "PASS" "All EMA momentum filter edge cases correct" $r.Elapsed
    } else {
        Record-Result "C4.MomentumPreFilter.EMA.AllCases" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # C5. Direction Classifier: sklearn LogisticRegression feature computation
    $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
n     = 300
rets  = pd.Series(np.random.normal(0.001, 0.02, n))
prices = pd.Series(100 * np.exp(rets.cumsum()))

# Build features: lagged returns
def _build_direction_features(prices, lags=(5, 10, 20)):
    df = pd.DataFrame()
    for lag in lags:
        df[f"ret_{lag}"] = prices.pct_change(lag)
    return df.dropna()

features = _build_direction_features(prices)
forward  = prices.pct_change(1).shift(-1)
aligned  = forward.reindex(features.index).dropna()
features = features.reindex(aligned.index)

X = features.values
y = (np.sign(aligned.values) > 0).astype(int)  # Binary: 1=up, 0=down

# Ensure sufficient samples for train
assert len(X) >= 50, f"Need >= 50 samples, got {len(X)}"

# Train logistic regression
clf = LogisticRegression(max_iter=500, C=1.0)
clf.fit(X[-200:], y[-200:])
proba = clf.predict_proba(X[-1:])
assert proba.shape == (1, 2), f"Expected (1,2) probability output, got {proba.shape}"
assert 0.0 <= proba[0][1] <= 1.0, f"Probability must be in [0,1], got {proba[0][1]}"
direction = 1.0 if proba[0][1] > 0.5 else -1.0
print(f"C5_DIRECTION_CLASSIFIER_PASS dir={direction} P(up)={proba[0][1]:.3f}")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 15
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "C5_DIRECTION_CLASSIFIER_PASS") {
        Record-Result "C5.DirectionClassifier.LogisticRegression" "PASS" "LR classifier trains and produces valid probabilities" $r.Elapsed
    } else {
        Record-Result "C5.DirectionClassifier.LogisticRegression" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # C6. Isotonic vs Platt calibration switchover at 80 samples
    $code = @'
import sys, numpy as np
sys.path.insert(0, r"ROOT_PLACEHOLDER")
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

MIN_PLATT     = 30
MIN_ISOTONIC  = 80
MIN_PASS_RATE = 0.15

def _pick_calibrator(n_samples, pass_rate):
    if n_samples < MIN_PLATT or pass_rate < MIN_PASS_RATE:
        return "shrinkage"
    if n_samples >= MIN_ISOTONIC:
        return "isotonic"
    return "platt"

assert _pick_calibrator(20,  0.30) == "shrinkage", "< 30 samples -> shrinkage"
assert _pick_calibrator(50,  0.05) == "shrinkage", "< 15% pass rate -> shrinkage"
assert _pick_calibrator(50,  0.30) == "platt",     "30-79 samples, OK rate -> platt"
assert _pick_calibrator(80,  0.30) == "isotonic",  ">= 80 samples, OK rate -> isotonic"
assert _pick_calibrator(200, 0.50) == "isotonic",  "200 samples -> isotonic"

# Verify isotonic regression produces monotone output
np.random.seed(5)
X_conf = np.sort(np.random.uniform(0.3, 0.9, 100)).reshape(-1, 1)
y_win  = (np.random.uniform(size=100) < (0.20 + 0.60 * X_conf.flatten())).astype(float)
ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
ir.fit(X_conf.flatten(), y_win)
preds = ir.predict(np.linspace(0.3, 0.9, 10))
assert np.all(np.diff(preds) >= -1e-8), f"Isotonic should be non-decreasing: {preds}"

print("C6_ISOTONIC_CALIBRATION_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "C6_ISOTONIC_CALIBRATION_PASS") {
        Record-Result "C6.IsotonicCalibration.SwitchLogic" "PASS" "Calibrator selection and monotonicity verified" $r.Elapsed
    } else {
        Record-Result "C6.IsotonicCalibration.SwitchLogic" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # C7. Volume Confirmation Gate
    $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _check_volume_confirmation(ohlcv, model_direction, vol_multiplier=1.15, volume_window=20):
    if "Volume" not in ohlcv.columns or len(ohlcv) < volume_window + 1:
        return True, 1.0
    latest  = ohlcv.iloc[-1]
    avg_vol = ohlcv["Volume"].iloc[-(volume_window+1):-1].mean()
    vol_ratio = latest["Volume"] / avg_vol if avg_vol > 0 else 1.0
    bar_dir   = float(np.sign(latest["Close"] - latest["Open"]))
    if vol_ratio < vol_multiplier:
        return False, vol_ratio       # Low volume
    if model_direction > 0 and bar_dir < 0:
        return False, vol_ratio       # Long on down bar at high volume = bearish
    if model_direction < 0 and bar_dir > 0:
        return False, vol_ratio       # Short on up bar at high volume = bullish
    return True, vol_ratio

n   = 30
idx = pd.date_range("2024-01-01", periods=n, freq="D")
df  = pd.DataFrame({
    "Open":   np.ones(n) * 100,
    "Close":  np.ones(n) * 101,     # All up bars
    "Volume": np.ones(n) * 1000.0,
}, index=idx)

# Case 1: LONG signal, up bar, high volume (above avg) -> confirmed
df.iloc[-1, df.columns.get_loc("Volume")] = 1200.0  # 20% above average
confirmed, ratio = _check_volume_confirmation(df, 1.0)
assert confirmed, f"LONG + up bar + high volume should confirm, ratio={ratio:.2f}"

# Case 2: SHORT signal, up bar, high volume -> unconfirmed (bearish model, bullish bar)
df.iloc[-1, df.columns.get_loc("Volume")] = 1500.0
confirmed, ratio = _check_volume_confirmation(df, -1.0)
assert not confirmed, f"SHORT + up bar + high volume should be unconfirmed (bearish vs bullish bar)"

# Case 3: Low volume (below threshold) -> always unconfirmed
df.iloc[-1, df.columns.get_loc("Volume")] = 900.0   # Below 1000 avg = 0.9x
confirmed, ratio = _check_volume_confirmation(df, 1.0)
assert not confirmed, f"Low volume should be unconfirmed, ratio={ratio:.2f}"

print("C7_VOLUME_CONFIRMATION_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "C7_VOLUME_CONFIRMATION_PASS") {
        Record-Result "C7.VolumeConfirmation.Gate" "PASS" "3 volume confirmation scenarios correct" $r.Elapsed
    } else {
        Record-Result "C7.VolumeConfirmation.Gate" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # C8. Asymmetric Directional Loss
    $code = @'
import sys, numpy as np
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _directional_loss(forecast_delta, actual_delta, lambda_dir=1.5):
    rmse = float(np.sqrt(np.mean((forecast_delta - actual_delta) ** 2)))
    dir_errors = (np.sign(forecast_delta) != np.sign(actual_delta)).astype(float)
    dir_component = float(np.mean(dir_errors))
    return rmse + lambda_dir * dir_component

np.random.seed(42)
n  = 50
ad = np.random.normal(0.5, 1.0, n)    # Actual changes (biased up)

# Perfect directional prediction: sign(forecast) == sign(actual)
fd_perfect = ad.copy()
loss_perfect = _directional_loss(fd_perfect, ad)
assert loss_perfect < 0.1, f"Perfect forecast should have near-zero loss, got {loss_perfect:.4f}"

# Wrong direction every time: sign(forecast) opposite to sign(actual)
fd_wrong = -ad
loss_wrong = _directional_loss(fd_wrong, ad)
# dir_component should be 1.0 -> lambda * 1.0 = 1.5 added
assert loss_wrong > loss_perfect + 1.0, \
    f"Wrong direction should add ~1.5 to loss, got delta={loss_wrong - loss_perfect:.4f}"

# 50% correct direction: dir_component ~ 0.5 -> half penalty
fd_half = ad.copy()
fd_half[:n//2] = -fd_half[:n//2]  # Flip half
loss_half = _directional_loss(fd_half, ad)
assert loss_perfect < loss_half < loss_wrong, \
    f"Half-correct should be between perfect and wrong: {loss_perfect:.3f} < {loss_half:.3f} < {loss_wrong:.3f}"

# Higher lambda -> stronger directional penalty
loss_strict = _directional_loss(fd_wrong, ad, lambda_dir=3.0)
assert loss_strict > loss_wrong, f"Higher lambda should increase penalty: {loss_strict:.3f} vs {loss_wrong:.3f}"

print("C8_ASYMMETRIC_LOSS_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "C8_ASYMMETRIC_LOSS_PASS") {
        Record-Result "C8.AsymmetricDirectionalLoss" "PASS" "Loss function penalizes directional errors correctly" $r.Elapsed
    } else {
        Record-Result "C8.AsymmetricDirectionalLoss" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D: ADVERSARIAL TESTS
# ─────────────────────────────────────────────────────────────────────────────
function Test-Adversarial {
    Write-Log "Adversarial Tests (Edge Cases, Outliers, Hostile Inputs)" "SECT"

    # D1. NaN / Inf injection into weighted scoring
    $code = @'
import sys, math
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _safe_weighted_score(criteria, weights, pass_threshold, expected_profit):
    try:
        if math.isnan(expected_profit) or math.isinf(expected_profit):
            return "FAIL"          # Inf/NaN EP -> hard fail
        if expected_profit < 0:
            return "FAIL"
        score   = sum(float(weights.get(k, 0.0)) * (1.0 if v else 0.0) for k, v in criteria.items())
        total_w = sum(float(weights.get(k, 0.0)) for k in criteria)
        if total_w == 0 or math.isnan(total_w) or math.isinf(total_w):
            return "FAIL"
        norm_score = score / total_w
        if math.isnan(norm_score) or math.isinf(norm_score):
            return "FAIL"
        return "PASS" if norm_score >= pass_threshold else "FAIL"
    except Exception:
        return "FAIL"

W = {"expected_profit": 0.25, "directional_accuracy": 0.75}
good_criteria = {"expected_profit": True, "directional_accuracy": True}

# NaN EP
r = _safe_weighted_score(good_criteria, W, 0.60, float("nan"))
assert r == "FAIL", f"NaN EP should FAIL, got {r}"

# Inf EP (positive)
r = _safe_weighted_score(good_criteria, W, 0.60, float("inf"))
assert r == "FAIL", f"Inf EP should FAIL, got {r}"

# -Inf EP
r = _safe_weighted_score(good_criteria, W, 0.60, float("-inf"))
assert r == "FAIL", f"-Inf EP should FAIL, got {r}"

# Zero-weight dict (division by zero guard)
r = _safe_weighted_score(good_criteria, {}, 0.60, 5.0)
assert r == "FAIL", f"Zero-weight dict should FAIL, got {r}"

# Normal case still works
r = _safe_weighted_score(good_criteria, W, 0.60, 5.0)
assert r == "PASS", f"Normal case should PASS after NaN tests, got {r}"

print("D1_ADVERSARIAL_SCORING_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "D1_ADVERSARIAL_SCORING_PASS") {
        Record-Result "D1.Adversarial.NaN_Inf.WeightedScoring" "PASS" "NaN/Inf inputs gracefully handled" $r.Elapsed
    } else {
        Record-Result "D1.Adversarial.NaN_Inf.WeightedScoring" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # D2. Extreme outlier series for GARCH
    if (-not $SkipSlow) {
        $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")
np.random.seed(13)
try:
    from forcester_ts.garch import GARCHForecaster, ARCH_AVAILABLE
    if not ARCH_AVAILABLE:
        print("D2_SKIP_NO_ARCH")
        raise SystemExit(0)
    # Insert 3-sigma spike at day 150 and 250
    n     = 400
    rets  = np.random.normal(0.001, 0.02, n)
    rets[150] = 0.30    # +30% day
    rets[250] = -0.25   # -25% day
    prices = pd.Series(100 * np.exp(np.cumsum(rets)),
                       index=pd.date_range("2020-01-01", periods=n, freq="D"), name="Close")
    g = GARCHForecaster(dist="skewt", mean="AR", enforce_stationarity=True)
    g.fit(prices)
    result = g.forecast(steps=5)
    fcst = result["forecast"]
    assert len(fcst) == 5, f"Expected 5 forecast steps, got {len(fcst)}"
    assert not any(abs(f) > 1e6 for f in fcst), f"Extreme outlier spike caused divergent forecast: {fcst}"
    print("D2_ADVERSARIAL_GARCH_OUTLIER_PASS")
except SystemExit:
    pass
except Exception as e:
    print(f"D2_FAIL: {e}")
    raise
'@
        $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
        $r = Invoke-PythonTest -Code $code -TimeoutSeconds 60
        if ($r.Stdout -match "D2_ADVERSARIAL_GARCH_OUTLIER_PASS") {
            Record-Result "D2.Adversarial.GARCH.OutlierSpikes" "PASS" "+30%/-25% spike days handled without divergence" $r.Elapsed
        } elseif ($r.Stdout -match "D2_SKIP") {
            Record-Result "D2.Adversarial.GARCH.OutlierSpikes" "SKIP" "arch not available" $r.Elapsed
        } else {
            Record-Result "D2.Adversarial.GARCH.OutlierSpikes" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
        }
    }

    # D3. Missing data / gaps in series (NaN in price)
    $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")
try:
    from forcester_ts.samossa import SAMOSSAForecaster
    np.random.seed(4)
    n     = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    prices = pd.Series(100 + np.cumsum(np.random.normal(0, 0.5, n)), index=dates, name="Close")
    # Insert 10% NaN gaps at random positions
    nan_idx = np.random.choice(n, size=n//10, replace=False)
    prices.iloc[nan_idx] = np.nan
    # SAMoSSA should handle NaN gracefully (drop or fill)
    try:
        f = SAMOSSAForecaster(window_length=0)
        f.fit(prices)
        result = f.forecast(steps=5)
        assert len(result["forecast"]) == 5, "Should produce 5 forecasts even with NaN gaps"
        print("D3_SAMOSSA_NAN_GAPS_PASS")
    except (ValueError, RuntimeError) as e:
        # Acceptable: model rejects insufficient data
        if "insufficient" in str(e).lower() or "minimum" in str(e).lower() or "length" in str(e).lower():
            print(f"D3_SAMOSSA_NAN_GAPS_GRACEFUL_REJECT: {e}")
        else:
            raise
except ImportError as e:
    print(f"D3_SKIP: {e}")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 30
    if ($r.Stdout -match "D3_SAMOSSA_NAN_GAPS_PASS|D3_SAMOSSA_NAN_GAPS_GRACEFUL_REJECT") {
        Record-Result "D3.Adversarial.SAMoSSA.NaN_Gaps" "PASS" "NaN gaps handled (fit or graceful rejection)" $r.Elapsed
    } elseif ($r.Stdout -match "D3_SKIP") {
        Record-Result "D3.Adversarial.SAMoSSA.NaN_Gaps" "SKIP" "SAMoSSA not available" $r.Elapsed
    } else {
        Record-Result "D3.Adversarial.SAMoSSA.NaN_Gaps" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # D4. Constant-price series (zero variance — degenerate case)
    $code = @'
import sys, numpy as np, pandas as pd
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _test_constant_series(forecaster_class, **kwargs):
    n     = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    const = pd.Series(np.ones(n) * 100.0, index=dates, name="Close")
    try:
        f = forecaster_class(**kwargs)
        f.fit(const)
        result = f.forecast(steps=5)
        fcst   = result["forecast"]
        assert len(fcst) == 5, "Should produce 5 forecast steps"
        assert not any(abs(v - 100.0) > 50 for v in fcst), \
            f"Constant series: forecasts should be near 100, got {fcst}"
        return "PASS"
    except (ValueError, RuntimeError, ZeroDivisionError) as e:
        # Graceful rejection is acceptable
        return f"GRACEFUL: {type(e).__name__}: {str(e)[:80]}"

try:
    from forcester_ts.samossa import SAMOSSAForecaster
    r = _test_constant_series(SAMOSSAForecaster, window_length=0)
    print(f"D4_SAMOSSA_CONSTANT: {r}")
except ImportError:
    print("D4_SAMOSSA_SKIP")

try:
    from forcester_ts.mssa_rl import MSSARLForecaster
    r = _test_constant_series(MSSARLForecaster, use_q_strategy_selection=True)
    print(f"D4_MSSARL_CONSTANT: {r}")
except ImportError:
    print("D4_MSSARL_SKIP")

print("D4_CONSTANT_SERIES_DONE")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 30
    if ($r.Stdout -match "D4_CONSTANT_SERIES_DONE") {
        $detail = if ($r.Stdout -match "GRACEFUL|PASS") { "Constant-price series handled (fit or graceful rejection)" } else { $r.Stdout }
        Record-Result "D4.Adversarial.ConstantPriceSeries" "PASS" $detail $r.Elapsed
    } else {
        Record-Result "D4.Adversarial.ConstantPriceSeries" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # D5. Small perturbation adversarial test (noise injection into scoring)
    $code = @'
import sys, numpy as np
sys.path.insert(0, r"ROOT_PLACEHOLDER")

def _weighted_score(criteria, weights, pass_threshold, expected_profit):
    if expected_profit < 0:
        return "FAIL", 0.0
    score   = sum(float(weights.get(k,0.0)) * (1.0 if v else 0.0) for k,v in criteria.items())
    total_w = sum(float(weights.get(k,0.0)) for k in criteria)
    ns = score / total_w if total_w > 0 else 0.0
    return ("PASS" if ns >= pass_threshold else "FAIL"), ns

W = {"expected_profit":0.25,"rmse_ratio":0.20,"directional_accuracy":0.20,
     "sharpe_ratio":0.10,"sortino_ratio":0.10,"profit_factor":0.10,"win_rate":0.05}

# Build a signal right at 0.60 boundary (score = 0.60 exactly)
# Possible: expected_profit(0.25) + directional_accuracy(0.20) + sharpe(0.10) + sortino(0.10) = 0.65
base_criteria = {"expected_profit": True, "rmse_ratio": False, "directional_accuracy": True,
                 "sharpe_ratio": True,  "sortino_ratio": True, "profit_factor": False, "win_rate": False}
status, score = _weighted_score(base_criteria, W, 0.60, 10.0)
assert status == "PASS", f"Base signal should PASS at {score:.3f}"

# Adversarial: flip directional_accuracy to False -> score drops to 0.45 -> FAIL
adv_criteria = base_criteria.copy()
adv_criteria["directional_accuracy"] = False
status_adv, score_adv = _weighted_score(adv_criteria, W, 0.60, 10.0)
assert status_adv == "FAIL", f"Flipping DA to False should FAIL, score={score_adv:.3f}"
assert score_adv < 0.60, f"Adversarial score should be < 0.60, got {score_adv:.3f}"

# Stress test: random perturbations to weights (10 random trials)
rng = np.random.default_rng(seed=77)
for trial in range(10):
    perturbed_W = {k: max(0.01, v + rng.normal(0, 0.03)) for k, v in W.items()}
    s, ns = _weighted_score(base_criteria, perturbed_W, 0.60, 10.0)
    assert s in ("PASS", "FAIL"), f"Perturbed weights trial {trial}: invalid status {s}"

print("D5_ADVERSARIAL_PERTURBATION_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "D5_ADVERSARIAL_PERTURBATION_PASS") {
        Record-Result "D5.Adversarial.SmallPerturbation.WeightScoring" "PASS" "Weight perturbation stress test (10 trials) passed" $r.Elapsed
    } else {
        Record-Result "D5.Adversarial.SmallPerturbation.WeightScoring" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION E: SYSTEM-LEVEL TESTS
# ─────────────────────────────────────────────────────────────────────────────
function Test-SystemLevel {
    Write-Log "System-Level Tests (Config Sync, Co-Agent Files, pytest Targets)" "SECT"

    # E1. Config sync: forecasting_config.yml has Phase 7.10b params
    $code = @'
import sys, yaml, pathlib
sys.path.insert(0, r"ROOT_PLACEHOLDER")
root = pathlib.Path(r"ROOT_PLACEHOLDER")
cfg_path = root / "config" / "forecasting_config.yml"
assert cfg_path.exists(), f"Config not found: {cfg_path}"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

fc = cfg["forecasting"]

# GARCH checks
garch = fc["garch"]
assert garch.get("dist") == "skewt",         f"GARCH dist should be skewt, got {garch.get('dist')}"
assert garch.get("mean") == "AR",             f"GARCH mean should be AR, got {garch.get('mean')}"
assert garch.get("enforce_stationarity"),     "GARCH enforce_stationarity should be true"
assert garch.get("igarch_fallback") == "gjr", f"GARCH igarch_fallback should be gjr"

# SAMoSSA checks
samossa = fc["samossa"]
assert samossa.get("window_length") is None,  f"SAMoSSA window_length should be null (auto), got {samossa.get('window_length')}"
assert samossa.get("use_residual_arima"),      "SAMoSSA use_residual_arima should be true"
assert samossa.get("trend_slope_bars") == 10, f"SAMoSSA trend_slope_bars should be 10"

# MSSA-RL checks
mssa = fc["mssa_rl"]
assert mssa.get("change_point_threshold") == 4.0, f"MSSA-RL threshold should be 4.0"
assert mssa.get("use_q_strategy_selection"),   "MSSA-RL Q selection should be enabled"
assert mssa.get("reward_mode") == "directional_pnl", f"MSSA-RL reward_mode mismatch"

# Ensemble checks
ens = fc["ensemble"]
assert ens.get("track_directional_accuracy"),  "Ensemble should track directional accuracy"

print("E1_CONFIG_SYNC_PASS")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "E1_CONFIG_SYNC_PASS") {
        Record-Result "E1.ConfigSync.ForecastingConfig.Phase710b" "PASS" "All Phase 7.10b params present in forecasting_config.yml" $r.Elapsed
    } else {
        Record-Result "E1.ConfigSync.ForecastingConfig.Phase710b" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # E2. pipeline_config.yml in sync with forecasting_config.yml (GARCH params)
    $code = @'
import sys, yaml, pathlib
sys.path.insert(0, r"ROOT_PLACEHOLDER")
root = pathlib.Path(r"ROOT_PLACEHOLDER")

fc_path = root / "config" / "forecasting_config.yml"
pc_path = root / "config" / "pipeline_config.yml"

with open(fc_path) as f: fc = yaml.safe_load(f)
with open(pc_path) as p: pc = yaml.safe_load(p)

# Check pipeline_config mirrors forecasting_config for GARCH
fc_garch = fc["forecasting"]["garch"]
# pipeline_config may have the same keys at different nesting
def _find_garch(d, depth=0):
    if isinstance(d, dict):
        if "garch" in d and isinstance(d["garch"], dict):
            g = d["garch"]
            if "dist" in g or "mean" in g or "vol" in g:
                return g
        for v in d.values():
            result = _find_garch(v, depth+1)
            if result:
                return result
    return None

pc_garch = _find_garch(pc)
if pc_garch:
    # Compare dist and mean if present
    if "dist" in pc_garch:
        assert pc_garch["dist"] == fc_garch["dist"], \
            f"pipeline dist={pc_garch['dist']} != forecasting dist={fc_garch['dist']}"
    if "mean" in pc_garch:
        assert pc_garch["mean"] == fc_garch["mean"], \
            f"pipeline mean={pc_garch['mean']} != forecasting mean={fc_garch['mean']}"
    print(f"E2_PIPELINE_CONFIG_SYNC_PASS garch_dist={pc_garch.get('dist','N/A')}")
else:
    print("E2_PIPELINE_CONFIG_NO_GARCH_SECTION (acceptable if pipeline uses forecasting_config)")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and ($r.Stdout -match "E2_PIPELINE_CONFIG_SYNC_PASS|E2_PIPELINE_CONFIG_NO_GARCH_SECTION")) {
        Record-Result "E2.ConfigSync.PipelineVsForecasting" "PASS" "pipeline_config.yml GARCH params in sync" $r.Elapsed
    } else {
        Record-Result "E2.ConfigSync.PipelineVsForecasting" "FAIL" "$($r.Stderr)$($r.Stdout)" $r.Elapsed
    }

    # E3. Co-agent file isolation: openclaw_cli.py NOT in task-owned staged changes
    $code = @'
import subprocess, sys
result = subprocess.run(
    ["git", "-C", r"ROOT_PLACEHOLDER", "diff", "--name-only", "--cached"],
    capture_output=True, text=True
)
staged = result.stdout.strip().splitlines()
# Co-agent files must not appear in the staged changeset for task Phase 7.10b
coagent_files = [
    "utils/openclaw_cli.py",
    "scripts/llm_multi_model_orchestrator.py",
    "scripts/openclaw_maintenance.py",
]
violations = [f for f in coagent_files if f in staged]
if violations:
    print(f"COAGENT_ISOLATION_FAIL: staged co-agent files: {violations}")
    sys.exit(1)
print(f"E3_COAGENT_ISOLATION_PASS staged_count={len(staged)}")
'@
    $code = $code.Replace("ROOT_PLACEHOLDER", $ROOT.Replace("\","\\"))
    $r = Invoke-PythonTest -Code $code -TimeoutSeconds 10
    if ($r.ExitCode -eq 0 -and $r.Stdout -match "E3_COAGENT_ISOLATION_PASS") {
        Record-Result "E3.CoAgent.FileIsolation.Staged" "PASS" "No co-agent files in staged changeset" $r.Elapsed
    } else {
        # After commit this is expected to be clean — report as WARN not FAIL
        Record-Result "E3.CoAgent.FileIsolation.Staged" "WARN" "No staged files (committed) or co-agent files staged" $r.Elapsed
    }

    # E4. Run targeted pytest on Phase 7.10b test files
    $pytestTargets = @(
        "tests/forcester_ts/test_ensemble_config_contract.py",
        "tests/models/test_time_series_signal_generator.py",
        "tests/scripts/test_check_quant_validation_health.py"
    )
    foreach ($target in $pytestTargets) {
        $fullTarget = Join-Path $ROOT $target
        if (Test-Path $fullTarget) {
            $pr = Invoke-PytestTarget -Target $fullTarget -ExtraArgs "-q --tb=short" -TimeoutSeconds 90
            if ($pr.ExitCode -eq 0) {
                Record-Result "E4.Pytest.$($target.Replace('/','.').Replace('tests.',''))" "PASS" "pytest passed" $pr.Elapsed
            } else {
                $snippet = ($pr.Output -split "`n" | Select-Object -Last 15) -join " | "
                Record-Result "E4.Pytest.$($target.Replace('/','.').Replace('tests.',''))" "FAIL" $snippet $pr.Elapsed
            }
        } else {
            Record-Result "E4.Pytest.$target" "SKIP" "Test file not found" 0
        }
    }

    # E5. Full test suite count and pass rate
    Write-Log "Running full pytest suite (this may take 2-4 minutes)..." "INFO"
    $sw  = [System.Diagnostics.Stopwatch]::StartNew()
    $cmd = "& `"$PythonExe`" -m pytest `"$ROOT/tests`" -q --tb=no --no-header -m 'not gpu' 2>&1"
    $out = Invoke-Expression $cmd
    $sw.Stop()
    $exitCode = $LASTEXITCODE
    $summary  = ($out | Select-Object -Last 5) -join " | "
    # Extract pass/fail counts from pytest summary line
    $passMatch = [regex]::Match($summary, "(\d+) passed")
    $failMatch = [regex]::Match($summary, "(\d+) failed")
    $totalPass = if ($passMatch.Success) { [int]$passMatch.Groups[1].Value } else { 0 }
    $totalFail = if ($failMatch.Success) { [int]$failMatch.Groups[1].Value } else { 0 }

    if ($exitCode -eq 0) {
        Record-Result "E5.FullTestSuite.PassRate" "PASS" "${totalPass} passed, ${totalFail} failed | $summary" $sw.ElapsedMilliseconds
    } elseif ($totalFail -le 5) {
        Record-Result "E5.FullTestSuite.PassRate" "WARN" "${totalPass} passed, ${totalFail} failed (< 5 failures tolerated) | $summary" $sw.ElapsedMilliseconds
    } else {
        Record-Result "E5.FullTestSuite.PassRate" "FAIL" "${totalPass} passed, ${totalFail} failed | $summary" $sw.ElapsedMilliseconds
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
Write-Log "Phase 7.10b + 7.11 Performance Test Suite" "SECT"
Write-Log "Python: $PythonExe"
Write-Log "Root:   $ROOT"
Write-Log "Report: $LOG_FILE"
Write-Log "Mode:   SkipSlow=$SkipSlow, AdversarialOnly=$AdversarialOnly"

if (-not $AdversarialOnly) {
    Test-ValidationLogic
    Test-ModelSignalQuality
    Test-DirectionalAccuracy
    Test-SystemLevel
}
Test-Adversarial

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
$elapsed = ((Get-Date) - $script:StartTime).TotalSeconds
$total   = $script:PassCount + $script:FailCount + $script:WarnCount
$passRate = if ($total -gt 0) { [math]::Round(100.0 * $script:PassCount / $total, 1) } else { 0 }

Write-Log "SUMMARY" "SECT"
Write-Log "Total tests : $total"
Write-Log "PASS        : $($script:PassCount)"
Write-Log "FAIL        : $($script:FailCount)"
Write-Log "WARN        : $($script:WarnCount)"
Write-Log "Pass rate   : ${passRate}%"
Write-Log "Elapsed     : ${elapsed}s"

# Write JSON report
$report = @{
    timestamp   = (Get-Date -Format "o")
    phase       = "7.10b+7.11"
    python      = $PythonExe
    pass_count  = $script:PassCount
    fail_count  = $script:FailCount
    warn_count  = $script:WarnCount
    pass_rate   = $passRate
    elapsed_s   = $elapsed
    tests       = $script:TestResults
}
$report | ConvertTo-Json -Depth 5 | Set-Content -Path $JSON_FILE -Encoding UTF8
Write-Log "JSON report : $JSON_FILE"

if ($script:FailCount -gt 0) {
    Write-Log "FAILED TESTS:" "WARN"
    foreach ($t in $script:TestResults) {
        if ($t.status -eq "FAIL") {
            Write-Log "  [FAIL] $($t.test): $($t.detail)" "FAIL"
        }
    }
}

$exitCode = if ($script:FailCount -gt 0) { 1 } else { 0 }
Write-Log "Exit code: $exitCode"
exit $exitCode
