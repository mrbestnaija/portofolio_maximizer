# setup_remote_workflow.ps1
# -------------------------
# One-click setup for OpenClaw v2026.3.13 remote workflow.
# Run from the portfolio_maximizer_v45 repo root.
#
# What this does:
#   1. Verifies openclaw CLI is available
#   2. Restarts the gateway (now in remote/lan mode)
#   3. Logs into Telegram channel (if not already linked)
#   4. Verifies WhatsApp session is still active
#   5. Starts the PMX Interactions API for webhook access
#   6. Runs remote workflow status check
#   7. Optionally launches ngrok tunnel for external access
#
# Usage:
#   .\bash\setup_remote_workflow.ps1
#   .\bash\setup_remote_workflow.ps1 -WithNgrok           # also start ngrok
#   .\bash\setup_remote_workflow.ps1 -SkipChannelLogin    # skip interactive login
#   .\bash\setup_remote_workflow.ps1 -StatusOnly          # only run status check

param(
    [switch]$WithNgrok,
    [switch]$SkipChannelLogin,
    [switch]$StatusOnly
)

$ErrorActionPreference = "Continue"
$PythonExe = ".\simpleTrader_env\Scripts\python.exe"
$OpenclawBin = "openclaw"

function Step($msg) { Write-Host "`n[STEP] $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  [OK] $msg"   -ForegroundColor Green }
function Warn($msg) { Write-Host "  [!]  $msg"   -ForegroundColor Yellow }
function Fail($msg) { Write-Host "  [X]  $msg"   -ForegroundColor Red }

Step "Remote Workflow Setup — OpenClaw v2026.3.13"

# ---------------------------------------------------------------------------
# Status only
# ---------------------------------------------------------------------------
if ($StatusOnly) {
    & $PythonExe scripts\openclaw_remote_workflow.py status
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# 1. Check openclaw CLI
# ---------------------------------------------------------------------------
Step "1/7  Verifying openclaw CLI"
$clawPath = Get-Command $OpenclawBin -ErrorAction SilentlyContinue
if (-not $clawPath) {
    Fail "openclaw not found in PATH"
    Write-Host "     Install: https://openclaw.io/docs/install" -ForegroundColor Gray
    exit 2
}
Ok "openclaw found: $($clawPath.Source)"

# ---------------------------------------------------------------------------
# 2. Restart gateway
# ---------------------------------------------------------------------------
Step "2/7  Restarting gateway (remote/lan mode)"
& $OpenclawBin gateway restart 2>&1 | Select-Object -Last 5
Start-Sleep -Seconds 5

# Verify gateway responds
$gatewayUp = $false
for ($i = 1; $i -le 6; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:18789/health" -TimeoutSec 4 -UseBasicParsing -ErrorAction Stop
        $gatewayUp = $true
        Ok "Gateway responding (attempt $i)"
        break
    } catch {
        Warn "Gateway not yet up (attempt $i/6)..."
        Start-Sleep -Seconds 3
    }
}
if (-not $gatewayUp) {
    Warn "Gateway health endpoint not reachable — proceeding anyway (may be auth-gated)"
}

# ---------------------------------------------------------------------------
# 3. Telegram channel login
# ---------------------------------------------------------------------------
Step "3/7  Telegram channel"
if ($SkipChannelLogin) {
    Warn "Skipping channel login (--SkipChannelLogin)"
} else {
    $tgStatus = & $OpenclawBin channels status --channel telegram --json 2>&1
    if ($LASTEXITCODE -eq 0) {
        Ok "Telegram channel already linked"
    } else {
        Write-Host "  Linking Telegram bot (token already in openclaw.json)..." -ForegroundColor Gray
        & $OpenclawBin channels login --channel telegram --account default --verbose
        if ($LASTEXITCODE -eq 0) {
            Ok "Telegram linked"
        } else {
            Warn "Telegram link failed — check bot token in openclaw.json"
        }
    }
}

# ---------------------------------------------------------------------------
# 4. WhatsApp session check
# ---------------------------------------------------------------------------
Step "4/7  WhatsApp session"
$waStatus = & $OpenclawBin channels status --channel whatsapp --json 2>&1
if ($LASTEXITCODE -eq 0) {
    Ok "WhatsApp session active"
} else {
    Warn "WhatsApp session may be inactive"
    Write-Host "  To relink: openclaw channels login --channel whatsapp --account default" -ForegroundColor Gray
}

# ---------------------------------------------------------------------------
# 5. PMX Interactions API
# ---------------------------------------------------------------------------
Step "5/7  PMX Interactions API"
try {
    $apiResp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
    Ok "Interactions API already running on :8000"
} catch {
    Write-Host "  Starting Interactions API in background..." -ForegroundColor Gray
    $apiJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        & ".\simpleTrader_env\Scripts\python.exe" scripts\pmx_interactions_api.py 2>&1
    }
    Start-Sleep -Seconds 4
    try {
        $apiResp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        Ok "Interactions API started (job: $($apiJob.Id))"
    } catch {
        Warn "Interactions API did not start — check INTERACTIONS_API_KEY in .env"
        Write-Host "  Manual: $PythonExe scripts\pmx_interactions_api.py" -ForegroundColor Gray
    }
}

# ---------------------------------------------------------------------------
# 6. Remote workflow status check
# ---------------------------------------------------------------------------
Step "6/7  Remote workflow status"
& $PythonExe scripts\openclaw_remote_workflow.py status

# ---------------------------------------------------------------------------
# 7. ngrok (optional)
# ---------------------------------------------------------------------------
Step "7/7  ngrok tunnel"
if ($WithNgrok) {
    $ngrokPath = Get-Command ngrok -ErrorAction SilentlyContinue
    if (-not $ngrokPath) {
        Warn "ngrok not found in PATH — install from https://ngrok.com/download"
    } else {
        Write-Host "  Starting ngrok tunnel for :8000..." -ForegroundColor Gray
        Start-Process -NoNewWindow -FilePath "ngrok" -ArgumentList "http", "8000"
        Start-Sleep -Seconds 3
        # Fetch public URL
        try {
            $tunnels = Invoke-WebRequest -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 5 -UseBasicParsing | ConvertFrom-Json
            $publicUrl = $tunnels.tunnels[0].public_url
            Ok "ngrok tunnel: $publicUrl"
            Write-Host "  Interactions API public URL: $publicUrl/interactions" -ForegroundColor Gray
            Write-Host "  Set in .env:  INTERACTIONS_ENDPOINT_URL=$publicUrl/interactions" -ForegroundColor Gray
        } catch {
            Warn "Could not read ngrok public URL — check http://localhost:4040"
        }
    }
} else {
    Write-Host "  Skip (use -WithNgrok to start ngrok tunnel)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "[DONE] Remote workflow setup complete." -ForegroundColor Green
Write-Host "  Failover check: $PythonExe scripts\openclaw_remote_workflow.py failover-test" -ForegroundColor Gray
Write-Host "  Cron health:    $PythonExe scripts\openclaw_remote_workflow.py cron-health" -ForegroundColor Gray
Write-Host "  Full diagnose:  $PythonExe scripts\openclaw_remote_workflow.py diagnose" -ForegroundColor Gray
