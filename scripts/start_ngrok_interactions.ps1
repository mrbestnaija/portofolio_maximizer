<#
Expose the local PMX Interactions API via ngrok (testing only).

What it does:
- Starts the local FastAPI server (scripts/pmx_interactions_api.py) if not already running
- Starts ngrok and discovers the public https URL via the local ngrok API
- Updates repo .env with:
  - INTERACTIONS_ENDPOINT_URL=https://<ngrok>/interactions
  - LINKED_ROLES_VERIFICATION_URL=https://<ngrok>/verify-roles

Security defaults:
- Refuses to run when CI=true or PMX_ENV=production unless -Force is provided.
- Refuses to run unless INTERACTIONS_API_KEY (strong) or Auth0 JWT config is present (avoid exposing an unauthenticated endpoint).
- Never prints .env secret values.

Run:
  powershell -ExecutionPolicy Bypass -File .\scripts\start_ngrok_interactions.ps1

Optional:
  powershell -ExecutionPolicy Bypass -File .\scripts\start_ngrok_interactions.ps1 -InstallNgrok
  powershell -ExecutionPolicy Bypass -File .\scripts\start_ngrok_interactions.ps1 -Port 8000 -DryRun
#>

[CmdletBinding()]
param(
  [int]$Port = 8000,
  [string]$Host = "127.0.0.1",

  [switch]$InstallNgrok,
  [switch]$DryRun,
  [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[start_ngrok_interactions] " + $Message) -ForegroundColor Cyan
}

function Write-Warn([string]$Message) {
  Write-Host ("[start_ngrok_interactions] WARN: " + $Message) -ForegroundColor Yellow
}

function Find-RepoRoot([string]$StartDir) {
  $dir = Resolve-Path -LiteralPath $StartDir
  while ($true) {
    if (Test-Path -LiteralPath (Join-Path $dir "scripts\\pmx_interactions_api.py")) { return $dir }
    $parent = Split-Path -Parent $dir
    if (-not $parent -or $parent -eq $dir) { break }
    $dir = $parent
  }
  throw "Repo root not found. cd into the repo and re-run."
}

function Get-PythonExe([string]$RepoRoot) {
  $venvPy = Join-Path $RepoRoot "simpleTrader_env\\Scripts\\python.exe"
  if (Test-Path -LiteralPath $venvPy) { return $venvPy }
  return "python"
}

function Is-ProductionEnv() {
  $ci = ($env:CI -as [string])
  if ($ci -and $ci.Trim().ToLower() -in @("1","true","yes","y","on")) { return $true }
  $pmxEnv = ($env:PMX_ENV -as [string])
  if (-not $pmxEnv) { $pmxEnv = ($env:ENV -as [string]) }
  if (-not $pmxEnv) { $pmxEnv = "local" }
  $pmxEnv = $pmxEnv.Trim().ToLower()
  return $pmxEnv -in @("prod","production","live")
}

function Ensure-NgrokInstalled() {
  $cmd = Get-Command ngrok -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }

  if (-not $InstallNgrok) {
    throw "ngrok not found on PATH. Install it (or pass -InstallNgrok)."
  }

  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "winget not found. Install ngrok from: https://ngrok.com/download"
  }

  Write-Step "Installing ngrok via winget..."
  winget install --id Ngrok.Ngrok -e --accept-package-agreements --accept-source-agreements | Out-Host
  Start-Sleep -Seconds 2

  $cmd2 = Get-Command ngrok -ErrorAction SilentlyContinue
  if (-not $cmd2) { throw "ngrok still not found after install. Re-open your shell and retry." }
  return $cmd2.Source
}

function Ensure-AuthConfigured([string]$RepoRoot, [string]$PythonExe) {
  $code = @"
import sys
import os
from pathlib import Path
sys.path.insert(0, r'$RepoRoot')
from etl.secret_loader import load_secret
key = (load_secret('INTERACTIONS_API_KEY') or '').strip()
key_l = key.lower()
bad = key_l in {'your_interactions_api_key_here','your_api_key_here','changeme','change_me','replace_me','replace-me','todo'} or (key_l.startswith('your_') and key_l.endswith('_here'))
min_len = int(os.getenv('INTERACTIONS_MIN_KEY_LENGTH') or '16')
key_ok = bool(key and (len(key) >= max(16, min_len)) and (not bad))

domain = (os.getenv('AUTH0_DOMAIN') or '').strip()
aud = (os.getenv('AUTH0_AUDIENCE') or '').strip()
jwt_ok = bool(domain and aud)

auth_mode = (os.getenv('INTERACTIONS_AUTH_MODE') or 'any').strip().lower()

if auth_mode == 'jwt-only':
    if not jwt_ok:
        print('[ERROR] INTERACTIONS_AUTH_MODE=jwt-only but AUTH0_DOMAIN/AUTH0_AUDIENCE not set.')
        sys.exit(2)
    sys.exit(0)
elif auth_mode == 'api-key-only':
    if not key_ok:
        print('[ERROR] INTERACTIONS_AUTH_MODE=api-key-only but INTERACTIONS_API_KEY not set or too short.')
        sys.exit(2)
    sys.exit(0)
else:
    sys.exit(0 if (key_ok or jwt_ok) else 2)
"@
  $p = Start-Process -FilePath $PythonExe -ArgumentList @("-c", $code) -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0) {
    $mode = if ($env:INTERACTIONS_AUTH_MODE) { $env:INTERACTIONS_AUTH_MODE } else { "any" }
    throw "No auth configured (mode=$mode). Set INTERACTIONS_API_KEY (>=16 chars, not a template placeholder) or AUTH0_DOMAIN + AUTH0_AUDIENCE before exposing endpoints publicly."
  }
}

function Upsert-DotenvVar([string]$EnvPath, [string]$Key, [string]$Value) {
  if (-not (Test-Path -LiteralPath $EnvPath)) { throw "Missing .env at: $EnvPath" }
  $lines = Get-Content -LiteralPath $EnvPath -ErrorAction Stop
  $found = $false
  $out = New-Object System.Collections.Generic.List[string]
  foreach ($line in $lines) {
    if ($line -match "^\s*$Key\s*=") {
      $out.Add("$Key=$Value")
      $found = $true
    } else {
      $out.Add($line)
    }
  }
  if (-not $found) {
    $out.Add("")
    $out.Add("$Key=$Value")
  }
  if ($DryRun) {
    Write-Step ("DRY-RUN set " + $Key)
    return
  }
  Set-Content -LiteralPath $EnvPath -Value $out -Encoding UTF8 -NoNewline:$false
}

function Start-Server([string]$RepoRoot, [string]$PythonExe, [int]$Port, [string]$Host) {
  $env:INTERACTIONS_PORT = [string]$Port
  $env:INTERACTIONS_BIND_HOST = [string]$Host
  # Keep server logs in the current terminal via -NoNewWindow.
  Write-Step ("Starting PMX interactions server on http://" + $Host + ":" + $Port)
  Start-Process -FilePath $PythonExe -ArgumentList @("scripts\\pmx_interactions_api.py") -WorkingDirectory $RepoRoot -NoNewWindow | Out-Null
  Start-Sleep -Seconds 1
}

function Start-Ngrok([string]$NgrokExe, [int]$Port) {
  Write-Step ("Starting ngrok tunnel to http://127.0.0.1:" + $Port)
  Start-Process -FilePath $NgrokExe -ArgumentList @("http", ("http://127.0.0.1:" + $Port)) -NoNewWindow | Out-Null
}

function Wait-ForNgrokUrl() {
  $api = "http://127.0.0.1:4040/api/tunnels"
  $deadline = (Get-Date).AddSeconds(15)
  while ((Get-Date) -lt $deadline) {
    try {
      $obj = Invoke-RestMethod -Uri $api -TimeoutSec 2
      if ($obj -and $obj.tunnels) {
        foreach ($t in $obj.tunnels) {
          if ($t.public_url -and $t.public_url.ToString().StartsWith("https://")) {
            return $t.public_url.ToString().TrimEnd("/")
          }
        }
      }
    } catch { }
    Start-Sleep -Milliseconds 500
  }
  throw "Could not discover ngrok public URL (is ngrok running? check http://127.0.0.1:4040)."
}

# --- main ---
$repoRoot = Find-RepoRoot (Get-Location).Path
Set-Location $repoRoot
Write-Step ("Repo root: " + $repoRoot)

if ((Is-ProductionEnv) -and (-not $Force)) {
  throw "Refusing to run in production/CI environment. Set PMX_ENV=local or pass -Force."
}

$py = Get-PythonExe $repoRoot
Write-Step ("Python: " + $py)

Ensure-AuthConfigured -RepoRoot $repoRoot -PythonExe $py

$authMode = if ($env:INTERACTIONS_AUTH_MODE) { $env:INTERACTIONS_AUTH_MODE.Trim().ToLower() } else { "any" }
Write-Step ("Auth mode: " + $authMode)
if ($authMode -eq "jwt-only") {
  Write-Step "JWT-only mode: API key auth will be rejected by the server."
} elseif ($authMode -eq "api-key-only") {
  Write-Step "API-key-only mode: JWT auth will be rejected by the server."
}

$ngrokExe = Ensure-NgrokInstalled
Write-Step ("ngrok: " + $ngrokExe)

Start-Server -RepoRoot $repoRoot -PythonExe $py -Port $Port -Host $Host
Start-Ngrok -NgrokExe $ngrokExe -Port $Port

$public = Wait-ForNgrokUrl
$interactionsUrl = $public + "/interactions"
$verifyUrl = $public + "/verify-roles"

Write-Step ("Public URL: " + $public)
Write-Step ("INTERACTIONS_ENDPOINT_URL: " + $interactionsUrl)
Write-Step ("LINKED_ROLES_VERIFICATION_URL: " + $verifyUrl)

$envPath = Join-Path $repoRoot ".env"
Upsert-DotenvVar -EnvPath $envPath -Key "INTERACTIONS_ENDPOINT_URL" -Value $interactionsUrl
Upsert-DotenvVar -EnvPath $envPath -Key "LINKED_ROLES_VERIFICATION_URL" -Value $verifyUrl

Write-Step "Done."
