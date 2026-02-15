<# 
PMX / OpenClaw Local LLM Setup (Windows, no paid keys required)

What it does:
- (Optional) Installs Ollama via winget
- (Optional) Starts Ollama and pulls a small recommended model set for RTX 4060 Ti 16GB class GPUs
- Configures OpenClaw model failover so it can use local Ollama models when available

Safe-by-default:
- Never prints secret values.
- Does NOT require OpenAI/Anthropic keys.

Run:
  powershell -ExecutionPolicy Bypass -File .\scripts\setup_openclaw_local_llm.ps1
  powershell -ExecutionPolicy Bypass -File .\scripts\setup_openclaw_local_llm.ps1 -InstallOllama -PullModels
#>

[CmdletBinding()]
param(
  [ValidateSet("auto", "local-first", "remote-first", "custom")]
  [string]$Strategy = "auto",

  [switch]$InstallOllama,
  [switch]$PullModels,

  [string]$OllamaHost = "",

  [switch]$SkipRestartGateway
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[setup_openclaw_local_llm] " + $Message) -ForegroundColor Cyan
}

function Write-Warn([string]$Message) {
  Write-Host ("[setup_openclaw_local_llm] WARN: " + $Message) -ForegroundColor Yellow
}

function Find-RepoRoot([string]$StartDir) {
  $dir = Resolve-Path -LiteralPath $StartDir
  while ($true) {
    if (Test-Path -LiteralPath (Join-Path $dir "scripts\\openclaw_models.py")) { return $dir }
    $parent = Split-Path -Parent $dir
    if (-not $parent -or $parent -eq $dir) { break }
    $dir = $parent
  }
  throw "Repo root not found. cd into the repo (folder that contains scripts\\openclaw_models.py) and re-run."
}

function Get-PythonExe([string]$RepoRoot) {
  $venvPy = Join-Path $RepoRoot "simpleTrader_env\\Scripts\\python.exe"
  if (Test-Path -LiteralPath $venvPy) { return $venvPy }
  return "python"
}

function Get-OllamaExe() {
  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }

  $candidates = @(
    (Join-Path ${env:ProgramFiles} "Ollama\\ollama.exe"),
    (Join-Path ${env:LOCALAPPDATA} "Programs\\Ollama\\ollama.exe")
  )
  foreach ($p in $candidates) {
    if ($p -and (Test-Path -LiteralPath $p)) { return $p }
  }
  return $null
}

function Ensure-OllamaInstalled() {
  $exe = Get-OllamaExe
  if ($exe) { return $exe }

  if (-not $InstallOllama) {
    Write-Warn "Ollama not found on PATH. Skipping install (pass -InstallOllama to auto-install)."
    return $null
  }

  if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Warn "winget not found. Install Ollama from: https://ollama.com/download/windows"
    return $null
  }

  Write-Step "Installing Ollama via winget (this may prompt you)..."
  winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements | Out-Host

  # Re-check
  Start-Sleep -Seconds 2
  return (Get-OllamaExe)
}

function Test-OllamaApi([string]$Host) {
  try {
    Invoke-RestMethod -Uri ($Host.TrimEnd("/") + "/api/tags") -TimeoutSec 3 | Out-Null
    return $true
  } catch {
    return $false
  }
}

function Start-OllamaIfNeeded([string]$OllamaExePath, [string]$Host) {
  if (-not $OllamaExePath) { return }

  if (Test-OllamaApi $Host) {
    Write-Step "Ollama API reachable at $Host"
    return
  }

  Write-Step "Starting Ollama (background): ollama serve"
  Start-Process -FilePath $OllamaExePath -ArgumentList "serve" -WindowStyle Hidden | Out-Null
  Start-Sleep -Seconds 2

  if (Test-OllamaApi $Host) {
    Write-Step "Ollama API reachable at $Host"
  } else {
    Write-Warn "Ollama API still not reachable at $Host. You may need to start it manually: ollama serve"
  }
}

function Pull-RecommendedModels([string]$OllamaExePath) {
  if (-not $OllamaExePath) { return }
  if (-not $PullModels) {
    Write-Step "Skipping model downloads (pass -PullModels to pull recommended models)."
    return
  }

  $models = @(
    "qwen:14b-chat-q4_K_M",
    "deepseek-coder:6.7b-instruct-q4_K_M",
    "codellama:13b-instruct-q4_K_M"
  )

  foreach ($m in $models) {
    Write-Step ("ollama pull " + $m)
    & $OllamaExePath "pull" $m | Out-Host
  }
}

# --- main ---
$repoRoot = Find-RepoRoot (Get-Location).Path
Set-Location $repoRoot
Write-Step ("Repo root: " + $repoRoot)

if (-not (Get-Command openclaw -ErrorAction SilentlyContinue)) {
  throw "openclaw CLI not found on PATH. Install it first (npm): npm i -g openclaw"
}

$py = Get-PythonExe $repoRoot
Write-Step ("Python: " + $py)

$resolvedOllamaHost = $OllamaHost
if (-not $resolvedOllamaHost -or $resolvedOllamaHost.Trim().Length -eq 0) {
  if ($env:OPENCLAW_OLLAMA_BASE_URL) {
    $resolvedOllamaHost = $env:OPENCLAW_OLLAMA_BASE_URL
  } elseif ($env:OLLAMA_HOST) {
    $resolvedOllamaHost = $env:OLLAMA_HOST
  } else {
    $resolvedOllamaHost = "http://127.0.0.1:11434"
  }
}

# OpenClaw `models.providers.ollama.baseUrl` is typically configured with `/v1`.
$openclawOllamaBaseUrl = $resolvedOllamaHost.TrimEnd("/")
if (-not $openclawOllamaBaseUrl.ToLower().EndsWith("/v1")) {
  $openclawOllamaBaseUrl = $openclawOllamaBaseUrl + "/v1"
}

# OpenClaw expects baseUrl typically with /v1, but our health check uses native endpoints.
$ollamaHealthHost = $openclawOllamaBaseUrl
if ($ollamaHealthHost.ToLower().EndsWith("/v1")) {
  $ollamaHealthHost = $ollamaHealthHost.Substring(0, $ollamaHealthHost.Length - 3)
}

# Ensure Python helper sees it (OpenClaw itself will read from config after apply).
$env:OPENCLAW_OLLAMA_BASE_URL = $openclawOllamaBaseUrl

$ollamaExe = Ensure-OllamaInstalled
if ($ollamaExe) {
  Write-Step ("Ollama: " + $ollamaExe)
  Start-OllamaIfNeeded -OllamaExePath $ollamaExe -Host $ollamaHealthHost
  Pull-RecommendedModels -OllamaExePath $ollamaExe
} else {
  Write-Warn "Proceeding without Ollama installed. OpenClaw will use remote model primary and keep local fallbacks configured for later."
}

Write-Step ("Configuring OpenClaw model failover (strategy=" + $Strategy + ")")
if (-not $SkipRestartGateway) {
  & $py scripts/openclaw_models.py apply --strategy $Strategy --restart-gateway | Out-Host
} else {
  & $py scripts/openclaw_models.py apply --strategy $Strategy | Out-Host
}

Write-Step "Status:"
& $py scripts/openclaw_models.py status | Out-Host

Write-Step "Test prompting:"
Write-Host "  python scripts/openclaw_notify.py --prompt --message 'Summarize latest run'"
