# Redirect logs to D: using NTFS junctions.
#
# What it does:
# - Moves the repo `logs/` directory to D: and replaces it with a junction.
# - Moves OpenClaw's default log directory `C:\tmp\openclaw` to D: and replaces it with a junction.
#
# Why junctions:
# - Requires no code changes across the repo; anything writing to `logs/` keeps working.
# - OpenClaw keeps writing to `C:\tmp\openclaw` but the bytes land on D:.
#
# Safety:
# - Never prints or touches `.env` values.
# - Use `-DryRun` first to preview actions.

param(
  [string]$PmxLogsTarget = "D:\pmx\logs",
  [string]$OpenClawLogsTarget = "D:\pmx\openclaw\tmp\openclaw",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[redirect_logs_to_d] " + $Message) -ForegroundColor Cyan
}

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    if ($DryRun) {
      Write-Step ("DRY-RUN mkdir " + $Path)
    } else {
      New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
  }
}

function Is-Junction([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return $false }
  $item = Get-Item -LiteralPath $Path -Force
  return [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

function Move-Dir-Contents([string]$From, [string]$To) {
  Ensure-Dir $To
  if (-not (Test-Path -LiteralPath $From)) { return }

  $items = Get-ChildItem -LiteralPath $From -Force -ErrorAction SilentlyContinue
  if (-not $items) { return }

  if ($DryRun) {
    Write-Step ("DRY-RUN move contents: " + $From + " -> " + $To)
    return
  }

  foreach ($it in $items) {
    Move-Item -LiteralPath $it.FullName -Destination $To -Force -ErrorAction Continue
  }
}

function Replace-With-Junction([string]$LinkPath, [string]$TargetPath) {
  Ensure-Dir $TargetPath

  if (Is-Junction $LinkPath) {
    Write-Step ("Already a junction: " + $LinkPath)
    return
  }

  if (Test-Path -LiteralPath $LinkPath) {
    # Move contents out first (best-effort), then remove dir.
    Move-Dir-Contents -From $LinkPath -To $TargetPath

    if ($DryRun) {
      Write-Step ("DRY-RUN remove dir " + $LinkPath)
    } else {
      try {
        Remove-Item -LiteralPath $LinkPath -Recurse -Force
      } catch {
        throw ("Failed to remove '" + $LinkPath + "'. If OpenClaw is running, stop it and retry. Underlying error: " + $_)
      }
    }
  }

  $parent = Split-Path -Parent $LinkPath
  Ensure-Dir $parent

  if ($DryRun) {
    Write-Step ("DRY-RUN junction " + $LinkPath + " -> " + $TargetPath)
    return
  }

  New-Item -ItemType Junction -Path $LinkPath -Target $TargetPath | Out-Null
  Write-Step ("Created junction " + $LinkPath + " -> " + $TargetPath)
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Step ("Repo root: " + $repoRoot)
Write-Step ("PMX logs target: " + $PmxLogsTarget)
Write-Step ("OpenClaw logs target: " + $OpenClawLogsTarget)

# 1) PMX repo logs/
$pmxLogsLink = Join-Path $repoRoot "logs"
Replace-With-Junction -LinkPath $pmxLogsLink -TargetPath $PmxLogsTarget

# 2) OpenClaw default log dir (seen in gateway output as \\tmp\\openclaw)
$openclawLogsLink = "C:\tmp\openclaw"
Replace-With-Junction -LinkPath $openclawLogsLink -TargetPath $OpenClawLogsTarget

Write-Step "Done."
