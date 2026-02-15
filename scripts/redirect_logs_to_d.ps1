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

function Get-DriveRoot([string]$Path) {
  try {
    return [System.IO.Path]::GetPathRoot($Path)
  } catch {
    return $null
  }
}

function Drive-Available([string]$Path) {
  $root = Get-DriveRoot $Path
  if (-not $root) { return $true }  # Relative paths, treat as local.
  return (Test-Path -LiteralPath $root)
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

function Ensure-Physical-Dir([string]$Path) {
  # If $Path is a junction (e.g., pointing to D:), rename it out of the way so new writes land locally.
  $parent = Split-Path -Parent $Path
  if ($parent) { Ensure-Dir $parent }

  if (Is-Junction $Path) {
    $stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
    $leaf = Split-Path -Leaf $Path
    $legacyName = ($leaf + "_junction_legacy_" + $stamp)
    $legacyPath = if ($parent) { Join-Path $parent $legacyName } else { $legacyName }

    if ($DryRun) {
      Write-Step ("DRY-RUN rename junction " + $Path + " -> " + $legacyPath)
    } else {
      try {
        Rename-Item -LiteralPath $Path -NewName $legacyName -ErrorAction Stop
        Write-Step ("Renamed junction to: " + $legacyPath)
      } catch {
        # Best-effort fallback: remove the junction itself (not the target).
        Write-Step ("WARN could not rename junction; attempting rmdir: " + $Path + " (" + $_.Exception.Message + ")")
        cmd.exe /c ("rmdir """ + $Path + """") | Out-Null
      }
    }
  }

  Ensure-Dir $Path
}

function Move-Dir-Contents([string]$From, [string]$To) {
  Ensure-Dir $To
  if (-not (Test-Path -LiteralPath $From)) { return }

  $items = Get-ChildItem -LiteralPath $From -Force -ErrorAction SilentlyContinue
  if (-not $items) { return }

  if ($DryRun) {
    Write-Step ("DRY-RUN move contents: " + $From + " -> " + $To)
    return @()
  }

  $failed = @()
  foreach ($it in $items) {
    try {
      Move-Item -LiteralPath $it.FullName -Destination $To -Force -ErrorAction Stop
    } catch {
      $failed += $it.FullName
      Write-Step ("WARN could not move: " + $it.FullName + " (" + $_.Exception.Message + ")")
    }
  }
  return $failed
}

function Replace-With-Junction([string]$LinkPath, [string]$TargetPath) {
  Ensure-Dir $TargetPath

  if (Is-Junction $LinkPath) {
    Write-Step ("Already a junction: " + $LinkPath)
    return
  }

  if (Test-Path -LiteralPath $LinkPath) {
    # Move contents out first (best-effort), then remove dir.
    $failedMoves = @(Move-Dir-Contents -From $LinkPath -To $TargetPath)

    if ($DryRun) {
      Write-Step ("DRY-RUN remove dir " + $LinkPath)
    } else {
      try {
        Remove-Item -LiteralPath $LinkPath -Recurse -Force
      } catch {
        # If something is still holding files open, try renaming the directory out of the way,
        # then create the junction for new writes.
        $stamp = (Get-Date).ToString("yyyyMMdd_HHmmss")
        $leaf = Split-Path -Leaf $LinkPath
        $parent = Split-Path -Parent $LinkPath
        $legacyName = ($leaf + "_legacy_" + $stamp)
        $legacyPath = Join-Path $parent $legacyName

        try {
          Rename-Item -LiteralPath $LinkPath -NewName $legacyName -ErrorAction Stop
          Write-Step ("Renamed locked dir to: " + $legacyPath)
        } catch {
          $msg = "Failed to remove or rename '" + $LinkPath + "'. Some files may be in use by another process."
          if ($failedMoves.Count -gt 0) {
            $msg += " Example locked item: " + $failedMoves[0]
          }
          $msg += " Stop the process using the log file(s) and re-run. Underlying error: " + $_
          throw $msg
        }
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
if (Drive-Available $PmxLogsTarget) {
  try {
    Replace-With-Junction -LinkPath $pmxLogsLink -TargetPath $PmxLogsTarget
  } catch {
    Write-Step ("WARN redirect failed; keeping logs on C: " + $_.Exception.Message)
    Ensure-Physical-Dir $pmxLogsLink
  }
} else {
  Write-Step ("WARN target drive not available for PMX logs (" + (Get-DriveRoot $PmxLogsTarget) + "); keeping logs on C")
  Ensure-Physical-Dir $pmxLogsLink
}

# 2) OpenClaw default log dir (seen in gateway output as \\tmp\\openclaw)
$openclawLogsLink = "C:\tmp\openclaw"
if (Drive-Available $OpenClawLogsTarget) {
  try {
    Replace-With-Junction -LinkPath $openclawLogsLink -TargetPath $OpenClawLogsTarget
  } catch {
    Write-Step ("WARN redirect failed; keeping OpenClaw logs on C: " + $_.Exception.Message)
    Ensure-Physical-Dir $openclawLogsLink
  }
} else {
  Write-Step ("WARN target drive not available for OpenClaw logs (" + (Get-DriveRoot $OpenClawLogsTarget) + "); keeping logs on C")
  Ensure-Physical-Dir $openclawLogsLink
}

Write-Step "Done."
