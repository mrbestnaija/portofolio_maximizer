Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message)
    Write-Output ("[observability] " + $Message)
}

function Resolve-RepoRoot {
    return (Split-Path -Parent $PSScriptRoot)
}

function Resolve-PreferredPython {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot
    )

    $candidates = @()
    if ($env:PMX_OBSERVABILITY_PYTHON) { $candidates += $env:PMX_OBSERVABILITY_PYTHON }
    $candidates += (Join-Path $RepoRoot "simpleTrader_env\Scripts\python.exe")
    $candidates += "python"

    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        if (Test-Path $candidate) { return $candidate }
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }

    throw "Unable to locate a usable Python interpreter for observability startup."
}

function Resolve-RequiredPath {
    param(
        [Parameter(Mandatory = $true)][string[]]$Candidates,
        [Parameter(Mandatory = $true)][string]$Label
    )

    foreach ($candidate in $Candidates) {
        if (-not $candidate) { continue }
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }

    throw "Unable to locate $Label. Checked: $($Candidates -join ', ')"
}

function Normalize-PathCandidates {
    param(
        [AllowEmptyCollection()][string[]]$Candidates
    )

    $normalized = @()
    foreach ($candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($candidate)) { continue }
        $normalized += $candidate
    }
    return $normalized
}

function Test-HttpHealthy {
    param(
        [string]$Url,
        [int]$TimeoutSec = 5
    )

    if ([string]::IsNullOrWhiteSpace($Url)) {
        return $false
    }

    try {
        $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec $TimeoutSec -Uri $Url
        return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300)
    }
    catch {
        return $false
    }
}

function Wait-HttpHealthy {
    param(
        [string]$Url,
        [int]$TimeoutSec = 20,
        [int]$PollMs = 500
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSec)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-HttpHealthy -Url $Url -TimeoutSec 3) {
            return $true
        }
        Start-Sleep -Milliseconds $PollMs
    }
    return $false
}

function Wait-PortClosed {
    param(
        [int]$Port,
        [int]$TimeoutSec = 10,
        [int]$PollMs = 250
    )

    if ($Port -le 0) {
        return $true
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSec)
    while ([DateTime]::UtcNow -lt $deadline) {
        $processIds = @(Get-ListeningProcessIdsByPort -Port $Port)
        if ($processIds.Count -eq 0) {
            return $true
        }
        Start-Sleep -Milliseconds $PollMs
    }
    return (@(Get-ListeningProcessIdsByPort -Port $Port).Count -eq 0)
}

function Request-LocalShutdown {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSec = 5
    )

    try {
        $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec $TimeoutSec -Method Post -Uri $Url
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
            Write-Status ("shutdown_requested " + $Label)
            return $true
        }
    }
    catch {
        Write-Status ("shutdown_request_skipped " + $Label + " error=" + $_.Exception.Message)
    }
    return $false
}

function Get-ObservedProcessIds {
    param(
        [string]$ExecutablePath = "",
        [string]$ExecutablePathLike = "",
        [string]$CommandLineMatch = ""
    )

    $resolvedExecutablePath = ""
    if (-not [string]::IsNullOrWhiteSpace($ExecutablePath)) {
        $resolved = Resolve-Path $ExecutablePath -ErrorAction SilentlyContinue
        $resolvedExecutablePath = if ($resolved) { $resolved.Path } else { $ExecutablePath }
    }

    $results = @(
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
            $keep = $true

            if ($resolvedExecutablePath) {
                $keep = $keep -and $_.ExecutablePath -and $_.ExecutablePath -ieq $resolvedExecutablePath
            }
            if ($ExecutablePathLike) {
                $keep = $keep -and $_.ExecutablePath -and $_.ExecutablePath -like $ExecutablePathLike
            }
            if ($CommandLineMatch) {
                $keep = $keep -and $_.CommandLine -and $_.CommandLine -match [Regex]::Escape($CommandLineMatch)
            }

            $keep
        } | Select-Object -ExpandProperty ProcessId
    )

    return @($results | Sort-Object -Unique)
}

function Get-ListeningProcessIdsByPort {
    param([int]$Port)

    if ($Port -le 0) {
        return @()
    }

    try {
        $results = @(
            Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction Stop |
                Select-Object -ExpandProperty OwningProcess
        )
        return @($results | Sort-Object -Unique)
    }
    catch {
        return @()
    }
}

function Start-RepoProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StdOutPath,
        [Parameter(Mandatory = $true)][string]$StdErrPath,
        [switch]$Foreground
    )

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StdOutPath) | Out-Null

    if ($Foreground) {
        & $FilePath @ArgumentList
        return $LASTEXITCODE
    }

    $proc = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdOutPath `
        -RedirectStandardError $StdErrPath `
        -PassThru
    return $proc.Id
}

function Ensure-RepoService {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory,
        [Parameter(Mandatory = $true)][string]$StdOutPath,
        [Parameter(Mandatory = $true)][string]$StdErrPath,
        [string]$HealthUrl = "",
        [AllowEmptyCollection()][int[]]$ExistingProcessIds = @(),
        [int]$StartupTimeoutSec = 20,
        [switch]$Foreground
    )

    if ($Foreground) {
        Start-RepoProcess `
            -FilePath $FilePath `
            -ArgumentList $ArgumentList `
            -WorkingDirectory $WorkingDirectory `
            -StdOutPath $StdOutPath `
            -StdErrPath $StdErrPath `
            -Foreground:$true
        return
    }

    $processIds = @($ExistingProcessIds | Where-Object { $_ })

    if ($HealthUrl -and (Test-HttpHealthy -Url $HealthUrl)) {
        if ($processIds.Count -gt 0) {
            Write-Status ("already_healthy " + $Label + " pid=" + ($processIds -join ","))
        }
        else {
            Write-Status ("already_healthy " + $Label)
        }
        return
    }

    if ($processIds.Count -gt 0) {
        Write-Status ("already_running " + $Label + " pid=" + ($processIds -join ","))
        return
    }

    $procId = Start-RepoProcess `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -StdOutPath $StdOutPath `
        -StdErrPath $StdErrPath

    if ($HealthUrl) {
        if (Wait-HttpHealthy -Url $HealthUrl -TimeoutSec $StartupTimeoutSec) {
            Write-Status ("started " + $Label + " pid=" + $procId)
        }
        else {
            Write-Status ("started_unverified " + $Label + " pid=" + $procId)
        }
    }
    else {
        Write-Status ("started " + $Label + " pid=" + $procId)
    }
}

function Stop-ObservedProcesses {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [AllowEmptyCollection()][int[]]$ProcessIds = @()
    )

    $ids = @($ProcessIds | Where-Object { $_ } | Sort-Object -Unique)
    if ($ids.Count -eq 0) {
        Write-Status ("already_stopped " + $Label)
        return
    }

    foreach ($id in $ids) {
        try {
            Stop-Process -Id $id -Force -ErrorAction Stop
            Write-Status ("stopped " + $Label + " pid=" + $id)
        }
        catch {
            $stillExists = Get-Process -Id $id -ErrorAction SilentlyContinue
            if ($stillExists) {
                Write-Status ("stop_unresolved " + $Label + " pid=" + $id + " error=" + $_.Exception.Message)
                continue
            }
            Write-Status ("already_stopped " + $Label + " pid=" + $id)
        }
    }
}
