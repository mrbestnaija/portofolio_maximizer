@echo off
setlocal EnableExtensions
REM Minimal Windows Task Scheduler wrapper for the core auto-trader cycle.
REM This only runs the trading heartbeat. OpenClaw reporting stays in the
REM audit/monitor jobs and does not control the cycle itself.

set "SCRIPT_DIR=%~dp0"

where wsl >nul 2>&1 || (
    echo [ERROR] WSL not found; cannot run auto_trader_core
    exit /b 1
)

echo [CORE] Starting auto_trader_core via bash/production_cron.sh
wsl --cd "%SCRIPT_DIR%" bash -lc "bash/production_cron.sh auto_trader_core"
set "EXIT_CODE=%ERRORLEVEL%"
echo [CORE] Exit code: %EXIT_CODE%
exit /b %EXIT_CODE%
