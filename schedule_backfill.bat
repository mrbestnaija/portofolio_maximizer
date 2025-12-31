@echo off
@echo off
REM Lightweight scheduled-task wrapper.
REM Default: trigger auto_trader_core via WSL to build TS evidence.
REM Optional: fallback to nightly TS forecaster audit/health check.

set SCRIPT_DIR=%~dp0
set PYTHON_BIN=%SCRIPT_DIR%simpleTrader_env\Scripts\python.exe
set AUDIT_SCRIPT=%SCRIPT_DIR%scripts\check_forecast_audits.py
set TASK=%1
if "%TASK%"=="" set TASK=auto_trader_core

if /I "%TASK%"=="auto_trader_core" (
  REM Use WSL to call the cron multiplexer for core tickers (halts after targets).
  where wsl >nul 2>&1 || (echo [ERROR] WSL not found; cannot run auto_trader_core & exit /b 1)
  wsl bash -lc "cd \"$(wslpath -a \"%SCRIPT_DIR%\")\" && bash/bash/production_cron.sh auto_trader_core"
  exit /b %ERRORLEVEL%
)

if not exist "%PYTHON_BIN%" (
  echo [ERROR] Python interpreter not found at %PYTHON_BIN%
  exit /b 1
)
if not exist "%AUDIT_SCRIPT%" (
  echo [ERROR] TS forecaster audit script not found at %AUDIT_SCRIPT%
  exit /b 1
)
"%PYTHON_BIN%" "%AUDIT_SCRIPT%" --audit-dir "%SCRIPT_DIR%logs\forecast_audits" --config-path "%SCRIPT_DIR%config\forecaster_monitoring.yml"
