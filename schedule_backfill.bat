@echo off
REM Scheduled task wrapper for nightly Time Series forecaster audit/health check
set SCRIPT_DIR=%~dp0
set PYTHON_BIN=%SCRIPT_DIR%simpleTrader_env\Scripts\python.exe
set AUDIT_SCRIPT=%SCRIPT_DIR%scripts\check_forecast_audits.py
if not exist "%PYTHON_BIN%" (
  echo [ERROR] Python interpreter not found at %PYTHON_BIN%
  exit /b 1
)
if not exist "%AUDIT_SCRIPT%" (
  echo [ERROR] TS forecaster audit script not found at %AUDIT_SCRIPT%
  exit /b 1
)
"%PYTHON_BIN%" "%AUDIT_SCRIPT%" --audit-dir "%SCRIPT_DIR%logs\forecast_audits" --config-path "%SCRIPT_DIR%config\forecaster_monitoring.yml"
