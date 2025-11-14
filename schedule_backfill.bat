@echo off
REM Scheduled task wrapper for nightly LLM signal validation backfill
set SCRIPT_DIR=%~dp0
set PYTHON_BIN=%SCRIPT_DIR%simpleTrader_env\Scripts\python.exe
set BACKFILL_SCRIPT=%SCRIPT_DIR%scripts\backfill_signal_validation.py
if not exist "%PYTHON_BIN%" (
  echo [ERROR] Python interpreter not found at %PYTHON_BIN%
  exit /b 1
)
if not exist "%BACKFILL_SCRIPT%" (
  echo [ERROR] Backfill script not found at %BACKFILL_SCRIPT%
  exit /b 1
)
"%PYTHON_BIN%" "%BACKFILL_SCRIPT%" --backtest-days 30 --lookback-days 60 --portfolio-value 10000 --db-path "%SCRIPT_DIR%data\portfolio_maximizer.db"
