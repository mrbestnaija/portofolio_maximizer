@echo off
setlocal

set "ROOT=%~dp0.."
set "PYTHON_BIN=%ROOT%\simpleTrader_env\Scripts\python.exe"
if not exist "%PYTHON_BIN%" set "PYTHON_BIN=C:\Python314\python.exe"
if not exist "%PYTHON_BIN%" set "PYTHON_BIN=python"

"%PYTHON_BIN%" "%ROOT%\scripts\windows_persistence_manager.py" ensure ^
  --status-json "%ROOT%\logs\persistence_manager_status.json" ^
  --db-path "%ROOT%\data\portfolio_maximizer.db" ^
  --audit-dir "%ROOT%\logs\forecast_audits" ^
  --watcher-tickers "AAPL,AMZN,GOOG,GS,JPM,META,MSFT,NVDA,TSLA,V"

exit /b %ERRORLEVEL%
