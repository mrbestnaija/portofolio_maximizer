@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "DASHBOARD_MANAGER_SCRIPT=%ROOT_DIR%\scripts\windows_dashboard_manager.py"
set "DB_PATH=%ROOT_DIR%\data\portfolio_maximizer.db"
set "PYTHON_BIN=%ROOT_DIR%\simpleTrader_env\Scripts\python.exe"

if not exist "%PYTHON_BIN%" set "PYTHON_BIN=python"

if not exist "%DASHBOARD_MANAGER_SCRIPT%" (
    echo [ERROR] Dashboard manager script missing at %DASHBOARD_MANAGER_SCRIPT%
    exit /b 1
)

"%PYTHON_BIN%" "%DASHBOARD_MANAGER_SCRIPT%" launch ^
    --root "%ROOT_DIR%" ^
    --python-bin "%PYTHON_BIN%" ^
    --db-path "%DB_PATH%" ^
    %*

exit /b %ERRORLEVEL%
