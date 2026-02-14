@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Portfolio Maximizer v45 - Environment Setup Script
REM Run this AFTER Python 3.10-3.12 is installed
REM See RECOVERY_PLAN.md for pre-requisites

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR:~0,-1%"
set "LOG_DIR=%SCRIPT_DIR%logs\setup"
set "AUDIT_DIR=%SCRIPT_DIR%logs\run_audit"
set "GATE_DIR=%SCRIPT_DIR%logs\audit_gate"
set "SECURITY_DIR=%SCRIPT_DIR%logs\security"
set "AUDIT_EVENT_SCRIPT=%SCRIPT_DIR%scripts\run_audit_event.py"
set "DASHBOARD_MANAGER_SCRIPT=%SCRIPT_DIR%scripts\windows_dashboard_manager.py"
set "SECURITY_PREFLIGHT_SCRIPT=%SCRIPT_DIR%scripts\security_preflight.py"
set "DB_PATH=%SCRIPT_DIR%data\portfolio_maximizer.db"
if "%ENABLE_DASHBOARD_API%"=="" set "ENABLE_DASHBOARD_API=1"
if "%AUTO_OPEN_DASHBOARD%"=="" set "AUTO_OPEN_DASHBOARD=1"
if "%DASHBOARD_PORT%"=="" set "DASHBOARD_PORT=8000"
if "%DASHBOARD_PERSIST%"=="" set "DASHBOARD_PERSIST=1"
if "%DASHBOARD_API_STRICT%"=="" set "DASHBOARD_API_STRICT=1"
if "%ENABLE_SECURITY_CHECKS%"=="" set "ENABLE_SECURITY_CHECKS=1"
if "%SECURITY_STRICT%"=="" set "SECURITY_STRICT=1"
if "%SECURITY_REQUIRE_PIP_AUDIT%"=="" set "SECURITY_REQUIRE_PIP_AUDIT=1"
if "%SECURITY_HARD_FAIL%"=="" set "SECURITY_HARD_FAIL=1"
if "%SECURITY_IGNORE_VULN_IDS%"=="" set "SECURITY_IGNORE_VULN_IDS="
if "%SECURITY_SQLITE_GUARDRAILS%"=="" set "SECURITY_SQLITE_GUARDRAILS=1"
if "%SECURITY_SQLITE_GUARDRAILS_HARD_FAIL%"=="" set "SECURITY_SQLITE_GUARDRAILS_HARD_FAIL=1"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%AUDIT_DIR%" mkdir "%AUDIT_DIR%"
if not exist "%GATE_DIR%" mkdir "%GATE_DIR%"
if not exist "%SECURITY_DIR%" mkdir "%SECURITY_DIR%"
set "STAMP="
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do if not "%%I"=="" set "STAMP=%%I"
if defined STAMP (
    set "STAMP=%STAMP:~0,8%_%STAMP:~8,6%"
) else (
    set "STAMP=%RANDOM%%RANDOM%"
)
set "RUN_ID=pmx_setup_%STAMP%_%RANDOM%%RANDOM%"
set "PARENT_RUN_ID=%PMX_PARENT_RUN_ID%"
set "PMX_RUN_ID=%RUN_ID%"
set "PMX_PARENT_RUN_ID=%PARENT_RUN_ID%"
set "LOG_FILE=%LOG_DIR%\setup_environment_%RUN_ID%.log"
set "AUDIT_FILE=%AUDIT_DIR%\setup_environment_%RUN_ID%.jsonl"
set "DASHBOARD_STATUS_JSON=%GATE_DIR%\dashboard_status_setup_%RUN_ID%.json"
set "SECURITY_JSON=%SECURITY_DIR%\security_preflight_%RUN_ID%.json"
set /a SUBPROC_SEQ=0

REM Security-oriented Python runtime hardening for all subprocesses.
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo [START] setup_environment at %DATE% %TIME% > "%LOG_FILE%"
echo [RUN] RUN_ID=%RUN_ID% PARENT_RUN_ID=%PARENT_RUN_ID% >> "%LOG_FILE%"
echo [CONFIG] DASHBOARD_API=%ENABLE_DASHBOARD_API% AUTO_OPEN=%AUTO_OPEN_DASHBOARD% PORT=%DASHBOARD_PORT% >> "%LOG_FILE%"
echo [CONFIG] SECURITY_CHECKS=%ENABLE_SECURITY_CHECKS% STRICT=%SECURITY_STRICT% REQUIRE_PIP_AUDIT=%SECURITY_REQUIRE_PIP_AUDIT% >> "%LOG_FILE%"
echo [CONFIG] SECURITY_IGNORE_VULN_IDS=%SECURITY_IGNORE_VULN_IDS% >> "%LOG_FILE%"

echo ========================================
echo Portfolio Maximizer v45 - Setup
echo ========================================
echo.
echo [LOG] %LOG_FILE%
echo.

REM Check Python
python --version >nul 2>&1
python --version >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10-3.12 first.
    echo See RECOVERY_PLAN.md section 1 for installation instructions.
    echo [ERROR] Python not found in PATH >> "%LOG_FILE%"
    pause
    exit /b 1
)

echo [OK] Python found
python --version >> "%LOG_FILE%" 2>&1
python --version
echo.

if not exist "%AUDIT_EVENT_SCRIPT%" (
    echo [ERROR] Audit event script missing at %AUDIT_EVENT_SCRIPT%
    echo [ERROR] Audit event script missing at %AUDIT_EVENT_SCRIPT% >> "%LOG_FILE%"
    pause
    exit /b 1
)
if not exist "%DASHBOARD_MANAGER_SCRIPT%" (
    echo [ERROR] Dashboard manager script missing at %DASHBOARD_MANAGER_SCRIPT%
    echo [ERROR] Dashboard manager script missing at %DASHBOARD_MANAGER_SCRIPT% >> "%LOG_FILE%"
    pause
    exit /b 1
)
if not exist "%SECURITY_PREFLIGHT_SCRIPT%" (
    echo [ERROR] Security preflight script missing at %SECURITY_PREFLIGHT_SCRIPT%
    echo [ERROR] Security preflight script missing at %SECURITY_PREFLIGHT_SCRIPT% >> "%LOG_FILE%"
    pause
    exit /b 1
)

call :audit_event "RUN_START" "STARTED" "bootstrap" "" "0" "setup_environment started"

if "%ENABLE_DASHBOARD_API%"=="1" (
    set /a SUBPROC_SEQ+=1
    set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
    call :audit_event "STEP_START" "RUNNING" "dashboard_api" "!SUBPROC_ID!" "0" "Ensuring dashboard API"
    set "DASHBOARD_OPEN_ARG="
    if "%AUTO_OPEN_DASHBOARD%"=="1" set "DASHBOARD_OPEN_ARG=--open-browser"
    set "DASHBOARD_PERSIST_ARG=--persist-snapshot"
    if "%DASHBOARD_PERSIST%"=="0" set "DASHBOARD_PERSIST_ARG=--no-persist-snapshot"
    set "DASHBOARD_STRICT_ARG=--no-strict"
    if "%DASHBOARD_API_STRICT%"=="1" set "DASHBOARD_STRICT_ARG=--strict"
    python "%DASHBOARD_MANAGER_SCRIPT%" ensure ^
        --root "%ROOT_DIR%" ^
        --python-bin python ^
        --port %DASHBOARD_PORT% ^
        --db-path "%DB_PATH%" ^
        --status-json "%DASHBOARD_STATUS_JSON%" ^
        --caller "setup_environment.bat" ^
        --run-id "%RUN_ID%" ^
        !DASHBOARD_PERSIST_ARG! ^
        !DASHBOARD_OPEN_ARG! ^
        !DASHBOARD_STRICT_ARG! >> "%LOG_FILE%" 2>&1
    set "DASH_RC=!ERRORLEVEL!"
    echo [DASHBOARD] Exit code: !DASH_RC! >> "%LOG_FILE%"
    if "!DASH_RC!"=="0" (
        call :audit_event "STEP_END" "SUCCESS" "dashboard_api" "!SUBPROC_ID!" "!DASH_RC!" "Dashboard API ready"
    ) else (
        call :audit_event "STEP_END" "FAIL" "dashboard_api" "!SUBPROC_ID!" "!DASH_RC!" "Dashboard API startup failed"
    )
) else (
    call :audit_event "STEP_END" "SKIPPED" "dashboard_api" "" "0" "Dashboard API disabled by ENABLE_DASHBOARD_API=0"
)

if "%ENABLE_SECURITY_CHECKS%"=="1" (
    set /a SUBPROC_SEQ+=1
    set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
    call :audit_event "STEP_START" "RUNNING" "security_preflight" "!SUBPROC_ID!" "0" "Running dependency and CVE preflight checks"
    set "SECURITY_STRICT_ARG=--strict"
    if "%SECURITY_STRICT%"=="0" set "SECURITY_STRICT_ARG=--no-strict"
    set "SECURITY_REQUIRE_ARG="
    if "%SECURITY_REQUIRE_PIP_AUDIT%"=="1" set "SECURITY_REQUIRE_ARG=--require-pip-audit"
    set "SECURITY_IGNORE_ARGS="
    if defined SECURITY_IGNORE_VULN_IDS (
        for %%V in (%SECURITY_IGNORE_VULN_IDS:,= %) do (
            if not "%%~V"=="" set "SECURITY_IGNORE_ARGS=!SECURITY_IGNORE_ARGS! --ignore-vuln-id %%~V"
        )
    )
    python "%SECURITY_PREFLIGHT_SCRIPT%" ^
        --python-bin python ^
        --output-json "%SECURITY_JSON%" ^
        --caller "setup_environment.bat" ^
        --run-id "%RUN_ID%" ^
        !SECURITY_STRICT_ARG! ^
        !SECURITY_REQUIRE_ARG! ^
        !SECURITY_IGNORE_ARGS! >> "%LOG_FILE%" 2>&1
    set "SEC_RC=!ERRORLEVEL!"
    echo [SECURITY] Exit code: !SEC_RC! >> "%LOG_FILE%"
    if "!SEC_RC!"=="0" (
        call :audit_event "STEP_END" "SUCCESS" "security_preflight" "!SUBPROC_ID!" "!SEC_RC!" "Security preflight passed"
    ) else (
        call :audit_event "STEP_END" "FAIL" "security_preflight" "!SUBPROC_ID!" "!SEC_RC!" "Security preflight failed"
        if not "%SECURITY_HARD_FAIL%"=="0" (
            call :audit_event "RUN_END" "FAIL" "security_preflight" "" "!SEC_RC!" "setup_environment aborted by security preflight"
            pause
            exit /b !SEC_RC!
        )
    )
) else (
    call :audit_event "STEP_END" "SKIPPED" "security_preflight" "" "0" "Security preflight disabled by ENABLE_SECURITY_CHECKS=0"
)

REM Verify we're in the right directory
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "verify_workspace" "!SUBPROC_ID!" "0" "Checking required files"
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found.
    echo Please run this script from: portfolio_maximizer_v45\portfolio_maximizer_v45\
    call :audit_event "STEP_END" "FAIL" "verify_workspace" "!SUBPROC_ID!" "1" "requirements.txt missing"
    pause
    exit /b 1
)
call :audit_event "STEP_END" "SUCCESS" "verify_workspace" "!SUBPROC_ID!" "0" "Workspace validated"

REM Check if venv exists
if exist "venv\" (
    echo [WARNING] venv\ already exists. Delete it to create fresh? (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        echo Removing old venv...
        echo [ACTION] Removing existing venv >> "%LOG_FILE%"
        rmdir /s /q venv
    )
)

REM Create virtual environment
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "create_venv" "!SUBPROC_ID!" "0" "Ensuring virtual environment"
if not exist "venv\" (
    echo [1/5] Creating virtual environment...
    python -m venv venv >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo [ERROR] Failed to create virtual environment >> "%LOG_FILE%"
        call :audit_event "STEP_END" "FAIL" "create_venv" "!SUBPROC_ID!" "1" "Virtual environment creation failed"
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [1/5] Using existing virtual environment
)
call :audit_event "STEP_END" "SUCCESS" "create_venv" "!SUBPROC_ID!" "0" "Virtual environment ready"
echo.

REM Activate venv and upgrade pip
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "upgrade_pip" "!SUBPROC_ID!" "0" "Upgrading pip"
echo [2/5] Upgrading pip...
call venv\Scripts\activate.bat >> "%LOG_FILE%" 2>&1
python -m pip install --upgrade pip --quiet >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    echo [ERROR] Failed to upgrade pip >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "upgrade_pip" "!SUBPROC_ID!" "1" "pip upgrade failed"
    pause
    exit /b 1
)
echo [OK] pip upgraded
call :audit_event "STEP_END" "SUCCESS" "upgrade_pip" "!SUBPROC_ID!" "0" "pip upgraded"
echo.

REM Install core dependencies
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "install_dependencies" "!SUBPROC_ID!" "0" "Installing requirements.txt"
echo [3/5] Installing core dependencies from requirements.txt...
echo This may take several minutes depending on internet speed...
pip install -r requirements.txt --no-cache-dir >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Try running manually: pip install -r requirements.txt
    echo [ERROR] pip install failed >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "install_dependencies" "!SUBPROC_ID!" "1" "Dependency installation failed"
    pause
    exit /b 1
)
echo [OK] Core dependencies installed
call :audit_event "STEP_END" "SUCCESS" "install_dependencies" "!SUBPROC_ID!" "0" "Dependencies installed"
echo.

REM Verify .env file
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "verify_env_file" "!SUBPROC_ID!" "0" "Checking .env configuration"
echo [4/5] Checking .env file...
if not exist ".env" (
    echo [WARNING] .env file not found
    echo [WARN] .env missing >> "%LOG_FILE%"
    if exist ".env.template" (
        echo Creating .env from template...
        copy .env.template .env >> "%LOG_FILE%" 2>&1
        echo [ACTION REQUIRED] Edit .env and add your API keys
    ) else (
        echo [ERROR] No .env or .env.template found
        echo [ERROR] Missing .env and .env.template >> "%LOG_FILE%"
    )
) else (
    echo [OK] .env file exists
)
call :audit_event "STEP_END" "SUCCESS" "verify_env_file" "!SUBPROC_ID!" "0" "Environment file check completed"
echo.

REM Test imports
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "import_smoke_test" "!SUBPROC_ID!" "0" "Running import smoke test"
echo [5/5] Testing core imports...
python -c "import pandas; import numpy; import torch; print('[OK] Core libraries imported successfully')" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [WARNING] Some core imports failed - verify installation
    echo [WARN] Core import check failed >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "import_smoke_test" "!SUBPROC_ID!" "1" "Import smoke test failed"
) else (
    echo [OK] Core imports successful
    call :audit_event "STEP_END" "SUCCESS" "import_smoke_test" "!SUBPROC_ID!" "0" "Import smoke test passed"
)
echo.

REM Summary
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Virtual environment: venv\
echo Activate with: venv\Scripts\activate.bat
echo.
echo Next steps:
echo 1. If WSL not installed, see RECOVERY_PLAN.md section 2
echo 2. Verify .env file has valid API keys
echo 3. Run tests: pytest tests\ -v
echo 4. Check RECOVERY_PLAN.md for model artifact recovery
echo.
echo Environment ready for development!
echo [END] setup_environment complete at %DATE% %TIME% >> "%LOG_FILE%"
call :audit_event "RUN_END" "COMPLETED" "finalize" "" "0" "setup_environment completed"
echo [LOG] %LOG_FILE%
echo [AUDIT] %AUDIT_FILE%
pause
goto :eof

:audit_event
set "AE_EVENT=%~1"
set "AE_STATUS=%~2"
set "AE_STEP=%~3"
set "AE_SUBPROC=%~4"
set "AE_EXIT=%~5"
set "AE_MSG=%~6"
python "%AUDIT_EVENT_SCRIPT%" ^
    --audit-file "%AUDIT_FILE%" ^
    --run-id "%RUN_ID%" ^
    --parent-run-id "%PARENT_RUN_ID%" ^
    --script-name "setup_environment.bat" ^
    --event "%AE_EVENT%" ^
    --status "%AE_STATUS%" ^
    --step "%AE_STEP%" ^
    --subprocess-id "%AE_SUBPROC%" ^
    --exit-code %AE_EXIT% ^
    --message "%AE_MSG%" ^
    --log-file "%LOG_FILE%" >nul 2>&1
exit /b 0
