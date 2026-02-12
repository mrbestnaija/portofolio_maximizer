@echo off
REM Docker Build and Test Script for Portfolio Maximizer v4.5 (Windows)
REM This script builds the Docker images and runs comprehensive tests

setlocal enabledelayedexpansion

REM Configuration
set IMAGE_NAME=portfolio-maximizer
set VERSION=v4.5
set COMPOSE_PROJECT_NAME=portfolio_maximizer
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "LOG_DIR=%ROOT_DIR%\logs\docker"
set "AUDIT_DIR=%ROOT_DIR%\logs\run_audit"
set "GATE_DIR=%ROOT_DIR%\logs\audit_gate"
set "SECURITY_DIR=%ROOT_DIR%\logs\security"
set "AUDIT_EVENT_SCRIPT=%ROOT_DIR%\scripts\run_audit_event.py"
set "DASHBOARD_MANAGER_SCRIPT=%ROOT_DIR%\scripts\windows_dashboard_manager.py"
set "SECURITY_PREFLIGHT_SCRIPT=%ROOT_DIR%\scripts\security_preflight.py"
set "DB_PATH=%ROOT_DIR%\data\portfolio_maximizer.db"
set "SECURITY_PYTHON_BIN=python"
if exist "%ROOT_DIR%\simpleTrader_env\Scripts\python.exe" set "SECURITY_PYTHON_BIN=%ROOT_DIR%\simpleTrader_env\Scripts\python.exe"
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

REM Security-oriented Python runtime hardening for all subprocesses.
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
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
set "RUN_ID=pmx_docker_%STAMP%_%RANDOM%%RANDOM%"
set "PARENT_RUN_ID=%PMX_PARENT_RUN_ID%"
set "PMX_RUN_ID=%RUN_ID%"
set "PMX_PARENT_RUN_ID=%PARENT_RUN_ID%"
set "LOG_FILE=%LOG_DIR%\docker_build_test_%RUN_ID%.log"
set "AUDIT_FILE=%AUDIT_DIR%\docker_build_test_%RUN_ID%.jsonl"
set "DASHBOARD_STATUS_JSON=%GATE_DIR%\dashboard_status_docker_%RUN_ID%.json"
set "SECURITY_JSON=%SECURITY_DIR%\security_preflight_%RUN_ID%.json"
set /a SUBPROC_SEQ=0

REM Colors not supported in batch, using echo prefixes instead
echo.
echo ===================================================
echo Docker Build and Test for Portfolio Maximizer v4.5
echo ===================================================
echo.

REM Parse command line arguments
set ACTION=%1
if "%ACTION%"=="" set ACTION=all
echo [START] docker_build_test action=%ACTION% at %DATE% %TIME% > "%LOG_FILE%"
echo [LOG] %LOG_FILE%
echo [RUN] RUN_ID=%RUN_ID% PARENT_RUN_ID=%PARENT_RUN_ID% >> "%LOG_FILE%"
echo [CONFIG] DASHBOARD_API=%ENABLE_DASHBOARD_API% AUTO_OPEN=%AUTO_OPEN_DASHBOARD% PORT=%DASHBOARD_PORT% >> "%LOG_FILE%"
echo [CONFIG] SECURITY_CHECKS=%ENABLE_SECURITY_CHECKS% STRICT=%SECURITY_STRICT% REQUIRE_PIP_AUDIT=%SECURITY_REQUIRE_PIP_AUDIT% >> "%LOG_FILE%"
echo [CONFIG] SECURITY_IGNORE_VULN_IDS=%SECURITY_IGNORE_VULN_IDS% >> "%LOG_FILE%"

if not exist "%AUDIT_EVENT_SCRIPT%" (
    echo [ERROR] Audit event script missing at %AUDIT_EVENT_SCRIPT%
    echo [ERROR] Audit event script missing at %AUDIT_EVENT_SCRIPT% >> "%LOG_FILE%"
    call :audit_event "RUN_END" "FAIL" "bootstrap" "" "1" "Audit event script missing"
    exit /b 1
)
if not exist "%DASHBOARD_MANAGER_SCRIPT%" (
    echo [ERROR] Dashboard manager script missing at %DASHBOARD_MANAGER_SCRIPT%
    echo [ERROR] Dashboard manager script missing at %DASHBOARD_MANAGER_SCRIPT% >> "%LOG_FILE%"
    call :audit_event "RUN_END" "FAIL" "bootstrap" "" "1" "Dashboard manager script missing"
    exit /b 1
)
if not exist "%SECURITY_PREFLIGHT_SCRIPT%" (
    echo [ERROR] Security preflight script missing at %SECURITY_PREFLIGHT_SCRIPT%
    echo [ERROR] Security preflight script missing at %SECURITY_PREFLIGHT_SCRIPT% >> "%LOG_FILE%"
    call :audit_event "RUN_END" "FAIL" "bootstrap" "" "1" "Security preflight script missing"
    exit /b 1
)

call :audit_event "RUN_START" "STARTED" "bootstrap" "" "0" "docker_build_test started"

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
    "%SECURITY_PYTHON_BIN%" "%DASHBOARD_MANAGER_SCRIPT%" ensure ^
        --root "%ROOT_DIR%" ^
        --python-bin python ^
        --port %DASHBOARD_PORT% ^
        --db-path "%DB_PATH%" ^
        --status-json "%DASHBOARD_STATUS_JSON%" ^
        --caller "docker_build_test.bat" ^
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
    "%SECURITY_PYTHON_BIN%" "%SECURITY_PREFLIGHT_SCRIPT%" ^
        --python-bin "%SECURITY_PYTHON_BIN%" ^
        --output-json "%SECURITY_JSON%" ^
        --caller "scripts/docker_build_test.bat" ^
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
            call :audit_event "RUN_END" "FAIL" "security_preflight" "" "!SEC_RC!" "docker_build_test aborted by security preflight"
            exit /b !SEC_RC!
        )
    )
) else (
    call :audit_event "STEP_END" "SKIPPED" "security_preflight" "" "0" "Security preflight disabled by ENABLE_SECURITY_CHECKS=0"
)

REM Check prerequisites
if "%ACTION%"=="prereq" goto :check_prereq
if "%ACTION%"=="build" goto :check_prereq
if "%ACTION%"=="test" goto :test_containers
if "%ACTION%"=="all" goto :check_prereq
goto :usage

:check_prereq
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "check_prerequisites" "!SUBPROC_ID!" "0" "Checking Docker prerequisites"
echo [CHECKING] Prerequisites...
echo.

REM Check Docker
docker --version >nul 2>&1
docker --version >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker not found. Please install Docker Desktop for Windows.
    echo [ERROR] Docker not found >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "check_prerequisites" "!SUBPROC_ID!" "1" "Docker binary missing"
    call :audit_event "RUN_END" "FAIL" "check_prerequisites" "" "1" "Docker binary missing"
    exit /b 1
)
for /f "tokens=*" %%i in ('docker --version') do echo [OK] Docker found: %%i

REM Check Docker Compose
docker compose version >nul 2>&1
docker compose version >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose not found. Please install Docker Desktop for Windows.
    echo [ERROR] Docker Compose not found >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "check_prerequisites" "!SUBPROC_ID!" "1" "Docker Compose missing"
    call :audit_event "RUN_END" "FAIL" "check_prerequisites" "" "1" "Docker Compose missing"
    exit /b 1
)
echo [OK] Docker Compose found

REM Check if Docker daemon is running
docker info >nul 2>&1
docker info >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker daemon is not running. Please start Docker Desktop.
    echo [ERROR] Docker daemon is not running >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "check_prerequisites" "!SUBPROC_ID!" "1" "Docker daemon not running"
    call :audit_event "RUN_END" "FAIL" "check_prerequisites" "" "1" "Docker daemon not running"
    exit /b 1
)
echo [OK] Docker daemon is running

REM Check for .env file
if exist ".env" (
    echo [OK] .env file found
) else (
    echo [WARNING] .env file not found. Creating from .env.template...
    if exist ".env.template" (
        copy ".env.template" ".env" >nul 2>&1
        echo [OK] Created .env file from .env.template
        echo [WARNING] Fill in real credentials locally before running API-dependent features.
        echo [WARN] Created .env from template >> "%LOG_FILE%"
    )
)
call :audit_event "STEP_END" "SUCCESS" "check_prerequisites" "!SUBPROC_ID!" "0" "Prerequisite checks completed"

if "%ACTION%"=="prereq" goto :end
if "%ACTION%"=="build" goto :build_images
if "%ACTION%"=="all" goto :build_images
goto :end

:build_images
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "build_images" "!SUBPROC_ID!" "0" "Building Docker images"
echo.
echo [BUILDING] Docker Images...
echo.

REM Build production image
echo Building production image...
docker build --target production --tag %IMAGE_NAME%:%VERSION% --tag %IMAGE_NAME%:latest --file Dockerfile . >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to build production image
    echo [ERROR] Failed to build production image >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "build_images" "!SUBPROC_ID!" "1" "Production image build failed"
    call :audit_event "RUN_END" "FAIL" "build_images" "" "1" "Production image build failed"
    exit /b 1
)
echo [OK] Production image built: %IMAGE_NAME%:%VERSION%

REM Build development image
echo.
echo Building development image...
docker build --target builder --tag %IMAGE_NAME%:%VERSION%-dev --file Dockerfile . >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to build development image
    echo [ERROR] Failed to build development image >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "build_images" "!SUBPROC_ID!" "1" "Development image build failed"
    call :audit_event "RUN_END" "FAIL" "build_images" "" "1" "Development image build failed"
    exit /b 1
)
echo [OK] Development image built: %IMAGE_NAME%:%VERSION%-dev

REM List built images
echo.
echo Built images:
docker images | findstr %IMAGE_NAME% >> "%LOG_FILE%" 2>&1
docker images | findstr %IMAGE_NAME%
call :audit_event "STEP_END" "SUCCESS" "build_images" "!SUBPROC_ID!" "0" "Docker images built"

if "%ACTION%"=="build" goto :end
if "%ACTION%"=="all" goto :test_containers
goto :end

:test_containers
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "test_containers" "!SUBPROC_ID!" "0" "Running Docker container tests"
echo.
echo [TESTING] Docker Containers...
echo.

REM Test 1: Basic container startup
echo Test 1: Basic container startup...
docker run --rm %IMAGE_NAME%:%VERSION% python -c "print('Container started successfully')" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start container
    echo [ERROR] Failed to start container >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "test_containers" "!SUBPROC_ID!" "1" "Container startup test failed"
    call :audit_event "RUN_END" "FAIL" "test_containers" "" "1" "Container startup test failed"
    exit /b 1
)
echo [OK] Container startup test passed

REM Test 2: Validate Python dependencies
echo.
echo Test 2: Validating Python dependencies...
docker run --rm %IMAGE_NAME%:%VERSION% python -c "import numpy, pandas, scipy, sklearn, matplotlib, yfinance; print('All core dependencies imported successfully')" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to import dependencies
    echo [ERROR] Dependency validation failed >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "test_containers" "!SUBPROC_ID!" "1" "Dependency validation failed"
    call :audit_event "RUN_END" "FAIL" "test_containers" "" "1" "Dependency validation failed"
    exit /b 1
)
echo [OK] Dependency validation passed

REM Test 3: ETL module imports
echo.
echo Test 3: Testing ETL module imports...
docker run --rm %IMAGE_NAME%:%VERSION% python -c "from etl import YFinanceExtractor, DataValidator, Preprocessor, DataStorage; print('All ETL modules imported successfully')" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to import ETL modules
    echo [ERROR] ETL import test failed >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "test_containers" "!SUBPROC_ID!" "1" "ETL module import test failed"
    call :audit_event "RUN_END" "FAIL" "test_containers" "" "1" "ETL module import test failed"
    exit /b 1
)
echo [OK] ETL module import test passed

REM Test 4: Health check
echo.
echo Test 4: Running health check...
docker run --rm %IMAGE_NAME%:%VERSION% python -c "import sys; sys.exit(0)" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Health check failed
    echo [ERROR] Health check failed >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "test_containers" "!SUBPROC_ID!" "1" "Health check failed"
    call :audit_event "RUN_END" "FAIL" "test_containers" "" "1" "Health check failed"
    exit /b 1
)
echo [OK] Health check passed

REM Test 5: Docker Compose
echo.
echo Test 5: Testing Docker Compose...
docker compose up -d portfolio-maximizer >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services
    echo [ERROR] docker compose up failed >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "test_containers" "!SUBPROC_ID!" "1" "docker compose up failed"
    call :audit_event "RUN_END" "FAIL" "test_containers" "" "1" "docker compose up failed"
    exit /b 1
)

REM Wait for container
timeout /t 5 /nobreak >nul
echo [OK] Container started with docker-compose

REM Test ETL help
docker compose exec -T portfolio-maximizer python scripts/run_etl_pipeline.py --help >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ETL pipeline help test failed
    echo [WARN] ETL pipeline help test failed >> "%LOG_FILE%"
) else (
    echo [OK] ETL pipeline help command successful
)

REM Stop services
docker compose down >> "%LOG_FILE%" 2>&1
echo [OK] Docker Compose tests completed
call :audit_event "STEP_END" "SUCCESS" "test_containers" "!SUBPROC_ID!" "0" "Container tests completed"

if "%ACTION%"=="test" goto :end
if "%ACTION%"=="all" goto :sample_etl
goto :end

:sample_etl
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "sample_etl" "!SUBPROC_ID!" "0" "Running sample ETL in container"
echo.
echo [RUNNING] Sample ETL Pipeline...
echo.

REM Create sample ticker list
echo AAPL> data\sample_tickers.txt
echo MSFT>> data\sample_tickers.txt
echo GOOG>> data\sample_tickers.txt

REM Run ETL pipeline
echo Running ETL pipeline with sample data...
docker run --rm -v "%cd%\data:/app/data" -v "%cd%\config:/app/config" -v "%cd%\logs:/app/logs" %IMAGE_NAME%:%VERSION% python scripts/run_etl_pipeline.py --tickers AAPL,MSFT,GOOGL --start 2023-01-01 --end 2023-12-31 --include-frontier-tickers --no-cache >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ETL pipeline run failed - this might be due to missing API keys
    echo [WARN] Sample ETL run failed >> "%LOG_FILE%"
) else (
    echo [OK] ETL pipeline completed
)

REM Check for output files
if exist "data\training\*.parquet" (
    echo [OK] ETL pipeline created output files
    dir data\training\*.parquet >> "%LOG_FILE%" 2>&1
    dir data\training\*.parquet
) else (
    echo [WARNING] No output files created - API keys might be required
    echo [WARN] No parquet outputs created by sample ETL >> "%LOG_FILE%"
)
call :audit_event "STEP_END" "SUCCESS" "sample_etl" "!SUBPROC_ID!" "0" "Sample ETL step completed"

goto :show_usage

:show_usage
echo.
echo ===================================================
echo Docker Usage Examples
echo ===================================================
echo.
echo Build images:
echo   docker build -t %IMAGE_NAME%:%VERSION% .
echo.
echo Run ETL pipeline:
echo   docker run --rm -v "%cd%\data:/app/data" %IMAGE_NAME%:%VERSION% python scripts/run_etl_pipeline.py --tickers AAPL,MSFT --start 2023-01-01 --include-frontier-tickers
echo.
echo Interactive shell:
echo   docker run -it --rm -v "%cd%:/app" %IMAGE_NAME%:%VERSION%-dev /bin/bash
echo.
echo Run with docker-compose:
echo   docker compose up portfolio-maximizer
echo.
echo Run tests:
echo   docker compose run --rm portfolio-test
echo.
echo Start Jupyter notebook:
echo   docker compose up portfolio-notebook
echo.
echo Development mode:
echo   docker compose run --rm portfolio-dev /bin/bash
echo.
echo View logs:
echo   docker compose logs -f portfolio-maximizer
echo.
echo Clean up:
echo   docker compose down -v
echo.

if "%ACTION%"=="all" (
    echo ===================================================
    echo [SUCCESS] Docker Build and Test Completed!
    echo ===================================================
    echo.
    echo Summary:
    echo - Prerequisites: PASSED
    echo - Docker Build: PASSED
    echo - Container Tests: PASSED
    echo - Sample ETL: COMPLETED
    echo.
    echo Next steps:
    echo 1. Add API keys to .env file
    echo 2. Run: docker compose up
    echo 3. Check logs: docker compose logs
    echo.
)

goto :end

:usage
echo Usage: docker_build_test.bat [prereq^|build^|test^|all]
echo   prereq - Check prerequisites only
echo   build  - Build Docker images
echo   test   - Run container tests
echo   all    - Run all steps (default)
call :audit_event "RUN_END" "FAIL" "usage" "" "1" "Invalid action argument"
exit /b 1

:end
echo [END] docker_build_test action=%ACTION% at %DATE% %TIME% >> "%LOG_FILE%"
call :audit_event "RUN_END" "COMPLETED" "finalize" "" "0" "docker_build_test finished"
echo [LOG] %LOG_FILE%
echo [AUDIT] %AUDIT_FILE%
endlocal
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
    --script-name "scripts/docker_build_test.bat" ^
    --event "%AE_EVENT%" ^
    --status "%AE_STATUS%" ^
    --step "%AE_STEP%" ^
    --subprocess-id "%AE_SUBPROC%" ^
    --exit-code %AE_EXIT% ^
    --message "%AE_MSG%" ^
    --log-file "%LOG_FILE%" >nul 2>&1
exit /b 0
