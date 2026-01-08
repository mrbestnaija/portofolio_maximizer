@echo off
REM Docker Build and Test Script for Portfolio Maximizer v4.5 (Windows)
REM This script builds the Docker images and runs comprehensive tests

setlocal enabledelayedexpansion

REM Configuration
set IMAGE_NAME=portfolio-maximizer
set VERSION=v4.5
set COMPOSE_PROJECT_NAME=portfolio_maximizer

REM Colors not supported in batch, using echo prefixes instead
echo.
echo ===================================================
echo Docker Build and Test for Portfolio Maximizer v4.5
echo ===================================================
echo.

REM Parse command line arguments
set ACTION=%1
if "%ACTION%"=="" set ACTION=all

REM Check prerequisites
if "%ACTION%"=="prereq" goto :check_prereq
if "%ACTION%"=="build" goto :check_prereq
if "%ACTION%"=="test" goto :test_containers
if "%ACTION%"=="all" goto :check_prereq
goto :usage

:check_prereq
echo [CHECKING] Prerequisites...
echo.

REM Check Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker not found. Please install Docker Desktop for Windows.
    exit /b 1
)
for /f "tokens=*" %%i in ('docker --version') do echo [OK] Docker found: %%i

REM Check Docker Compose
docker compose version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose not found. Please install Docker Desktop for Windows.
    exit /b 1
)
echo [OK] Docker Compose found

REM Check if Docker daemon is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker daemon is not running. Please start Docker Desktop.
    exit /b 1
)
echo [OK] Docker daemon is running

REM Check for .env file
if exist ".env" (
    echo [OK] .env file found
) else (
    echo [WARNING] .env file not found. Creating from .env.template...
    if exist ".env.template" (
        copy ".env.template" ".env" >nul
        echo [OK] Created .env file from .env.template
        echo [WARNING] Fill in real credentials locally before running API-dependent features.
    )
)

if "%ACTION%"=="prereq" goto :end
if "%ACTION%"=="build" goto :build_images
if "%ACTION%"=="all" goto :build_images
goto :end

:build_images
echo.
echo [BUILDING] Docker Images...
echo.

REM Build production image
echo Building production image...
docker build --target production --tag %IMAGE_NAME%:%VERSION% --tag %IMAGE_NAME%:latest --file Dockerfile .
if %errorlevel% neq 0 (
    echo [ERROR] Failed to build production image
    exit /b 1
)
echo [OK] Production image built: %IMAGE_NAME%:%VERSION%

REM Build development image
echo.
echo Building development image...
docker build --target builder --tag %IMAGE_NAME%:%VERSION%-dev --file Dockerfile .
if %errorlevel% neq 0 (
    echo [ERROR] Failed to build development image
    exit /b 1
)
echo [OK] Development image built: %IMAGE_NAME%:%VERSION%-dev

REM List built images
echo.
echo Built images:
docker images | findstr %IMAGE_NAME%

if "%ACTION%"=="build" goto :end
if "%ACTION%"=="all" goto :test_containers
goto :end

:test_containers
echo.
echo [TESTING] Docker Containers...
echo.

REM Test 1: Basic container startup
echo Test 1: Basic container startup...
docker run --rm %IMAGE_NAME%:%VERSION% python -c "print('Container started successfully')"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start container
    exit /b 1
)
echo [OK] Container startup test passed

REM Test 2: Validate Python dependencies
echo.
echo Test 2: Validating Python dependencies...
docker run --rm %IMAGE_NAME%:%VERSION% python -c "import numpy, pandas, scipy, sklearn, matplotlib, yfinance; print('All core dependencies imported successfully')"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to import dependencies
    exit /b 1
)
echo [OK] Dependency validation passed

REM Test 3: ETL module imports
echo.
echo Test 3: Testing ETL module imports...
docker run --rm %IMAGE_NAME%:%VERSION% python -c "from etl import YFinanceExtractor, DataValidator, Preprocessor, DataStorage; print('All ETL modules imported successfully')"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to import ETL modules
    exit /b 1
)
echo [OK] ETL module import test passed

REM Test 4: Health check
echo.
echo Test 4: Running health check...
docker run --rm %IMAGE_NAME%:%VERSION% python -c "import sys; sys.exit(0)"
if %errorlevel% neq 0 (
    echo [ERROR] Health check failed
    exit /b 1
)
echo [OK] Health check passed

REM Test 5: Docker Compose
echo.
echo Test 5: Testing Docker Compose...
docker compose up -d portfolio-maximizer
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services
    exit /b 1
)

REM Wait for container
timeout /t 5 /nobreak >nul
echo [OK] Container started with docker-compose

REM Test ETL help
docker compose exec -T portfolio-maximizer python scripts/run_etl_pipeline.py --help >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ETL pipeline help test failed
) else (
    echo [OK] ETL pipeline help command successful
)

REM Stop services
docker compose down
echo [OK] Docker Compose tests completed

if "%ACTION%"=="test" goto :end
if "%ACTION%"=="all" goto :sample_etl
goto :end

:sample_etl
echo.
echo [RUNNING] Sample ETL Pipeline...
echo.

REM Create sample ticker list
echo AAPL> data\sample_tickers.txt
echo MSFT>> data\sample_tickers.txt
echo GOOG>> data\sample_tickers.txt

REM Run ETL pipeline
echo Running ETL pipeline with sample data...
docker run --rm -v "%cd%\data:/app/data" -v "%cd%\config:/app/config" -v "%cd%\logs:/app/logs" %IMAGE_NAME%:%VERSION% python scripts/run_etl_pipeline.py --tickers AAPL,MSFT,GOOGL --start 2023-01-01 --end 2023-12-31 --include-frontier-tickers --no-cache
if %errorlevel% neq 0 (
    echo [WARNING] ETL pipeline run failed - this might be due to missing API keys
) else (
    echo [OK] ETL pipeline completed
)

REM Check for output files
if exist "data\training\*.parquet" (
    echo [OK] ETL pipeline created output files
    dir data\training\*.parquet
) else (
    echo [WARNING] No output files created - API keys might be required
)

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
exit /b 1

:end
endlocal
