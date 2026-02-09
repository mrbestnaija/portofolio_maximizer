@echo off
REM Portfolio Maximizer v45 - Environment Setup Script
REM Run this AFTER Python 3.10-3.12 is installed
REM See RECOVERY_PLAN.md for pre-requisites

echo ========================================
echo Portfolio Maximizer v45 - Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10-3.12 first.
    echo See RECOVERY_PLAN.md section 1 for installation instructions.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Verify we're in the right directory
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found.
    echo Please run this script from: portfolio_maximizer_v45\portfolio_maximizer_v45\
    pause
    exit /b 1
)

REM Check if venv exists
if exist "venv\" (
    echo [WARNING] venv\ already exists. Delete it to create fresh? (Y/N)
    set /p choice=
    if /i "%choice%"=="Y" (
        echo Removing old venv...
        rmdir /s /q venv
    )
)

REM Create virtual environment
if not exist "venv\" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [1/5] Using existing virtual environment
)
echo.

REM Activate venv and upgrade pip
echo [2/5] Upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    pause
    exit /b 1
)
echo [OK] pip upgraded
echo.

REM Install core dependencies
echo [3/5] Installing core dependencies from requirements.txt...
echo This may take several minutes depending on internet speed...
pip install -r requirements.txt --no-cache-dir
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Try running manually: pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK] Core dependencies installed
echo.

REM Verify .env file
echo [4/5] Checking .env file...
if not exist ".env" (
    echo [WARNING] .env file not found
    if exist ".env.template" (
        echo Creating .env from template...
        copy .env.template .env
        echo [ACTION REQUIRED] Edit .env and add your API keys
    ) else (
        echo [ERROR] No .env or .env.template found
    )
) else (
    echo [OK] .env file exists
)
echo.

REM Test imports
echo [5/5] Testing core imports...
python -c "import pandas; import numpy; import torch; print('[OK] Core libraries imported successfully')" 2>nul
if errorlevel 1 (
    echo [WARNING] Some core imports failed - verify installation
) else (
    echo [OK] Core imports successful
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
pause
