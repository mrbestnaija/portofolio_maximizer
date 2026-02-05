@echo off
REM Daily automated trader with position persistence and intraday passes.
REM Designed for Windows Task Scheduler.
REM
REM Usage:
REM   run_daily_trader.bat
REM   set TICKERS=AAPL,MSFT,NVDA && run_daily_trader.bat
REM
REM Schedule via Task Scheduler:
REM   schtasks /create /tn "PortfolioMaximizer_DailyTrader" /tr "%~dp0run_daily_trader.bat" /sc DAILY /st 18:00 /f

set SCRIPT_DIR=%~dp0
set PYTHON_BIN=%SCRIPT_DIR%simpleTrader_env\Scripts\python.exe
set LOG_DIR=%SCRIPT_DIR%logs\daily_runs

REM Configurable via environment
if "%TICKERS%"=="" set TICKERS=AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,JPM,GS,V
if "%CYCLES%"=="" set CYCLES=1
if "%LOOKBACK_DAYS%"=="" set LOOKBACK_DAYS=365
if "%INITIAL_CAPITAL%"=="" set INITIAL_CAPITAL=25000
if "%RISK_MODE%"=="" set RISK_MODE=research_production
if "%INTRADAY_INTERVAL%"=="" set INTRADAY_INTERVAL=1h
if "%INTRADAY_HORIZON%"=="" set INTRADAY_HORIZON=6
if "%INTRADAY_LOOKBACK%"=="" set INTRADAY_LOOKBACK=30
if "%INTRADAY_CYCLES%"=="" set INTRADAY_CYCLES=3

REM Timestamp for log files
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DATETIME=%%I
set STAMP=%DATETIME:~0,8%_%DATETIME:~8,6%

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%PYTHON_BIN%" (
    echo [ERROR] Python not found at %PYTHON_BIN% >> "%LOG_DIR%\daily_trader_%STAMP%.log"
    exit /b 1
)

echo [START] Daily trader run at %DATE% %TIME% >> "%LOG_DIR%\daily_trader_%STAMP%.log"

REM === PASS 1: Daily signals ===
echo [PASS 1] Daily signals (interval=1d) >> "%LOG_DIR%\daily_trader_%STAMP%.log"
"%PYTHON_BIN%" "%SCRIPT_DIR%scripts\run_auto_trader.py" ^
    --tickers %TICKERS% ^
    --lookback-days %LOOKBACK_DAYS% ^
    --initial-capital %INITIAL_CAPITAL% ^
    --cycles %CYCLES% ^
    --sleep-seconds 10 ^
    --resume ^
    --bar-aware ^
    --persist-bar-state >> "%LOG_DIR%\daily_trader_%STAMP%.log" 2>&1

echo [PASS 1] Exit code: %ERRORLEVEL% >> "%LOG_DIR%\daily_trader_%STAMP%.log"

REM === PASS 2: Intraday signals (1h) ===
echo [PASS 2] Intraday signals (interval=%INTRADAY_INTERVAL%) >> "%LOG_DIR%\daily_trader_%STAMP%.log"
"%PYTHON_BIN%" "%SCRIPT_DIR%scripts\run_auto_trader.py" ^
    --tickers %TICKERS% ^
    --yfinance-interval %INTRADAY_INTERVAL% ^
    --lookback-days %INTRADAY_LOOKBACK% ^
    --forecast-horizon %INTRADAY_HORIZON% ^
    --initial-capital %INITIAL_CAPITAL% ^
    --cycles %INTRADAY_CYCLES% ^
    --sleep-seconds 10 ^
    --resume ^
    --bar-aware ^
    --persist-bar-state >> "%LOG_DIR%\daily_trader_%STAMP%.log" 2>&1

echo [PASS 2] Exit code: %ERRORLEVEL% >> "%LOG_DIR%\daily_trader_%STAMP%.log"

echo [END] Daily trader complete at %DATE% %TIME% >> "%LOG_DIR%\daily_trader_%STAMP%.log"
