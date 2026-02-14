1>2# : ^
r'''
@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Daily automated trader with position persistence and intraday passes.
REM Designed for Windows Task Scheduler.
REM
REM Usage:
REM   run_daily_trader.bat
REM   set TICKERS=AAPL,MSFT,NVDA && run_daily_trader.bat
REM   set PROOF_MODE=1 && set REQUIRE_PROFITABLE=1 && run_daily_trader.bat
REM
REM Schedule via Task Scheduler:
REM   schtasks /create /tn "PortfolioMaximizer_DailyTrader" /tr "%~dp0run_daily_trader.bat" /sc DAILY /st 18:00 /f

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR:~0,-1%"
set "PYTHON_BIN=%SCRIPT_DIR%simpleTrader_env\Scripts\python.exe"
set "AUTO_TRADER_SCRIPT=%SCRIPT_DIR%scripts\run_auto_trader.py"
set "GATE_LIFT_REPLAY_SCRIPT=%SCRIPT_DIR%scripts\run_gate_lift_replay.py"
set "PRODUCTION_GATE_SCRIPT=%SCRIPT_DIR%scripts\production_audit_gate.py"
set "AUDIT_SCRIPT=%SCRIPT_DIR%scripts\run_audit_event.py"
set "DASHBOARD_MANAGER_SCRIPT=%SCRIPT_DIR%scripts\windows_dashboard_manager.py"
set "SECURITY_PREFLIGHT_SCRIPT=%SCRIPT_DIR%scripts\security_preflight.py"
set "LOG_DIR=%SCRIPT_DIR%logs\daily_runs"
set "GATE_DIR=%SCRIPT_DIR%logs\audit_gate"
set "AUDIT_DIR=%SCRIPT_DIR%logs\run_audit"
set "SECURITY_DIR=%SCRIPT_DIR%logs\security"
set "TS_FORECAST_AUDIT_DIR=%SCRIPT_DIR%logs\forecast_audits"
set "TS_FORECAST_MONITOR_CONFIG=%SCRIPT_DIR%config\forecaster_monitoring.yml"
set "DB_PATH=%SCRIPT_DIR%data\portfolio_maximizer.db"

REM Configurable via environment
if "%TICKERS%"=="" set "TICKERS=AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,JPM,GS,V"
if "%CYCLES%"=="" set "CYCLES=1"
if "%LOOKBACK_DAYS%"=="" set "LOOKBACK_DAYS=365"
if "%INITIAL_CAPITAL%"=="" set "INITIAL_CAPITAL=25000"
if "%RISK_MODE%"=="" set "RISK_MODE=research_production"
if "%INTRADAY_INTERVAL%"=="" set "INTRADAY_INTERVAL=1h"
if "%INTRADAY_HORIZON%"=="" set "INTRADAY_HORIZON=6"
if "%INTRADAY_LOOKBACK%"=="" set "INTRADAY_LOOKBACK=30"
if "%INTRADAY_CYCLES%"=="" set "INTRADAY_CYCLES=3"
if "%PROOF_MODE%"=="" set "PROOF_MODE=1"
if "%SKIP_PRODUCTION_GATE%"=="" set "SKIP_PRODUCTION_GATE=0"
if "%REQUIRE_PROFITABLE%"=="" set "REQUIRE_PROFITABLE=1"
if "%ALLOW_INCONCLUSIVE_LIFT%"=="" set "ALLOW_INCONCLUSIVE_LIFT=0"
if "%STRICT_HOLDING_PERIOD%"=="" set "STRICT_HOLDING_PERIOD=0"
if "%EXECUTION_MODE%"=="" set "EXECUTION_MODE=live"
if "%ENABLE_DATA_CACHE%"=="" set "ENABLE_DATA_CACHE=0"
if "%ENABLE_CACHE_DELTAS%"=="" set "ENABLE_CACHE_DELTAS=0"
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
if "%ENABLE_GATE_LIFT_REPLAY%"=="" set "ENABLE_GATE_LIFT_REPLAY=0"
if "%GATE_LIFT_REPLAY_DAYS%"=="" set "GATE_LIFT_REPLAY_DAYS=0"
if "%GATE_LIFT_REPLAY_START_OFFSET_DAYS%"=="" set "GATE_LIFT_REPLAY_START_OFFSET_DAYS=1"
if "%GATE_LIFT_REPLAY_INTERVAL%"=="" set "GATE_LIFT_REPLAY_INTERVAL=1d"
if "%GATE_LIFT_REPLAY_STRICT%"=="" set "GATE_LIFT_REPLAY_STRICT=0"

REM Security-oriented Python runtime hardening for all subprocesses.
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

REM Force real-time provider path for production-style runs.
set "ENABLE_SYNTHETIC_PROVIDER="
set "ENABLE_SYNTHETIC_DATA_SOURCE="
set "SYNTHETIC_ONLY="
set "RUN_SYNTHETIC="
set "SYNTHETIC_DATA_MODE="

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%GATE_DIR%" mkdir "%GATE_DIR%"
if not exist "%AUDIT_DIR%" mkdir "%AUDIT_DIR%"
if not exist "%SECURITY_DIR%" mkdir "%SECURITY_DIR%"
if not exist "%TS_FORECAST_AUDIT_DIR%" mkdir "%TS_FORECAST_AUDIT_DIR%"

if not exist "%PYTHON_BIN%" (
    echo [ERROR] Python not found at %PYTHON_BIN%
    exit /b 1
)
if not exist "%AUTO_TRADER_SCRIPT%" (
    echo [ERROR] Auto trader script not found at %AUTO_TRADER_SCRIPT%
    exit /b 1
)
if not exist "%PRODUCTION_GATE_SCRIPT%" (
    echo [ERROR] Production gate script not found at %PRODUCTION_GATE_SCRIPT%
    exit /b 1
)
if not exist "%AUDIT_SCRIPT%" (
    echo [ERROR] Audit event script not found at %AUDIT_SCRIPT%
    exit /b 1
)
if not exist "%DASHBOARD_MANAGER_SCRIPT%" (
    echo [ERROR] Dashboard manager script not found at %DASHBOARD_MANAGER_SCRIPT%
    exit /b 1
)
if not exist "%SECURITY_PREFLIGHT_SCRIPT%" (
    echo [ERROR] Security preflight script not found at %SECURITY_PREFLIGHT_SCRIPT%
    exit /b 1
)
if "%ENABLE_GATE_LIFT_REPLAY%"=="1" if not exist "%GATE_LIFT_REPLAY_SCRIPT%" (
    echo [ERROR] Gate-lift replay script not found at %GATE_LIFT_REPLAY_SCRIPT%
    exit /b 1
)

set "STAMP="
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do if not "%%I"=="" set "STAMP=%%I"
if defined STAMP (
    set "STAMP=%STAMP:~0,8%_%STAMP:~8,6%"
) else (
    set "STAMP=%RANDOM%%RANDOM%"
)

set "RUN_ID=pmx_daily_%STAMP%_%RANDOM%%RANDOM%"
set "PARENT_RUN_ID=%PMX_PARENT_RUN_ID%"
set "PMX_RUN_ID=%RUN_ID%"
set "PMX_PARENT_RUN_ID=%PARENT_RUN_ID%"
set "LOG_FILE=%LOG_DIR%\daily_trader_%RUN_ID%.log"
set "GATE_JSON=%GATE_DIR%\production_gate_%RUN_ID%.json"
set "DASHBOARD_STATUS_JSON=%GATE_DIR%\dashboard_status_%RUN_ID%.json"
set "AUDIT_FILE=%AUDIT_DIR%\run_daily_trader_%RUN_ID%.jsonl"
set "SECURITY_JSON=%SECURITY_DIR%\security_preflight_%RUN_ID%.json"
set "REPLAY_JSON=%GATE_DIR%\gate_lift_replay_%RUN_ID%.json"
set /a SUBPROC_SEQ=0

set "PROOF_ARG="
if /I "%PROOF_MODE%"=="1" set "PROOF_ARG=--proof-mode"
set "REQUIRE_PROFITABLE_ARG="
if not "%REQUIRE_PROFITABLE%"=="0" set "REQUIRE_PROFITABLE_ARG=--require-profitable"
set "ALLOW_INCONCLUSIVE_ARG="
if "%ALLOW_INCONCLUSIVE_LIFT%"=="1" set "ALLOW_INCONCLUSIVE_ARG=--allow-inconclusive-lift"
set "STRICT_HOLDING_ARG="
if "%STRICT_HOLDING_PERIOD%"=="1" set "STRICT_HOLDING_ARG=--require-holding-period"

set "EXIT_CODE=0"

echo [START] Daily trader run at %DATE% %TIME% > "%LOG_FILE%"
echo [RUN] RUN_ID=%RUN_ID% PARENT_RUN_ID=%PARENT_RUN_ID% >> "%LOG_FILE%"
echo [CONFIG] EXECUTION_MODE=%EXECUTION_MODE% RISK_MODE=%RISK_MODE% >> "%LOG_FILE%"
echo [CONFIG] PROOF_MODE=%PROOF_MODE% REQUIRE_PROFITABLE=%REQUIRE_PROFITABLE% >> "%LOG_FILE%"
echo [CONFIG] TICKERS=%TICKERS% >> "%LOG_FILE%"
echo [CONFIG] DASHBOARD_API=%ENABLE_DASHBOARD_API% AUTO_OPEN=%AUTO_OPEN_DASHBOARD% PORT=%DASHBOARD_PORT% >> "%LOG_FILE%"
echo [CONFIG] SECURITY_CHECKS=%ENABLE_SECURITY_CHECKS% STRICT=%SECURITY_STRICT% REQUIRE_PIP_AUDIT=%SECURITY_REQUIRE_PIP_AUDIT% >> "%LOG_FILE%"
echo [CONFIG] SECURITY_IGNORE_VULN_IDS=%SECURITY_IGNORE_VULN_IDS% >> "%LOG_FILE%"
echo [CONFIG] GATE_LIFT_REPLAY=%ENABLE_GATE_LIFT_REPLAY% DAYS=%GATE_LIFT_REPLAY_DAYS% START_OFFSET=%GATE_LIFT_REPLAY_START_OFFSET_DAYS% STRICT=%GATE_LIFT_REPLAY_STRICT% >> "%LOG_FILE%"

call :audit_event "RUN_START" "STARTED" "bootstrap" "" "0" "Daily trader started"

if "%ENABLE_DASHBOARD_API%"=="1" (
    set /a SUBPROC_SEQ+=1
    set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
    "%PYTHON_BIN%" "%AUDIT_SCRIPT%" ^
        --audit-file "%AUDIT_FILE%" ^
        --run-id "%RUN_ID%" ^
        --parent-run-id "%PARENT_RUN_ID%" ^
        --script-name "run_daily_trader.bat" ^
        --event "STEP_START" ^
        --status "RUNNING" ^
        --step "dashboard_api" ^
        --subprocess-id "!SUBPROC_ID!" ^
        --exit-code 0 ^
        --message "Ensuring dashboard bridge localhost API" ^
        --log-file "%LOG_FILE%" >nul 2>&1
    set "DASHBOARD_OPEN_ARG="
    if "%AUTO_OPEN_DASHBOARD%"=="1" set "DASHBOARD_OPEN_ARG=--open-browser"
    set "DASHBOARD_PERSIST_ARG=--persist-snapshot"
    if "%DASHBOARD_PERSIST%"=="0" set "DASHBOARD_PERSIST_ARG=--no-persist-snapshot"
    set "DASHBOARD_STRICT_ARG=--strict"
    if "%DASHBOARD_API_STRICT%"=="0" set "DASHBOARD_STRICT_ARG=--no-strict"
    "%PYTHON_BIN%" "%DASHBOARD_MANAGER_SCRIPT%" ensure ^
        --root "%ROOT_DIR%" ^
        --python-bin "%PYTHON_BIN%" ^
        --port %DASHBOARD_PORT% ^
        --db-path "%DB_PATH%" ^
        --status-json "%DASHBOARD_STATUS_JSON%" ^
        --caller "run_daily_trader.bat" ^
        --run-id "%RUN_ID%" ^
        --require-bridge ^
        !DASHBOARD_PERSIST_ARG! ^
        !DASHBOARD_OPEN_ARG! ^
        !DASHBOARD_STRICT_ARG! >> "%LOG_FILE%" 2>&1
    set "DASH_RC=!ERRORLEVEL!"
    echo [DASHBOARD] Exit code: !DASH_RC! >> "%LOG_FILE%"
    if "!DASH_RC!"=="0" (
        "%PYTHON_BIN%" "%AUDIT_SCRIPT%" ^
            --audit-file "%AUDIT_FILE%" ^
            --run-id "%RUN_ID%" ^
            --parent-run-id "%PARENT_RUN_ID%" ^
            --script-name "run_daily_trader.bat" ^
            --event "STEP_END" ^
            --status "SUCCESS" ^
            --step "dashboard_api" ^
            --subprocess-id "!SUBPROC_ID!" ^
            --exit-code !DASH_RC! ^
            --message "Dashboard API ready" ^
            --log-file "%LOG_FILE%" >nul 2>&1
    ) else (
        "%PYTHON_BIN%" "%AUDIT_SCRIPT%" ^
            --audit-file "%AUDIT_FILE%" ^
            --run-id "%RUN_ID%" ^
            --parent-run-id "%PARENT_RUN_ID%" ^
            --script-name "run_daily_trader.bat" ^
            --event "STEP_END" ^
            --status "FAIL" ^
            --step "dashboard_api" ^
            --subprocess-id "!SUBPROC_ID!" ^
            --exit-code !DASH_RC! ^
            --message "Dashboard API startup failed" ^
            --log-file "%LOG_FILE%" >nul 2>&1
        set "EXIT_CODE=!DASH_RC!"
    )
) else (
    "%PYTHON_BIN%" "%AUDIT_SCRIPT%" ^
        --audit-file "%AUDIT_FILE%" ^
        --run-id "%RUN_ID%" ^
        --parent-run-id "%PARENT_RUN_ID%" ^
        --script-name "run_daily_trader.bat" ^
        --event "STEP_END" ^
        --status "SKIPPED" ^
        --step "dashboard_api" ^
        --subprocess-id "" ^
        --exit-code 0 ^
        --message "Dashboard API disabled by ENABLE_DASHBOARD_API=0" ^
        --log-file "%LOG_FILE%" >nul 2>&1
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
    "%PYTHON_BIN%" "%SECURITY_PREFLIGHT_SCRIPT%" ^
        --python-bin "%PYTHON_BIN%" ^
        --output-json "%SECURITY_JSON%" ^
        --caller "run_daily_trader.bat" ^
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
        set "EXIT_CODE=!SEC_RC!"
        if not "%SECURITY_HARD_FAIL%"=="0" goto :finalize
    )
) else (
    call :audit_event "STEP_END" "SKIPPED" "security_preflight" "" "0" "Security preflight disabled by ENABLE_SECURITY_CHECKS=0"
)

REM === PASS 1: Daily signals (real-time/live) ===
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "pass1_daily_signals" "!SUBPROC_ID!" "0" "Daily signal pass started"
echo [PASS 1] Daily signals (interval=1d) >> "%LOG_FILE%"
"%PYTHON_BIN%" "%AUTO_TRADER_SCRIPT%" ^
    --tickers %TICKERS% ^
    --lookback-days %LOOKBACK_DAYS% ^
    --initial-capital %INITIAL_CAPITAL% ^
    --cycles %CYCLES% ^
    --sleep-seconds 10 ^
    --resume ^
    --bar-aware ^
    --persist-bar-state ^
    %PROOF_ARG% >> "%LOG_FILE%" 2>&1

set "PASS1_RC=%ERRORLEVEL%"
echo [PASS 1] Exit code: %PASS1_RC% >> "%LOG_FILE%"
if "%PASS1_RC%"=="0" (
    call :audit_event "STEP_END" "SUCCESS" "pass1_daily_signals" "!SUBPROC_ID!" "%PASS1_RC%" "Daily signal pass completed"
) else (
    call :audit_event "STEP_END" "FAIL" "pass1_daily_signals" "!SUBPROC_ID!" "%PASS1_RC%" "Daily signal pass failed"
)
if not "%PASS1_RC%"=="0" set "EXIT_CODE=%PASS1_RC%"

REM === PASS 2: Intraday signals (1h) ===
set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "pass2_intraday_signals" "!SUBPROC_ID!" "0" "Intraday signal pass started"
echo [PASS 2] Intraday signals (interval=%INTRADAY_INTERVAL%) >> "%LOG_FILE%"
"%PYTHON_BIN%" "%AUTO_TRADER_SCRIPT%" ^
    --tickers %TICKERS% ^
    --yfinance-interval %INTRADAY_INTERVAL% ^
    --lookback-days %INTRADAY_LOOKBACK% ^
    --forecast-horizon %INTRADAY_HORIZON% ^
    --initial-capital %INITIAL_CAPITAL% ^
    --cycles %INTRADAY_CYCLES% ^
    --sleep-seconds 10 ^
    --resume ^
    --bar-aware ^
    --persist-bar-state ^
    %PROOF_ARG% >> "%LOG_FILE%" 2>&1

set "PASS2_RC=%ERRORLEVEL%"
echo [PASS 2] Exit code: %PASS2_RC% >> "%LOG_FILE%"
if "%PASS2_RC%"=="0" (
    call :audit_event "STEP_END" "SUCCESS" "pass2_intraday_signals" "!SUBPROC_ID!" "%PASS2_RC%" "Intraday signal pass completed"
) else (
    call :audit_event "STEP_END" "FAIL" "pass2_intraday_signals" "!SUBPROC_ID!" "%PASS2_RC%" "Intraday signal pass failed"
)
if not "%PASS2_RC%"=="0" set "EXIT_CODE=%PASS2_RC%"

set "RUN_GATE_LIFT_REPLAY=0"
if "%ENABLE_GATE_LIFT_REPLAY%"=="1" if not "%GATE_LIFT_REPLAY_DAYS%"=="0" set "RUN_GATE_LIFT_REPLAY=1"

if "%RUN_GATE_LIFT_REPLAY%"=="1" (
    set /a SUBPROC_SEQ+=1
    set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
    call :audit_event "STEP_START" "RUNNING" "gate_lift_replay" "!SUBPROC_ID!" "0" "Running historical replay for gate-lift evidence"
    echo [REPLAY] Running gate-lift replay (days=%GATE_LIFT_REPLAY_DAYS%, start_offset=%GATE_LIFT_REPLAY_START_OFFSET_DAYS%) >> "%LOG_FILE%"
    set "REPLAY_STRICT_ARG="
    if "%GATE_LIFT_REPLAY_STRICT%"=="1" set "REPLAY_STRICT_ARG=--strict"
    "%PYTHON_BIN%" "%GATE_LIFT_REPLAY_SCRIPT%" ^
        --python-bin "%PYTHON_BIN%" ^
        --auto-trader-script "%AUTO_TRADER_SCRIPT%" ^
        --tickers "%TICKERS%" ^
        --lookback-days %LOOKBACK_DAYS% ^
        --initial-capital %INITIAL_CAPITAL% ^
        --days %GATE_LIFT_REPLAY_DAYS% ^
        --start-offset-days %GATE_LIFT_REPLAY_START_OFFSET_DAYS% ^
        --yfinance-interval %GATE_LIFT_REPLAY_INTERVAL% ^
        %PROOF_ARG% ^
        --resume ^
        --run-id "%RUN_ID%" ^
        --parent-run-id "%PARENT_RUN_ID%" ^
        --audit-file "%AUDIT_FILE%" ^
        --log-file "%LOG_FILE%" ^
        --output-json "%REPLAY_JSON%" ^
        !REPLAY_STRICT_ARG! >> "%LOG_FILE%" 2>&1
    set "REPLAY_RC=!ERRORLEVEL!"
    echo [REPLAY] Exit code: !REPLAY_RC! >> "%LOG_FILE%"
    if "!REPLAY_RC!"=="0" (
        call :audit_event "STEP_END" "SUCCESS" "gate_lift_replay" "!SUBPROC_ID!" "!REPLAY_RC!" "Gate-lift replay completed"
    ) else (
        call :audit_event "STEP_END" "FAIL" "gate_lift_replay" "!SUBPROC_ID!" "!REPLAY_RC!" "Gate-lift replay failed"
        if "%GATE_LIFT_REPLAY_STRICT%"=="1" set "EXIT_CODE=!REPLAY_RC!"
    )
) else (
    if "%ENABLE_GATE_LIFT_REPLAY%"=="1" (
        call :audit_event "STEP_END" "SKIPPED" "gate_lift_replay" "" "0" "Gate-lift replay enabled but days=0"
    ) else (
        call :audit_event "STEP_END" "SKIPPED" "gate_lift_replay" "" "0" "Gate-lift replay disabled by configuration"
    )
)

if "%SKIP_PRODUCTION_GATE%"=="1" (
    echo [GATE] Skipped ^(SKIP_PRODUCTION_GATE=1^) >> "%LOG_FILE%"
    call :audit_event "STEP_END" "SKIPPED" "production_audit_gate" "" "0" "Production gate skipped"
) else (
    set /a SUBPROC_SEQ+=1
    set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
    call :audit_event "STEP_START" "RUNNING" "production_audit_gate" "!SUBPROC_ID!" "0" "Running production profitability gate"
    echo [GATE] Running production audit gate >> "%LOG_FILE%"
    "%PYTHON_BIN%" "%PRODUCTION_GATE_SCRIPT%" ^
        --db "%DB_PATH%" ^
        --audit-dir "%TS_FORECAST_AUDIT_DIR%" ^
        --monitor-config "%TS_FORECAST_MONITOR_CONFIG%" ^
        --max-files 500 ^
        --output-json "%GATE_JSON%" ^
        %REQUIRE_PROFITABLE_ARG% ^
        %ALLOW_INCONCLUSIVE_ARG% ^
        %STRICT_HOLDING_ARG% >> "%LOG_FILE%" 2>&1
    set "GATE_RC=!ERRORLEVEL!"
    echo [GATE] Exit code: !GATE_RC! >> "%LOG_FILE%"
    if "!GATE_RC!"=="0" (
        call :audit_event "STEP_END" "SUCCESS" "production_audit_gate" "!SUBPROC_ID!" "!GATE_RC!" "Production gate passed"
    ) else (
        call :audit_event "STEP_END" "FAIL" "production_audit_gate" "!SUBPROC_ID!" "!GATE_RC!" "Production gate failed"
    )
if not "!GATE_RC!"=="0" set "EXIT_CODE=!GATE_RC!"
)

:finalize
echo [END] Daily trader complete at %DATE% %TIME% (exit=%EXIT_CODE%) >> "%LOG_FILE%"
set "RUN_STATUS=SUCCESS"
if not "%EXIT_CODE%"=="0" set "RUN_STATUS=FAIL"
call :audit_event "RUN_END" "%RUN_STATUS%" "finalize" "" "%EXIT_CODE%" "Daily trader finished"
echo [LOG] %LOG_FILE%
echo [AUDIT] %AUDIT_FILE%
if not "%SKIP_PRODUCTION_GATE%"=="1" echo [GATE] %GATE_JSON%
exit /b %EXIT_CODE%

:audit_event
set "AE_EVENT=%~1"
set "AE_STATUS=%~2"
set "AE_STEP=%~3"
set "AE_SUBPROC=%~4"
set "AE_EXIT=%~5"
set "AE_MSG=%~6"
"%PYTHON_BIN%" "%AUDIT_SCRIPT%" ^
    --audit-file "%AUDIT_FILE%" ^
    --run-id "%RUN_ID%" ^
    --parent-run-id "%PARENT_RUN_ID%" ^
    --script-name "run_daily_trader.bat" ^
    --event "%AE_EVENT%" ^
    --status "%AE_STATUS%" ^
    --step "%AE_STEP%" ^
    --subprocess-id "%AE_SUBPROC%" ^
    --exit-code %AE_EXIT% ^
    --message "%AE_MSG%" ^
    --log-file "%LOG_FILE%" >nul 2>&1
exit /b 0

goto :eof
'''
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    # Support accidental `python run_daily_trader.bat` invocation by delegating to cmd.
    bat_path = str(Path(__file__).resolve())
    exit_code = subprocess.call(["cmd.exe", "/d", "/c", bat_path, *sys.argv[1:]])
    raise SystemExit(exit_code)
