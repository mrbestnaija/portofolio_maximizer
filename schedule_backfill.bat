@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Lightweight scheduled-task wrapper with production gate logging.
REM Default task: auto_trader_core (WSL cron wrapper).
REM Optional tasks: forecast_gate, production_gate, or any production_cron.sh task.

set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR:~0,-1%"
set "PYTHON_BIN=%SCRIPT_DIR%simpleTrader_env\Scripts\python.exe"
set "FORECAST_AUDIT_SCRIPT=%SCRIPT_DIR%scripts\check_forecast_audits.py"
set "PRODUCTION_GATE_SCRIPT=%SCRIPT_DIR%scripts\production_audit_gate.py"
set "AUDIT_EVENT_SCRIPT=%SCRIPT_DIR%scripts\run_audit_event.py"
set "DASHBOARD_MANAGER_SCRIPT=%SCRIPT_DIR%scripts\windows_dashboard_manager.py"
set "SECURITY_PREFLIGHT_SCRIPT=%SCRIPT_DIR%scripts\security_preflight.py"
set "LOG_DIR=%SCRIPT_DIR%logs\scheduled_tasks"
set "AUDIT_DIR=%SCRIPT_DIR%logs\run_audit"
set "GATE_DIR=%SCRIPT_DIR%logs\audit_gate"
set "SECURITY_DIR=%SCRIPT_DIR%logs\security"
set "TS_FORECAST_AUDIT_DIR=%SCRIPT_DIR%logs\forecast_audits"
set "TS_FORECAST_MONITOR_CONFIG=%SCRIPT_DIR%config\forecaster_monitoring.yml"
set "DB_PATH=%SCRIPT_DIR%data\portfolio_maximizer.db"

set "TASK=%~1"
if "%TASK%"=="" set "TASK=auto_trader_core"

if "%RUN_PRODUCTION_GATE%"=="" (
    if /I "%TASK%"=="auto_trader_core" (
        set "RUN_PRODUCTION_GATE=1"
    ) else (
        set "RUN_PRODUCTION_GATE=0"
    )
)
if "%REQUIRE_PROFITABLE%"=="" set "REQUIRE_PROFITABLE=1"
if "%ALLOW_INCONCLUSIVE_LIFT%"=="" set "ALLOW_INCONCLUSIVE_LIFT=0"
if "%STRICT_HOLDING_PERIOD%"=="" set "STRICT_HOLDING_PERIOD=0"
if "%EXECUTION_MODE%"=="" set "EXECUTION_MODE=live"
if "%RISK_MODE%"=="" set "RISK_MODE=research_production"
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

REM Security-oriented Python runtime hardening for all subprocesses.
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

REM Force live/provider-backed runs.
set "ENABLE_SYNTHETIC_PROVIDER="
set "ENABLE_SYNTHETIC_DATA_SOURCE="
set "SYNTHETIC_ONLY="
set "RUN_SYNTHETIC="
set "SYNTHETIC_DATA_MODE="

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%AUDIT_DIR%" mkdir "%AUDIT_DIR%"
if not exist "%GATE_DIR%" mkdir "%GATE_DIR%"
if not exist "%SECURITY_DIR%" mkdir "%SECURITY_DIR%"
if not exist "%TS_FORECAST_AUDIT_DIR%" mkdir "%TS_FORECAST_AUDIT_DIR%"

if not exist "%PYTHON_BIN%" (
    echo [ERROR] Python interpreter not found at %PYTHON_BIN%
    exit /b 1
)
if not exist "%FORECAST_AUDIT_SCRIPT%" (
    echo [ERROR] Forecast audit script not found at %FORECAST_AUDIT_SCRIPT%
    exit /b 1
)
if not exist "%PRODUCTION_GATE_SCRIPT%" (
    echo [ERROR] Production gate script not found at %PRODUCTION_GATE_SCRIPT%
    exit /b 1
)
if not exist "%AUDIT_EVENT_SCRIPT%" (
    echo [ERROR] Audit event script not found at %AUDIT_EVENT_SCRIPT%
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

set "STAMP="
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do if not "%%I"=="" set "STAMP=%%I"
if defined STAMP (
    set "STAMP=%STAMP:~0,8%_%STAMP:~8,6%"
) else (
    set "STAMP=%RANDOM%%RANDOM%"
)

set "RUN_ID=pmx_schedule_%TASK%_%STAMP%_%RANDOM%%RANDOM%"
set "PARENT_RUN_ID=%PMX_PARENT_RUN_ID%"
set "PMX_RUN_ID=%RUN_ID%"
set "PMX_PARENT_RUN_ID=%PARENT_RUN_ID%"
set "LOG_FILE=%LOG_DIR%\%TASK%_%RUN_ID%.log"
set "GATE_JSON=%LOG_DIR%\production_gate_%RUN_ID%.json"
set "DASHBOARD_STATUS_JSON=%GATE_DIR%\dashboard_status_%RUN_ID%.json"
set "AUDIT_FILE=%AUDIT_DIR%\schedule_backfill_%RUN_ID%.jsonl"
set "SECURITY_JSON=%SECURITY_DIR%\security_preflight_%RUN_ID%.json"
set /a SUBPROC_SEQ=0

set "REQUIRE_PROFITABLE_ARG="
if not "%REQUIRE_PROFITABLE%"=="0" set "REQUIRE_PROFITABLE_ARG=--require-profitable"
set "ALLOW_INCONCLUSIVE_ARG="
if "%ALLOW_INCONCLUSIVE_LIFT%"=="1" set "ALLOW_INCONCLUSIVE_ARG=--allow-inconclusive-lift"
set "STRICT_HOLDING_ARG="
if "%STRICT_HOLDING_PERIOD%"=="1" set "STRICT_HOLDING_ARG=--require-holding-period"

set "EXIT_CODE=0"
echo [START] schedule_backfill task=%TASK% at %DATE% %TIME% > "%LOG_FILE%"
echo [RUN] RUN_ID=%RUN_ID% PARENT_RUN_ID=%PARENT_RUN_ID% >> "%LOG_FILE%"
echo [CONFIG] EXECUTION_MODE=%EXECUTION_MODE% RISK_MODE=%RISK_MODE% >> "%LOG_FILE%"
echo [CONFIG] RUN_PRODUCTION_GATE=%RUN_PRODUCTION_GATE% REQUIRE_PROFITABLE=%REQUIRE_PROFITABLE% >> "%LOG_FILE%"
echo [CONFIG] DASHBOARD_API=%ENABLE_DASHBOARD_API% AUTO_OPEN=%AUTO_OPEN_DASHBOARD% PORT=%DASHBOARD_PORT% >> "%LOG_FILE%"
echo [CONFIG] SECURITY_CHECKS=%ENABLE_SECURITY_CHECKS% STRICT=%SECURITY_STRICT% REQUIRE_PIP_AUDIT=%SECURITY_REQUIRE_PIP_AUDIT% >> "%LOG_FILE%"
echo [CONFIG] SECURITY_IGNORE_VULN_IDS=%SECURITY_IGNORE_VULN_IDS% >> "%LOG_FILE%"

call :audit_event "RUN_START" "STARTED" "bootstrap" "" "0" "Scheduled backfill wrapper started"

if "%ENABLE_DASHBOARD_API%"=="1" (
    set /a SUBPROC_SEQ+=1
    set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
    call :audit_event "STEP_START" "RUNNING" "dashboard_api" "!SUBPROC_ID!" "0" "Ensuring dashboard bridge + localhost API"
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
        --caller "schedule_backfill.bat" ^
        --run-id "%RUN_ID%" ^
        --require-bridge ^
        !DASHBOARD_PERSIST_ARG! ^
        !DASHBOARD_OPEN_ARG! ^
        !DASHBOARD_STRICT_ARG! >> "%LOG_FILE%" 2>&1
    set "DASH_RC=!ERRORLEVEL!"
    echo [DASHBOARD] Exit code: !DASH_RC! >> "%LOG_FILE%"
    if "!DASH_RC!"=="0" (
        call :audit_event "STEP_END" "SUCCESS" "dashboard_api" "!SUBPROC_ID!" "!DASH_RC!" "Dashboard API ready"
    ) else (
        call :audit_event "STEP_END" "FAIL" "dashboard_api" "!SUBPROC_ID!" "!DASH_RC!" "Dashboard API startup failed"
        set "EXIT_CODE=!DASH_RC!"
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
    "%PYTHON_BIN%" "%SECURITY_PREFLIGHT_SCRIPT%" ^
        --python-bin "%PYTHON_BIN%" ^
        --output-json "%SECURITY_JSON%" ^
        --caller "schedule_backfill.bat" ^
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
        if not "%SECURITY_HARD_FAIL%"=="0" goto :finish
    )
) else (
    call :audit_event "STEP_END" "SKIPPED" "security_preflight" "" "0" "Security preflight disabled by ENABLE_SECURITY_CHECKS=0"
)

if /I "%TASK%"=="auto_trader_core" (
    set /a SUBPROC_SEQ+=1
    set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
    call :audit_event "STEP_START" "RUNNING" "auto_trader_core" "!SUBPROC_ID!" "0" "Starting auto_trader_core task"
    where wsl >nul 2>&1 || (
        echo [ERROR] WSL not found; cannot run auto_trader_core >> "%LOG_FILE%"
        call :audit_event "STEP_END" "FAIL" "auto_trader_core" "!SUBPROC_ID!" "1" "WSL runtime missing"
        set "EXIT_CODE=1"
        goto :after_task
    )
    echo [TASK] Running auto_trader_core via production_cron.sh >> "%LOG_FILE%"
    wsl --cd "%SCRIPT_DIR%" bash -lc "EXECUTION_MODE=live RISK_MODE=%RISK_MODE% bash/production_cron.sh auto_trader_core" >> "%LOG_FILE%" 2>&1
    set "TASK_RC=!ERRORLEVEL!"
    echo [TASK] auto_trader_core exit=!TASK_RC! >> "%LOG_FILE%"
    if "!TASK_RC!"=="0" (
        call :audit_event "STEP_END" "SUCCESS" "auto_trader_core" "!SUBPROC_ID!" "!TASK_RC!" "auto_trader_core completed"
    ) else (
        call :audit_event "STEP_END" "FAIL" "auto_trader_core" "!SUBPROC_ID!" "!TASK_RC!" "auto_trader_core failed"
    )
    if not "!TASK_RC!"=="0" set "EXIT_CODE=!TASK_RC!"
    goto :after_task
)

if /I "%TASK%"=="forecast_gate" (
    set /a SUBPROC_SEQ+=1
    set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
    call :audit_event "STEP_START" "RUNNING" "forecast_gate" "!SUBPROC_ID!" "0" "Running forecast lift gate"
    echo [TASK] Running forecast lift gate >> "%LOG_FILE%"
    "%PYTHON_BIN%" "%FORECAST_AUDIT_SCRIPT%" ^
        --audit-dir "%TS_FORECAST_AUDIT_DIR%" ^
        --config-path "%TS_FORECAST_MONITOR_CONFIG%" ^
        --max-files 500 >> "%LOG_FILE%" 2>&1
    set "TASK_RC=!ERRORLEVEL!"
    echo [TASK] forecast_gate exit=!TASK_RC! >> "%LOG_FILE%"
    if "!TASK_RC!"=="0" (
        call :audit_event "STEP_END" "SUCCESS" "forecast_gate" "!SUBPROC_ID!" "!TASK_RC!" "Forecast gate passed"
    ) else (
        call :audit_event "STEP_END" "FAIL" "forecast_gate" "!SUBPROC_ID!" "!TASK_RC!" "Forecast gate failed"
    )
    if not "!TASK_RC!"=="0" set "EXIT_CODE=!TASK_RC!"
    goto :after_task
)

if /I "%TASK%"=="production_gate" (
    set "RUN_PRODUCTION_GATE=1"
    call :audit_event "STEP_END" "SKIPPED" "scheduled_task_body" "" "0" "production_gate task selected; skipping cron body"
    echo [TASK] production_gate selected; skipping cron task body >> "%LOG_FILE%"
    goto :after_task
)

set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "production_cron_%TASK%" "!SUBPROC_ID!" "0" "Running production_cron.sh task"
where wsl >nul 2>&1 || (
    echo [ERROR] WSL not found; cannot run production_cron task %TASK% >> "%LOG_FILE%"
    call :audit_event "STEP_END" "FAIL" "production_cron_%TASK%" "!SUBPROC_ID!" "1" "WSL runtime missing"
    set "EXIT_CODE=1"
    goto :after_task
)
echo [TASK] Running generic production_cron task: %TASK% >> "%LOG_FILE%"
wsl --cd "%SCRIPT_DIR%" bash -lc "EXECUTION_MODE=live RISK_MODE=%RISK_MODE% bash/production_cron.sh %TASK%" >> "%LOG_FILE%" 2>&1
set "TASK_RC=%ERRORLEVEL%"
echo [TASK] %TASK% exit=%TASK_RC% >> "%LOG_FILE%"
if "%TASK_RC%"=="0" (
    call :audit_event "STEP_END" "SUCCESS" "production_cron_%TASK%" "!SUBPROC_ID!" "%TASK_RC%" "production_cron task completed"
) else (
    call :audit_event "STEP_END" "FAIL" "production_cron_%TASK%" "!SUBPROC_ID!" "%TASK_RC%" "production_cron task failed"
)
if not "%TASK_RC%"=="0" set "EXIT_CODE=%TASK_RC%"

:after_task
if not "%RUN_PRODUCTION_GATE%"=="1" (
    call :audit_event "STEP_END" "SKIPPED" "production_audit_gate" "" "0" "Production gate disabled for this task"
    goto :finish
)

set /a SUBPROC_SEQ+=1
set "SUBPROC_ID=%RUN_ID%_S!SUBPROC_SEQ!"
call :audit_event "STEP_START" "RUNNING" "production_audit_gate" "!SUBPROC_ID!" "0" "Running production profitability gate"
echo [GATE] Running production profitability gate >> "%LOG_FILE%"
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
echo [GATE] exit=!GATE_RC! >> "%LOG_FILE%"
if "!GATE_RC!"=="0" (
    call :audit_event "STEP_END" "SUCCESS" "production_audit_gate" "!SUBPROC_ID!" "!GATE_RC!" "Production gate passed"
) else (
    call :audit_event "STEP_END" "FAIL" "production_audit_gate" "!SUBPROC_ID!" "!GATE_RC!" "Production gate failed"
)
if not "!GATE_RC!"=="0" set "EXIT_CODE=!GATE_RC!"

:finish
echo [END] schedule_backfill complete (exit=%EXIT_CODE%) at %DATE% %TIME% >> "%LOG_FILE%"
set "RUN_STATUS=SUCCESS"
if not "%EXIT_CODE%"=="0" set "RUN_STATUS=FAIL"
call :audit_event "RUN_END" "%RUN_STATUS%" "finalize" "" "%EXIT_CODE%" "schedule_backfill finished"
echo [LOG] %LOG_FILE%
echo [AUDIT] %AUDIT_FILE%
if "%RUN_PRODUCTION_GATE%"=="1" echo [GATE] %GATE_JSON%
exit /b %EXIT_CODE%

:audit_event
set "AE_EVENT=%~1"
set "AE_STATUS=%~2"
set "AE_STEP=%~3"
set "AE_SUBPROC=%~4"
set "AE_EXIT=%~5"
set "AE_MSG=%~6"
"%PYTHON_BIN%" "%AUDIT_EVENT_SCRIPT%" ^
    --audit-file "%AUDIT_FILE%" ^
    --run-id "%RUN_ID%" ^
    --parent-run-id "%PARENT_RUN_ID%" ^
    --script-name "schedule_backfill.bat" ^
    --event "%AE_EVENT%" ^
    --status "%AE_STATUS%" ^
    --step "%AE_STEP%" ^
    --subprocess-id "%AE_SUBPROC%" ^
    --exit-code %AE_EXIT% ^
    --message "%AE_MSG%" ^
    --log-file "%LOG_FILE%" >nul 2>&1
exit /b 0
