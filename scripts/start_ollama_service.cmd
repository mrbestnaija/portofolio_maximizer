@echo off
REM Ollama Auto-Start Service for Portfolio Maximizer
REM Ensures Ollama is running for local LLM inference (deepseek-r1:8b, deepseek-r1:32b, qwen3:8b)
REM Registered as Windows scheduled task: "Ollama Serve (PMX)"

setlocal

set OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
set LOG_DIR=%~dp0..\logs\ollama
set LOG_FILE=%LOG_DIR%\ollama_serve_%DATE:~-4%%DATE:~4,2%%DATE:~7,2%.log

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Check if Ollama is already running
tasklist /FI "IMAGENAME eq ollama.exe" /NH | findstr /I "ollama.exe" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [%DATE% %TIME%] Ollama already running, skipping start >> "%LOG_FILE%"
    exit /b 0
)

REM Start Ollama serve in background
echo [%DATE% %TIME%] Starting Ollama serve... >> "%LOG_FILE%"
start "" /B "%OLLAMA_EXE%" serve >> "%LOG_FILE%" 2>&1

REM Wait for startup
timeout /t 5 /nobreak >nul

REM Verify it started
tasklist /FI "IMAGENAME eq ollama.exe" /NH | findstr /I "ollama.exe" >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [%DATE% %TIME%] Ollama started successfully >> "%LOG_FILE%"
) else (
    echo [%DATE% %TIME%] ERROR: Ollama failed to start >> "%LOG_FILE%"
    exit /b 1
)

endlocal
