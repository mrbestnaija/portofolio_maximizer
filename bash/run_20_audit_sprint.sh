#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve Python binary
if [[ -x "${ROOT_DIR}/simpleTrader_env/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/bin/python"
elif [[ -x "${ROOT_DIR}/simpleTrader_env/Scripts/python.exe" ]]; then
    PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/Scripts/python.exe"
else
    echo "[ERROR] Virtual environment Python not found under ${ROOT_DIR}/simpleTrader_env"
    exit 1
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
AUDIT_RUNS="${AUDIT_RUNS:-20}"
LOG_DIR="${ROOT_DIR}/logs/audit_sprint/${RUN_TAG}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run.log"

touch "${LOG_FILE}"

log() {
    echo "$*" | tee -a "${LOG_FILE}"
}

# === Environment defaults ===
export EXECUTION_MODE="${EXECUTION_MODE:-live}"
export TS_FORECAST_AUDIT_DIR="${TS_FORECAST_AUDIT_DIR:-${ROOT_DIR}/logs/forecast_audits}"
export TS_FORECAST_MONITOR_CONFIG="${TS_FORECAST_MONITOR_CONFIG:-${ROOT_DIR}/config/forecaster_monitoring.yml}"
export DASHBOARD_PERSIST="${DASHBOARD_PERSIST:-1}"
export DASHBOARD_KEEP_ALIVE="${DASHBOARD_KEEP_ALIVE:-1}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
export ENABLE_GPU_PARALLEL="${ENABLE_GPU_PARALLEL:-1}"
export PIPELINE_DEVICE="${PIPELINE_DEVICE:-cuda}"
export RISK_MODE="${RISK_MODE:-research_production}"
export ENABLE_DATA_CACHE="${ENABLE_DATA_CACHE:-0}"
export ENABLE_CACHE_DELTAS="${ENABLE_CACHE_DELTAS:-0}"
unset RUN_SYNTHETIC SYNTHETIC_DATA_MODE

TICKERS="${TICKERS:-AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,JPM,GS,V}"
CYCLES="${CYCLES:-1}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-25000}"
INTRADAY_INTERVAL="${INTRADAY_INTERVAL:-1h}"
INTRADAY_HORIZON="${INTRADAY_HORIZON:-6}"
INTRADAY_LOOKBACK="${INTRADAY_LOOKBACK:-30}"
INTRADAY_CYCLES="${INTRADAY_CYCLES:-3}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-15}"
WAIT_BETWEEN_RUNS_SECONDS="${WAIT_BETWEEN_RUNS_SECONDS:-0}"
AUTO_WAIT_FOR_NEW_BARS="${AUTO_WAIT_FOR_NEW_BARS:-1}"
PROOF_MODE="${PROOF_MODE:-1}"

mkdir -p "${TS_FORECAST_AUDIT_DIR}"

# Normalize config path if relative
if [[ "${TS_FORECAST_MONITOR_CONFIG}" != /* && "${TS_FORECAST_MONITOR_CONFIG}" != [A-Za-z]:* ]]; then
    TS_FORECAST_MONITOR_CONFIG="${ROOT_DIR}/${TS_FORECAST_MONITOR_CONFIG}"
fi

interval_to_seconds() {
    local raw="$1"
    raw="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${raw}" =~ ^([0-9]+)(m|min)$ ]]; then
        echo $((BASH_REMATCH[1] * 60))
        return
    fi
    if [[ "${raw}" =~ ^([0-9]+)h$ ]]; then
        echo $((BASH_REMATCH[1] * 3600))
        return
    fi
    if [[ "${raw}" =~ ^([0-9]+)d$ ]]; then
        echo $((BASH_REMATCH[1] * 86400))
        return
    fi
    if [[ "${raw}" =~ ^([0-9]+)w$ ]]; then
        echo $((BASH_REMATCH[1] * 604800))
        return
    fi
    echo 0
}

if [[ "${AUTO_WAIT_FOR_NEW_BARS}" == "1" ]] && [[ "${WAIT_BETWEEN_RUNS_SECONDS}" -eq 0 ]]; then
    inferred_wait="$(interval_to_seconds "${INTRADAY_INTERVAL}")"
    if [[ "${inferred_wait}" -gt 0 ]]; then
        WAIT_BETWEEN_RUNS_SECONDS="${inferred_wait}"
    fi
fi

log "[RUNBOOK] 20-audit sprint started: ${RUN_TAG}"
log "[RUNBOOK] Repo: ${ROOT_DIR}"
log "[RUNBOOK] Python: ${PYTHON_BIN}"
log "[RUNBOOK] Logs: ${LOG_DIR}"
log "[RUNBOOK] EXECUTION_MODE=${EXECUTION_MODE} | PIPELINE_DEVICE=${PIPELINE_DEVICE} | ENABLE_GPU_PARALLEL=${ENABLE_GPU_PARALLEL}"
log "[RUNBOOK] RISK_MODE=${RISK_MODE}"
log "[RUNBOOK] PROOF_MODE=${PROOF_MODE}"
log "[RUNBOOK] ENABLE_DATA_CACHE=${ENABLE_DATA_CACHE} | ENABLE_CACHE_DELTAS=${ENABLE_CACHE_DELTAS}"
log "[RUNBOOK] TS_FORECAST_MONITOR_CONFIG=${TS_FORECAST_MONITOR_CONFIG}"
log "[RUNBOOK] TS_FORECAST_AUDIT_DIR=${TS_FORECAST_AUDIT_DIR}"
log "[RUNBOOK] TICKERS=${TICKERS}"
log "[RUNBOOK] AUDIT_RUNS=${AUDIT_RUNS}"
log ""
if [[ "${WAIT_BETWEEN_RUNS_SECONDS}" -eq 0 ]]; then
    log "[WARN] WAIT_BETWEEN_RUNS_SECONDS=0; sequential runs may reuse cached market data."
else
    log "[RUNBOOK] WAIT_BETWEEN_RUNS_SECONDS=${WAIT_BETWEEN_RUNS_SECONDS}"
fi

GPU_MONITOR_PID=""
start_gpu_monitor() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_log="${LOG_DIR}/gpu_util.csv"
        echo "timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total" > "${gpu_log}"
        nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "${gpu_log}" || true
        (
            while true; do
                nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits >> "${gpu_log}" || true
                sleep "${GPU_POLL_SECONDS}"
            done
        ) &
        GPU_MONITOR_PID=$!
        log "[GPU] Logging to ${gpu_log} (PID=${GPU_MONITOR_PID})"
    else
        log "[GPU] nvidia-smi not found; skipping GPU utilization log"
    fi
}

cleanup() {
    local exit_code=$?
    if [[ -n "${GPU_MONITOR_PID}" ]] && kill -0 "${GPU_MONITOR_PID}" >/dev/null 2>&1; then
        kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    fi
    if [[ "${exit_code}" -eq 0 ]]; then
        log "[RUNBOOK] Completed successfully"
    else
        log "[RUNBOOK] Failed with exit code ${exit_code}"
    fi
}
trap cleanup EXIT INT TERM

run_and_log() {
    local label="$1"
    shift
    local target_log="${LOG_DIR}/${label}.log"
    log ""
    log "[STEP] ${label}"
    if ! "$@" 2>&1 | tee -a "${target_log}" | tee -a "${LOG_FILE}"; then
        log "[ERROR] ${label} failed; see ${target_log}"
        exit 1
    fi
}

start_gpu_monitor

PROOF_ARGS=()
if [[ "${PROOF_MODE}" == "1" ]]; then
    PROOF_ARGS+=(--proof-mode)
fi

for ((run_idx=1; run_idx<=AUDIT_RUNS; run_idx++)); do
    log ""
    log "============================================================"
    log "[AUDIT] Run ${run_idx}/${AUDIT_RUNS} - $(date -Iseconds)"
    log "============================================================"

    run_and_log "audit_${run_idx}_daily" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_auto_trader.py" \
        --tickers "${TICKERS}" \
        --lookback-days "${LOOKBACK_DAYS}" \
        --initial-capital "${INITIAL_CAPITAL}" \
        --cycles "${CYCLES}" \
        --sleep-seconds 10 \
        --resume \
        --bar-aware \
        --persist-bar-state \
        "${PROOF_ARGS[@]}"

    run_and_log "audit_${run_idx}_intraday" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_auto_trader.py" \
        --tickers "${TICKERS}" \
        --yfinance-interval "${INTRADAY_INTERVAL}" \
        --lookback-days "${INTRADAY_LOOKBACK}" \
        --forecast-horizon "${INTRADAY_HORIZON}" \
        --initial-capital "${INITIAL_CAPITAL}" \
        --cycles "${INTRADAY_CYCLES}" \
        --sleep-seconds 10 \
        --resume \
        --bar-aware \
        --persist-bar-state \
        "${PROOF_ARGS[@]}"

    run_and_log "gate_${run_idx}_forecast_audits" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_forecast_audits.py" \
        --config-path "${TS_FORECAST_MONITOR_CONFIG}" \
        --max-files 500

    run_and_log "gate_${run_idx}_quant_health" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_quant_validation_health.py"

    run_and_log "gate_${run_idx}_dashboard_audit" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/audit_dashboard_payload_sources.py"

    if [[ "${WAIT_BETWEEN_RUNS_SECONDS}" -gt 0 ]] && [[ "${run_idx}" -lt "${AUDIT_RUNS}" ]]; then
        log "[WAIT] Sleeping ${WAIT_BETWEEN_RUNS_SECONDS}s before next run"
        sleep "${WAIT_BETWEEN_RUNS_SECONDS}"
    fi
done
