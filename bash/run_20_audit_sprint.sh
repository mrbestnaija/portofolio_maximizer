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

# === Concurrent process guard ===
# Prevents multiple sprint/auto_trader instances from clobbering portfolio state.
LOCKFILE="${ROOT_DIR}/data/.sprint.lock"

acquire_lock() {
    if [[ -f "${LOCKFILE}" ]]; then
        local stale_pid
        stale_pid="$(cat "${LOCKFILE}" 2>/dev/null || echo "")"
        if [[ -n "${stale_pid}" ]] && kill -0 "${stale_pid}" 2>/dev/null; then
            echo "[ERROR] Another sprint/auto_trader is running (PID ${stale_pid}). Aborting."
            echo "[ERROR] If this is stale, remove ${LOCKFILE} manually."
            exit 1
        fi
        echo "[WARN] Removing stale lockfile (PID ${stale_pid} no longer running)."
        rm -f "${LOCKFILE}"
    fi
    echo $$ > "${LOCKFILE}"
}

release_lock() {
    rm -f "${LOCKFILE}"
}

acquire_lock

# Kill any rogue auto_trader Python processes using production DB.
# This prevents the concurrent-write bug that wiped portfolio state in sprint 20260212.
rogue_pids="$(pgrep -f 'run_auto_trader.py' 2>/dev/null || true)"
if [[ -n "${rogue_pids}" ]]; then
    for pid in ${rogue_pids}; do
        # Don't kill ourselves
        if [[ "${pid}" != "$$" ]]; then
            echo "[WARN] Killing rogue auto_trader process PID=${pid}"
            kill "${pid}" 2>/dev/null || true
        fi
    done
    sleep 2
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
# Optional holdout-audit backfill: run the same pipeline as if "today" were a
# previous date to create unique dataset windows for forecast_audit gating.
AS_OF_START_DATE="${AS_OF_START_DATE:-}"
AS_OF_STEP_DAYS="${AS_OF_STEP_DAYS:-1}"
ALLOW_FORECAST_GATE_FAILURE="${ALLOW_FORECAST_GATE_FAILURE:-}"
FORECAST_GATE_FAILURE_COUNT=0

# Determine whether we're in a local/holdout audit (non-live or as-of backfill).
IS_HOLDOUT_AUDIT=0
if [[ -n "${AS_OF_START_DATE}" ]]; then
    IS_HOLDOUT_AUDIT=1
fi
IS_LOCAL_AUDIT=0
if [[ "${EXECUTION_MODE}" != "live" ]]; then
    IS_LOCAL_AUDIT=1
fi
ALLOW_GATE_CONTEXT=0
if [[ "${IS_HOLDOUT_AUDIT}" == "1" || "${IS_LOCAL_AUDIT}" == "1" ]]; then
    ALLOW_GATE_CONTEXT=1
fi

# Integrity gate: orphaned-position max-age sensitivity.
# - Live runs: keep strict default (3 days) so stale opens trip the HIGH gate quickly.
# - Holdout/local runs: widen the window to avoid false positives from historical "as-of" dates.
if [[ -z "${INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS:-}" ]]; then
    if [[ "${ALLOW_GATE_CONTEXT}" == "1" ]]; then
        export INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS="60"
    else
        export INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS="3"
    fi
else
    export INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS
fi

if [[ -z "${ALLOW_FORECAST_GATE_FAILURE}" ]]; then
    if [[ "${ALLOW_GATE_CONTEXT}" == "1" ]]; then
        ALLOW_FORECAST_GATE_FAILURE="1"
    else
        ALLOW_FORECAST_GATE_FAILURE="0"
    fi
fi

if [[ "${ALLOW_FORECAST_GATE_FAILURE}" == "1" && "${ALLOW_GATE_CONTEXT}" != "1" ]]; then
    log "[WARN] ALLOW_FORECAST_GATE_FAILURE=1 is only permitted for local/holdout audits; enforcing fatal gate."
    ALLOW_FORECAST_GATE_FAILURE="0"
fi

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
if [[ "${ALLOW_FORECAST_GATE_FAILURE}" == "1" ]]; then
    log "[RUNBOOK] Forecast gate: non-fatal (local/holdout audit)"
else
    log "[RUNBOOK] Forecast gate: fatal"
fi
if [[ -n "${AS_OF_START_DATE}" ]]; then
    log "[RUNBOOK] AS_OF_START_DATE=${AS_OF_START_DATE} | AS_OF_STEP_DAYS=${AS_OF_STEP_DAYS}"
fi
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
    release_lock
    if [[ -n "${GPU_MONITOR_PID}" ]] && kill -0 "${GPU_MONITOR_PID}" >/dev/null 2>&1; then
        kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    fi
    if [[ "${exit_code}" -eq 0 ]]; then
        if [[ "${FORECAST_GATE_FAILURE_COUNT}" -gt 0 ]]; then
            log "[RUNBOOK] Completed with ${FORECAST_GATE_FAILURE_COUNT} non-fatal forecast gate failure(s)."
        else
            log "[RUNBOOK] Completed successfully"
        fi
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

run_and_log_allow_fail() {
    local label="$1"
    shift
    local target_log="${LOG_DIR}/${label}.log"
    log ""
    log "[STEP] ${label}"
    set +e
    "$@" 2>&1 | tee -a "${target_log}" | tee -a "${LOG_FILE}"
    local exit_code=${PIPESTATUS[0]}
    set -e
    if [[ "${exit_code}" -ne 0 ]]; then
        log "[WARN] ${label} failed (exit=${exit_code}); see ${target_log}"
    fi
    return "${exit_code}"
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

    AS_OF_ARGS=()
    if [[ -n "${AS_OF_START_DATE}" ]]; then
        # offset can be negative or positive; GNU date accepts "YYYY-MM-DD -N days".
        offset_days=$(( (run_idx - 1) * AS_OF_STEP_DAYS ))
        AS_OF_DATE="$(date -d "${AS_OF_START_DATE} ${offset_days} days" +%Y-%m-%d)"
        AS_OF_ARGS+=(--as-of-date "${AS_OF_DATE}")
        log "[AUDIT] as_of_date=${AS_OF_DATE}"
    fi

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
        "${AS_OF_ARGS[@]}" \
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
        "${AS_OF_ARGS[@]}" \
        "${PROOF_ARGS[@]}"

    if [[ "${ALLOW_FORECAST_GATE_FAILURE}" == "1" ]]; then
        gate_exit=0
        run_and_log_allow_fail "gate_${run_idx}_forecast_audits" \
            "${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_forecast_audits.py" \
            --config-path "${TS_FORECAST_MONITOR_CONFIG}" \
            --max-files 500 || gate_exit=$?
        if [[ "${gate_exit}" -ne 0 ]]; then
            FORECAST_GATE_FAILURE_COUNT=$((FORECAST_GATE_FAILURE_COUNT + 1))
        fi
    else
        run_and_log "gate_${run_idx}_forecast_audits" \
            "${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_forecast_audits.py" \
            --config-path "${TS_FORECAST_MONITOR_CONFIG}" \
            --max-files 500
    fi

    run_and_log "gate_${run_idx}_quant_health" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_quant_validation_health.py"

    run_and_log "gate_${run_idx}_dashboard_audit" \
        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/audit_dashboard_payload_sources.py"

    if [[ "${WAIT_BETWEEN_RUNS_SECONDS}" -gt 0 ]] && [[ "${run_idx}" -lt "${AUDIT_RUNS}" ]]; then
        log "[WAIT] Sleeping ${WAIT_BETWEEN_RUNS_SECONDS}s before next run"
        sleep "${WAIT_BETWEEN_RUNS_SECONDS}"
    fi
done
