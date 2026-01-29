#!/usr/bin/env bash
# Daily automated trader with position persistence and intraday passes.
#
# Runs two passes per invocation:
#   1. Daily pass (1d interval) - standard daily signals
#   2. Intraday pass (1h interval) - hourly signals for faster position lifecycle
#
# Both passes use --resume so positions carry across passes and sessions.
#
# Usage:
#   bash bash/run_daily_trader.sh
#   TICKERS=AAPL,MSFT RISK_MODE=research_production bash bash/run_daily_trader.sh
#
# Environment variables:
#   TICKERS             - Comma-separated ticker list (default: diversified set)
#   CYCLES              - Daily pass cycles (default: 1)
#   LOOKBACK_DAYS       - Daily lookback window (default: 365)
#   INITIAL_CAPITAL     - Starting capital (default: 25000)
#   RISK_MODE           - Risk mode override (default: research_production)
#   INTRADAY_INTERVAL   - Intraday bar interval (default: 1h)
#   INTRADAY_HORIZON    - Intraday forecast horizon in bars (default: 6)
#   INTRADAY_LOOKBACK   - Intraday lookback days (default: 30)
#   INTRADAY_CYCLES     - Intraday pass cycles (default: 3)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve Python binary
if [[ -f "${ROOT_DIR}/simpleTrader_env/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/bin/python"
elif [[ -f "${ROOT_DIR}/simpleTrader_env/Scripts/python.exe" ]]; then
    PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/Scripts/python.exe"
else
    echo "[ERROR] Virtual environment Python not found"
    exit 1
fi

# Defaults
TICKERS="${TICKERS:-AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,JPM,GS,V}"
CYCLES="${CYCLES:-1}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-25000}"
INTRADAY_INTERVAL="${INTRADAY_INTERVAL:-1h}"
INTRADAY_HORIZON="${INTRADAY_HORIZON:-6}"
INTRADAY_LOOKBACK="${INTRADAY_LOOKBACK:-30}"
INTRADAY_CYCLES="${INTRADAY_CYCLES:-3}"

export RISK_MODE="${RISK_MODE:-research_production}"

LOG_DIR="${ROOT_DIR}/logs/daily_runs"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/daily_trader_${STAMP}.log"

echo "[DAILY] Starting daily trader (RISK_MODE=${RISK_MODE})"
echo "[DAILY] Tickers: ${TICKERS}"
echo "[DAILY] Log: ${LOG_FILE}"

{
    echo "[START] $(date -Iseconds)"

    # === PASS 1: Daily signals ===
    echo ""
    echo "========================================"
    echo "[PASS 1] Daily signals (interval=1d, cycles=${CYCLES})"
    echo "========================================"
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_auto_trader.py" \
        --tickers "${TICKERS}" \
        --lookback-days "${LOOKBACK_DAYS}" \
        --initial-capital "${INITIAL_CAPITAL}" \
        --cycles "${CYCLES}" \
        --sleep-seconds 10 \
        --resume \
        --bar-aware \
        --persist-bar-state \
        "$@" || echo "[PASS 1] exited with code $?"

    # === PASS 2: Intraday signals ===
    echo ""
    echo "========================================"
    echo "[PASS 2] Intraday signals (interval=${INTRADAY_INTERVAL}, horizon=${INTRADAY_HORIZON}, cycles=${INTRADAY_CYCLES})"
    echo "========================================"
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
        "$@" || echo "[PASS 2] exited with code $?"

    echo ""
    echo "[END] $(date -Iseconds)"
} 2>&1 | tee "${LOG_FILE}"

echo "[DAILY] Complete. Log: ${LOG_FILE}"
