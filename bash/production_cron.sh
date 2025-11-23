#!/usr/bin/env bash
# Production cron task multiplexer for Portfolio Maximizer v45.
#
# This script is designed to be called from system cron and routes
# to the appropriate Python entrypoints inside the authorised
# virtual environment (simpleTrader_env).
#
# Usage (from project root):
#   bash/bash/production_cron.sh <task> [extra args...]
# Typical cron examples are documented in Documentation/CRON_AUTOMATION.md.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${PROJECT_ROOT}/simpleTrader_env"

detect_python() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "${VENV_DIR}/bin/python"
  elif [[ -x "${VENV_DIR}/Scripts/python.exe" ]]; then
    echo "${VENV_DIR}/Scripts/python.exe"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo ""
  fi
}

PYTHON_BIN="$(detect_python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[CRON] ERROR: No Python interpreter found. Ensure simpleTrader_env exists or python3 is on PATH." >&2
  exit 1
fi

LOG_DIR="${PROJECT_ROOT}/logs/cron"
mkdir -p "${LOG_DIR}"

TASK="${1:-help}"
shift || true

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${TASK}_${TIMESTAMP}.log"

run_with_logging() {
  local desc="$1"; shift
  {
    echo "[CRON] $(date -Iseconds) :: ${desc}"
    "$@"
  } >> "${LOG_FILE}" 2>&1
}

case "${TASK}" in
  daily_etl)
    # Once-per-day ETL refresh using the main pipeline.
    # Defaults can be overridden via environment:
    #   CRON_TICKERS, CRON_START_DATE, CRON_END_DATE, CRON_EXEC_MODE
    TICKERS="${CRON_TICKERS:-AAPL,MSFT,GOOGL}"
    START_DATE="${CRON_START_DATE:-2020-01-01}"
    END_DATE="${CRON_END_DATE:-$(date +%Y-%m-%d)}"
    EXEC_MODE="${CRON_EXEC_MODE:-live}"
    run_with_logging "daily_etl: tickers=${TICKERS} mode=${EXEC_MODE}" \
      "${PYTHON_BIN}" scripts/run_etl_pipeline.py \
        --tickers "${TICKERS}" \
        --start "${START_DATE}" \
        --end "${END_DATE}" \
        --execution-mode "${EXEC_MODE}"
    ;;

  auto_trader)
    # High-frequency trading loop (paper trading engine).
    # Intended to be run every N minutes; behaviour is driven by
    # scripts/run_auto_trader.py and config/pipeline_config.yml.
    run_with_logging "auto_trader: run_auto_trader.py" \
      "${PYTHON_BIN}" scripts/run_auto_trader.py "$@"
    ;;

  nightly_backfill)
    # Nightly signal validation backfill.
    # NOTE: scripts/backfill_signal_validation.py is still under
    # modernization per implementation_checkpoint.md; this wiring
    # is provided as a stub for when that work is complete.
    if [[ -f "scripts/backfill_signal_validation.py" ]]; then
      run_with_logging "nightly_backfill: backfill_signal_validation.py" \
        "${PYTHON_BIN}" scripts/backfill_signal_validation.py "$@"
    else
      run_with_logging "nightly_backfill: stub (script missing)" \
        echo "backfill_signal_validation.py not present; skipping."
    fi
    ;;

  monitoring)
    # Lightweight health/latency monitor for LLM + pipeline.
    # Safe to run hourly; see monitor_llm_system.py for details.
    if [[ -f "scripts/monitor_llm_system.py" ]]; then
      run_with_logging "monitoring: monitor_llm_system.py" \
        "${PYTHON_BIN}" scripts/monitor_llm_system.py "$@"
    else
      run_with_logging "monitoring: stub (script missing)" \
        echo "monitor_llm_system.py not present; skipping."
    fi
    ;;

  env_sanity)
    # Environment sanity check before trading hours.
    if [[ -f "scripts/validate_environment.py" ]]; then
      run_with_logging "env_sanity: validate_environment.py" \
        "${PYTHON_BIN}" scripts/validate_environment.py "$@"
    else
      run_with_logging "env_sanity: stub (script missing)" \
        echo "validate_environment.py not present; skipping."
    fi
    ;;

  ticker_discovery_stub)
    # Future Phase 5.2+ ticker discovery integration
    # (see Documentation/arch_tree.md, TASK 5.2.x).
    run_with_logging "ticker_discovery_stub" \
      echo "Ticker discovery cron stub – integrate etl/ticker_discovery.* loaders in Phase 5.x."
    ;;

  optimizer_stub)
    # Future Phase 5.3+ portfolio optimizer runs
    # (see Documentation/arch_tree.md TASK 5.3.x).
    run_with_logging "optimizer_stub" \
      echo "Optimizer cron stub – integrate run_optimizer_pipeline once portfolio selection is live."
    ;;

  help|*)
    cat <<EOF
production_cron.sh – Portfolio Maximizer v45 cron multiplexer

Usage:
  bash/bash/production_cron.sh <task> [extra args...]

Tasks:
  daily_etl            Run full ETL pipeline once (default live mode).
  auto_trader          Invoke scripts/run_auto_trader.py (paper trading loop).
  nightly_backfill     Run signal validation backfill (stub until modernised).
  monitoring           Run LLM/pipeline monitoring (if available).
  env_sanity           Run environment validation checks (if available).
  ticker_discovery_stub  Placeholder for future ticker discovery cron.
  optimizer_stub       Placeholder for future optimizer pipeline cron.

Cron examples and production guidance:
  See Documentation/CRON_AUTOMATION.md.
EOF
    ;;
esac

