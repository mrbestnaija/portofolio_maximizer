#!/usr/bin/env bash
# Orchestrate live ETL + Auto-Trader + Dashboard update.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/simpleTrader_env/bin/python"
PIPELINE_SCRIPT="$ROOT_DIR/scripts/run_etl_pipeline.py"
TRADER_SCRIPT="$ROOT_DIR/scripts/run_auto_trader.py"
LOG_DIR="$ROOT_DIR/logs/end_to_end"

mkdir -p "$LOG_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="$LOG_DIR/pipeline_${RUN_STAMP}.log"
TRADER_LOG="$LOG_DIR/auto_trader_${RUN_STAMP}.log"

# Defaults (override via env)
TICKERS="${TICKERS:-AAPL,MSFT}"
START_DATE="${START_DATE:-2018-01-01}"
END_DATE="${END_DATE:-2024-01-01}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-180}"
FORECAST_HORIZON="${FORECAST_HORIZON:-10}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-25000}"
PIPELINE_EXECUTION_MODE="${PIPELINE_EXECUTION_MODE:-auto}"
PIPELINE_USE_CV="${PIPELINE_USE_CV:-0}"
ENABLE_LLM="${ENABLE_LLM:-0}"
LLM_MODEL="${LLM_MODEL:-deepseek-coder:6.7b-instruct-q4_K_M}"

echo "=== [1/3] Running ETL pipeline (${PIPELINE_EXECUTION_MODE}) ==="
PIPE_CMD=(
  "$PYTHON_BIN" "$PIPELINE_SCRIPT"
  --tickers "$TICKERS"
  --start "$START_DATE"
  --end "$END_DATE"
  --execution-mode "$PIPELINE_EXECUTION_MODE"
)
if [[ "$PIPELINE_USE_CV" == "1" ]]; then PIPE_CMD+=(--use-cv); fi
if [[ "$ENABLE_LLM" == "1" ]]; then PIPE_CMD+=(--enable-llm --llm-model "$LLM_MODEL"); fi

set +e
"${PIPE_CMD[@]}" 2>&1 | tee "$PIPELINE_LOG"
PIPE_EXIT=$?
set -e
if [[ "$PIPE_EXIT" -ne 0 ]]; then
  echo "Pipeline failed (exit $PIPE_EXIT). See $PIPELINE_LOG" >&2
  exit "$PIPE_EXIT"
fi
echo "Pipeline complete. Log: $PIPELINE_LOG"

echo "=== [2/3] Running Auto-Trader ==="
TRADER_CMD=(
  "$PYTHON_BIN" "$TRADER_SCRIPT"
  --tickers "$TICKERS"
  --lookback-days "$LOOKBACK_DAYS"
  --forecast-horizon "$FORECAST_HORIZON"
  --initial-capital "$INITIAL_CAPITAL"
  --cycles "${CYCLES:-1}"
  --sleep-seconds "${SLEEP_SECONDS:-0}"
)
if [[ "$ENABLE_LLM" == "1" ]]; then TRADER_CMD+=(--enable-llm --llm-model "$LLM_MODEL"); fi

set +e
"${TRADER_CMD[@]}" 2>&1 | tee "$TRADER_LOG"
TRADER_EXIT=$?
set -e
if [[ "$TRADER_EXIT" -ne 0 ]]; then
  echo "Auto-trader failed (exit $TRADER_EXIT). See $TRADER_LOG" >&2
  exit "$TRADER_EXIT"
fi
echo "Auto-trader complete. Log: $TRADER_LOG"

echo "=== [3/3] Dashboard refresh ==="
DASH_PATH="$ROOT_DIR/visualizations/dashboard_data.json"
if [[ -f "$DASH_PATH" ]]; then
  echo "Dashboard JSON updated at: $DASH_PATH"
else
  echo "Dashboard JSON not found (expected at $DASH_PATH); ensure run_auto_trader emits it." >&2
fi

echo "End-to-end run finished @ $RUN_STAMP"
