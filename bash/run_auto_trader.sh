#!/usr/bin/env bash
# Launch the automated trader with quality gating and dashboard emission.
# Defaults can be overridden via environment variables or CLI flags.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/simpleTrader_env/bin/python"
TRADER_SCRIPT="$ROOT_DIR/scripts/run_auto_trader.py"
LOG_DIR="$ROOT_DIR/logs/auto_runs"
RUN_LABEL="${RUN_LABEL:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$TRADER_SCRIPT" ]]; then
  echo "Auto-trader script missing at $TRADER_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/auto_trader_${RUN_STAMP}${RUN_LABEL:+_${RUN_LABEL}}.log"

TICKERS="${TICKERS:-AAPL,MSFT}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-180}"
FORECAST_HORIZON="${FORECAST_HORIZON:-362}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-25000}"
CYCLES="${CYCLES:-10}"
SLEEP_SECONDS="${SLEEP_SECONDS:-20}"
ENABLE_LLM="${ENABLE_LLM:-1}"
LLM_MODEL="${LLM_MODEL:-deepseek-coder:6.7b-instruct-q4_K_M}"
INCLUDE_FRONTIER_TICKERS="${INCLUDE_FRONTIER_TICKERS:-0}"
VERBOSE="${VERBOSE:-0}"
HYPEROPT_ROUNDS="${HYPEROPT_ROUNDS:-0}"

if [[ "$HYPEROPT_ROUNDS" != "0" ]]; then
  echo "Hyperopt mode enabled via HYPEROPT_ROUNDS=$HYPEROPT_ROUNDS; delegating to bash/run_post_eval.sh"
  ROOT_DIR="$ROOT_DIR" HYPEROPT_ROUNDS="$HYPEROPT_ROUNDS" TICKERS="$TICKERS" \
    START="${START_DATE:-2018-01-01}" END="${END_DATE:-$(date +%Y-%m-%d)}" \
    "$ROOT_DIR/bash/run_post_eval.sh"
  exit 0
fi

CMD=(
  "$PYTHON_BIN" "$TRADER_SCRIPT"
  --tickers "$TICKERS"
  --lookback-days "$LOOKBACK_DAYS"
  --forecast-horizon "$FORECAST_HORIZON"
  --initial-capital "$INITIAL_CAPITAL"
  --cycles "$CYCLES"
  --sleep-seconds "$SLEEP_SECONDS"
)

if [[ "$ENABLE_LLM" == "1" ]]; then
  CMD+=(--enable-llm --llm-model "$LLM_MODEL")
fi

if [[ "$INCLUDE_FRONTIER_TICKERS" == "1" ]]; then
  CMD+=(--include-frontier-tickers)
fi

if [[ "$VERBOSE" == "1" ]]; then
  CMD+=(--verbose)
fi

# Allow extra overrides via CLI.
CMD+=("$@")

echo "Running auto-trader @ ${RUN_STAMP}"
echo "Command: ${CMD[*]}"
echo "Streaming output to $LOG_FILE"

set +e
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=$?
set -e

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "Auto-trader failed (exit code $EXIT_CODE). See $LOG_FILE for details." >&2
  exit "$EXIT_CODE"
fi

DASH_PATH="$ROOT_DIR/visualizations/dashboard_data.json"
if [[ -f "$DASH_PATH" ]]; then
  echo "Dashboard JSON updated: $DASH_PATH"
fi

echo "Auto-trader run complete. Logs: $LOG_FILE"
