#!/usr/bin/env bash
# Launch the automated trader with quality gating and dashboard emission.
# Defaults can be overridden via environment variables or CLI flags.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/bash/lib/common.sh"
pmx_require_wsl_simpletrader_runtime "${ROOT_DIR}"
PYTHON_BIN="$(pmx_require_venv_python "${ROOT_DIR}")"
TRADER_SCRIPT="$ROOT_DIR/scripts/run_auto_trader.py"
LOG_DIR="$ROOT_DIR/logs/auto_runs"
RUN_LABEL="${RUN_LABEL:-}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
DASHBOARD_AUTO_SERVE="${DASHBOARD_AUTO_SERVE:-1}"
# Default ON: persist dashboard snapshots for auditability.
DASHBOARD_PERSIST="${DASHBOARD_PERSIST:-1}"
# Default ON: keep dashboard server running after script exits.
DASHBOARD_KEEP_ALIVE="${DASHBOARD_KEEP_ALIVE:-1}"

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
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"
FORECAST_HORIZON="${FORECAST_HORIZON:-362}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-25000}"
CYCLES="${CYCLES:-10}"
SLEEP_SECONDS="${SLEEP_SECONDS:-20}"
ENABLE_LLM="${ENABLE_LLM:-0}"
LLM_MODEL="${LLM_MODEL:-deepseek-coder:6.7b-instruct-q4_K_M}"
INCLUDE_FRONTIER_TICKERS="${INCLUDE_FRONTIER_TICKERS:-1}"
VERBOSE="${VERBOSE:-0}"
HYPEROPT_ROUNDS="${HYPEROPT_ROUNDS:-1}"
BACKTEST_LOOKBACK_DAYS="${BACKTEST_LOOKBACK_DAYS:-365}"
BACKTEST_HORIZON="${BACKTEST_HORIZON:-10}"
# Optional fast intraday smoke (set INTRADAY_SMOKE=1 to enable)
INTRADAY_SMOKE="${INTRADAY_SMOKE:-0}"
INTRADAY_INTERVAL="${INTRADAY_INTERVAL:-1h}"
INTRADAY_FORECAST_HORIZON="${INTRADAY_FORECAST_HORIZON:-6}"

if [[ "$HYPEROPT_ROUNDS" != "0" ]]; then
  echo "Hyperopt mode enabled via HYPEROPT_ROUNDS=$HYPEROPT_ROUNDS; delegating to bash/run_post_eval.sh"
  ROOT_DIR="$ROOT_DIR" HYPEROPT_ROUNDS="$HYPEROPT_ROUNDS" TICKERS="$TICKERS" \
    START="${START_DATE:-2018-01-01}" END="${END_DATE:-$(date +%Y-%m-%d)}" \
  "$ROOT_DIR/bash/run_post_eval.sh"
  exit 0
fi

# Live dashboard wiring (DB -> JSON -> static HTML).
if [[ "${DASHBOARD_AUTO_SERVE}" == "1" ]]; then
  pmx_ensure_dashboard "${ROOT_DIR}" "${PYTHON_BIN}" "${DASHBOARD_PORT}" "${DASHBOARD_PERSIST}" "${DASHBOARD_KEEP_ALIVE}" "${ROOT_DIR}/data/portfolio_maximizer.db"
  if [[ "${DASHBOARD_KEEP_ALIVE}" != "1" ]]; then
    trap pmx_dashboard_cleanup EXIT
  fi
fi

# Best-effort liquidation of any open paper trades before a new run.
set +e
"$PYTHON_BIN" "$ROOT_DIR/scripts/liquidate_open_trades.py" --db-path "$ROOT_DIR/data/portfolio_maximizer.db" --pricing-policy neutral >/dev/null 2>&1 || true
set -e

# Pre-flight horizon-consistent backtest (full 365d) to log expectations.
BACKTEST_REPORT="$ROOT_DIR/reports/horizon_backtest_${RUN_STAMP}.json"
set +e
"$PYTHON_BIN" "$ROOT_DIR/scripts/run_horizon_consistent_backtest.py" \
  --tickers "$TICKERS" \
  --lookback-days "$BACKTEST_LOOKBACK_DAYS" \
  --forecast-horizon "$BACKTEST_HORIZON" \
  --report-path "$BACKTEST_REPORT" >/dev/null 2>&1 || true
set -e

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

if [[ "$INTRADAY_SMOKE" == "1" ]]; then
  CMD+=(--yfinance-interval "$INTRADAY_INTERVAL" --forecast-horizon "$INTRADAY_FORECAST_HORIZON" --cycles 3 --sleep-seconds 0)
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

# Validate dashboard assets/wiring (non-blocking).
set +e
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev_dashboard_smoke.py" >/dev/null 2>&1 || true
set -e

set +e
"$PYTHON_BIN" "$ROOT_DIR/scripts/dashboard_db_bridge.py" --once >/dev/null 2>&1 || true
set -e

DASH_PATH="$ROOT_DIR/visualizations/dashboard_data.json"
if [[ -f "$DASH_PATH" ]]; then
  echo "Dashboard JSON updated: $DASH_PATH"
  echo "Dashboard URL: http://127.0.0.1:${DASHBOARD_PORT}/visualizations/live_dashboard.html"
  echo "Dashboard HTML: $ROOT_DIR/visualizations/live_dashboard.html"
fi

echo "Auditing dashboard payload sources..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/audit_dashboard_payload_sources.py" \
  --db-path "$ROOT_DIR/data/portfolio_maximizer.db" \
  --audit-db-path "$ROOT_DIR/data/dashboard_audit.db" \
  --dashboard-json "$ROOT_DIR/visualizations/dashboard_data.json"

pmx_sanitize_logs "$ROOT_DIR"

echo "Auto-trader run complete. Logs: $LOG_FILE"
