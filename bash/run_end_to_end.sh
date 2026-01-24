#!/usr/bin/env bash
# Orchestrate live ETL + Auto-Trader + Dashboard update.

set -euo pipefail

# Production-safe defaults: ensure diagnostic shortcuts used by
# brutal/diagnostic helpers do not leak into live profit runs.
unset DIAGNOSTIC_MODE TS_DIAGNOSTIC_MODE EXECUTION_DIAGNOSTIC_MODE LLM_FORCE_FALLBACK || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/bash/lib/common.sh"
pmx_require_wsl_simpletrader_runtime "${ROOT_DIR}"
PYTHON_BIN="$(pmx_require_venv_python "${ROOT_DIR}")"
PIPELINE_SCRIPT="$ROOT_DIR/scripts/run_etl_pipeline.py"
TRADER_SCRIPT="$ROOT_DIR/scripts/run_auto_trader.py"
LOG_DIR="$ROOT_DIR/logs/end_to_end"
DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
DASHBOARD_AUTO_SERVE="${DASHBOARD_AUTO_SERVE:-1}"
# Default ON: persist dashboard snapshots for auditability.
DASHBOARD_PERSIST="${DASHBOARD_PERSIST:-1}"
# Default ON: keep dashboard server running after script exits.
DASHBOARD_KEEP_ALIVE="${DASHBOARD_KEEP_ALIVE:-1}"

mkdir -p "$LOG_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="$LOG_DIR/pipeline_${RUN_STAMP}.log"
TRADER_LOG="$LOG_DIR/auto_trader_${RUN_STAMP}.log"

# Live dashboard wiring (DB -> JSON -> static HTML).
if [[ "${DASHBOARD_AUTO_SERVE}" == "1" ]]; then
  pmx_ensure_dashboard "${ROOT_DIR}" "${PYTHON_BIN}" "${DASHBOARD_PORT}" "${DASHBOARD_PERSIST}" "${DASHBOARD_KEEP_ALIVE}" "${ROOT_DIR}/data/portfolio_maximizer.db"
  if [[ "${DASHBOARD_KEEP_ALIVE}" != "1" ]]; then
    trap pmx_dashboard_cleanup EXIT
  fi
fi

# Defaults (override via env)
TICKERS="${TICKERS:-AAPL,MSFT, GOOGL}"
START_DATE="${START_DATE:-2015-01-01}"
END_DATE="${END_DATE:-2024-01-01}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"
FORECAST_HORIZON="${FORECAST_HORIZON:-10}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-25000}"
PIPELINE_EXECUTION_MODE="${PIPELINE_EXECUTION_MODE:-auto}"
PIPELINE_USE_CV="${PIPELINE_USE_CV:-0}"
ENABLE_LLM="${ENABLE_LLM:-0}"
LLM_MODEL="${LLM_MODEL:-deepseek-coder:6.7b-instruct-q4_K_M}"
BACKTEST_LOOKBACK_DAYS="${BACKTEST_LOOKBACK_DAYS:-365}"
BACKTEST_HORIZON="${BACKTEST_HORIZON:-10}"

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
echo "Data source snapshot (from $PIPELINE_LOG):"
grep -E "Primary:" "$PIPELINE_LOG" | tail -n 1 || echo "  (no primary source line found)"
grep -E "OK Successfully extracted" "$PIPELINE_LOG" | tail -n 1 || echo "  (no extraction success line found)"

# Higher-order hyper-parameter exploration (project-wide default)
HYPEROPT_ROUNDS="${HYPEROPT_ROUNDS:-0}"

if [[ "$HYPEROPT_ROUNDS" != "0" ]]; then
  echo "=== Hyperopt mode enabled for end-to-end run (rounds=$HYPEROPT_ROUNDS) ==="
  ROOT_DIR="$ROOT_DIR" HYPEROPT_ROUNDS="$HYPEROPT_ROUNDS" TICKERS="$TICKERS" \
    START="$START_DATE" END="$END_DATE" \
    "$ROOT_DIR/bash/run_post_eval.sh"
  echo "End-to-end hyperopt run finished @ $RUN_STAMP"
  exit 0
fi

echo "=== [2/3] Running Auto-Trader ==="
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

echo "=== Performance summary (SQLite: data/portfolio_maximizer.db) ==="
"$PYTHON_BIN" - <<'PY'
import os
from datetime import datetime, timedelta

from etl.database_manager import DatabaseManager

start_env = os.getenv("MVS_START_DATE")
end_env = os.getenv("MVS_END_DATE")
window_days = os.getenv("MVS_WINDOW_DAYS")

start_date = start_env
end_date = end_env

if not start_date and window_days:
    try:
        days = int(window_days)
        end = datetime.utcnow().date()
        start = end - timedelta(days=days)
        start_date = start.isoformat()
        end_date = end.isoformat()
    except ValueError:
        start_date = None
        end_date = None

db = DatabaseManager()
perf = db.get_performance_summary(start_date=start_date, end_date=end_date)
db.close()

total_trades = perf.get("total_trades", 0)
total_profit = perf.get("total_profit", 0.0) or 0.0
win_rate = perf.get("win_rate", 0.0) or 0.0
profit_factor = perf.get("profit_factor", 0.0) or 0.0

window_label = "full history"
if start_date or end_date:
    window_label = f"{start_date or '...'} -> {end_date or '...'}"

print(f"Window         : {window_label}")
print(f"Total trades   : {total_trades}")
print(f"Total profit   : {total_profit:.2f} USD")
print(f"Win rate       : {win_rate:.1%}")
print(f"Profit factor  : {profit_factor:.2f}")

mvs_passed = (
    total_profit > 0.0
    and win_rate > 0.45
    and profit_factor > 1.0
    and total_trades >= 30
)
print(f"MVS Status     : {'PASS' if mvs_passed else 'FAIL'}")
PY

# Validate dashboard assets/wiring (non-blocking).
set +e
"$PYTHON_BIN" "$ROOT_DIR/scripts/dev_dashboard_smoke.py" >/dev/null 2>&1 || true
set -e

set +e
"$PYTHON_BIN" "$ROOT_DIR/scripts/dashboard_db_bridge.py" --once >/dev/null 2>&1 || true
set -e

echo "Auditing dashboard payload sources..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/audit_dashboard_payload_sources.py" \
  --db-path "$ROOT_DIR/data/portfolio_maximizer.db" \
  --audit-db-path "$ROOT_DIR/data/dashboard_audit.db" \
  --dashboard-json "$ROOT_DIR/visualizations/dashboard_data.json"

pmx_sanitize_logs "$ROOT_DIR"

echo "End-to-end run finished @ $RUN_STAMP"
echo "Dashboard URL: http://127.0.0.1:${DASHBOARD_PORT}/visualizations/live_dashboard.html"
echo "Dashboard HTML: ${ROOT_DIR}/visualizations/live_dashboard.html"
