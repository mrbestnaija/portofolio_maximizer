#!/usr/bin/env bash
# Production cron task multiplexer for Portfolio Maximizer v45.
#
# This script is designed to be called from system cron and routes
# to the appropriate Python entrypoints inside the authorised
# virtual environment (simpleTrader_env).
#
# Usage (from project root):
#   bash/production_cron.sh <task> [extra args...]
# Typical cron examples are documented in Documentation/CRON_AUTOMATION.md.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck source=bash/lib/common.sh
source "${PROJECT_ROOT}/bash/lib/common.sh"

PYTHON_BIN="$(pmx_resolve_python "${PROJECT_ROOT}")"

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

  auto_trader_core)
    # Core tickers only, with a simple trade-count gate to stop once
    # enough evidence has accumulated (>= total + per-ticker closed trades).
    CORE_TICKERS="${CRON_CORE_TICKERS:-AAPL,MSFT,GC=F,COOP}"
    CORE_DB_PATH="${CRON_CORE_DB_PATH:-data/portfolio_maximizer.db}"
    CORE_TOTAL_TARGET="${CRON_CORE_TOTAL_TARGET:-30}"
    CORE_PER_TARGET="${CRON_CORE_PER_TICKER_TARGET:-10}"

    "${PYTHON_BIN}" - <<PY
import sqlite3, sys, pathlib
db_path = pathlib.Path(r"${CORE_DB_PATH}")
tickers = [t.strip().upper() for t in "${CORE_TICKERS}".split(",") if t.strip()]
total_target = int("${CORE_TOTAL_TARGET}")
per_target = int("${CORE_PER_TARGET}")
if not db_path.exists():
    sys.exit(0)
try:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT ticker, COUNT(*) FROM trade_executions "
        "WHERE realized_pnl IS NOT NULL GROUP BY ticker"
    )
    rows = cur.fetchall()
    counts = {str(t or "").upper(): int(n or 0) for t, n in rows}
    total = sum(counts.values())
    if total >= total_target and all(counts.get(t, 0) >= per_target for t in tickers):
        print(f"[CRON] Core targets met (total={total} >= {total_target}; "
              f"per_ticker min={per_target}); skipping auto_trader_core.")
        sys.exit(1)
except Exception:
    # On errors, allow the run to proceed to avoid blocking automation.
    sys.exit(0)
finally:
    try:
        conn.close()
    except Exception:
        pass
PY
    gate_status=$?
    if [[ ${gate_status} -eq 1 ]]; then
      # Targets met; exit gracefully without running auto_trader.
      exit 0
    fi
    run_with_logging "auto_trader_core: run_auto_trader.py (tickers=${CORE_TICKERS})" \
      "${PYTHON_BIN}" scripts/run_auto_trader.py --tickers "${CORE_TICKERS}" "$@"
    ;;

  synthetic_refresh)
    # Generate a synthetic dataset (config-driven) for offline regression/smoke tests.
    SYN_CONFIG="${CRON_SYNTHETIC_CONFIG:-config/synthetic_data_config.yml}"
    SYN_TICKERS="${CRON_SYNTHETIC_TICKERS:-AAPL,MSFT}"
    SYN_OUTPUT_ROOT="${CRON_SYNTHETIC_OUTPUT_ROOT:-}"
    export ENABLE_SYNTHETIC_PROVIDER=1
    run_with_logging "synthetic_refresh: generate_synthetic_dataset.py" \
      "${PYTHON_BIN}" scripts/generate_synthetic_dataset.py \
        --config "${SYN_CONFIG}" \
        --tickers "${SYN_TICKERS}" \
        ${SYN_OUTPUT_ROOT:+--output-root "${SYN_OUTPUT_ROOT}"}
    # Validate latest synthetic dataset if present
    LATEST_DATASET_DIR=$(ls -dt ${SYN_OUTPUT_ROOT:-data/synthetic}/* 2>/dev/null | head -n 1 || true)
    if [[ -n "${LATEST_DATASET_DIR}" ]]; then
      PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" run_with_logging "synthetic_refresh: validate_synthetic_dataset.py" \
        "${PYTHON_BIN}" scripts/validate_synthetic_dataset.py \
          --dataset-path "${LATEST_DATASET_DIR}"
    fi
    ;;

  sanitize_caches)
    RETENTION="${CRON_SANITIZE_RETENTION:-14}"
    PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" run_with_logging "sanitize_cache_and_logs" \
      "${PYTHON_BIN}" scripts/sanitize_cache_and_logs.py \
        --retention-days "${RETENTION}" \
        ${CRON_SANITIZE_DATA_DIRS:+--data-dirs "${CRON_SANITIZE_DATA_DIRS}"} \
        ${CRON_SANITIZE_LOG_DIRS:+--log-dirs "${CRON_SANITIZE_LOG_DIRS}"} \
        ${CRON_SANITIZE_PATTERNS:+--patterns "${CRON_SANITIZE_PATTERNS}"}
    ;;

  ts_threshold_sweep)
    # Weekly/monthly TS threshold sweep to grow evidence for per-ticker settings.
    sweep_args=()
    if [[ -n "${CRON_TS_SWEEP_TICKERS:-}" ]]; then
      sweep_args+=(--tickers "${CRON_TS_SWEEP_TICKERS}")
    fi
    run_with_logging "ts_threshold_sweep: sweep_ts_thresholds.py" \
      "${PYTHON_BIN}" scripts/sweep_ts_thresholds.py \
        --lookback-days "${CRON_TS_SWEEP_LOOKBACK:-365}" \
        --grid-confidence "${CRON_TS_SWEEP_CONFIDENCE:-0.50,0.55,0.60}" \
        --grid-min-return "${CRON_TS_SWEEP_MIN_RETURN:-0.001,0.002,0.003}" \
        --min-trades "${CRON_TS_SWEEP_MIN_TRADES:-10}" \
        --output "${CRON_TS_SWEEP_OUTPUT:-logs/automation/ts_threshold_sweep.json}" \
        "${sweep_args[@]}"
    ;;

  transaction_costs)
    # Monthly/quarterly transaction cost estimation for friction-aware thresholds.
    cost_args=()
    if [[ -n "${CRON_COST_AS_OF:-}" ]]; then
      cost_args+=(--as-of "${CRON_COST_AS_OF}")
    fi
    if [[ -n "${CRON_COST_DB_PATH:-}" ]]; then
      cost_args+=(--db-path "${CRON_COST_DB_PATH}")
    fi
    run_with_logging "transaction_costs: estimate_transaction_costs.py" \
      "${PYTHON_BIN}" scripts/estimate_transaction_costs.py \
        --lookback-days "${CRON_COST_LOOKBACK:-365}" \
        --grouping "${CRON_COST_GROUPING:-asset_class}" \
        --min-trades "${CRON_COST_MIN_TRADES:-5}" \
        --output "${CRON_COST_OUTPUT:-logs/automation/transaction_costs.json}" \
        "${cost_args[@]}"
    ;;

  weekly_sleeve_maintenance)
    # Weekly sleeve summary + promotion/demotion recommendations.
    run_with_logging "weekly_sleeve_maintenance: summarize + promotion plan" \
      bash/weekly_sleeve_maintenance.sh
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
  bash/production_cron.sh <task> [extra args...]

Tasks:
  daily_etl            Run full ETL pipeline once (default live mode).
  auto_trader          Invoke scripts/run_auto_trader.py (paper trading loop).
  nightly_backfill     Run signal validation backfill (stub until modernised).
  monitoring           Run LLM/pipeline monitoring (if available).
  env_sanity           Run environment validation checks (if available).
  ticker_discovery_stub  Placeholder for future ticker discovery cron.
  optimizer_stub       Placeholder for future optimizer pipeline cron.
  ts_threshold_sweep   Sweep TS thresholds over realised trades (logs/automation JSON).
  transaction_costs    Estimate transaction costs grouped by ticker/asset class.
  auto_trader_core     Core tickers with trade-count gate (defaults: AAPL,MSFT,GC=F,COOP).
  synthetic_refresh    Generate synthetic dataset artifacts for offline regression.

Cron examples and production guidance:
  See Documentation/CRON_AUTOMATION.md.
EOF
    ;;
esac
