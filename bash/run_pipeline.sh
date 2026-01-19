#!/usr/bin/env bash
# Unified ETL pipeline launcher (live + auto + synthetic + dry-run).
#
# This script consolidates the duplicated logic previously spread across:
# - bash/run_pipeline_live.sh
# - bash/run_pipeline_dry_run.sh
#
# It is safe to call from anywhere (it self-resolves the repo root).
#
# Usage:
#   bash bash/run_pipeline.sh --mode live [pipeline args...]
#   bash bash/run_pipeline.sh --mode dry-run [pipeline args...]
#   bash bash/run_pipeline.sh --mode auto [pipeline args...]
#   bash bash/run_pipeline.sh --mode synthetic [pipeline args...]
#   bash bash/run_pipeline.sh --tickers AAPL --start 2020-01-01 --end 2024-01-01
#
# Environment overrides (common):
#   TICKERS, START_DATE, END_DATE, DATA_SOURCE, USE_CV, ENABLE_LLM, LLM_MODEL,
#   EXECUTION_MODE, INCLUDE_FRONTIER_TICKERS, TS_FORECAST_AUDIT_DIR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=bash/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

pmx_log "INFO" "Runtime fingerprint (enforced):"
# Capture only the resolved interpreter path (pmx_resolve_python may log to stdout).
PYTHON_BIN="$(pmx_resolve_python "${ROOT_DIR}" | tail -n 1)"
pmx_require_executable "${PYTHON_BIN}"

PIPELINE_SCRIPT="${ROOT_DIR}/scripts/run_etl_pipeline.py"
pmx_require_file "${PIPELINE_SCRIPT}"

DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
DASHBOARD_AUTO_SERVE="${DASHBOARD_AUTO_SERVE:-1}"
# Default ON: persist dashboard snapshots for auditability.
DASHBOARD_PERSIST="${DASHBOARD_PERSIST:-1}"
# Default ON: keep dashboard server running after script exits.
DASHBOARD_KEEP_ALIVE="${DASHBOARD_KEEP_ALIVE:-1}"

MODE="${PIPELINE_MODE:-live}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2 || true
      ;;
    --help|-h)
      cat <<'EOF'
bash/run_pipeline.sh - Unified ETL pipeline launcher

Usage:
  bash bash/run_pipeline.sh --mode live [pipeline args...]
  bash bash/run_pipeline.sh --mode dry-run [pipeline args...]
  bash bash/run_pipeline.sh --mode auto [pipeline args...]
  bash bash/run_pipeline.sh --mode synthetic [pipeline args...]

Env:
  TICKERS, START_DATE, END_DATE, DATA_SOURCE, USE_CV, ENABLE_LLM, LLM_MODEL,
  EXECUTION_MODE, INCLUDE_FRONTIER_TICKERS, TS_FORECAST_AUDIT_DIR
EOF
      exit 0
      ;;
    --)
      shift || true
      break
      ;;
    *)
      break
      ;;
  esac
done

###############################################################################
# Live dashboard wiring (DB -> JSON -> static HTML)
###############################################################################

if [[ "${DASHBOARD_AUTO_SERVE}" == "1" ]]; then
  pmx_ensure_dashboard "${ROOT_DIR}" "${PYTHON_BIN}" "${DASHBOARD_PORT}" "${DASHBOARD_PERSIST}" "${DASHBOARD_KEEP_ALIVE}" "${ROOT_DIR}/data/portfolio_maximizer.db"
  if [[ "${DASHBOARD_KEEP_ALIVE}" != "1" ]]; then
    trap pmx_dashboard_cleanup EXIT
  fi
fi

case "${MODE}" in
  dry-run|dry_run|dry)
    MODE="dry-run"
    LOG_DIR="${ROOT_DIR}/logs/dry_runs"
    LOG_PREFIX="pipeline_dry_run"
    DEFAULT_START="2024-01-02"
    DEFAULT_END="2024-01-19"
    ;;
  synthetic)
    MODE="synthetic"
    LOG_DIR="${ROOT_DIR}/logs/synthetic_runs"
    LOG_PREFIX="pipeline_synthetic"
    DEFAULT_START="2020-01-01"
    DEFAULT_END="$(date +%Y-%m-%d)"
    ;;
  auto)
    MODE="auto"
    LOG_DIR="${ROOT_DIR}/logs/live_runs"
    LOG_PREFIX="pipeline_auto"
    DEFAULT_START="2020-01-01"
    DEFAULT_END="$(date +%Y-%m-%d)"
    ;;
  *)
    MODE="live"
    LOG_DIR="${ROOT_DIR}/logs/live_runs"
    LOG_PREFIX="pipeline_live"
    DEFAULT_START="2020-01-01"
    DEFAULT_END="$(date +%Y-%m-%d)"
    ;;
esac

mkdir -p "${LOG_DIR}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${LOG_PREFIX}_${RUN_STAMP}.log"

TICKERS="${TICKERS:-AAPL,MSFT}"
START_DATE="${START_DATE:-${DEFAULT_START}}"
END_DATE="${END_DATE:-${DEFAULT_END}}"
DATA_SOURCE="${DATA_SOURCE:-}"
USE_CV="${USE_CV:-0}"
ENABLE_LLM="${ENABLE_LLM:-1}"
LLM_MODEL="${LLM_MODEL:-}"
INCLUDE_FRONTIER_TICKERS="${INCLUDE_FRONTIER_TICKERS:-1}"

EXECUTION_MODE="${EXECUTION_MODE:-}"
if [[ -z "${EXECUTION_MODE}" ]]; then
  case "${MODE}" in
    auto) EXECUTION_MODE="auto" ;;
    synthetic) EXECUTION_MODE="synthetic" ;;
    live) EXECUTION_MODE="live" ;;
    dry-run) EXECUTION_MODE="" ;;
    *) EXECUTION_MODE="live" ;;
  esac
fi

# Enable TS forecast audit logs (forecester_ts/instrumentation.py).
export TS_FORECAST_AUDIT_DIR="${TS_FORECAST_AUDIT_DIR:-${ROOT_DIR}/logs/forecast_audits}"

CMD=(
  "${PYTHON_BIN}" "${PIPELINE_SCRIPT}"
  --tickers "${TICKERS}"
  --start "${START_DATE}"
  --end "${END_DATE}"
)

if [[ "${MODE}" == "dry-run" ]]; then
  CMD+=(--dry-run)
else
  CMD+=(--execution-mode "${EXECUTION_MODE}")
fi

if [[ -n "${DATA_SOURCE}" ]]; then
  CMD+=(--data-source "${DATA_SOURCE}")
fi
if [[ "${USE_CV}" == "1" ]]; then
  CMD+=(--use-cv)
fi
if [[ "${ENABLE_LLM}" == "1" ]]; then
  CMD+=(--enable-llm)
fi
if [[ -n "${LLM_MODEL}" ]]; then
  CMD+=(--llm-model "${LLM_MODEL}")
fi
if [[ "${INCLUDE_FRONTIER_TICKERS}" == "1" ]]; then
  CMD+=(--include-frontier-tickers)
fi

# Remaining args are passed straight through to scripts/run_etl_pipeline.py.
CMD+=("$@")

echo "Running pipeline (${MODE}) @ ${RUN_STAMP}"
set +e
pmx_run_tee "${LOG_FILE}" "${CMD[@]}"
EXIT_CODE=$?
set -e

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  echo "Pipeline run failed (exit code ${EXIT_CODE}). See ${LOG_FILE} for details." >&2
  exit "${EXIT_CODE}"
fi

PIPELINE_ID="$(grep -oE 'pipeline_[0-9]{8}_[0-9]{6}' "${LOG_FILE}" | tail -n 1 || true)"
if [[ -z "${PIPELINE_ID}" ]]; then
  echo "Unable to determine pipeline ID from ${LOG_FILE}" >&2
  exit 2
fi

DB_PATH_OVERRIDE=""
for ((i=0; i<${#CMD[@]}; i++)); do
  arg="${CMD[i]}"
  if [[ "${arg}" == "--db-path" && $((i+1)) -lt ${#CMD[@]} ]]; then
    DB_PATH_OVERRIDE="${CMD[i+1]}"
  elif [[ "${arg}" == --db-path=* ]]; then
    DB_PATH_OVERRIDE="${arg#--db-path=}"
  fi
done
DB_PATH_FOR_SUMMARY="${DB_PATH_OVERRIDE:-${PORTFOLIO_DB_PATH:-data/portfolio_maximizer.db}}"

echo ""
echo "Pipeline ID: ${PIPELINE_ID}"
echo "Stage timing summary:"

ROOT_DIR="${ROOT_DIR}" PIPELINE_ID="${PIPELINE_ID}" DB_PATH="${DB_PATH_FOR_SUMMARY}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from etl.database_manager import DatabaseManager

root = Path(os.environ["ROOT_DIR"])
pipeline_id = os.environ["PIPELINE_ID"]
db_path_raw = os.environ.get("DB_PATH") or "data/portfolio_maximizer.db"
db_path = Path(db_path_raw)
if not db_path.is_absolute():
    db_path = root / db_path

events_path = root / "logs" / "events" / "events.log"
quant_log_path = root / "logs" / "signals" / "quant_validation.jsonl"
config_path = root / "config" / "quant_success_config.yml"

tail_entries = 10
if config_path.exists():
    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
        tail_entries = int(
            cfg.get("quant_validation", {})
            .get("logging", {})
            .get("tail_entries", tail_entries)
        )
    except Exception:
        tail_entries = 10

stage_rows = []
if events_path.exists():
    with events_path.open() as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("pipeline_id") != pipeline_id:
                continue
            if record.get("event_type") == "stage_complete":
                stage = record.get("stage")
                duration = record.get("metadata", {}).get("duration_seconds")
                if stage and duration is not None:
                    stage_rows.append((stage, duration))

if not stage_rows:
    print("  (no stage metrics found)")
else:
    for stage, duration in stage_rows:
        print(f"  - {stage}: {duration:0.2f}s")

print("\nLatest dataset artifacts:")
data_root = root / "data"
for split in ("training", "validation", "testing"):
    split_dir = data_root / split
    if not split_dir.exists():
        continue
    candidates = sorted(
        split_dir.glob("*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    latest = candidates[0] if candidates else None
    if latest:
        size_kb = latest.stat().st_size / 1024
        print(f"  - {split}: {latest.name} ({size_kb:0.1f} KiB)")

print("\nQuant validation summary:")
if not quant_log_path.exists():
    print("  (quant validation log not found)")
else:
    entries = []
    with quant_log_path.open() as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries.append(record)

    matching = [r for r in entries if r.get("pipeline_id") == pipeline_id and r.get("pipeline_id")]
    target = matching if matching else entries
    target = target[-tail_entries:]

    if not target:
        print("  (no quant validation entries yet)")
    else:
        for record in target:
            ticker = record.get("ticker", "UNKNOWN")
            status = record.get("status") or record.get("quant_validation", {}).get("status", "UNKNOWN")
            action = (record.get("action") or record.get("signal", {}).get("action") or "UNKNOWN").upper()
            confidence = record.get("confidence")
            expected_return = record.get("expected_return")
            directional_edge = None
            if isinstance(expected_return, (int, float)):
                directional_edge = expected_return
                if action in {"SELL", "SHORT"}:
                    directional_edge = -directional_edge
            conf_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
            edge_str = f"{directional_edge:.2%}" if isinstance(directional_edge, (int, float)) else "n/a"
            viz = record.get("visualization_path") or "n/a"
            print(f"  - {ticker}: {status} action={action} (conf={conf_str}, edge={edge_str}) viz={viz}")

print("\nPortfolio performance summary:")
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

if not db_path.exists():
    print(f"  (db not found at {db_path})")
else:
    db = DatabaseManager(db_path=str(db_path))
    perf = db.get_performance_summary(start_date=start_date, end_date=end_date, run_id=pipeline_id)
    lifetime = db.get_performance_summary(start_date=start_date, end_date=end_date)
    db.close()

    total_trades = perf.get("total_trades", 0)
    total_profit = perf.get("total_profit", 0.0) or 0.0
    win_rate = perf.get("win_rate", 0.0) or 0.0
    profit_factor = perf.get("profit_factor", 0.0) or 0.0

    window_label = "full history"
    if start_date or end_date:
        window_label = f"{start_date or '...'} -> {end_date or '...'}"

    if total_trades <= 0:
        print(f"  run_id={pipeline_id}: no realized trades for this run; skipping PnL/MVS.")
    else:
        print(f"  Window         : {window_label} (run-scoped)")
        print(f"  Total trades   : {total_trades}")
        print(f"  Total profit   : {total_profit:.2f} USD")
        print(f"  Win rate       : {win_rate:.1%}")
        print(f"  Profit factor  : {profit_factor:.2f}")

        mvs_passed = (
            total_profit > 0.0
            and win_rate > 0.45
            and profit_factor > 1.0
            and total_trades >= 30
        )
        print(f"  MVS Status     : {'PASS' if mvs_passed else 'FAIL'}")

    # Still surface lifetime context to aid operators.
    lifetime_trades = lifetime.get("total_trades", 0)
    if lifetime_trades:
        print(f"  Lifetime       : trades={lifetime_trades}, profit={float(lifetime.get('total_profit', 0.0) or 0.0):.2f} USD, win_rate={(lifetime.get('win_rate') or 0.0):.1%}, profit_factor={(lifetime.get('profit_factor') or 0.0):.2f}")
print("")
PY

echo ""
echo "Data source snapshot (from ${LOG_FILE}):"
grep -E "Primary:" "${LOG_FILE}" | tail -n 1 || echo "  (no primary source line found)"
grep -E "OK Successfully extracted" "${LOG_FILE}" | tail -n 1 || echo "  (no extraction success line found)"

# Validate dashboard assets/wiring (non-blocking).
set +e
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/dev_dashboard_smoke.py" >/dev/null 2>&1 || true
set -e

set +e
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/dashboard_db_bridge.py" --once >/dev/null 2>&1 || true
set -e

DASH_PATH="${ROOT_DIR}/visualizations/dashboard_data.json"
echo "Log captured at: ${LOG_FILE}"
if [[ -f "${DASH_PATH}" ]]; then
  echo "Dashboard JSON available at: ${DASH_PATH}"
  echo "Dashboard URL: http://127.0.0.1:${DASHBOARD_PORT}/visualizations/live_dashboard.html"
  echo "Dashboard HTML: ${ROOT_DIR}/visualizations/live_dashboard.html"
fi

echo "Auditing dashboard payload sources..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/audit_dashboard_payload_sources.py" \
  --db-path "${ROOT_DIR}/data/portfolio_maximizer.db" \
  --audit-db-path "${ROOT_DIR}/data/dashboard_audit.db" \
  --dashboard-json "${ROOT_DIR}/visualizations/dashboard_data.json"
echo "Pipeline run complete."
