#!/usr/bin/env bash
# Legacy real-time pipeline smoke runner (non-destructive).
#
# This script is kept for backward compatibility with documentation references.
# Prefer the canonical entrypoints:
# - `bash/run_pipeline.sh` (live/auto/synthetic/dry-run)
# - `bash/comprehensive_brutal_test.sh` (full evidence bundle)
#
# What this does:
# - Runs a minimal live-mode pipeline on a single ticker into an isolated DB.
# - Runs a minimal synthetic-mode pipeline (optionally with frontier tickers) into an isolated DB.
# - Optionally emits an LLM report artifact for the live DB.
#
# Env knobs:
#   RT_ENABLE_LLM=0|1
#   RT_LIVE_TICKERS="AAPL"
#   RT_LIVE_START_DATE="2020-01-01"
#   RT_LIVE_END_DATE="$(date +%Y-%m-%d)"
#   RT_SYNTH_TICKERS="AAPL,MSFT,GOOGL"
#   RT_SYNTH_START_DATE="2020-01-01"
#   RT_SYNTH_END_DATE="2024-01-01"
#   RT_INCLUDE_FRONTIER=0|1
#   RT_REFRESH_SYNTHETIC=0|1
#   RT_SYNTHETIC_DATASET_ID="latest"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=bash/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

PYTHON_BIN="$(pmx_resolve_python "${ROOT_DIR}")"
pmx_require_executable "${PYTHON_BIN}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="logs/automation/real_time_pipeline_${RUN_STAMP}"
mkdir -p "${ARTIFACT_DIR}"

RT_ENABLE_LLM="${RT_ENABLE_LLM:-0}"

LIVE_DB_PATH="data/test_real_time_pipeline_live_${RUN_STAMP}.db"
SYN_DB_PATH="data/test_real_time_pipeline_synth_${RUN_STAMP}.db"

check_db_tables() {
  local db_path="$1"
  "${PYTHON_BIN}" - <<'PY' "${db_path}"
import sqlite3
import sys
from pathlib import Path

db_path = Path(sys.argv[1])
if not db_path.exists():
    raise SystemExit(f"db missing: {db_path}")

conn = sqlite3.connect(str(db_path))
try:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    table_count = int(cur.fetchone()[0] or 0)
finally:
    conn.close()

print(f"[DB] {db_path} tables={table_count}")
PY
}

echo "=========================================="
echo "REAL-TIME PIPELINE SMOKE"
echo "=========================================="
echo "Artifacts: ${ARTIFACT_DIR}"
echo "DB (live) : ${LIVE_DB_PATH}"
echo "DB (synth): ${SYN_DB_PATH}"
echo "LLM       : ${RT_ENABLE_LLM}"
echo ""

echo "== [0/3] Optional synthetic refresh =="
if [[ "${RT_REFRESH_SYNTHETIC}" == "1" ]]; then
  echo "Refreshing synthetic dataset via production_cron.sh synthetic_refresh..."
  bash bash/production_cron.sh synthetic_refresh
else
  echo "(skipped; set RT_REFRESH_SYNTHETIC=1 to regenerate)"
fi

echo ""
echo "== [1/3] Live-mode pipeline (isolated DB) =="
RT_LIVE_TICKERS="${RT_LIVE_TICKERS:-AAPL}"
RT_LIVE_START_DATE="${RT_LIVE_START_DATE:-2020-01-01}"
RT_LIVE_END_DATE="${RT_LIVE_END_DATE:-$(date +%Y-%m-%d)}"
TICKERS="${RT_LIVE_TICKERS}" START_DATE="${RT_LIVE_START_DATE}" END_DATE="${RT_LIVE_END_DATE}" \
  ENABLE_LLM="${RT_ENABLE_LLM}" INCLUDE_FRONTIER_TICKERS=0 \
  bash "${SCRIPT_DIR}/run_pipeline.sh" --mode live --db-path "${LIVE_DB_PATH}"
check_db_tables "${LIVE_DB_PATH}"

echo ""
echo "== [2/3] Synthetic-mode pipeline (isolated DB) =="
RT_SYNTH_TICKERS="${RT_SYNTH_TICKERS:-AAPL,MSFT,GOOGL}"
RT_INCLUDE_FRONTIER="${RT_INCLUDE_FRONTIER:-1}"
RT_SYNTHETIC_DATASET_ID="${RT_SYNTHETIC_DATASET_ID:-latest}"
RT_SYNTH_START_DATE="${RT_SYNTH_START_DATE:-2020-01-01}"
RT_SYNTH_END_DATE="${RT_SYNTH_END_DATE:-2024-01-01}"

export ENABLE_SYNTHETIC_PROVIDER=1
export SYNTHETIC_ONLY=1
export SYNTHETIC_DATASET_ID="${RT_SYNTHETIC_DATASET_ID}"

TICKERS="${RT_SYNTH_TICKERS}" START_DATE="${RT_SYNTH_START_DATE}" END_DATE="${RT_SYNTH_END_DATE}" \
  ENABLE_LLM="${RT_ENABLE_LLM}" INCLUDE_FRONTIER_TICKERS="${RT_INCLUDE_FRONTIER}" \
  bash "${SCRIPT_DIR}/run_pipeline.sh" --mode synthetic --data-source synthetic --db-path "${SYN_DB_PATH}"
check_db_tables "${SYN_DB_PATH}"

echo ""
echo "== [3/3] Optional LLM report artifact (live DB) =="
if [[ "${RT_ENABLE_LLM}" == "1" ]]; then
  PORTFOLIO_DB_PATH="${LIVE_DB_PATH}" "${PYTHON_BIN}" scripts/generate_llm_report.py \
    --period all \
    --format json \
    --output "${ARTIFACT_DIR}/llm_report_live.json" || true
  echo "LLM report: ${ARTIFACT_DIR}/llm_report_live.json"
else
  echo "(skipped; set RT_ENABLE_LLM=1 to generate)"
fi

echo ""
echo "Smoke complete."
