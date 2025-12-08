#!/usr/bin/env bash
# GPU-parallel, power-aware auto-trader runner.
# Shards tickers per GPU, skips shards that already meet realised trade targets,
# then refreshes slippage and TS sweeps.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Preferred Python
if [[ -x "${ROOT_DIR}/simpleTrader_env/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/bin/python"
elif [[ -x "${ROOT_DIR}/simpleTrader_env/Scripts/python.exe" ]]; then
  PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/Scripts/python.exe"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

# Defaults (override via env)
GPU_LIST=(${GPU_LIST:-0})                   # GPUs to use
TARGET_TRADES="${TARGET_TRADES:-30}"        # realised trades gate per shard
CYCLES="${CYCLES:-4}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-50000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
FORECAST_HORIZON="${FORECAST_HORIZON:-10}"
DB_PATH="${DB_PATH:-data/portfolio_maximizer_new.db}"
# Shard tickers by asset class/liquidity (tune as needed)
TICKER_SHARDS=(
  "${SHARD1:-MTN,MSFT,AAPL}"
  "${SHARD2:-CL=F}"
)

trade_count() {
  local tickers="$1"
  local db_path="${DB_PATH:-data/portfolio_maximizer_new.db}"
  "${PYTHON_BIN}" - <<PY
import sqlite3, pathlib, sys
db = pathlib.Path("${db_path}")
if not db.exists():
    print(0); sys.exit(0)
try:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    qs = ",".join("?"*len("${tickers}".split(",")))
    cur.execute(f"SELECT COUNT(*) FROM trade_executions WHERE realized_pnl IS NOT NULL AND ticker IN ({qs})",
                [t.strip() for t in "${tickers}".split(",")])
    print(cur.fetchone()[0] or 0)
except Exception:
    print(0)
finally:
    try:
        conn.close()
    except Exception:
        pass
PY
}

run_shard() {
  local gpu="$1"; shift
  local shard="$1"
  local run_label
  run_label="gpu${gpu}_$(echo "$shard" | tr ',= ' '_' | tr -cd 'A-Za-z0-9_')"
  local count
  count=$(trade_count "$shard")
  local diag="${FORCE_DIAGNOSTIC:-0}"
  if [[ "$count" -ge "$TARGET_TRADES" ]]; then
    echo "[shard $shard] target met ($count >= $TARGET_TRADES); skipping"
    return 0
  fi
  echo "[shard $shard] launching on GPU $gpu (current trades=$count)"
  CUDA_VISIBLE_DEVICES="$gpu" \
  DIAGNOSTIC_MODE="$diag" \
  TS_DIAGNOSTIC_MODE="$diag" \
  EXECUTION_DIAGNOSTIC_MODE="$diag" \
  ENABLE_LLM=0 \
  PORTFOLIO_DB_PATH="$DB_PATH" \
  TICKERS="$shard" \
  RUN_LABEL="$run_label" \
  LOOKBACK_DAYS="$LOOKBACK_DAYS" \
  FORECAST_HORIZON="$FORECAST_HORIZON" \
  INITIAL_CAPITAL="$INITIAL_CAPITAL" \
  CYCLES="$CYCLES" \
  SLEEP_SECONDS="$SLEEP_SECONDS" \
    bash "${ROOT_DIR}/bash/run_auto_trader.sh"
}

export -f run_shard trade_count
export PYTHON_BIN TARGET_TRADES LOOKBACK_DAYS INITIAL_CAPITAL CYCLES SLEEP_SECONDS FORECAST_HORIZON ROOT_DIR

idx=0
for shard in "${TICKER_SHARDS[@]}"; do
  # skip empty shards if user leaves SHARD vars blank
  [[ -z "${shard//,/}" ]] && continue
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  run_shard "$gpu" "$shard" &
  idx=$((idx+1))
done
wait

echo "[post] refreshing slippage..."
"${PYTHON_BIN}" scripts/analyze_slippage_windows.py \
  --db-path "$DB_PATH" \
  --execution-log logs/automation/execution_log.jsonl \
  --output logs/automation/slippage_windows.json || true

echo "[post] running sweep + proposals..."
 CRON_COST_DB_PATH="$DB_PATH" \
 CRON_TS_SWEEP_DB_PATH="$DB_PATH" \
  bash "${ROOT_DIR}/bash/run_ts_sweep_and_proposals.sh" || true

echo "[done] Check logs/auto_runs/ and logs/automation/ for outputs."
