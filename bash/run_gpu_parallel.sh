#!/usr/bin/env bash
# GPU-parallel runner (synthetic-first or auto-trader).
# Modes:
#   MODE=synthetic (default here): parallel synthetic dataset generation + validation
#   MODE=auto_trader: legacy GPU-parallel auto-trader shards with trade-count gates
# Follows GPU_PARALLEL_RUNNER_CHECKLIST and synthetic isolation guardrails to
# avoid polluting production data.

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

# Mode selection
MODE="${MODE:-synthetic}"  # synthetic | auto_trader

# Shared defaults (override via env)
GPU_LIST=(${GPU_LIST:-0})  # GPUs to use (round-robin if shards > GPUs)
# Shard tickers by liquidity; defaults favour liquid names for efficiency.
TICKER_SHARDS=(
  "${SHARD1:-AAPL,MSFT}"
  "${SHARD2:-GC=F,CL=F}"
)

# Synthetic mode defaults
SYN_CONFIG="${SYN_CONFIG:-config/synthetic_data_config.yml}"
SYN_OUTPUT_ROOT="${SYN_OUTPUT_ROOT:-data/synthetic}"
SYN_START="${SYN_START:-}"
SYN_END="${SYN_END:-}"
SYN_SEED="${SYN_SEED:-}"
SYN_FREQ="${SYN_FREQ:-}"
SYN_VALIDATE="${SYN_VALIDATE:-1}"
SYN_DATASET_PREFIX="${SYN_DATASET_PREFIX:-syn}"

# Auto-trader mode defaults
TARGET_TRADES="${TARGET_TRADES:-30}"        # realised trades gate per shard
CYCLES="${CYCLES:-4}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-50000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
FORECAST_HORIZON="${FORECAST_HORIZON:-10}"
DB_PATH="${DB_PATH:-data/test_database.db}" # never point to production when synthetic/testing

mkdir -p logs/automation logs/auto_runs

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

run_shard_synthetic() {
  local gpu="$1"; shift
  local shard="$1"
  # Avoid dataset collisions by stamping dataset_id with run_label + epoch
  local run_label="syn_gpu${gpu}_$(echo "$shard" | tr ',= ' '_' | tr -cd 'A-Za-z0-9_')"
  local dataset_id="${SYN_DATASET_PREFIX}_${run_label}_$(date +%s)"
  local log_file="logs/automation/${run_label}.log"
  echo "[synthetic][${run_label}] generating (GPU $gpu) -> ${dataset_id}" | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES="$gpu" \
  ENABLE_SYNTHETIC_PROVIDER=1 \
  SYNTHETIC_ONLY=1 \
  PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" scripts/generate_synthetic_dataset.py \
    --config "${SYN_CONFIG}" \
    --tickers "${shard}" \
    --dataset-id "${dataset_id}" \
    ${SYN_START:+--start-date "${SYN_START}"} \
    ${SYN_END:+--end-date "${SYN_END}"} \
    ${SYN_SEED:+--seed "${SYN_SEED}"} \
    ${SYN_FREQ:+--frequency "${SYN_FREQ}"} \
    --output-root "${SYN_OUTPUT_ROOT}" | tee -a "${log_file}"

  local dataset_path="${SYN_OUTPUT_ROOT}/${dataset_id}"
  if [[ "${SYN_VALIDATE}" == "1" ]]; then
    echo "[synthetic][${run_label}] validating ${dataset_path}" | tee -a "${log_file}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" \
    ENABLE_SYNTHETIC_PROVIDER=1 \
    SYNTHETIC_ONLY=1 \
      "${PYTHON_BIN}" scripts/validate_synthetic_dataset.py \
        --dataset-path "${dataset_path}" \
        --config "${SYN_CONFIG}" | tee -a "${log_file}"
  fi

  echo "[synthetic][${run_label}] complete -> ${dataset_path}" | tee -a "${log_file}"
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
  if [[ "${MODE}" == "synthetic" ]]; then
    run_shard_synthetic "$gpu" "$shard" &
  else
    run_shard "$gpu" "$shard" &
  fi
  idx=$((idx+1))
done
wait

if [[ "${MODE}" == "synthetic" ]]; then
  echo "[done] Synthetic shards complete. Review logs/automation/*.log and manifests under ${SYN_OUTPUT_ROOT}/."
else
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
fi
