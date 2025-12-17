#!/usr/bin/env bash
# Synthetic pipeline smoke test: generates a synthetic dataset and runs the ETL pipeline using it.
# This is a local-only helper to confirm persistence paths and DataSourceManager wiring.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/bin/python"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found at ${PYTHON_BIN}" >&2
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-config/synthetic_data_config.yml}"
TICKERS="${TICKERS:-AAPL,MSFT}"
START_DATE="${START_DATE:-}"
END_DATE="${END_DATE:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

echo "Generating synthetic dataset (config=${CONFIG_PATH})..."
GEN_CMD=("${PYTHON_BIN}" "scripts/generate_synthetic_dataset.py" "--config" "${CONFIG_PATH}" "--tickers" "${TICKERS}")
if [[ -n "${START_DATE}" ]]; then GEN_CMD+=("--start" "${START_DATE}"); fi
if [[ -n "${END_DATE}" ]]; then GEN_CMD+=("--end" "${END_DATE}"); fi
if [[ -n "${OUTPUT_ROOT}" ]]; then GEN_CMD+=("--output-root" "${OUTPUT_ROOT}"); fi

DATASET_ID="$("${GEN_CMD[@]}" | tee /dev/tty | awk '/Synthetic dataset generated:/ {print $NF}' | tail -n1)"

if [[ -z "${DATASET_ID}" ]]; then
  echo "Failed to parse dataset_id from generator output; aborting." >&2
  exit 1
fi

export ENABLE_SYNTHETIC_PROVIDER=1
export SYNTHETIC_DATASET_ID="${DATASET_ID}"
# Constrain DataSourceManager to synthetic only for this smoke.
export SYNTHETIC_ONLY=1

echo "Running pipeline with synthetic provider (dataset_id=${DATASET_ID})..."
"${PYTHON_BIN}" scripts/run_etl_pipeline.py \
  --tickers "${TICKERS}" \
  --execution-mode synthetic \
  --data-source synthetic \
  --config "config/pipeline_config.yml" \
  --start "${START_DATE:-2020-01-01}" \
  --end "${END_DATE:-2024-01-01}"

echo "Synthetic pipeline smoke test completed."
