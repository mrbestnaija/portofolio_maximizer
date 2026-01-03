#!/usr/bin/env bash
# Synthetic pipeline smoke test: generates a synthetic dataset and runs the ETL pipeline using it.
# This is a local-only helper to confirm persistence paths and DataSourceManager wiring.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=bash/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

PYTHON_BIN="$(pmx_resolve_python "${ROOT_DIR}")"
pmx_require_executable "${PYTHON_BIN}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

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

# Generator prints the dataset_id on its last line; capture the final non-empty token.
DATASET_ID="$("${GEN_CMD[@]}" | tee /dev/tty | awk 'NF {last=$NF} END {print last}')"

if [[ -z "${DATASET_ID}" ]]; then
  echo "Failed to parse dataset_id from generator output; aborting." >&2
  exit 1
fi

export ENABLE_SYNTHETIC_PROVIDER=1
export SYNTHETIC_DATASET_ID="${DATASET_ID}"
# Constrain DataSourceManager to synthetic only for this smoke.
export SYNTHETIC_ONLY=1

PIPELINE_START="${START_DATE:-2020-01-01}"
PIPELINE_END="${END_DATE:-2024-01-01}"

# Preserve historical behaviour: synthetic smoke does NOT enable LLM by default.
ENABLE_LLM="${ENABLE_LLM:-0}"
INCLUDE_FRONTIER_TICKERS="${INCLUDE_FRONTIER_TICKERS:-0}"

echo "Running pipeline with synthetic provider (dataset_id=${DATASET_ID})..."
TICKERS="${TICKERS}" START_DATE="${PIPELINE_START}" END_DATE="${PIPELINE_END}" \
  ENABLE_LLM="${ENABLE_LLM}" INCLUDE_FRONTIER_TICKERS="${INCLUDE_FRONTIER_TICKERS}" \
  bash "${SCRIPT_DIR}/run_pipeline.sh" --mode synthetic --data-source synthetic --config "config/pipeline_config.yml" "$@"

echo "Synthetic pipeline smoke test completed."
