#!/usr/bin/env bash
# Run the ETL pipeline against an existing synthetic dataset (default: "latest").
#
# To generate/refresh the synthetic dataset first, use:
#   bash/production_cron.sh synthetic_refresh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=bash/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

PYTHON_BIN="$(pmx_resolve_python "${ROOT_DIR}")"
pmx_require_executable "${PYTHON_BIN}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export ENABLE_SYNTHETIC_PROVIDER=1
export SYNTHETIC_ONLY=1
export SYNTHETIC_DATASET_ID="${SYNTHETIC_DATASET_ID:-latest}"

# Preserve historical behaviour: synthetic runs do NOT enable LLM by default.
ENABLE_LLM="${ENABLE_LLM:-0}"
INCLUDE_FRONTIER_TICKERS="${INCLUDE_FRONTIER_TICKERS:-0}"

ENABLE_LLM="${ENABLE_LLM}" INCLUDE_FRONTIER_TICKERS="${INCLUDE_FRONTIER_TICKERS}" \
  bash "${SCRIPT_DIR}/run_pipeline.sh" --mode synthetic --data-source synthetic "$@"
