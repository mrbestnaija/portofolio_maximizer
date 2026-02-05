#!/usr/bin/env bash
# Live/auto orchestration for Portfolio Maximizer ETL pipeline.
#
# This is a thin wrapper around `bash/run_pipeline.sh --mode live` to keep
# the public entrypoint stable while centralizing implementation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=bash/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

# Production-safe defaults: clear diagnostic shortcuts so live runs keep
# quant validation and latency guards enabled.
pmx_unset_diagnostics

# Default to true live mode; callers can override to 'auto'/'synthetic' via EXECUTION_MODE.
export EXECUTION_MODE="${EXECUTION_MODE:-live}"

bash "${SCRIPT_DIR}/run_pipeline.sh" --mode live "$@"

echo "Bestman's Portfolio Maximizer v45 Live pipeline run complete."

