#!/usr/bin/env bash
# Comprehensive dry-run orchestration for Portfolio Maximizer ETL pipeline.
#
# Thin wrapper around `bash/run_pipeline.sh --mode dry-run` to keep the public
# entrypoint stable while centralizing implementation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_pipeline.sh" --mode dry-run "$@"

echo "Dry-run complete."

