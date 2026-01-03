#!/usr/bin/env bash
# DEPRECATED: use `bash/run_llm_tests.sh full`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[DEPRECATED] bash/test_llm_integration.sh -> bash/run_llm_tests.sh full"

# Best-effort connectivity hint; do not fail the suite on healthcheck output.
bash "${SCRIPT_DIR}/run_llm_tests.sh" healthcheck || true

bash "${SCRIPT_DIR}/run_llm_tests.sh" full "$@"

