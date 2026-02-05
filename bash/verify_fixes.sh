#!/usr/bin/env bash
# DEPRECATED: use `bash/run_llm_tests.sh quick`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[DEPRECATED] bash/verify_fixes.sh -> bash/run_llm_tests.sh quick"
bash "${SCRIPT_DIR}/run_llm_tests.sh" quick "$@"

