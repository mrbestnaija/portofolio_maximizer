#!/usr/bin/env bash
#
# Portfolio Maximizer v45 - Full Test Runner (canonical location)
# Executes the complete pytest suite using the simpleTrader_env virtual environment.
#
# NOTE: This script mirrors the legacy `.bash/full_test_run.sh` entrypoint.
# Prefer this `bash/` path going forward; the `.bash/` variant is kept only
# for backward compatibility and may be removed once all docs/scripts are updated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/bash/lib/common.sh"
pmx_require_wsl_simpletrader_runtime "${PROJECT_ROOT}"
PYTHON_BIN="$(pmx_require_venv_python "${PROJECT_ROOT}")"
PYTEST_CMD=("${PYTHON_BIN}" -m pytest)

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] simpleTrader_env Python interpreter not found at: ${PYTHON_BIN}" >&2
    echo "        Ensure the environment exists: python3 -m venv simpleTrader_env" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -m pytest --version >/dev/null 2>&1; then
    echo "[ERROR] pytest is not available in simpleTrader_env." >&2
    echo "        Activate the environment and install dependencies (pip install -r requirements.txt)." >&2
    exit 1
fi

printf -- "\n=============================================\n"
printf -- " Portfolio Maximizer v45 - Full Test Run\n"
printf -- " Interpreter : %s\n" "${PYTHON_BIN}"
printf -- " Working Dir : %s\n" "${PROJECT_ROOT}"
printf -- "=============================================\n\n"

TEST_ARGS=("$@")
if [[ ${#TEST_ARGS[@]} -eq 0 ]]; then
    TEST_ARGS=("tests")
fi

START_TIME=$(date +%s)

set +e
"${PYTEST_CMD[@]}" \
    --maxfail=1 \
    --durations=15 \
    --cache-clear \
    "${TEST_ARGS[@]}"
TEST_STATUS=$?
set -e

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

printf -- "\n---------------------------------------------\n"
if [[ ${TEST_STATUS} -eq 0 ]]; then
    printf -- "✅ Full test suite passed in %ss\n" "${ELAPSED}"
else
    printf -- "❌ Test suite failed in %ss (exit code %s)\n" "${ELAPSED}" "${TEST_STATUS}"
fi
printf -- "---------------------------------------------\n\n"

exit "${TEST_STATUS}"
