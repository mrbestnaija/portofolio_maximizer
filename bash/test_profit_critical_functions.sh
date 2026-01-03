#!/usr/bin/env bash
# Profit-critical test runner (money-impacting invariants).
#
# This script is intentionally narrow: it only executes the tests that validate
# PnL math, profit factor / win rate, and report generation correctness.
#
# Canonical full-suite gate remains: `bash/comprehensive_brutal_test.sh`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=bash/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

PYTHON_BIN="$(pmx_resolve_python "${ROOT_DIR}")"
pmx_require_executable "${PYTHON_BIN}"

echo "=========================================="
echo "PROFIT-CRITICAL FUNCTION TESTS"
echo "=========================================="
echo "Python: ${PYTHON_BIN}"
echo ""

echo "[1/4] Profit-critical database functions"
"${PYTHON_BIN}" -m pytest \
  -q \
  --maxfail=1 \
  tests/integration/test_profit_critical_functions.py::TestProfitCriticalDatabaseFunctions

echo ""
echo "[2/4] MVS criteria validation"
"${PYTHON_BIN}" -m pytest \
  -q \
  --maxfail=1 \
  tests/integration/test_profit_critical_functions.py::TestMVSCriteriaValidation

echo ""
echo "[3/4] Profit report generation accuracy"
"${PYTHON_BIN}" -m pytest \
  -q \
  --maxfail=1 \
  tests/integration/test_llm_report_generation.py::TestProfitReportAccuracy

echo ""
echo "[4/4] System performance requirements"
"${PYTHON_BIN}" -m pytest \
  -q \
  --maxfail=1 \
  tests/integration/test_profit_critical_functions.py::TestSystemPerformanceRequirements || true

echo ""
echo "✅ Profit-critical checks complete."

