#!/usr/bin/env bash
# Unified LLM test runner.
#
# This consolidates legacy scripts:
# - bash/test_llm_quick.sh
# - bash/test_llm_integration.sh
# - bash/verify_fixes.sh
#
# Usage:
#   bash bash/run_llm_tests.sh quick
#   bash bash/run_llm_tests.sh full
#   bash bash/run_llm_tests.sh healthcheck

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# shellcheck source=bash/lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

PYTHON_BIN="$(pmx_resolve_python "${ROOT_DIR}")"
pmx_require_executable "${PYTHON_BIN}"

MODE="${1:-quick}"
shift || true

case "${MODE}" in
  healthcheck)
    if [[ -f "${SCRIPT_DIR}/ollama_healthcheck.sh" ]]; then
      bash "${SCRIPT_DIR}/ollama_healthcheck.sh"
      exit 0
    fi
    pmx_die "Missing bash/ollama_healthcheck.sh"
    ;;

  quick)
    "${PYTHON_BIN}" -m pytest \
      -q \
      --maxfail=1 \
      tests/ai_llm/test_ollama_client.py \
      tests/ai_llm/test_market_analyzer.py \
      tests/ai_llm/test_llm_parsing.py \
      "$@"
    ;;

  full)
    "${PYTHON_BIN}" -m pytest \
      -q \
      --maxfail=1 \
      tests/ai_llm/ \
      "$@"
    ;;

  *)
    cat <<EOF
run_llm_tests.sh - Unified LLM test runner

Usage:
  bash bash/run_llm_tests.sh quick [pytest args...]
  bash bash/run_llm_tests.sh full [pytest args...]
  bash bash/run_llm_tests.sh healthcheck
EOF
    exit 2
    ;;
esac

