#!/usr/bin/env bash
# Manual wrapper to run the signal validation backfill with logging.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck source=bash/lib/common.sh
source "${PROJECT_ROOT}/bash/lib/common.sh"

PYTHON_BIN="$(pmx_resolve_python "${PROJECT_ROOT}")"

pmx_require_file "scripts/backfill_signal_validation.py"

LOG_DIR="${LOG_DIR:-logs/automation}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/backfill_${STAMP}.log"

echo "[INFO] $(date -Is) starting backfill -> $LOG_FILE"
"${PYTHON_BIN}" scripts/backfill_signal_validation.py "$@" | tee "$LOG_FILE"
