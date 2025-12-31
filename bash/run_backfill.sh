#!/usr/bin/env bash
# Manual wrapper to run the signal validation backfill with logging.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY_BIN="${PYTHON_BIN:-}"
if [[ -z "$PY_BIN" && -x "simpleTrader_env/bin/python3" ]]; then
  PY_BIN="simpleTrader_env/bin/python3"
fi
PY_BIN="${PY_BIN:-python3}"

LOG_DIR="${LOG_DIR:-logs/automation}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/backfill_${STAMP}.log"

echo "[INFO] $(date -Is) starting backfill -> $LOG_FILE"
"$PY_BIN" scripts/backfill_signal_validation.py "$@" | tee "$LOG_FILE"
