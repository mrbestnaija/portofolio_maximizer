#!/usr/bin/env bash
# Clear persisted portfolio state for a clean start.
#
# Usage:
#   bash bash/reset_portfolio.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve Python binary
if [[ -f "${ROOT_DIR}/simpleTrader_env/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/bin/python"
elif [[ -f "${ROOT_DIR}/simpleTrader_env/Scripts/python.exe" ]]; then
    PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/Scripts/python.exe"
else
    echo "[ERROR] Virtual environment Python not found"
    exit 1
fi

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -c "
import sys
sys.path.insert(0, '.')
from etl.database_manager import DatabaseManager
dm = DatabaseManager()
dm.clear_portfolio_state()
dm.close()
print('[OK] Portfolio state cleared. Next --resume run will start fresh.')
"
