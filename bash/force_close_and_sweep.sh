#!/usr/bin/env bash
# Diagnostic chain: force-close open trades (set realized_pnl=0), then sweep + proposals.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=bash/lib/common.sh
source "${PROJECT_ROOT}/bash/lib/common.sh"

PYTHON_BIN="$(pmx_resolve_python "${PROJECT_ROOT}")"

echo "[FORCE] Liquidating open trades with mark-to-market pricing (yfinance fallback)."
"${PYTHON_BIN}" scripts/liquidate_open_trades.py

echo "[FORCE] Running sweep + proposals"
bash ./bash/run_ts_sweep_and_proposals.sh
