#!/usr/bin/env bash
# Power-aware helper to rebuild trade evidence, refresh slippage, and rerun sweeps.
# Skips the trading loop if the DB already has enough realised trades.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Configurable knobs
TARGET_TRADES="${TARGET_TRADES:-30}"            # Minimum realised trades to skip trading loop
TICKERS="${TICKERS:-MTN,CL=F,AAPL,MSFT}"        # Low-notional set to conserve capital
CYCLES="${CYCLES:-6}"                           # Short diagnostic cycles
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-25000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"

# Prefer project venv for consistency
if [[ -x "${ROOT_DIR}/simpleTrader_env/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/bin/python"
elif [[ -x "${ROOT_DIR}/simpleTrader_env/Scripts/python.exe" ]]; then
  PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/Scripts/python.exe"
else
  PYTHON_BIN="python"
fi

trade_count() {
  "${PYTHON_BIN}" - <<'PY'
import sqlite3, pathlib, sys
db = pathlib.Path("data/portfolio_maximizer.db")
if not db.exists():
    print(0)
    sys.exit(0)
try:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM trade_executions WHERE realized_pnl IS NOT NULL")
    print(cur.fetchone()[0] or 0)
except Exception:
    print(0)
finally:
    try:
        conn.close()
    except Exception:
        pass
PY
}

current_trades="$(trade_count)"
echo "[auto_rebuild_and_sweep] Realised trades in DB: ${current_trades}"

if [[ "${current_trades}" -lt "${TARGET_TRADES}" ]]; then
  echo "[auto_rebuild_and_sweep] Running diagnostic loop to accumulate trades..."
  ENABLE_LLM=0 \
  TICKERS="${TICKERS}" \
  CYCLES="${CYCLES}" \
  LOOKBACK_DAYS="${LOOKBACK_DAYS}" \
  INITIAL_CAPITAL="${INITIAL_CAPITAL}" \
  SLEEP_SECONDS="${SLEEP_SECONDS}" \
    bash "${ROOT_DIR}/bash/run_auto_trader_diagnostic.sh"
else
  echo "[auto_rebuild_and_sweep] Target met; skipping trading loop."
fi

echo "[auto_rebuild_and_sweep] Refreshing slippage analysis..."
"${PYTHON_BIN}" scripts/analyze_slippage_windows.py \
  --execution-log logs/automation/execution_log.jsonl \
  --output logs/automation/slippage_windows.json || true

echo "[auto_rebuild_and_sweep] Running sweep + proposals..."
bash "${ROOT_DIR}/bash/run_ts_sweep_and_proposals.sh" || true

echo "[auto_rebuild_and_sweep] Complete. Check logs/automation/ for outputs."
