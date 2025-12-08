#!/usr/bin/env bash
# Run the core auto-trader loop once via production_cron.sh (WSL-friendly).
# Defaults to AAPL,MSFT,GC=F,COOP and respects the built-in trade-count gate.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CRON_CORE_TICKERS="${CRON_CORE_TICKERS:-AAPL,MSFT,GC=F,COOP}"
export CRON_CORE_DB_PATH="${CRON_CORE_DB_PATH:-data/portfolio_maximizer.db}"
export CRON_CORE_TOTAL_TARGET="${CRON_CORE_TOTAL_TARGET:-30}"
export CRON_CORE_PER_TICKER_TARGET="${CRON_CORE_PER_TICKER_TARGET:-10}"

echo "[CORE] Starting auto_trader_core (tickers=${CRON_CORE_TICKERS})"
./bash/production_cron.sh auto_trader_core "$@"
echo "[CORE] Complete. Check logs/cron/auto_trader_core*.out for details."
