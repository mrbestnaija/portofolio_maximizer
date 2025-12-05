#!/usr/bin/env bash
# Diagnostic helper to drive a high-trade-count run (target >=30 opportunities)
# Uses relaxed DIAGNOSTIC_MODE gates and a broader ticker set with multiple cycles.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults tuned to surface at least ~8 tickers x 6 cycles ≈ 48 trade opportunities.
: "${TICKERS:=MTN,SOL,GC=F,EURUSD=X,BTC-USD,CL=F,AAPL,MSFT}"
: "${CYCLES:=8}"
: "${FORECAST_HORIZON:=10}"
: "${LOOKBACK_DAYS:=180}"
: "${INITIAL_CAPITAL:=50000}"
: "${ENABLE_LLM:=1}"
: "${VERBOSE:=0}"

# Relaxed gates
export DIAGNOSTIC_MODE=1
export TS_DIAGNOSTIC_MODE=1
export EXECUTION_DIAGNOSTIC_MODE=1
export LLM_FORCE_FALLBACK=0

# Invoke the standard launcher with the diagnostic-friendly defaults.
exec env \
  TICKERS="$TICKERS" \
  CYCLES="$CYCLES" \
  FORECAST_HORIZON="$FORECAST_HORIZON" \
  LOOKBACK_DAYS="$LOOKBACK_DAYS" \
  INITIAL_CAPITAL="$INITIAL_CAPITAL" \
  ENABLE_LLM="$ENABLE_LLM" \
  VERBOSE="$VERBOSE" \
  bash "$ROOT_DIR/bash/run_auto_trader.sh" "$@"
