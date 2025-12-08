#!/usr/bin/env bash
# Orchestrate TS threshold sweep + transaction cost estimation + config proposals.
# Safe to run manually or from cron. Read-only with respect to configs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_DIR="${PROJECT_ROOT}/simpleTrader_env"

detect_python() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "${VENV_DIR}/bin/python"
  elif [[ -x "${VENV_DIR}/Scripts/python.exe" ]]; then
    echo "${VENV_DIR}/Scripts/python.exe"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo ""
  fi
}

PYTHON_BIN="$(detect_python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[SWEEP] ERROR: No Python interpreter found. Ensure simpleTrader_env exists or python3 is on PATH." >&2
  exit 1
fi

# Sweep parameters (env-overridable)
SWEEP_TICKERS="${SWEEP_TICKERS:-}"
SWEEP_LOOKBACK="${SWEEP_LOOKBACK:-365}"
SWEEP_GRID_CONFIDENCE="${SWEEP_GRID_CONFIDENCE:-0.50,0.55,0.60}"
SWEEP_GRID_MIN_RETURN="${SWEEP_GRID_MIN_RETURN:-0.001,0.002,0.003}"
SWEEP_MIN_TRADES="${SWEEP_MIN_TRADES:-10}"
SWEEP_SEL_MIN_PF="${SWEEP_SEL_MIN_PF:-1.1}"
SWEEP_SEL_MIN_WR="${SWEEP_SEL_MIN_WR:-0.5}"
SWEEP_OUTPUT="${SWEEP_OUTPUT:-logs/automation/ts_threshold_sweep.json}"

# Transaction cost parameters (env-overridable)
COST_DB_PATH="${COST_DB_PATH:-${CRON_COST_DB_PATH:-data/portfolio_maximizer.db}}"
COST_LOOKBACK="${COST_LOOKBACK:-365}"
COST_GROUPING="${COST_GROUPING:-asset_class}"
COST_MIN_TRADES="${COST_MIN_TRADES:-5}"
COST_OUTPUT="${COST_OUTPUT:-logs/automation/transaction_costs.json}"

# Proposals output
PROPOSALS_OUTPUT="${PROPOSALS_OUTPUT:-logs/automation/config_proposals.json}"

# Sweep DB (defaults to cost DB if not provided explicitly)
SWEEP_DB_PATH="${SWEEP_DB_PATH:-${CRON_TS_SWEEP_DB_PATH:-${COST_DB_PATH}}}"

echo "[SWEEP] Running transaction cost estimation -> ${COST_OUTPUT}"
"${PYTHON_BIN}" scripts/estimate_transaction_costs.py \
  --db-path "${COST_DB_PATH}" \
  --lookback-days "${COST_LOOKBACK}" \
  --grouping "${COST_GROUPING}" \
  --min-trades "${COST_MIN_TRADES}" \
  --output "${COST_OUTPUT}"

SWEEP_ARGS=(
  --lookback-days "${SWEEP_LOOKBACK}"
  --grid-confidence "${SWEEP_GRID_CONFIDENCE}"
  --grid-min-return "${SWEEP_GRID_MIN_RETURN}"
  --min-trades "${SWEEP_MIN_TRADES}"
  --selection-min-profit-factor "${SWEEP_SEL_MIN_PF}"
  --selection-min-win-rate "${SWEEP_SEL_MIN_WR}"
  --output "${SWEEP_OUTPUT}"
  --db-path "${SWEEP_DB_PATH}"
)
if [[ -n "${SWEEP_TICKERS}" ]]; then
  SWEEP_ARGS+=(--tickers "${SWEEP_TICKERS}")
fi

echo "[SWEEP] Running TS threshold sweep -> ${SWEEP_OUTPUT}"
"${PYTHON_BIN}" scripts/sweep_ts_thresholds.py "${SWEEP_ARGS[@]}"

echo "[SWEEP] Generating config proposals -> ${PROPOSALS_OUTPUT}"
"${PYTHON_BIN}" scripts/generate_config_proposals.py \
  --ts-sweep-path "${SWEEP_OUTPUT}" \
  --costs-path "${COST_OUTPUT}" \
  --output "${PROPOSALS_OUTPUT}"

echo "[SWEEP] Complete. Outputs:"
echo "  - Threshold sweep : ${SWEEP_OUTPUT}"
echo "  - Cost estimates  : ${COST_OUTPUT}"
echo "  - Config proposals: ${PROPOSALS_OUTPUT}"
