#!/usr/bin/env bash
set -euo pipefail

# Post-implementation evaluation runner:
# 1) ETL with CV drift checks on mixed/uncorrelated basket.
# 2) Auto-trader backtest pass to populate dashboard/metrics.

TICKERS=${TICKERS:-"AAPL,MSFT,CL=F,GC=F,BTC-USD,EURUSD=X"}
START=${START:-"2025-11-15"}
END=${END:-"2025-11-23"}
RUN_ID="eval_$(date +%Y%m%d_%H%M%S)"

# Higher-order hyper-parameter exploration defaults (30/70 explore/exploit)
HYPEROPT_ROUNDS=${HYPEROPT_ROUNDS:-0}
HYPEROPT_EXPLORE_PCT=${HYPEROPT_EXPLORE_PCT:-30}
HYPEROPT_EXPLOIT_PCT=$((100 - HYPEROPT_EXPLORE_PCT))

# Candidate spaces derived from regime-aware historic profitability
# - Window candidates are dynamic lookbacks anchored on END (default regime ~14d),
#   plus the explicit [START, END] provided by the caller.
DEFAULT_LOOKBACK_DAYS=${DEFAULT_LOOKBACK_DAYS:-14}
WINDOW_CANDIDATES=()
for days in "${DEFAULT_LOOKBACK_DAYS}" 30 60 90; do
  # Prefer GNU date (-d), fall back to BSD date (-v); on failure, use START.
  start_candidate="$(date -d \"${END} -${days} days\" +%Y-%m-%d 2>/dev/null || date -j -v -\"${days}\"d -f \"%Y-%m-%d\" \"${END}\" +\"%Y-%m-%d\" 2>/dev/null || echo \"${START}\")"
  WINDOW_CANDIDATES+=("${start_candidate}:${END}")
done
WINDOW_CANDIDATES+=("${START}:${END}")

# Profit threshold candidates tightened using quant_validation statistics for
# historically profitable tickers (AAPL, COOP, GC=F, EURUSD=X):
# - Most positive expected_profit values cluster in the 25–200 range.
MIN_PROFIT_CANDIDATES=(25 50 100 200)

# Expected return thresholds aligned with profitable regimes where useful moves
# are typically 0.1–0.3% rather than 2%:
MIN_RETURN_CANDIDATES=(0.0005 0.001 0.002 0.003)

QUANT_CFG_BASE="config/quant_success_config.yml"
SIGNAL_CFG_BASE="config/signal_routing_config.yml"
QUANT_CFG_HYPER="config/quant_success_config.hyperopt.yml"
SIGNAL_CFG_HYPER="config/signal_routing_config.hyperopt.yml"

write_quant_override() {
  local target=$1
  local min_profit=$2
  python3 - <<'PY' "$QUANT_CFG_BASE" "$target" "$min_profit"
import sys, yaml, pathlib
base, target, min_profit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), float(sys.argv[3])
payload = yaml.safe_load(base.read_text()) or {}
try:
    payload["quant_validation"]["success_criteria"]["min_expected_profit"] = min_profit
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"quant_success_config missing expected shape: {exc}")
target.write_text(yaml.safe_dump(payload, sort_keys=False))
PY
}

write_signal_override() {
  local target=$1
  local min_return=$2
  python3 - <<'PY' "$SIGNAL_CFG_BASE" "$target" "$min_return"
import sys, yaml, pathlib
base, target, min_return = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), float(sys.argv[3])
payload = yaml.safe_load(base.read_text()) or {}
try:
    payload["signal_routing"]["time_series"]["min_expected_return"] = min_return
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"signal_routing_config missing expected shape: {exc}")
target.write_text(yaml.safe_dump(payload, sort_keys=False))
PY
}

score_db_total_profit() {
  local db_path=$1
  python3 - <<'PY' "$db_path"
import sys
from datetime import datetime, timedelta, UTC
try:
    from etl.database_manager import DatabaseManager
except Exception as exc:  # pragma: no cover
    print(-1e12)
    raise SystemExit(f"failed to import DatabaseManager: {exc}")

db_path = sys.argv[1]
end = datetime.now(UTC).date()
start = end - timedelta(days=14)
try:
    db = DatabaseManager(db_path=db_path)
    summary = db.get_performance_summary(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    ) or {}
    print(summary.get("total_profit") or 0.0)
except Exception:
    print(-1e12)
PY
}

run_stack() {
  local run_id=$1
  local start_date=$2
  local end_date=$3
  local min_profit=$4
  local min_return=$5

  local db_path="data/portfolio_maximizer_${run_id}.db"
  write_quant_override "${QUANT_CFG_HYPER}" "${min_profit}"
  write_signal_override "${SIGNAL_CFG_HYPER}" "${min_return}"

  export QUANT_SUCCESS_CONFIG_PATH="${QUANT_CFG_HYPER}"
  export SIGNAL_ROUTING_CONFIG_PATH="${SIGNAL_CFG_HYPER}"

  echo "== [${run_id}] ETL + CV Drift (start=${start_date}, end=${end_date}) ==" >&2
  python3 scripts/run_etl_pipeline.py \
    --tickers "${TICKERS}" \
    --use-cv --n-splits 5 \
    --start "${start_date}" --end "${end_date}" \
    --execution-mode live \
    --db-path "${db_path}" >&2

  echo "== [${run_id}] Backtest / Auto-trader (min_return=${min_return}) ==" >&2
  python3 scripts/run_auto_trader.py \
    --tickers "${TICKERS}" \
    --lookback-days 120 \
    --forecast-horizon 14 \
    --cycles 5 \
    --sleep-seconds 3 \
    --initial-capital 25000 \
    --enable-llm \
    --include-frontier-tickers >&2

  echo "== [${run_id}] Stochastic Strategy Optimization (default regime) ==" >&2
  python3 scripts/run_strategy_optimization.py \
    --config-path config/strategy_optimization_config.yml \
    --db-path "${db_path}" \
    --n-candidates 32 \
    --regime default >&2

  score_db_total_profit "${db_path}"
}

hyperopt_search() {
  local rounds=$1
  local best_score=-1e12
  local best_window="${START}:${END}"
  local best_profit=50
  local best_return=0.002
  local explore_pct=${HYPEROPT_EXPLORE_PCT}

  mkdir -p logs/hyperopt
  local log_file="logs/hyperopt/hyperopt_${RUN_ID}.log"

  for ((i=1; i<=rounds; i++)); do
    local mode="explore"
    local roll=$((RANDOM % 100))
    if (( roll >= explore_pct )); then
      mode="exploit"
    fi

    local window="${best_window}"
    local min_profit="${best_profit}"
    local min_return="${best_return}"

    if [[ "${mode}" == "explore" ]]; then
      # Hold-one-out: perturb exactly one axis.
      case $((RANDOM % 3)) in
        0) window=${WINDOW_CANDIDATES[$((RANDOM % ${#WINDOW_CANDIDATES[@]}))]} ;;
        1) min_profit=${MIN_PROFIT_CANDIDATES[$((RANDOM % ${#MIN_PROFIT_CANDIDATES[@]}))]} ;;
        2) min_return=${MIN_RETURN_CANDIDATES[$((RANDOM % ${#MIN_RETURN_CANDIDATES[@]}))]} ;;
      esac
    fi

    IFS=: read -r trial_start trial_end <<< "${window}"
    local trial_id="${RUN_ID}_h${i}"

    local score
    score=$(run_stack "${trial_id}" "${trial_start}" "${trial_end}" "${min_profit}" "${min_return}")
    printf "%s\tmode=%s\twindow=%s\tmin_profit=%s\tmin_return=%s\tscore=%s\n" \
      "${trial_id}" "${mode}" "${window}" "${min_profit}" "${min_return}" "${score}" | tee -a "${log_file}"

    if awk "BEGIN {exit !(${score} > ${best_score})}"; then
      best_score=${score}
      best_window=${window}
      best_profit=${min_profit}
      best_return=${min_return}
      explore_pct=$(( explore_pct > 10 ? explore_pct - 5 : explore_pct ))
    else
      explore_pct=$(( explore_pct < 90 ? explore_pct + 5 : explore_pct ))
    fi
  done

  echo "== Hyperopt best =="
  echo "  window: ${best_window}"
  echo "  min_expected_profit: ${best_profit}"
  echo "  min_expected_return: ${best_return}"
  echo "  score (total_profit): ${best_score}"

  IFS=: read -r best_start best_end <<< "${best_window}"
  run_stack "${RUN_ID}_best" "${best_start}" "${best_end}" "${best_profit}" "${best_return}"
  FINAL_RUN_ID="${RUN_ID}_best"
}

FINAL_RUN_ID="${RUN_ID}"
if (( HYPEROPT_ROUNDS > 0 )); then
  echo "== Hyper-parameter exploration enabled (rounds=${HYPEROPT_ROUNDS}, explore=${HYPEROPT_EXPLORE_PCT}%, exploit=${HYPEROPT_EXPLOIT_PCT}%) =="
  hyperopt_search "${HYPEROPT_ROUNDS}"
else
  run_stack "${RUN_ID}" "${START}" "${END}" 50 0.002
fi

echo "Artifacts:"
echo "  - DB: data/portfolio_maximizer_${FINAL_RUN_ID}.db (plus hyperopt variants if run)"
echo "  - Drift JSON: visualizations/split_drift_latest.json"
echo "  - Dashboard JSON/PNG: visualizations/dashboard_data.json, visualizations/dashboard_snapshot.png"
