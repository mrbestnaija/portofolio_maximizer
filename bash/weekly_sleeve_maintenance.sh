#!/usr/bin/env bash
set -euo pipefail

# Weekly sleeve maintenance pipeline:
# 1) Aggregate sleeve performance
# 2) Propose promotions/demotions (writes logs/automation/sleeve_promotion_plan.json)
# 3) Build a shadow-first NAV rebalance sidecar (writes logs/automation/nav_rebalance_plan_latest.json)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=bash/lib/common.sh
source "${ROOT_DIR}/bash/lib/common.sh"

PYTHON_BIN="$(pmx_resolve_python "${ROOT_DIR}")"

DB_PATH="${DB_PATH:-data/portfolio_maximizer.db}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"
MIN_TRADES="${MIN_TRADES:-5}"
SUMMARY_PATH="${SUMMARY_PATH:-logs/automation/sleeve_summary.json}"
PROMO_PLAN_PATH="${PROMO_PLAN_PATH:-logs/automation/sleeve_promotion_plan.json}"
NAV_REBALANCE_PATH="${NAV_REBALANCE_PATH:-logs/automation/nav_rebalance_plan_latest.json}"
ELIGIBILITY_PATH="${ELIGIBILITY_PATH:-logs/ticker_eligibility.json}"
ELIGIBILITY_GATES_PATH="${ELIGIBILITY_GATES_PATH:-logs/ticker_eligibility_gates.json}"
METRICS_SUMMARY_PATH="${METRICS_SUMMARY_PATH:-visualizations/performance/metrics_summary.json}"

echo "[weekly_sleeve_maintenance] summarizing sleeves..."
"${PYTHON_BIN}" scripts/summarize_sleeves.py \
  --db-path "$DB_PATH" \
  --lookback-days "$LOOKBACK_DAYS" \
  --min-trades "$MIN_TRADES" \
  --output "$SUMMARY_PATH"

echo "[weekly_sleeve_maintenance] generating promotion/demotion plan..."
"${PYTHON_BIN}" scripts/evaluate_sleeve_promotions.py \
  --summary-path "$SUMMARY_PATH" \
  --config-path "config/barbell.yml" \
  --output "$PROMO_PLAN_PATH" \
  --min-trades "$MIN_TRADES"

echo "[weekly_sleeve_maintenance] building shadow-first NAV rebalance plan..."
"${PYTHON_BIN}" scripts/build_nav_rebalance_plan.py \
  --eligibility-path "$ELIGIBILITY_PATH" \
  --eligibility-gates-path "$ELIGIBILITY_GATES_PATH" \
  --sleeve-summary-path "$SUMMARY_PATH" \
  --sleeve-plan-path "$PROMO_PLAN_PATH" \
  --metrics-summary-path "$METRICS_SUMMARY_PATH" \
  --risk-buckets-path "config/risk_buckets.yml" \
  --output "$NAV_REBALANCE_PATH"

echo "[weekly_sleeve_maintenance] complete. Review $PROMO_PLAN_PATH and $NAV_REBALANCE_PATH before applying config changes."
