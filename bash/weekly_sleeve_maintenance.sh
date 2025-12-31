#!/usr/bin/env bash
set -euo pipefail

# Weekly sleeve maintenance pipeline:
# 1) Aggregate sleeve performance
# 2) Propose promotions/demotions (writes logs/automation/sleeve_promotion_plan.json)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "${ROOT_DIR}/simpleTrader_env/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/bin/python"
elif [[ -x "${ROOT_DIR}/simpleTrader_env/Scripts/python.exe" ]]; then
  PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/Scripts/python.exe"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

DB_PATH="${DB_PATH:-data/portfolio_maximizer.db}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"
MIN_TRADES="${MIN_TRADES:-5}"
SUMMARY_PATH="${SUMMARY_PATH:-logs/automation/sleeve_summary.json}"
PROMO_PLAN_PATH="${PROMO_PLAN_PATH:-logs/automation/sleeve_promotion_plan.json}"

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

echo "[weekly_sleeve_maintenance] complete. Review $PROMO_PLAN_PATH before applying config changes."
