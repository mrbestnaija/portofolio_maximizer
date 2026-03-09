#!/usr/bin/env bash
# run_exp_r5_phase3_accumulate.sh
#
# EXP-R5-001 Phase 3 re-accumulation under RC1-RC4 redesigned residual model.
#
# Runs 10 ETL pipeline passes with distinct --end dates, then runs the Phase 3
# backfill (realized-price patching) and truth check to evaluate M3 promotion.
#
# USAGE
#   bash bash/run_exp_r5_phase3_accumulate.sh
#   DRY_RUN=1 bash bash/run_exp_r5_phase3_accumulate.sh   # skip pipeline, only backfill+truth
#   TICKER=MSFT bash bash/run_exp_r5_phase3_accumulate.sh  # use a different ticker
#
# KEY FACTS
#   - All end dates are within 2020-2024 so the widest checkpoint
#     (pipeline_20260308_184327_data_extraction_*.parquet) covers realized prices.
#   - Each run produces one uniquely fingerprinted audit (SHA1 of ticker+start+end+len+horizon).
#     Runs with identical --start/--end produce the same fingerprint and are deduplicated
#     by phase3_backfill.py -- so distinct end dates are required.
#   - set -euo pipefail: any pipeline pass failure exits immediately with its pass number.
#
# INTERPRETATION
#   After the backfill, the script prints the M3 promotion check:
#     mean_rmse_ratio < 1.0 AND mean_corr(e, e_hat) >= 0.30 => PROMOTE
#     Otherwise: accumulate more windows or investigate

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve Python binary (Windows venv or Linux venv)
if [[ -x "${ROOT_DIR}/simpleTrader_env/Scripts/python.exe" ]]; then
    PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/Scripts/python.exe"
elif [[ -x "${ROOT_DIR}/simpleTrader_env/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/bin/python"
else
    echo "[ERROR] Virtual environment Python not found under ${ROOT_DIR}/simpleTrader_env"
    exit 1
fi

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
TICKER="${TICKER:-AAPL}"
START_DATE="${START_DATE:-2020-01-01}"
DRY_RUN="${DRY_RUN:-0}"

# 10 end dates spanning 2021-2024 in ~3-month steps.
# Must all fall within the checkpoint coverage window so realized prices exist.
END_DATES=(
    "2021-03-01"
    "2021-06-01"
    "2021-09-01"
    "2021-12-01"
    "2022-03-01"
    "2022-06-01"
    "2022-09-01"
    "2023-01-01"
    "2023-06-01"
    "2024-01-01"
)

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/exp_r5_phase3/${RUN_TAG}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/phase3_accumulate.log"

# Tee all output to both terminal and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================================"
echo "[EXP-R5-001 Phase 3] Starting accumulation run ${RUN_TAG}"
echo "Ticker   : ${TICKER}"
echo "Start    : ${START_DATE}"
echo "End dates: ${END_DATES[*]}"
echo "Log      : ${LOG_FILE}"
echo "DRY_RUN  : ${DRY_RUN}"
echo "========================================================"

# -----------------------------------------------------------------------
# Step 1: 10 ETL pipeline passes
# -----------------------------------------------------------------------
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] DRY_RUN=1 -- skipping pipeline passes"
else
    PASS=0
    for END_DATE in "${END_DATES[@]}"; do
        PASS=$((PASS + 1))
        echo ""
        echo "------------------------------------------------------------"
        echo "[Pass ${PASS}/10] --tickers ${TICKER} --start ${START_DATE} --end ${END_DATE}"
        echo "------------------------------------------------------------"

        PASS_LOG="${LOG_DIR}/pass_${PASS}_${END_DATE}.log"

        "${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_etl_pipeline.py" \
            --tickers "${TICKER}" \
            --start "${START_DATE}" \
            --end "${END_DATE}" \
            --execution-mode synthetic \
            2>&1 | tee "${PASS_LOG}"

        EXIT_CODE=${PIPESTATUS[0]}
        if [[ "${EXIT_CODE}" -ne 0 ]]; then
            echo "[ERROR] Pass ${PASS} (--end ${END_DATE}) exited with code ${EXIT_CODE}"
            echo "[ERROR] See ${PASS_LOG} for details"
            exit "${EXIT_CODE}"
        fi

        echo "[Pass ${PASS}/10] DONE (--end ${END_DATE})"
    done

    echo ""
    echo "[INFO] All 10 pipeline passes completed."
fi

# -----------------------------------------------------------------------
# Step 2: Phase 3 backfill (realized-price patching)
# -----------------------------------------------------------------------
echo ""
echo "========================================================"
echo "[Step 2] Running Phase 3 backfill (realized-price patch)"
echo "========================================================"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/residual_experiment_phase3_backfill.py" \
    2>&1 | tee "${LOG_DIR}/phase3_backfill.log"

BACKFILL_EXIT=${PIPESTATUS[0]}
if [[ "${BACKFILL_EXIT}" -ne 0 ]]; then
    echo "[ERROR] phase3_backfill.py exited with code ${BACKFILL_EXIT}"
    exit "${BACKFILL_EXIT}"
fi

# -----------------------------------------------------------------------
# Step 3: Run quality pipeline to refresh residual summary JSON
# -----------------------------------------------------------------------
echo ""
echo "========================================================"
echo "[Step 3] Refreshing residual summary via quality pipeline"
echo "========================================================"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_quality_pipeline.py" \
    --enable-residual-experiment \
    2>&1 | tee "${LOG_DIR}/quality_pipeline.log"

# Non-fatal: quality pipeline may exit 1 if other gates fail (not residual-related)
echo "[INFO] Quality pipeline complete (residual summary refreshed)"

# -----------------------------------------------------------------------
# Step 4: Truth check and M3 evaluation
# -----------------------------------------------------------------------
echo ""
echo "========================================================"
echo "[Step 4] EXP-R5-001 truth check"
echo "========================================================"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/residual_experiment_truth.py" \
    2>&1 | tee "${LOG_DIR}/truth_check.log"

TRUTH_EXIT=${PIPESTATUS[0]}

# -----------------------------------------------------------------------
# Step 5: Print M3 promotion summary from residual_experiment_summary.json
# -----------------------------------------------------------------------
echo ""
echo "========================================================"
echo "[Step 5] M3 Promotion Summary"
echo "========================================================"

SUMMARY_JSON="${ROOT_DIR}/visualizations/performance/residual_experiment_summary.json"
if [[ -f "${SUMMARY_JSON}" ]]; then
    "${PYTHON_BIN}" - <<'PYEOF'
import json, sys
from pathlib import Path

summary_path = Path("visualizations/performance/residual_experiment_summary.json")
try:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
except Exception as e:
    print(f"[ERROR] Cannot read summary: {e}")
    sys.exit(0)

n_windows = s.get("n_windows_with_residual_metrics", s.get("n_windows_with_realized_residual_metrics", 0))
rmse_ratio = s.get("rmse_ratio_mean", s.get("mean_rmse_ratio", None))
corr       = s.get("corr_mean", s.get("mean_corr", None))
status     = s.get("status", "UNKNOWN")

print(f"  n_windows_with_realized_metrics : {n_windows}")
print(f"  mean_rmse_ratio                 : {rmse_ratio}")
print(f"  mean_corr(epsilon, epsilon_hat) : {corr}")
print(f"  summary status                  : {status}")
print()

# M3 promotion criteria
RMSE_THRESHOLD = 1.0
CORR_THRESHOLD = 0.30
MIN_WINDOWS    = 10

promote = (
    n_windows is not None and n_windows >= MIN_WINDOWS
    and rmse_ratio is not None and rmse_ratio < RMSE_THRESHOLD
    and corr is not None and corr >= CORR_THRESHOLD
)

if n_windows is not None and n_windows < MIN_WINDOWS:
    print(f"[M3] INSUFFICIENT DATA: only {n_windows} windows (need >= {MIN_WINDOWS})")
    print("[M3] Run more pipeline passes with additional end dates.")
elif promote:
    print(f"[M3] PROMOTE: mean_rmse_ratio={rmse_ratio:.4f} < 1.0  AND  "
          f"mean_corr={corr:.4f} >= 0.30 over {n_windows} windows")
    print("[M3] EXP-R5-001 redesign is an improvement. Ready for Agent C promotion decision.")
else:
    reasons = []
    if rmse_ratio is not None and rmse_ratio >= RMSE_THRESHOLD:
        reasons.append(f"rmse_ratio={rmse_ratio:.4f} >= {RMSE_THRESHOLD}")
    if corr is not None and corr < CORR_THRESHOLD:
        reasons.append(f"corr={corr:.4f} < {CORR_THRESHOLD}")
    print(f"[M3] INCONCLUSIVE / REDESIGN_REQUIRED: {'; '.join(reasons)}")
    print("[M3] Investigate root causes before re-accumulating.")
PYEOF
else
    echo "[WARN] Summary JSON not found at ${SUMMARY_JSON}"
    echo "[WARN] Run scripts/run_quality_pipeline.py --enable-residual-experiment manually."
fi

echo ""
echo "========================================================"
echo "[EXP-R5-001 Phase 3] Complete. Logs: ${LOG_DIR}"
echo "========================================================"

if [[ "${TRUTH_EXIT}" -ne 0 ]]; then
    echo "[WARN] Truth check exited ${TRUTH_EXIT} (possible contradiction in summary)"
fi

exit 0
