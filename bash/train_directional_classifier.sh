#!/usr/bin/env bash
# bash/train_directional_classifier.sh
# ------------------------------------
# Phase 9: Build directional training dataset and train the classifier.
#
# Usage:
#   bash bash/train_directional_classifier.sh [--fallback-to-pnl-label] [--skip-build]
#
# Steps:
#   1. Build labeled dataset from quant_validation.jsonl + checkpoint parquets
#   2. Train directional classifier (LogisticRegression, walk-forward CV)
#   3. Print summary and gate status
#
# Exit codes:
#   0  — model trained and saved
#   2  — cold start (not enough labeled examples yet, model NOT saved)
#   1  — hard error (missing files, import failure, etc.)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Resolve venv Python (Windows: Scripts/python.exe  Linux/Mac: bin/python)
# ---------------------------------------------------------------------------
if [[ -x "${ROOT_DIR}/simpleTrader_env/Scripts/python.exe" ]]; then
    PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/Scripts/python.exe"
elif [[ -x "${ROOT_DIR}/simpleTrader_env/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/simpleTrader_env/bin/python"
else
    echo "[ERROR] Virtual environment Python not found under ${ROOT_DIR}/simpleTrader_env"
    echo "        Run: simpleTrader_env\\Scripts\\activate  (Windows)"
    echo "             source simpleTrader_env/bin/activate  (Linux/Mac)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
FALLBACK_FLAG=""
SKIP_BUILD=0

for arg in "$@"; do
    case "${arg}" in
        --fallback-to-pnl-label) FALLBACK_FLAG="--fallback-to-pnl-label" ;;
        --skip-build)             SKIP_BUILD=1 ;;
        -h|--help)
            sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# *//'
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: ${arg}"
            exit 1
            ;;
    esac
done

cd "${ROOT_DIR}"

# Ensure project root is on PYTHONPATH so forcester_ts / models packages are importable
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== Phase 9: Directional Classifier Training ==="
echo "    Root:   ${ROOT_DIR}"
echo "    Python: ${PYTHON_BIN}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Build labeled dataset
# ---------------------------------------------------------------------------
if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    echo "--- Step 1: Build training dataset ---"
    BUILD_RC=0
    "${PYTHON_BIN}" scripts/build_directional_training_data.py ${FALLBACK_FLAG} || BUILD_RC=$?

    if [[ "${BUILD_RC}" -eq 1 ]]; then
        echo ""
        echo "[ERROR] Dataset build failed (exit 1)."
        echo "        Check that logs/signals/quant_validation.jsonl exists."
        exit 1
    fi

    # Print summary if written
    SUMMARY="logs/directional_training_latest.json"
    if [[ -f "${SUMMARY}" ]]; then
        echo ""
        echo "  Dataset summary:"
        "${PYTHON_BIN}" -c "
import json, sys
s = json.loads(open('${SUMMARY}', encoding='utf-8').read())
print(f\"    n_labeled    : {s.get('n_labeled', 0)}\")
print(f\"    n_positive   : {s.get('n_positive', 0)}\")
print(f\"    n_negative   : {s.get('n_negative', 0)}\")
print(f\"    win_rate     : {s.get('win_rate')}\")
print(f\"    cold_start   : {s.get('cold_start')}\")
if s.get('cold_start_reason'):
    print(f\"    reason       : {s.get('cold_start_reason')}\")
"

        # Short-circuit: if dataset is cold start, skip training
        COLD_START="$( "${PYTHON_BIN}" -c "import json; s=json.loads(open('${SUMMARY}',encoding='utf-8').read()); print('1' if s.get('cold_start') else '0')" 2>/dev/null || echo '1' )"
        if [[ "${COLD_START}" == "1" ]]; then
            echo ""
            echo "[COLD_START] Dataset cold start -- skipping training step."
            echo ""
            echo "  Need >= 60 labeled examples with class balance >= 10 per class."
            echo "  Current count is in ${SUMMARY}"
            echo ""
            echo "  Next steps:"
            echo "    - Run the pipeline to accumulate signals with classifier_features in JSONL"
            echo "    - Re-run this script once sufficient data is available"
            echo "    - Use --fallback-to-pnl-label to use PnL win/loss labels as a stopgap"
            exit 2
        fi
    fi
    echo ""
else
    echo "--- Step 1: Skipped (--skip-build) ---"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Train classifier
# ---------------------------------------------------------------------------
echo "--- Step 2: Train directional classifier ---"
TRAIN_RC=0
"${PYTHON_BIN}" scripts/train_directional_classifier.py || TRAIN_RC=$?

echo ""

case "${TRAIN_RC}" in
    0)
        echo "[OK] Classifier trained and saved."
        echo ""
        # Print meta summary
        META="data/classifiers/directional_v1.meta.json"
        if [[ -f "${META}" ]]; then
            "${PYTHON_BIN}" -c "
import json
m = json.loads(open('${META}', encoding='utf-8').read())
print(f\"  n_train          : {m.get('n_train')}\")
print(f\"  walk_forward_DA  : {m.get('walk_forward_da')}\")
print(f\"  best_C           : {m.get('best_c')}\")
print(f\"  top features:\")
for f in (m.get('top3_features') or []):
    print(f\"    {f['name']}: {f['coef']:+.4f}\")
print()
da = m.get('walk_forward_da', 0)
print('[GATE] Activation ready:', 'YES' if da > 0.50 and m.get('n_train', 0) >= 60 else 'NO (DA <= 50% or n < 60)')
print('       To activate: set directional_classifier.enabled: true in config/signal_routing_config.yml')
"
        fi
        exit 0
        ;;
    2)
        echo "[COLD_START] Not enough labeled examples to train — model NOT saved."
        echo ""
        echo "  Need >= 60 labeled examples with class balance >= 10 per class."
        echo "  Current count is in logs/directional_training_latest.json"
        echo ""
        echo "  Next steps:"
        echo "    - Run the pipeline to accumulate more signals with classifier_features in JSONL"
        echo "    - Re-run this script once sufficient data is available"
        echo "    - Use --fallback-to-pnl-label to use PnL win/loss labels as a stopgap"
        exit 2
        ;;
    *)
        echo "[ERROR] Training failed (exit ${TRAIN_RC})."
        exit 1
        ;;
esac
