#!/usr/bin/env bash
# bash/overnight_classifier_bootstrap.sh
# ----------------------------------------
# Phase 9: Overnight directional classifier bootstrap + PnL A/B comparison.
#
# What this script does:
#   Phase 1 — Bootstrap: 12 historical synthetic cycles (2020-2024) to
#              generate JSONL entries with classifier_features.
#   Phase 2 — Build dataset + train directional classifier.
#   Phase 3 — Control run: 4 holdout dates with gate DISABLED.
#   Phase 4 — Treatment run: same 4 holdout dates with gate ENABLED.
#   Phase 5 — Report: PnL / WR / trade-count delta (treatment - control).
#
# Usage:
#   bash bash/overnight_classifier_bootstrap.sh [--tickers AAPL,MSFT,NVDA,GS,AMZN]
#                                               [--skip-bootstrap]
#                                               [--skip-train]
#                                               [--dry-run]
#
# Options:
#   --tickers TEXT   Comma-separated tickers (default: AAPL,MSFT,NVDA,GS,AMZN)
#   --skip-bootstrap Skip Phase 1 (use existing JSONL entries; saves ~30-60 min)
#   --skip-train     Skip Phase 2 (use existing model in data/classifiers/)
#   --dry-run        Print all commands without running them
#
# Estimated runtime: 90-150 minutes (all phases)
#
# Log: logs/run_audit/classifier_bootstrap_YYYYMMDD_HHMMSS.log
# Summary: logs/run_audit/classifier_bootstrap_YYYYMMDD_HHMMSS_summary.txt

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Resolve venv Python
# ---------------------------------------------------------------------------
if [[ -x "${ROOT_DIR}/simpleTrader_env/Scripts/python.exe" ]]; then
    PYTHON="${ROOT_DIR}/simpleTrader_env/Scripts/python.exe"
elif [[ -x "${ROOT_DIR}/simpleTrader_env/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/simpleTrader_env/bin/python"
else
    echo "[ERROR] Virtual environment not found under ${ROOT_DIR}/simpleTrader_env"
    exit 1
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# Defaults / argument parsing
# ---------------------------------------------------------------------------
TICKERS="AAPL,MSFT,NVDA,GS,AMZN"
SKIP_BOOTSTRAP=0
SKIP_TRAIN=0
DRY_RUN=0

for arg in "$@"; do
    case "${arg}" in
        --tickers=*)      TICKERS="${arg#*=}" ;;
        --tickers)        shift; TICKERS="$1" ;;
        --skip-bootstrap) SKIP_BOOTSTRAP=1 ;;
        --skip-train)     SKIP_TRAIN=1 ;;
        --dry-run)        DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# *//'
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/run_audit"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/classifier_bootstrap_${TS}.log"
SUMMARY="${LOG_DIR}/classifier_bootstrap_${TS}_summary.txt"

log()         { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }
log_section() {
    echo "" | tee -a "${LOG}"
    echo "============================================================" | tee -a "${LOG}"
    log "$*"
    echo "============================================================" | tee -a "${LOG}"
}
log_warn()  { log "[WARN]  $*"; }
log_pass()  { log "[PASS]  $*"; }
log_error() { log "[ERROR] $*"; }

run_or_dry() {
    local label="$1"; shift
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "[DRY_RUN] Would run: $*"
        return 0
    fi
    log "--- ${label}"
    local rc=0
    "$@" >> "${LOG}" 2>&1 || rc=$?
    if [[ ${rc} -eq 0 ]]; then
        log_pass "${label}"
    else
        log_warn "${label} exited ${rc}"
    fi
    return ${rc}
}

log_section "Overnight Classifier Bootstrap — ${TS}"
log "Python   : ${PYTHON}"
log "Root     : ${ROOT_DIR}"
log "Tickers  : ${TICKERS}"
log "Log      : ${LOG}"
log "Options  : skip_bootstrap=${SKIP_BOOTSTRAP} skip_train=${SKIP_TRAIN} dry_run=${DRY_RUN}"
log "Estimated runtime: 90-150 min (all phases)"

ERRORS=0

# ---------------------------------------------------------------------------
# Helper: snapshot canonical DB metrics → JSON string
# ---------------------------------------------------------------------------
db_snapshot() {
    "${PYTHON}" -c "
import json, sys
sys.path.insert(0, '${ROOT_DIR}')
from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
with PnLIntegrityEnforcer('data/portfolio_maximizer.db') as e:
    m = e.get_canonical_metrics()
    print(json.dumps({
        'round_trips': m.total_round_trips,
        'total_pnl': round(m.total_realized_pnl, 4) if m.total_realized_pnl else 0.0,
        'win_count': m.win_count,
        'loss_count': m.loss_count,
        'win_rate': round(m.win_rate, 4) if m.win_rate else 0.0,
    }))
" 2>/dev/null || echo '{"round_trips":0,"total_pnl":0,"win_count":0,"loss_count":0,"win_rate":0}'
}

# Helper: extract field from JSON string
jq_lite() {
    local json="$1" field="$2"
    "${PYTHON}" -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get(sys.argv[2],'n/a'))" \
        "${json}" "${field}" 2>/dev/null || echo "n/a"
}

# Helper: toggle directional_classifier.enabled in signal_routing_config.yml
set_gate_enabled() {
    local enabled="$1"  # "true" or "false"
    "${PYTHON}" -c "
import pathlib, re
p = pathlib.Path('config/signal_routing_config.yml')
txt = p.read_text(encoding='utf-8')
txt = re.sub(
    r'(directional_classifier:.*?enabled:\s*)(?:true|false)',
    lambda m: m.group(1) + '${enabled}',
    txt, flags=re.DOTALL, count=1
)
p.write_text(txt, encoding='utf-8')
print('set directional_classifier.enabled=${enabled}')
" 2>&1 | tee -a "${LOG}"
}

# ---------------------------------------------------------------------------
# STEP 0: Record pre-script DB baseline
# ---------------------------------------------------------------------------
log_section "STEP 0: DB baseline snapshot"
BASELINE_SNAP="$(db_snapshot)"
log "Baseline: $(jq_lite "${BASELINE_SNAP}" round_trips) closed trades, PnL=$(jq_lite "${BASELINE_SNAP}" total_pnl), WR=$(jq_lite "${BASELINE_SNAP}" win_rate)"

# ---------------------------------------------------------------------------
# PHASE 1: Bootstrap — accumulate classifier_features in JSONL
# ---------------------------------------------------------------------------
log_section "PHASE 1/5: Bootstrap synthetic cycles (training window 2020-2024)"

# 12 dates spread across 2020-2024 → diverse feature distributions
BOOTSTRAP_DATES=(
    2020-06-01  2020-12-01
    2021-06-01  2021-12-01
    2022-03-01  2022-09-01
    2023-01-02  2023-07-03
    2024-01-02  2024-04-01
    2024-07-01  2024-10-01
)

if [[ "${SKIP_BOOTSTRAP}" -eq 1 ]]; then
    log "Skipping Phase 1 (--skip-bootstrap)"
else
    BOOT_PASS=0
    BOOT_FAIL=0
    for AS_OF in "${BOOTSTRAP_DATES[@]}"; do
        RC=0
        run_or_dry "bootstrap as-of ${AS_OF}" \
            "${PYTHON}" scripts/run_auto_trader.py \
                --tickers "${TICKERS}" \
                --cycles 1 \
                --execution-mode synthetic \
                --as-of-date "${AS_OF}" \
                --no-resume \
                --sleep-seconds 0 || RC=$?
        if [[ ${RC} -eq 0 ]]; then
            BOOT_PASS=$((BOOT_PASS + 1))
        else
            BOOT_FAIL=$((BOOT_FAIL + 1))
            ERRORS=$((ERRORS + 1))
        fi
    done
    log "Bootstrap complete: ${BOOT_PASS} passed / ${BOOT_FAIL} failed"
fi

# Count new JSONL entries with classifier_features
FEATURES_COUNT="$(${PYTHON} -c "
import json, pathlib
p = pathlib.Path('logs/signals/quant_validation.jsonl')
if not p.exists():
    print(0); exit()
lines = [l for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
entries = []
for l in lines:
    try: entries.append(json.loads(l))
    except: pass
with_feat = [e for e in entries if e.get('classifier_features')]
print(len(with_feat))
" 2>/dev/null || echo 0)"
log "JSONL entries with classifier_features: ${FEATURES_COUNT}"

# ---------------------------------------------------------------------------
# PHASE 2: Build dataset + train classifier
# ---------------------------------------------------------------------------
log_section "PHASE 2/5: Build training dataset + train classifier"

if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
    log "Skipping Phase 2 (--skip-train)"
    TRAIN_RC=0
    MODEL_TRAINED=1
else
    MODEL_TRAINED=0

    # Step 2a: build dataset
    log "--- build_directional_training_data.py --fallback-to-pnl-label"
    BUILD_RC=0
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${PYTHON}" scripts/build_directional_training_data.py \
            --fallback-to-pnl-label >> "${LOG}" 2>&1 || BUILD_RC=$?
    fi

    if [[ "${BUILD_RC}" -ne 0 ]]; then
        log_error "Dataset build failed — check ${LOG}"
        ERRORS=$((ERRORS + 1))
    fi

    # Print dataset summary
    if [[ -f "logs/directional_training_latest.json" && "${DRY_RUN}" -eq 0 ]]; then
        "${PYTHON}" -c "
import json
s = json.loads(open('logs/directional_training_latest.json', encoding='utf-8').read())
print(f'  n_labeled   : {s.get(\"n_labeled\", 0)}')
print(f'  n_positive  : {s.get(\"n_positive\", 0)}')
print(f'  n_negative  : {s.get(\"n_negative\", 0)}')
print(f'  win_rate    : {s.get(\"win_rate\")}')
print(f'  cold_start  : {s.get(\"cold_start\")}')
if s.get('cold_start_reason'):
    print(f'  reason      : {s.get(\"cold_start_reason\")}')
" 2>/dev/null | tee -a "${LOG}" || true
    fi

    # Step 2b: train
    log "--- train_directional_classifier.py"
    TRAIN_RC=0
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${PYTHON}" scripts/train_directional_classifier.py >> "${LOG}" 2>&1 || TRAIN_RC=$?
    fi

    case "${TRAIN_RC}" in
        0)
            MODEL_TRAINED=1
            log_pass "Classifier trained and saved"
            if [[ -f "data/classifiers/directional_v1.meta.json" ]]; then
                "${PYTHON}" -c "
import json
m = json.loads(open('data/classifiers/directional_v1.meta.json', encoding='utf-8').read())
print(f'  n_train         : {m.get(\"n_train\")}')
print(f'  walk_forward_DA : {m.get(\"walk_forward_da\")}')
print(f'  best_C          : {m.get(\"best_c\")}')
for f in (m.get('top3_features') or []):
    print(f'  feature: {f[\"name\"]} coef={f[\"coef\"]:+.4f}')
" 2>/dev/null | tee -a "${LOG}" || true
            fi
            ;;
        2)
            log_warn "Cold start — not enough labeled data. A/B test skipped."
            log "  Run Phase 1 again across more dates, or use --fallback-to-pnl-label"
            ERRORS=$((ERRORS + 1))
            ;;
        *)
            log_error "Training failed (exit ${TRAIN_RC})"
            ERRORS=$((ERRORS + 1))
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# PHASE 3: Control run — gate DISABLED (baseline PnL measurement)
# ---------------------------------------------------------------------------
log_section "PHASE 3/5: Control evaluation — gate DISABLED"

# 4 holdout dates in 2025 (unseen relative to 2020-2024 training window)
EVAL_DATES=(
    2025-01-06
    2025-04-01
    2025-07-01
    2025-10-01
)

BEFORE_CONTROL="$(db_snapshot)"
log "DB before control: $(jq_lite "${BEFORE_CONTROL}" round_trips) trades, PnL=$(jq_lite "${BEFORE_CONTROL}" total_pnl)"

# Ensure gate is disabled
if [[ "${DRY_RUN}" -eq 0 ]]; then
    set_gate_enabled "false"
fi

CTRL_PASS=0
CTRL_FAIL=0
for AS_OF in "${EVAL_DATES[@]}"; do
    RC=0
    run_or_dry "control as-of ${AS_OF}" \
        "${PYTHON}" scripts/run_auto_trader.py \
            --tickers "${TICKERS}" \
            --cycles 1 \
            --execution-mode synthetic \
            --as-of-date "${AS_OF}" \
            --no-resume \
            --sleep-seconds 0 || RC=$?
    if [[ ${RC} -eq 0 ]]; then
        CTRL_PASS=$((CTRL_PASS + 1))
    else
        CTRL_FAIL=$((CTRL_FAIL + 1))
        ERRORS=$((ERRORS + 1))
    fi
done

AFTER_CONTROL="$(db_snapshot)"
log "DB after control : $(jq_lite "${AFTER_CONTROL}" round_trips) trades, PnL=$(jq_lite "${AFTER_CONTROL}" total_pnl)"
log "Control cycles   : ${CTRL_PASS} passed / ${CTRL_FAIL} failed"

# Compute control-phase delta
CONTROL_DELTA_TRADES="$(($(jq_lite "${AFTER_CONTROL}" round_trips) - $(jq_lite "${BEFORE_CONTROL}" round_trips)))"
CONTROL_DELTA_PNL="$( "${PYTHON}" -c "print(round($(jq_lite "${AFTER_CONTROL}" total_pnl) - $(jq_lite "${BEFORE_CONTROL}" total_pnl), 4))" 2>/dev/null || echo 'n/a' )"
CONTROL_WIN_RATE="$(jq_lite "${AFTER_CONTROL}" win_rate)"

log "Control result   : trades_added=${CONTROL_DELTA_TRADES}, pnl_delta=${CONTROL_DELTA_PNL}, wr=${CONTROL_WIN_RATE}"

# ---------------------------------------------------------------------------
# PHASE 4: Treatment run — gate ENABLED
# ---------------------------------------------------------------------------
log_section "PHASE 4/5: Treatment evaluation — gate ENABLED"

if [[ "${MODEL_TRAINED}" -eq 0 ]]; then
    log_warn "Skipping Phase 4 — classifier not trained (Phase 2 cold start or skip)"
    TREAT_DELTA_TRADES="n/a"
    TREAT_DELTA_PNL="n/a"
    TREAT_WIN_RATE="n/a"
else
    # Check if classifier file exists
    if [[ ! -f "${ROOT_DIR}/data/classifiers/directional_v1.pkl" && "${DRY_RUN}" -eq 0 ]]; then
        log_warn "data/classifiers/directional_v1.pkl not found — skipping treatment run"
        TREAT_DELTA_TRADES="n/a"
        TREAT_DELTA_PNL="n/a"
        TREAT_WIN_RATE="n/a"
    else
        BEFORE_TREAT="$(db_snapshot)"
        log "DB before treatment: $(jq_lite "${BEFORE_TREAT}" round_trips) trades, PnL=$(jq_lite "${BEFORE_TREAT}" total_pnl)"

        # Enable gate
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            set_gate_enabled "true"
        fi

        TREAT_PASS=0
        TREAT_FAIL=0
        for AS_OF in "${EVAL_DATES[@]}"; do
            RC=0
            run_or_dry "treatment as-of ${AS_OF}" \
                "${PYTHON}" scripts/run_auto_trader.py \
                    --tickers "${TICKERS}" \
                    --cycles 1 \
                    --execution-mode synthetic \
                    --as-of-date "${AS_OF}" \
                    --no-resume \
                    --sleep-seconds 0 || RC=$?
            if [[ ${RC} -eq 0 ]]; then
                TREAT_PASS=$((TREAT_PASS + 1))
            else
                TREAT_FAIL=$((TREAT_FAIL + 1))
                ERRORS=$((ERRORS + 1))
            fi
        done

        AFTER_TREAT="$(db_snapshot)"
        log "DB after treatment : $(jq_lite "${AFTER_TREAT}" round_trips) trades, PnL=$(jq_lite "${AFTER_TREAT}" total_pnl)"
        log "Treatment cycles   : ${TREAT_PASS} passed / ${TREAT_FAIL} failed"

        TREAT_DELTA_TRADES="$(($(jq_lite "${AFTER_TREAT}" round_trips) - $(jq_lite "${BEFORE_TREAT}" round_trips)))"
        TREAT_DELTA_PNL="$( "${PYTHON}" -c "print(round($(jq_lite "${AFTER_TREAT}" total_pnl) - $(jq_lite "${BEFORE_TREAT}" total_pnl), 4))" 2>/dev/null || echo 'n/a' )"
        TREAT_WIN_RATE="$(jq_lite "${AFTER_TREAT}" win_rate)"
        log "Treatment result   : trades_added=${TREAT_DELTA_TRADES}, pnl_delta=${TREAT_DELTA_PNL}, wr=${TREAT_WIN_RATE}"

        # Restore gate to disabled
        if [[ "${DRY_RUN}" -eq 0 ]]; then
            set_gate_enabled "false"
            log "Gate restored to disabled"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# PHASE 5: Final report
# ---------------------------------------------------------------------------
log_section "PHASE 5/5: Results"

GATE_VERDICT="n/a"
if [[ "${TREAT_DELTA_PNL}" != "n/a" && "${CONTROL_DELTA_PNL}" != "n/a" ]]; then
    PNL_IMPROVEMENT="$( "${PYTHON}" -c "
ctrl = ${CONTROL_DELTA_PNL}
treat = ${TREAT_DELTA_PNL}
delta = treat - ctrl
sign = '+' if delta >= 0 else ''
print(f'{sign}{delta:.4f}')
" 2>/dev/null || echo 'n/a' )"

    GATE_VERDICT="$( "${PYTHON}" -c "
ctrl = ${CONTROL_DELTA_PNL}
treat = ${TREAT_DELTA_PNL}
if treat > ctrl:
    print('IMPROVEMENT')
elif treat < ctrl:
    print('REGRESSION')
else:
    print('NO_CHANGE')
" 2>/dev/null || echo 'n/a' )"
else
    PNL_IMPROVEMENT="n/a"
fi

{
    echo ""
    echo "=============================="
    echo " DIRECTIONAL CLASSIFIER A/B "
    echo "=============================="
    echo ""
    echo "  Bootstrap cycles  : ${#BOOTSTRAP_DATES[@]} historical dates x tickers"
    echo "  JSONL w/ features : ${FEATURES_COUNT}"
    echo "  Eval dates        : ${#EVAL_DATES[@]} holdout dates (2025)"
    echo ""
    echo "  CONTROL  (gate off) | trades=${CONTROL_DELTA_TRADES}  pnl=${CONTROL_DELTA_PNL}  wr=${CONTROL_WIN_RATE}"
    echo "  TREATMENT (gate on) | trades=${TREAT_DELTA_TRADES}  pnl=${TREAT_DELTA_PNL}  wr=${TREAT_WIN_RATE}"
    echo ""
    echo "  PnL delta (treatment - control): ${PNL_IMPROVEMENT}"
    echo "  Verdict: ${GATE_VERDICT}"
    echo ""
    if [[ "${GATE_VERDICT}" == "IMPROVEMENT" ]]; then
        echo "  [ACTION] Gate shows positive PnL impact."
        echo "           To enable permanently:"
        echo "           Set directional_classifier.enabled: true"
        echo "           in config/signal_routing_config.yml"
    elif [[ "${GATE_VERDICT}" == "REGRESSION" ]]; then
        echo "  [INFO] Gate shows PnL regression — classifier needs more training data."
        echo "         Continue accumulating JSONL entries and re-run this script."
    elif [[ "${GATE_VERDICT}" == "n/a" ]]; then
        echo "  [INFO] Classifier was not trained (cold start)."
        echo "         Need >= 60 labeled examples. Check logs/directional_training_latest.json"
    fi
    echo ""
    echo "  Errors (non-fatal): ${ERRORS}"
    echo "  Full log  : ${LOG}"
    echo "  Summary   : ${SUMMARY}"
    echo ""
    echo "  Next steps:"
    echo "    python scripts/update_platt_outcomes.py"
    echo "    python scripts/production_audit_gate.py --allow-inconclusive-lift"
    echo "    bash bash/train_directional_classifier.sh"
} | tee -a "${LOG}" | tee "${SUMMARY}"

exit 0
