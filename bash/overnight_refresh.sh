#!/usr/bin/env bash
# Overnight refresh: clear legacy quant_validation window + accumulate Platt scaling data
# Run: bash bash/overnight_refresh.sh
# Check log: logs/run_audit/overnight_refresh_YYYYMMDD_HHMMSS.log
# Purpose:
#   1. Adversarial forecaster suite  -> directional accuracy re-baseline
#   2. Pipeline per non-AAPL ticker  -> refresh quant_validation.jsonl with post-7.10b entries
#      (AMZN, GOOG, GS, JPM, META, MSFT, NVDA, TSLA, V)
#   3. Final health check            -> confirm GREEN + headroom

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/simpleTrader_env/Scripts/python.exe"
if [ ! -f "$PYTHON" ]; then
    PYTHON="$(which python)"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="${REPO_ROOT}/logs/run_audit/overnight_refresh_${TS}.log"
SUMMARY="${REPO_ROOT}/logs/run_audit/overnight_refresh_${TS}_summary.txt"
mkdir -p "${REPO_ROOT}/logs/run_audit"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log_section() { echo "" | tee -a "$LOG"; echo "========================================" | tee -a "$LOG"; log "$*"; echo "========================================" | tee -a "$LOG"; }

log_section "Overnight refresh started"
log "Python: $PYTHON"
log "Repo: $REPO_ROOT"
log "Log: $LOG"
log "Tickers: AMZN GOOG GS JPM META MSFT NVDA TSLA V"
log "Estimated runtime: 60-120 minutes"

cd "$REPO_ROOT"
ERRORS=0

# ---------------------------------------------------------------------------
# STEP 1: Adversarial forecaster suite (directional accuracy re-baseline)
# ---------------------------------------------------------------------------
log_section "STEP 1/3: Adversarial forecaster suite (DA re-baseline)"
if "$PYTHON" scripts/run_adversarial_forecaster_suite.py \
    >> "$LOG" 2>&1; then
    log "[PASS] Adversarial suite completed"
else
    log "[WARN] Adversarial suite returned non-zero (check log for details)"
    ERRORS=$((ERRORS + 1))
fi

# ---------------------------------------------------------------------------
# STEP 2: Pipeline per non-AAPL ticker -- populate quant_validation.jsonl
# Each run generates ~7-10 CV fold entries with post-7.10b weighted scoring
# ---------------------------------------------------------------------------
log_section "STEP 2/3: Pipeline refresh for non-AAPL tickers"

# Reconcile JSONL outcomes before pipeline runs so calibration uses
# any newly available trade outcomes from previous sessions.
log "--- update_platt_outcomes.py (Platt scaling outcome reconciliation)"
"$PYTHON" scripts/update_platt_outcomes.py >> "$LOG" 2>&1 || true

TICKERS="AMZN GOOG GS JPM META MSFT NVDA TSLA V"
TICKER_PASS=0
TICKER_FAIL=0

for TICKER in $TICKERS; do
    log "--- Pipeline: $TICKER (start 2024-01-01 end 2026-01-01 synthetic)"
    if "$PYTHON" scripts/run_etl_pipeline.py \
            --tickers "$TICKER" \
            --start 2024-01-01 \
            --end 2026-01-01 \
            --execution-mode synthetic \
            >> "$LOG" 2>&1; then
        log "[PASS] $TICKER pipeline completed"
        TICKER_PASS=$((TICKER_PASS + 1))
    else
        log "[WARN] $TICKER pipeline returned non-zero"
        TICKER_FAIL=$((TICKER_FAIL + 1))
        ERRORS=$((ERRORS + 1))
    fi
done

log "Ticker results: ${TICKER_PASS} passed / ${TICKER_FAIL} failed"

# ---------------------------------------------------------------------------
# STEP 2.5: Execute signals via auto_trader (synthetic cycle) -- Phase 7.13-A3
# Pipeline above only generates JSONL; no DB trade rows are created.
# This step executes the generated signals so trade_executions accumulates rows
# that update_platt_outcomes.py can reconcile against quant_validation.jsonl.
# ---------------------------------------------------------------------------
log_section "STEP 2.5/3: Synthetic auto_trader cycle (Platt scaling data accumulation)"

ALL_TICKERS="AMZN,GOOG,GS,JPM,META,MSFT,NVDA,TSLA,V"
log "--- run_auto_trader.py --tickers $ALL_TICKERS --cycles 1 --execution-mode synthetic --no-resume"
# NOTE: --proof-mode removed (Phase 7.14-A4). Proof-mode forces max_holding=5 bars -> unnatural
# tight exits -> Platt calibration data reflects test behavior, not production behavior.
if "$PYTHON" scripts/run_auto_trader.py \
        --tickers "$ALL_TICKERS" \
        --cycles 1 \
        --execution-mode synthetic \
        --no-resume \
        --sleep-seconds 0 \
        >> "$LOG" 2>&1; then
    log "[PASS] Synthetic auto_trader cycle completed"
else
    log "[WARN] Synthetic auto_trader cycle returned non-zero (Platt pairs may not accumulate)"
    ERRORS=$((ERRORS + 1))
fi

log "--- update_platt_outcomes.py (reconcile new trade outcomes)"
"$PYTHON" scripts/update_platt_outcomes.py >> "$LOG" 2>&1 || true

# ---------------------------------------------------------------------------
# STEP 2.6: Historical Platt bootstrap -- seed (confidence, outcome) pairs
# from backtested dates so calibration has material to work with immediately.
# Set PLATT_BOOTSTRAP=1 to activate (off by default to keep overnight fast).
# Uses --as-of-date to rewind market data to each historical date.
# ---------------------------------------------------------------------------
if [[ "${PLATT_BOOTSTRAP:-0}" == "1" ]]; then
    log_section "STEP 2.6/3: Platt bootstrap -- seeding pairs from 2021-2024 historical data"
    for AS_OF in 2021-01-01 2021-07-01 2022-01-01 2022-07-01 2023-01-01 2023-07-01 2024-01-01 2024-07-01; do
        log "--- bootstrap as-of $AS_OF"
        "$PYTHON" scripts/run_auto_trader.py \
            --tickers "$ALL_TICKERS" \
            --cycles 1 \
            --execution-mode synthetic \
            --as-of-date "$AS_OF" \
            --no-resume \
            --sleep-seconds 0 >> "$LOG" 2>&1 || true
        "$PYTHON" scripts/update_platt_outcomes.py >> "$LOG" 2>&1 || true
    done
    log "[DONE] Platt bootstrap complete"
fi

# ---------------------------------------------------------------------------
# STEP 3: Health check + headroom measurement
# ---------------------------------------------------------------------------
log_section "STEP 3/3: Final health check"

log "--- check_quant_validation_health.py"
"$PYTHON" scripts/check_quant_validation_health.py >> "$LOG" 2>&1 || true

log "--- quant_validation_headroom.py --json"
"$PYTHON" scripts/quant_validation_headroom.py --json >> "$LOG" 2>&1 || true

# Phase 7.13-C2: Refresh forecast audit cache so production_audit_gate has current data at 7 AM cron.
log "--- production_audit_gate.py (refresh forecast_audits_cache/latest_summary.json)"
"$PYTHON" scripts/production_audit_gate.py >> "$LOG" 2>&1 || true

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
log_section "OVERNIGHT REFRESH COMPLETE"
log "Errors: $ERRORS"
log "Log: $LOG"

{
    echo "=== Overnight Refresh Summary ==="
    echo "Completed: $(date)"
    echo "Errors: ${ERRORS}"
    echo "Tickers refreshed: ${TICKER_PASS} passed / ${TICKER_FAIL} failed"
    echo ""
    echo "Next steps:"
    echo "  1. Check log: $LOG"
    echo "  2. Re-run: python scripts/quant_validation_headroom.py --json"
    echo "  3. Re-run: python scripts/check_quant_validation_health.py"
    echo "  4. If headroom rolling window < 71%: Phase 7.10b fully validated"
    echo "  5. If 60+ (conf,win) pairs: implement Platt scaling (B5)"
    echo ""
    echo "Platt scaling data check:"
    echo "  python -c \""
    echo "    import json, pathlib"
    echo "    log = pathlib.Path('logs/signals/quant_validation.jsonl')"
    echo "    entries = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]"
    echo "    pairs = [(e.get('confidence'), e.get('outcome')) for e in entries"
    echo "             if e.get('confidence') is not None and e.get('outcome') is not None]"
    echo "    print(f'Platt scaling pairs available: {len(pairs)}')\""
} | tee "$SUMMARY" | tee -a "$LOG"

exit $ERRORS
