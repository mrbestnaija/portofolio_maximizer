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
TICKERS="AMZN GOOG GS JPM META MSFT NVDA TSLA V"
mkdir -p "${REPO_ROOT}/logs/run_audit"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
log_section() { echo "" | tee -a "$LOG"; echo "========================================" | tee -a "$LOG"; log "$*"; echo "========================================" | tee -a "$LOG"; }

run_best_effort() {
    local label="$1"
    shift
    log "--- ${label}"
    if "$@" >> "$LOG" 2>&1; then
        log "[PASS] ${label}"
    else
        log "[WARN] ${label} returned non-zero"
    fi
}

run_synthetic_auto_trader() {
    local as_of_date="${1:-}"
    # Production step 2.5: 1 cycle, no proof-mode -> live-comparable behavior.
    local -a cmd=(
        "$PYTHON" scripts/run_auto_trader.py
        --tickers "$ALL_TICKERS"
        --cycles 1
        --execution-mode synthetic
        --no-resume
        --sleep-seconds 0
    )
    if [[ -n "$as_of_date" ]]; then
        cmd+=(--as-of-date "$as_of_date")
    fi
    "${cmd[@]}" >> "$LOG" 2>&1
}

run_platt_bootstrap_cycle() {
    # Platt bootstrap requires CLOSED ts_* trades for update_platt_outcomes.py.
    #
    # Adversarial finding (2026-02-25): fixed --as-of-date with --cycles 8 and
    # default --bar-aware produces cycle-1 executions then SKIPPED_SAME_BAR for
    # cycles 2..8, so bars do not progress and time exits never trigger.
    #
    # Fix: run 1 cycle per historical date and carry state across dates
    # (resume after first window). This advances bar timestamps across runs and
    # allows proof-mode holding exits to realize ts_* closes.
    local as_of_date="${1:-}"
    local resume_flag="${2:---resume}"
    local -a cmd=(
        "$PYTHON" scripts/run_auto_trader.py
        --tickers "$ALL_TICKERS"
        --cycles 1
        --execution-mode synthetic
        --sleep-seconds 0
        --proof-mode
        "$resume_flag"
    )
    if [[ -n "$as_of_date" ]]; then
        cmd+=(--as-of-date "$as_of_date")
    fi
    "${cmd[@]}" >> "$LOG" 2>&1
}

reconcile_platt_outcomes() {
    local context="$1"
    run_best_effort "update_platt_outcomes.py (${context})" "$PYTHON" scripts/update_platt_outcomes.py
}

log_section "Overnight refresh started"
log "Python: $PYTHON"
log "Repo: $REPO_ROOT"
log "Log: $LOG"
log "Tickers: $TICKERS"
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
reconcile_platt_outcomes "pre-run reconciliation"
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
if run_synthetic_auto_trader; then
    log "[PASS] Synthetic auto_trader cycle completed"
else
    log "[WARN] Synthetic auto_trader cycle returned non-zero (Platt pairs may not accumulate)"
    ERRORS=$((ERRORS + 1))
fi

reconcile_platt_outcomes "post-auto-trader reconciliation"

# ---------------------------------------------------------------------------
# STEP 2.6: Historical Platt bootstrap -- seed (confidence, outcome) pairs
# from backtested dates so calibration has material to work with immediately.
# Set PLATT_BOOTSTRAP=1 to activate (off by default to keep overnight fast).
# Uses --as-of-date to rewind market data to each historical date.
# ---------------------------------------------------------------------------
if [[ "${PLATT_BOOTSTRAP:-0}" == "1" ]]; then
    log_section "STEP 2.6/3: Platt bootstrap -- seeding pairs from 2021-2024 historical data"
    BOOTSTRAP_FIRST=1
    for AS_OF in 2021-01-01 2021-07-01 2022-01-01 2022-07-01 2023-01-01 2023-07-01 2024-01-01 2024-07-01; do
        log "--- bootstrap as-of $AS_OF"
        if [[ "$BOOTSTRAP_FIRST" -eq 1 ]]; then
            resume_flag="--no-resume"
            BOOTSTRAP_FIRST=0
        else
            resume_flag="--resume"
        fi
        if run_platt_bootstrap_cycle "$AS_OF" "$resume_flag"; then
            log "[PASS] bootstrap as-of $AS_OF"
        else
            log "[WARN] bootstrap as-of $AS_OF returned non-zero"
        fi
        reconcile_platt_outcomes "bootstrap as-of $AS_OF"
    done
    log "[DONE] Platt bootstrap complete"
fi

# ---------------------------------------------------------------------------
# STEP 2.7: Audit gate bootstrap -- seed 20 unique forecast audit windows
# ---------------------------------------------------------------------------
# Problem: check_forecast_audits.py deduplicates by (dataset.start, dataset.end,
# dataset.length, forecast_horizon). The fixed overnight date range always yields
# ~11 unique windows (< holding_period_audits=20). This step generates 20 unique
# dataset windows so the lift gate exits INCONCLUSIVE and makes a definitive verdict.
# Set AUDIT_GATE_BOOTSTRAP=1 to activate (off by default; run once after config changes).
# ---------------------------------------------------------------------------
if [[ "${AUDIT_GATE_BOOTSTRAP:-0}" == "1" ]]; then
    log_section "STEP 2.7/3: Audit gate bootstrap (20 AS_OF windows for lift gate)"
    AUDIT_WIN_PASS=0
    AUDIT_WIN_FAIL=0
    for AS_OF in \
        2022-01-03 2022-04-04 2022-07-05 2022-10-03 \
        2023-01-03 2023-04-03 2023-07-03 2023-10-02 \
        2024-01-02 2024-04-01 2024-07-01 2024-10-01 \
        2025-01-02 2025-04-01 2025-07-01 2025-10-01 \
        2026-01-02 2026-01-15 2026-02-02 2026-02-16; do
        log "--- audit window as-of $AS_OF"
        if "$PYTHON" scripts/run_auto_trader.py \
                --tickers "$ALL_TICKERS" \
                --cycles 1 \
                --execution-mode synthetic \
                --as-of-date "$AS_OF" \
                --no-resume \
                --sleep-seconds 0 >> "$LOG" 2>&1; then
            log "[PASS] audit window as-of $AS_OF"
            AUDIT_WIN_PASS=$((AUDIT_WIN_PASS + 1))
        else
            log "[WARN] audit window as-of $AS_OF returned non-zero"
            AUDIT_WIN_FAIL=$((AUDIT_WIN_FAIL + 1))
        fi
    done
    log "Audit window results: ${AUDIT_WIN_PASS} passed / ${AUDIT_WIN_FAIL} failed"
    log "[DONE] Audit gate bootstrap complete"
fi

# ---------------------------------------------------------------------------
# STEP 3: Health check + headroom measurement
# ---------------------------------------------------------------------------
log_section "STEP 3/3: Final health check"

run_best_effort "check_quant_validation_health.py" "$PYTHON" scripts/check_quant_validation_health.py
run_best_effort "quant_validation_headroom.py --json" "$PYTHON" scripts/quant_validation_headroom.py --json
# Phase 7.13-C2: Refresh forecast audit cache so production_audit_gate has current data at 7 AM cron.
# Phase 7.15: --allow-inconclusive-lift: holding_period_audits=20 requires 20 unique AS_OF date
# windows; overnight refresh (fixed 2024-01-01->2026-01-01 range) is structurally capped at ~11
# unique audit windows after dedup. Inconclusive during holding period = non-failing per config.
run_best_effort "production_audit_gate.py (refresh forecast_audits_cache/latest_summary.json)" "$PYTHON" scripts/production_audit_gate.py --allow-inconclusive-lift

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
