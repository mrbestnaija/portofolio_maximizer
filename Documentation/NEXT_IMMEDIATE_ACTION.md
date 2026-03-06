# Next Immediate Action

**Last Updated**: 2026-03-06
**Status**: PHASE 1 — Evidence Stabilization
**Protocol**: [Research-Grade Experiment Protocol v3](RESEARCH_EXPERIMENT_PROTOCOL.md)

---

## Autonomous Decision (2026-03-06)

Applying the protocol decision hierarchy:

```
execution_log_freshness = 63h  (threshold: <= 24h)
→ PHASE 1: repair execution telemetry freshness
```

All Phase-5 strategy experiments are blocked until Phases 1–4 are complete.

---

## Phase 1 Actions (Unblock Now)

### Action 1.1 — Restore execution telemetry freshness

The execution log is 63 hours stale. Cron job `[P1] Trading Cycles` should handle this
automatically (daily 10 AM), but can be triggered manually:

```bash
source simpleTrader_env/Scripts/activate
python scripts/run_auto_trader.py \
    --tickers NVDA,MSFT,GOOG,JPM \
    --cycles 3 \
    --execution-mode auto
```

**Success signal**: `project_runtime_status.py` shows `execution_age_hours <= 24`.

### Action 1.2 — Verify integrity gate clears

```bash
python scripts/capital_readiness_check.py --json
```

Expected after Action 1.1:
- `R6` → cleared (0 lifecycle violations — already fixed)
- `R3` → FAIL until WR crosses 45% (data-driven, not fixable by a single run)
- `R2` → passes once `run_all_gates.py` is run post-execution

### Action 1.3 — Run production gates to refresh gate artifact

```bash
python scripts/run_all_gates.py --json
```

Writes `logs/gate_status_latest.json`. R2 requires this artifact to be < 26h old.

---

## Phase 2 Preview (Linkage Coverage)

Once Phase 1 is complete, run:

```bash
python scripts/update_platt_outcomes.py
python scripts/outcome_linkage_attribution_report.py --json
```

Target: `outcome_matched >= 10`, `matched/eligible >= 0.80`.

Current state: `matched = 0/0` (thin linkage — no matched outcomes yet).

---

## Phase 3 Preview (Evaluation Completeness)

Many forecast audit files are missing `evaluation_metrics`. Fix by running fresh audit windows:

```bash
bash bash/overnight_refresh.sh
python scripts/check_model_improvement.py --layer 1 --json
```

Target: `evaluation_metrics_coverage >= 80%`.
Current state: ~30% coverage (many legacy windows missing the field).

---

## Phase 4 Preview (Attribution)

Once Phases 1–3 pass, run the full attribution suite:

```bash
python scripts/exit_quality_audit.py --json
python scripts/compute_ticker_eligibility.py --json
python scripts/compute_context_quality.py --json
python scripts/check_model_improvement.py --json
```

Largest known leak sources (from current data):
- `STOP_LOSS` exits dominating losses — avg loss $75.91, PF 0.80
- Ensemble lift CI [-0.085, -0.035] → ensemble rarely beats best-single (1.4% of windows)
- WR=40% below 45% R3 gate — signal quality, not execution, is the bottleneck

---

## Phase 5 Experiment Backlog (Blocked)

| ID | Hypothesis | Blocked by |
|----|-----------|------------|
| EXP-001 | Tighter ATR stop reduces tail-loss magnitude | Phases 1-4 |
| EXP-002 | Longer max_holding reduces correct-direction-but-loss exits | Phases 1-4 |
| EXP-003 | Regime filter blocks WEAK ticker entries in high-vol | Phases 1-4 |
| EXP-004 | Higher confidence gate improves entry quality | Phases 1-4 |
| EXP-005 | Reduced position size for LAB_ONLY tickers improves PF | Phases 1-4 |

---

## Capital Readiness Snapshot (2026-03-06)

| Gate | Status | Detail |
|------|--------|--------|
| R1 adversarial | PASS | 0/21 confirmed, all cleared |
| R2 gate artifact | STALE | run_all_gates needed |
| R3 trade quality | FAIL | WR=40% < 45%, PF=0.80 < 1.30 |
| R4 calibration | PASS | Brier=0.235, tier=jsonl |
| R5 lift CI | WARNING | CI [-0.085, -0.035], advisory only |
| R6 lifecycle | PASS | 0 violations (fixed 2026-03-06) |

---

## Observability Commands (Run Any Time)

```bash
# Full readiness snapshot
python scripts/capital_readiness_check.py --json

# Execution freshness
python scripts/project_runtime_status.py --json

# Model quality
python scripts/check_model_improvement.py --json

# Trade quality
python -m integrity.pnl_integrity_enforcer

# Adversarial health
python scripts/adversarial_diagnostic_runner.py --severity HIGH --json
```
