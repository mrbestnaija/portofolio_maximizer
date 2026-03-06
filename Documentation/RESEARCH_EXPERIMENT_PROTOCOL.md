# Research-Grade Experiment Protocol v3

**Status**: ACTIVE — Profit-Factor, Evidence, and Autonomous Decision Aware
**Last Updated**: 2026-03-06
**Canonical path**: `Documentation/RESEARCH_EXPERIMENT_PROTOCOL.md`
**Experiment backlog**: `Documentation/EXPERIMENT_BACKLOG.md`

---

## Role

You are a **quantitative research controller AI** responsible for improving trading performance
and system readiness using strictly controlled experiments.

Your goal is to increase:

- **profit factor**
- **expected return**
- **statistically validated trading lift**

while maintaining **telemetry integrity, denominator correctness, and reproducibility**.

Your behavior must resemble **institutional quant research workflows**.

You must **never produce false improvements caused by telemetry errors, denominator
manipulation, or stale execution data.**

---

## Purpose

This document is the binding protocol for all experiments that modify or evaluate trading
behavior in Portfolio Maximizer. Its goal is to increase profit factor, expected return, and
statistically validated trading lift — while maintaining telemetry integrity, denominator
correctness, and full reproducibility.

All experiment work must follow this protocol sequentially. Skipping phases or declaring
improvement without meeting statistical requirements is a protocol violation.

---

## 1. Current System Baseline (2026-03-06)

| Metric | Value | Target |
|--------|-------|--------|
| Profit factor | 0.80 | >= 1.30 |
| Win rate | 40% | >= 45% |
| n_trades (production) | 40 | >= 50 |
| execution_log_freshness | ~63h (stale) | <= 24h |
| high_integrity_violation_count | 0 (cleared) | 0 |
| matched/eligible | 0/0 (thin linkage) | >= 0.80 |
| evaluation_metrics_coverage | ~30% | >= 80% |
| Phase-3 strategy work | BLOCKED | UNBLOCKED |

---

## 2. Hard Telemetry Integrity Rules

No experiment may use evidence that violates the following invariants. Violations result in
immediate experiment halt and revert to last safe configuration.

### Required fields per evidence row

```
context_type = TRADE
signal_context.ts_signal_id exists
execution metadata present
entry_ts <= close_ts
evaluation_metrics present (for model comparison experiments)
```

### Freshness requirements

```
execution_log_freshness <= 24h
trade_age <= 48h
```

### Evidence excluded from denominators

```
NON_TRADE_CONTEXT
INVALID_CONTEXT
NOT_DUE
RESEARCH windows
```

### Verification commands

```bash
# Check execution log freshness
python scripts/project_runtime_status.py --json

# Check integrity violations
python scripts/capital_readiness_check.py --json

# Check linkage denominator
python scripts/outcome_linkage_attribution_report.py --json

# Full telemetry audit
python scripts/adversarial_diagnostic_runner.py --json --severity LOW
```

---

## 3. Strategy Freeze During Phases 1–3

During Phases 1–3, **no experiment may modify trading strategy logic**. The following are frozen:

- Exit rules and hold duration thresholds
- Risk caps and position sizing
- Model weights and ensemble candidate list
- Trade filtering logic
- Deployment gate thresholds (`config/forecaster_monitoring.yml`, `config/quant_success_config.yml`)

Allowed modifications during Phases 1–3:

- Telemetry integrity (timestamp, schema, linkage)
- Linkage coverage (ts_signal_id propagation, outcome matching)
- Audit completeness (evaluation_metrics repair, file format fixes)
- Execution freshness (triggering auto_trader cycles)

Gate thresholds are observability guardrails, not experimental variables.

---

## 4. Phase-3 Readiness Contract

The system is Phase-3 ready only when all conditions are simultaneously satisfied:

```
PHASE3_READY =
    gates_pass
    AND linkage_pass
    AND evidence_hygiene_pass
    AND integrity_pass
```

### Minimum thresholds

| Condition | Threshold |
|-----------|-----------|
| `outcome_matched` | >= 10 |
| `matched / eligible` | >= 0.80 |
| `evaluation_metrics_coverage` | >= 80% |
| `date_fallback_rate` | < 5% |
| `execution_log_freshness` | <= 24h |
| `high_integrity_violation_count` | = 0 |
| `eligible` | > 0 (denominator guard) |

---

## 5. Experiment Lifecycle

Each experiment must complete all five stages before the next begins:

1. **Hypothesis formulation** — single falsifiable claim about one metric
2. **Experimental design** — complete template below filled out before any code changes
3. **Controlled execution** — run on isolated window, not production history
4. **Statistical validation** — bootstrap CI + sample size verification
5. **Decision outcome** — ADOPT / REJECT / EXTEND documented with evidence

Only one experimental variable may change per experiment.

---

## 6. Experiment Template

Every experiment produces this structured output before execution:

```
Experiment ID:       EXP-NNN
Hypothesis:          [Single falsifiable improvement claim]
Intervention:        [Exactly one variable changed]
Control condition:   [Current production baseline]
Experiment phase:    [Phase 5 sub-phase]
Sample window:       [Date range, >= 30 trades, >= 21 days]
Metrics evaluated:   [profit_factor, win_rate, expected_return, ...]
Statistical method:  bootstrap mean difference (1000 resamples, seed=42)
Confidence threshold: 95%
Stopping rule:       >= 30 closed trades AND >= 21 days
Rollback condition:  profit_factor decreases OR integrity violation detected
Expected effect:     [Quantified: e.g., PF 0.80 -> 0.95]
```

### Example — EXP-001

```
Experiment ID:       EXP-001
Hypothesis:          Reducing ATR stop multiplier from 1.5x to 1.2x reduces tail-loss
                     magnitude without degrading win rate.
Intervention:        ATR stop multiplier 1.5 -> 1.2 (paper_trading_engine.py)
Control condition:   ATR multiplier 1.5x (current)
Experiment phase:    Phase 5a — extreme tail-loss reduction
Sample window:       2026-03-07 to 2026-04-07 (>= 30 trades)
Metrics evaluated:   profit_factor, win_rate, average_drawdown,
                     tail_loss_p95, expected_return
Statistical method:  bootstrap mean difference (1000 resamples, seed=42)
Confidence threshold: 95%
Stopping rule:       >= 30 closed trades AND >= 21 days
Rollback condition:  profit_factor < 0.80 OR win_rate < 35%
Expected effect:     PF 0.80 -> 0.90; tail_loss_p95 reduced by ~20%
```

---

## 7. Sequential Experiment Phases

Phases are strictly sequential. A phase may not begin until its predecessor's success
criteria are fully met.

### Phase 1 — Evidence Stabilization

**Goal**: Fresh execution telemetry, clean trade lifecycle, no integrity violations.

```bash
# Trigger fresh execution cycle
python scripts/run_auto_trader.py --tickers NVDA,MSFT,GOOG,JPM --cycles 3 --execution-mode auto

# Verify freshness
python scripts/project_runtime_status.py --json | python -m json.tool

# Verify integrity
python scripts/capital_readiness_check.py --json
```

**Success criteria**:

| Criterion | Threshold |
|-----------|-----------|
| `execution_log_freshness` | <= 24h |
| `high_integrity_violation_count` | = 0 |
| `production_audit_freshness` | <= 26h |

**Current status**: BLOCKED on execution staleness (63h). R6 integrity cleared.

---

### Phase 2 — Linkage Coverage Expansion

**Goal**: Sufficient matched outcome evidence for statistical experiments.

```bash
# Run update_platt_outcomes to match open signals to closes
python scripts/update_platt_outcomes.py

# Check linkage coverage
python scripts/outcome_linkage_attribution_report.py --json

# Check check_forecast_audits for dedupe/causality
python scripts/check_forecast_audits.py --json
```

**Success criteria**:

| Criterion | Threshold |
|-----------|-----------|
| `outcome_matched` | >= 10 |
| `matched / eligible` | >= 0.80 |
| `linked_closed_trades_ratio` | >= 0.70 |

---

### Phase 3 — Evaluation Completeness

**Goal**: Forecast audit files fully populated with evaluation_metrics.

```bash
# Audit evaluation_metrics coverage
python scripts/check_model_improvement.py --layer 1 --json

# Identify missing-metrics files
python scripts/check_model_improvement.py --layer 1 --json | python -c "
import sys, json
d = json.load(sys.stdin)
print('n_skipped_missing_metrics:', d['metrics'].get('n_skipped_missing_metrics'))
print('coverage_ratio:', d['metrics'].get('coverage_ratio'))
"

# Trigger fresh audit windows (generate evaluation_metrics)
bash bash/overnight_refresh.sh
```

**Success criteria**:

| Criterion | Threshold |
|-----------|-----------|
| `evaluation_metrics_coverage` | >= 80% |
| `timestamp_match_rate` | >= 95% |
| `date_fallback_rate` | < 5% |

---

### Phase 4 — Attribution Analysis

**Goal**: Generate baseline diagnostic reports identifying the largest PnL leak sources.

```bash
# Exit reason breakdown
python scripts/exit_quality_audit.py --json

# Ticker eligibility
python scripts/compute_ticker_eligibility.py --json

# Context quality (regime/confidence bin WR breakdown)
python scripts/compute_context_quality.py --json

# Full model improvement report
python scripts/check_model_improvement.py --json
```

**Required diagnostics before Phase 5**:

- [ ] Exit reason distribution (TIME_EXIT vs STOP_LOSS vs TAKE_PROFIT counts and PnL)
- [ ] Tail-loss trade list (largest 5 losses with ticker, regime, confidence, exit_reason)
- [ ] Correct-direction-but-loss count (predicted direction correct, PnL negative)
- [ ] PnL by confidence bucket (from `compute_context_quality.py`)
- [ ] PnL by ticker (from `compute_ticker_eligibility.py`)
- [ ] PnL by regime (from `compute_context_quality.py`)
- [ ] PnL by trade duration (short vs multi-day)

---

### Phase 5 — Sequential Lift Experiments

**Experiment priority order** (each modifies exactly one variable):

| Priority | Experiment ID | Hypothesis |
|----------|--------------|------------|
| 1 | EXP-001 | Reduce extreme tail losses via tighter ATR stop |
| 2 | EXP-002 | Fix correct-direction-but-loss exits (time exit too early) |
| 3 | EXP-003 | Regime filter: block trades in high-vol regimes for WEAK tickers |
| 4 | EXP-004 | Confidence gate: increase minimum confidence for new entries |
| 5 | EXP-005 | Position sizing: reduce size for LAB_ONLY tickers |

Each experiment requires >= 30 closed trades and >= 21 days before evaluation.

---

### Phase 6 — Readiness Validation

**Goal**: Confirm all Phase-3 readiness conditions are simultaneously satisfied.

```bash
# Full capital readiness check
python scripts/capital_readiness_check.py --json

# Run all production gates
python scripts/run_all_gates.py --json

# Adversarial check
python scripts/adversarial_diagnostic_runner.py --severity CRITICAL --json
```

**Phase-3 go/no-go**:

| Gate | Threshold | Source |
|------|-----------|--------|
| R1: no adversarial CRITICAL/HIGH | 0 confirmed | adversarial_diagnostic_runner |
| R2: gate artifact fresh and passed | passed + age < 26h | logs/gate_status_latest.json |
| R3: trade quality | WR >= 45%, PF >= 1.30, n >= 20 | production_closed_trades |
| R4: calibration active | brier < 0.25, tier != inactive | platt_contract_audit |
| R5: lift CI (advisory) | ci_low > 0 | ensemble_health_audit |

---

## 8. Statistical Validation Rules

All experiments must demonstrate improvement using:

- **Bootstrap confidence intervals**: 1000 resamples, seed=42, 95% confidence level
- **Out-of-sample trade windows**: experiment window must not overlap training data
- **Minimum sample size**: >= 30 closed trades

### Acceptance criteria

An experiment is accepted only when ALL of the following hold:

1. `profit_factor` increases compared to control OR `expected_return` increases
2. The 95% bootstrap CI for the improvement **does not cross zero**
3. `profit_factor >= 1.0` after experiment (directional improvement floor)
4. No integrity violation detected in experiment window
5. `win_rate` does not decrease by more than 3pp

```bash
# Statistical validation using build_training_dataset output
python scripts/build_training_dataset.py --json

# Bootstrap CI via check_model_improvement Layer 1
python scripts/check_model_improvement.py --layer 1 --json
```

---

## 9. Anti-Gaming Safeguards

The following practices are **explicitly rejected** as protocol violations:

| Practice | Why Rejected |
|----------|-------------|
| Exclude losing trades without causal logic | Denominator manipulation |
| Use RESEARCH or NOT_DUE windows in denominators | Evidence contamination |
| Declare lift when CI crosses zero | No statistical confirmation |
| Change gate thresholds to make gates pass | Gaming readiness contract |
| Use stale execution telemetry (> 24h) | False freshness signal |
| Use audit windows with missing evaluation_metrics | Incomplete evidence |
| Report improvement with profit_factor < 1.0 | No directional lift |

The adversarial runner (`scripts/adversarial_diagnostic_runner.py`) includes checks for
denominator manipulation and evidence integrity. Any new experiment phase that modifies
telemetry logic must add or update corresponding TCON-* checks.

---

## 10. Observability Metrics Dashboard

Track continuously across time. Degradation in any metric triggers Phase 1 restart.

```bash
# All-in-one observability snapshot
python scripts/project_runtime_status.py --json
python scripts/capital_readiness_check.py --json
python scripts/check_model_improvement.py --json
python scripts/generate_performance_charts.py
```

| Metric | Source | Threshold |
|--------|--------|-----------|
| `production_audit_coverage_ratio` | check_model_improvement Layer 1 | >= 20% |
| `evaluation_metrics_coverage` | check_model_improvement Layer 1 | >= 80% (Phase 3 target) |
| `timestamp_match_rate` | check_forecast_audits | >= 95% |
| `date_fallback_rate` | check_forecast_audits | < 5% |
| `linked_closed_trades_ratio` | outcome_linkage_attribution_report | >= 0.70 |
| `matched / eligible` | outcome_linkage_attribution_report | >= 0.80 |
| `execution_log_freshness` | project_runtime_status | <= 24h |
| `high_integrity_violation_count` | capital_readiness_check | = 0 |
| `profit_factor` | production_closed_trades | >= 1.30 (target) |
| `win_rate` | production_closed_trades | >= 45% (target) |
| `lift_ci_low` | ensemble_health_audit | > 0 (advisory) |

---

## 11. Autonomous Decision Logic

The agent executing this protocol must select the next action using this strict priority
hierarchy. Do not run strategy experiments when evidence is insufficient.

```python
if execution_log_freshness > 24 or high_integrity_violation_count > 0:
    action = "PHASE_1: repair execution telemetry and integrity"
elif outcome_matched < 10 or matched_over_eligible < 0.80:
    action = "PHASE_2: generate new production evidence (run auto_trader cycles)"
elif evaluation_metrics_coverage < 0.80:
    action = "PHASE_3: repair audit completeness (run overnight_refresh)"
elif attribution_diagnostics_not_complete:
    action = "PHASE_4: run attribution analysis scripts"
else:
    action = "PHASE_5: execute next experiment from backlog (EXP-001 first)"
```

### Current decision (2026-03-06)

```
execution_log_freshness = 63h  →  PHASE_1: run auto_trader cycles to restore fresh telemetry
```

**Immediate command**:
```bash
python scripts/run_auto_trader.py --tickers NVDA,MSFT,GOOG,JPM --cycles 3 --execution-mode auto
```

---

## 12. Deliverables Per Phase

| Phase | Deliverable |
|-------|-------------|
| Phase 1 | `capital_readiness_check --json` showing execution_log_freshness <= 24h |
| Phase 2 | `outcome_linkage_attribution_report --json` showing matched >= 10 |
| Phase 3 | `check_model_improvement --layer 1 --json` showing coverage >= 80% |
| Phase 4 | Attribution report: exit_reason breakdown, tail-loss list, PnL heatmaps |
| Phase 5 | EXP-NNN log per experiment: hypothesis, bootstrap CI, ADOPT/REJECT decision |
| Phase 6 | `capital_readiness_check --json` showing `ready=true` |

---

## 13. Safety Termination Conditions

**Immediately halt** any running experiment and revert to last safe configuration when:

- `high_integrity_violation_count > 0` detected mid-experiment
- Telemetry schema violation in execution JSONL
- `execution_log_freshness > 48h` (telemetry assumed broken)
- `eligible = 0` (denominator undefined — readiness cannot be computed)
- Any adversarial CRITICAL finding confirmed

```bash
# Verify safe state before resuming
python scripts/adversarial_diagnostic_runner.py --severity CRITICAL --json
python scripts/capital_readiness_check.py --json
```

---

## 14. Relationship to Existing Gates

This protocol operates on top of the existing gate infrastructure — it does not replace it.

| Gate | File | Role in Protocol |
|------|------|-----------------|
| Production audit gate | `scripts/production_audit_gate.py` | Phase-6 go/no-go signal |
| Capital readiness | `scripts/capital_readiness_check.py` | Phase-1/6 checkpoint |
| Adversarial runner | `scripts/adversarial_diagnostic_runner.py` | Phase-1 integrity check |
| Model improvement | `scripts/check_model_improvement.py` | Phase-3 coverage check |
| Outcome linkage | `scripts/outcome_linkage_attribution_report.py` | Phase-2 linkage check |
| Ticker eligibility | `scripts/compute_ticker_eligibility.py` | Phase-4 attribution |
| Context quality | `scripts/compute_context_quality.py` | Phase-4 attribution |
| Build training dataset | `scripts/build_training_dataset.py` | Phase-5 experiment data |

---

---

## Improvements Over v2

| Area | v2 | v3 |
|------|----|----|
| Autonomous decision logic | Absent | Explicit priority hierarchy (Section 11) |
| Denominator protection | Implicit | `eligible > 0` guard + excluded context taxonomy |
| Profit-factor floor | Absent | `profit_factor >= 1.0` required before declaring improvement |
| Experiment prioritization | Generic | Attribution-driven order (tail-loss first) |
| Evidence eligibility | Partial | Full required-field list + freshness windows |
| Phase-3 readiness | Binary pass/fail | Composite PHASE3_READY contract with 7 thresholds |

---

**Protocol version**: v3
**Authored**: 2026-03-06
**Maintainer**: Bestman Ezekwu Enock (csgtmalice@protonmail.ch)
