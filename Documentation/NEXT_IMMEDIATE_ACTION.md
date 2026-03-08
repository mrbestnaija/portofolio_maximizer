# Next Immediate Action

**Last Updated**: 2026-03-08
**Status**: Denominator recovery only â€” no readiness or linkage-improvement claims
**Protocol**: [RESEARCH_EXPERIMENT_PROTOCOL.md](RESEARCH_EXPERIMENT_PROTOCOL.md)

---

## Current Evidence State (verified 2026-03-08)

```text
fresh_trade_rows = 1
fresh_linkage_included = 1
fresh_production_valid_matched = 0
-> stay in denominator-recovery mode
```

No readiness, linkage-improvement, or strategy-experiment claim is allowed until:

- fresh `TRADE` exclusions stay near zero across multiple cycles
- `fresh_linkage_included > 1`
- at least one fresh production-valid matched row appears

Next window for new evidence: **Monday, 2026-03-10** (first trading day). Weekend idle
is not progress or regression.

---

## Gate Status (verified 2026-03-08)

| Gate | Status | Root Cause |
|---|---|---|
| `ci_integrity_gate` | **PASS** | ALL PASSED; orphan whitelist extended for ids 249,250,251,253 |
| `check_quant_validation_health` | **PASS** | â€” |
| `production_audit_gate` | **FAIL** | profitable=False, matched=0/1, EVIDENCE_HYGIENE_FAIL â€” data-driven, no code can fix |
| `institutional_unattended_gate` | **FAIL** | cascades from production_audit_gate |

## Capital Readiness (verified 2026-03-08)

| Gate | Status | Root Cause |
|---|---|---|
| R1 adversarial | PASS | 0/21 confirmed adversarial findings |
| R2 gate artifact | FAIL | cascades from production_audit_gate |
| R3 trade quality | FAIL | WR=40%<45%, PF=0.80<1.30 â€” real trade outcomes, not wiring |
| R4 calibration | PASS | Brier below threshold |
| R5 lift CI | FAIL | CI=[-0.1139,-0.0572] definitively negative, 162 windows (now correctly hard FAIL â€” Phase 7.40 fixed wiring) |
| R6 lifecycle | PASS | cleared |

R2 and R3 require live trading cycles. R5 requires the ensemble architecture to produce
non-negative lift â€” this is a research problem, not a wiring problem.

## Resolved This Session (2026-03-08)

- `ci_integrity_gate` ORPHANED_POSITION: ids 249,250,251,253 whitelisted (AAPL batch-replay duplicates)
- Dashboard truth: stale positions â†’ fallback, `None` vs `0.0` unknown metrics, `exit_reason` in trade events
- Layer 1 CI semantics: definitively-negative CI now hard FAIL (Phase 7.40, `n_used` â†’ `n_used_windows` key bug fixed)
- `regime_detector.py` flat-series Hurst stability: `hurst=0.5` for constant series (Phase 7.40)
- `etl/regime_detector.py` now finite-clamps degenerate t-test outputs (`confidence`/`transition_probability` no longer NaN on flat inputs)
- Gate wiring now publishes `pre_institutional` status artifact before institutional P4 and fails closed on stale/missing prior-gate evidence
- Agent coordination docs: `AGENT_COORDINATION_PROTOCOL_2026-03-08.md` created
- `visualizations/live_dashboard.html` integrated: JS parse corruption removed and `performance_unknown` now renders `N/A`

---

## Immediate Actions

### Action 1 â€” Keep the fresh cohort strictly TRADE-only

```bash
python scripts/run_live_denominator_overnight.py \
  --tickers AAPL,AMZN,GOOG,GS,JPM,META,MSFT,NVDA,TSLA,V \
  --cycles 30 \
  --sleep-seconds 86400 \
  --resume \
  --stop-on-progress
```

Interpret the watcher using only these three signals:

- fresh `TRADE`-context exclusion counts
- fresh linkage denominator growth
- fresh production-valid matched rows

`NON_TRADE_CONTEXT` rows are diagnostics only. They must never be reintroduced into the
fresh TRADE denominator.

### Action 2 â€” Let the watcher accumulate daily evidence, not intraday noise

- Daily bars make sub-daily polling mostly noise.
- The watcher sleeps for `86400` seconds and skips weekends by default.
- No new production watcher cycles should be expected until Monday, 2026-03-10.

### Action 3 â€” Keep dashboard startup wiring active after reboot

```bash
python scripts/windows_dashboard_manager.py ensure --status-json logs/dashboard_manager_status.json
```

### Action 4 - Keep gate artifact contract intact

`scripts/run_all_gates.py` now writes a `pre_institutional` gate status artifact before
calling `institutional_unattended_gate.py`, then rewrites final status after all gates.

Operational dependency:

- preserve `logs/gate_status_latest.json` write path
- do not bypass pre-institutional write in unattended flows

---

## What To Watch Next (Monday 2026-03-10)

Only three watcher signals matter:

1. Fresh `TRADE`-context exclusion counts â€” target: near zero across multiple cycles
2. Fresh linkage denominator growth â€” target: `fresh_linkage_included > 1`
3. Fresh production-valid matched rows â€” target: `fresh_production_valid_matched >= 1`

Anything else is secondary until those three conditions are met.

---

## Observability Commands

```bash
# Watcher status without waiting for the next trading day
python scripts/run_live_denominator_overnight.py --tickers AAPL,AMZN,GOOG,GS,JPM,META,MSFT,NVDA,TSLA,V --cycles 1 --sleep-seconds 0 --dry-run

# Gate snapshot
python scripts/run_all_gates.py --json

# Capital readiness
python scripts/capital_readiness_check.py --json

# Integrity check
python -m integrity.pnl_integrity_enforcer

# Dashboard stack + watcher startup
python scripts/windows_dashboard_manager.py ensure --status-json logs/dashboard_manager_status.json
```
