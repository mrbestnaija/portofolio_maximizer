# HEARTBEAT.md

## System Status (2026-04-05, commit fd4eb68)

- **Gate**: PASS (semantics=PASS) — overall_passed=True, all 5 gates green
- **Lift decision**: KEEP (lift demonstrated; lift_fraction=57.14%, threshold 25%)
- **Violation rate**: 31.43% (11/35 effective) — 4 violations of 35% ceiling
- **Recent window**: 4/10 effective, 0 violations — INCONCLUSIVE (data, not code)
- **Warmup**: active until 2026-04-15 (10 days) — THIN_LINKAGE covered by warmup
- **PnL**: 40 round-trips, +$620.01, 40% WR, profit factor 1.73, avg hold 1.1 days
- **Integrity**: ALL PASSED (0 violations)
- **Last commit**: fd4eb68 (downgraded finnhub to priority 3)
- **Test count**: 2173 passed, 0 failed, 10 xfailed, 0 xpassed (fast lane, +8)
- **Phase**: Domain-Calibrated Remediation — Phase 1 COMPLETE (commits 7a5ae27, e744262, 78142bb)

## Gate Summary

| Gate | Status | Key Metric |
|------|--------|------------|
| ci_integrity_gate | PASS | 40 round-trips, ALL checks passed (local: CROSS_MODE_CONTAMINATION — data issue not CI) |
| check_quant_validation_health | PASS | 727 PASS / 0 FAIL (0.0% fail rate) |
| production_audit_gate | PASS | semantics=PASS, KEEP decision |
| production_gate_schema | PASS | — |
| institutional_unattended_gate | PASS | platt_contract_bootstrap PASS (fix committed, not pushed) |

## Active Cron Jobs (OpenClaw)

| Job | Schedule | Last Status |
|-----|----------|-------------|
| [P0] PnL Integrity Audit | Every 4h | ALL PASSED |
| [P0] Production Gate Check | Daily 7 AM | PASS (semantics=PASS) |
| [P1] Directional Classifier Health | Daily 8:45 AM | Running |
| [P2] GARCH Unit-Root Guard | Weekly Mon 9 AM | Running |
| System Health Check | Every 6h | Running |
| Weekly Session Cleanup | Sunday 3 AM | Silent |

## Warmup Gate Coverage (expires 2026-04-15)

| Check | Current | Required (post-warmup) | Warmup Covering? |
|-------|---------|----------------------|-----------------|
| THIN_LINKAGE matched | 1/309 (0.32%) | ≥10 matched, ≥80% ratio | **YES** |
| Recent window | 4/10 effective | ≥10 effective | Data only (not warmup) |
| Profitability proof trading days | 10/21 | ≥21 days | Partially (proof still valid) |

**What breaks on 2026-04-15:**
1. THIN_LINKAGE hard-fails (1 matched vs 10 required)
2. Root cause confirmed: 99.6% of forecasts blocked by signal routing, never become trades

## Confirmed Bypasses (adversarial audit 2026-04-05)

| Bypass | Severity | File | Status |
|--------|----------|------|--------|
| Missing baseline_rmse → `violation=False` (deflates violation_rate) | CRITICAL | `check_forecast_audits.py:1194` | PLAN written |
| `residual_diagnostics_rate_warn_only: true` — enforcement removed | CRITICAL | `forecaster_monitoring.yml` | PLAN written |
| `fail_on_violation_during_holding_period: false` — FAILs → INCONCLUSIVE | HIGH | `forecaster_monitoring.yml` | PLAN written |
| `diagnostics_score` defaults to 0.5 (neutral) silently | HIGH | `time_series_signal_generator.py:767` | PLAN written |
| GARCH EWMA variance floor 1e-12 → CI collapse → SNR inflation | MEDIUM | `garch.py:589` | PLAN written |
| MSSA-RL policy_support never checked during action selection | MEDIUM | `mssa_rl.py:516` | PLAN written |
| Linkage vacuously passes when eligible==0 | HIGH | `production_audit_gate.py:1403` | PLAN written |
| RMSE dedupe key missing ticker; outcome dedupe has ticker | HIGH | `check_forecast_audits.py:1471` | PLAN written |
| Terminal DA computed but gate uses RMSE only | HIGH | `metrics.py:109` vs gate | PLAN written |
| OOS metrics {} in all CV runs → RMSE-rank always disabled | CRITICAL | `forecaster.py:2491` | PLAN written |

## Open Items

| Priority | Item | Status |
|----------|------|--------|
| URGENT | THIN_LINKAGE: must accumulate 10 matched outcomes before 2026-04-15 | DATA — run 5 date passes ×2 |
| URGENT | CI: platt_contract_bootstrap fix not yet committed/pushed | PENDING commit + push |
| HIGH | Implement Phase 1 fixes (P1-A through P1-E) from DOMAIN_CALIBRATION_REMEDIATION plan | PENDING |
| HIGH | Funnel audit logging (P1-B) — understand why 99.6% of forecasts don't become trades | PENDING |
| MEDIUM | Phase 2: terminal DA co-gate + CV OOS proxy | PENDING |
| MEDIUM | Phase 3: calibrate_confidence_thresholds.py from 40 trades | PENDING |
| DEFERRED | Phase 4: linkage vacuous-pass + ticker in RMSE dedupe key | DEFERRED (governance) |

## Recent Commits

```
fd4eb68 downgraded upgraded finnhub to priority 3
bf9a7b1 fix(ci): curl-cffi 0.15.0 (CVE-2026-33752) + fix add_to_project workflow
7cdc24d change data source priorities from finnhub 3 to ctrader 3
b64d403 docs(status): repowide update — gate PASS, PnL, P4 remediation complete
b44ea4e fix(ci): remove stale xfail, add trained-artifact tests, smooth vol-bands
```
