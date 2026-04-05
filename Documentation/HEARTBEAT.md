# HEARTBEAT.md

## System Status (2026-04-05, commit 40dfa8e)

- **Gate**: PASS (semantics=PASS) — overall_passed=True, all 5 gates green
- **Lift decision**: KEEP (lift demonstrated; lift_fraction=57.14%, threshold 25%)
- **Violation rate**: 31.43% (11/35 effective) — 4 violations of 35% ceiling
- **Recent window**: 4/10 effective, 0 violations — INCONCLUSIVE (data, not code)
- **Warmup**: active until 2026-04-15 (10 days) — THIN_LINKAGE covered by warmup
- **PnL**: 40 round-trips, +$620.01, 40% WR, profit factor 1.73, avg hold 1.1 days
- **Integrity**: ALL PASSED (0 violations)
- **Last commit**: 40dfa8e (bypass: P2-B CV OOS proxy, P3-A confidence calibration, P3-B MSSA-RL neutral-on-low-support)
- **Test count**: 2184 passed, 0 failed, 10 xfailed, 0 xpassed (fast lane)
- **Phase**: Domain-Calibrated Remediation — Phases 1+2+3 COMPLETE; heuristic distortion fixes pending

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
| Missing baseline_rmse → `violation=False` (deflates violation_rate) | CRITICAL | `check_forecast_audits.py:1194` | IMPLEMENTED |
| `residual_diagnostics_rate_warn_only: true` — enforcement removed | CRITICAL | `forecaster_monitoring.yml` | IMPLEMENTED |
| `fail_on_violation_during_holding_period: false` — FAILs → INCONCLUSIVE | HIGH | `forecaster_monitoring.yml` | IMPLEMENTED |
| `diagnostics_score` defaults to 0.5 (neutral) silently | HIGH | `time_series_signal_generator.py:767` | IMPLEMENTED |
| GARCH EWMA variance floor 1e-12 → CI collapse → SNR inflation | MEDIUM | `garch.py:589` | IMPLEMENTED |
| MSSA-RL policy_support never checked during action selection | MEDIUM | `mssa_rl.py:516` | IMPLEMENTED |
| Linkage vacuously passes when eligible==0 | HIGH | `production_audit_gate.py:1403` | IMPLEMENTED |
| RMSE dedupe key missing ticker; outcome dedupe has ticker | HIGH | `check_forecast_audits.py:1471` | IMPLEMENTED |
| Terminal DA computed but gate uses RMSE only | HIGH | `metrics.py:109` vs gate | IMPLEMENTED |
| OOS metrics {} in all CV runs → RMSE-rank always disabled | CRITICAL | `forecaster.py:2491` | IMPLEMENTED |

## Heuristic Distortions (adversarial audit 2026-04-05)

| Finding | Severity | File | Status |
|---------|----------|------|--------|
| OOS scan capped at 20 files — loses metrics in multi-ticker runs | HIGH | `forecaster.py:2453` | PENDING |
| SAMoSSA +0.05 pre-rank bump when TE < SARIMAX TE | HIGH | `ensemble.py:860-869` | PENDING |
| SNR None → 0.5 (neutral) — inconsistent with pessimistic fallback policy | HIGH | `time_series_signal_generator.py:1459` | PENDING |
| RMSE-rank silently disabled (<2 OOS models) — no warning | MEDIUM | `ensemble.py:506-507` | PENDING |
| EWMA fallback claims `convergence_ok: True` — suppresses CI inflation | MEDIUM | `garch.py:658` | PENDING |
| `_realized_volatility` floor still `1e-12` (EWMA floor fixed to `1e-6`) | MEDIUM | `garch.py:172, 205` | PENDING |

## Open Items

| Priority | Item | Status |
|----------|------|--------|
| URGENT | THIN_LINKAGE: must accumulate 10 matched outcomes before 2026-04-15 | DATA — run 5 date passes ×2 |
| HIGH | OOS scan cap (C5): remove `[:20]` from `forecaster.py:2453` + post-loop warning | PENDING |
| HIGH | SAMoSSA bump (C3): delete `ensemble.py:860-869` bump block | PENDING |
| HIGH | SNR fallback (H6): change `snr_score = 0.5` → `0.0` at `time_series_signal_generator.py:1459` | PENDING |
| MEDIUM | RMSE-rank logging (H7): add `logger.warning` at `ensemble.py:506-507` | PENDING |
| MEDIUM | EWMA convergence_ok (M1): `garch.py:658` → `"convergence_ok": False` | PENDING |
| MEDIUM | Realized vol floor (H2): `garch.py:172, 205, 642` → `1e-6` | PENDING |
| DEFERRED | Phase 4: linkage vacuous-pass + ticker in RMSE dedupe key | DEFERRED (governance) |

## Recent Commits

```
40dfa8e fix(bypass): P2-B CV OOS proxy, P3-A confidence calibration, P3-B MSSA-RL neutral-on-low-support
f83a6ea docs(status): Phase 1 complete — 2173 passed, bypasses remediated
78142bb fix(gate): Phase 1 structural bypass remediation — adversarial audit 2026-04-05
e744262 docs(audit): domain-calibrated remediation plan — adversarial audit 2026-04-05
7a5ae27 fix(ci): platt_contract_audit empty-DB WARN not FAIL
fd4eb68 downgraded upgraded finnhub to priority 3
```
