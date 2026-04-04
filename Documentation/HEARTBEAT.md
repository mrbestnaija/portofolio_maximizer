# HEARTBEAT.md

## System Status (2026-04-04, commit b44ea4e)

- **Gate**: PASS (semantics=PASS) — overall_passed=True, all 5 gates green
- **Lift decision**: KEEP (lift demonstrated; lift_fraction=58.82%, threshold 25%)
- **Violation rate**: 32.35% (11/34 effective) — 2 violations of 35% ceiling
- **Recent window**: 3/10 effective, 0 violations — INCONCLUSIVE (data, not code)
- **Warmup**: active until 2026-04-15 (11 days)
- **PnL**: 40 round-trips, +$620.01, 40% WR, profit factor 1.73, avg hold 1.1 days
- **Integrity**: ALL PASSED (0 violations)
- **Last commit**: b44ea4e (fix(ci): remove stale xfail, add trained-artifact tests, smooth vol-bands)
- **Test count**: 2165 passed, 0 failed, 10 xfailed, 0 xpassed (fast lane)
- **Phase**: Post-P4 Adversarial Remediation — Items 1/2/4 complete; Item 3 data-driven

## Gate Summary

| Gate | Status | Key Metric |
|------|--------|------------|
| ci_integrity_gate | PASS | 40 round-trips, ALL checks passed |
| check_quant_validation_health | PASS | — |
| production_audit_gate | PASS | semantics=PASS, KEEP decision |
| production_gate_schema | PASS | — |
| institutional_unattended_gate | PASS | — |

## Active Cron Jobs (OpenClaw)

| Job | Schedule | Last Status |
|-----|----------|-------------|
| [P0] PnL Integrity Audit | Every 4h | ALL PASSED |
| [P0] Production Gate Check | Daily 7 AM | PASS (semantics=PASS) |
| [P1] Directional Classifier Health | Daily 8:45 AM | Running |
| [P2] GARCH Unit-Root Guard | Weekly Mon 9 AM | Running |
| System Health Check | Every 6h | Running |
| Weekly Session Cleanup | Sunday 3 AM | Silent |

## Open Items

| Priority | Item | Status |
|----------|------|--------|
| URGENT | Evidence generation: run 5 `--as-of-date` dates x2 (>=2 min apart) before warmup 2026-04-15 | PENDING |
| DEFERRED | Ticker in RMSE dedup key — gate-contract change, governance decision required | DEFERRED |
| OPEN | MSSA-RL Q-table stub cleanup, GARCH lam=0.94 externalization | OPEN (P4 backlog) |

## Recent Commits

```
b44ea4e fix(ci): remove stale xfail, add trained-artifact tests, smooth vol-bands
37bcc56 docs(audit): update gate lift audit to 2026-04-04 state after P4 merge
dd9a633 feat(mssa-rl): offline policy artifact + forecaster hardening (Phase P4)
503d5f2 fix(gate): ticker fail-open, as-of-date audit routing, institutional latch
```
