# HEARTBEAT.md

## System Status (2026-04-15, commit 29806ac)

- **Gate**: PASS (semantics=PASS) — warmup EXPIRED today; THIN_LINKAGE now hard-enforced
- **Lift decision**: KEEP (lift_fraction=57.14%, threshold 25%)
- **Violation rate**: 31.43% (11/35 effective) — within 35% ceiling
- **Recent window**: 4/10 effective — INCONCLUSIVE (data, not code)
- **Warmup**: EXPIRED 2026-04-15 — THIN_LINKAGE now requires matched >= 10
- **THIN_LINKAGE**: matched=2 (need 10) — BLOCKING; 8 more live round-trips required
- **MSSA-RL**: NOW ONLINE — white_noise hard gate removed (warn-only); policy_support=43-47 adequate
- **GARCH**: Universal EWMA fallback on all tickers (high-vol regime, volatility_ratio 15-22, persistence 0.92-0.99) — working as designed; 1.5x CI inflation active
- **PnL**: 40 round-trips, +$620.01, 40% WR, profit factor 1.73, avg hold 1.1 days
- **Integrity**: ALL PASSED (0 violations)
- **Last commit**: 29806ac (GARCH convergence_ok grid loop isolation + seen_windows dedup fix)
- **Test count**: 2371 passed, 0 failed, 14 xfailed (fast lane, 2026-04-15)
- **Phase**: Post-DCR Hardening — quant gate forward-looking, MSSA-RL online, GARCH convergence isolated

## Gate Summary

| Gate | Status | Key Metric |
|------|--------|------------|
| ci_integrity_gate | PASS | 40 round-trips, ALL checks passed |
| check_quant_validation_health | PASS | 0.0% fail rate |
| production_audit_gate | PASS (semantics=PASS) | KEEP decision; warmup expired |
| THIN_LINKAGE | **FAIL** | matched=2/10 — blocking post-warmup; need 8 more round-trips |
| MSSA-RL | ONLINE | policy_status=ready on all tickers (white_noise warn-only) |

## Active Cron Jobs (OpenClaw)

| Job | Schedule | Last Status |
|-----|----------|-------------|
| [P0] PnL Integrity Audit | Every 4h | ALL PASSED |
| [P0] Production Gate Check | Daily 7 AM | PASS (semantics=PASS) |
| [P1] Live Trading Cycles — NYSE Hours (AAPL,MSFT,GS) | 14:30 + 17:00 UTC Mon-Fri | Active |
| [P1] Directional Classifier Health | Daily 8:45 AM | Running |
| [P2] GARCH Unit-Root Guard | Weekly Mon 9 AM | Running |
| System Health Check | Every 6h | Running |
| Weekly Session Cleanup | Sunday 3 AM | Silent |

## THIN_LINKAGE Critical Path (warmup EXPIRED 2026-04-15)

| Check | Current | Required | Status |
|-------|---------|----------|--------|
| matched | 2 | >= 10 | **FAIL — need 8 more closed round-trips** |
| matched/eligible ratio | 100% | >= 80% | PASS |
| eligible | 2 | > 0 | PASS |

**Root cause resolved (2026-04-15 funnel fixes):** 4 bugs were blocking ALL live trades (weight_coverage FAIL, AAPL 80bps threshold, deprecated DA gate, SYNTHETIC_ONLY raw getenv). All fixed. Next live cycles will produce genuine trade opportunities.

**What still needs data:**
1. 8 more live closed round-trips linked to audit windows
2. MSFT market data stale to 2026-03-02 — clear `data/raw/MSFT*.parquet` before next run
3. GS OOS metrics absent — run `python scripts/run_etl_pipeline.py --tickers GS` once

## Confirmed Fixes — 2026-04-15 Session

| Bug | Severity | File:Line | Commit |
|-----|----------|-----------|--------|
| GARCH convergence_ok poisoned by rejected grid candidates | HIGH | `garch.py:302` | 29806ac |
| seen_windows all-None dedup collapses audit files without dataset section | MEDIUM | `forecaster.py:2394` | 29806ac |
| MSSA-RL white_noise hard gate blocks all forecasts in high-vol regimes | HIGH | `mssa_rl.py:773` | 48fc989 |
| strict_preselection_min_effective_audits fallback 1 vs 3 | MEDIUM | `forecaster.py:2610` | 48fc989 |
| omega_ratio/payoff_asymmetry from backward-looking daily bar returns | CRITICAL | `time_series_signal_generator.py` | 5e55b2b |
| production_eval/ RMSE files displaced by production/ mtime | HIGH | `check_forecast_audits.py` | 136a193 |
| SYNTHETIC_ONLY raw os.getenv() truthy on "0" | HIGH | `data_source_manager.py` | 5e55b2b |

## Confirmed Architectural Gaps (not bugs, design gaps)

| Gap | Severity | Description | Action |
|-----|----------|-------------|--------|
| Preselection gate dead in live mode | MEDIUM | `_audit_history_stats` reads `production/`; RMSE data lives in `production_eval/`; effective_n=0 always in live mode — gate defaults to KEEP | Low priority; gate is conservative-safe |
| EWMA lambda=0.94 hardcoded | LOW | No per-ticker/regime adaptation | P4 backlog |
| GS OOS metrics absent | MEDIUM | 180 audit files scanned, no GS/horizon=30 eval_metrics | Run ETL for GS once |

## Quant Gate — Forward-Looking Signal Metrics (2026-04-15)

omega_ratio, omega_robustness_score, and payoff_asymmetry are now computed from the signal's
own trade parameters (entry_price, target_price, stop_loss, confidence) using a Bernoulli
synthetic distribution — NOT from 365-day historical daily bar returns.

| Metric | Old source | New source |
|--------|-----------|-----------|
| omega_ratio | 365d market returns vs 28% NGN annual hurdle | Bernoulli(n=120): conf*upside / (1-conf)*stop, tau=0 |
| payoff_asymmetry | historical avg_win / avg_loss from daily bars | upside_pct / stop_pct (forward R:R) |
| omega_robustness_score | same historical returns | omega_robustness_summary on Bernoulli distribution |

Threshold changes: `min_payoff_asymmetry` 1.25 → 0.60 (now correct scale for forward R:R, not historical ratio).

## Heuristic Distortion Status (DCR audit 2026-04-05, verified clean 2026-04-13)

| Fix | File | Status |
|-----|------|--------|
| Missing-baseline bypass | `check_forecast_audits.py:1502-1505` | FIXED (returns None, excludes window) |
| diagnostics_score pessimistic 0.5→0.0 | `time_series_signal_generator.py` | FIXED |
| GARCH EWMA variance floor 1e-12→1e-6 | `garch.py:172, 205, 642` | FIXED |
| MSSA-RL neutral-on-low-support | `mssa_rl.py` | FIXED |
| OOS scan cap removed | `forecaster.py:2491` | FIXED |
| SAMoSSA bump block removed | `ensemble.py` | FIXED |
| SNR fallback 0.5→0.0 | `time_series_signal_generator.py` | FIXED |
| RMSE-rank silent-disable warning added | `ensemble.py` | FIXED |
| EWMA fallback convergence_ok=False | `garch.py` | FIXED |

## Recent Commits (last 10)

```
29806ac fix(garch): isolate convergence_ok to selected model; fix seen_windows dedup
48fc989 fix(mssa-rl): white_noise gate warn-only; fix preselection min_effective fallback
5e55b2b feat(quant-gate): forward-looking signal metrics + verifiable gate logging
136a193 fix(audit): production_eval/ files always win RMSE dedup over production/ files
5dc3264 fix(audit): funnel dedup guard + clarify ensemble-missing gate semantics
69be06b feat(anti-omega): harden barbell acceptance against 4 Omega failure modes
b4157ee feat(barbell): full barbell objective hardening — anti-omega failure modes
42085d0 feat(integrity): barbell-objective canonical metrics in integrity report
dd17efd docs(readme): update to Phase 11 / 2026-04-11 repo truth
521c37e feat(live-funnel): lot-aware close linkage + gate hardening
```
