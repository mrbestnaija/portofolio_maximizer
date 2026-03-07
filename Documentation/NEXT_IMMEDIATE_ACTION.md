# Next Immediate Action

**Last Updated**: 2026-03-07
**Status**: Denominator recovery only - no readiness or linkage-improvement claims
**Protocol**: [RESEARCH_EXPERIMENT_PROTOCOL.md](RESEARCH_EXPERIMENT_PROTOCOL.md)

---

## Autonomous Decision (2026-03-07)

Applying the current evidence hierarchy:

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

---

## Immediate Actions

### Action 1 - Keep the fresh cohort strictly TRADE-only

Use only the corrected watcher outputs:

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

`NON_TRADE_CONTEXT` rows are diagnostics only. They must never be reintroduced into the fresh TRADE denominator.

### Action 2 - Let the watcher accumulate daily evidence, not intraday noise

- Daily bars make sub-daily polling mostly noise.
- The watcher now sleeps for `86400` seconds and skips weekends by default.
- Because today is Saturday, March 7, 2026, no new production watcher cycles should be expected until Monday, March 9, 2026.

### Action 3 - Keep dashboard startup wiring active after reboot

```bash
python scripts/windows_dashboard_manager.py ensure --status-json logs/dashboard_manager_status.json
```

This keeps the dashboard bridge, local HTTP server, and live denominator watcher connected from one entry point.

### Action 4 - Correct the reporting-layer architectural mismatches before trusting dashboard/readiness surfaces

Current verified review findings on `master`:

- `capital_readiness_check.py` still treats definitively negative lift CI as advisory only
- `dashboard_db_bridge.py` still trusts stale `portfolio_positions` rows and can replay raw executions without production filters
- unavailable performance metrics can still surface as `0.0`
- recent trade events still omit `exit_reason`
- `RegimeDetector` still emits non-finite values on flat inputs

These are audit/reporting hardening tasks, not strategy changes. They should be fixed before any new readiness claim based on dashboard or capital-readiness surfaces.

---

## What To Watch Next

Only three watcher signals matter right now:

1. Fresh `TRADE`-context exclusion counts
   - target: near zero across multiple cycles
2. Fresh linkage denominator growth
   - target: `fresh_linkage_included > 1`, then stable growth toward `5-10`
3. Fresh production-valid matched rows
   - target: `fresh_production_valid_matched >= 1`

Anything else is secondary until those three conditions are met.

---

## Capital Readiness Snapshot (2026-03-07)

| Gate | Status | Detail |
|------|--------|--------|
| R1 adversarial | PASS | telemetry contract and TCON checks remain active |
| R2 gate artifact | FAIL | `run_all_gates.py` currently fails on `production_audit_gate` |
| R3 trade quality | FAIL | current metrics are below threshold (`win_rate=40.0%`, `profit_factor=0.80`) |
| R4 calibration | PASS | last verified Brier path remains below hard-fail threshold |
| R5 lift CI | WARNING | negative CI remains advisory only in `capital_readiness_check.py` |
| R6 lifecycle | PASS | lifecycle integrity remains cleared |

---

## Observability Commands

```bash
# Watcher status without waiting for the next trading day
python scripts/run_live_denominator_overnight.py --tickers AAPL,AMZN,GOOG,GS,JPM,META,MSFT,NVDA,TSLA,V --cycles 1 --sleep-seconds 0 --dry-run

# Dashboard stack + watcher startup
python scripts/windows_dashboard_manager.py ensure --status-json logs/dashboard_manager_status.json

# Gate snapshot
python scripts/run_all_gates.py --json
```
