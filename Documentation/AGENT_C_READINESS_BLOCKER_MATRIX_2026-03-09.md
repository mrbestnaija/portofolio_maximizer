# Agent C Readiness Blocker Matrix (2026-03-09)

Doc Type: blocker_matrix
Authority: temporary Agent C blocker tracker; not a gate-semantics source of truth
Owner: Agent C
Last Verified: 2026-03-09
Supersedes: `Documentation/AGENT_C_READINESS_BLOCKER_MATRIX_2026-03-08.md`
Expires When: superseded by a newer dated blocker matrix or merged into canonical runtime status docs

Purpose: keep Agent C on measurement, sequencing, and evidence only.

This document does not authorize strategy changes, experiment execution, or
readiness claims.

## Changes from 2026-03-08 Matrix

1. **Eligibility gate semantics: OPEN → FIXED** (commit fc6b921)
   - `apply_ticker_eligibility_gates.py` is now fail-closed on missing/corrupt evidence
   - `status="FAIL"`, `reason="missing_eligibility_evidence"`, exit code 1
   - 5 tests passing; 2 new tests cover corrupt input + CLI exit code

2. **EXP-R5-001 post-redesign canary: complete**
   - Fresh pipeline run (AAPL 2022-01-01 to 2024-01-01, synthetic mode) produced one new audit
   - RC1–RC4 observability fields confirmed populated in `artifacts.residual_experiment`
   - See Section 7 for full canary snapshot

3. **INT-02 adversarial finding: CONFIRMED (pre-existing, not introduced this session)**
   - `test_critical_confirmed_findings_exist_in_production_codebase` now fails on INT-02
   - This is a pre-existing finding — `DUPLICATE_CLOSE detection requires entry_trade_id (bypassed when NULL)`
   - Not caused by any change in this session

4. **CLOSE_WITHOUT_ENTRY_LINK violation (trade 255): ROOT CAUSE CONFIRMED — whitelist justified**
   - Trade 255: TSLA BUY `is_close=1`, 2026-03-06, live, `run_id=20260309_063607`, `realized_pnl=-$629.77`
   - **Created 2026-03-09 06:36** — run just before canary; NOT a historical artifact
   - The dry-run proposed linking to trade 114 (TSLA synthetic SELL, 2021-06-14) — **rejected** (different epoch, synthetic vs live, multi-year gap)
   - **Root cause (confirmed 2026-03-09 via DB trace)**:
     - Trade 247 (`id=247`, `is_synthetic=1`, `execution_mode=synthetic`, `run_id=20260304_045850`) is the true opener: TSLA SELL, 2026-03-04, `price=82.13343940353356`, `position_after=-2.0`
     - Trade 255 `entry_price=82.13343940353356` — **exact price match to trade 247**; PTE populated `entry_price` from `portfolio_state` which held the synthetic position
     - **Mechanism**: `portfolio_state` table has no `is_synthetic` column. The synthetic run (`20260304_045850`) wrote TSLA=-2 to `portfolio_state`. The next live run (`20260309_063607`) with `--resume` read TSLA=-2 as a live position and closed it as live trade 255.
     - **This is a `portfolio_state` mode-contamination bug, not an entry_trade_id wiring bug**. The close-path wiring is correct — trade 255 has no live opener because the opener was synthetic.
   - Whitelist is **justified**: opener exists (id=247) but is `is_synthetic=1`; linking live close to synthetic open is semantically wrong; opener is irrecoverable for legitimate live attribution.
   - Do NOT apply the dry-run repair (id=114 or id=247 — both synthetic, both wrong)
   - **35 other synthetic orphan SELL opens** exist in `trade_executions` with no close leg (same contamination class — see Section 9)
   - **Whitelisted in `.env`, `bash/run_20_audit_sprint.sh`, `bash/production_cron.sh`** (INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS=66,75,255)
   - `INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS=66,75,255 pnl_integrity_enforcer` reports **ALL PASSED** (verified 2026-03-09)

## Current State

### 1. Runtime Health

- overall status: `degraded` (unchanged)
- production gate still: `GATES_FAIL`, `THIN_LINKAGE`, `EVIDENCE_HYGIENE_FAIL`, `matched=0/1`
- evidence-bound, not code-fixable

### 2. Capital Readiness (unchanged from 2026-03-08)

| Gate | Status | Root Cause |
|---|---|---|
| R1 adversarial | PASS | 0 confirmed CRITICAL/HIGH findings (INT-02 is MEDIUM) |
| R2 gate artifact | PASS | current gate artifact is fresh |
| R3 trade quality | FAIL | WR=40% < 45%, PF=0.80 < 1.30 |
| R4 calibration | PASS | Brier below threshold |
| R5 lift CI | FAIL | CI=[-0.1139,-0.0572] definitively negative across 162 windows |
| R6 lifecycle | PASS | no high lifecycle violations |

### 3. Layer 1 Lift (unchanged from 2026-03-08)

- `status = FAIL`
- `lift_ci_low = -0.1139`, `lift_ci_high = -0.0572`
- `lift_win_fraction = 3.1%`

### 4. Dashboard Truth (unchanged from 2026-03-08)

- `payload_schema_version = 2`
- `positions_stale = true`
- `positions_source = trade_executions_fallback_stale`
- running bridge matches current schema

### 5. Eligibility Gate Semantics

**FIXED** — commit fc6b921 (2026-03-09)

Before:
- missing input → `status="WARN"`, exit 0 (fail-open)

After:
- missing input → `status="FAIL"`, `reason="missing_eligibility_evidence"`, exit 1 (fail-closed)
- corrupt JSON → same FAIL path
- invalid tickers dict → same FAIL path
- clean input → `status="PASS"` (unchanged)

Test coverage: 5 tests all pass.

### 6. Watcher Lane (unchanged from 2026-03-08)

- `status = WAITING`
- `fresh_trade_rows = 1`
- `fresh_linkage_included = 1`
- `fresh_production_valid_matched = 0`
- next evidence window: live trading session (market hours)

### 7. EXP-R5-001 Post-Redesign Canary

Source: `python scripts/run_etl_pipeline.py --tickers AAPL --start 2022-01-01 --end 2024-01-01 --execution-mode synthetic`
Audit: `logs/forecast_audits/forecast_audit_20260309_065611.json`

Post-redesign artifact snapshot (from `artifacts.residual_experiment`):

```json
{
  "phi_hat": 0.99,
  "intercept_hat": 0.233,
  "n_train_residuals": 52,
  "oos_n_used": 52,
  "skip_reason": null,
  "residual_status": "active",
  "residual_active": true,
  "residual_mean": 12.42,
  "residual_std": 0.95,
  "n_corrected": 30
}
```

Canary interpretation:
- `phi_hat = 0.99` — strong AR(1), above `_MIN_PHI=0.15`; phi gate did not fire
- `skip_reason = null` — fit succeeded as expected
- `intercept_hat = 0.233` — near-zero (RC1 demeaning effective; pre-redesign intercept was +1.10 with DC bias)
- `oos_n_used = 52` — RC3 proportional OOS (was fixed at max(20, horizon)=20 before)
- Corrections consistently positive (~+10 to +14 over 30 steps) — reflects genuine OOS anchor under-forecast on AAPL 2022-2024; requires Phase 3 realized-price comparison to evaluate

Log evidence during run:
```
[EXP-R5-001] ResidualModel fitted: phi=0.9317 intercept=-0.1009 n_train=122 oos_n=122
[EXP-R5-001] ResidualModel fitted: phi=0.9057 intercept=0.1255 n_train=45 oos_n=45
[EXP-R5-001] ResidualModel fitted: phi=0.9900 intercept=0.2334 n_train=52 oos_n=52
```

Note: multiple folds run (5 CV builds); three distinct fits shown above.

Canary verdict: **PASS** — RC1–RC4 producing valid observability fields, no gate fires on strong autocorrelation, intercept near-zero confirming DC-bias removal.

**Next step for EXP-R5-001**: Phase 3 re-accumulation. Run 10+ additional pipeline windows with different `--end` dates to get fresh realized metrics under the redesigned model. Compute `rmse_ratio` and `corr(ε,ε_hat)` via `scripts/residual_experiment_phase3_backfill.py` when realized prices are available.

### 9. portfolio_state Synthetic Contamination Bug (NEW — 2026-03-09)

**Bug identified** during trade 255 root-cause trace.

**Schema**: `portfolio_state` has no `is_synthetic` column. Columns: `id, ticker, shares, entry_price, entry_timestamp, stop_loss, target_price, max_holding_days, holding_bars, entry_bar_timestamp, last_bar_timestamp, updated_at`.

**Mechanism**:
1. Synthetic run writes a short position to `portfolio_state` (e.g., TSLA=-2, entry_price=$82.13)
2. Synthetic run ends without closing the position
3. Next live run with `--resume` reads `portfolio_state` → sees TSLA=-2 → treats it as a live position
4. Live run closes it as a live BUY (no matching live SELL opener in `trade_executions`)
5. Result: live close (is_synthetic=0) with no live opener → CLOSE_WITHOUT_ENTRY_LINK violation

**Scope**: 35 synthetic SELL opens with no close leg currently in `trade_executions` — each represents a potential future contamination if any of those tickers are still in `portfolio_state` when a live `--resume` run happens.

**Current `portfolio_state`** (2026-03-09 post-run):
- AAPL: 4 shares (live, from run 20260309_063607)
- NVDA: 2 shares (live, from run 20260309_063607)
- These appear to be legitimate live positions, not synthetic contamination.

**Required fix** (not yet implemented):
- Add `is_synthetic INTEGER DEFAULT 0` to `portfolio_state` schema
- PTE writes `is_synthetic` when saving positions
- On `--resume` in live mode: skip positions where `is_synthetic=1`, log a warning

**Risk without fix**: Any future synthetic run that does not close all positions will contaminate the next live `--resume` run. This has already happened once (trade 255). The fix is low-risk and should be added before the next controlled live cycle.

### 8. CI State

- Targeted residual suite (`tests/forcester_ts/test_residual_ensemble.py`): PASS (1733 total at RC1-RC4 commit)
- Fast lane: 1060 passed (not-slow mark), 1 pre-existing failure (INT-02 adversarial runner)
- INT-02 is pre-existing; see section above

## Blocker Matrix

| Surface | Current result | Blocking owner | Unblock condition |
|---|---|---|---|
| Runtime health | `degraded` | Trading cycles + gate owners | `production_gate` passes |
| Production audit gate | FAIL | Trading cycles | clear `GATES_FAIL`, `THIN_LINKAGE`, `EVIDENCE_HYGIENE_FAIL` |
| Capital readiness | FAIL | Trading cycles + ensemble architecture | R3 clears and R5 no longer definitively negative |
| Dashboard truth | PASS | Agent B | keep served payload aligned with bridge schema |
| Eligibility gate semantics | **FIXED** | Done (commit fc6b921) | fail-closed on missing evidence — complete |
| Fresh TRADE denominator | `linkage_included=1`, `matched=0` | Trading cycles | `fresh_linkage_included > 1` and `fresh_production_valid_matched >= 1` |
| EXP-R5-001 fast-lane | PASS | Shared | targeted residual suite passing |
| Repo-wide fast lane | 1 pre-existing FAIL (INT-02) | Shared | INT-02 duplicate-close detection needs entry_trade_id fix |
| EXP-R5-001 canary | **PASS** | Done (2026-03-09) | RC1-RC4 producing valid artifacts with phi_hat, skip_reason=null |
| EXP-R5-001 Phase 3 re-accumulation | NOT STARTED | Agent B | run 10+ new windows post-redesign; compute realized rmse_ratio + corr |
| portfolio_state synthetic contamination | OPEN BUG | Agent A | add `is_synthetic` column; skip synthetic positions on live --resume |
| Experiment execution | blocked | Agent C protocol | all preconditions below satisfied |

## Gate-Lifting Order (corrected 2026-03-09)

Verified sequence for production audit gate advancement:

1. **Whitelist trade 255** in `INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS` — DONE
   — Root cause confirmed: synthetic opener id=247 (`is_synthetic=1`, run 20260304_045850) contaminated `portfolio_state`; live resume read it and produced a live close with no live opener
   — Opener is irrecoverable for live attribution; whitelist is semantically justified
   — Do NOT apply any repair (id=114 or id=247 — both synthetic, both wrong)

2. **Fix `portfolio_state` synthetic contamination bug before next live `--resume` cycle**
   — Add `is_synthetic INTEGER DEFAULT 0` to `portfolio_state`
   — PTE writes `is_synthetic` when saving state
   — On `--resume` in live mode: skip synthetic positions with WARNING log
   — Without this fix, next synthetic run ending with open positions will repeat trade 255

3. **Confirm live close path wiring** — VERIFIED
   — PTE entry_trade_id auto-population: correct (lines 1354, 1364-1366)
   — DB save function: correct (line 2155, `entry_trade_id INTEGER` column)
   — No code changes required for the close path itself

3. **Synthetic cycles: plumbing check only** (not gate-lifting)
   — Synthetic rows excluded from `production_closed_trades` view (DB-level filter: `is_synthetic=0`)
   — Gate artifact: no `production_audit_only` field (validator correction 2026-03-09); audit_dir is flat `logs/forecast_audits`
   — Synthetic useful for: linkage plumbing, close-leg wiring, residual canary
   — Synthetic NOT useful for: clearing `THIN_LINKAGE`, `EVIDENCE_HYGIENE_FAIL`

4. **Live non-synthetic cycle** (market hours, next trading session)
   — Each live cycle is a data point; re-check gate after each one
   — Gate needs: `fresh_linkage_included > 1` and `fresh_production_valid_matched >= 1`

5. **Gate re-check after each cycle** (not batched)

## Agent C Operating Rules

- Do not start experiments.
- Do not recommend strategy changes.
- Do not interpret `WAITING` as progress.
- Report only verified outputs from commands and artifacts.

## Promotion Rule For Agent C

Agent C may move from blocker tracking to experiment-ready planning only when all
of the following are true:

1. fresh `TRADE` exclusions stay near zero across multiple cycles
2. `fresh_linkage_included > 1`
3. at least one fresh production-valid matched row appears
4. `production_audit_gate` passes
5. `capital_readiness_check.py` clears R3
6. live dashboard/runtime truth blockers are explicitly resolved

Items 1-5 are not yet satisfied. Item 6 is currently resolved.
Item 7 (eligibility gate fail-closed): **NOW RESOLVED**.
