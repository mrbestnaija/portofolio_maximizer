# Domain-Calibrated Remediation Plan — 2026-04-05

> Historical evidence note: objective semantics are now governed by
> `Documentation/REPO_WIDE_MATRIX_FIRST_REMEDIATION_2026-04-08.md`.
> Where this note frames Sharpe, win rate, or other symmetric metrics as primary, treat that wording
> as historical context rather than the current repo-wide default.

## Purpose

This document captures the findings and implementation plan from the adversarial audit
conducted on 2026-04-05, covering structural bypasses, heuristic distortions, architectural
stubs, threshold dodges, and mismatched wiring discovered across the forecasting and gating
stack.

The gate is currently `PASS (semantics=PASS)` — but the audit distinguishes:
- **"Warmup exemption covers the gap"** — passes only because of warmup leniency
- **"Genuinely passes on its own"** — would pass without warmup

Warmup expires **2026-04-15**. This plan must be actioned before that date.

---

## Confirmed State (2026-04-05, commit fd4eb68)

| Metric | Value | Threshold | Warmup Covering? |
|--------|-------|-----------|-----------------|
| Violation rate | 31.43% (11/35 effective) | ≤35% | No — passes on merit |
| Lift fraction | 57.14% (20/35) | ≥25% | No — passes on merit |
| Effective audits | 35 | ≥30 holding period | No — passes on merit |
| Recent window | 4/10 effective | need 10 | INCONCLUSIVE (data) |
| THIN_LINKAGE | 1/309 matched (0.32%) | ≥10 matched, ≥80% ratio | **YES — warmup masks** |
| Residuals non-WN | 100% | ≤75% (warn_only=true) | **YES — enforcement removed** |
| PnL | +$620.01, PF=1.73, 40% WR | profitable | No |
| Trading days | 10/21 required | ≥21 for full proof | Partially — proof still valid |

**What breaks on 2026-04-15 (warmup expiry):**
1. THIN_LINKAGE hard-fails: needs 10 matched outcomes, has 1
2. Recent window remains INCONCLUSIVE until 6 more audits accumulate
3. Profitability proof loses `is_proof_valid=True` until 11 more trading days

---

## Root Cause: The Linkage Problem Is a Funnel Problem

Confirmed with live data from the trade DB and production audit directory:

- **260 production audit files** contain `ts_signal_id` (forecasts generated since Phase 10c)
- **42 closed trades** contain `ts_signal_id` in the DB
- **Intersection: 1** — only 1 audit file matches a closed trade

This is not a data-accumulation problem. The 259 unmatched audits represent forecasts
that were **blocked by signal routing** (confidence < 0.55, SNR < 1.5, min_return < 20bps)
and never became executed trades. The 41 closed trades without audits were executed
before Phase 10c's audit-routing fix (pre-2026-03-16).

**LINKAGE is actually measuring "forecast → trade conversion rate"** — currently 0.4%.
The warmup exemption hides that 99.6% of forecasts are filtered out. Whether those
filters are correctly calibrated or destroying alpha is unknown — the funnel has never
been audited.

---

## Cascade of Silent Failures

```
OOS metrics missing in CV runs
  → RMSE-rank disabled → uniform confidence
    → select_weights insertion-order dependent
      → suboptimal model selected → RMSE regresses
        → violation_rate rises (fail_on_violation_during_holding_period=false)
          → gate stays INCONCLUSIVE → warmup covers it
            → linkage stays at 1/309 → warmup covers it
              → System appears PASS but is in silent-failure regime
```

---

## Confirmed Bugs and Bypasses

### Category 1: Threshold Dodges (config changes made to pass, not derived)

| Setting | Current Value | Original | Why It Distorts |
|---------|-------------|---------|----------------|
| `max_non_white_noise_rate` | 0.75 | 0.25 | 100% non-WN currently; limit is meaningless |
| `residual_diagnostics_rate_warn_only` | true | false | Removes all enforcement from residual check |
| `fail_on_violation_during_holding_period` | false | true | Converts hard FAILs to INCONCLUSIVE silently |
| `strict_preselection_max_rmse_ratio` | 1.1 | 1.0 | Allows 10% RMSE regression at forecast time |

### Category 2: Short-Circuit Bypasses (code paths that produce wrong result on error)

| Bug | File:Line | Effect |
|-----|-----------|--------|
| Missing `baseline_rmse` → `violation=False` | `check_forecast_audits.py:1194-1200` | Window counted as non-violation; deflates violation_rate |
| GARCH EWMA fallback doesn't reset `convergence_ok` | `garch.py:356-365` | forecaster.py CI inflation (1.5x) may not fire when it should |
| Linkage vacuously passes when `eligible==0` | `production_audit_gate.py:1403` | 100+ not_due windows could bypass linkage check forever |

### Category 3: Stubs and Silent Defaults (pretend to compute but return constants)

| Bug | File:Line | Effect |
|-----|-----------|--------|
| `diagnostics_score` defaults to 0.5 (neutral) when missing | `time_series_signal_generator.py:767` | System pretends diagnostics are neutral; should be pessimistic |
| GARCH EWMA variance floor at 1e-12 | `garch.py:589` | Near-zero variance → CI collapses → SNR inflates → spurious confidence |
| MSSA-RL `policy_support` never checked during action selection | `mssa_rl.py:516-517` | Low-support states return same action as well-trained states |

### Category 4: Mismatched Wiring (data computed but not used)

| Bug | File:Line | Effect |
|-----|-----------|--------|
| `terminal_directional_accuracy` computed but gate only uses RMSE | `metrics.py:109` vs `check_forecast_audits.py:1194` | Directionally correct but RMSE-heavy forecasts penalised |
| `_oos_da` extracted but only used in `select_weights`, not confidence scoring | `forecaster.py:2023-2042` | DA-aware path runs but doesn't influence output confidence |
| RMSE dedupe key lacks ticker; outcome dedupe includes ticker | `check_forecast_audits.py:1471-1494` | Multi-ticker audits create denominator divergence between RMSE and outcome counts |
| `_load_trailing_oos_metrics()` returns `{}` in all CV runs | `forecaster.py:2394-2491` | RMSE-rank always disabled in CV; gate measures ensemble that was never scored |

### Category 5: Architectural Stubs

| Stub | File | Reality |
|------|------|---------|
| MSSA-RL has no `_update_q_table()` | `mssa_rl.py` | Offline policy, never learns; `_select_action()` returns static best_action_by_state |
| `_calibrate_confidence()` silently bypasses Platt when n < 43 | `time_series_signal_generator.py:2642-2707` | Returns raw confidence unchanged; no flag set; consumers can't distinguish |

---

## Remediation Plan

### Phase 1 — Fix Structural Bypasses (Priority: before warmup expiry)

**P1-A: Missing baseline_rmse → exclude window (not non-violation)**
- File: `scripts/check_forecast_audits.py` line 1194
- Change: `return None` instead of `return AuditCheckResult(..., violation=False)`
- Callers filter `None`: `results = [r for r in raw_results if r is not None]`
- Test: assert window with missing baseline_rmse absent from effective_count, not counted as non-violation
- Risk: effective_count may drop; verify still ≥ 30 (holding period floor)

**P1-B: Funnel audit logging**
- File: `scripts/run_auto_trader.py` in `_execute_signal()` / signal-blocking paths
- Add: structured JSONL log (`logs/funnel_audit.jsonl`) recording every blocked signal:
  `{ts_signal_id, ticker, reason, confidence, snr, expected_return, terminal_da_from_audit}`
- Purpose: observability only; basis for future threshold calibration after ≥10 cycles
- Test: assert blocked signals produce funnel_audit.jsonl entries with correct fields
- Do NOT change any threshold until funnel audit shows `blocked_terminal_DA > 0.52`

**P1-C: diagnostics_score pessimistic fallback**
- File: `models/time_series_signal_generator.py` line 767
- Change: `diagnostics.get("score", 0.5)` → `diagnostics.get("score")` with `None` → log WARNING + use 0.0
- Test: conf(missing_score) < conf(score=0.5) — penalty for missing diagnostics
- Risk: low; may reduce confidence on sparse forecasts (correct behaviour)

**P1-D: GARCH EWMA variance floor**
- File: `forcester_ts/garch.py` line 589
- Change: `max(ewma_var, 1e-12)` → `max(ewma_var, 1e-6)` (0.1% daily vol floor)
- Constant: `MIN_EWMA_VARIANCE = 1e-6`
- Test: constant series → GARCH CI width ≥ `2 * sqrt(MIN_EWMA_VARIANCE) * sqrt(horizon)`
- Risk: negligible; 1e-6 is below any real equity daily variance

**P1-E: Residual diagnostics enforcement by model type**
- Files: `config/forecaster_monitoring.yml` + `scripts/check_forecast_audits.py`
- Change in config: add `residual_diagnostics_model_type_exemptions: [samossa, mssa_rl]`
- Change in code: skip non-WN count for windows where primary model is in exemption list
- Set `residual_diagnostics_rate_warn_only: false` once exemptions are wired
- Test: SAMoSSA-primary window excluded from non-WN count; GARCH-primary window included
- Risk: violation_rate for residuals may rise (correct — the system was genuinely non-compliant)

---

### Phase 2 — Domain Calibration (after Phase 1 is clean)

**P2-A: Terminal DA as RMSE co-gate**
- File: `scripts/check_forecast_audits.py` around `_check_rmse_violation()`
- Prerequisite: verify `terminal_directional_accuracy` is present in audit JSONs' `evaluation_metrics`
- Rule: if `ensemble_rmse > (1+tolerance)*baseline_rmse` AND `ensemble_terminal_DA >= baseline_terminal_DA - 0.05`, clear the violation (direction compensates for magnitude)
- Backtest: run against 35 existing audits; report how many violations change; verify result is positive
- Test: RMSE=1.12x, ensemble_tda=0.60, baseline_tda=0.58 → NOT a violation

**P2-B: CV OOS proxy from fold metrics**
- File: `forcester_ts/forecaster.py` in `_load_trailing_oos_metrics()`
- Change: when no disk audit found AND `self._cv_fold_metrics` is populated, return fold metrics as OOS proxy
- Effect: RMSE-rank re-enabled in CV; ensemble scoring becomes meaningful during cross-validation
- Test: CV context with no pre-existing audits → `derive_model_confidence` receives non-empty `oos_metrics`

---

### Phase 3 — Calibration from Realized Trades (after Phase 2)

**P3-A: Confidence → win_rate calibration script**
- New file: `scripts/calibrate_confidence_thresholds.py`
- Input: `production_closed_trades` view (40 trades)
- Output: `logs/confidence_calibration.json` plus immutable history under
  `logs/confidence_calibration_history/` — bins of `(conf_low, conf_high) → win_rate, n`
- Use: basis for future signal routing threshold changes; do NOT change thresholds yet
- Test: with 40 trades, produces JSON with non-decreasing win_rate per confidence bin

**P3-B: MSSA-RL policy_support gate**
- File: `forcester_ts/mssa_rl.py` in `_resolve_active_action()`
- Change: if `policy_support[current_state] < min_support_for_action` → return neutral action (1)
- Test: low-support state returns 1 regardless of `best_action_by_state` value

---

### Phase 4 — Architectural Cleanup

**P4-A: Linkage vacuous-pass hardening**
- File: `scripts/production_audit_gate.py` line 1403
- Change: vacuous pass only if total audit windows < 5 (genuine early accumulation)
- If ≥5 windows exist but all are not_due → FAIL with reason `LINKAGE_ALL_NOT_DUE`
- Test: 100 not_due windows → FAIL

**P4-B: Ticker in RMSE dedupe key**
- File: `scripts/check_forecast_audits.py` in `_rmse_dedupe_key_from_audit()`
- Change: add ticker to key → `(ticker, start, end, length, horizon)`
- Effect: effective count rises from ~35 to ~130; violation_rate recalculated
- Backtest: run on existing audit files; report new counts before merging
- This resolves the previously deferred governance decision (Item 5 of P4 plan)
- Test: same-date, different-ticker audits each counted separately in effective_count

---

## Anti-Patterns (Explicitly Forbidden)

1. **Do NOT lower `min_lift_fraction` or raise `max_violation_rate`** if Phase 1 fixes cause
   violation_rate to rise. A higher violation rate after honest counting is the correct signal —
   it means the system was worse than the gate showed.

2. **Do NOT change confidence or SNR thresholds** (in `signal_routing_config.yml`) until
   Phase 1-B's funnel audit has run for ≥10 cycles and proves blocked forecasts had
   `terminal_DA > 0.52`. Threshold changes without evidence = threshold dodging.

3. **Do NOT add more warmup exemptions** to cover Phase 4-B's contract change. The current
   warmup covers THIN_LINKAGE. Additional exemptions compound the bypass pattern.

4. **Do NOT implement P2-A** until confirming that `terminal_directional_accuracy` is
   actually written to audit JSON files. Adding a co-gate on a field that is absent is
   a dead code path that silently clears violations.

---

## Verification Sequence

```bash
# Phase 1 verification
pytest tests/forcester_ts/test_garch_guardrails.py -v                    # P1-D
pytest tests/forcester_ts/test_residual_diagnostics.py -v                # P1-E
pytest tests/models/test_time_series_signal_generator.py -k "diag" -v    # P1-C

# Backtest P1-A impact on audit effective count
./simpleTrader_env/Scripts/python.exe scripts/check_forecast_audits.py \
    | grep -E "Effective audits|Violation rate|usable"

# Phase 2 backtest (run AFTER P1 is merged)
./simpleTrader_env/Scripts/python.exe scripts/check_forecast_audits.py --json \
    | python -c "import sys,json; d=json.load(sys.stdin); print('effective:', d.get('effective_audits'), 'violation_rate:', d.get('violation_rate'))"

# Full gate verification
./simpleTrader_env/Scripts/python.exe scripts/run_all_gates.py --json \
    | python -c "import sys,json; d=json.load(sys.stdin); print('overall_passed:', d.get('overall_passed'))"
```

---

## Files to Modify

| File | Phase | Change |
|------|-------|--------|
| `scripts/check_forecast_audits.py:1194` | P1-A | Missing baseline → exclude (None), not non-violation |
| `scripts/check_forecast_audits.py:1471` | P4-B | Add ticker to RMSE dedupe key |
| `scripts/run_auto_trader.py` | P1-B | Funnel audit JSONL logging |
| `models/time_series_signal_generator.py:767` | P1-C | diagnostics_score → 0.0 pessimistic fallback |
| `forcester_ts/garch.py:589` | P1-D | EWMA variance floor → 1e-6 |
| `forcester_ts/mssa_rl.py` | P3-B | policy_support gate in _resolve_active_action |
| `forcester_ts/forecaster.py:2491` | P2-B | CV OOS proxy from fold metrics |
| `config/forecaster_monitoring.yml` | P1-E | Model-type exemptions for residual diagnostics |
| `scripts/production_audit_gate.py:1403` | P4-A | Linkage vacuous-pass hardening |
| `scripts/check_forecast_audits.py` | P2-A | Terminal DA co-gate (after verifying field presence) |
| `scripts/calibrate_confidence_thresholds.py` | P3-A | New script (confidence→win_rate lookup) |

---

## Related Documentation

- `Documentation/GATE_LIFT_FIRST_PRINCIPLES_AUDIT_20260329.md` — Phase 10c OOS wiring root cause
- `Documentation/REPO_WIDE_GATE_LIFT_REMEDIATION_2026-03-29.md` — Phase 10c remediation
- `Documentation/ADVERSARIAL_AUDIT_20260216.md` — Original adversarial findings
- `Documentation/PHASE_7.14_GATE_RECALIBRATION.md` — Config sanitization (Phase 7.14)
- `Documentation/MSSA_RL_OFFLINE_REMEDIATION_20260404.md` — MSSA-RL offline policy notes
- `Documentation/HEARTBEAT.md` — Current system status

---

**Last updated**: 2026-04-05
**Author**: Adversarial audit + domain calibration review
**Status**: PLAN — implementation pending
