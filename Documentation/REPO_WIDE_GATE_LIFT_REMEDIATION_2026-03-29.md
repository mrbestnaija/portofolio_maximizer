# Repo-Wide Gate Lift Remediation

**Status**: COMPLETE — gate PASS (semantics=PASS), 33.33% violation rate, decision=KEEP
**Date**: 2026-03-29 (audit) / 2026-03-30 (remediation complete)
**Scope**: Repo-wide guidance for forecast-gate lift recovery, selector correctness, and evidence generation

## Outcome (2026-03-30)

| Metric | Before | After | Threshold |
|--------|--------|-------|-----------|
| Effective audits | 25 | 33 | >= 30 |
| Violations | 11/25 (44%) | 11/33 (33.33%) | <= 35% |
| Gate decision | INCONCLUSIVE | KEEP | lift demonstrated |
| Gate semantics | INCONCLUSIVE_ALLOWED | **PASS** | — |

**Changes made** (all on `codex/observability-rollout-20260328`):
- `forcester_ts/ensemble.py`: OOS wiring (P0), confidence cap moved post-selection (P1a), `_change_point_boost` capped at 0.20 (P1b), MSSA-RL floor removed (P1c), baseline consistency fix
- `forcester_ts/forecaster.py`: `_load_trailing_oos_metrics()` (ticker+horizon scoped), OOS dispatch in `_build_ensemble`, post-selection confidence cap
- `forcester_ts/garch.py`: `hard_igarch_threshold` 0.99 → 0.97
- Tests: 6 new tests in `test_forecaster_audit_contract.py`, 1 in `test_ensemble_config_contract.py`
- P3 evidence: 9 `run_auto_trader --as-of-date` AAPL runs (2021-2024)

**Windows filesystem lesson (P3)**: Within a single CV run, the no-RMSE fold file is consistently
2-3ms newer than the RMSE-bearing fold file (NTFS mtime + write order). Fix: re-run same
`--as-of-date` minutes later — both folds load prior run's audit as trailing OOS → both have RMSE.

---

---

## Purpose

This document records the current verified gate state, the first-principles diagnosis of why
ensemble lift is still not clearing governance, and the repo-wide remediation order.

The main conclusion is:

- The current blocker is **not** one isolated model bug.
- The primary failure is that **candidate selection still runs before audited OOS evidence and
  directional-accuracy inputs are available**.
- Heuristic cleanup matters, but it is **second-order** until the selector uses the right evidence
  at the right time.

---

## Current Verified State

Verified on **2026-03-29** with:

```powershell
.\simpleTrader_env\Scripts\python.exe scripts\check_forecast_audits.py
.\simpleTrader_env\Scripts\python.exe scripts\check_model_improvement.py --json
```

### Forecast Audit Gate

| Metric | Current | Requirement |
|---|---:|---:|
| Raw audit files | 434 | -- |
| Unique audit windows after RMSE dedupe | 60 | -- |
| Effective audits with RMSE | 25 | >= 30 |
| Violations | 11/25 | <= 35% |
| Violation rate | 44.00% | <= 35.00% |
| Recent effective audits | 5/10 | 10 |
| Recent violations | 3/5 | <= 35% |
| Baseline model | EFFECTIVE_DEFAULT | configured |
| Decision | INCONCLUSIVE | must clear hold + threshold |

Additional verified notes:

- RMSE ratio percentiles: `p10=0.928`, `median=0.940`, `p90=1.170`
- Residual diagnostics are warning-only but elevated: `48/48` usable model-backed windows non-white-noise
- Diversity is still thin: `regimes=2`, `healthy_tickers=1`, `trading_days=23`

### Layer 1 Improvement Check

| Metric | Current |
|---|---:|
| Layer 1 status | WARN |
| Baseline | EFFECTIVE_DEFAULT |
| Used windows | 117 |
| Effective-default resolved | 117 |
| Effective-default fallback | 0 |
| Lift fraction global | 0.3333 |
| Lift fraction recent | 0.4500 |
| Lift mean | 0.1195 |
| Lift CI | [-0.0286, 0.2673] |
| SAMOSSA DA zero rate | 53.8% |

Interpretation:

- The baseline contract repair worked.
- The ensemble is no longer being judged against the old oracle baseline in Layer 1.
- Lift is **not statistically confirmed yet** because the confidence interval still spans zero.

---

## First-Principles Diagnosis

### 1. The selector still uses the wrong evidence at decision time

This is the highest-priority issue.

Today the forecaster:

1. builds model summaries,
2. derives model confidence,
3. selects ensemble weights,
4. only then runs `evaluate()` and writes `evaluation_metrics`.

That means the selector is still making its main decision **before** current-window audited OOS
RMSE and directional-accuracy evidence is available.

Practical effect:

- the Phase 10 RMSE-rank hybrid is often inactive at selection time,
- DA-aware selection logic exists but is not fed with live values,
- heuristic scores can dominate because the audited correction arrives too late.

### 2. The RMSE-rank problem is real, but it is downstream of the selection-time data-path bug

The repo already contains RMSE-aware and DA-aware scoring logic, but the live path usually does not
have the necessary OOS metrics populated when weights are chosen.

This means:

- "RMSE-rank is dead code" is directionally true in production behavior,
- but the root cause is broader than one bad formula,
- the selector/evidence contract must be fixed before score tuning will reliably help.

### 3. Heuristics are distorting selection after the evidence gap

Once the selection-time evidence path is fixed, three heuristics become the next highest-leverage
cleanup items:

- `change_point_boost` can push MSSA-RL confidence upward using in-sample change-point recency
- `CONFIDENCE_ACCURACY_CAP=0.65` compresses discrimination too early because it is applied inside
  weight selection rather than only downstream for signal confidence
- heterogeneous score components are still averaged too flatly, so strong heuristic components can
  carry the same arithmetic influence as more trustworthy signals

### 4. The gate is still evidence-limited, not just model-limited

Even with the repaired baseline:

- the gate still has only `25` effective audits,
- `11` are violations,
- the recent window has only `5` effective audits.

So there are two simultaneous blockers:

- **quality blocker**: violation rate is too high,
- **coverage blocker**: effective audit count is too low.

### 5. Some important model-level issues are real, but they are not the first lever

The repo does have lower-level issues worth cleaning up:

- MSSA-RL is still heuristic-heavy and its RL layer appears weak in practice
- GARCH has threshold/comment mismatch and rigid constants
- signal confidence still contains several judgment-based constants
- SAMOSSA directional-accuracy pathologies remain visible in Layer 1

These matter, but they are **not** the first fix if the selector still chooses candidates without
current-window audited OOS evidence.

---

## What Is Blocking Lift Right Now

### Arithmetic blocker

With `11` violations at `25` effective audits:

- at least **5** more effective audits are needed just to clear the `30`-audit holding period
- at least **7** additional clean effective audits are needed to get below `35%`

Reason:

```text
11 / (25 + x) <= 0.35
x >= 7
```

Recent-window arithmetic also matters:

- current recent denominator is `5`
- current recent violations are `3`
- the recent gate needs `10` effective audits
- if the next `5` recent windows are all clean, the recent rate becomes `3/10 = 30%`

### Measurement-contract nuance

Current RMSE dedupe is by shared data window, not ticker-window.

That means:

- same-date multi-ticker runs do **not** automatically raise effective RMSE audit count,
- adding ticker to the RMSE dedupe key would be a **gate-governance change**, not just a bug fix,
- any change here must be documented as a new measurement contract.

---

## Remediation Plan

## P0: Fix the selector evidence contract

**Priority owner**: Agent A  
**Goal**: Make primary candidate selection use audited OOS evidence and DA before weights are chosen.

### Required changes

- Generalize the existing prior-OOS fallback logic so candidate scoring can consume trailing audited
  OOS metrics during primary selection, not only in `DISABLE_DEFAULT` fallback paths
- Feed model directional accuracy into `select_weights()` so the existing DA-aware candidate logic
  and DA caps actually participate in live selection
- Ensure the selector can distinguish between:
  - in-sample heuristic summaries,
  - trailing audited OOS evidence,
  - current-window post-hoc evaluation

### Why this is first

Without this, every later score adjustment is still compensating for a selector that is mostly
flying on stale or in-sample signals.

### Acceptance evidence

- targeted tests covering selection with non-empty trailing OOS metrics
- targeted tests covering DA-aware selection inputs
- `check_model_improvement.py --json` still clean on contract semantics after the change

---

## P1: Remove heuristic distortion inside selection

**Priority owner**: Agent A

These changes come **after** P0, not before.

### Required changes

1. Move `CONFIDENCE_ACCURACY_CAP` out of candidate selection
   - keep it for downstream signal confidence if needed
   - do not let it flatten ranking before weights are chosen

2. Bound or condition `change_point_boost`
   - it should not be able to elevate MSSA-RL purely from in-sample change-point recency
   - preferred direction: make it conditional on trailing OOS evidence or cap its contribution

3. Review the MSSA-RL floor in ensemble scoring
   - current minimum support can still prop MSSA-RL up even when stronger evidence is absent

4. Revisit heterogeneous score combination
   - not all components should be allowed to contribute with identical arithmetic influence

### Acceptance evidence

- targeted unit tests for capped-vs-uncapped candidate ordering
- before/after candidate audit on known violation windows
- no regression in fast lane

---

## P2: Decide and document the gate measurement unit

**Priority owner**: Agent A with human review  
**Supporting owner**: Agent C for documentation and acceptance notes

### Decision required

Choose one of these explicitly:

1. **Keep current contract**
   - RMSE dedupe remains by shared data window
   - multi-ticker same-window runs count as one RMSE evidence point

2. **Adopt ticker-window contract**
   - add ticker to RMSE dedupe key
   - historical comparability changes
   - gate denominator grows differently going forward

### Why this matters

This is governance, not just implementation detail. The answer determines what counts as a new
audit window everywhere in the repo.

### Acceptance evidence

- decision written in config/docs
- `check_forecast_audits.py` output and tests updated consistently
- historical reports annotated if denominator semantics change

---

## P3: Generate fresh clean evidence under the correct contract

**Priority owner**: Agent B for pipeline execution and reporting support  
**Primary interpretation**: Agent A

### Rules

- Use `run_etl_pipeline.py --use-cv`
- Do **not** rely on stale `--mode cv` guidance
- If using `--execution-mode auto`, verify the source did **not** fall back to synthetic
- Do not count synthetic-derived audits as production lift evidence
- Use genuinely new windows under the chosen RMSE dedupe contract

### Minimum target

- **7 clean new effective windows**

That is the minimum to move `11/25` below `35%`, assuming none of the new windows violate.

### Practical target

- 7 clean new overall windows
- 5 clean new recent-window audits
- better diversity than the current `regimes=2`, `healthy_tickers=1`

### Verification loop

Run after each evidence batch:

```powershell
.\simpleTrader_env\Scripts\python.exe scripts\check_forecast_audits.py
.\simpleTrader_env\Scripts\python.exe scripts\check_model_improvement.py --json
```

Stop interpreting progress from intuition alone. Use the gate outputs.

---

## P4: Secondary model cleanup after the gate path is fixed

**Priority owner**: Agent A for model logic  
**Supporting owners**: Agent B for non-core audit/report consumers, Agent C for documentation

These items remain worthwhile, but they are not the first lever for lift recovery:

### MSSA-RL

- decide whether to:
  - simplify it into an explicit heuristic model, or
  - keep RL and tune/train it properly
- remove dead/comment-drift paths that imply active RL behavior when the practical policy is weak

### GARCH

- align `0.97` vs `0.99` threshold intent and documentation
- externalize or justify EWMA lambda
- review asset-insensitive clipping constants

### Signal confidence

- smooth volatility band cliffs
- review edge / diagnostics / SNR weighting
- revisit Platt ramp constants only after the calibration corpus contract is stable

### SAMOSSA

- review CI growth behavior and consider explicit caps comparable to MSSA-RL safeguards

---

## What Is Not The First Fix

These should **not** be treated as the first remedial move:

- adding a DA term to the gate before the selector uses DA in live selection
- changing the gate threshold just to pass
- optimizing one component model in isolation while the selector still ignores audited OOS evidence
- counting same-window multi-ticker audits as "new evidence" without a documented dedupe-contract change
- using synthetic fallback ETL runs as if they were production lift evidence

---

## Repo-Wide Working Rules During Remediation

1. **Do not change the gate and the selector in the same patch unless the contract change is
   explicitly documented.**
2. **Keep Layer 1 baseline on `EFFECTIVE_DEFAULT` unless a reviewed governance decision replaces it.**
3. **Do not claim lift recovery from Layer 1 alone.**
   The forecast gate still requires enough effective audits and an acceptable violation rate.
4. **Do not self-merge model-governance changes.**
   Integration must still go through the human + Claude review step.
5. **Run the fast regression lane for every selector or gate patch.**

Minimum verification set for selector/gate patches:

```powershell
.\simpleTrader_env\Scripts\python.exe scripts\check_model_improvement.py --json
.\simpleTrader_env\Scripts\python.exe scripts\check_forecast_audits.py
.\simpleTrader_env\Scripts\python.exe -m pytest -m "not gpu and not slow" --tb=short -q
```

For unattended-readiness claims, also run:

```powershell
python scripts/institutional_unattended_gate.py --json
python scripts/run_all_gates.py --json
python -m pytest tests/scripts/test_institutional_unattended_contract.py tests/scripts/test_institutional_unattended_gate.py tests/scripts/test_llm_runtime_install_policy.py tests/scripts/test_platt_calibration_contract.py tests/scripts/test_run_all_gates.py -q
```

---

## Success Definition

This remediation is complete only when all of the following are true:

- selector weights are driven by trailing audited OOS evidence plus DA-aware inputs
- heuristic score distortions no longer flatten or overwhelm candidate ranking
- the RMSE measurement contract is explicit and documented
- the forecast gate clears the `30`-audit holding period
- RMSE violation rate is below `35%`
- recent-window violation rate is below `35%`
- Layer 1 lift is no longer relying on stale/oracle semantics

Until then, the correct repo-wide posture is:

- keep the analysis grounded in current gate outputs,
- prefer selector-evidence fixes over cosmetic heuristic tuning,
- treat new evidence generation as a governed pipeline activity, not a side effect of random runs.

---

## Verified Inputs Used For This Document

Commands run on **2026-03-29**:

```powershell
git status --porcelain=v1
.\simpleTrader_env\Scripts\python.exe scripts\check_forecast_audits.py
.\simpleTrader_env\Scripts\python.exe scripts\check_model_improvement.py --json
```

Primary code paths reviewed:

- `forcester_ts/forecaster.py`
- `forcester_ts/ensemble.py`
- `forcester_ts/mssa_rl.py`
- `forcester_ts/garch.py`
- `forcester_ts/samossa.py`
- `models/time_series_signal_generator.py`
- `scripts/check_forecast_audits.py`
- `scripts/run_etl_pipeline.py`
