# Agent C Experiment Briefs: C1, C2, C4 (Parked)

Status: planned only. Not authorized for implementation or rollout.

This document supersedes `AGENT_C_EXPERIMENT_BRIEFS_C1_C2_C4_2026-03-07.md`.

Global block (verified 2026-03-08):

- `fresh_linkage_included` is still `1`
- `fresh_production_valid_matched` is still `0`
- `production_audit_gate`: FAIL (GATES_FAIL, THIN_LINKAGE, EVIDENCE_HYGIENE_FAIL)
- Phase 3 readiness remains blocked

No experiment below may begin until the global block is cleared.

## Global Entry Criteria

All three experiments remain blocked until:

1. fresh `TRADE` exclusions stay near zero across multiple cycles
2. `fresh_linkage_included > 1`
3. `fresh_production_valid_matched >= 1`
4. `production_audit_gate` passes (includes `profitable=True`)
5. `capital_readiness_check.py` R3 clears (WR>=45%, PF>=1.30)
6. parallel merge caveats are closed (see status column below)

| Caveat | Status |
|---|---|
| Layer 1 lift wording mislabels fully-negative CI as `spans zero` | **RESOLVED 2026-03-08** (Phase 7.40 fixed `n_used` → `n_used_windows`; Layer 1 now hard-FAILs definitively-negative CI) |
| C2 flat-series Hurst numerical stability | **RESOLVED 2026-03-08** (Phase 7.40 landed `hurst=0.5` for constant series in `regime_detector.py`) |
| C2 Hurst flat-series semantic contract explicitly tested and documented | **OPEN** — no test asserts `hurst=0.5` on a constant-price series; no doc names the semantic intent. One line required before C2 may begin. |

Items 1-5 require Monday trading cycles at earliest (2026-03-10).
Item 6 (C2 Hurst semantic doc) may be done at any time; it does not unblock the experiments
by itself — it only satisfies one caveat within the larger block.

## C1 - Directional Consensus Gate

Reference: `Documentation/QUANT_FAIL_RATE_RECOVERY_PLAN_2026-02-19.md`

- Objective:
  - abstain when active TS models split directionally instead of forcing a trade
- Intended implementation surface:
  - `models/time_series_signal_generator.py`
  - `config/quant_success_config.yml`
- Expected benefit:
  - reduce false positives from model disagreement
- Expected cost:
  - lower signal frequency

### Entry Criteria

- global entry criteria satisfied
- at least one full linked evaluation window exists after denominator recovery

### Success Metrics

- improved directional accuracy on the production-valid linked subset
- no degradation in profit factor on the same subset
- no increase in gate failures attributable to reduced evidence volume

### Rollback Condition

- linked directional accuracy does not improve, or
- trade frequency drops enough to worsen denominator accumulation materially, or
- any gate semantics become harder to interpret after the change

## C2 - Hurst Exponent Directional Policy

Reference: `Documentation/QUANT_FAIL_RATE_RECOVERY_PLAN_2026-02-19.md`

- Objective:
  - follow signals in trending regimes and fade them in strongly mean-reverting regimes
- Intended implementation surface:
  - `models/time_series_signal_generator.py`
  - `config/forecasting_config.yml`
- Hard dependency (original):
  - Agent A must make `forcester_ts/regime_detector.py` numerically stable on flat and near-flat inputs
  - Agent A must define the flat-series Hurst contract explicitly

### C2 Dependency Status (2026-03-08)

| Dependency | Status |
|---|---|
| `regime_detector.py` numerically stable on flat inputs | **MET** — Phase 7.40 clamps `tau` to `max(t, 1e-12)` before `np.log(tau)`; constant series no longer produces `-inf` Hurst |
| Flat-series Hurst value is `0.5` (random-walk neutral) | **MET** — Phase 7.40 returns `hurst=0.5` for constant series |
| `hurst=0.5` semantic contract documented and tested | **NOT MET** — no targeted test or docstring states `constant series → hurst=0.5 (random-walk neutral, no trend bias)`. One documentation addition (comment or docstring in `regime_detector.py`) and one assertion-level test are required. |

**Agent A action required before C2 can be marked fully pre-qualified:**
Add a single docstring note to `RegimeDetector.compute_hurst()` (or equivalent) and a
one-assertion test that covers the constant-series path. Agent A does not need to change
any logic — the numerical fix is already in place.

### Entry Criteria

- global entry criteria satisfied
- regime outputs are finite on degenerate inputs (MET)
- Hurst flat-series value is `0.5` (MET)
- `hurst=0.5` semantic contract explicitly tested and documented (NOT MET)
- live Layer 1 / capital-readiness semantics are no longer internally contradictory (RESOLVED 2026-03-08)

### Success Metrics

- improved directional accuracy on linked trades segmented by regime
- no non-finite regime outputs in regression tests
- no new ambiguity in signal direction semantics

### Rollback Condition

- any NaN/inf regime output appears in tests or live diagnostics
- linked subset shows no measurable directional improvement
- policy makes model direction harder to interpret in audits

## C4 - EMA Momentum Pre-Filter

Reference: `Documentation/QUANT_FAIL_RATE_RECOVERY_PLAN_2026-02-19.md`

- Objective:
  - suppress low-confidence signals that fight strong short-term momentum
- Intended implementation surface:
  - `models/time_series_signal_generator.py`
  - `config/quant_success_config.yml`
- Expected benefit:
  - reduce whipsaw entries

### Entry Criteria

- global entry criteria satisfied
- linked subset is large enough to measure "correct direction but negative PnL" before/after

### Success Metrics

- reduced rate of "correct direction but negative PnL" on linked trades
- no increase in stop-loss concentration
- no denominator collapse from over-filtering

### Rollback Condition

- signal suppression starves evidence growth
- linked PnL quality does not improve
- the filter produces hidden abstention semantics that are not clear in telemetry

## Execution Order Once Unblocked

1. C1
2. C2
3. C4

Rationale:

- C1 is the least structurally invasive
- C2 depends on numerical-stability cleanup (met) and semantic documentation (pending)
- C4 should be measured after a clean baseline exists for directional disagreement and regime behavior

## Agent C Guardrail

Agent C may prepare acceptance tests and evaluation templates for these experiments, but
may not request implementation while the global block remains active.

The global block is not cleared until all conditions in the Global Entry Criteria section are
met and verified by running commands — not by inference. Agent C does not self-certify
readiness.
