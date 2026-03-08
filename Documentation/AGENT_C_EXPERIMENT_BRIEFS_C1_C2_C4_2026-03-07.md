# Agent C Experiment Briefs: C1, C2, C4 (Parked)

Status: planned only. Not authorized for implementation or rollout.

Global block:

- `fresh_linkage_included` is still `1`
- `fresh_production_valid_matched` is still `0`
- Phase 3 readiness remains blocked

No experiment below may begin until the global block is cleared.

## Global Entry Criteria

All three experiments remain blocked until:

1. fresh `TRADE` exclusions stay near zero across multiple cycles
2. `fresh_linkage_included > 1`
3. `fresh_production_valid_matched >= 1`
4. Agent A confirms gate/reporting semantics are trustworthy enough to evaluate downstream effects
5. parallel merge caveats are closed:
   - Layer 1 lift wording no longer mislabels fully negative CI as `spans zero`
   - C2 regime semantics are explicitly tested, not just finite

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
- Hard dependency:
  - Agent A must first make `forcester_ts/regime_detector.py` numerically stable on flat and near-flat inputs
  - Agent A must also define the flat-series Hurst contract explicitly (`0.5` neutral vs another intentional value)

### Entry Criteria

- global entry criteria satisfied
- regime outputs are finite on degenerate inputs
- Hurst semantics are explicitly tested and documented
- live Layer 1 / capital-readiness semantics are no longer internally contradictory

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
- C2 depends on numerical-stability cleanup
- C4 should be measured after a clean baseline exists for directional disagreement and regime behavior

## Agent C Guardrail

Agent C may prepare acceptance tests and evaluation templates for these experiments, but may not request implementation while the global block remains active.
