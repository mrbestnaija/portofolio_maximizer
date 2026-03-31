# Gate Lift: First-Principles Model & Pipeline Audit

**Date**: 2026-03-29
**Branch**: `codex/observability-rollout-20260328`
**Head commit**: `5610ce5`
**Gate state**: `PASS (INCONCLUSIVE_ALLOWED)` — warmup expires 2026-04-15
**Trigger**: RMSE violation rate 44% (11/25 effective audits), need <35% with ≥30 effective audits

---

## Executive Summary

A systematic audit of the ensemble selection pipeline, all model implementations, and the gate
measurement contract revealed a layered set of bugs — from a root data-path ordering failure
through unvalidated heuristics to secondary model stubs. The findings are ordered by gate-lift
impact; the same ordering governs the remedial plan.

**Root cause**: The ensemble selector (`derive_model_confidence` → `select_weights`) fires before
out-of-sample metrics exist. Every "fix" applied since Phase 10 to improve selection quality
(RMSE-rank hybrid, DA-aware candidates) has had zero runtime effect because the data it reads is
never populated at selection time.

---

## Part 1: Gate Measurement Diagnosis

### Current effective audit counts

| Metric | Value | Required |
|--------|-------|---------|
| Raw audit files | 434 | — |
| After RMSE dedupe | 60 unique | — |
| With RMSE data (effective) | 25 | ≥ 30 |
| Violations (ratio > 1.10) | 11 | — |
| Violation rate | 44% | ≤ 35% |
| Recent window (5/10) | 60% | ≤ 35% |

### Why 434 files become 25 effective audits

**Deduplication key** (`check_forecast_audits.py:1471`):
```python
(dataset.start, dataset.end, dataset.length, dataset.forecast_horizon)
```
No ticker in the key. AAPL + MSFT + AMZN on the same date range collapse to one window; the
latest file wins. 434 files → 60 unique data windows → 25 with RMSE data (only ETL pipeline
CV runs populate `evaluation_metrics`; auto-trader runs produce `{}`).

**Arithmetic to clear the gate**:
- Need ≥ 30 effective audits: 5 more ETL CV runs with any outcome
- Need ≤ 35% violation rate: `11 / (25 + x) ≤ 0.35` → x ≥ 7 non-violating new windows

Adding auto-trader cycles adds zero effective audits. Running the same date range for multiple
tickers adds zero effective audits (deduped away). Only new `(start, end, length, horizon)` tuples
from `run_etl_pipeline.py` with a CV step add effective audits.

### Violation pattern

All 11 violations have `primary_model = samossa`. SAMoSSA minimises EVR by SSA construction and
will almost always beat any blend on RMSE in a smooth trending market. The gate asks "does blending
beat what you would have picked anyway?" — with a broken weight selector, the ensemble frequently
picks MSSA-RL-heavy candidates that are worse than SAMoSSA alone.

---

## Part 2: Selection-Time Data-Path Bug (P0 — Root Cause)

### The ordering problem

**`forecaster.py:1987-1988`**:
```python
confidence = derive_model_confidence(summaries)          # line 1987
weights, score = weighter.select_weights(confidence)     # line 1988
```

**`forecaster.py:2391`** (after the forecast loop):
```python
_latest_metrics = evaluation_metrics   # written here — AFTER selection
```

`derive_model_confidence` and `select_weights` fire in the *forecast phase*. `evaluation_metrics`
(OOS RMSE, DA, CI coverage from `evaluate()`) are computed in the *evaluation phase*, which runs
after the forecast loop completes. The selector therefore operates on zero OOS evidence.

### RMSE-rank hybrid is dead code

**`ensemble.py:438-455`** (Phase 10 RMSE-rank addition):
```python
_rmse_values = {
    model: float(m.get("rmse"))
    for model, m in metrics_map.items()
    if m.get("rmse") is not None ...
}
```
`metrics_map` reads `component_summaries["regression_metrics"]`. In every production audit file,
`component_summaries.regression_metrics = {}` (empty dict). Therefore `_rmse_values = {}`,
`_rmse_rank_scores = {}`, and no RMSE-rank component ever enters `_combine_scores`.

This has been true since Phase 10 shipped. The feature was never active.

### DA-aware selection never runs

**`ensemble.py:177-224`**: DA-weighted candidate construction and DA capping are implemented.
**`forecaster.py:1988`**: `select_weights(confidence)` — `model_directional_accuracy` parameter
is never passed. The entire DA path in `select_weights` is dead code at runtime.

### Impact of the ordering bug

Without OOS RMSE-rank and without DA, the only live inputs to `derive_model_confidence` are:
- SAMoSSA: EVR (always ~1.0 by SSA construction)
- MSSA-RL: `baseline_variance` score + `_change_point_boost`
- GARCH: AIC/BIC domain-normalized to [0.28, 0.58]
- SARIMAX: AIC/BIC raw

The `_change_point_boost` for MSSA-RL can reach 1.0 when `recent_change_point_days=0`, which
then wins the quantile ranking over SAMoSSA's EVR=0.95. This is why the AAPL violation shows
`mssa_rl:0.622, samossa:0.170` ensemble weights when SAMoSSA has the lowest RMSE.

---

## Part 3: Heuristic Distortions Inside Selection (P1)

### `_change_point_boost` — unvalidated formula with 1.0 ceiling

**`ensemble.py:590-601`**:
```python
def _change_point_boost(summary):
    ...
    if recent_days <= 7:
        recency = max(0.0, 1 - (recent_days / 7.0))
        boost = 0.2 + 0.6 * recency + 0.2 * min(density * 10.0, 1.0)
        return float(np.clip(boost, 0.0, 1.0))
```
When `recent_change_point_days=0`: `recency=1.0`, `boost=0.2+0.6+0.2=1.0` (maximum).

Constants `0.2`, `0.6`, `0.2`, `7`, `10.0` have no documented derivation and have not been
A/B tested. The boost is based entirely on in-sample change-point detection with no OOS
validation. It enters `_combine_scores` on equal footing with all other components.

**Concrete effect (AAPL violation, `forecast_audit_20260316_055920.json`)**:
```
mssa_rl: baseline_var_score=0.224, change_point_boost=1.0 → combined=0.612
samossa: EVR=0.9999998 → combined=0.950
After quantile normalisation + 0.65 cap: mssa_rl=0.65, samossa=0.65 (tied)
Winner by insertion order: mssa_rl-heavy candidate
Result: weights {mssa_rl:0.62, garch:0.21, samossa:0.17}, ensemble RMSE=13.80 vs samossa RMSE=9.77
```

### `CONFIDENCE_ACCURACY_CAP=0.65` applied inside candidate scoring

**`ensemble.py:786-795`**:
```python
CONFIDENCE_ACCURACY_CAP = 0.65
calibrated_confidence = {
    model: float(min(score, CONFIDENCE_ACCURACY_CAP))
    for model, score in calibrated_confidence.items()
}
```
This cap fires before `select_weights` runs. It was intended to prevent overconfident position
sizing downstream, but it collapses the discrimination between the best and worst models:
- SAMoSSA (RMSE=9.77, raw score=0.95) → capped to 0.65
- MSSA-RL (RMSE=16.53, raw score=0.612) → 0.625

With two models within 4% of each other in confidence, any candidate using both scores
identically. The winner is the first tied candidate in the `scored_candidates` list.

**Fix**: Apply `CONFIDENCE_ACCURACY_CAP` only to the confidence value used for downstream
position sizing — not inside the candidate scoring loop.

### `_combine_scores` — unweighted mean across heterogeneous signal types

**`ensemble.py:457-465`**:
```python
def _combine_scores(*scores):
    valid = [clip(s) for s in scores if s is not None]
    return clip(mean(valid))
```
AIC/BIC, EVR, `baseline_variance`, `change_point_boost`, F-test, and RMSE-rank all enter with
equal weight. A `change_point_boost=1.0` carries the same arithmetic weight as `EVR=0.95`. No
hierarchy or reliability weighting exists between signal types of fundamentally different meaning.

### MSSA-RL hard floor at 0.40

**`ensemble.py:730-734`**:
```python
if mssa_score is not None:
    mssa_score = max(mssa_score, 0.40)
```
Even after removing the boost, this floor ensures MSSA-RL always enters the quantile normaliser
with a score that puts it in the middle tier, regardless of its actual OOS performance.

### GARCH domain normalization — three-parameter formula

**`ensemble.py:657`**:
```python
return float(np.clip(0.42 + 0.16 * float(np.clip(raw_aic_bic_score, 0.0, 1.0)), 0.28, 0.58))
```
Parameters `0.42`, `0.16`, `0.28`, `0.58` were set to put GARCH in a "neutral participation
band" by judgment. The fallback score of `0.28` for IGARCH and `0.45` for converged-but-no-IC
are also judgment values. None have been backtested.

### `_relative_rmse_score` and `_relative_te_score` — inconsistent penalty curves

**`ensemble.py:474`**: RMSE penalty `1.0 / (1.0 + 1.5 * (ratio - 1.0))`
**`ensemble.py:483`**: TE penalty `1.0 / (1.0 + 1.2 * (ratio - 1.0))`

Different scaling constants (1.5 vs 1.2) for analogous metrics with no documented rationale
for the difference.

---

## Part 4: MSSA-RL is Functionally a Stub (P4 after fixing selector)

### Q-table never learns a meaningful policy

Production Q-table values across all 434 audits: range `[-0.025, +0.004]`, indistinguishable
from the `0.0` default for unseen `(state, action)` pairs.

For all 4 observed states, `best_action=1` (neutral, `q_direction_weight=0.0`) always wins.
The RL component has no effect on forecast output.

**`mssa_rl.py:119-121`**: `alpha=0.3, gamma=0.85, epsilon=0.1` are standard RL textbook
defaults. No domain calibration has been performed. With tiny rewards (directional PnL on a
30-step horizon) and 11-entry Q-tables, learning cannot converge meaningfully within any
single fit call.

### Action degeneracy

**`mssa_rl.py:145-146`**: `action_rank_cutoffs={0:0.25, 1:0.90, 2:1.00}`
**Production `rank_by_action`**: `{0:1, 1:2, 2:30}` (most common observed)

Action 0 uses rank 1 (lowest-frequency component). Action 1 uses rank 2. For short series
these are near-identical reconstructions. The agent nominally has 3 choices; it has at most
2 materially different behaviors, and always picks action 1 (neutral) in practice.

### Legacy slope blend is dead code

**`mssa_rl.py:514-516`**:
```python
# Legacy slope-direction signal (retained; blended at 0.5 weight below).
effective_slope = slope + q_direction_weight * abs(slope) * 0.5
```
Since `best_action` is always 1, `q_direction_weight = {0:-1.0, 1:0.0, 2:1.0}[1] = 0.0`.
The legacy term is always zero. The comment accurately describes the intent; the behavior
never fires.

### Design decision required

MSSA-RL should either:
1. Be simplified to a pure heuristic model (remove Q-table, keep reconstruction selection based
   on deterministic variance cutoffs)
2. Be properly trained with a defined reward signal, adequate state space, and offline Q-value
   initialisation before being used as a live signal source

---

## Part 5: GARCH Implementation Issues (P4)

### Persistence threshold code/comment mismatch

**`garch.py:8`** (module docstring): "GJR-GARCH fallback when persistence >= **0.97**"
**`garch.py:316`** (inline comment): "if alpha+beta >= **0.97** the"
**`garch.py:57`** (actual code): `hard_igarch_threshold: float = 0.99`

The code triggers IGARCH at **0.99** but all documentation says **0.97**. Models with
persistence 0.97–0.98 pass as "converged GARCH" when they should trigger the GJR-GARCH →
EWMA fallback chain. This is a silent 2pp gap in a near-unit-root region.

**Fix**: Align code to 0.97 (matching documentation) or document why 0.99 is correct.

### EWMA lambda hardcoded

**`garch.py:532`**: `lam = 0.94` — RiskMetrics 1994 daily lambda. No config path exists.

NVDA annual vol ~58%, AAPL ~27%. The same decay constant produces over-smoothing on NVDA
and under-smoothing on AAPL. The EWMA fallback is already a degraded path; an asset-class
aware lambda would reduce damage when it fires.

### Outlier clip uses arbitrary multiplier

**`garch.py:132`**: `cap = max(p995, med * 10.0)` — the 10× median multiplier clips extreme
returns before GARCH fitting. For high-vol assets in crisis, legitimate observations can
exceed 10× daily median. For stable equities it may be too loose. No asset-class branching.

---

## Part 6: Signal Confidence Heuristics (P4)

### Discrete volatility bands create cliff edges

**`time_series_signal_generator.py:1457-1462`**:
```python
if vol >= 0.60:   vol_factor = 0.60
elif vol >= 0.40: vol_factor = 0.75
elif vol <= 0.15: vol_factor = 1.05
```
A 1bp vol change at 0.40 boundary causes a 25pp confidence drop. This is a step function,
not a soft penalty. Any ticker near a band boundary produces erratic signal confidence across
otherwise similar market conditions.

**Fix**: Replace with a smooth sigmoid or piecewise-linear transition between bands.

### Confidence weights give edge disproportionate influence

**`time_series_signal_generator.py:1468-1472`**:
```python
core = (
    0.20 * diagnostics_score
    + 0.20 * model_agreement
    + 0.20 * snr_score
    + 0.40 * edge_score
)
```
`edge_score = clamp01(net_return / (3 × threshold))`. At 3× threshold, edge_score=1.0,
contributing `core=0.40` even with zero diagnostics, zero agreement, and zero SNR. The 40%
edge weight allows a single factor to drive high confidence on low-quality signals.

### SNR mapping assumes a range that rarely applies

**`time_series_signal_generator.py:1451`**:
```python
snr_score = clamp01((snr - 0.5) / 1.5)   # 0.5 → 0, 2.0 → 1
```
SNR = `E[return] / CI_half_width`. For NVDA with CI half-width 20% and expected return 0.3%,
SNR=0.015 → `snr_score=0`. The entire bottom half of the SNR scale maps to zero. The formula
compresses most real observations into the lowest scores, making SNR a weak discriminator.

### Platt ramp constants are backward-derived

**`time_series_signal_generator.py:2764`**:
```python
ramp_raw_weight = 0.80 - 0.30 * min(1.0, max(0.0, (n - 43) / 57.0))
```
`43` = minimum pairs floor; `57 = 100 - 43` (desired full-weight at n=100). Constants were
derived backward from the desired endpoint, not from empirical calibration of how quickly
Platt becomes trustworthy. The 0.80 starting weight and 0.30 range are judgment values.

The `max_downside=0.15, max_upside=0.10` asymmetry (**`signal_generator.py:2768-2775`**)
assumes Platt primarily corrects downward, which is not supported by the win-rate data.

---

## Part 7: CI Growth Factual Record

**MSSA-RL** (`mssa_rl.py:565`): CI scale capped at `sqrt(max(steps/2, 1.0))` — **bounded**.
**SAMoSSA** (`samossa.py:526`): CI scale `sqrt(step+1)` — **unbounded**.

For SAMoSSA on a 30-step horizon with high `baseline_variance`, CI lower bound can go negative
for a positive-price asset. MSSA-RL has the Phase 10b cap; SAMoSSA does not.

The `0.85 * base + 0.15 * last_recon` blend appears in **MSSA-RL** (`mssa_rl.py:468`) only.
SAMoSSA does not use this formula (earlier session note was incorrect).

---

## Part 8: Gate Measurement Contract

### Current design (intentional)

`check_forecast_audits.py:1471`: RMSE dedupe key = `(start, end, length, forecast_horizon)`
`check_forecast_audits.py:1479`: comment — "latest evidence wins"

This counts each shared data window once. Multi-ticker runs on the same date range do not
increase effective audit count.

### Proposed change (governance decision required)

Adding `ticker` to the dedupe key would allow AAPL, MSFT, AMZN on the same dates to count
as 3 separate windows — immediately multiplying effective audits from existing data.

**This changes the gate contract**: it shifts the measurement from "how often does ensemble
beat best-single on a given data window?" to "how often does it beat per-ticker-window?"
The violation rate would change (likely improve) not because the model improved but because
the denominator changed. This must be an explicit governance decision with documentation,
not a mechanical fix.

---

## Remedial Plan (Ordered by Gate-Lift Impact)

### P0 — Fix selector evidence contract
**Target**: `forecaster.py:1987-1988`, `ensemble.py:427-455`, `ensemble.py:177-224`

1. Thread trailing audited OOS metrics (from prior `evaluation_metrics` in audit files)
   into `derive_model_confidence()` at selection time, using the existing previous-window
   OOS fallback stubs at `forecaster.py:1504` and `forecaster.py:2101`
2. Pass `model_directional_accuracy` (from prior audit files) to `select_weights()` so
   `ensemble.py:177-224` DA-aware candidate logic actually runs
3. Populate `_rmse_rank_scores` from those trailing OOS values instead of in-sample
   `component_summaries["regression_metrics"]`

**Expected effect**: Candidate selection will use the model that demonstrably performed
best OOS over prior windows, not the model with the best EVR or most recent change point.

### P1 — Remove heuristic distortions
**Target**: `ensemble.py:590-601`, `ensemble.py:786-795`, `ensemble.py:730-734`

1. Move `CONFIDENCE_ACCURACY_CAP=0.65` out of `select_weights` — apply only to the
   confidence value returned to signal generation
2. Cap `_change_point_boost` contribution inside `_combine_scores` at ≤0.20 weighting,
   or make it conditional on trailing OOS evidence rather than always firing on in-sample
   change points
3. Remove or reduce MSSA-RL hard floor after boost cleanup — let OOS evidence determine rank

### P2 — Decide gate measurement unit
**Target**: `check_forecast_audits.py:1471`

Document explicitly: keep "shared window" contract (no change) or add ticker and document
as a gate-contract update with a dated comment in `forecaster_monitoring.yml`.

### P3 — Generate fresh clean evidence
**Target**: `scripts/run_etl_pipeline.py`

```bash
# Target crisis/high-vol windows where GARCH/MSSA-RL add real value
python scripts/run_etl_pipeline.py --tickers AAPL --start 2020-01-01 --end 2020-09-01 --execution-mode auto
python scripts/run_etl_pipeline.py --tickers MSFT --start 2022-01-01 --end 2022-12-31 --execution-mode auto
python scripts/run_etl_pipeline.py --tickers AMZN --start 2022-06-01 --end 2023-06-01 --execution-mode auto
python scripts/run_etl_pipeline.py --tickers NVDA --start 2021-01-01 --end 2021-12-31 --execution-mode auto
```

Each run must produce a **new** `(start, end, length, horizon)` tuple to count as a new
effective audit. Same date range re-runs replace the existing entry.

Do P0+P1 first — without fixing the selector, new audits will be scored by the wrong
inputs and violations may not improve.

### P4 — Secondary model cleanup
**After gate path is fixed and violation rate is improving:**

1. **MSSA-RL**: decide — simplify to deterministic heuristic (remove Q-table) or define
   proper offline training pipeline. Current Q-table is non-functional.
2. **GARCH**: align persistence threshold `0.97`/`0.99`; add `ewma_lambda` to config;
   review `10×median` clip for high-vol assets
3. **Signal confidence**: replace discrete vol bands with smooth transitions; review
   `0.40` edge weight; fix SNR mapping to cover realistic range
4. **SAMoSSA**: add CI cap analogous to MSSA-RL `sqrt(horizon/2)` — prevents negative
   lower bounds on positive-price assets

---

## Key Files Referenced

| File | Lines | Issue |
|------|-------|-------|
| `forcester_ts/forecaster.py` | 1987-1988 | Selector fires before OOS metrics exist |
| `forcester_ts/forecaster.py` | 2391 | `evaluation_metrics` written here (after selection) |
| `forcester_ts/forecaster.py` | 1504, 2101 | Existing OOS fallback stubs to generalize |
| `forcester_ts/ensemble.py` | 427-455 | RMSE-rank reads empty `component_summaries` |
| `forcester_ts/ensemble.py` | 177-224 | DA-aware path — never runs (DA never passed) |
| `forcester_ts/ensemble.py` | 590-601 | `_change_point_boost` — 1.0 ceiling, unvalidated |
| `forcester_ts/ensemble.py` | 786-795 | Confidence cap inside candidate scoring |
| `forcester_ts/ensemble.py` | 730-734 | MSSA-RL hard floor 0.40 |
| `forcester_ts/ensemble.py` | 457-465 | `_combine_scores` unweighted mean |
| `forcester_ts/ensemble.py` | 657 | GARCH domain norm 3-param formula |
| `forcester_ts/mssa_rl.py` | 119-121 | RL textbook defaults, never trained |
| `forcester_ts/mssa_rl.py` | 145-146 | Action degeneracy |
| `forcester_ts/mssa_rl.py` | 477 | Q-table always picks action 1 |
| `forcester_ts/mssa_rl.py` | 514-516 | Legacy slope blend — dead code |
| `forcester_ts/mssa_rl.py` | 565 | CI cap at `sqrt(horizon/2)` (bounded) |
| `forcester_ts/garch.py` | 8, 57, 316 | Persistence threshold 0.99 vs 0.97 mismatch |
| `forcester_ts/garch.py` | 532 | EWMA `lam=0.94` hardcoded |
| `forcester_ts/samossa.py` | 526 | CI growth uncapped `sqrt(step+1)` |
| `models/time_series_signal_generator.py` | 1457-1462 | Discrete vol bands |
| `models/time_series_signal_generator.py` | 1468-1472 | 0.40 edge weight |
| `models/time_series_signal_generator.py` | 2764 | Platt ramp constants |
| `scripts/check_forecast_audits.py` | 1471, 1479 | RMSE dedupe key (no ticker) |

---

## Bottom Line

> The selector reads the wrong evidence. Fix that first.
> After that, fix the confidence cap and change-point boost.
> Only then do the RL stub, GARCH constants, and signal-confidence constants become the right next levers.

Adding data volume without P0 will score new audits by the same broken inputs.
The violation rate may not improve regardless of how many ETL runs are added.
