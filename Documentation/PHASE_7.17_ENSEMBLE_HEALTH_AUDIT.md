# Phase 7.17: Ensemble Health Audit & Adaptive Weighting

**Status**: COMPLETE
**Regression**: 1332 passed, 1 skipped, 28 deselected, 7 xfailed
**Date completed**: 2026-03-01

---

## Motivation

Overnight refresh exposed three structural problems killing ensemble lift:

| Problem | Evidence |
|---------|----------|
| SAMOSSA DA=0 anomaly | SAMOSSA wins best-single RMSE on 6/11 windows (55%) but with `directional_accuracy=0.000`. Near-flat forecasts win on magnitude but contribute zero directional signal, diluting mssa_rl in the blend. |
| Static candidate weights | 10 fixed candidates in `forecasting_config.yml` never adapt to recent model performance. No mechanism to down-weight chronically underperforming models. |
| 14pp exit-quality gap | Forecast DA 55.2% but trade win-rate 41%. Root cause unknown (stop-too-tight? holding-too-short?). |
| 4 duplicate audit windows | Identical metrics for same ticker/date inflate the denominator for lift calculation. |

---

## Architecture

```
logs/forecast_audits/*.json
        |
        v
scripts/dedupe_audit_windows.py     (Step 2.8: report duplicates, dry-run nightly)
        |
        v
scripts/ensemble_health_audit.py    (Step 2.9: per-model stats + Shapley + adaptive weights)
        |
        +-> logs/ensemble_health/ensemble_health_{YYYYMMDD}.md   (markdown report)
        +-> config/forecasting_config.yml  (writes adaptive_candidate_weights section)
        |
        v
forcester_ts/ensemble.py            (DA penalty cap + read adaptive weights)
        |
        v
scripts/exit_quality_audit.py       (diagnose forecast-DA -> trade-WR gap)
        |
        v
scripts/run_overnight_refresh.py    (Steps 2.8 + 2.9 integrated)
```

---

## New Scripts

### `scripts/dedupe_audit_windows.py`

Fingerprints each audit JSON by SHA1 of `(ticker, window_start, window_end, n_observations, horizon)`.
Keeps the newest file per fingerprint hash.

```bash
# Dry-run (report only)
python scripts/dedupe_audit_windows.py --audit-dir logs/forecast_audits

# Apply (delete older duplicates)
python scripts/dedupe_audit_windows.py --audit-dir logs/forecast_audits --apply
```

Exit codes: `0` = no duplicates, `1` = duplicates found (non-blocking CI).

### `scripts/ensemble_health_audit.py`

Loads all forecast audit JSONs, computes per-model diagnostics, Shapley attribution, and adaptive
candidate weights. Writes markdown report and optionally updates `forecasting_config.yml`.

```bash
python scripts/ensemble_health_audit.py \
    [--audit-dir logs/forecast_audits] \
    [--recent-n 20] \
    [--write-config] \
    [--write-report]
```

**Key outputs**:
- `logs/ensemble_health/ensemble_health_{DATE}.md` — markdown report with model table, Shapley, adaptive candidates
- `config/forecasting_config.yml` (optional) — `adaptive_candidate_weights` section
- `GOLDEN_METRICS` structured JSON log line for CI/monitoring

**GOLDEN_METRICS alert thresholds**:
- `lift_fraction < 0.05` → ensemble under-performing (WARN)
- `samossa_da_zero_pct > 0.40` → SAMOSSA DA=0 anomaly active (WARN)

### `scripts/exit_quality_audit.py`

Reads `production_closed_trades` (is_close=1, non-diagnostic, non-synthetic) and decomposes
win/loss by exit_reason to explain the forecast-DA → trade win-rate gap.

```bash
python scripts/exit_quality_audit.py [--db data/portfolio_maximizer.db] [--tail-n 100]
```

**Interpretation rules**:
- `stop_too_tight`: stop-loss exits > 40% of trades
- `holding_too_short`: time-exits > 40% AND time-exit win-rate < 45%
- `mix`: both populations need inspection

Exit code: always 0 (non-blocking diagnostic).

---

## `forcester_ts/ensemble.py` Changes

### Change A: DA-penalty cap (`_apply_da_cap`)

New module-level helper that caps and proportionally redistributes weights for models
with chronic near-zero directional accuracy.

```python
def _apply_da_cap(
    weights: Dict[str, float],
    da_scores: Dict[str, float],
    da_floor: float,      # from EnsembleConfig.da_floor (default 0.10)
    da_weight_cap: float, # from EnsembleConfig.da_weight_cap (default 0.10)
) -> Dict[str, float]:
```

**CONTRACT**:
- If ALL models are DA-penalized (DA < da_floor), returns `{}` → caller skips candidate.
- Otherwise returns normalized dict where every penalized model has weight ≤ da_weight_cap.
- Budget freed from capping redistributes ONLY to non-penalized models (DA ≥ da_floor).
- Runtime invariant guard: logs ERROR and self-corrects if output sum ≠ 1.0.

Applied inside `select_weights()` after `_apply_minimum_component_weight()`:
```python
if model_directional_accuracy:
    normalized = _apply_da_cap(
        normalized, model_directional_accuracy,
        self.config.da_floor, self.config.da_weight_cap
    )
    if not normalized:
        continue  # all models DA-penalized → skip this candidate
```

### Change B: Adaptive candidates prepended

`EnsembleConfig` gains three new fields (all config-driven via `forecasting_config.yml`):
```python
adaptive_candidate_weights: list[dict] = field(default_factory=list)
da_floor: float = 0.10
da_weight_cap: float = 0.10
```

`select_weights()` prepends adaptive candidates before static ones:
```python
candidate_list = list(adaptive_candidate_weights) + list(candidate_weights)
```

---

## `config/forecasting_config.yml` Changes

```yaml
ensemble:
  da_floor: 0.10       # DA below this triggers weight penalty (SAMOSSA DA=0 fix)
  da_weight_cap: 0.10  # Max allowed weight for DA-penalized models after normalization

  # Written by scripts/ensemble_health_audit.py --write-config:
  adaptive_candidate_weights:
    computed_at: "YYYY-MM-DD"
    recent_n: 20
    lambda_decay: 1.0
    da_floor: 0.10
    da_cap_weight: 0.10
    weights:
      - {garch: 0.xx, samossa: 0.xx, mssa_rl: 0.xx}  # primary adaptive
      - {top_model: 0.70, second_model: 0.30}          # top-2 hedge
      - {top_model: 1.0}                                # pure winner
```

---

## Overnight Refresh Integration

Steps 2.8 and 2.9 added between the audit-gate bootstrap (Step 2.7) and final health check (Step 3):

```python
# STEP 2.8: Deduplicate audit windows (dry-run — report only)
rc = py("scripts/dedupe_audit_windows.py", allow_fail=True)
if rc == 1:
    log("[WARN] Duplicate audit windows detected — run --apply to remove")

# STEP 2.9: Ensemble health audit + adaptive weight update
rc = py("scripts/ensemble_health_audit.py",
        "--write-config", "--write-report", "--recent-n", "20",
        allow_fail=True)
if rc != 0:
    log("[WARN] Ensemble health audit returned non-zero — check logs/ensemble_health/")
```

Both steps are `allow_fail=True` (non-blocking diagnostics, not gatekeeping).

---

## Adaptive Weight Algorithm (`compute_adaptive_weights`)

```
1. Use last `recent_n` windows (by window_id timestamp order)
2. For each model: mean_rmse = mean(rmse) over recent windows
3. rmse_median = median across models
4. raw_weight = exp(-lambda_decay * mean_rmse / rmse_median)
5. Hard zero: if mean_rmse > 1.2 * rmse_median → weight = 0.0
6. DA penalty: if mean_da < da_floor → cap at da_cap_weight
   All-DA-zero fallback: if ALL models have DA < da_floor, skip DA penalty
   and weight by RMSE-only. Record degraded_da_fallback=True.
7. Normalize to sum=1.0
8. Diversity guard: if top model weight > 0.90 after normalization,
   clamp to 0.90 and redistribute excess proportionally.
   Log diversity_clamped=True.
9. Output 3 candidates:
   [primary adaptive, top-2 hedge (0.70/0.30), pure winner (1.0)]
```

**Key redistribution invariant** (enforced by Hypothesis property tests):
Budget freed from capping must flow ONLY to non-penalized models (DA ≥ da_floor).
Sending budget to penalized-but-below-cap models allows them to grow above da_cap_weight.

---

## Bugs Found and Fixed by Hypothesis Property Tests

Three bugs were found by `hypothesis==6.151.9` during the hardening pass:

### Bug 1: `_apply_da_cap()` — redistribution to penalized-but-below-cap models

**Falsifying example**:
```python
weights={'garch': 0.4, 'samossa': 0.4, 'mssa_rl': 0.2}
da_scores={'garch': 0.0, 'samossa': 0.0, 'mssa_rl': 0.0}
da_floor=0.25, da_weight_cap=0.25
```

**Old behavior**: garch (0.4 > 0.25) and samossa (0.4 > 0.25) were capped;
mssa_rl (0.2 ≤ 0.25) was classified as "uncapped" and received freed budget,
growing from 0.20 → 0.50 (> da_weight_cap=0.25).

**Fix**: identify ALL penalized models (DA < da_floor) first; redistribute ONLY to
non-penalized models (DA ≥ da_floor); if none exist, return `{}`.

### Bug 2: `compute_adaptive_weights()` post-normalization cap — same structural flaw

Same root cause as Bug 1 in `ensemble_health_audit.py`. Fixed identically:
`uncapped` replaced with `non_penalized` (models with DA ≥ da_floor).

### Bug 3: `round(adaptive[m], 4)` — midpoint rounds up past da_cap_weight

**Falsifying example**:
`da_cap_weight=0.109375` (= 7/64, exact binary fraction) with `round(0.109375, 4) = 0.1094`.
Python rounds the midpoint (digit 5) up, so `0.1094 > da_cap_weight=0.109375`.

**Fix**: changed primary candidate construction from `round(..., 4)` to `round(..., 6)`.
`0.109375` with 6 decimal places is exactly representable; no rounding violation.

---

## Test Coverage

| File | Tests Added |
|------|-------------|
| `tests/scripts/test_dedupe_audit_windows.py` | 10 (new file) |
| `tests/scripts/test_ensemble_health_audit.py` | 33 (new file, includes 5 Hypothesis property tests) |
| `tests/scripts/test_exit_quality_audit.py` | 21 (new file) |
| `tests/forcester_ts/test_ensemble_config_contract.py` | +10 (6 Phase 7.17 unit tests + 3 Hypothesis property tests + 1 regression) |

**Hypothesis property tests** (`hypothesis==6.151.9`):
- `TestApplyDaCapProperties` (3 tests × 300 examples each):
  - `test_result_is_empty_or_normalized`
  - `test_penalized_models_respect_cap` ← found Bug 1
  - `test_all_penalized_returns_empty`
- `TestComputeAdaptiveWeightsProperties` (5 tests × 150-250 examples each):
  - `test_candidates_always_normalized`
  - `test_no_negative_weights`
  - `test_all_da_zero_triggers_fallback`
  - `test_primary_candidate_respects_diversity_guard`
  - `test_penalized_models_respect_cap` ← found Bugs 2 + 3

---

## Repo-Wide Architecture Audit (2026-03-01)

A systematic audit was performed across all Phase 7.17 files and key gate scripts,
checking for: short-circuit falsy-traps, return-value discards, parameter wiring
mismatches, threshold dodges, numerical stability issues, and stub implementations.

### Confirmed Non-Issues (False Positives Ruled Out)

| Finding | Verdict | Reason |
|---------|---------|--------|
| `exit_quality_audit.py:132` division by `total` | OK | `if trades.empty: return` early guard ensures `total > 0` |
| `ensemble.py` `if not normalized: continue` | OK | Empty dict is intentional sentinel per contract |
| `check_forecast_audits.py` `violation_rate=0.0` when `effective_n=0` | OK | `warmup_required` gate at line 724 exits before violation_rate check |
| `run_overnight_refresh.py` unchecked `rc` | OK | All `rc =` assignments have explicit checks after |
| `da_floor`/`da_weight_cap` not wired from YAML | OK | `ensemble_kwargs = {k: v for k, v in ensemble_cfg.items()}` passes all YAML keys |
| `min_lift_fraction` not enforced | OK | Enforced at `check_forecast_audits.py:794`; soft gate when `disable_if_no_lift=False` |
| `ensemble_health_audit.py` division by zero | OK | `if total <= EPSILON` guard on line 359 |
| NaN propagation in `exit_quality_audit.py` | OK | Intentional: `risk_unit.replace(0.0, np.nan)` produces NaN; pandas median() skips NaN |

### Documented Threshold Values (Operational Reference)

| Config | Field | Value | Notes |
|--------|-------|-------|-------|
| `forecaster_monitoring.yml` | `quant_validation.max_fail_fraction` | 0.85 | Hard gate (was 0.95 in Phase 7.14) |
| `forecaster_monitoring.yml` | `quant_validation.warn_fail_fraction` | 0.80 | Yellow zone |
| `forecaster_monitoring.yml` | `regression_metrics.max_violation_rate` | 0.35 | RMSE gate |
| `forecaster_monitoring.yml` | `regression_metrics.min_lift_fraction` | 0.25 | Soft (fail-open when `disable_ensemble_if_no_lift=false`) |
| `forecaster_monitoring.yml` | `regression_metrics.disable_ensemble_if_no_lift` | false | Fail-open; flip condition documented in AGENTS.md |
| `forecasting_config.yml` | `ensemble.diversity_tolerance` | 0.05 | Strict blend diversity gate |
| `forecasting_config.yml` | `ensemble.da_floor` | 0.10 | DA penalty floor (Phase 7.17) |
| `forecasting_config.yml` | `ensemble.da_weight_cap` | 0.10 | Max weight for DA-penalized model (Phase 7.17) |
| `check_forecast_audits.py` | fallback `max_violation_rate` | 0.25 | Tighter than YAML (0.35) — safe-fail when config absent |

### Noted Inconsistency (Non-Critical)

`max_violation_rate` hardcoded Python fallback (0.25) is tighter than the YAML value (0.35).
This is a **safe-fail posture**: if `forecaster_monitoring.yml` is absent (e.g., bare CI run),
the gate uses a stricter threshold. In production the YAML controls.

---

## Files Added / Modified

### New Files
- `scripts/dedupe_audit_windows.py`
- `scripts/ensemble_health_audit.py`
- `scripts/exit_quality_audit.py`
- `tests/scripts/test_dedupe_audit_windows.py`
- `tests/scripts/test_ensemble_health_audit.py`
- `tests/scripts/test_exit_quality_audit.py`
- `Documentation/PHASE_7.17_ENSEMBLE_HEALTH_AUDIT.md` (this file)

### Modified Files
- `forcester_ts/ensemble.py` — `_apply_da_cap()`, `EnsembleConfig` fields, `select_weights()`
- `config/forecasting_config.yml` — `da_floor`, `da_weight_cap` added to ensemble section
- `scripts/run_overnight_refresh.py` — Steps 2.8 + 2.9
- `requirements.txt` — `hypothesis==6.151.9`
- `.gitignore` — `.hypothesis/` excluded
- `tests/forcester_ts/test_ensemble_config_contract.py` — Phase 7.17 unit + Hypothesis tests

---

## Usage Runbook

```bash
# 1. Run dedupe check (dry-run)
python scripts/dedupe_audit_windows.py
# Exit 1 = duplicates found; run --apply to remove

# 2. Run ensemble health audit (dry-run)
python scripts/ensemble_health_audit.py
# Check: logs/ensemble_health/ensemble_health_YYYYMMDD.md

# 3. Run ensemble health audit + write adaptive weights to config
python scripts/ensemble_health_audit.py --write-config --write-report

# 4. Verify adaptive weights written
python -c "
import yaml
cfg = yaml.safe_load(open('config/forecasting_config.yml'))
aw = cfg.get('adaptive_candidate_weights', 'NOT FOUND')
print('computed_at:', aw.get('computed_at'))
print('weights:', aw.get('weights'))
"

# 5. Run exit quality audit
python scripts/exit_quality_audit.py
# Expect: interpretation label (stop_too_tight | holding_too_short | mix)

# 6. Targeted tests
pytest tests/scripts/test_ensemble_health_audit.py \
       tests/scripts/test_dedupe_audit_windows.py \
       tests/scripts/test_exit_quality_audit.py \
       tests/forcester_ts/test_ensemble_config_contract.py -v

# 7. Full regression baseline
pytest -m "not gpu and not slow" -q
# Expect: 1332 passed, 1 skipped, 28 deselected, 7 xfailed
```
