# Phase 7.39 — Paranoid Architectural Review

**Date**: 2026-03-07
**Regression baseline**: 1683 passed, 1 skipped, 28 deselected, 7 xfailed
**Commit**: cddd477
**Branch**: chore/repo-sanitization-p0-p2

---

## Overview

A paranoid systems-architect review of diagnostic/reporting scripts targeting:
- Short-circuit and mismatched wiring inconsistencies
- Threshold dodges (hardcoded vs config-driven)
- Stubs presented as live data
- Dead/misleading code
- Numerical stability gaps

Three confirmed architectural bugs were fixed, plus a new ticker eligibility gating
pipeline and several supporting hardening fixes.

---

## Bug 1 — Layer 1 Warn Threshold Misalignment (THRESHOLD DODGE)

**File**: `scripts/check_model_improvement.py`
**Severity**: HIGH — operators saw Layer 1 PASS while the production gate simultaneously FAILed

### Root Cause

`run_layer1_forecast_quality()` had `warn_lift_threshold: float = 0.05` hardcoded.
The production gate (`run_all_gates.py` Gate 2) reads `min_lift_fraction` from
`config/forecaster_monitoring.yml`, currently set to `0.25`.

A lift fraction of 20% would produce:
- Layer 1: PASS (20% > 5% hardcoded threshold)
- Gate 2: FAIL (20% < 25% config threshold)

The WIRE-01 fix in Phase 7.33 correctly aligned the RMSE *ratio* threshold by adding
`_load_layer1_lift_threshold()`, but left a second, independent `warn_lift_threshold`
parameter using a separate hardcoded default.

### Fix

Added `_load_min_lift_fraction()`:

```python
def _load_min_lift_fraction() -> float:
    # Phase 7.39: read min_lift_fraction from forecaster_monitoring.yml so the Layer 1
    # WARN threshold aligns with the gate's requirement.
    try:
        import yaml as _yaml_mlf
        monitoring_path = REPO_ROOT / "config" / "forecaster_monitoring.yml"
        if monitoring_path.exists():
            monitoring_cfg = _yaml_mlf.safe_load(monitoring_path.read_text(encoding="utf-8")) or {}
            return float(
                monitoring_cfg.get("forecaster_monitoring", {})
                .get("regression_metrics", {})
                .get("min_lift_fraction", 0.05)
            )
    except Exception:
        pass
    return 0.05
```

Changed `warn_lift_threshold` to `Optional[float] = None`; defaults to config value
when `None`, allowing explicit test overrides to still pass a numeric value.

**Contract test added**: `test_layer1_warn_threshold_aligned_with_gate_config` — a lift
fraction of 20% (above old 5% hardcode, below new 25% config) must produce WARN, not PASS.

---

## Bug 2 — Excursion Stub Undocumented (STUB)

**File**: `scripts/outcome_linkage_attribution_report.py`
**Severity**: MEDIUM — consumers had no machine-readable signal that MAE/MFE fields are always null

### Root Cause

`excursion_min_pct` and `excursion_max_pct` in every record were always `None`.
These require Maximum Adverse/Favorable Excursion data — per-bar OHLC during the
full holding period — which is not stored in the current `trade_executions` schema.
The fields existed for future use, but the summary contained no flag indicating they
were stubs, leaving consumers (dashboards, alerting) to silently treat `None` as
missing data rather than a known architectural limitation.

### Fix

Added `"excursion_data_available": False` to the summary dict, with an explanatory
comment on the record-level stub fields:

```python
# MAE/MFE fields: reserved for future bar-level excursion data.
# Requires per-bar OHLC during holding period which is not stored in the DB.
# Always None until bar storage is added; see summary["excursion_data_available"].
"excursion_min_pct": None,
"excursion_max_pct": None,
```

**Contract test added**: `test_build_report_excursion_stub_flag` — asserts
`summary["excursion_data_available"] is False` and all record fields are `None`.

---

## Bug 3 — Dead Status Computation (DEAD CODE / MISLEADING)

**File**: `scripts/generate_performance_charts.py`, `_build_metrics_summary()`
**Severity**: LOW — no functional impact, but misleading to readers and future maintainers

### Root Cause

`_build_metrics_summary()` computed a `"status"` key based on internal `warnings` and
`errors` lists. However, the caller (`generate_performance_artifacts()`) always
overwrote `metrics["status"]` with its own authoritative evaluation:

```python
# caller (lines 487-490):
if errors:
    metrics["status"] = "ERROR"
elif warnings:
    metrics["status"] = "WARN"
else:
    metrics["status"] = "PASS"
```

The computation inside `_build_metrics_summary` was never visible to any consumer.

### Fix

Removed the dead `"status"` key from the return value of `_build_metrics_summary`.
Added a comment directing readers to the caller for the authoritative assignment.

---

## Additional Fixes

### Ticker Eligibility Gate Pipeline

**Problem**: `scripts/apply_ticker_eligibility_gates.py` existed but was not wired into
the daily quality pipeline run.

**Fix**: `scripts/run_quality_pipeline.py` Step 1b now calls `apply_eligibility_gates()`
after `compute_ticker_eligibility`. The gate step result is recorded in `steps[]` with
`lab_only_tickers`, `gate_written`, and `status` fields.

`models/time_series_signal_generator.py` `_load_lab_only_override()` blocks lab-only
tickers at signal generation time (TTL-aware, fail-open when file absent/stale).

### Audit Path Collision Fix

**Problem**: `forcester_ts/forecaster.py` named audit files `forecast_audit_{timestamp}.json`
using only second-level granularity — concurrent forecasters could produce the same filename.

**Fix**: `_next_audit_path()` uses microsecond timestamp + UUID-4 8-char hex suffix:
```
forecast_audit_20260307_142351_123456_a1b2c3d4.json
```

`save_audit_report()` also backfills a `signal_context` block when the payload lacks
one (FORECAST_ONLY context), ensuring all audit files are parseable by downstream
attribution scripts.

### Exit Quality Audit NumPy Dtype Fix

**File**: `scripts/exit_quality_audit.py:93`

**Problem**: `df.loc[has_atr, "atr_proxy"] = df.loc[...] - df.loc[...]` triggered
pandas FutureWarning on mixed-dtype assignment.

**Fix**: Replaced with `np.where(has_atr, atr_series, np.nan)` (dtype-safe).

---

## Files Changed

| File | Change |
|------|--------|
| `scripts/check_model_improvement.py` | `_load_min_lift_fraction()` + `warn_lift_threshold=None` default |
| `scripts/outcome_linkage_attribution_report.py` | `excursion_data_available: False` in summary |
| `scripts/generate_performance_charts.py` | Dead `"status"` removed from `_build_metrics_summary` |
| `scripts/apply_ticker_eligibility_gates.py` | New: autonomous ticker gate file writer |
| `scripts/run_quality_pipeline.py` | Step 1b: apply eligibility gates |
| `scripts/exit_quality_audit.py` | NumPy dtype fix for ATR proxy |
| `forcester_ts/forecaster.py` | `_next_audit_path()` + UUID suffix + `signal_context` backfill |
| `models/time_series_signal_generator.py` | `_load_lab_only_override()` TTL-aware gate check |
| `tests/scripts/test_check_model_improvement.py` | `test_layer1_warn_threshold_aligned_with_gate_config` |
| `tests/scripts/test_outcome_linkage_attribution_report.py` | `test_build_report_excursion_stub_flag` |
| `tests/scripts/test_apply_ticker_eligibility_gates.py` | New: 16 eligibility gate tests |
| `tests/scripts/test_run_quality_pipeline.py` | Pipeline gate step integration tests |
| `tests/forcester_ts/test_forecaster_audit_contract.py` | Audit path + signal_context contract tests |
| `tests/test_execution_logging.py` | Forecaster audit report contract tests |

---

## Regression Results

```
pytest -m "not gpu and not slow" -q --tb=short
1683 passed, 1 skipped, 28 deselected, 7 xfailed in 217.61s
```

**Pre-existing failure** (not introduced by Phase 7.39):
- `tests/integration/test_time_series_signal_wiring_scaling.py::test_signal_scaling_invariant_under_price_rescale`
  — fails identically on the commit before Phase 7.39 changes.
