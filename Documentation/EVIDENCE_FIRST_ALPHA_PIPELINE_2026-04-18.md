# Evidence-First Alpha Pipeline

Date: 2026-04-18

## Scope

This document is the repo-wide reference for the evidence-first alpha pipeline.
It explains how preprocessing integrity, OOS evidence quality, provenance, and
gate semantics now flow through the stack and how that is verified.

## What The Pipeline Now Requires

- Production admission must be backed by observed OOS evidence, not proxy-only
  or heuristic-only evidence.
- Preprocessing must pass a post-fill / post-pad validator before a frame can
  be treated as production-ready.
- Provenance must remain visible from ingest through forecast, trade, audit,
  dashboard, and gate surfaces.
- Raw upside metrics such as Omega and payoff asymmetry remain visible, but
  they are not sufficient for acceptance by themselves.
- Continued deployment must fail closed when evidence coverage, freshness, or
  preprocessing quality erodes.

## Repo-Wide Flow

1. `scripts/run_etl_pipeline.py`
   - Runs extraction, validation, preprocessing, and storage.
   - Applies the post-preprocess validator after fill/pad.
   - Persists `preprocess_health` into the processed artifact metadata.

2. `forcester_ts/forecaster.py`
   - Loads OOS audit evidence and emits structured `evidence_health`.
   - Marks heuristic/proxy/stale evidence as research-only.
   - Ensures audit artifacts always carry an evidence snapshot, even on
     fallback-only or non-ensemble paths.

3. `scripts/check_model_improvement.py`
   - Treats Layer 1 as evidence health, not just diagnostics.
   - Surfaces `coverage_ratio`, `n_used_windows`, `n_skipped_missing_metrics`,
     `source_kind`, `freshness_status`, `rmse_rank_active`, and
     `evidence_health`.
   - Preserves the existing WARN/FAIL semantics while making evidence quality
     explicit.

4. `scripts/production_audit_gate.py` and `risk/barbell_promotion_gate.py`
   - Keep one enforcement path.
   - Expose explicit reasons for hygiene failures instead of hiding them behind
     a generic pass/fail.
   - Keep raw Omega and payoff asymmetry visible while requiring threshold,
     tail, path, regime, evidence, and provenance checks to pass.

5. `scripts/run_auto_trader.py`
   - Carries `preprocess_health` and `evidence_health` through candidate prep,
     execution reports, orchestration summaries, and dashboard payloads.
   - Fails closed in the live lane when evidence or preprocessing is not
     production-ready.

6. `scripts/dashboard_db_bridge.py`, `scripts/check_dashboard_health.py`, and
   `visualizations/live_dashboard.html`
   - Surface the same evidence and preprocessing state that the gates see.
   - Keep operational visibility aligned with production admission rules.

## Explicit Reason Codes

When evidence hygiene blocks production or demotes a path to research-only, the
failure should be explainable.

- `OOS_COVERAGE_THIN`
- `OOS_MISSING_METRICS`
- `PREPROCESS_DISTORTION`
- `HEURISTIC_FALLBACK`
- `PROVENANCE_UNTRUSTED`

## Preprocess Contract

The new post-preprocess validator classifies frames into three practical
states:

- `PASS` for clean frames that meet production contract thresholds.
- `WARN` for sparse, heavily imputed, or over-padded frames that remain usable
  for research but are not production-ready.
- `FAIL` for structural problems such as duplicates, non-monotonic dates, or
  non-finite values.

The validator records:

- `imputed_fraction`
- `padding_fraction`
- duplicate rows
- monotonic date checks
- finite-value checks
- minimum usable bars

## Evidence Contract

The structured evidence bundle now carries:

- `coverage_ratio`
- `n_used_windows`
- `n_skipped_missing_metrics`
- `lift_ci_low`
- `lift_ci_high`
- `samossa_da_zero_pct`
- `oos_metrics_available`
- `rmse_rank_active`
- `source_kind`
- `freshness_status`
- `fallback_class`

Production-ready evidence must be observed, fresh, and non-heuristic. Guarded
fallbacks can remain visible, but they stay out of the production bar unless
they have their own proof.

## Robust Testing

The pipeline was validated with both targeted contracts and the repo-wide fast
lane.

Targeted slice:

- `python -m pytest tests/etl/test_preprocessor.py tests/etl/test_data_source_manager.py tests/etl/test_data_source_manager_chunking.py tests/scripts/test_check_model_improvement.py tests/scripts/test_check_dashboard_health.py tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_dashboard_payload_invariants.py tests/scripts/test_live_dashboard_wiring.py tests/scripts/test_production_cron_hygiene.py tests/scripts/test_run_live_denominator_overnight.py tests/scripts/test_run_auto_trader_integrity.py tests/scripts/test_run_auto_trader_config_guard.py tests/scripts/test_run_auto_trader_rejection_taxonomy.py tests/scripts/test_run_etl_pipeline_ticker_universe.py tests/forcester_ts/test_forecaster_audit_contract.py -q`

Observed result:

- `159 passed`

Repo fast lane:

- `python -m pytest -m "not gpu and not slow" --tb=short -q`

Observed result:

- `2534 passed, 6 skipped, 45 deselected, 11 xfailed`

Static verification:

- Python compilation sweep completed successfully on the repo Python surface.
- `git diff --check` remained clean aside from expected CRLF normalization
  warnings on edited files.

## Residual Risk

- The live lane is intentionally narrow. Some sparse or heuristic-backed paths
  remain valuable for research, but they must not be mistaken for production
  evidence.
- Thresholds are staged and should be ratcheted only when backfill and coverage
  improve, not relaxed to force passage.

