# Repository Sanitation Guardrails

Status: Mandatory
Scope: `scripts/`, `models/`, `config/`, `integrity/`, `tests/`
Last updated: 2026-03-07

## Purpose

Prevent repository drift that silently degrades readiness, linkage, telemetry truthfulness,
and PnL integrity through duplication, mismatched wiring, and fail-open semantics.

This document is a standing anti-regression contract for sanitation work.

## Non-Negotiable Rules

1. Single source of truth per critical workflow.
- Duplicate gate/linkage/eligibility logic across scripts is not allowed unless one path is explicitly marked compatibility-only and tested.

2. No silent denominator distortion.
- Readiness/linkage denominators must be explicit and context-aware.
- `NON_TRADE_CONTEXT` and `INVALID_CONTEXT` must be surfaced and excluded by flags, not silently dropped.

3. No fail-open status promotion.
- Missing required artifacts, invalid context, or blocking integrity violations cannot be reported as PASS/OK.

4. No hidden timestamp fallback semantics.
- Timestamp-primary matching remains canonical; any date fallback must emit a reason code and be measured.
- Date-only truncation must not silently replace timestamp matching in readiness paths.

5. No config no-op drift.
- Config keys must be either enforced in code or explicitly marked deprecated with warnings.
- Unsupported knobs cannot silently appear in new configs.

6. No unbounded threshold copies.
- Readiness and gate thresholds must be centralized or mapped through shared config/default helpers.

7. Keep additive compatibility.
- Backward-compatible outputs are additive only in this phase; do not remove legacy keys used by downstream tooling.

8. No numerical-stability blind spots.
- Non-finite metrics (`NaN`, `Inf`) in gating/readiness paths must be surfaced as errors, never coerced into PASS-adjacent states.

## High-Risk Drift Patterns to Block

1. Duplicate SQL for lifecycle integrity checks in multiple gate scripts.
2. Repeated ticker eligibility aggregation logic across charting and gating scripts.
3. Parallel linkage implementations with divergent dedupe keys or window semantics.
4. Divergent artifact freshness logic (timestamp field vs filesystem mtime).
5. Hardcoded readiness thresholds drifting from config-driven gate policies.
6. Report/dashboard PASS or OK states when required chart/sidecar artifacts are missing or stale.
7. Placeholder/dead compatibility stubs that mask missing implementations in active paths.

## Required Review Checklist (Before Merge)

1. Duplication scan
- `rg -n "SELECT|FROM trade_executions|close_ts < entry_ts|forecast_horizon|eligible|matched" scripts integrity`

2. Config contract scan
- `rg -n "enable_|routing_mode|sidecar_max_age" config models scripts`

3. Denominator/status scan
- `rg -n "NON_TRADE_CONTEXT|INVALID_CONTEXT|counts_toward_|phase3_ready|reason_code|DATE_FALLBACK_USED" scripts tests`

4. Targeted anti-regression tests (minimum)
- `python -m pytest tests/scripts/test_run_all_gates.py tests/scripts/test_production_audit_gate.py tests/scripts/test_update_platt_outcomes.py tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_telemetry_contract_policy.py -q`

5. Fast regression lane
- `python -m pytest -m "not gpu and not slow" --tb=short -q`

If any item fails, sanitation is not complete.

## Repository Cleanup Actions (Current Baseline)

1. `scripts/production_audit_gate.py` vs `scripts/capital_readiness_check.py`
- Category: DUPLICATION
- Action: `MERGE_INTO scripts/capital_readiness_check.py` (shared lifecycle integrity helper in one module)

2. `scripts/compute_ticker_eligibility.py` vs `scripts/generate_performance_charts.py`
- Category: DUPLICATION
- Action: `MERGE_INTO scripts/compute_ticker_eligibility.py` (reuse one eligibility aggregation function)

3. `scripts/check_forecast_audits.py`, `scripts/run_live_denominator_overnight.py`, `scripts/outcome_linkage_attribution_report.py`
- Category: DUPLICATION / UNALIGNED_IMPL
- Action: `CONSOLIDATE TO SINGLE SOURCE OF TRUTH` for linkage status taxonomy and denominator logic

4. `config/signal_routing_config.yml` vs `models/signal_router.py`
- Category: UNALIGNED_IMPL
- Action: `ENFORCE_CONFIG` for supported keys; `MARK_DEPRECATED` for unsupported knobs with explicit warnings

5. `config/signal_routing_config.hyperopt.yml`
- Category: BLOAT
- Action: `MARK_DEPRECATED` then remove after reference scan confirms no inbound usage

## Documentation and Evidence Requirements

1. Any sanitation change must update this file when it changes contracts or baseline risks.
2. Sanitation findings must be recorded in `Documentation/SESSION_COMPLETE_SUMMARY.md` or in a dedicated sanitation artifact under `Documentation/`.
3. Do not claim readiness improvements without command-level evidence from the checklist above.

## Definition of Done for Sanitation Passes

1. No unresolved critical duplication in readiness/linkage/gate code paths.
2. Config-to-code contract scan shows no silent no-op keys in active config.
3. Anti-regression test set and fast lane pass.
4. Documentation links remain current in:
- `Documentation/DOCS_INDEX.md`
- `Documentation/DOCUMENTATION_INDEX.md`
- `README.md`
