# Agent B Dashboard Runtime Truth Handoff (2026-03-08)

Doc Type: handoff_note
Authority: temporary Agent B implementation handoff from Agent C
Owner: Agent C
Last Verified: 2026-03-08
Verification Commands:
- `python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db --output logs\dashboard_data_review_tmp.json`
- `python -m pytest tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_live_dashboard_wiring.py -q`
Artifacts:
- `visualizations/dashboard_data.json`
- `logs/dashboard_data_review_tmp.json`
Supersedes: none
Expires When: superseded by a newer dashboard handoff note or retired after Agent B merges the remaining truth-path fixes

Owner: Agent B
Prepared by: Agent C
Scope: dashboard bridge/runtime truth path only

## Current Scope

The original runtime-drift issue is closed:

- the served payload now carries the current schema fields
- the served payload now reports `data_origin = mixed` when trade sources are mixed

The remaining Agent B lane is narrower:

1. keep served payload parity with one-shot bridge output
2. preserve `data_origin = mixed` in mixed-source cases
3. make stale/unknown performance and position semantics unambiguous in both payload and UI

## Verified Evidence

Commands run:

```powershell
python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db --output logs\dashboard_data_review_tmp.json
python -m pytest tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_live_dashboard_wiring.py -q
```

Observed:

```json
{
  "data_origin": "mixed",
  "payload_schema_version": 2,
  "performance_unknown": false,
  "positions_stale": true,
  "positions_source": "trade_executions_fallback_stale",
  "trade_sources": {
    "synthetic": 156,
    "yfinance": 89
  },
  "checks": [
    "performance_metrics missing; run performance aggregation.",
    "portfolio_positions stale (as_of=2026-02-19, max_age_days=14); using filtered trade_executions fallback."
  ]
}
```

Interpretation:

- runtime/schema delivery is now aligned
- mixed provenance labeling is now correct
- the remaining issue is reporting semantics for stale and fallback-derived values

## Remaining Issue 1 - Stale/Unknown Truth Semantics

### Problem

The served payload exposes:

- `positions_stale = true`
- `positions_source = trade_executions_fallback_stale`
- `performance_unknown = false`
- `checks[]` explicitly says performance metrics are missing and positions are stale

This can still be misread unless the UI clearly distinguishes:

1. realized canonical values that are available
2. advanced analytics or position state that are stale, fallback-derived, or unavailable

### Acceptance

After Agent B's fix:

- stale position state is visibly marked as stale in the UI
- fallback-derived position state is labeled by source
- unavailable advanced performance fields are not presented as fully known/healthy
- `checks[]` remains surfaced as the operator-facing explanation

## Remaining Issue 2 - Truth-Path Discipline Must Stay Green

### Problem

The mixed-provenance fix is now live, but it is important enough to keep under regression protection.

### Acceptance

When both synthetic and non-synthetic trade sources are present:

- `meta.data_origin` remains `mixed`
- the audit console must not label the payload as purely synthetic
- the served payload and one-shot payload must agree

## Recommended Implementation Order

1. keep served payload parity with one-shot bridge output
2. preserve `data_origin = mixed` in mixed-source payloads
3. refine stale/unknown UI semantics without changing gate logic

## Suggested Verification After Agent B Changes

```powershell
python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db --output logs\dashboard_data_review_tmp.json
python -m pytest tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_live_dashboard_wiring.py -q
python scripts/project_runtime_status.py --pretty
```

Field parity required between:

- `visualizations/dashboard_data.json`
- `logs/dashboard_data_review_tmp.json`

Minimum required truth fields:

- `meta.payload_schema_version`
- `meta.payload_digest`
- `performance_unknown`
- `positions_stale`
- `positions_source`
- `meta.data_origin = mixed`

## Non-Scope

- no changes to gate scripts
- no changes to trading logic
- no changes to readiness thresholds
