# Agent B Dashboard Runtime Truth Handoff (2026-03-08)

Owner: Agent B  
Prepared by: Agent C  
Scope: dashboard bridge/runtime truth path only

This handoff is based on verified runtime evidence, not just local tests.

## Why This Is Highest ROI

Current checked-in bridge code and tests are ahead of the actively served
dashboard payload. That means:

1. users are still seeing stale semantics even though parts of the code are fixed
2. readiness/reporting conclusions can still be wrong in the live UI
3. this can be corrected without waiting for new trading evidence

This is the highest implementation ROI that does not require strategy changes or
new market data.

## Verified Evidence

Commands run:

```powershell
python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db --output logs\dashboard_data_review_tmp.json
python -c "import json; from pathlib import Path; served=json.loads(Path('visualizations/dashboard_data.json').read_text(encoding='utf-8')); fresh=json.loads(Path('logs/dashboard_data_review_tmp.json').read_text(encoding='utf-8')); print({'served_missing_fields':[k for k in ['payload_schema_version','payload_digest','performance_unknown','positions_stale','positions_source'] if k not in served and k not in served.get('meta',{})], 'fresh_truth': {'performance_unknown': fresh.get('performance_unknown'), 'positions_stale': fresh.get('positions_stale'), 'positions_source': fresh.get('positions_source'), 'data_origin': fresh.get('meta',{}).get('data_origin'), 'trade_sources': fresh.get('meta',{}).get('provenance',{}).get('trade_sources')}, 'checks': fresh.get('checks')})"
python -m pytest tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_live_dashboard_wiring.py -q
```

Observed:

```json
{
  "served_missing_fields": [
    "payload_schema_version",
    "payload_digest",
    "performance_unknown",
    "positions_stale",
    "positions_source"
  ],
  "fresh_truth": {
    "performance_unknown": false,
    "positions_stale": true,
    "positions_source": "trade_executions_fallback_stale",
    "data_origin": "synthetic",
    "trade_sources": {
      "synthetic": 156,
      "yfinance": 89
    }
  },
  "checks": [
    "performance_metrics missing; run performance aggregation.",
    "portfolio_positions stale (as_of=2026-02-19, max_age_days=14); using filtered trade_executions fallback."
  ]
}
```

Targeted tests:
- `test_dashboard_db_bridge.py`: pass
- `test_live_dashboard_wiring.py`: pass

Interpretation:
- the bridge code path can emit the new truth fields
- the served artifact is still stale-schema
- provenance labeling remains semantically wrong for mixed-source state

## Defect 1 - Active Dashboard Producer Drift

### Problem

The actively served `visualizations/dashboard_data.json` is missing fields that
the checked-in bridge now emits:

- `meta.payload_schema_version`
- `meta.payload_digest`
- `performance_unknown`
- `positions_stale`
- `positions_source`

This is a live runtime drift issue.

### Files / Surfaces

- `scripts/dashboard_db_bridge.py`
- `visualizations/dashboard_data.json`
- active bridge daemon process

### Likely Root Causes

One of:

1. the running bridge process has not been restarted after code changes
2. another write path is still producing the older payload shape
3. merge-with-existing behavior is preserving a stale top-level structure

### Acceptance

After Agent B’s fix:

```powershell
python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db --output logs\dashboard_data_review_tmp.json
```

and the actively served `visualizations/dashboard_data.json` must both contain:

- `meta.payload_schema_version`
- `meta.payload_digest`
- `performance_unknown`
- `positions_stale`
- `positions_source`

No schema drift between one-shot and served payload should remain.

## Defect 2 - Mixed Provenance Mislabeling

### Problem

The current bridge output reports:

- `data_origin = synthetic`
- while `trade_sources = {"synthetic": 156, "yfinance": 89}`

That is semantically mixed, not purely synthetic.

The UI then escalates synthetic origin as a blocker, so this is not cosmetic.

### Relevant Logic

Current provenance origin logic appears to consider non-synthetic `ohlcv_sources`
but not non-synthetic `trade_sources` strongly enough for the mixed case.

### Acceptance

When both synthetic and non-synthetic trade sources are present:

- `meta.data_origin` must be `mixed`
- the audit console must not label the payload as purely synthetic

Add/keep a regression test for this exact case.

## Defect 3 - Unknown/Stale Truth Must Reach the Served UI

### Problem

Fresh one-shot output shows:

- `positions_stale = true`
- `positions_source = trade_executions_fallback_stale`
- `checks[]` includes stale positions and missing `performance_metrics`

But the served payload does not expose those fields.

That means the UI cannot honestly render stale/unknown state in live use.

### Acceptance

The served payload must preserve:

- `positions_stale`
- `positions_source`
- `performance_unknown`
- `checks[]`

The live UI must visibly reflect them after a normal dashboard refresh.

## Defect 4 - Performance Truth Is Still Semantically Weak

### Problem

Fresh one-shot output currently shows:

- `performance_unknown = false`
- `checks[]` says `performance_metrics missing`
- performance source is still effectively the PnL integrity fallback

That can be defensible only if the UI and payload clearly distinguish:

- realized PnL metrics available from integrity
- richer performance analytics unavailable from `performance_metrics`

If not, this remains misleading.

### Acceptance

Agent B should ensure the served payload/UI distinguishes:

1. realized canonical PnL metrics
2. unavailable advanced performance metrics

If top-level performance is partly derived from integrity and partly unavailable,
the UI should not collapse that into a clean “all known” state.

## Recommended Implementation Order

1. Fix the active producer drift first
2. Fix mixed provenance classification second
3. Verify stale/unknown fields survive into the served artifact
4. Only then refine UI messaging if needed

This order is important because otherwise UI work may target fields that the
live producer still fails to emit.

## Suggested Verification After Agent B Changes

```powershell
python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db --output logs\dashboard_data_review_tmp.json
python -m pytest tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_live_dashboard_wiring.py -q
python scripts/project_runtime_status.py --pretty
```

And compare:

- `visualizations/dashboard_data.json`
- `logs/dashboard_data_review_tmp.json`

Field parity required:

- `meta.payload_schema_version`
- `meta.payload_digest`
- `performance_unknown`
- `positions_stale`
- `positions_source`
- corrected `meta.data_origin`

## Limitations / Dependencies

### Temporal

- this work does not depend on the next trading day
- it can be completed while the live evidence lane is idle

### Spatial

- validation is specific to the current Windows workstation and its running bridge daemon

### Technical

- the active runtime process must be restarted or otherwise reconciled if it is serving stale-schema output
- do not mix this with gate-threshold or strategy changes

## Non-Scope

- no changes to `production_audit_gate.py`
- no changes to `capital_readiness_check.py`
- no changes to trading strategy logic
- no changes to denominator acceptance policy
