# Live Dashboard Optimization TODO

Purpose: keep the dashboard audit-relevant without building a second stack.

Status:
- The dashboard already has payload freshness/status handling, audit snapshot persistence, pause/refresh controls, run-id-aware diagnostics refresh, missing-asset TTL, live denominator surfacing, and robustness/evidence status precedence.
- The remaining work is the small set of gaps that are not already implemented in code.

## Completed

### Reliability and payload integrity
- [x] Payload schema version is emitted by the bridge and validated on load in the dashboard UI.
- [x] Data freshness and stale-sidecar status are surfaced from bridge sidecars.
- [x] Derived metric rendering is guarded against missing data.
- [x] Safe-mode style warnings are shown when payload sections are missing or partial.
- [x] Health summary block is present in the header (`Data: OK/Partial/Stale/Error`).
- [x] `run_id` is propagated through the payload and used to control diagnostics refresh behavior.

### Observability and auditability
- [x] Dashboard payload snapshots are persisted by the bridge.
- [x] Payload digest hash is emitted and shown in the UI.
- [x] Active DB path / read path / mirror path are exposed in payload storage metadata and surfaced in the UI.
- [x] Quant-validation status is summarized in the bridge payload and surfaced in the UI.
- [x] Robustness payload includes cache status and forecast-audit sidecar context for the audit console.

### Interactivity and workflow
- [x] Auto-refresh can be paused and resumed from the UI.
- [x] Refresh cadence is configurable from the UI.
- [x] Diagnostics refresh is decoupled from normal polling and only forces image reloads when `run_id` changes or when a user retargets diagnostics.
- [x] Missing diagnostics assets use a local TTL to suppress repeated reload noise.
- [x] Audit console allows focus filtering (`All`, `Blockers`, `Freshness`, `Linkage`, `Coverage`) and click-through into relevant dashboard sections.

### Testing and runbook
- [x] Dashboard bridge tests cover payload schema metadata and quant-validation summary wiring.
- [x] Dashboard wiring tests cover polling controls, audit-console anchors, and robustness status precedence.
- [x] Windows dashboard manager health/status command exists and is exercised by tests.

## Remaining gaps worth implementing

### P1
- [ ] Upstream persistence hardening for `portfolio_positions` and `performance_metrics` so the dashboard relies less on fallback derivations.
- [ ] Surface the last successful pipeline run explicitly from provenance metadata, not just the current dashboard payload timestamp.
- [ ] Add a compact key-decisions strip for highest-risk position, biggest recent PnL move, and top actionable signal.
- [ ] Add realized vs unrealized PnL split in the header.

### P2
- [ ] Add position risk table with exposure, hold time, and simple volatility proxy.
- [ ] Add per-ticker provenance tags (source, last refresh, row count) in the universe panel.
- [ ] Add download/export button for the current dashboard JSON payload.
- [ ] Add a small client-side diff/highlight path so unchanged sections are not re-rendered every refresh.

### P3
- [ ] Add a headless render/smoke test that fails on JavaScript console errors.
- [ ] Add a targeted synthetic diagnostics image test for the diagnostics panel.
- [ ] Document local-only security expectations and retention policy for dashboard audit snapshots.

## Anti-bloat guardrails

- Do not build another dashboard payload layer when the bridge already emits the data.
- Do not add a second ticker-universe orchestration path; ETL and trading should both reuse `etl.data_universe.resolve_ticker_universe`.
- Do not add readiness or linkage language to the dashboard unless the underlying telemetry is production-valid.

## Acceptance criteria for the next dashboard slice

- [ ] No stale TODO item remains unchecked when the code is already implemented.
- [ ] Dashboard UI exposes payload schema, digest, storage path, quant-validation status, and audit-console focus state.
- [ ] ETL ticker resolution uses the shared universe resolver instead of duplicating discovery logic.
- [ ] New dashboard work adds either a test or a bridge/UI verification path in the same change.
