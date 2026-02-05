# Live Dashboard Optimization & Upgrade TODO (Institutional-Grade)

**Purpose**: Upgrade the live dashboard to institutional-grade reliability, clarity, and operational usefulness.
**Scope**: UI/UX, data integrity, performance, observability, and governance.
**Status**: Draft checklist

**P0 Reliability & Data Integrity**
- [ ] Standardize the dashboard payload schema version and validate on load; show a clear warning if schema mismatch.
- [ ] Persist `portfolio_positions` and `performance_metrics` reliably so the dashboard does not depend on fallbacks.
- [ ] Enforce single source of truth for `run_id` and expose it in all sections (signals, trades, charts).
- [ ] Add a data freshness indicator with hard thresholds (e.g., stale if >10 minutes old).
- [ ] Gate all derived metrics on non-empty data; avoid divide-by-zero and placeholder noise.

**P0 Operational Safeguards**
- [ ] Implement a safe-mode banner when data is partial or missing (no trades, no price series, or missing performance metrics).
- [ ] Add a health summary block with red/amber/green status for core data tables.
- [ ] Add a “last successful pipeline run” reference linked to DB provenance metadata.

**P1 Performance & Efficiency**
- [ ] Decouple diagnostics refresh from data refresh; only refresh diagnostics when `run_id` changes or manually requested.
- [ ] Add configurable polling interval and default to 15–30 seconds for production.
- [ ] Add client-side caching for static images and local TTL for missing assets.
- [ ] Add selective rendering to avoid full page updates when only a small subset changes.

**P1 Observability & Auditability**
- [ ] Persist dashboard snapshots (JSON) on each refresh to `data/dashboard_audit.db` with run metadata.
- [ ] Add a “payload digest” hash to detect unexpected changes between refreshes.
- [ ] Show the DB path and mirror path currently in use (read-only)
- [ ] Surface quant-validation status (pass/fail counts) if available.

**P1 UX/Decision Support**
- [ ] Add a compact “Key Decisions” panel with: top signals, highest risk position, and biggest PnL swing.
- [ ] Show realized vs unrealized PnL split in the header area.
- [ ] Add a position risk table with exposure, volatility proxy, and days held.
- [ ] Add multi-ticker diagnostics selection and quick switch to “last trade ticker.”

**P2 Data Quality & Drift**
- [ ] Display data quality snapshot per ticker (missing %, coverage, outlier frac) with thresholds.
- [ ] Add PSI/vol_psi drift indicators and a simple trend banner.
- [ ] Add data provenance tags per ticker (source, last refreshed, row count).

**P2 Charts & Visual Analytics**
- [ ] Add an equity curve with drawdown overlay and max drawdown marker.
- [ ] Add confidence distribution chart and rolling hit-rate chart.
- [ ] Add a trade timeline with entry/exit markers and regime labels.

**P2 Controls & Workflow**
- [ ] Add explicit “Refresh Now” and “Pause Auto Refresh” states to the UI.
- [ ] Add a manual diagnostics refresh button and visual last-refresh time.
- [ ] Add a “download JSON” button for audits.

**P3 Security & Compliance**
- [ ] Ensure no secrets are exposed in the payload (strip tokens, credentials, and internal file paths).
- [ ] Add a security banner noting the dashboard is local-only unless explicitly hosted.
- [ ] Add a data retention policy for `dashboard_audit.db` snapshots.

**P3 Testing & CI**
- [ ] Add a dashboard payload contract test to CI (schema validation and required fields).
- [ ] Add a headless snapshot test to confirm UI renders without console errors.
- [ ] Add a synthetic dataset test for diagnostics image rendering.

**P3 Deployment & Runbook**
- [ ] Document recommended local serving method and ports.
- [ ] Add troubleshooting steps for missing images and empty sections.
- [ ] Add a minimal “dashboard health check” command.

**Acceptance Criteria (Promotion Gate)**
- [ ] All P0 items completed and verified.
- [ ] Dashboard shows non-empty positions, trades, and performance for a live run.
- [ ] No repeated 404s for diagnostics assets after one refresh cycle.
- [ ] Data freshness banner correctly switches from OK to stale after threshold.

