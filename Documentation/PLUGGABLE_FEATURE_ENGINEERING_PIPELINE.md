# Pluggable Feature Engineering Pipeline (Add-ons, Profit-Gated)

## Scope

This document defines a **pluggable data + feature engineering pipeline** for Portfolio Maximizer where:

- New feature blocks (called **add-ons**) can be plugged into the existing ETL stages.
- Add-ons are **off by default** and **only promoted** when they measurably improve:
  - model/forecast quality,
  - trading signal quality,
  - and overall compute efficiency (or acceptable, measured overhead).
- **OpenClaw AI** is the default notification surface for promotion decisions (when configured).

This is intentionally conservative. The system should prefer "no add-on" over unproven complexity.

Related files:

- Pipeline stages: `config/pipeline_config.yml`
- Preprocessing hooks (scaffold): `config/preprocessing_config.yml`
- Sentiment scaffold (strict gating, disabled): `config/sentiment.yml`
- Sentiment plan (profit-gated): `Documentation/SENTIMENT_SIGNAL_INTEGRATION_PLAN.md`
- Feature engineering roadmap: `Documentation/FEATURE_ENGINEERING_PIPELINE_TODO.md`
- OpenClaw integration: `Documentation/OPENCLAW_INTEGRATION.md`

## Core Principles (Non-Negotiable)

- **Profit-gated by default**: add-ons stay dormant until the strategy is already profitable and quant-health is green.
- **Leak-free by construction**:
  - Training features must be computed using only past information.
  - Any feature derived from "future known" information is forbidden.
  - All join/aggregation operations must be time-aligned and shift-safe.
- **Compute-aware**:
  - Add-ons must publish their measured overhead and any external API costs.
  - If the add-on makes the pipeline slower, it must buy its way in with measured lift.
- **Operationally safe**:
  - Add-ons must degrade gracefully: missing data => passthrough, no crashes.
  - Live mode should prefer cached artifacts over live network calls.
- **Auditable**:
  - Every add-on must emit: feature list/version, provenance, and gating decisions into `logs/` and `reports/`.

## Where Add-ons Plug In (Conceptual)

Current core stages (see `config/pipeline_config.yml`):

1. `data_extraction`
2. `data_validation`
3. `data_preprocessing`
4. `data_storage`
5. `time_series_forecasting`
6. `time_series_signal_generation`
7. `signal_router`

Add-ons can plug in at three points:

- **A) Exogenous feature add-ons (preferred)**: between `data_storage` and `time_series_forecasting`
  - Output: feature frames that can be consumed by SARIMAX-X / other models as exogenous inputs.
- **B) Signal overlay add-ons**: between `time_series_signal_generation` and `signal_router`
  - Output: confidence/size adjustments (bounded, reversible), never primary signal generation.
- **C) Reporting-only add-ons**: after routing
  - Output: dashboards/alerts context only, no effect on trading.

Sentiment can be implemented as (A) and/or (B), but should start as (C) then (B) (shadow -> limited impact) before being considered for (A).

## Add-on Contract (What "Suitable" Means)

An add-on is "suitable" only if it produces features that are:

- **Time-indexed**: keyed by `(ticker, timestamp)` or equivalent.
- **Numerically safe**: no NaN/inf at export boundaries; clamped where appropriate.
- **Stable schema**: feature names are explicit and versioned.
- **Join-safe**: clearly defined merge semantics (left-join onto price bars by default).
- **Leak-safe**: any aggregation window uses only historical rows; default shift rules are explicit.

Recommended naming conventions:

- Prefix features by domain: `sent_*`, `regime_*`, `tail_*`, `xs_*`.
- If a feature is shifted for leakage prevention, include it in metadata (do not rename columns just to encode shift).

## Gating Policy (When an Add-on Is Allowed to Run)

There are two gates:

### 1) Global gate (system readiness)

Add-ons are only allowed to influence signals when:

- Profitability gates are met (see sentiment defaults in `config/sentiment.yml`):
  - trailing positive PnL windows (90d and 180d),
  - Sharpe >= 1.1,
  - max drawdown <= 0.22,
  - win rate >= 0.52,
  - trade count >= minimum.
- Quant validation is GREEN per `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md`.
- Pipeline health is stable (brutal/integration suites green; no DB corruption; no persistent stage errors).

If global gate fails: add-ons must remain **dormant** or **shadow-only** (compute + log, no trading impact).

### 2) Per add-on gate (measured incremental lift)

An add-on is only promoted to affect models/signals if an A/B evaluation shows:

- **No degradation** in risk:
  - max drawdown not worse (or within an explicitly allowed epsilon),
  - tail metrics not worse (CVaR proxy, worst-day loss).
- **Measurable improvement** in at least one primary objective:
  - trading: Sharpe, profit factor, or net PnL on pre-declared windows,
  - forecasting: RMSE/sMAPE, directional accuracy, or calibration metrics.
- **Compute overhead within budget**:
  - target: < 15% walltime overhead vs baseline pipeline run, or a documented exception.

Promotion decisions must be recorded as artifacts and (by default) notified via OpenClaw.

## Compute Budgets (Practical Defaults)

Add-ons must make their compute footprint explicit:

- Stage walltime: measured via pipeline stage timings (see `etl/pipeline_logger.py`).
- External calls:
  - API call count,
  - bytes downloaded,
  - rate-limit behavior.
- Inference:
  - batch size,
  - device used (CPU/GPU),
  - model name and version.

If an add-on requires network:

- Default behavior in live mode should be "use cache; do not block trading".
- Network calls should be disabled unless explicitly enabled and budgeted.

## OpenClaw AI Defaults (Organic, Low-Friction)

OpenClaw AI is the default place to send:

- add-on promotion decisions (PASS/FAIL + deltas),
- compute budget violations,
- unexpected regressions (e.g., negative lift detected).

This repo already supports OpenClaw notifications (optional, no-op until configured):

- Config: `Documentation/OPENCLAW_INTEGRATION.md`
- Manual send: `python scripts/openclaw_notify.py --to "<target>" --message "..."`.

Recommended message format for add-on decisions (keep it short):

- `ADDON=<name> STATUS=<shadow|rejected|promoted> DELTA_SHARPE=... DELTA_DD=... DELTA_RUNTIME=... ARTIFACT=<path>`

## Implementation Sketch (Docs-First)

No code is required to adopt this policy, but the intended integration points are:

- Config-driven enablement:
  - sentiment: `config/sentiment.yml`
  - future add-ons: `config/<addon>.yml` or a consolidated `config/feature_addons.yml`
- Stage wiring (future):
  - add a pipeline stage (exogenous features) after `data_storage`
  - add a pipeline stage (signal overlays) before `signal_router`
- Tests:
  - each add-on must have a "safe defaults" test like `tests/sentiment/test_sentiment_config_scaffold.py`
  - add-on must have a smoke test in synthetic mode (offline) validating:
    - schema stability,
    - no NaN/inf at output,
    - join alignment.

If you only do one thing: keep add-ons disabled until they prove lift with evidence bundles under `reports/`.

### Example Pipeline Stage Stub (Future)

If you decide to wire an add-on stage into the orchestrator, keep it optional and dependency-scoped.

Example `config/pipeline_config.yml` stage (illustrative only; requires implementation code):

```yaml
    - name: "feature_addons"
      description: "Optional exogenous features and overlays (profit-gated)"
      module: "etl.feature_addons"
      class: "FeatureAddonRunner"
      config_file: "config/feature_addons.yml"
      timeout_seconds: 180
      retry_attempts: 1
      required: false
      enabled: false
      depends_on: ["data_storage"]
```
