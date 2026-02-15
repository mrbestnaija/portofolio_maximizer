# Feature Add-on Promotion Checklist (Measured Lift, Measured Cost)

## Goal

This checklist prevents "feature creep" by requiring evidence that an add-on improves performance/signal quality with an acceptable compute trade-off before it is allowed to affect models or trades.

Applies to all add-ons, including sentiment (see `Documentation/SENTIMENT_FEATURE_ADDON.md`).

## Definitions (Keep It Simple)

- **Baseline**: pipeline run with the add-on disabled.
- **Shadow**: add-on computes features and logs them, but **does not** affect models/signals/trades.
- **A/B**: pipeline runs with and without the add-on on the same pre-declared windows, with metrics compared.
- **Promotion**: enabling an add-on to influence either model inputs (exogenous features) or signal overlays.

## 1) Global Readiness Gate (Must Be True)

Do not promote any add-on unless:

- Profitability and quant-health are already GREEN.
- Pipeline is stable (no recurring errors, no DB corruption).
- A reproducible runtime is used for evidence-grade results (see `Documentation/RUNTIME_GUARDRAILS.md`).

Recommended: run the production gate and record its output artifact.

## 2) Evidence Bundle Requirements (What You Must Save)

For every promotion decision, save a bundle containing:

- commit SHA (`git rev-parse HEAD`)
- config hashes for:
  - `config/pipeline_config.yml`
  - add-on configs (e.g., `config/sentiment.yml`)
  - relevant signal/quant configs (e.g., `config/signal_routing_config.yml`)
- run IDs / pipeline IDs
- metrics summary (baseline vs add-on)
- compute summary (walltime deltas, API usage/cost if applicable)
- artifact paths under `logs/` and `reports/`

## 3) What "Measurable Lift" Means (Promotion Bar)

Pick 1-3 primary metrics and declare them before running A/B.

Recommended primary metrics:

- Trading: Sharpe, net PnL, max drawdown, profit factor
- Forecasting: directional accuracy, RMSE/sMAPE, calibration metrics

Promotion requires all of:

- No material risk degradation (drawdown and tail metrics do not worsen beyond a declared epsilon).
- Positive lift on at least one primary metric on multiple regimes/windows.
- Compute overhead within budget:
  - target: < 15% walltime overhead vs baseline (unless the lift justifies an exception).

If the add-on improves one window but degrades another, it is not ready for promotion (keep shadow-only).

## 4) Compute Budget Checklist (Hard Requirements)

Before any non-shadow promotion:

- The add-on has a caching strategy (avoid recomputing).
- Missing data/outages do not crash live runs (passthrough behavior).
- External API usage is rate-limited and capped.
- The add-on's worst-case runtime is known and recorded.

For sentiment specifically:

- Transformer scoring must be bounded (batching, caps on docs, optional sampling).
- Live mode should use cached aggregates; do not block trading on network calls.

## 5) OpenClaw AI Default (Organic Decision Loop)

Promotion decisions should be sent to OpenClaw AI by default (if configured) so you have a durable, low-friction record of what changed and why.

Configure (see `Documentation/OPENCLAW_INTEGRATION.md`):

- `OPENCLAW_TARGETS` (recommended) or `OPENCLAW_TO`
- optional `OPENCLAW_COMMAND`

Manual send helper:

```bash
python scripts/openclaw_notify.py --to "<target>" --message "ADDON=sentiment STATUS=shadow DELTA_SHARPE=... DELTA_DD=... DELTA_RUNTIME=... ARTIFACT=reports/feature_addons/<...>.md"
```

## 6) Report Template (Copy For Each Add-on Decision)

Create a decision record under `reports/feature_addons/`:

```markdown
# Feature Add-on Decision: <ADDON_NAME>

## Summary

- Status: shadow | rejected | promoted
- Reason: <1-3 bullets, objective>

## Provenance

- Date (UTC):
- Commit SHA:
- Runtime fingerprint:
- Config files:
  - config/pipeline_config.yml (hash: ...)
  - config/<addon>.yml (hash: ...)

## Evaluation Windows

- Train/val/test definitions:
- Backtest windows:
- Regimes covered:

## Metrics (Baseline vs Add-on)

- Primary metrics:
  - Sharpe:
  - Max drawdown:
  - Net PnL:
  - Profit factor:
- Secondary metrics:
  - Directional accuracy:
  - RMSE/sMAPE:
  - Win rate:

## Compute / Cost

- Baseline walltime:
- Add-on walltime:
- Overhead:
- External API calls:
- Notes:

## Failure Modes Observed

- Outages:
- Missing data:
- Stability issues:

## Decision

- Promote? yes/no
- If promoted:
  - scope (shadow -> overlay -> exogenous):
  - caps/limits:
  - rollback plan:
```
