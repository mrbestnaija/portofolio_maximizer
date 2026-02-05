# Ensemble Model Status (Source Of Truth)

**Date**: 2026-02-04

This doc is the canonical reference for how the **time-series ensemble** behaves in the current codebase and how to interpret “RESEARCH_ONLY” vs “KEEP” across tools/logs.

---

## What “Ensemble Active” Means Here

The system **does compute and use** an ensemble forecast bundle (SAMOSSA/MSSA-RL/GARCH/SARIMAX candidates) as the primary time-series forecast output when the ensemble build succeeds.

Concrete wiring:
- Forecaster builds and returns the ensemble bundle via `results["ensemble_forecast"]` and sets `results["mean_forecast"] = ensemble["forecast_bundle"]`: `../forcester_ts/forecaster.py`.
- Signal generation consumes the returned forecast bundle + metadata (weights/confidence/diagnostics), not a separate “portfolio allocation” layer: `../models/time_series_signal_generator.py`.

**Important**: the system also computes an **ensemble policy label** (`KEEP` / `RESEARCH_ONLY` / `DISABLE_DEFAULT`) for governance/monitoring. Today this label is recorded into metadata (and into model events) but **does not automatically replace** the `mean_forecast` bundle with a best-single fallback.

---

## Two Different “Statuses” You Will See

### 1) Per-forecast policy label (runtime metadata)

Location:
- `../forcester_ts/forecaster.py` (`_enforce_ensemble_safety`)

Meaning:
- `KEEP`: ensemble is within tolerance and not blocked by promotion/no-lift rules.
- `RESEARCH_ONLY`: ensemble did not meet the **promotion margin** (default 2% RMSE lift) for this evaluated window.
- `DISABLE_DEFAULT`: ensemble is materially worse than baseline beyond tolerance.

This is logged as:
- `[TS_MODEL] ENSEMBLE policy_decision :: status=...`

### 2) Aggregate audit gate decision (gate script)

Tool:
- `../scripts/check_forecast_audits.py`

Meaning:
- This is a **rolling governance gate** over recent `forecast_audit_*.json` files.
- It answers: “Across recent unique windows, are we within the configured RMSE tolerance and violation rate, and did we show enough lift across the holding period?”

This prints:
- `Decision: KEEP (...)` on pass
- exits non-zero on failure modes (violation-rate exceed, no-lift after holding period, etc.)

---

## Current Evidence (Reproducible)

Run:
```bash
simpleTrader_env/bin/python scripts/check_forecast_audits.py \
  --audit-dir logs/forecast_audits \
  --config-path config/forecaster_monitoring.yml \
  --max-files 500
```

As of **2026-02-04**, the output reports:
- Effective audits with RMSE: **25**
- Violations: **3** (12.00% violation rate; max allowed 25%)
- Lift fraction: **12.00%** (min required 10%)
- Decision: **KEEP (lift demonstrated during holding period)**

---

## External-Facing Reporting Guidance (Non-Negotiable)

When writing public/external summaries (e.g., README, deployment roadmaps), do **not** claim:
- “Ensemble forecasts are computed but discarded / unused”
- “System falls back to single model by default because it’s RESEARCH_ONLY”

Correct phrasing:
- “Ensemble forecast bundle is active and is the primary TS forecast output; governance labels may mark individual windows as RESEARCH_ONLY when the 2% promotion margin isn’t met, but the aggregate audit gate currently passes (Decision KEEP).”
