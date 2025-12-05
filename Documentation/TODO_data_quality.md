# Data Quality & Model Readiness TODOs (TS + LLM)

- Enforce temporal cleanliness: normalize DatetimeIndex with explicit freq (B/D), pad only small gaps, reject windows with large holes; keep train/val/test strictly chronological and fit scalers on train only (persist with checkpoints).
- Missing/outlier handling: cap ffill/interp to short gaps (≤3 bars); winsorize returns/vol; drop non-positive prices/absurd volumes; keep “raw” vs “cleaned” views.
- Per-window quality scoring: compute coverage, missing%, outlier count, stationarity flags (ADF/KPSS), regime tags; derive a quality score; gate TS/LLM routing on the score and persist it with forecasts/signals.
- Feature/label builder: standard TS features (log returns, rolling vol, momentum 5/20/60, RSI/trend slope, drawdown, turnover); regime features (vol band, stationarity, calendar tags); labels as log-return horizons clipped (e.g., ±5σ); store net_expected_return after friction.
- Source provenance/caching: persist source used (yfinance/alpha_vantage/ctrader), cache hit/miss, latency, and failovers; reconcile multi-source conflicts within a tolerance and annotate confidence.
- Signal/forecast persistence: for every forecast/signal store ts/ticker/source (TS/LLM), confidence, net_expected_return, risk score, quality score, data source, latency; emit run-level metrics (routing counts, latencies) to DB and dashboard JSON.
- LLM prompt hygiene: feed structured stats (recent returns/vol band, quality score, regime tags, risk constraints); cap prompt size; enforce latency guard with retry/backoff.
- Monitoring/audit: extend dashboard JSON with quality/source/latency; show recent signals with quality badges; add DB inserts for performance snapshots (PnL, win rate if available), forecast quality, and signal provenance.
- Normalization/scale control: standardize features per ticker with stored scalers; clamp expected_return for routing but log raw values for audit.
- Checklist alignment: map completed items against AGENT_DEV_CHECKLIST, OPTIMIZATION/SEQUENCED plans, SARIMAX/SAMOSSA checklists; update NEXT_TO_DO_SEQUENCED.md and CRITICAL_REVIEW.md when gates (quality scoring, persistence, dashboards) land.
- Quant validation & TS health gates: wire `logs/signals/quant_validation.jsonl`, `scripts/check_quant_validation_health.py`, and `scripts/compare_forecast_models.py` into brutal/CI so pipelines fail fast when TS signal quality (expected_profit, RMSE/sMAPE, directional accuracy vs SAMOSSA baseline) degrades beyond configured limits.
