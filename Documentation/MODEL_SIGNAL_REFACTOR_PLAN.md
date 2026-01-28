Reference: Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md

# Model & Signal Refactor Plan (Institutional-Grade Path)

## Purpose
Replace proprietary forecasters/LLM-driven signalers with time-tested, auditable models while preventing performance regressions. Transition uses canary/shadowing so we erase legacy artifacts only after evidence shows non-degradation and profitability upside. Logs show three distinct blockers: (1) DB CHECK prevents ENSEMBLE saves (plumbing), (2) model key/metadata drift drops GARCH, and (3) decision logic / confidence scoring is saturating to winner-takes-all (largest risk).

## Guardrails (must hold before any code is merged)
- Runtime: WSL `simpleTrader_env/bin/python` only (see Documentation/RUNTIME_GUARDRAILS.md); record `which python`, `python -V`, and torch fingerprint with every run.
- Profitability evidence: no rollout without backtests showing >10% annual return and RMSE/sMAPE wins vs. current baseline; cite log artifacts, not projections.
- Config-driven only: no hardcoded model orders or ad-hoc thresholds; changes must route through config files and existing search utilities.
- Backward compatibility: new paths must be feature-flagged and able to revert to current behavior instantly.
- Line-count/complexity: keep refactors scoped (<500 LoC per phase) and avoid new paid dependencies.

## Current Problems to Eliminate (ranked)
- Confidence/selection miscalibration: ensemble degenerates to BEST_SINGLE with saturated scores (e.g., SAMoSSA at 0.9999 while RMSE gate fails). Scoring may be monotonicity-violating or using incomparable scales.
- Model metadata drift: inconsistent keys (`garch` vs `GARCH`) and missing summaries/confidence entries silently drop candidates.
- Plumbing: DB CHECK constraint blocks ENSEMBLE persistence; migrations collide with long-running pipelines.
- Proprietary/LLM signalers and bespoke artifacts entangled with forecasting outputs.
- Weak observability: insufficient audit of signal provenance and model metrics.

## Reality Check from Recent Logs
- Weights observed: `{'sarimax': 1.0}` / `{'samossa': 1.0}` → acting as BEST_SINGLE, not a blend.
- Policy gate disables ensemble for high RMSE ratio, yet SAMoSSA confidence is 0.9999 → confidence scaling likely saturating or defaulting to “perfect.”
- GARCH absent from confidence dict and summaries; RMSE ratios remain high (MSFT 1.682x, NVDA 1.223x).

## Target Architecture (time-tested components)
- Forecasting sleeves: data-learned SARIMAX (with exogenous features), data-learned GARCH for volatility, SAMOSSA retained only with measured lift; optional ETS/ARIMA variants via config if needed.
- Signal library: canonical factor signals (trend/momentum, mean-reversion, volatility breakout, carry/term-structure proxies, cross-sectional risk parity adjustments). Each signal exposes: input requirements, lookback window, normalization, and guardrails (liquidity filter, min history).
- Ensembling: confidence-driven weight selection with fallbacks for missing metrics, deterministic logging of candidate weights and confidence inputs.
- Routing/execution: signals flow through existing gates (quant validation, MTM/liquidation policies) without bypassing risk controls.

## Immediate Fix Order (must do before any model swap)
1) DB constraint: add `ENSEMBLE` (or drop CHECK/use ref table) and set `PRAGMA busy_timeout` in connection factory; run migrations before pipelines.
2) Canonical model keys: normalize to lowercase at entry to `forecasts` / `summaries` / confidence; map to DB enums only at persistence boundary.
3) Instrument summaries: log keys and presence of `regression_metrics` per model; confirm GARCH summary is written post-fit.
4) Confidence/selection math: fix saturation; ensure scores spread and are monotonic with RMSE (lower RMSE → higher score). Verify coordinator is not hard argmax unless policy gate demands fallback.
5) Alignment checks: ensure all model forecasts share index/horizon/scale (no log-price vs price mix, no off-by-one).

## Refactor Phases (sequential, canary-first)
1) Inventory & Baseline
   - Catalog all forecasters, signal generators, and artifacts (configs, caches, DB enums). Map dependencies in `arch_tree.md`.
   - Establish baseline metrics: RMSE/sMAPE per model, ensemble RMSE ratios, live/paper PnL, turnover, and win rate. Persist to `METRICS_AND_EVALUATION.md`.
2) Observability & Data Hygiene
   - Ensure every model writes regression_metrics and audit summaries; instrument logs with summary keys and metric presence per model.
   - Expand logging for signal provenance (inputs, filters applied, pre/post normalization).
   - Fix schema blockers (e.g., add `ENSEMBLE` to model_type CHECK) before rollout; set DB busy_timeout to avoid lock flaps.
3) Forecasting Layer Cleanup
   - Standardize summaries keys (lowercase `garch`, etc.) and confidence derivation; prevent silent drops from missing keys or metrics.
   - Move model configuration to shared profiles (`config/model_profiles.yml`); prohibit ad-hoc orders.
   - Add deterministic canary toggles: `forecasting.canary.enabled`, per-ticker canary allowlist, shadow forecasts persisted for comparison.
4) Signal Library Refactor
   - Define a `signals/` registry: each signal exposes `compute(df) -> series`, metadata, and risk filters; remove LLM/proprietary signal code paths.
   - Implement canonical signals first (SMA/EMA crossover, z-score mean reversion, ATR/vol breakout, cross-sectional momentum, volume/liquidity filter).
   - Add signal quality metrics (stability, hit-rate, turnover, cost-adjusted PnL) and log them to audit artifacts.
5) Ensemble & Routing Hardening
   - Require confidence dict coverage for all active models; ensure DB persistence works for ENSEMBLE outputs; assert confidence spread (no 0.9999 saturation).
   - Add shadow-mode ensemble: old vs. new weights side-by-side; emit comparison artifacts for gatekeepers; verify selected model matches best holdout RMSE when ensemble disabled.
6) Canary Rollout & Validation
   - Run new models/signals in shadow; compare against baseline with fixed horizons. Promotion only if: RMSE ratio improves or stays within +2%, PnL improves by >=1% annualized, and risk metrics (max DD, VaR proxy) are no worse.
   - Document decisions in `implementation_checkpoint.md` and `PROJECT_WIDE_OPTIMIZATION_ROADMAP.md`; include evidence files.
7) Decommission Legacy Artifacts
   - Remove unused proprietary/LLM signal code and configs once canary passes. Drop stale model artifacts and caches after a retention window.
   - Update schema enums and tests; ensure rollbacks remain possible via feature flags.

## GARCH Positioning (industry-aligned)
- Prefer GARCH as volatility/risk input: drive position sizing, stop widths, regime flags. Keep mean ensemble focused on directional models (SARIMAX/ETS/SSA). This avoids fake regression_metrics for variance forecasts.
- If GARCH must stay in mean ensemble: ensure comparable mean forecasts and holdout metrics; never synthesize confidence from unrelated AIC/BIC.
- Always lowercase keys internally; map to uppercase enums only when persisting to DB/logs.

## Confidence/Selection Fixes (concrete actions)
- Inspect `_score_from_metrics()` and coordinator selection for argmax-only behavior; require smooth score mapping (relative RMSE ratio → bounded spread, avoid clipping at 0.9999).
- Enforce monotonicity test: lower RMSE should yield higher score; add unit test that asserts all enabled candidates appear in confidence and that scores are not all identical/saturated.
- Validate forecast alignment (index/horizon/scale) before scoring; fail loud on mismatches.

## Deliverables Checklist
- Updated configs enabling canary toggles and registry entries for institutional signals/models.
- Logging/audit additions for model summaries, signal provenance, and ensemble confidence inputs.
- Migration script for DB CHECK constraints (including `ENSEMBLE`).
- Documentation updates: `METRICS_AND_EVALUATION.md` (baseline + post-canary), `implementation_checkpoint.md` (decisions), `CORE_PROJECT_DOCUMENTATION.md` (architecture delta).

## Success Criteria
- Confidence dict always includes all active models; no silent exclusions or score saturation; monotonicity tests pass.
- Ensemble RMSE ratio <= baseline (target <=1.10x) on canary runs; PnL > baseline by >=1% annualized with unchanged or lower drawdown.
- GARCH and SARIMAX hyperparameters remain data-learned; no manual orders introduced; GARCH primarily used for volatility unless explicitly justified as mean forecaster.
- Legacy/LLM signal paths removed or fully feature-flagged off without breaking existing pipelines.

## Neural Forecaster Integration (GPU-aligned, 1-hour horizon)
- Horizon: 1-hour forecasts; targets: returns/direction with volatility and cross-sectional features; also maintain price-only mode for ablations.
- Training cadence: real-time incremental retrain plus daily batch refresh; keep canary/shadow enabled before promotion.
- GPU profile (local): RTX 4060 Ti 16GB, CUDA 12.9, low-util headroom (per GPU_PARALLEL_RUNNER_CHECKLIST.md); default GPU_LIST=0.
- Stack: PatchTST or NHITS via Nixtla NeuralForecast for mean/directional; skforecast + XGBoost GPU for feature-driven directional edge; Chronos-Bolt as read-only zero-shot benchmark; GARCH kept for volatility sizing/stops.
- Feature sets: (a) price-only; (b) price + volume + realized vol + cross-sectional factors (e.g., rank returns, liquidity filters). Must be config-driven and logged with provenance.
- Orchestration: enable through feature flags in `forecasting.canary` and GPU runner configs; align shard/GPU mapping with `Documentation/GPU_PARALLEL_RUNNER_CHECKLIST.md` and keep ETL/routing on CPU. Ensure busy_timeout set on DB connections for long GPU runs.
