# Signal Quality and Data Enrichment Review and Plan (2026-02-25)

## Scope

This document captures the current signal-quality bottlenecks and a concrete implementation plan for feature/data enhancement and enrichment.

Goals:

1. Increase directional signal quality without introducing lookahead bias.
2. Remove silent feature degradation paths.
3. Unblock audit-gate evidence accumulation.
4. Enforce delivery discipline: every new implementation ships with targeted unit tests and regression coverage.

## Layer 1: Data Sources (Current Gaps)

| Source | Status | Signal Impact |
|---|---|---|
| OHLCV (yfinance daily) | Active | Primary source, stable quality |
| OHLCV intraday (1h) | Active | Used in audit sprint / intraday checks |
| Cross-sectional OHLCV | Degraded | Neutral fallback is frequently used because many runs are single-ticker |
| Macro (yield curve, VIX, CPI) | Not started | Documented but not wired into production feature flow |
| Sentiment (GDELT, FinBERT) | Scaffolded, disabled | `config/sentiment.yml` exists but gate-locked |
| Alternative data | Not started | Only specification-level coverage in `Documentation/` |

Key gap:

- Cross-sectional context is often missing during overnight runs, so rank/z-score features lose discriminative power.

## Layer 2: Feature Quality (Current State)

Canonical implementation path:

- `etl/time_series_feature_builder.py`

Features that fire reliably:

- Price lags: 1, 5, 10, 20
- Return lags and rolling mean/std/skew windows: 5, 10, 20, 60
- Drift and regime proxies: `drift_intensity`, `vol_regime_flag`
- Tail-risk proxies: `downside_vol_20`, `drawdown_depth_60`, `cvar_proxy_95`
- Microstructure proxy: ATR-style rolling true range
- Seasonal decomposition when enough history is available
- Calendar flags

Features that silently degrade:

- `cross_sectional_rank_5d` falls back to `0.5` when multi-ticker context is absent (`etl/time_series_feature_builder.py:88` onward).
- `cross_sectional_zscore_20d` falls back to `0.0` for the same reason.
- Seasonal decomposition can be skipped when historical depth is insufficient; current behavior logs but does not fail hard.

## Layer 3: Configuration Quality (Post Phase 7.14)

Validated values:

- `confidence_threshold: 0.55`
- `min_expected_return: 0.0030` (30 bps)
- `max_risk_score: 0.70`
- `min_signal_to_noise: 1.5`

Observed positive chain:

- Better GARCH convergence and CI behavior improves SNR gating stability.

Remaining gap:

- Platt calibration remains under-powered due to insufficient `(confidence, outcome)` pairs in production evidence windows.
- Confidence remains materially above realized win rate until proof-runway accumulation completes.

## Layer 4: Audit Gate Structural Bottleneck

Current dedupe behavior:

- `scripts/check_forecast_audits.py` deduplicates by:
  - `dataset.start`
  - `dataset.end`
  - `dataset.length`
  - `dataset.forecast_horizon`

Implication:

- Re-running the same fixed date window repeatedly does not increase unique effective audit windows.
- With `holding_period_audits: 20`, a fixed-range overnight routine can stall below required unique-window evidence.

Canonical fix pattern:

- Use AS-OF date diversification from `bash/run_20_audit_sprint.sh:293-300`.
- Vary effective dataset windows per run to increase unique dedupe keys.

## Implementation Plan

## Workstream A: Data Enrichment and Coverage

1. Cross-sectional run mode:
   - Ensure nightly jobs include a multi-ticker cohort path for feature generation.
   - Do not rely on single-ticker-only batches for model-selection evidence.
2. Macro enrichment:
   - Add feature ingestion path for VIX, yield-curve slope, CPI surprise proxy.
   - Gate with config flags and availability checks.
3. Sentiment activation path:
   - Keep feature-flagged rollout from `config/sentiment.yml`.
   - Add explicit quality thresholds (coverage %, freshness, source health) before enabling.

Acceptance criteria:

- Cross-sectional fallback rate below threshold (target: <20% of scored rows).
- Macro features present in persisted feature snapshots for enabled runs.
- Sentiment remains disabled unless quality gates pass.

## Workstream B: Feature Quality Hardening

1. Add explicit feature-health metrics to logs/artifacts:
   - Fallback counters for cross-sectional features.
   - Seasonal decomposition availability rate.
2. Promote silent degradation to visible diagnostics:
   - Warn when fallback ratio breaches threshold.
   - Fail fast in strict modes when critical features are absent.
3. Preserve backward compatibility:
   - Keep neutral fallbacks for runtime continuity, but expose hard telemetry.

Acceptance criteria:

- Every run emits feature-health summary.
- Alerting rule exists for prolonged cross-sectional neutralization.

## Workstream C: Calibration and Confidence Reliability

1. Expand confidence-outcome pair collection in production-like runs.
2. Keep Platt calibration gated until minimum sample quality is met.
3. Track confidence-to-win-rate drift as a first-class metric.

Acceptance criteria:

- Calibration dataset count and freshness visible in run summary.
- Confidence calibration error trend is non-worsening across release windows.

## Workstream D: Audit Evidence Through Window Diversification

1. Default nightly audit routines to AS-OF diversified windows.
2. Keep fixed-window mode only for controlled reproducibility checks.
3. Record dedupe-key cardinality per run in audit output.

Acceptance criteria:

- Unique-window count grows with each diversified run.
- `holding_period_audits` requirements can be satisfied without synthetic duplication.

## Test and Validation Plan (Mandatory DevOps Gate)

For every new implementation in these workstreams:

1. Run targeted unit tests for touched modules.
2. Run fast regression lane:
   - `pytest -m "not gpu and not slow"`
3. Run compile/smoke checks for touched entry points.
4. Record exact commands and outcomes in delivery notes.

Minimum evidence per change:

- Unit tests for new/changed logic.
- Regression lane result.
- No hidden fallback regressions in feature-health telemetry.

## Rollout Sequence

1. Workstream D (audit-window diversification) first, to unblock evidence accumulation.
2. Workstream B (feature health observability), so degradations become measurable.
3. Workstream A (macro/sentiment/coverage enrichment), staged behind flags.
4. Workstream C (calibration refinement), once enough quality data exists.

## Risks and Controls

| Risk | Control |
|---|---|
| Overfitting from feature expansion | Walk-forward validation + strict holdout discipline |
| Noisy external data | Source quality gates and config feature flags |
| False confidence improvements | Confidence-vs-realized calibration monitoring |
| Audit inflation from duplicates | Dataset-window dedupe visibility and AS-OF diversification |

## Execution Checklist

- [ ] Enable AS-OF diversified audit routine in nightly workflow.
- [ ] Add feature-health fallback telemetry for cross-sectional and seasonal features.
- [ ] Add/refresh unit tests for each implementation path.
- [ ] Run regression lane before every merge.
- [ ] Update project status snapshot after first completed diversified audit cycle.
