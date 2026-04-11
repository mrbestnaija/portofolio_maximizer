# Portfolio Maximizer – Autonomous Profit Engine

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Phase 11 Active](https://img.shields.io/badge/Phase%2011-Active-blue.svg)](Documentation/)
[![Fast Lane: 2355 passing](https://img.shields.io/badge/fast%20lane-2355%20passing-success.svg)](tests/)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-informational.svg)](Documentation/)
[![Research Ready](https://img.shields.io/badge/research-reproducible-purple.svg)](#-research--reproducibility)

> End-to-end quantitative automation that ingests data, forecasts regimes, routes signals, and executes trades hands-free with profit as the north star.

> Canonical objective policy: [Documentation/REPO_WIDE_MATRIX_FIRST_REMEDIATION_2026-04-08.md](Documentation/REPO_WIDE_MATRIX_FIRST_REMEDIATION_2026-04-08.md)
> **Barbell asymmetry is the primary economic objective. The system optimizes for asymmetric upside with bounded downside (omega_ratio > 1 vs NGN 31% annual hurdle), not for symmetric textbook efficiency metrics.**

**Version**: 4.5
**Status**: Phase 11 active — Nigeria production path, lot-aware close linkage, domain-calibrated barbell objective
**Last Updated**: 2026-04-11

## Contributing

Contribution policy lives in [CONTRIBUTING.md](CONTRIBUTING.md).
Telemetry changes must follow the Evidence Integrity Contract (schema version bump + adversarial coverage update).

## Current Repo Truth (2026-04-11)

- **Canonical objective**: `domain_utility` barbell mode — omega_ratio vs NGN 31% annual hurdle
  (`DAILY_NGN_THRESHOLD = (1.31)^(1/252)-1 ≈ 0.00108/day`) is the primary metric. Win rate is
  a diagnostic, not a gate. See `Documentation/REPO_WIDE_MATRIX_FIRST_REMEDIATION_2026-04-08.md`.

- **Live funnel: lot-aware close linkage (commit 521c37e)** — `trade_close_allocations` table
  + `trade_close_linkages` view enable a single closing trade to link multiple openers (e.g. NVDA
  close id=322 against lots 316+317). All gate queries route through the view. `repair_unlinked_closes.py`
  hardened to reject synthetic-ancestry matches. `PaperTradingEngine` does FIFO lot matching
  at close time and persists allocation rows automatically.

- **Gate state (2026-04-11)**:
  - `ci_integrity_gate`: **PASS** — CROSS_MODE_CONTAMINATION fixed; only untagged synthetic-opener
    closes are flagged (tagged closes already excluded from `production_closed_trades`).
  - `check_quant_validation_health`: **PASS** (728 PASS, 0 FAIL)
  - `production_audit_gate`: **PASS** (INCONCLUSIVE_ALLOWED) — warmup until 2026-04-24;
    `Phase3 ready=True`, `matched=2/2`, `phase3_reason=READY`
  - `institutional_unattended_gate`: FAIL — `prior_gate_execution` latch only;
    `WARMUP_COVERED_PASS ≠ GENUINE_PASS` by design (resolves at warmup expiry + real lift evidence)
  - `overall_passed`: False — institutional latch; not a live-funnel failure

- **Default tickers**: `AMZN,NVDA,MSFT` — fastest-closing live evidence candidates.
  NVDA avg hold 0.25d; AMZN has clean producer-native NOT_DUE open evidence.

- **DCR (Domain-Calibrated Remediation) phases 1-3 complete**:
  - P1: missing-baseline bypass, residual enforcement, diagnostics_score pessimistic fallback,
    GARCH variance floor 1e-12→1e-6, funnel audit logging
  - P2-B: CV OOS proxy; P3-A: confidence calibration script; P3-B: MSSA-RL neutral-on-low-support
  - Remaining heuristic distortion fixes (C5 OOS scan cap, C3 SAMoSSA bump, H6 SNR fallback,
    H7 RMSE-rank silent disable, M1 EWMA convergence_ok, H2 realized_vol floor): pending

- **Phase 11-A (Nigeria math extension)**: `omega_ratio()`, `fractional_kelly_fat_tail()`,
  `effective_ngn_return()`, `portfolio_metrics_ngn()` in `etl/portfolio_math.py`.
  Phases B-E (fx_layer, broker_cost_model, oanda executor) blocked until GENUINE_PASS.

- **Adversarial suite barbell extension (commit 58b07a1)**: 4 new NGN-domain scenarios
  (ngn_high_inflation, asymmetric_vol, fat_tail_crash, crisis_recovery), `compute_barbell_per_run`,
  `summarize_barbell`, `evaluate_barbell_thresholds`. Denominator false-PASS bug fixed. 4→47 tests.

- **Funnel audit observability**: `logs/funnel_audit.jsonl` — 68 entries; 60 SNR_GATE (MSFT, conf≈0.37,
  SNR≈0.79 < 1.5 threshold), 8 QUANT_VALIDATION_FAIL. SNR_GATE blocks are the primary routing
  bottleneck. Confidence below 0.55 floor is the secondary.

- **Latest verification**:
  - `python -m pytest tests/ -m "not slow and not gpu and not integration" -q`
  - Result: **2355 passed, 1 skipped, 10 xfailed** (2026-04-11)
  - `python scripts/ci_integrity_gate.py` → `[PASS] All integrity checks passed.`
  - `python scripts/production_audit_gate.py --unattended-profile` → `PASS, Phase3 ready=1`

Historical sections below preserve earlier phase notes for chronology; use this section as the
current repo baseline.

---

## 🎯 Overview

Portfolio Maximizer is a self-directed trading stack that marries institutional-grade ETL with autonomous execution. It continuously extracts, validates, preprocesses, forecasts, and trades financial time series so profit-focused decisions are generated without human babysitting.

### Current Phase & Scope (Mar 2026)

**Phase 10 Complete** - SARIMAX Re-enable, RMSE-Rank Hybrid Confidence, Production Gate Unblock:

- **SARIMAX re-enabled**: `sarimax_enabled: true` by default. Adds ARIMA/state-space
  class diversity orthogonal to GARCH, spectral (SAMoSSA), and RL (MSSA-RL) models.
  Validator (`scripts/validate_forecasting_configs.py`) updated to accept `enabled: true`
  and to skip sarimax-in-regime-candidates check when SARIMAX is enabled.
- **Hybrid RMSE-rank scoring**: `forcester_ts/ensemble.py` computes rank-normalized
  RMSE scores per model and injects them into `_combine_scores()`. Prevents SAMoSSA's
  always-high EVR (~1.0 by SSA construction) from inflating confidence regardless of
  actual forecast accuracy.
- **15-candidate ensemble**: expanded from 10 with SARIMAX-anchored positions (1-2),
  MSSA-RL elevated (3-4), and a single-model SARIMAX anchor (15).
- **Production audit gate unblocked**: EVIDENCE_HYGIENE_FAIL and THIN_LINKAGE cleared;
  early-credit bypass for already-closed trades eliminates NOT_DUE procrastination.
  THIN_LINKAGE warmup provision (30-day window, 1-match floor) prevents accumulation
  phase from hard-blocking CI. Gate: `phase3_reason=GATES_FAIL, matched=1/1`.
- **OpenClaw gateway restored**: fixed placeholder remote URL; switched to local mode.
  WhatsApp + Telegram channels online. 22 cron jobs operational.

**Phase 9 (Binary Directional Classifier)** — also complete:

- **Binary Directional Classifier**: `P(price_up_in_N_bars)` trained offline via `scripts/train_directional_classifier.py`. CalibratedClassifierCV (Platt/sigmoid) wraps a LogisticRegression pipeline with walk-forward CV hyperparameter selection. Schema v2 pkl with `feature_names` persisted for inference-time mismatch detection.
- **Parquet-scan labeler**: `scripts/generate_classifier_training_labels.py` scans price parquets directly instead of relying on JSONL timestamps, solving the wall-clock alignment gap that blocked label generation.
- **Pre-flight validator**: `scripts/validate_pipeline_inputs.py` — V1-V6 checks (filename convention, parquet coverage, JSONL alignment, eval date coverage, duplicate-parquet collision, edge cases). CLI returns exit 0/1 for CI gating.
- **Overnight bootstrap**: `bash/overnight_classifier_bootstrap.ps1` — 5-phase PS1 pipeline (pre-flight, label gen, train, A/B eval, report) with exit-code capture, ASCII-safe logging, and auto execution-mode.
- **Evaluation harness**: `scripts/evaluate_directional_classifier.py` produces ECE (10-bin), walk-forward DA, gate-lift counterfactual, and feature importance. Artifacts written to `visualizations/directional_eval.txt`.
- **Phase 9 metrics (2026-03-18)**: DA=0.562 (PASS), ECE=0.075 (PASS), gate-lift=-0.025 (WARN — gate at p_up>0.55 slightly under-selects; 290 labeled examples, AAPL).

**Phase 7.9 Complete** - PnL Integrity Enforcement, Adversarial Audit, OpenClaw Automation:

- **PnL Integrity Framework**: Database-level constraints preventing double-counting, orphaned positions, and diagnostic contamination
- **Adversarial Audit**: 10-finding stress test revealing 94.2% quant FAIL rate, broken confidence calibration, ensemble underperformance
- **Forecast Audit Gate**: PASS (21.4% violation rate, 28 effective audits, threshold 25%)
- **OpenClaw Cron Automation**: 9 audit-aligned cron jobs (P0-P2 priority) with real script execution via agentTurn mode
- **Interactions API**: Security-hardened FastAPI with auth mode enforcement (JWT/API-key/any), CORS, rate limiting
- **3-Model Local LLM**: deepseek-r1:8b (fast reasoning), deepseek-r1:32b (heavy reasoning), qwen3:8b (tool orchestrator)

**Production Metrics (2026-02-14)**:
- 37 round-trips, $673.22 total PnL, 43.2% win rate, 1.85 profit factor
- Integrity: ALL PASSED (0 violations with whitelist)
- System survives on magnitude asymmetry (avg win $91.59 vs avg loss $34.54 = 2.65x ratio)

**System Architecture**:
- Regime-aware ensemble routing with adaptive model selection
- 4 forecasting models: SARIMAX (off by default), GARCH, SAMOSSA, MSSA-RL
- Quantile-based confidence calibration (Phase 7.4)
- PnL integrity enforcement with canonical views (Phase 7.9)
- OpenClaw-driven monitoring and notifications
- SARIMAX disabled by default for 15x single-forecast speedup

### Key Features

- **🚀 Intelligent Caching**: 20x speedup with cache-first strategy (24h validity)
- **📊 Advanced Analysis**: MIT-standard time series analysis (ADF, ACF/PACF, stationarity)
- **📈 Publication-Quality Visualizations**: 8 professional plots with 150 DPI quality
- **🔄 Robust ETL Pipeline**: 4-stage pipeline with comprehensive validation
- **✅ Comprehensive Testing**: 2355+ tests with high coverage across ETL, LLM, forecaster, execution, integrity, and security modules
- **⚡ High Performance**: Vectorized operations, Parquet format (10x faster than CSV)
- **🧠 Modular Orchestration**: Dataclass-driven pipeline runner coordinating CV splits, neural/TS stages, and ticker discovery with auditable logging
- **🔐 Resilient Data Access**: Hardened Yahoo Finance extraction with pooling to reduce transient failures
- **🤖 Autonomous Profit Engine**: `scripts/run_auto_trader.py` keeps the signal router + trading engine firing so positions are sized and executed automatically

---

### Latest Enhancements (Apr 2026)

**Phase 11-A + Live Funnel Integrity (2026-04-11)**:

- **Lot-aware close linkage** (`etl/database_manager.py`, `execution/paper_trading_engine.py`):
  New `trade_close_allocations` table and `trade_close_linkages` UNION-ALL view allow one closing
  trade to reference multiple openers. `PaperTradingEngine` does FIFO lot matching via
  `_build_close_allocations()` and persists rows via `save_trade_close_allocations()`.
  Legacy scalar `entry_trade_id` closes still work via the view fallback arm.
  `load_open_entry_lots()` reconstructs the lot map on resume, filtering `is_synthetic=0`.
- **repair_unlinked_closes.py hardened**: rejects synthetic-ancestry matches; fail-closes unless
  the close matches a live lot from the current position run. Prevented fake-green repair of
  close 322 to stale synthetic opener 211; correctly wrote 322 → {316, 317} allocations.
- **CROSS_MODE_CONTAMINATION gate fix**: `_check_cross_mode_contamination()` now flags only
  untagged closes (is_contaminated=0 with synthetic opener). Tagged closes are already excluded
  from `production_closed_trades` — no manual whitelist needed for future contaminated closes.
- **Resume backfill synthetic filter**: `AND COALESCE(is_synthetic, 0) = 0` in the entry_trade_id
  backfill query prevents a later synthetic opener from shadowing the real live opener on resume,
  which would cause the close to be tagged contaminated and excluded from THIN_LINKAGE evidence.
- **Win-rate quick-fail removed** from auto-trader cycle loop: barbell runs with 10-15% WR and
  high PF are valid; only PF < 0.5 retains action-required status.
- **Barbell adversarial suite** (`scripts/run_adversarial_forecaster_suite.py`): 4 new
  NGN-domain scenarios, `compute_barbell_per_run`, `summarize_barbell`, `evaluate_barbell_thresholds`.
  Denominator false-PASS bug (max(1, runs-errors)) fixed to produce `nan`. 4→47 tests.
- **Nigeria math layer** (`etl/portfolio_math.py`): `omega_ratio()`, `fractional_kelly_fat_tail()`,
  `effective_ngn_return()`, `portfolio_metrics_ngn()` with `NGN_ANNUAL_INFLATION=0.28`,
  `NGN_P2P_FRICTION=0.03`, `DAILY_NGN_THRESHOLD`. 39 tests. No existing functions modified.
- **Default tickers updated**: `AMZN,NVDA,MSFT` — fastest historical close speeds and clean
  producer-native NOT_DUE open evidence (ids 315, 316/317).
- **Regression baseline**: 2355 passed, 1 skipped, 10 xfailed (fast lane, 2026-04-11).

**Phase 10c + DCR (2026-03-30 to 2026-04-05)**:

- **OOS selector wiring** (`forcester_ts/ensemble.py`, `forcester_ts/forecaster.py`):
  `derive_model_confidence` now accepts trailing OOS metrics from audit files scoped to current
  ticker + horizon. RMSE-rank hybrid scoring and DA extraction are active in live auto_trader runs.
- **Heuristic distortion cleanup**: `_change_point_boost` capped at 0.20; MSSA-RL hard floor
  `max(mssa_score, 0.40)` removed; `CONFIDENCE_ACCURACY_CAP=0.65` moved post-selection.
- **DCR P1**: missing-baseline bypass fixed; residual diagnostics enforced by model type;
  `diagnostics_score` defaults 0.5→0.0 (pessimistic); GARCH EWMA variance floor 1e-12→1e-6.
- **DCR P2-B**: CV OOS proxy from fold metrics. **DCR P3-A**: confidence calibration script.
  **DCR P3-B**: MSSA-RL neutral-on-low-support guard.
- **Gate**: PASS (semantics=PASS, 33.33% violation rate, warmup expired 2026-04-24).

**Phase 9: Binary Directional Classifier + Pre-Flight Validator (2026-03-18)**:

- **Directional classifier pipeline** (13 commits, fully production-ready):
  - `scripts/generate_classifier_training_labels.py`: parquet-scan labeler; scans `data/checkpoints/*data_extraction*.parquet` files, applies N-bar forward-return threshold to assign BUY/SELL/HOLD labels. Same-parquet collision guard (`--auto-parquet` + multi-ticker) prevents meaningless training from shared synthetic price files.
  - `scripts/train_directional_classifier.py`: walk-forward TimeSeriesSplit CV for C selection; final model wrapped in `CalibratedClassifierCV(method='sigmoid', cv=2|3)` (Platt scaling). Saves `data/classifiers/directional_v1.pkl` + `.meta.json` (schema v2: `feature_names`, `calibration_method`, `schema_version=2`).
  - `forcester_ts/directional_classifier.py`: lazy-load inference wrapper with feature-name mismatch guard — if `meta["feature_names"]` diverges from current `_FEATURE_NAMES`, scoring is disabled with an ERROR log rather than silently mis-mapping coefficients.
  - `scripts/evaluate_directional_classifier.py`: ECE (10-bin calibration error), walk-forward DA, gate-lift counterfactual (gated WR vs baseline WR at configurable `p_up_threshold`), feature importance from calibration fold coefs. Report written to `visualizations/directional_eval.txt`.
  - `bash/overnight_classifier_bootstrap.ps1`: 5-phase PowerShell pipeline: pre-flight (V1-V6), label generation, training, A/B holdout evaluation across configurable eval dates, structured Phase 5 report. ASCII-safe logging, `$LASTEXITCODE` captured before PS resets it, execution-mode `auto` (not synthetic).
- **Pre-flight validator** (`scripts/validate_pipeline_inputs.py`):
  - V1: filename convention (ticker-named vs unnamed fallback vs missing)
  - V2: parquet coverage map (Close column, min length 100, constant-price synthetic detection)
  - V3: JSONL timestamp alignment — WARN (not FAIL) when 0% align, advisory that parquet-scan path is unaffected
  - V4: eval date coverage (date within parquet range for each ticker)
  - V5: duplicate-parquet collision (multi-ticker same-file = synthetic contamination)
  - V6: edge cases (empty parquets, null JSONL timestamps, stale training dataset, missing checkpoint dir)
  - CLI: `--json` for machine-readable output; exits 0/1 for CI gating; `run_all_checks()` callable from other scripts
- **Tests**: 37 new tests (`tests/scripts/test_validate_pipeline_inputs.py` 26 tests + `tests/scripts/test_train_directional_classifier.py` 11 tests including D4 Platt calibration, schema v2, feature-name mismatch)
- **Regression baseline**: **1916 passed, 6 skipped, 12 xfailed** (2026-03-18)
- **Phase 9 classifier metrics** (AAPL, 290 labeled examples): DA=0.562 (PASS >0.52), ECE=0.075 (PASS <0.10), gate-lift=-0.025 (WARN — accumulate more labeled examples to lift)

**Phase 7.35: Outcome Linkage Denominator + Causality Hardening (2026-03-06)**:

- **Ticker-aware outcome dedupe (P0)**:
  - `scripts/check_forecast_audits.py` now keeps separate dedupe strategies:
    - RMSE gate: `(start, end, length, forecast_horizon)`
    - Outcome linkage: `(ticker, start, end, length, forecast_horizon)`
  - Prevents cross-ticker evidence suppression in linkage denominators.
- **Eligibility anchor fixed to signal context (P0)**:
  - Added `compute_expected_close(signal_context, dataset)` and made signal-context timing primary:
    - `expected_close = entry_ts + signal_context.forecast_horizon`
    - dataset fallback only when signal context is absent.
- **Fail-closed context integrity guards (P0)**:
  - Added explicit `INVALID_CONTEXT` classification and reasons:
    - `CAUSALITY_VIOLATION` (`expected_close < entry_ts`)
    - `HORIZON_MISMATCH` (dataset horizon != signal horizon)
    - `MISSING_SIGNAL_ID`, `EXPECTED_CLOSE_UNAVAILABLE`, `AMBIGUOUS_MATCH`
  - Added explicit `NOT_DUE` status for open outcome windows (prevents false `OUTCOME_MISSING`).
- **Linkage denominator truth metrics (P1)**:
  - `scripts/outcome_linkage_attribution_report.py` now reports:
    - `linked_ts_trades / total_ts_trades`
    - `ts_trade_coverage = total_ts_trades / total_closed_trades`
  - Keeps global linkage ratio while separating legacy-ID denominator drag.
- **Adversarial anti-regression coverage (P2)**:
  - Added telemetry contract checks in `scripts/adversarial_diagnostic_runner.py`:
    - `TCON-06` outcome ticker-aware dedupe enforcement
    - `TCON-07` signal-anchored expected-close enforcement
    - `TCON-08` explicit `NOT_DUE` status enforcement
  - Added test coverage in:
    - `tests/scripts/test_check_forecast_audits.py`
    - `tests/scripts/test_outcome_linkage_attribution_report.py`
    - `tests/scripts/test_adversarial_diagnostic_runner.py`

**Phase 7.34: Paranoid Hardening (2026-03-05)**:

- **Data sufficiency fail-closed fixes**:
  - `scripts/data_sufficiency_monitor.py` now treats `profit_factor=0.0` as below the R3 gate (no threshold dodge path).
  - Non-finite numeric metrics (`NaN`, `Inf`) are classified as `DATA_ERROR`.
  - CLI exit codes are now contract-stable: `0=SUFFICIENT`, `1=INSUFFICIENT`, `2=DATA_ERROR` for both JSON and text modes.
- **Visualization truthfulness hardening**:
  - `scripts/generate_performance_charts.py` now validates required chart artifacts after generation.
  - In strict mode, missing chart files produce `chart_missing:<name>` errors and force `status=ERROR`.
  - `scripts/dashboard_db_bridge.py` now validates chart paths before reporting robustness `OK`; missing chart artifacts downgrade robustness to `WARN`.
- **Wiring and short-circuit hardening**:
  - `scripts/run_quality_pipeline.py` now passes one precomputed sufficiency snapshot into chart generation to avoid duplicated evaluation drift.
  - Eligibility/context/chart-stage surfaced errors now escalate pipeline status to `ERROR`.
  - `scripts/compute_ticker_eligibility.py` now surfaces DB/query failures via explicit `errors` and `eligibility_query_error` warnings.
- **OpenClaw host enforcement consistency**:
  - `scripts/run_openclaw_maintenance.ps1` now runs `scripts/enforce_openclaw_exec_environment.py` on both Windows and WSL paths before maintenance execution.
  - `scripts/project_runtime_status.py` now emits explicit exec-environment signals for invalid `tools.exec.host`, sandbox mode drift, and missing ACP default agent.
- **OpenClaw notification storm guard hardening**:
  - `utils/openclaw_cli.py` now applies persistent adaptive cooldown for repeated transport failures (DNS/listener/timeout/gateway churn) instead of repeatedly attempting sends.
  - New controls: `OPENCLAW_STORM_GUARD_ENABLED`, `OPENCLAW_STORM_BASE_COOLDOWN_SECONDS`, `OPENCLAW_STORM_MAX_COOLDOWN_SECONDS`, `OPENCLAW_STORM_BACKOFF_MULTIPLIER`, `OPENCLAW_STORM_RESET_WINDOW_SECONDS`.
- **Verification evidence (2026-03-05)**:
  - `python scripts/adversarial_diagnostic_runner.py --json --severity LOW --fix-report`
  - `python -m pytest tests/scripts/test_data_sufficiency_monitor.py tests/scripts/test_generate_performance_charts.py tests/scripts/test_run_quality_pipeline.py tests/scripts/test_dashboard_db_bridge.py -q`
  - `python -m pytest -m "not gpu and not slow" --tb=short -q`
  - Latest fast-lane result in this cycle: `1590 passed, 3 skipped, 28 deselected, 7 xfailed`.

**Phase 7.17: Ensemble Health Audit & Adaptive Weighting (2026-03-01)**:

- **SAMOSSA DA=0 fix**: `_apply_da_cap()` in `forcester_ts/ensemble.py` caps and penalizes models with chronic near-zero directional accuracy; budget redistributed only to non-penalized models (DA >= da_floor)
- **`scripts/ensemble_health_audit.py`**: per-model RMSE/DA stats, proxy Shapley attribution, adaptive candidate weights (3 candidates: primary, top-2 hedge, pure winner), markdown report, GOLDEN_METRICS structured JSON log
- **`scripts/dedupe_audit_windows.py`**: SHA1 fingerprint deduplication of audit windows, exit-1 on duplicates
- **`scripts/exit_quality_audit.py`**: exit-reason breakdown (stop_loss / time_exit / signal_exit) to diagnose the 14pp forecast-DA → trade-WR gap
- **`scripts/check_model_improvement.py`**: unified 4-layer CLI — forecast quality, gate status, trade quality, and calibration in one command
- **Hypothesis property testing**: 8 property-based tests (300 + 250 examples each); found and fixed 3 bugs: redistribution-to-penalized-models, same bug in health audit, `round(4)` midpoint rounding violation
- **Regression**: 1332 passed, 1 skipped, 28 deselected, 7 xfailed

**Phase 7.9 Achievements**:

- **PnL Integrity Enforcement**: Database-level constraints (opening legs NULL PnL, entry_trade_id linkage, diagnostic/synthetic flags), canonical views (`production_closed_trades`, `round_trips`), CI gate
- **Adversarial Audit**: 10-finding stress test documented in [ADVERSARIAL_AUDIT_20260216.md](Documentation/ADVERSARIAL_AUDIT_20260216.md)
- **OpenClaw Cron Automation**: 9 priority-ranked cron jobs running real PMX scripts (P0: PnL integrity every 4h, production gate daily; P1: signal linkage, ticker health; P2: GARCH unit-root, overnight hold)
- **Tavily Search Integration**: `scripts/tavily_search.py` for quota-safe web grounding without Brave dependency
- **Interactions API**: FastAPI with auth mode enforcement (`jwt-only`/`api-key-only`/`any`), CORS, rate limiting, ngrok integration
- **3-Model Local LLM Stack**: deepseek-r1:8b + deepseek-r1:32b + qwen3:8b via Ollama with multi-model orchestration
- **Cross-session persistence**: portfolio_state + portfolio_cash_state tables via `--resume`
- **Proof mode**: Tight max_holding, ATR stops/targets, flatten-before-reverse for round-trip validation
- **SARIMAX off by default**: 15x single-forecast speedup (0.18s vs 2.74s)
- **Secrets leak guard**: Pre-commit hook + CI check preventing credential leaks

**Phase 7.8 Achievements**:

- All-regime weight optimization (3/6 regimes) with ~60-65% RMSE improvement for CRISIS/MODERATE_TRENDING
- SAMOSSA dominance finding: 72-90% across ALL optimized regimes
- Comprehensive documentation: [PHASE_7.8_RESULTS.md](Documentation/PHASE_7.8_RESULTS.md)

**Infrastructure Improvements**:
- Security hardening: secrets_guard pre-commit hook, API key rotation, credential validation
- SQLite read-only connections with immutable URI mode (WSL/DrvFS robustness)
- Concurrent process guard with lockfile + PID-based stale detection
- Adversarial test isolation with `_IsolatedConnection` wrapper (always rolls back)

## Verifying Model Improvement

Run all 4 measurement layers with one command:

```bash
python scripts/check_model_improvement.py
```

| Layer | What it measures | Green signal | Script |
|-------|-----------------|--------------|--------|
| 1 — Forecast Quality | Ensemble lift over best single model; SAMOSSA DA anomaly; data coverage | lift_global >= 5%, samossa_da_zero < 40%, n_used >= 50 | `ensemble_health_audit.py` |
| 2 — Gate Status | 4 CI gates: integrity, quant health, audit lift, institutional | overall_passed = True | `run_all_gates.py` |
| 3 — Trade Quality | Win rate, profit factor, exit-reason gap diagnosis | win_rate >= 45%, pf >= 1.3 | `exit_quality_audit.py` |
| 4 — Calibration | Platt scaling active tier, Brier score, ECE | tier = db_local/jsonl, brier < 0.25 | `platt_contract_audit.py` |

> **SKIP != PASS.** SKIP means "no measurement data available" (empty audit dir, DB not found,
> etc.). A SKIP layer provides no health signal.

Save a baseline before model changes and compare after:

```bash
python scripts/check_model_improvement.py --save-baseline logs/baseline_before.json
# ... make changes, run overnight pipeline ...
python scripts/check_model_improvement.py --baseline logs/baseline_before.json
```

Layer-specific runs:

```bash
python scripts/check_model_improvement.py --layer 1          # forecast quality only
python scripts/check_model_improvement.py --layer 3 --json   # trade quality as JSON
```

---

## Academic Rigor & Reproducibility (MIT-style)

- **Traceable artifacts**: Log config + commit hashes alongside experiment IDs; keep hashes for data snapshots and generated plots (`logs/artifacts_manifest.jsonl` when present).
- **Deterministic runs**: Set and record seeds (`PYTHONHASHSEED`, RNG, hyper-opt samplers, RL) for every reported experiment; prefer config overrides over ad hoc flags.
- **Executable evidence**: Each figure/table used for publication should have a runnable script/notebook (target: `reproducibility/` folder) that regenerates it from logged artifacts.
- **Transparency**: Document MTM assumptions, cost models, and cron wiring in experiment notes; link back to `Documentation/RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md` for the publication plan and replication checklist.
- **Archiving plan**: Package replication bundles (configs, logs, plots, minimal sample data) for Zenodo/Dataverse deposit before submitting any paper/thesis.

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Phase 7.8 Results](#-phase-78-results-all-regime-optimization)
- [Phase 7.9 Status](#-phase-79-cross-session-persistence--proof-mode)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Research & Reproducibility](#-research--reproducibility)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎖️ Phase 7.8 Results: All-Regime Optimization

### Key Results

**3/6 Regimes Optimized** with SAMOSSA-dominant weights:

| Regime | Samples | Folds | RMSE Before | RMSE After | Improvement | Optimal Weights |
|--------|---------|-------|-------------|------------|-------------|-----------------|
| **CRISIS** | 25 | 5 | 17.15 | 6.74 | **+60.69%** | 72% SAMOSSA, 23% SARIMAX, 5% MSSA-RL |
| **MODERATE_MIXED** | 20 | 4 | 17.63 | 16.52 | +6.30% | 73% SAMOSSA, 22% MSSA-RL, 5% SARIMAX |
| **MODERATE_TRENDING** | 50 | 10 | 20.86 | 7.29 | **+65.07%** | 90% SAMOSSA, 5% SARIMAX, 5% MSSA-RL |

### Major Finding: SAMOSSA Dominance

**SAMOSSA dominates ALL optimized regimes (72-90%)**, contradicting initial hypothesis that GARCH would be optimal for CRISIS regime.

- Pattern recognition outperforms volatility modeling across all market conditions
- CRISIS regime: SAMOSSA (72%) + SARIMAX (23%) provides best defensive configuration
- MODERATE_TRENDING: Confirms Phase 7.7 results with 2x sample size validation

### Configuration Updates

```yaml
# config/forecasting_config.yml (lines 98-115)
regime_candidate_weights:
  CRISIS:
    - {sarimax: 0.23, samossa: 0.72, mssa_rl: 0.05}
  MODERATE_MIXED:
    - {sarimax: 0.05, samossa: 0.73, mssa_rl: 0.22}
  MODERATE_TRENDING:
    - {sarimax: 0.05, samossa: 0.90, mssa_rl: 0.05}
```

### Regimes Not Optimized (Insufficient Samples)

| Regime | Reason | Recommendation |
|--------|--------|----------------|
| **HIGH_VOL_TRENDING** | Rare in AAPL 2024-2026 data | Test with NVDA (higher volatility) |
| **MODERATE_RANGEBOUND** | Rare in trending market | Use default weights |
| **LIQUID_RANGEBOUND** | Very rare (stable markets) | Use default weights |

**Full Results**: [Documentation/PHASE_7.8_RESULTS.md](Documentation/PHASE_7.8_RESULTS.md)

---

## 🚀 Phase 7.9: Complete (PnL Integrity & Automation)

### Objective

Establish reliable round-trip trade execution with cross-session persistence, PnL integrity enforcement, adversarial validation, and autonomous monitoring via OpenClaw.

### Final Status (2026-02-17)

- **Round-trips**: 37 validated, $673.22 total PnL, 43.2% win rate, 1.85 profit factor
- **Forecast audit gate**: PASS (21.4% violation rate, 28 effective audits, threshold 25%)
- **PnL integrity**: ALL PASSED (0 CRITICAL/HIGH violations)
- **OpenClaw cron**: 9 jobs active (P0-P2 priority, agentTurn mode)
- **Adversarial audit**: 10 findings documented, structural weaknesses identified

### Key Components

- **PnL Integrity Enforcer**: `integrity/pnl_integrity_enforcer.py` -- 6 integrity checks, canonical metrics, CI gate
- **Cross-session persistence**: `portfolio_state` + `portfolio_cash_state` tables via `--resume`
- **Proof mode** (`--proof-mode`): Tight max_holding (5d/6h), ATR stops/targets, flatten-before-reverse
- **Audit sprint**: `bash/run_20_audit_sprint.sh` with lockfile + gate enforcement
- **OpenClaw Cron**: 9 audit-aligned jobs running via `agentTurn` (P0 every 4h, P1 daily, P2 weekly)
- **Interactions API**: `scripts/pmx_interactions_api.py` with auth mode enforcement + ngrok tunnel
- **3-Model LLM**: deepseek-r1:8b/32b for reasoning, qwen3:8b for tool orchestration

### Validation Commands

```bash
# Run PnL integrity audit
python -m integrity.pnl_integrity_enforcer --db data/portfolio_maximizer.db

# Check production gate
python scripts/production_audit_gate.py

# Check canonical metrics (correct way)
python -c "
from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
with PnLIntegrityEnforcer('data/portfolio_maximizer.db') as e:
    m = e.get_canonical_metrics()
    print(f'Round-trips: {m.total_trades}, PnL: \${m.total_realized_pnl:+,.2f}, WR: {m.win_rate:.1%}')
"

# Check OpenClaw cron status
openclaw cron list
```

### Success Criteria

- [x] Cross-session position persistence working
- [x] Proof mode creates guaranteed round trips
- [x] UTC-aware timestamps across all layers
- [x] Forecast audit gate: PASS (28 audits, 21.4% violation rate)
- [x] PnL integrity enforcement deployed with CI gate
- [x] OpenClaw cron automation with audit-aligned jobs
- [x] Adversarial audit documented

### Phase 7.10: Production Hardening (Next)

Prerequisites:

- Address adversarial findings (94.2% quant FAIL rate, ensemble underperformance)
- Improve directional accuracy (currently below coin-flip at 41% WR)
- Fix confidence calibration (0.9+ confidence yields only 41% win rate)
- Widen proof-mode max_holding (5 -> 8-10 bars) for better risk/reward

---

## 🏗️ Architecture

### System Architecture (7 Layers)

```
┌─────────────────────────────────────────────────────────┐
│              Portfolio Maximizer                          │
│              Production-Ready System                     │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
  Layer 1:        Layer 2:         Layer 3:
  Extraction      Storage          Validation
  (yfinance &     (Parquet         (Quality
   multi-source)  Format)          Checks)
        │               │               │
        └───────────────┼───────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  Layer 4:        Layer 5:         Layer 6:
  Preprocessing   Organization     Analysis &
  (Transform)     (Train/Val/Test) Visualization
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
                   Layer 7:
                   Output
                   (Reports, Plots)
```

### Core Components

| Component | Lines | Purpose |
|-----------|-------|---------|
| **yfinance_extractor.py** | 327 | Yahoo Finance data extraction with intelligent caching |
| **data_validator.py** | 117 | Statistical validation and outlier detection |
| **preprocessor.py** | 101 | Missing data handling and normalization |
| **data_storage.py** | 158 | Parquet-based storage with train/val/test split |
| **portfolio_math.py** | 45 | Financial calculations (returns, volatility, Sharpe) |
| **time_series_analyzer.py** | 500+ | MIT-standard time series analysis |
| **visualizer.py** | 600+ | Publication-quality visualization engine |

---

## 🚀 Installation

### Prerequisites

- Python 3.10-3.12
- pip package manager
- Virtual environment (recommended)
- **Ollama** (optional, for LLM features): Local LLM server for market analysis, signal generation, and OpenClaw orchestration
  - Installation: `curl -s https://raw.githubusercontent.com/ollama/ollama/main/install.sh | sh`
  - Start server: `ollama serve`
  - Pull models: `ollama pull deepseek-r1:8b && ollama pull deepseek-r1:32b && ollama pull qwen3:8b`
  - See [OpenClaw Integration](Documentation/OPENCLAW_INTEGRATION.md) for the 3-model strategy

### Setup

```bash
# Clone the repository
git clone https://github.com/example-org/portofolio_maximizer.git
cd portofolio_maximizer

# Create virtual environment
python -m venv simpleTrader_env

# Activate virtual environment
# On Linux/Mac:
source simpleTrader_env/bin/activate
# On Windows:
simpleTrader_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# API Keys (if needed)
ALPHA_VANTAGE_API_KEY=your_key_here

# Cache Settings
CACHE_VALIDITY_HOURS=24
```

### LLM Integration (Optional and being phased out)

LLM/Ollama integration is disabled by default to avoid unnecessary startup delays when Ollama is not running. The roadmap prioritizes TS/GPU forecasters; if you still need the legacy LLM path for experiments, set `PM_ENABLE_OLLAMA=1` and use `--enable-llm` where supported.

---

## ⚡ Quick Start

### Run the ETL Pipeline

```bash
# Activate virtual environment
source simpleTrader_env/bin/activate

# Recommended: live run with automatic synthetic fallback
 python scripts/run_etl_pipeline.py \
   --tickers AAPL,MSFT \
   --include-frontier-tickers \
   --start 2020-01-01 \
   --end 2024-01-01 \
   --execution-mode auto

# Force live-only execution (fails fast on network/API issues)
python scripts/run_etl_pipeline.py --execution-mode live

# Offline validation (synthetic data, no network)
python scripts/run_etl_pipeline.py --execution-mode synthetic --include-frontier-tickers

# Expected output:
# ✓ Extraction complete (cache hit: <0.1s)
# ✓ Validation complete (0.1s)
# ✓ Preprocessing complete (0.2s)
# ✓ Storage complete (0.1s)
# Total time: varies with mode (synthetic ≈ 1s, live depends on APIs)

# Shortcut runner (auto mode with logs):
./bash/run_pipeline_live.sh
```

`--include-frontier-tickers` automatically adds the Nigeria → Bulgaria frontier symbols
curated in `etl/frontier_markets.py` (see `Documentation/arch_tree.md`) so every multi-ticker
training or validation run exercises less-liquid market scenarios. Synthetic mode is
recommended until provider-specific ticker mappings are finalized.

### Launch The Autonomous Trading Loop

```bash
python scripts/run_auto_trader.py \
  --tickers AAPL,MSFT,NVDA \
  --include-frontier-tickers \
  --lookback-days 365 \
  --forecast-horizon 30 \
  --initial-capital 25000 \
  --cycles 5 \
  --sleep-seconds 900
```

Auto-trader **resumes persisted positions by default** (`--resume` is on). Use `--no-resume` to start fresh from `--initial-capital`, or run `bash/reset_portfolio.sh` to clear the saved state. Existing databases should run the one-time migration: `python scripts/migrate_add_portfolio_state.py`.

For scheduled daily+intraday passes, use `bash/run_daily_trader.sh` (WSL/Linux) or `run_daily_trader.bat` (Windows Task Scheduler); both runs keep positions via `--resume`.

Add `--enable-llm` (plus `PM_ENABLE_OLLAMA=1`) to activate the legacy Ollama-backed fallback router whenever the ensemble hesitates. Each cycle:

1. Streams fresh OHLCV windows via `DataSourceManager` with cache-first failover.
2. Validates, imputes, and feeds the data into the SARIMAX/SAMOSSA/GARCH/MSSA-RL ensemble.
3. Routes the highest-confidence trade and executes it through `PaperTradingEngine`, tracking cash, PnL, and open positions in real time.

### Higher‑Order Hyper‑Parameter Optimization (Default Orchestration Mode)

For post‑implementation evaluation and regime‑aware tuning, the project includes a higher‑order
hyper‑parameter driver that wraps ETL → auto‑trader → strategy optimization in a stochastic loop.
This driver treats configuration knobs such as:

- Time window (`START` / `END` evaluation dates),
- Quant success `min_expected_profit`,
- Time Series `time_series.min_expected_return`

as higher‑order hyper‑parameters and searches over them non‑convexly using a bandit‑style
explore/exploit policy (30% explore / 70% exploit by default, dynamically adjusted).

The canonical entrypoint is:

```bash
# Run a 5‑round higher‑order hyper‑parameter search
HYPEROPT_ROUNDS=5 bash/bash/run_post_eval.sh
```

Each round:
- Generates temporary override configs (`config/quant_success_config.hyperopt.yml`,
  `config/signal_routing_config.hyperopt.yml`),
- Runs `scripts/run_etl_pipeline.py`, `scripts/run_auto_trader.py`,
  and `scripts/run_strategy_optimization.py` against a dedicated DB,
- Scores the run by realized `total_profit` over a short evaluation window,
- Logs trial parameters and scores to `logs/hyperopt/hyperopt_<RUN_ID>.log`,
- Maintains a 30/70 explore/exploit policy that slowly shifts toward exploitation
  as better configurations are discovered.

The best configuration is re‑run as `<RUN_ID>_best` and surfaced in
`visualizations/dashboard_data.json` so dashboards and downstream tools can treat it
as the current regime‑specific optimum (without hardcoding it in code).

### Analyze Dataset

```bash
# Run time series analysis on training data
python scripts/analyze_dataset.py \
    --dataset data/training/training_20251001_210734_20251001.parquet \
    --column Close \
    --output analysis_results.json

# Output: ADF test, ACF/PACF, statistical summary
```

### Generate Visualizations

```bash
# Create publication-quality plots
python scripts/visualize_dataset.py \
    --dataset data/training/training_20251001_210734_20251001.parquet \
    --column Close \
    --output-dir visualizations/

# Generates 8 plots:
# - Time series overview
# - Distribution analysis
# - ACF/PACF plots
# - Decomposition (trend/seasonal/residual)
# - Rolling statistics
# - Spectral density
# - Comprehensive dashboard
```

---

## 📖 Usage

### cTrader Credentials Precedence (Demo/Live)

The cTrader client resolves credentials in this order:

1. **Environment‑specific keys** (`CTRADER_DEMO_*` or `CTRADER_LIVE_*`)
2. **Generic keys** (`USERNAME_CTRADER` / `CTRADER_USERNAME`, `PASSWORD_CTRADER` / `CTRADER_PASSWORD`, `APPLICATION_NAME_CTRADER` / `CTRADER_APPLICATION_ID`)
3. **Email fallback** (`EMAIL_CTRADER` / `CTRADER_EMAIL`) if username is missing

This allows demo + live to run side‑by‑side without cross‑env leakage.

### 1. Data Extraction

```python
from etl.yfinance_extractor import YFinanceExtractor

# Initialize extractor
extractor = YFinanceExtractor()

# Extract data with intelligent caching
df = extractor.extract_data(
    ticker='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Cache hit: <0.1s, Cache miss: ~20s
```

### 2. Data Validation

```python
from etl.data_validator import DataValidator

# Initialize validator
validator = DataValidator()

# Validate data quality
validation_results = validator.validate_dataframe(df)

# Check for:
# - Price positivity
# - Volume non-negativity
# - Outliers (3σ threshold)
# - Missing data percentage
```

### 3. Data Preprocessing

```python
from etl.preprocessor import Preprocessor

# Initialize preprocessor
preprocessor = Preprocessor()

# Handle missing data
df_filled = preprocessor.handle_missing_data(df, method='forward')

# Normalize data (Z-score)
df_normalized = preprocessor.normalize_data(df_filled, method='zscore')
```

### 4. Time Series Analysis

```python
from etl.time_series_analyzer import TimeSeriesAnalyzer

# Initialize analyzer
analyzer = TimeSeriesAnalyzer()

# Run comprehensive analysis
results = analyzer.analyze(
    data=df['Close'],
    column_name='Close'
)

# Results include:
# - ADF test (stationarity)
# - ACF/PACF (autocorrelation)
# - Statistical summary (μ, σ², skewness, kurtosis)
```

### 5. Visualization

```python
from etl.visualizer import Visualizer

# Initialize visualizer
viz = Visualizer()

# Create comprehensive dashboard
viz.plot_comprehensive_dashboard(
    data=df,
    column='Close',
    save_path='visualizations/dashboard.png'
)

# Creates 8-panel publication-quality plot
```

---

## 📁 Project Structure

```
portfolio_maximizer/
│
├── config/                          # Configuration files (YAML)
│   ├── pipeline_config.yml          # Main pipeline orchestration
│   ├── forecasting_config.yml       # Model parameters + ensemble config
│   ├── llm_config.yml              # LLM integration (3-model strategy)
│   ├── quant_success_config.yml    # Trading success criteria
│   ├── signal_routing_config.yml   # Signal routing logic
│   └── yfinance_config.yml         # Yahoo Finance settings
│
├── data/                            # Data storage (organized by ETL stage)
│   ├── raw/                         # Original extracted data + cache
│   ├── training/                    # Training set (70%)
│   ├── validation/                  # Validation set (15%)
│   ├── testing/                     # Test set (15%)
│   └── portfolio_maximizer.db      # SQLite database
│
├── Documentation/                   # Comprehensive documentation (174 files)
│   ├── ADVERSARIAL_AUDIT_20260216.md # Current adversarial audit findings
│   ├── OPENCLAW_INTEGRATION.md     # OpenClaw + LLM + Interactions API
│   ├── EXIT_ELIGIBILITY_AND_PROOF_MODE.md # Proof-mode spec
│   └── PHASE_7.*.md               # Phase-specific documentation
│
├── etl/                             # ETL pipeline modules
│   ├── yfinance_extractor.py       # Yahoo Finance extraction
│   ├── openbb_extractor.py         # Multi-provider via OpenBB SDK
│   ├── data_validator.py           # Data quality validation
│   ├── preprocessor.py             # Data preprocessing
│   ├── data_storage.py             # Data persistence
│   ├── database_manager.py         # SQLite with integrity columns
│   ├── timestamp_utils.py          # UTC-aware timestamp utilities
│   └── time_series_analyzer.py     # Time series analysis
│
├── integrity/                       # PnL integrity enforcement (Phase 7.9)
│   ├── __init__.py
│   └── pnl_integrity_enforcer.py   # 6 integrity checks, canonical metrics, CI gate
│
├── forcester_ts/                    # Time series forecasting models
│   ├── forecaster.py               # Main forecasting engine
│   ├── ensemble.py                 # Ensemble coordinator
│   ├── garch.py                    # GARCH implementation
│   └── _freq_compat.py            # Pandas frequency compatibility
│
├── models/                          # Signal generation and routing
│   └── time_series_signal_generator.py  # Signal router
│
├── execution/                       # Order management and paper trading
│   ├── paper_trading_engine.py     # Risk-managed paper trading
│   └── order_manager.py           # Order lifecycle management
│
├── ai_llm/                         # LLM integration
│   ├── ollama_client.py            # Local LLM server integration
│   ├── signal_generator.py         # LLM-powered signal generation
│   └── market_analyzer.py          # Fundamental analysis via LLM
│
├── scripts/                         # Executable scripts
│   ├── run_etl_pipeline.py         # Main ETL orchestration
│   ├── run_auto_trader.py          # Autonomous profit loop
│   ├── production_audit_gate.py    # Production readiness gate
│   ├── ci_integrity_gate.py        # CI integrity gate
│   ├── openclaw_models.py          # OpenClaw model management
│   ├── pmx_interactions_api.py     # Interactions API (FastAPI)
│   ├── llm_multi_model_orchestrator.py  # Multi-model LLM orchestrator
│   ├── start_ngrok_interactions.ps1     # ngrok tunnel launcher
│   ├── validate_credentials.py     # Credential validation (no values)
│   └── migrate_*.py               # Database migrations
│
├── tools/                           # Development tools
│   ├── secrets_guard.py            # Pre-commit secrets leak guard
│   └── pmx_git_askpass.py          # Git credential helper
│
├── tests/                           # Test suite (2355+ tests)
│   ├── etl/                        # ETL module tests
│   ├── forecaster/                 # Forecaster tests
│   ├── execution/                  # Execution tests
│   ├── integration/                # Integration tests
│   ├── security/                   # Security tests
│   └── utils/                      # Utility tests (OpenClaw CLI, etc.)
│
├── bash/                            # Shell scripts
│   ├── run_20_audit_sprint.sh      # Audit sprint with lockfile
│   ├── run_pipeline_live.sh        # Live pipeline shortcut
│   └── run_auto_trader.sh          # Auto-trader defaults
│
├── CLAUDE.md                        # Agent guidance (Claude Code)
├── AGENTS.md                        # Agent guardrails + cron rules
├── .env.template                    # Environment variable template
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## ⚡ Performance

### Benchmarks (AAPL Dataset: 1,006 rows)

| Operation | Time | Notes |
|-----------|------|-------|
| **Cache Hit** | <0.1s | Instant retrieval from local parquet |
| **Cache Miss** | ~20s | Network fetch + validation + save |
| **Validation** | <0.1s | Vectorized quality checks |
| **Preprocessing** | <0.2s | Missing data + normalization |
| **Train/Val/Test Split** | <0.1s | Chronological slicing |
| **Full Analysis** | 1.2s | ADF + ACF + statistics (704 rows) |
| **Single Visualization** | 0.3s | One plot at 150 DPI |
| **All Visualizations** | 2.5s | 8 plots at publication quality |
| **Full ETL Pipeline (cached)** | <1s | All 4 stages with 100% cache hit |
| **Full ETL Pipeline (no cache)** | ~25s | First run with network fetch |

### Cache Performance

- **Hit Rate**: 100% (after first run)
- **Speedup**: 20x compared to network fetch
- **Storage**: 54 KB per ticker (Parquet compressed)
- **Validity**: 24 hours (configurable)
- **Network Savings**: 100% on cache hit

---

## 🧪 Testing

### Run Test Suite

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=etl --cov-report=html

# Run specific test module
pytest tests/etl/test_yfinance_cache.py -v

# Run with verbose output
pytest tests/ -v --tb=short
```

### Test Coverage Summary

| Test Suite | Tests | Status | Coverage |
|------------|-------|--------|----------|
| **ETL Tests** | 300+ | Passing | Core pipeline, caching, checkpoints |
| **LLM Integration Tests** | 30+ | Passing | Market analysis, signals, risk |
| **Forecaster Tests** | 150+ | Passing | SARIMAX, GARCH, SAMOSSA, MSSA-RL, ensemble |
| **Integration Tests** | 100+ | Passing | End-to-end workflows |
| **Execution Tests** | 80+ | Passing | Order management, paper trading |
| **Security Tests** | 20+ | Passing | Data protection, credentials |
| **Total** | 731 | 718 passing, 6 skipped, 7 xfailed | Comprehensive coverage |

---

## 📚 Documentation

### Core Documentation

- **[Core Project Documentation](Documentation/CORE_PROJECT_DOCUMENTATION.md)**: Canonical docs, evidence standards, and verification ladder
- **[Metrics & Evaluation](Documentation/METRICS_AND_EVALUATION.md)**: Unambiguous metric definitions (PF/WR/Sharpe/DM-style tests)
- **[Architecture Tree](Documentation/arch_tree.md)**: Complete architecture overview
- **[OpenClaw Integration](Documentation/OPENCLAW_INTEGRATION.md)**: OpenClaw + 3-model LLM strategy + Interactions API security
- **[OpenClaw Implementation Policy](Documentation/OPENCLAW_IMPLEMENTATION_POLICY.md)**: Repo-wide implementation contracts and anti-regression evidence requirements
- **[Adversarial Audit](Documentation/ADVERSARIAL_AUDIT_20260216.md)**: 10-finding stress test with P0-P3 recommendations
- **[Project Status](Documentation/PROJECT_STATUS.md)**: Current verified snapshot + reproducible commands

### Phase 7.9 Documentation (Current)

- **[EXIT_ELIGIBILITY_AND_PROOF_MODE.md](Documentation/EXIT_ELIGIBILITY_AND_PROOF_MODE.md)**: Exit diagnosis + proof-mode specification
- **[ADVERSARIAL_AUDIT_20260216.md](Documentation/ADVERSARIAL_AUDIT_20260216.md)**: Production stress test findings
- **[INTEGRITY_STATUS_20260212.md](Documentation/INTEGRITY_STATUS_20260212.md)**: PnL integrity framework status
- **[SECURITY_AUDIT_AND_HARDENING.md](Documentation/SECURITY_AUDIT_AND_HARDENING.md)**: Security hardening plan

### Phase 7 Documentation (Regime Detection & Optimization)

**Phase 7.7-7.8 - Regime Optimization**:
- **[PHASE_7.8_RESULTS.md](Documentation/PHASE_7.8_RESULTS.md)**: All-regime optimization results and weights
- **[PHASE_7.7_FINAL_SUMMARY.md](Documentation/PHASE_7.7_FINAL_SUMMARY.md)**: Per-regime optimization handoff

**Phase 7.5 - Regime Detection Integration**:
- **[PHASE_7.5_VALIDATION.md](Documentation/PHASE_7.5_VALIDATION.md)**: Single-ticker validation results
- **[PHASE_7.5_MULTI_TICKER_RESULTS.md](Documentation/PHASE_7.5_MULTI_TICKER_RESULTS.md)**: Multi-ticker analysis

### Operational Documentation

- **[AGENTS.md](AGENTS.md)**: Agent guardrails, cron notification rules, tool-use protocol
- **[logs/README.md](logs/README.md)**: Log structure, search patterns, retention policies
- **[Cron Automation](Documentation/CRON_AUTOMATION.md)**: Production-style scheduling + evidence freshness wiring
- **[Production Security + Profitability Runbook](Documentation/PRODUCTION_SECURITY_AND_PROFITABILITY_RUNBOOK.md)**: CVE defaults, overrides, gate-clearance workflow
- **[Implementation Checkpoint](Documentation/implementation_checkpoint.md)**: Development status

---

## 🔬 Research & Reproducibility

### For Researchers and Academics

This project follows MIT-standard statistical rigor and reproducibility practices:

**Reproducibility Standards**:
- All experiments logged with commit hashes and configuration snapshots
- Deterministic runs with documented seed values (PYTHONHASHSEED, RNG, hyper-opt samplers)
- Traceable artifacts in `logs/artifacts_manifest.jsonl` (when present)
- Executable evidence: Every figure/table has a runnable script for regeneration

**Research Documentation**:
- **[Research Progress and Publication Plan](Documentation/RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md)**: Research questions, protocols, and replication checklist
- **[Agent Development Checklist](Documentation/AGENT_DEV_CHECKLIST.md)**: Overall project status and audit trail

### Citation

If you use this work in academic research, please cite:

```bibtex
@software{bestman2026portfolio,
  title={Portfolio Maximizer: Autonomous Quantitative Trading with Regime-Adaptive Ensemble},
  author={Portfolio Maximizer Team},
  year={2026},
  version={4.3},
  url={https://github.com/example-org/portofolio_maximizer},
  note={Phase 7.9: PnL integrity enforcement, adversarial audit, OpenClaw automation}
}
```

### Key Research Results

**Phase 7.7 Optimization** (January 2026):
- **Method**: Rolling cross-validation with scipy.optimize.minimize
- **Dataset**: AAPL (2023-01-01 to 2026-01-18, 3+ years)
- **Results**: 65% RMSE reduction for MODERATE_TRENDING regime (19.26 → 6.74)
- **Optimal Configuration**: 90% SAMOSSA, 5% SARIMAX, 5% MSSA-RL
- **Validation**: Multi-ticker testing (AAPL, MSFT, NVDA) with 53% adaptation rate

**Phase 7.5 Regime Detection** (January 2026):
- **Regimes Identified**: 6 market regimes based on volatility, trend strength, Hurst exponent
- **Adaptation Performance**: 53% of forecasts switched to regime-specific weights
- **Cross-Ticker Validation**: Consistent regime detection across 3 tickers
- **Historical note**: early regime-weight experiments showed RMSE regressions on some windows; current governance is driven by the forecast-audit gate (see `scripts/check_forecast_audits.py` and `Documentation/ENSEMBLE_MODEL_STATUS.md`).

### Replication Instructions

**Full Replication Package**:

1. **Environment Setup**:
   ```bash
   git clone https://github.com/example-org/portofolio_maximizer.git
   cd portofolio_maximizer
   python -m venv simpleTrader_env
   source simpleTrader_env/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

2. **Reproduce Phase 7.7 Results**:
   ```bash
   # Run optimization (matches reported results)
   python scripts/optimize_ensemble_weights.py \
       --source rolling_cv \
       --tickers AAPL \
       --start-date 2023-01-01 \
       --end-date 2026-01-18 \
       --horizon 5 \
       --min-train-size 180 \
       --step-size 20 \
       --max-folds 10 \
       --min-samples-per-regime 25 \
       --output data/phase7.7_replication.json

   # Validate results
   python scripts/run_etl_pipeline.py \
       --tickers AAPL \
       --start 2024-07-01 \
       --end 2026-01-18 \
       --execution-mode auto
   ```

3. **Access Artifacts**:
   - Configuration: `config/forecasting_config.yml` (lines 87, 98-109)
   - Results: `data/phase7.7_optimized_weights.json`
   - Logs: `logs/phase7.7/phase7.7_weight_optimization.log`
   - Documentation: `Documentation/PHASE_7.7_*.md`

**Data Availability**:
- Market data: Yahoo Finance (publicly available via yfinance library)
- Synthetic data generator: `scripts/generate_synthetic_dataset.py`
- Database schema: SQLite with documented migrations in `scripts/migrate_*.py`

### Transparency & Assumptions

**Mark-to-Market Assumptions**:
- Transaction costs: 0.8 bps for liquid US stocks (configurable in `config/execution_cost_model.yml`)
- Slippage: Market impact model with square-root scaling
- Position sizing: Risk-managed via `risk/barbell_policy.py`

**Model Assumptions**:
- Stationarity: ADF test validation before forecasting
- Seasonality: Automatic detection and detrending
- Volatility clustering: GARCH modeling for time-varying volatility

**Known Limitations**:
- Only 3/6 regimes optimized (remaining regimes lack samples in the current AAPL window)
- Ensemble governance labels can mark individual windows `RESEARCH_ONLY` when the 2% promotion margin is not met; this is a monitoring/promotion label, not proof the ensemble forecast is unused (see `Documentation/ENSEMBLE_MODEL_STATUS.md`).
- Limited to US equity markets (frontier markets in synthetic mode)
- No live broker integration (paper trading engine)

---

## 🛣️ Roadmap

### Phase 7: Regime Detection & Ensemble Optimization (Complete)

**All Phases Completed**:
- ✅ Phase 7.3: GARCH ensemble integration with confidence calibration
- ✅ Phase 7.4: Quantile-based confidence calibration (29% RMSE improvement)
- ✅ Phase 7.5: Regime detection integration (6 market regimes, multi-ticker validation)
- ✅ Phase 7.6: Threshold tuning experiments
- ✅ Phase 7.7: Per-regime weight optimization (65% RMSE reduction for MODERATE_TRENDING)
- ✅ Phase 7.8: All-regime optimization (3/6 regimes optimized; SAMOSSA dominance confirmed)
- ✅ Phase 7.9: PnL integrity enforcement, adversarial audit, OpenClaw automation, forecast gate PASS

**Next**:
- Phase 7.10: Production hardening (address adversarial findings, improve directional accuracy)

### Phase 8: Neural Forecasters & GPU Acceleration (Planned)

- PatchTST/NHITS integration with 1-hour horizon
- skforecast + XGBoost GPU for directional edge
- Chronos-Bolt as zero-shot benchmark
- Real-time + daily batch training cadence
- RTX 4060 Ti (CUDA 12.9) utilization

### Phase 9: Portfolio Optimization (Future)

- Mean-variance optimization (Markowitz)
- Risk parity portfolio
- Black-Litterman model
- Constraint handling (long-only, sector limits)

### Phase 10: Advanced Risk Modeling (Future)

- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Expected Shortfall (CVaR)
- Maximum Drawdown analysis with regime conditioning
- Stress testing framework

### Infrastructure Enhancements

**Caching**:
- ✅ 20x speedup with intelligent caching (24h validity)
- 🔄 Smart cache invalidation (market close triggers)
- Distributed caching (Redis/Memcached) for multi-node
- Cache analytics dashboard

**Monitoring & Observability**:
- Enhanced log organization with phase-specific directories
- Grafana/Loki integration (documented in logs/README.md)
- Real-time model health monitoring
- Automated performance degradation alerts

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Follow code standards**:
   - Vectorized operations only
   - Type hints required
   - Comprehensive docstrings
   - MIT statistical standards
4. **Write tests** (maintain >95% coverage)
5. **Commit changes** (`git commit -m 'feat: Add amazing feature'`)
6. **Push to branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/portofolio_maximizer.git

# Add upstream remote
git remote add upstream https://github.com/example-org/portofolio_maximizer.git

# Create feature branch
git checkout -b feature/your-feature

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests before committing
pytest tests/ --cov=etl
```

### Client-only sync (GitHub as source of truth)

For follower/client PCs where GitHub `master` is the definitive source and local changes should never push upstream, use the guarded sync helper:

```bash
# Sync current branch from GitHub (auto-stash dirty worktrees)
bash/git_syn_to_local.sh

# Sync a specific branch from GitHub
bash/git_syn_to_local.sh master
```

- Lives at `bash/git_syn_to_local.sh` (run from repo root).
- Auto-stashes uncommitted work, fetches, rebases, and restores the stash when safe.
- Never pushes; warns if local-only commits exist so you can reconcile from the master PC.

---

## 📄 License

This project is documented as MIT-licensed in the repo badges and policy docs.

---

## 👤 Author

**Project Maintainer**

- GitHub: [example-org](https://github.com/example-org)
- Public contact: See Support section

---

## 🙏 Acknowledgments

- **MIT OpenCourseWare**: Micro Masters in Statistics and Data Science (MMSDS)
- **Yahoo Finance**: Market data API

---

## 📊 Project Statistics

### Code Metrics

- **Total Production Code**: 15,000+ lines
- **Test Code**: 5,000+ lines
- **Test Suite**: 731 tests (718 passing, 6 skipped, 7 xfailed)
- **Test Coverage**: Comprehensive across all modules
- **Documentation**: 174 files in Documentation/ + root guides

### Performance Metrics

- **Cache Performance**: 20x speedup with intelligent caching
- **Data Quality**: 0% missing data (after preprocessing)
- **Optimization Results**: 65% RMSE reduction (Phase 7.7, MODERATE_TRENDING)
- **Regime Detection**: 53% adaptation rate across multi-ticker validation
- **Model Ensemble**: 4 forecasters (SARIMAX, GARCH, SAMOSSA, MSSA-RL)

### Phase 7 Progress

- **Phases Completed**: 10 (7.0 - 7.9)
- **Current Phase**: 7.10 (Production hardening, planned)
- **Regimes Optimized**: 3/6 (CRISIS, MODERATE_MIXED, MODERATE_TRENDING)
- **Round-Trips**: 37 validated ($673.22 PnL, 43.2% WR, 1.85 PF)
- **Forecast Gate**: PASS (21.4% violation rate, 28 effective audits)
- **PnL Integrity**: ALL PASSED (0 CRITICAL/HIGH violations)

---

## 🔧 Troubleshooting

### Common Issues

**1. Cache not working**
```bash
# Check cache directory permissions
ls -la data/raw/

# Clear cache if corrupted
rm data/raw/*.parquet

# Verify cache configuration
cat config/yfinance_config.yml
```

**2. Import errors**
```bash
# Ensure virtual environment is activated
source simpleTrader_env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**3. Test failures**
```bash
# Run specific failing test with verbose output

# Check Python version
python --version  # Should be 3.10+
```

---

## 📞 Support

For questions or issues:

1. **Check Documentation**: `Documentation/` directory
2. **Search Issues**: [GitHub Issues](https://github.com/example-org/portofolio_maximizer/issues)
3. **Open New Issue**: Provide reproducible example
4. **Public contact email**: redacted

---

---

## 🎯 Current Status Summary

**Phase 7.9**: ✅ Complete (PnL integrity, adversarial audit, OpenClaw automation)
**Production Status**: Research Phase (addressing adversarial findings before production deployment)

**Latest Achievements**:
- PnL integrity enforcement with canonical views and CI gate
- Adversarial audit: 10 findings documented with P0-P3 recommendations
- Forecast audit gate: PASS (21.4% violation rate, 28 effective audits)
- OpenClaw cron: 9 audit-aligned jobs with real script execution
- Interactions API: Auth mode enforcement, CORS, rate limiting, ngrok
- 3-model LLM stack: deepseek-r1:8b/32b + qwen3:8b
- 37 round-trips validated ($673.22 PnL, 1.85 profit factor)
- SARIMAX disabled by default (15x single-forecast speedup)
- Secrets leak guard with pre-commit hook + CI check
- 731 tests collected (718 passing)

**Next Steps** (Phase 7.10):
1. Address 94.2% quant FAIL rate (0.8% from RED gate)
2. Fix ensemble underperformance (worse than best single 92% of the time)
3. Improve directional accuracy (41% WR below coin-flip)
4. Fix confidence calibration (0.9+ confidence yields 41% WR)
5. Widen proof-mode max_holding for better risk/reward

---

**Built with Python, NumPy, Pandas, and SciPy**

**Version**: 4.3
**Status**: Phase 7.9 Complete - PnL integrity enforcement, adversarial audit, OpenClaw automation
**Last Updated**: 2026-02-17
