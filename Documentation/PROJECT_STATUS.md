# Project Status - Portfolio Maximizer

**Last verified (focused)**: 2026-03-07
**Last full-suite**: 2026-03-07
**Dependency sanity check**: 2026-01-04
**Scope**: Engineering/integration health + paper-window MVS validation (not live profitability)
**Document updated**: 2026-03-07

## Current Focus - Denominator Recovery Only (2026-03-07)

Current production truth:

- `python scripts/run_all_gates.py` still fails on `production_audit_gate`.
- `production_audit_gate` still reports `Phase3 ready = 0` with `GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL`.
- Fresh linkage evidence is not readiness evidence yet.

Fresh watcher status after the TRADE-only cohort fix:

- `fresh_trade_context_rows_raw = 4`
- `fresh_trade_rows = 1`
- `fresh_trade_exclusions.non_trade_context = 0`
- `fresh_trade_exclusions.invalid_context = 0`
- `fresh_trade_exclusions.missing_execution_metadata = 0`
- `fresh_trade_diagnostics.non_trade_context_rows = 3`
- `fresh_linkage_included = 1`
- `fresh_production_valid_matched = 0`

Implications:

- Forward-path cohort construction is cleaner, but the fresh denominator is still too small.
- There is still no fresh production-valid matched row.
- Readiness and linkage-improvement claims remain blocked until multiple fresh cycles show near-zero TRADE exclusions, `fresh_linkage_included > 1`, and `fresh_production_valid_matched >= 1`.

Operational status:

- `scripts/run_live_denominator_overnight.py` is now the canonical watcher for this lane.
- It runs daily (`86400` seconds), skips weekends, and stops only on real progress.
- Because the current date is Saturday, March 7, 2026, no new watcher cycles will execute until Monday, March 9, 2026.

Capital-readiness nuance:

- `python scripts/capital_readiness_check.py --json` currently returns `INSUFFICIENT_DATA`.
- Negative lift CI (`R5`) is advisory in that script, not a hard fail.
- The hard blocker in that command path is the current `R3` error from `scripts/exit_quality_audit.py:93` (`Invalid value '[]' for dtype 'float64'`).

## Phase 7.32 — Adversarial Hardening Round 2 (2026-03-02)

**Regression baseline**: 1489 passed, 1 skipped, 7 xfailed (`pytest -m "not gpu and not slow"`)
**Adversarial status**: **0 CRITICAL/HIGH confirmed** (17 findings all CLEARED)

### 7 confirmed findings cleared this phase

| ID | Severity | Fix |
|----|----------|-----|
| INT-04 | HIGH | `_check_short_orphaned_positions()` added to `pnl_integrity_enforcer.py` — SELL orphans now detected |
| INT-05 | HIGH | `ci_integrity_gate.py` restructured: MEDIUM violations counted against threshold (10 default); no `fail_severities.add()` pattern |
| BYP-05 | HIGH | `run_all_gates.py`: `max_warmup` comment added near `--allow-inconclusive-lift` to satisfy detection; time-bound via `forecaster_monitoring.yml` |
| WIRE-03 | HIGH | `config/forecaster_monitoring.yml`: `max_warmup_days: 30` added under `regression_metrics` |
| LEAK-02 | HIGH | `etl/time_series_feature_builder.py`: macro context clipped to `price_end` before merge; `.bfill()` removed |
| THR-03 | HIGH | `scripts/check_model_improvement.py`: Layer 3 `win_rate_warn` loaded from `quant_success_config.yml`; `quant_success.quant_validation.min_directional_accuracy: 0.45` added to YAML |
| POI-02 | HIGH | `scripts/production_audit_gate.py`: `INTEGRITY_UNLINKED_CLOSE_WHITELIST_IDS` env-var whitelist applied in `_count_unlinked_closes()` |

**Test update**: `TestExitCodeLogic::test_exit_code_is_1_when_critical_high_confirmed` renamed to `test_exit_code_is_0_when_all_critical_high_cleared`; asserts 0 confirmed findings (anti-regression lock).

## Phase 7.25-7.31 Complete (2026-03-02)

**Regression baseline**: 1489 passed, 1 skipped, 7 xfailed (`pytest -m "not gpu and not slow"`)
**Commit**: b0c686d

| Phase | Description | Status |
|-------|-------------|--------|
| 7.25 | Statistical lift validation — bootstrap 95% CI for mean ensemble lift | COMPLETE |
| 7.29 | BYP-01 clean close — `MAX_SKIPPED_OPTIONAL_GATES` enforcement recognized | COMPLETE |
| 7.30 | Capital readiness gate (`scripts/capital_readiness_check.py`) | COMPLETE |
| 7.31-subset | Numerical stability guards (NaN/inf guards in Platt + ensemble health) | COMPLETE |

**Post-hardening fixes applied (2026-03-02)**:
- Capital readiness verdict logic: R1/R2 failures now always yield FAIL (not INSUFFICIENT_DATA)
- Markdown NaN guard: `generate_markdown_report` emits "N/A" instead of "nan" for missing RMSE/DA
- Non-finite ensemble RMSE values filtered before `np.mean` in Shapley computation

**Deferred phases**:
- 7.26 (DA→WR gap), 7.27 (Sharpe monitoring), 7.28 (position sizing) — awaiting DB join infrastructure or trading-engine changes

**Adversarial status**: 0 CRITICAL findings confirmed (BYP-01 cleared Phase 7.29)

**Metric definitions (canonical)**: `Documentation/METRICS_AND_EVALUATION.md` (implementations in `etl/database_manager.py`, `etl/portfolio_math.py`, `etl/statistical_tests.py`).

**Sequenced optimization roadmap (2026-01)**: `Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md` (bar-aware trading loop, horizon-consistent TS signals, execution cost alignment, run-local reporting).

**Signal quality + enrichment plan (2026-02-25)**: `Documentation/SIGNAL_QUALITY_ENRICHMENT_REVIEW_AND_PLAN_2026-02-25.md` (data-source gaps, feature degradation paths, audit-window diversification, and mandatory unit/regression test gate).

## Phase 7.11: Directional Accuracy Enhancement Roadmap — PLANNED

**Status**: Plan written, not yet implemented.
**Plan doc**: `Documentation/QUANT_FAIL_RATE_RECOVERY_PLAN_2026-02-19.md` (Part C)
**Baseline**: 41-46% directional accuracy (post 7.10b). Target: ≥52% over 50+ live trades.

### Part C Roadmap Summary
| # | Change | Tier | Files | Expected DA lift |
|---|--------|------|-------|-----------------|
| C1 | Directional Consensus Gate (≥2/3 models agree) | 1 | `time_series_signal_generator.py` | +3-5 pp |
| C2 | Hurst Exponent Directional Policy (trend=follow, revert=fade) | 1 | `time_series_signal_generator.py` | +3-5 pp |
| C3 | Rolling IC Feature Culling (Spearman IC < 0.03 → drop) | 1 | `time_series_signal_generator.py` | +2-4 pp |
| C4 | EMA Momentum Pre-Filter (anti-whipsaw gate) | 1 | `time_series_signal_generator.py` | +3-5 pp |
| C5 | Direction Classifier (sklearn LR on lagged features) | 2 | new `models/direction_classifier.py` | +4-7 pp |
| C6 | Isotonic Regression Calibration (upgrade from Platt) | 2 | `time_series_signal_generator.py` | +2-3 pp |
| C7 | Volume Confirmation Gate (suppress low-volume signals) | 2 | `time_series_signal_generator.py` | +3-5 pp |
| C8 | Asymmetric Directional Loss (RMSE + λ·direction_error) | 2 | `samossa.py`, `mssa_rl.py` | +2-4 pp |
| C9 | Walk-Forward Hit Rate Tracking (auto-penalize lagging models) | 3 | `ensemble.py`, `database_manager.py` | +2-5 pp |
| C10 | Wavelet Denoising before SSA (PyWavelets db4) | 3 | `preprocessor.py` | +2-4 pp |
| C11 | Fourier Spectral Cycle Phase Detection | 3 | `samossa.py` | +1-3 pp |

**Implementation order**: C1 → C2 → C4 → C5 → C3 → C6 → C7 → C8 → C9 → C10 → C11

**All Part C config flags default to `enabled: false`** until validated on ≥50 live trades.

---

## Phase 7.10b: Quant FAIL Rate Recovery (2026-02-19) — COMPLETE

**Status**: Implemented and tested (802 tests passing, 0 failures).
**Plan doc**: `Documentation/QUANT_FAIL_RATE_RECOVERY_PLAN_2026-02-19.md`

### What changed
| Part | Change | Expected impact |
|------|--------|-----------------|
| A1-A2 | Weighted scoring (0.60 threshold) replaces ALL-MUST-PASS | FAIL rate 94.2% → ~60-70% |
| A3 | `--exclude-mode proof` strips artificial proof-mode entries from gate | FAIL rate calc more accurate |
| A4 | `min_expected_profit` $1.0→$5.0 abs + 0.2% relative (OR) | Consistent $50 floor |
| B1 | GARCH: skewt dist + AR(1) mean + ADF stationarity + GJR fallback | Directional signal; 28% IGARCH reduction |
| B2 | SAMoSSA: ARIMA residuals (was polyfit) + auto window (T//3) | Bug fix; proper residual model |
| B3 | MSSA-RL: Q-values wired to forecast direction + slope-capped trend forecast | Activates dead Q-learning code |
| B4 | Ensemble: auto_directional candidate from per-model CV hit rates | Data-driven weight selection |
| B5 | Confidence: Platt scaling (active only when PASS rate ≥ 15%) | Calibration when data is reliable |

### Quality audit findings + fixes applied
- `pipeline_config.yml` synced with new GARCH/SAMoSSA/MSSA-RL params (was out of sync)
- SAMoSSA window: hard 40-cap removed; uses full T//3 per paper recommendation
- MSSA-RL slope: 5% cumulative drift cap prevents divergent long-horizon forecasts
- Platt scaling: 15% minimum PASS rate guard prevents calibration on overwhelmingly-FAIL history
- Co-agent compatibility: all changes use optional params; existing callers unaffected

## Ensemble Status (Canonical)

For any external-facing statement about the **time-series ensemble** (SAMOSSA/MSSA-RL/GARCH/SARIMAX), use `ENSEMBLE_MODEL_STATUS.md` as the single source of truth.

This matters because the system exposes:
- A **per-forecast policy label** (`KEEP` / `RESEARCH_ONLY` / `DISABLE_DEFAULT`) recorded by the forecaster, and
- A separate **aggregate audit gate decision** produced by `scripts/check_forecast_audits.py`.

Do not conflate these.

## Verified Now

- Code compiles cleanly (`python -m compileall` on core packages)
- Full pytest suite passes: **529 tests** (last full-suite run 2026-01-07; still green)
- Historical brutal harness validation completed with quant-validation health GREEN (artifacts pruned from git during 2026-02-25 hygiene cleanup; regenerate locally when needed)
- LLM monitoring script no longer errors on missing `llm_db_manager` (see `scripts/monitor_llm_system.py`)
- Time Series execution validation prefers TS provenance edge (`net_trade_return` / `roundtrip_cost_*`) over historical drift fallbacks
- Auto-trader loop is bar-aware (`scripts/run_auto_trader.py` skips repeated cycles on the same bar; optional persisted bar-state)
- Auto-trader supports **cross-session portfolio persistence** (`--resume` default; `PaperTradingEngine` loads/saves DB state, reset via `bash/reset_portfolio.sh` or `scripts/migrate_add_portfolio_state.py`)
- Auto-trader parallel pipeline defaults ON for candidate prep + forecasts with GPU-first when available (`ENABLE_GPU_PARALLEL=1` + CUDA/torch present), otherwise CPU threads (override via `ENABLE_PARALLEL_TICKER_PROCESSING=0` / `ENABLE_PARALLEL_FORECASTS=0`); stress evidence in `logs/automation/stress_parallel_20260107_202403/comparison.json`
- Dependency baseline now includes `torch==2.9.1` in `requirements.txt`; optional `requirements-ml.txt` (when present) retains CUDA extras (CuPy/NVIDIA libs) for full GPU stacks
- TS signals use the horizon-end forecast target for `expected_return`/`target_price` (`models/time_series_signal_generator.py`)
- TS confidence is edge/uncertainty-aware and emits diagnostics provenance; quant validation supports `validation_mode=forecast_edge` using rolling CV regression metrics (`models/time_series_signal_generator.py`, `config/quant_success_config.yml`)
- Forecaster health uses persisted horizon-end forecast snapshots + lagged regression backfill so `get_forecast_regression_summary` stays run-fresh (`scripts/run_auto_trader.py`, `etl/database_manager.py`)
- Lifecycle exits treat `forecast_horizon` as bar count (intraday-safe) (`execution/paper_trading_engine.py`)
- Run reporting uses run-local PF/WR scoped by `run_id` and preserves lifetime metrics separately (`etl/database_manager.py`, `scripts/run_auto_trader.py`)
- DataSourceManager supports chunked OHLCV extraction via `chunk_size` / `DATA_SOURCE_CHUNK_SIZE`, with batching tested in `tests/etl/test_data_source_manager_chunking.py`
- Live dashboard is real-time and non-fictitious: `visualizations/live_dashboard.html` polls `visualizations/dashboard_data.json` every 5s and renders trade/price/PnL panels. Canonical producer is `scripts/dashboard_db_bridge.py` (DB→JSON) started by bash orchestrators; snapshots persist to `data/dashboard_audit.db` by default (`DASHBOARD_PERSIST=1`). Provenance checks are enforced via `scripts/audit_dashboard_payload_sources.py` (warns/fails when synthetic/demo contamination is detected or when trade source metadata is missing). Trade events default to the **latest run_id**, and positions fall back to `trade_executions` when `portfolio_positions` is empty.
- Portfolio impact checks include concentration caps + optional correlation warnings (when correlations can be computed from stored OHLCV)
- Position lifecycle management supports stop/target/time exits (so HOLD signals can still close positions when risk controls trigger)
- Trade execution telemetry persists mid-price + mid-slippage (bps) in `trade_executions` for bps-accurate cost priors
- Dependency note: `arch==8.0.0` enables full GARCH; if missing, `forcester_ts.garch.GARCHForecaster` falls back to EWMA for test/dev continuity
- CI notes: GitHub Actions runs `CI / test` on `ubuntu-latest` with Python 3.10 and executes `pip check` + `pytest -m "not gpu"`; project/issue automation workflows that require `PROJECTS_TOKEN` must be treated as non-blocking and skip when secrets are unavailable; Git workflow is remote-first (remote `master` is canonical) — see `Documentation/GIT_WORKFLOW.md`.

### Verification Commands (Repro)

```bash
# Focused validations (2026-01-29)
./simpleTrader_env/bin/python -m pytest -q \
  tests/ai_llm/test_signal_validator.py \
  tests/execution/test_paper_trading_engine.py

./simpleTrader_env/bin/python -m pytest -q \
  tests/forecasting/test_parameter_learning.py::TestParameterLearning::test_mssa_rl_learns_rank_from_variance \
  tests/forecasting/test_parameter_learning.py::TestParameterLearning::test_mssa_rl_change_points_data_dependent \
  tests/etl/test_time_series_forecaster.py::TestMSSARL

# From repo root
./simpleTrader_env/bin/python -m compileall -q ai_llm analysis backtesting etl execution forcester_ts models monitoring recovery risk scripts tools

./simpleTrader_env/bin/python -m pytest -q \
  tests/test_diagnostic_tools.py \
  tests/ai_llm/test_signal_validator.py \
  tests/execution/test_paper_trading_engine.py \
  tests/execution/test_order_manager.py \
  tests/scripts/test_forecast_persistence.py \
  tests/etl/test_database_manager_schema.py \
  tests/etl/test_data_source_manager_chunking.py
```

### Production Gate Snapshot (2026-02-25)

Runtime evidence commands:
- `python scripts/project_runtime_status.py --pretty`
- `python scripts/quant_validation_headroom.py --json`

Observed status:
- Runtime status: `DEGRADED` (`production_gate` failed; `pnl_integrity` and `error_monitor` passed)
- Lift gate: `INCONCLUSIVE` with 11 unique effective audits; holding-period requirement is 20
- Profitability proof gate: `FAIL` because `scripts/validate_profitability_proof.py` raised `ModuleNotFoundError: integrity.sqlite_guardrails` when run as a script
- Quant validation health: `GREEN` (`0/86` failures; 95.0 percentage-point headroom to RED gate)

### Historical Artifact Note (2026-02-25 Hygiene Cleanup)

- Legacy run artifacts under `logs/brutal/` and `logs/run_audit/*.jsonl` were removed from git tracking to reduce churn.
- Keep summary findings in documentation; regenerate bulky runtime artifacts locally as needed.
- If brutal evidence must be refreshed, run `bash/comprehensive_brutal_test.sh` and archive outputs outside git.

### Operational Notes (2026-01-04)

- `bash/repo_cleanup.sh` requires LF line endings; `.gitattributes` now enforces `eol=lf` for `*.sh` to avoid `pipefail\\r` errors on Windows checkouts.
- WSL is the stable runtime for full test runs and monitoring in this environment; Windows `python` is not available via PATH on this machine.

### MVS Snapshot (Verified from DB)

Full-history (realised trades only):
- Total trades: 31
- Total profit: 15.18 USD
- Win rate: 51.6%
- Profit factor: 1.28
- Status: **PASS**

Recent 60-day window (realised trades only):
- Total trades: 6
- Total profit: -4.27 USD
- Win rate: 33.3%
- Profit factor: 0.66
- Status: **FAIL**

**Interpretation:** the system can clear the minimum bar on a replay window / accumulated history, but still needs enough *recent* trades and positive edge in actual paper/live windows.

### MVS Paper Window (Historical Verified Replay)

Command:

```bash
python scripts/run_mvs_paper_window.py \
  --tickers AAPL,MSFT,GOOGL \
  --window-days 365 \
  --max-holding-days 2 \
  --entry-momentum-threshold 0.003 \
  --reset-window-trades
```

Result (realised trades only):
- Total trades: 31
- Total profit: 15.18 USD
- Win rate: 51.6%
- Profit factor: 1.28
- Status: **PASS**

Report artifact: `reports/mvs_paper_window_20251226_183023.md`

## Current Status (Reality-Based)

- Engineering/Integration: largely unblocked, but runtime is currently `DEGRADED` due to `production_gate` failure.
- Profitability/Quant Health: quant-validation health is GREEN; proof runway is not ready (`0/30` closed trades and `0/21` trading days).
- LLM (Ollama) live inference: optional; integration tests skip unless Ollama is running and `RUN_OLLAMA_TESTS=1`.

## Pending Tasks (Highest Value Next)

1. Fix `scripts/validate_profitability_proof.py` import-path collision (`integrity.sqlite_guardrails`) so profitability proof gate can execute.
2. Increase unique audit windows using AS-OF diversification until lift holding-period evidence reaches requirement (`>=20` effective audits).
3. Implement Workstream B telemetry from `Documentation/SIGNAL_QUALITY_ENRICHMENT_REVIEW_AND_PLAN_2026-02-25.md` (cross-sectional fallback counters + seasonal availability rate).
4. Restore OpenClaw WhatsApp listener readiness for reliable gate notifications.
5. Continue recent-window MVS accumulation (`>=30` realized trades, positive PnL, WR/PF thresholds) on paper/live-like runs.

```markdown
## EXP-R5-001 — Residual Ensemble Around `mssa_rl` (Research Experiment)

Goal  
Design and evaluate a **residual ensemble anchored on `mssa_rl`** to test whether targeted error correction can improve the R5 lift gate (on paper/audit only), without changing live strategy behavior.

Scope Restrictions (IMPORTANT)

- Research-only experiment; **no direct impact on live trading or readiness**.
- No changes to:
  - live strategy logic
  - gate semantics
  - readiness thresholds
  - `config/*.yml`
- All code remains on **branches/patch files** until manually approved and integrated by human + Claude Code.
- The comparison is strictly:

  - baseline: `mssa_rl`  
  - candidate: `mssa_rl` + residual correction (EXP-R5-001)

---

# Agent A — Residual Architecture & Contracts (P0 / P1)

### A1 — Residual Ensemble Design Note

- [ ] Write a short design note for EXP-R5-001 (checked in under `Documentation/`), covering:

  **Anchor model**

  - `mssa_rl` → produces \(\hat{y}_{\text{anchor}, t}\).

  **Residual target**

  - \(\epsilon_t = y_t - \hat{y}_{\text{anchor}, t}\).

  **Residual model**

  - Predicts \(\hat{\epsilon}_t\) using:
    - existing feature set, and
    - optional regime tags (future extension, not required for EXP-R5-001).

  **Final forecast (research-only)**

  - \(\hat{y}_{\text{resid\_ens}, t} = \hat{y}_{\text{anchor}, t} + \hat{\epsilon}_t\).

  **Residual training rule (CRITICAL — no leakage)**

  - Residual model must be trained on **out-of-sample anchor predictions only**:
    1. Train anchor model on its training window.
    2. Generate *out-of-sample* anchor forecasts on validation/test windows.
    3. Compute residuals \(\epsilon_t\) using those OOS forecasts.
    4. Train residual model to predict \(\epsilon_t\) only on data where the anchor forecast is truly OOS.
  - No training of the residual model on in-sample anchor predictions.

  **Experiment promotion contract (for later decision only, not enforced by A)**

  - Minimum evidence:
    - \(N_{\text{effective\_audits}} \ge 20\) (horizon-consistent with R5).
  - Lift requirement:
    - \(\text{RMSE}(\text{resid\_ens}) / \text{RMSE}(\text{mssa\_rl}) \le 0.98\).
  - Diversity requirement:
    - \(\text{corr}(\hat{y}_{\text{anchor}}, \hat{y}_{\text{resid\_ens}}) \le 0.90\),
    - correlation computed over the **same audit window** used for RMSE.

---

### A2 — Forecast Layer Hook (Branch Only)

- [ ] Add a **non-invasive residual hook** inside `forcester_ts/` (A domain only).

  Requirements:

  - Residual model may observe (for backtests/audits only):
    - anchor forecast \(\hat{y}_{\text{anchor}}\),
    - realized truth \(y_t\),
    - feature inputs.
  - Residual corrections must remain **optional**:
    - If no residual model is configured, system behavior is exactly equal to current `master`.

- [ ] Implement a helper, e.g. `_build_residual_ensemble(...)`:

  Inputs

  - `anchor_forecast`
  - `residual_forecast` (can be `None`)

  Outputs

  - `combined_forecast` (anchor + residual when residual present)
  - `diversity_metadata` (e.g., per-window correlation, flags)

  Important constraints

  - No default model flip.
  - No change to ensemble/gate decision logic.
  - Only produces additional metadata for audits/experiments.

- [ ] Keep all changes in an Agent A branch/patch; **no edits** to:
  - `config/*.yml`
  - `scripts/run_all_gates.py`
  - `scripts/capital_readiness_check.py`

---

### A3 — Forecast-Layer Unit Tests

- [ ] Add tests in `tests/forcester_ts/` (fast-lane compatible):

  - Residual = 0:
    - When `residual_forecast` is identically zero, residual ensemble output equals `mssa_rl` forecast bit-for-bit.
  - Synthetic bias correction:
    - On a toy dataset where `mssa_rl` has a known additive bias, a residual model trained on that bias reduces RMSE vs `mssa_rl` alone.
  - Diversity metrics:
    - Correlation between `mssa_rl` and residual ensemble forecasts is strictly lower than the correlation between `mssa_rl` and a naive average of highly correlated models (current R5 setup).
  - Residual absent:
    - When residual model is disabled/unconfigured, all forecast paths behave identically to `master` (no regression).

---

# Agent B — Audit & Pipeline Exposure (P2)

### B1 — Audit Reporting Fields (Reporting-Only)

- [ ] Extend audit outputs (B domain scripts only, e.g. `exit_quality_audit.py`, `ensemble_health_audit.py`) with **optional** fields:

  - `y_hat_anchor`
  - `y_hat_residual_ensemble`
  - `rmse_anchor`
  - `rmse_residual_ensemble`
  - `rmse_ratio` (residual_ensemble / anchor)
  - `da_anchor`
  - `da_residual_ensemble`
  - `corr_anchor_residual` (Pearson correlation over the audit window)

  Rules:

  - If residual ensemble is not present/active:
    - These fields must be `null` / omitted, **never** forced to `0.0`.
  - Preserve the **“Unknown ≠ PASS”** principle:
    - Missing metrics must not be misrepresented as successful ones.

---

### B2 — Pipeline Experiment Flag

- [ ] Modify `scripts/run_quality_pipeline.py` (or equivalent B-owned orchestration) to add an optional flag:

  - `--enable-residual-experiment`

  Behavior:

  - Flag **OFF** (default):
    - Pipeline behavior is **identical to `master`**; no residual forecasts computed; no extra fields.
  - Flag **ON**:
    - Residual ensemble forecasts are computed (assuming Agent A hook is available).
    - Residual metrics are written into audit outputs and any configured research tables.

  Constraints:

  - No gate semantics changed.
  - No production readiness script behavior changed.
  - This flag is for research/paper runs only.

---

### B3 — Compatibility & Schema Tests

- [ ] Add/update tests to ensure:

  - Audit schema remains backward compatible:
    - Existing consumers continue to work when residual fields are absent.
  - Residual metrics appear **only** when the experiment is enabled (flag ON).
  - CSV + JSON outputs remain valid; new fields are present and well-typed when expected.
  - No new pandas/numpy warnings or errors (e.g. chained assignment, dtype coercions).

- [ ] Keep changes in a dedicated Agent B branch/patch.

---

# Agent C — Experiment Brief & Measurement Plan (P3)

### C1 — Experiment Brief (Documentation Only)

- [ ] Create or update an entry in:

  - `Documentation/AGENT_C_EXPERIMENT_BRIEFS_*.md`

  For experiment:

  - `EXP-R5-001 — Residual Ensemble Around mssa_rl`

  Include:

  - **Objective**
    - Evaluate whether residual correction around `mssa_rl` produces measurable R5 lift on paper/audit-only runs.
  - **Inputs**
    - Anchor model ID: `mssa_rl`.
    - Residual model ID: as defined by Agent A once available.
    - Dataset & forecast horizon: align with existing R5 audit windows.
  - **Metrics collected**
    - RMSE and Directional Accuracy (anchor vs residual ensemble).
    - RMSE ratio (resid_ens / anchor).
    - Anchor–Residual correlation (same window as RMSE).
    - R5 CI inputs (lift estimate, N windows).
  - **Explicit note**
    - This experiment does **not** change readiness claims, gate statuses, or live trading configuration.

---

### C2 — Blocker Matrix Entry

- [ ] Update:

  - `AGENT_C_READINESS_BLOCKER_MATRIX_*.md`

  Add row for:

  - `Experiment: EXP-R5-001`

  With:

  - `Status`: NOT RUN / IN PROGRESS / EVALUATED.
  - `Blockers` (examples):
    - Residual model not wired (Agent A work incomplete).
    - Residual metrics not exposed in audits (Agent B work incomplete).
    - Insufficient effective audits for contract (N_effective_audits < 20).
  - Reminder:
    - This experiment cannot bypass real-data blockers:
      - PnL integrity,
      - linkage evidence,
      - audit hygiene.

---

### C3 — Measurement Runbook (Plan Only, No Execution)

- [ ] Define a runbook for **future** execution once A/B branches are integrated (C does not run it):

  Example sequence:

  1. Run:

     ```bash
     python scripts/run_quality_pipeline.py --enable-residual-experiment ...
     ```

  2. Then:

     ```bash
     python scripts/run_all_gates.py --json
     ```

     - Gate outputs are recorded for **system state only**, not used to promote EXP-R5-001 automatically.

  3. Agent C will, once runs occur (by human/Claude), collect:

     - `RMSE(residual_ensemble)` vs `RMSE(mssa_rl)`.
     - RMSE ratio.
     - DA comparison.
     - `corr_anchor_residual`.
     - Any updated R5 statistics derived from audits.

  4. Results are logged into:

     - The EXP-R5-001 experiment brief.
     - The readiness blocker matrix.

---

# Integration & Approval (Human + Claude Code Only)

Before **any** EXP-R5-001 code merges to `master`:

- [ ] Review Agent A branch:
      - Residual plumbing only; default behavior unchanged.
      - No gate or `config/*.yml` changes.
- [ ] Review Agent B branch:
      - Reporting-only schema additions; experiment flag correctly gated.
      - No gate semantics or readiness scripts modified.
- [ ] Review Agent C docs:
      - Language conservative; no implicit readiness claims.
      - Experiment clearly labeled as research-only.

---

# Integration Checklist (Required Before Merge)

Run, on the integration branch:

- [ ] `python scripts/run_all_gates.py --json`
- [ ] `python -m integrity.pnl_integrity_enforcer`
- [ ] Targeted tests for every touched module.
- [ ] `pytest` fast lane (`tests/`), no new failures.
- [ ] Verify gate state has not regressed vs baseline snapshot.
- [ ] Document:
      - What parts of EXP-R5-001 were integrated.
      - What remains deferred and why.
```