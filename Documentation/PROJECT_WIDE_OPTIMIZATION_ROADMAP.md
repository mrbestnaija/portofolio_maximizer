# Project-Wide Optimization Roadmap (TS ↔ Execution ↔ Reporting)

> **RUNTIME GUARDRAIL (WSL `simpleTrader_env` ONLY)**
> Supported runtime: WSL + Linux venv `simpleTrader_env/bin/python` (`source simpleTrader_env/bin/activate`).
> **Do not** use Windows interpreters/venvs (incl. `py`, `python.exe`, `.venv`, `simpleTrader_env\\Scripts\\python.exe`) — results are invalid.
> Before reporting runs, include the runtime fingerprint (command + output): `which python`, `python -V`, `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` (see `Documentation/RUNTIME_GUARDRAILS.md`).

**Last updated**: 2026-01-18
**Status**: Active (sequenced, evidence-driven)
**Scope**: Fix Time Series (TS) model wiring, execution realism, and reporting so “live/paper” runs are bar-aware, horizon-consistent, cost-aware, and measurable.

## Delta (2026-01-18)

- Live dashboard is now real-time (polls `visualizations/dashboard_data.json` every 5s) and uses empty states when data is missing (no embedded demo data).
- Dashboard payload includes `positions`, `price_series`, and `trade_events` for trade/price/PnL visualization; canonical producer is `scripts/dashboard_db_bridge.py` (DB→JSON) started by bash orchestrators and snapshot-persisted into `data/dashboard_audit.db` by default.
- Forecast audit monitoring dedupe fixed in `scripts/check_forecast_audits.py` (newest audit per dataset window wins).
- Profitability remediation is documented in `Documentation/CRITICAL_PROFITABILITY_ANALYSIS_AND_REMEDIATION_PLAN.md` (synthetic/test contamination, lifecycle exits). Keep future fixes aligned with that ledger.
- Workflow hygiene: remote remains canonical; rebase/push small deltas early to avoid dashboard/forecaster drift and stash conflicts.

## Delta (2026-01-29)

- Auto-trader resumes **persisted portfolio state** by default (`--resume`), allowing multi-session audit accumulation without forced liquidation (`bash/reset_portfolio.sh` for clean starts).
- Dashboard bridge filters `trade_events` to the **latest run_id by default** and backfills positions from `trade_executions` when `portfolio_positions` is empty.
- Ops helpers added for daily evidence collection: `bash/run_daily_trader.sh` and `run_daily_trader.bat`.

This roadmap is a **sequenced To‑Do list** designed to avoid disruptive rewrites. Each phase is intended to be **small, testable, and reversible**.

---

## 0. Guiding Principles (Non‑Negotiables)

- **Bar-aware, not loop-aware**: do not “trade faster” than the data supports (daily bars ≠ intraday trading).
- **Horizon-consistent decisions**: entry logic, exit logic, and evaluation horizons must match.
- **Cost-aware everywhere**: routing friction and execution simulation must use the same cost model inputs/units.
- **Evidence before tightening**: use quant validation/monitoring as *measurement* first, then promotion gates.
- **No silent optimism**: confidence/diagnostics must be tied to real predictive power, not “forecast stayed near price”.

---

## Phase Workflow (Implement → Test → Run → Review)

Use this loop after each phase so changes are incremental and auditable:

1) **Implement** a small, reversible change (prefer config flags + minimal diffs).
2) **Test** the tightest scope first:

```bash
./simpleTrader_env/bin/python -m py_compile scripts/run_auto_trader.py
./simpleTrader_env/bin/python -m pytest -q \
  tests/models/test_time_series_signal_generator.py \
  tests/models/test_signal_router.py \
  tests/execution/test_paper_trading_engine.py \
  tests/integration/test_time_series_signal_integration.py
```

3) **Run** a single-cycle paper/live check:

```bash
CYCLES=1 SLEEP_SECONDS=0 ENABLE_LLM=0 bash bash/run_auto_trader.sh
```

4) **Review** the same artifacts every time (baseline-comparable):

```bash
tail -n 1 logs/automation/run_summary.jsonl
tail -n 10 logs/automation/execution_log.jsonl
./simpleTrader_env/bin/python scripts/summarize_quant_validation.py | head -n 80
```

5) **Decide**: proceed, tighten thresholds, or roll back before moving to the next phase.

### CI stance (keep the repo mergeable)

- Default CI must run with feature flags in their safe defaults (barbell/options/sentiment disabled unless explicitly enabled) so the baseline remains stable.
- Prefer a fast-fail smoke subset first (targeted tests), then full suite/brutal runs when promoting changes.
- For dependency/security updates: keep diffs minimal, run `pip check` and `python -m pip_audit -r requirements.txt`, and ensure `CI / test` is green before merging.

---

## 1. Initial Setup and Planning

### 1.1 Review current project setup (baseline capture)

- [x] Record baseline configs + code state (paths to review):
  - `config/yfinance_config.yml` (bar interval/cadence)
  - `scripts/run_auto_trader.py` (loop + forecasting orchestration)
  - `models/time_series_signal_generator.py` (horizon/edge/confidence/quant gate)
  - `execution/paper_trading_engine.py` (slippage/cost simulation + sizing)
  - `config/execution_cost_model.yml` + `config/signal_routing_config.yml` (cost priors + thresholds)
  - `config/quant_success_config.yml` + `config/forecaster_monitoring.yml` (quant gate + monitoring tiers)

**Baseline run command (paper/live):**

```bash
CYCLES=1 SLEEP_SECONDS=0 ENABLE_LLM=0 bash bash/run_auto_trader.sh
```

**Artifacts to snapshot (baseline evidence):**

- `logs/automation/run_summary.jsonl` (last line)
- `logs/automation/execution_log.jsonl` (last ~10 lines)
- `logs/signals/quant_validation.jsonl` (summary via `scripts/summarize_quant_validation.py`)
- `visualizations/dashboard_data.json` (meta + forecaster_health + quant_validation_health)

**Baseline snapshot helper (copies configs/code + tails artefacts):**

```bash
./simpleTrader_env/bin/python scripts/capture_baseline_snapshot.py --tag phase1_baseline
```

### 1.2 Define the scope for required fixes

- [ ] **Horizon alignment**: stop using “first forecast step” as a proxy for a multi-day trade thesis.
- [ ] **Cost model alignment**: routing friction vs execution cost simulation use consistent bps/fraction semantics.
- [ ] **Reporting**: compute run-local metrics (this run) separately from DB lifetime metrics.

### 1.3 Plan iterative approach (minimize disruption)

- [ ] Implement behind **config flags** or **safe defaults** first.
- [ ] After each phase:
  - run unit/integration tests,
  - run 1-cycle auto-trader,
  - inspect the same artifacts for regressions,
  - only then proceed.

---

## 2. Bar‑Aware Trading Loop

### 2.1 Trigger only on a new bar

- [x] Modify `scripts/run_auto_trader.py` to **skip signal generation/execution** unless a new bar timestamp is observed for the ticker(s).
  - Daily data (`interval: 1d`) ⇒ at most **1 actionable cycle per day** per ticker.
  - Intraday data ⇒ gate on bar timestamp using the configured interval.
- [x] Persist “last seen bar” state (options):
  - in-memory per run (default), and/or
  - optional JSON persistence via `--persist-bar-state` (or env `PERSIST_BAR_STATE=1`) using `--bar-state-path` / `BAR_STATE_PATH`.

**Test (mock feed / deterministic window):**

- [x] Add a unit/integration test that feeds identical last-bar timestamps across two cycles and asserts **no second trade attempt** (`tests/scripts/test_bar_aware_trading_loop.py`).

**Expected outcome:**

- The system only acts on real market updates (no repeated trading on the same candle).

### 2.2 Intraday fallback (optional)

- [x] If switching to intraday, update:
  - `config/yfinance_config.yml` interval (e.g., `1h`, `30m`) and
  - signal/validator lookbacks to avoid “365-day” nonsense windows for intraday bars.

**Implemented support:**
- `scripts/run_auto_trader.py` now uses a smaller minimum lookback for intraday intervals (30 days vs 365 for daily) when building the OHLCV window.
- Optional interval override without editing config: `./simpleTrader_env/bin/python scripts/run_auto_trader.py --yfinance-interval 1h ...` (or env `YFINANCE_INTERVAL=1h`).

---

## 3. Fixing Forecast Horizon Misuse (Entry/Exit/Evaluation Consistency)

### 3.1 Align forecast target with strategy timing

- [x] In `models/time_series_signal_generator.py`, stop treating the **first forecast step** as the trade thesis.
- [x] Choose and document a consistent target (examples):
  - horizon-end price (`forecast.iloc[-1]`),
  - horizon mean/median,
  - or a weighted path metric (if calibrated).
- [x] Ensure targets/stops/holding period reflect the same horizon definition (targets/stops now follow the same horizon-end target used for expected_return).

**Test:**

- [x] Add a unit test that fixes a forecast path and asserts expected_return is computed from the chosen horizon target (`tests/models/test_time_series_signal_generator.py`).
- [x] Backtest a fixed window with the same horizon used for entry, exits, and evaluation (`scripts/run_horizon_consistent_backtest.py`).

```bash
./simpleTrader_env/bin/python scripts/run_horizon_consistent_backtest.py \
  --tickers AAPL,MSFT \
  --lookback-days 180 \
  --forecast-horizon 14
```

**Expected outcome:**

- Trades are justified by the intended time horizon and exits are not arbitrary.

### 3.2 Align `max_holding_days`

- [x] Ensure `execution.paper_trading_engine.PaperTradingEngine` lifecycle logic uses the same horizon basis:
  - `forecast_horizon` now maps to **bar-count holding** (intraday-safe); daily bars remain equivalent to days.

---

## 4. Improve Confidence Calculation (Make It Discriminative)

### 4.1 Refactor confidence to reflect predictive power

- [x] In `models/time_series_signal_generator.py`, refactor `_calculate_confidence` to:
  - penalize tiny net edges after costs,
  - avoid rewarding “forecast stayed near price”,
  - incorporate uncertainty (CI/SNR) in a calibrated way.
- [ ] Prefer model-quality inputs already available in the stack:
  - `forcester_ts` instrumentation summaries,
  - regression metrics (RMSE/sMAPE/tracking error) when available,
  - directional accuracy / hit-rate estimates.

**Test:**

- [x] Compare confidence outputs for:
  - small-edge forecasts vs large-edge forecasts,
  - high-uncertainty vs low-uncertainty,
  - and confirm confidence distribution is not pinned ~0.8–0.9.
  - Evidence: `tests/models/test_time_series_signal_generator.py` (`test_confidence_penalizes_small_net_edge`, `test_confidence_penalizes_wide_ci`).

**Expected outcome:**

- Confidence is meaningful and correlates with realized edge/hit-rate.

### 4.2 Validate across varied regimes

- [ ] Run on:
  - trend regime,
  - flat regime,
  - high-vol regime,
  - and confirm confidence is not uniformly optimistic.

---

## 5. Fix Diagnostics Scoring (No Fake “Quality Bumps”)

### 5.1 Use real diagnostics inputs

- [x] Refactor diagnostics scoring to use **real metrics** present in the forecast bundle / instrumentation:
  - realized regression metrics when available (RMSE/sMAPE/TE),
  - internal stability diagnostics (convergence flags, change-point stability),
  - and/or cross-validated summaries (preferred).

**Test:**

- [x] Add regression tests for confidence/diagnostics separation via small-edge vs large-edge and wide-CI vs narrow-CI (`tests/models/test_time_series_signal_generator.py`).

**Expected outcome:**

- Diagnostics scores no longer inflate without evidence.

### 5.2 Make poor models self-describing

- [x] Ensure signals carry a short “why low quality” summary in provenance for dashboards/logs (`models/time_series_signal_generator.py` now emits `provenance.diagnostics` + optional `provenance.why_low_quality` when the diagnostics score is low).

---

## 6. Revamp Quant Validation (Measure Forecast Edge, Not “Always Long/Short”)

### 6.1 Redesign the quant success signal

- [x] Update quant validation logic to answer: **“does this forecast add edge?”**
  - incremental performance vs baselines (random-walk / SAMOSSA baseline),
  - forecast error improvements,
  - directional accuracy conditioned on signals.

**Test:**

- [x] Add unit test for `validation_mode=forecast_edge` (monkeypatched rolling-CV metrics) to ensure the gate keys off regression metrics (`tests/models/test_time_series_signal_generator.py`).

**Expected outcome:**

- Good models aren’t rejected simply because the recent drift was negative/flat.

### 6.2 Reduce the role of directional “drift proxy”

- [x] Phase out “always long/short recent returns” as the primary validation substrate (when `validation_mode=forecast_edge`).
- [x] Keep it only as a weak prior / monitoring metric, not a gate (drift metrics still logged under `quant_validation.metrics`, but PASS/FAIL is driven by forecast-edge criteria unless `include_drift_proxy_criteria: true`).

---

## 7. Address Cost Model Mismatch (Routing ↔ Execution)

### 7.1 Align routing friction with execution costs

- [x] Ensure `models/time_series_signal_generator.py` and `execution/paper_trading_engine.py`:
  - agree on what “round trip cost” includes (spread + slippage + commissions),
  - use consistent units (bps vs fraction),
  - and share priors from `config/execution_cost_model.yml` / `config/signal_routing_config.yml`.

**Test:**

- [x] Add unit coverage for deterministic LOB fallback and microstructure-aware cost usage (`tests/execution/test_paper_trading_engine.py`).

**Expected outcome:**

- Trades that clear routing gates aren’t systematically negative after simulation costs.

### 7.2 Slippage handling review

- [x] Audit slippage in `execution/paper_trading_engine.py` and ensure it is not:
  - double-counting spread + slippage,
  - or ignoring asset-class spreads entirely.

---

## 8. Execution Simulation Refinements (LOB + Low Liquidity)

### 8.1 LOB fallback when depth data is missing

- [x] Use `config/execution_cost_model.yml` depth profiles even when live depth/spread fields are absent.
- [x] Establish a deterministic fallback:
  - infer half-spread from profile when Bid/Ask missing,
  - infer depth_notional from profile when Depth missing.

**Test:**

- [x] Simulate the same trade on a “liquid” vs “illiquid” asset-class profile and ensure slippage differs (`tests/execution/test_paper_trading_engine.py`).

**Expected outcome:**

- Execution costs reflect liquidity assumptions rather than defaulting to a generic slippage percent.

### 8.2 Keep paper trading aligned with routing

- [x] Ensure the same cost model drives both:
  - TS net-edge gating (signal generator),
  - and fill simulation (paper trading engine).

---

## 9. Reporting Fixes (Run‑Local Metrics + Forecaster Health)

### 9.1 Run-local win_rate / profit_factor

- [x] Update reporting so “this run” metrics are computed from:
  - trades executed during the run, and
  - realized PnL for those trades (and/or unrealized MTM separately).
- [x] Keep DB lifetime metrics, but label them explicitly as “lifetime”.

**Test:**

- [x] Execute a 2‑cycle run and confirm run_summary reports only those cycles/trades (run-local PF/WR now scoped by `run_id`).

**Expected outcome:**

- Live reporting reflects actual run performance, not historical DB artifacts.

### 9.2 Persist forecast regression metrics for monitoring

- [x] Persist forecast snapshots + regression metrics so `forecaster_health` is not stale:
  - persist horizon-end snapshots per bar via `etl.database_manager.DatabaseManager.save_forecast`,
  - lagged backfill evaluation writes per-row `regression_metrics`,
  - dashboards/summaries consume `get_forecast_regression_summary`.

**Tests:**

- [x] Forecast snapshot + backfill helpers (`tests/scripts/test_forecast_persistence.py`)
- [x] DB API coverage (`tests/etl/test_database_manager_schema.py`)

### 9.3 Ensemble gate outcome (2026-01-11)

- Research-profile RMSE gate (`scripts/check_forecast_audits.py --config-path config/forecaster_monitoring.yml --max-files 500`) on 27 effective audits: violation_rate=3.7% (<=25% cap) but lift_fraction=0% (<10% required) ⇒ **Decision: DISABLE ensemble as default**. `config/forecasting_config.yml` now sets `ensemble.enabled: false`; BEST_SINGLE baseline remains the source of truth until lift is demonstrated over ≥20 effective audits with sufficient lift.
- **Parameter learning policy**: TS models run in auto-learned mode only (no manual SARIMAX/GARCH orders). Performance is controlled via capped/compact search (`order_search_mode`, `max_p/max_q/...`). SARIMAX-X exogenous features are wired by default in `forcester_ts/forecaster.py`.

---

## 10. Continuous Testing and Validation

### 10.1 Unit tests

- [x] Add/extend tests for:
  - horizon-consistent expected_return (`tests/models/test_time_series_signal_generator.py`)
  - confidence discriminativeness (`tests/models/test_time_series_signal_generator.py`)
  - diagnostics scoring (`tests/models/test_time_series_signal_generator.py`)
  - cost-model alignment (`tests/execution/test_paper_trading_engine.py`, `tests/execution/test_lob_simulator.py`)

### 10.2 Backtesting (iterative)

- [x] After each phase, run a small walk-forward/backtest and compare to baseline:
  - run horizon-consistent harness: `scripts/run_horizon_consistent_backtest.py`
  - capture artefact bundle: `scripts/capture_baseline_snapshot.py` (also captures latest horizon backtest report when present)
  - compare snapshots: `scripts/compare_baseline_snapshots.py`

### 10.3 Real-time monitoring

- [x] Track and review:
  - `logs/automation/run_summary.jsonl`,
  - `logs/automation/execution_log.jsonl`,
  - `logs/signals/quant_validation.jsonl`,
  - dashboard payloads under `visualizations/`.
  - Helper CLI: `scripts/summarize_latest_run.py` (run-scoped profitability/liquidity/forecaster/next-actions + execution-log summary).

### 10.4 Feedback loop

- [x] Use the above artifacts to decide whether to tighten or relax thresholds (run_summary `next_actions` + `scripts/summarize_latest_run.py`).

---

## 11. Final Optimization (Simulated Live → Live)

- [ ] Run simulated capital with bar-aware loop enabled.
- [ ] Only then enable barbell/risk-bucket features as per:
  - `Documentation/NAV_RISK_BUDGET_ARCH.md`,
  - `Documentation/NAV_BAR_BELL_TODO.md`,
  - `Documentation/BARBELL_OPTIONS_MIGRATION.md`.

---

## 12. Documentation & Reporting

- [ ] Update docs at each phase boundary:
  - add “what changed”, “how tested”, “evidence artifact paths”.
- [ ] Maintain a baseline vs current comparison in `reports/` after major phases.

---

## 13. Final Iteration & Ongoing Monitoring

- [ ] Full review: stability + profitability + liquidity + forecasting metrics trend.
- [ ] Continue iterative updates based on recent-window evidence, not full-history noise.
