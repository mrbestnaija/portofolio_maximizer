# Portfolio Maximizer – Research Progress & Publication Plan

**Status**: Draft research log  
**Intended audience**: Future MIT‑level Master’s thesis / publication in quantitative finance / algorithmic trading  
**Last updated**: 2026-01-03  

This document tracks the Portfolio Maximizer project as a scientific research artefact rather than only as a codebase. It is meant to evolve into the backbone of a Master’s‑level thesis or journal submission, with clear hypotheses, methods, experiments, and reproducible evidence.

It complements the implementation-focused documents in `Documentation/` (e.g. `QUANTIFIABLE_SUCCESS_CRITERIA.md`, `TIME_SERIES_FORECASTING_IMPLEMENTATION.md`, `SYSTEM_STATUS_2025-10-22.md`) by giving a single, publication-oriented view. For current system status and verification evidence, see `Documentation/arch_tree.md` and `Documentation/implementation_checkpoint.md`.

---

## 0. How To Use This Document (Context + Rigor)

This file is the research-facing single source of truth for turning the codebase into publishable evidence: clear hypotheses, pre-declared evaluation protocols, reproducible artifacts, and auditable assumptions.

**What counts as evidence in this repo**

- Immutable provenance: Git commit SHA + config snapshots + environment snapshot (`requirements.txt`, Python version, OS).
- Re-runnable commands: CLI invocations that regenerate metrics, tables, and plots from logged artifacts.
- Traceable outputs: logs/reports under `logs/` and dashboards/plots under `visualizations/`.

**Where to find current system status**

- Implementation and verification: `Documentation/implementation_checkpoint.md`
- Architecture and operational map: `Documentation/arch_tree.md`
- Automation wiring: `Documentation/CRON_AUTOMATION.md`
- Evidence standards (institutional): `Documentation/CORE_PROJECT_DOCUMENTATION.md`
- Metric definitions (math): `Documentation/METRICS_AND_EVALUATION.md`

**Operational evidence-generation entry points**

- TS evidence runs: `bash/production_cron.sh auto_trader_core` (Windows wrapper: `schedule_backfill.bat`, defaults to `auto_trader_core`).
- Quant validation summaries: `scripts/summarize_quant_validation.py` (reads `logs/signals/quant_validation.jsonl` when present).
- Forecast audit/health checks: `scripts/check_forecast_audits.py` (writes/reads `logs/forecast_audits/`).

---

## 1. Research Questions & Hypotheses

**Primary research question**

- Under realistic transaction costs and execution constraints, do time-series forecasting ensembles (SARIMAX/SAMOSSA/GARCH/MSSA-RL) deliver statistically significant improvements in forecast accuracy and net risk-adjusted performance relative to mature baselines (random walk, ARIMA, historical mean/volatility) across liquid and frontier markets?

**Secondary questions**

- Does forecast combination (ensemble + convex blending) yield more stable out-of-sample performance than single-model signals across regime shifts (trend vs mean-reversion, low vs high volatility)?
- How much incremental value, if any, does an LLM fallback add once TS signals are quant-validated and cost-aware, and does it avoid degrading performance when capped as a fallback?
- Does cost-aware signal gating (quant validation + microstructure-based transaction cost model) reduce turnover and tail risk compared with ungated TS signals without materially lowering net returns?
- Are predictability and signal durability meaningfully different in frontier markets vs developed markets after controlling for liquidity and transaction costs?

**Working hypotheses**

1. Forecast combination reduces forecast error (RMSE/sMAPE) relative to any single model and to random-walk baselines, with improvements that are statistically significant under standard forecast comparison tests (e.g., Diebold-Mariano or MCS).
2. Cost-aware quant gating lowers turnover and drawdown while maintaining or improving net risk-adjusted performance (profit factor, win rate) versus ungated TS signals, especially in high-vol regimes.
3. Frontier markets exhibit higher raw predictability but weaker net performance after realistic costs unless liquidity-adjusted cost models and stricter gating are applied.
4. LLM-only signals do not consistently outperform the TS ensemble out-of-sample; when capped as a fallback, they are non-degrading and occasionally additive rather than primary.
  
Each hypothesis will need explicit acceptance/rejection based on the experiments in Section 5.

---

## 2. System Overview (Experiment Platform)

The project implements a full end‑to‑end experimental platform:

- **Data & ETL** (`etl/`):
  - Multi‑source OHLCV extraction with 24h caching.
  - Validation (`etl/data_validator.py`) and preprocessing (`etl/preprocessor.py`).
  - Storage and CV utilities (`etl/data_storage.py`, `etl/dashboard_loader.py`).
- **Forecasting & Signals**:
  - TS ensemble in `forcester_ts/` + `etl/time_series_forecaster.py`.
  - Default TS signal generator (`models/time_series_signal_generator.py`).
  - LLM signal generator (`ai_llm/signal_generator.py`) and Ollama client.
  - Signal routing (`models/signal_router.py`) and barbell‑aware risk buckets (`risk/barbell_policy.py`).
- **Execution & Portfolio**:
  - Paper trading engine (`execution/paper_trading_engine.py`).
  - Order management (`execution/order_manager.py`) with cTrader client.
  - Performance metrics and trade logging via `etl/database_manager.py`.
- **Quant Validation & Monitoring**:
  - Quant success helper and config (`config/quant_success_config.yml`).
  - Quant validation logging (`logs/signals/quant_validation.jsonl`) and CLIs:
    - `scripts/summarize_quant_validation.py`
    - `scripts/check_quant_validation_health.py`
  - Global monitoring policy (`config/forecaster_monitoring.yml`, `QUANT_VALIDATION_MONITORING_POLICY.md`).
- **Hyper‑Parameter & Post‑Evaluation Loop**:
  - Higher‑order hyperopt driver (`bash/run_post_eval.sh`).
  - Strategy optimisation (`scripts/run_strategy_optimization.py`).
  - Candidate backtesting scaffolds (`backtesting/candidate_backtester.py`, `scripts/run_backtest_for_candidate.py`).

The **canonical experiment entry points** are:

- `scripts/run_etl_pipeline.py` – deterministic ETL + forecasting runs.
- `scripts/run_auto_trader.py` – iterative paper‑trading runs (core PnL experiments).
- `bash/run_post_eval.sh` – stochastic higher‑order optimisation.

---

## 3. Experimental Design

### 3.1 Datasets & Instruments

- **Data sources**: yfinance (primary), Alpha Vantage, Finnhub (configured but optional).
- **Tickers**:
  - Core equities: AAPL, MSFT, NVDA, COOP, etc.
  - Futures/commodities: CL=F (Crude), GC=F (Gold).
  - FX: EURUSD=X and major pairs.
  - Frontier markets: curated NGX → Bulgaria list via `etl/frontier_markets.py` and `--include-frontier-tickers`.
- **Sampling**:
  - Daily bars, adjusted close prices.
  - Typical windows: 3–5 years (e.g. 2020‑01‑01 to 2024‑01‑01) for TS training; shorter rolling windows for live runs.

### 3.2 Evaluation Protocols

- **Recency-bias guardrails**:
  - Pre-declare evaluation windows (or window-selection rules) before running experiments.
  - Report results across multiple non-overlapping regimes (e.g., bull/bear, low/high vol), not only the most recent period.

- **Backtesting / TS experiments**:
  - Walk‑forward or blocked CV via ETL pipeline and TS ensemble.
  - Hyper‑parameter sweeps / bandit search via `bash/run_post_eval.sh` and `scripts/run_strategy_optimization.py`.
- **Paper‑trading experiments**:
  - `scripts/run_auto_trader.py` with:
    - `--lookback-days` (default 365).
    - `--forecast-horizon` (default 30).
    - `--initial-capital` (e.g. 25,000).
    - `--cycles` (time slices; cron wires this into periodic intraday runs).
  - Quant validation gating active (TS trades demoted to HOLD when quant_success FAILs).
- **Higher‑order Hyper‑opt**:
  - Search space over:
    - Time windows: evaluation `START` / `END` dates.
    - Quant/TS thresholds: `min_expected_profit`, `time_series.min_expected_return`, per‑ticker overrides.
  - Bandit‑style explore/exploit balancing with logs in `logs/hyperopt/hyperopt_<RUN_ID>.log`.

### 3.3 Metrics

Per‑trade and aggregate metrics, drawn from `etl/database_manager.py`, quant validation, and forecaster monitoring:

- **Return / PnL**:
  - Total profit, annualised return, average PnL per trade.
  - Equity curves via `get_equity_curve`.
- **Risk & Risk‑adjusted**:
  - Profit factor, win rate, maximum drawdown.
  - Sharpe / Sortino (TS and portfolio level).
  - CVaR (planned; see risk roadmap).
- **Forecast quality**:
  - RMSE/sMAPE vs SAMOSSA baseline (forecaster regression metrics).
  - Directional accuracy, information ratio (where benchmark data is present).
- **Quant validation tiers**:
  - PASS/FAIL fractions per ticker and globally.
  - GREEN/YELLOW/RED tiers for both per‑ticker regimes and global health (see `QUANT_VALIDATION_MONITORING_POLICY.md`).

---

## 4. Implementation Phases & Milestone Tracking

The project already uses several roadmap documents (`arch_tree.md`, `UNIFIED_ROADMAP.md`, `REFACTORING_STATUS.md`). This section aligns them with a research‑oriented view.

| Phase | Focus                                    | Implementation References                                                   | Research Status |
|------:|------------------------------------------|-----------------------------------------------------------------------------|-----------------|
| 1     | Core ETL & Validation                    | `etl/*.py`, `validation_config.yml`, `SANITY_CHECK_COMPLETE.md`             | ✅ Baseline complete |
| 2     | TS Forecasting Stack                     | `forcester_ts/*`, `TIME_SERIES_FORECASTING_IMPLEMENTATION.md`              | 🟡 Hardened; more experiments needed |
| 3     | TS Signal Generation & Routing           | `models/time_series_signal_generator.py`, `models/signal_router.py`        | 🟡 Functional; calibration in progress |
| 4     | Barbell Risk & TS‑first Architecture     | `NAV_RISK_BUDGET_ARCH.md`, `NAV_BAR_BELL_TODO.md`, `risk/barbell_policy.py`| 🟡 Design documented; limited empirical evaluation |
| 5     | LLM Integration & Monitoring             | `LLM_ETL_INTEGRATION_COMPLETE.md`, `LLM_PERFORMANCE_REVIEW.md`             | 🟡 Latency + quality constraints under study |
| 6     | Hyper‑opt & Quant Validation Automation  | `STOCHASTIC_PNL_OPTIMIZATION.md`, `QUANT_VALIDATION_AUTOMATION_TODO.md`    | 🟡 Tools scaffolded; trials pending |

For each phase, target experiments and acceptance criteria should be logged as they complete (see Section 5).

---

## 5. Experiment Log (Template)

This section is a structured log template for specific experiments. Fill it with concrete runs as you accumulate evidence.

### 5.1 Example Experiment Record (template)

- **ID**: EXP_TS_2025_001  
- **Date**: YYYY‑MM‑DD  
- **Objective**: e.g. “Compare TS ensemble vs buy‑and‑hold on AAPL/MSFT under quant validation gate, 2020‑01‑01 → 2024‑01‑01.”  
- **Setup**:
  - Command(s):  
    - `python scripts/run_etl_pipeline.py ...`  
    - `python scripts/run_auto_trader.py --tickers AAPL,MSFT ...`  
  - Config hashes: `analysis_config.yml`, `quant_success_config.yml`, `signal_routing_config.yml`.  
  - Repo commit: `git rev-parse HEAD` (and branch name if not `master`).  
  - Randomness controls: `PYTHONHASHSEED`, RNG seeds (Python/NumPy), and any hyperopt/RL sampler seeds.  
  - Data snapshot: location of raw and training parquet files.  
  - Evidence pointers: paths to the exact `logs/` report(s), quant validation JSONL slice(s), and any DB snapshot used for metrics.  
- **Results (key metrics)**:
  - Total trades, win rate, profit factor, annualised return, max drawdown.
  - Quant validation: PASS/FAIL counts, tiers by ticker (`scripts/summarize_quant_validation.py` output).  
- **Interpretation**:
  - Did the experiment support or refute the relevant hypothesis?
  - Any observed regime‑dependence (e.g. high‑vol vs low‑vol periods)?  
- **Follow‑ups**:
  - Adjust thresholds?
  - Schedule additional runs for robustness?

Populate this section with one entry per important run, using identifiers that can be cross‐linked to logs and DB snapshots.

---

## 6. Reproducibility & Environment

To target graduate‐level publication standards, every major result should be reproducible with:

- **Environment specification**:
  - Python version and OS (see `SYSTEM_STATUS_2025-10-22.md` for examples).
  - Exact `requirements.txt` snapshot.
  - Instructions for `simpleTrader_env` creation and activation.
- **Data provenance**:
  - Ticker list, data sources (yfinance / Alpha Vantage / Finnhub).
  - Date ranges and cache state (whether runs used cached or live data).
  - Seeds and randomness where applicable (for hyper‐opt and RL components).
- **Command logs**:
  - CLI invocations (ETL, auto‐trader, hyper‐opt).
  - Cron schedule snippets for long‐running experiments (see `CRON_AUTOMATION.md`).
- **Artifacts**:
  - Plots under `visualizations/`.
  - Dashboard JSON snapshots under `visualizations/dashboard_data.json`.
  - Quant validation logs under `logs/signals/quant_validation.jsonl`.
  - Hyper‐opt logs under `logs/hyperopt/`.

### 6.1 Research‐grade reproducibility checklist (MIT / Google Scholar expectations)

- **Traceable inputs**: Pin dataset snapshots by SHA256 and log acquisition dates; keep DOI/URL for any external benchmarks. Store hashes in `logs/artifacts_manifest.jsonl`.
- **Deterministic runs**: Record `PYTHONHASHSEED`, RNG seeds, and sampler seeds for hyper‐opt and RL. Prefer immutable config overrides over ad hoc flags.
- **Config immutability**: Attach commit SHA + config hashes (`quant_success_config.yml`, `signal_routing_config.yml`, `forecaster_monitoring.yml`) to every experiment entry (see Section 5).
- **Executable notebooks / scripts**: For each reported figure/table, keep a runnable script or notebook under `reproducibility/` that regenerates it from the logged artifacts.
- **External reviewability**: Include a short “Methods at a glance” appendix summarizing assumptions, constraints, and any deviations from preregistered hypotheses.
- **Archiving plan**: For publishable results, prepare a replication bundle (configs, logs, plots, and minimal sample data) suitable for Zenodo/Dataverse deposit.

### 6.2 MTM Pricing Policy Assumptions

For all current experiments that rely on `scripts/liquidate_open_trades.py` to close out open trades synthetically:

- **Default TS backtests / paper‑trading runs**:
  - Use `--pricing-policy neutral` unless otherwise stated.
  - Spot instruments (equities/ETFs/crypto/FX) are marked using the DB → vendor → entry hierarchy:
    - Latest close from `ohlcv_data` when available,
    - Otherwise best‑effort yfinance close,
    - Otherwise entry price.
- **Stress / conservative NAV scenarios**:
  - Use `--pricing-policy conservative` to avoid marking unrealised gains:
    - Long positions: `mtm_price = min(spot, entry)`,
    - Short positions: `mtm_price = max(spot, entry)`.
- **Options**:
  - Under both policies above, options are currently valued **intrinsic‑only** (no time value):
    - Calls: `max(S − K, 0)`,
    - Puts: `max(K − S, 0)`,
    - With `S` obtained via the same spot hierarchy as equities.
  - A future `bs_model` policy (Black–Scholes with realised vol) is planned but **not yet used** in reported experiments.
- **Synthetic instruments**:
  - At present, trades without explicit leg definitions are treated conservatively by marking to **entry price** only; synthetic leg decomposition is reserved for future work.

Publications and internal reports should explicitly state which `pricing_policy` was used for each experiment so MTM choices are transparent and reproducible.

Consider adding a dedicated `reproducibility/` section or appendix in the final thesis that lists experiment IDs, configs, and scripts.

---

## 7. Publication Outline (MIT‑Level Thesis / Paper)

Suggested structure for a Master’s thesis or research paper based on this project:

1. **Introduction**
   - Motivation: retail‑accessible, fully automated quantitative trading.
   - Research questions & hypotheses (from Section 1).
2. **Related Work**
   - Classical TS forecasting and quantitative trading literature.
   - Recent work on LLMs for financial decision‑making.
3. **System Architecture**
   - Overview of the 7‑layer architecture and barbell risk shell.
   - Design of the TS ensemble and signal routing.
4. **Methodology**
   - Data sources, preprocessing, validation.
   - Quant success criteria and monitoring policies.
   - Hyper‑parameter search strategy and bandit‑like optimisation.
5. **Experiments**
   - Backtests across regimes (bull, bear, sideways).
   - TS vs LLM vs hybrid routing.
   - Barbell vs non‑barbell risk profiles.
   - Hyper‑opt improvement vs static baselines.
6. **Results**
   - Summary tables and plots (PnL, Sharpe, PF, drawdown).
   - Robustness checks and ablation studies.
7. **Discussion**
   - Interpretation of results, overfitting risks, regime dependence.
   - Limitations (data, execution assumptions, LLM brittleness).
8. **Conclusion & Future Work**
   - Final assessment of hypotheses.
   - Directions for extending to live trading, options, or new asset classes.
9. **Appendices**
   - Config listings.
   - Additional plots / diagnostics.
   - Reproducibility checklist.

This document should be kept in sync with the code and other docs as the system matures, so that converting it into a formal thesis or paper is primarily a matter of polishing, not reverse‑engineering past decisions.
