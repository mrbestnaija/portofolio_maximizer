# Institutional Synthetic Data Generator (High‑Dimensional) – TODO

**Last updated**: 2025-12-18  
**Scope**: Add an institutional-grade, config-driven synthetic data generation pipeline (high-dimensional, multi-asset) as a first-class local data source for testing/regression/training runs, with persistence + versioning (DVC), rigorous validation, and cron-based refresh—while preserving backward compatibility.

This roadmap is aligned with:
- `Documentation/BRUTAL_TEST_README.md`
- `Documentation/NUMERIC_INVARIANTS_AND_SCALING_TESTS.md`
- `Documentation/CHECKPOINTING_AND_LOGGING.md`
- `Documentation/SYSTEM_ERROR_MONITORING_GUIDE.md`
- `Documentation/CRITICAL_REVIEW.md`
- `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md`
- `Documentation/QUANT_VALIDATION_AUTOMATION_TODO.md`
- `Documentation/CRON_AUTOMATION.md`
- `Documentation/STUB_IMPLEMENTATION_PLAN.md`
- `Documentation/GPU_PARALLEL_RUNNER_CHECKLIST.md`
- `Documentation/arch_tree.md`
- `Documentation/TIME_SERIES_FORECASTING_IMPLEMENTATION.md`
- `Documentation/implementation_checkpoint.md`

---

## Pre‑production directive (synthetic‑first)

Default data sources are currently inadequate for exercising the full TS/LLM pipeline and execution loop, so **pre‑production testing must run in synthetic‑first mode until the generator roadmap below is delivered and validated**.

- **Scope of use**: Run `scripts/run_etl_pipeline.py` and bash orchestrators with `--execution-mode synthetic` (and `--data-source synthetic` once the adapter lands). Prefer persisted datasets over in‑process generation so runs are reproducible; the current smoke dataset `data/synthetic/syn_685360ffc97f` passed validation (`validation.json`, quality_score=1.0) and can be used as a baseline until Phase 0 scaffolding is complete.
- **Guardrails to observe**: Follow the architectural/testing guides named here: `Documentation/arch_tree.md`, `TIME_SERIES_FORECASTING_IMPLEMENTATION.md`, `implementation_checkpoint.md`, `AGENT_INSTRUCTION.md`, `AGENT_DEV_CHECKLIST.md`, `QUANT_TIME_SERIES_STACK.md`, `NUMERIC_INVARIANTS_AND_SCALING_TESTS.md`, `CHECKPOINTING_AND_LOGGING.md`, `SYSTEM_ERROR_MONITORING_GUIDE.md`, `CRITICAL_REVIEW.md`, `QUANT_VALIDATION_MONITORING_POLICY.md`, `QUANT_VALIDATION_AUTOMATION_TODO.md`, `CRON_AUTOMATION.md`, `OPTIMIZATION_IMPLEMENTATION_PLAN.md`, `REWARD_TO_EFFORT_INTEGRATION_PLAN.md`, `SAMOSSA_IMPLEMENTATION_CHECKLIST.md`, and `SARIMAX_IMPLEMENTATION_CHECKLIST.md`.
- **Testing stance**: Treat synthetic datasets as the default input for pre‑prod/brutal runs until quant health gates turn GREEN on real data. Enforce schema/metric invariants from `NUMERIC_INVARIANTS_AND_SCALING_TESTS.md` and data‑quality checks from `TODO_data_quality.md` for every dataset_id; log evidence via the hooks in `CHECKPOINTING_AND_LOGGING.md` and `SYSTEM_ERROR_MONITORING_GUIDE.md`.
- **Execution/lifecycle**: Keep live trading disabled; auto‑trader/PaperTradingEngine demos may run on synthetic datasets for routing/execution smoke tests, but promotion to live data requires the quant validation gates defined in `QUANT_VALIDATION_MONITORING_POLICY.md` and sequencing in `NEXT_TO_DO_SEQUENCED.md` / `SEQUENCED_IMPLEMENTATION_PLAN.md`.
- **Isolation after brutal validation**: Once brutal/quant validation on synthetic passes, flip back to live‑only by (a) disabling `ENABLE_SYNTHETIC_PROVIDER`/`SYNTHETIC_ONLY`, (b) removing `--data-source synthetic`/`--execution-mode synthetic` flags from production runners, (c) ensuring `PORTFOLIO_DB_PATH` points to the production DB (not `data/test_database.db`), and (d) rejecting synthetic‑sourced rows during training/scoring/dashboard export. Synthetic datasets must never pollute live dashboards, model training data, or execution once the pre‑production validation phase ends.
- **Progress 2025-12-17 (synthetic smoke set)**: Generated dataset `syn_1dcce391f1ea` via `scripts/generate_synthetic_dataset.py --tickers AAPL,MSFT` and validated it (`logs/automation/synthetic_validation_syn_1dcce391f1ea.json`, passed=true, rows=2088). To lock pipelines/brutal runs to this dataset, export `ENABLE_SYNTHETIC_PROVIDER=1` and either `SYNTHETIC_DATASET_PATH=data/synthetic/syn_1dcce391f1ea` or `SYNTHETIC_DATASET_ID=syn_1dcce391f1ea` alongside `--execution-mode synthetic --data-source synthetic`; cron `synthetic_refresh` will pick up `CRON_SYNTHETIC_*` overrides when scheduled.
- **Progress 2025-12-17 (GPU-parallel synthetic runner)**: `bash/run_gpu_parallel.sh` now defaults to `MODE=synthetic` for parallel dataset generation/validation shards (env-driven: `SYN_CONFIG`, `SYN_OUTPUT_ROOT`, `SYN_VALIDATE`, `GPU_LIST`, `SHARD*`). Uses `ENABLE_SYNTHETIC_PROVIDER=1`/`SYNTHETIC_ONLY=1`, writes manifests under `data/synthetic/`, and keeps DB/LLM off to avoid production pollution. Switch `MODE=auto_trader` to restore prior behaviour.
- **Progress 2025-12-17 (synthetic run logging + retention)**: Synthetic generation/validation scripts now append JSONL events to `logs/automation/synthetic_runs.log` and auto-prune synthetic logs/reports older than 14 days via `scripts/prune_synthetic_logs.py` (invoked on each run). Aligns with log hygiene/retention guidance from `TIME_SERIES_FORECASTING_IMPLEMENTATION.md`, `QUANT_VALIDATION_MONITORING_POLICY.md`, and `CHECKPOINTING_AND_LOGGING.md`.
- **Progress 2025-12-18 (latest pointer)**: `scripts/generate_synthetic_dataset.py` now writes `data/synthetic/latest.json` with the newest dataset_id/path/manifest; `etl/synthetic_extractor.py` accepts `SYNTHETIC_DATASET_ID=latest` (or `SYNTHETIC_DATASET_PATH=.../latest.json`) to load the pointer for automation.
- **Progress 2025-12-18 (visual + TS stack green)**: Synthetic pipeline now runs end-to-end with SARIMAX/MSSA/GARCH enabled and visualization deps (`matplotlib`, `seaborn`) installed; persisted dataset `syn_714ae868f78b` is the latest pointer target.
- **Progress 2025-12-18 (GPU default runner)**: `bash/run_gpu_parallel.sh` auto-detects available GPUs (falls back to CPU) and wires synthetic/auto-trader shards to prefer CUDA when present, keeping synthetic-first isolation by default.
- **Progress 2025-12-18 (GAN stub wiring)**: Added `scripts/train_gan_stub.py` + `bash/run_gan_stub.sh` that honor `PIPELINE_DEVICE` for future TimeGAN/diffusion training; currently a device-aware no-op consuming persisted synthetic data.
- **Context (liquidity gaps)**: Synthetic-first pre-production remains mandatory because free-tier data sources lack depth for commodity/illiquid classes. All testing/regression must use persisted synthetic datasets until quant health on live data turns GREEN; align with `NAV_RISK_BUDGET_ARCH.md`, `NAV_BAR_BELL_TODO.md`, `BARBELL_OPTIONS_MIGRATION.md`, `MTM_AND_LIQUIDATION_IMPLEMENTATION_PLAN.md`, and `OPTIMIZATION_IMPLEMENTATION_PLAN.md` to keep risk gates intact while using synthetic inputs.
- **Progress 2025-12-19 (microstructure validation)**: Synthetic extractor now validates microstructure channels (`Spread`, `Slippage`, `Depth`, `OrderImbalance`) for non-negative/finite values, and unit tests assert these columns exist when microstructure is enabled. Config `config/synthetic_data_config.yml` retains depth/order-flow knobs plus liquidity shock events to emulate illiquid markets.
- **Progress 2025-12-17 (pipeline smoke, dataset_id propagation)**: `scripts/run_etl_pipeline.py --execution-mode synthetic --data-source synthetic` exercised with `SYNTHETIC_DATASET_ID=syn_1dcce391f1ea`; pipeline logged dataset_id/generator_version in `pipeline_events` and checkpoints. Latest smoke (`pipeline_20251217_220920`) completed end-to-end on syn_1dcce391f1ea after restoring `numpy`/`scipy`/`pyarrow`; warnings were limited to CV fold overlap/drift and missing viz deps (`kiwisolver`) while time-series/validation/storage/signals/routing all passed.
- **Progress 2025-12-17 (CLI synthetic selection)**: `scripts/run_etl_pipeline.py` now accepts `--synthetic-dataset-id`, `--synthetic-dataset-path`, and `--synthetic-config` CLI options (bridge to env), enabling persisted synthetic dataset selection without manual env exports.
- **Progress 2025-12-17 (synthetic default provider)**: `config/data_sources_config.yml` now prioritizes the `synthetic` provider (priority=1) so synthetic is the default source for all runs unless explicitly overridden; live providers remain enabled as lower-priority fallbacks.
- **Progress 2025-12-19 (Phase 2/3 hooks)**: Synthetic generator now emits liquidity-proxy channels (Depth, OrderImbalance) and a `liquidity_shock` event in `event_library`, plus configurable microstructure depth/imbalance knobs in `config/synthetic_data_config.yml` to better exercise illiquid classes and spread/impact logic without live feeds.

---

## 0) Non‑Negotiables (Design Contract)

- [ ] **Backward compatible defaults**
  - Keep `scripts/run_etl_pipeline.py --execution-mode synthetic` behaviour stable (deterministic, in-process OHLCV) unless explicitly overridden by config.
  - Add new synthetic generator behind a feature flag / config switch (no breaking CLI changes).
- [ ] **No hardcoding**
  - Every generator choice (models, regimes, correlations, tickers, frequencies, seeds, microstructure, GAN training) must be driven by YAML + env overrides + CLI flags (no constants embedded in code paths).
- [ ] **Modular + self-contained**
  - Synthetic generator must plug into `etl/data_source_manager.py` via the existing adapter registry mechanism (provider = `synthetic`) and continue to work in `.bash/` and `bash/**` runners.
- [ ] **Local-first (no paid external services)**
  - All persistence, versioning, and model training must work locally (DVC remote can be local filesystem).
- [ ] **Reproducible + versioned artifacts**
  - Every dataset is identified by a **dataset_id** (config hash + seed + generator version + git SHA) and can be regenerated deterministically.
- [ ] **Institutional QA**
  - Statistical validation + unit tests for each model phase; integration tests for pipeline and bash runners; monitoring hooks and logs compatible with existing “brutal” and quant validation policies.

---

## 1) Target Integration Outcome (What “Done” Looks Like)

- [ ] `synthetic` appears as a provider in `config/data_sources_config.yml` and can be selected via:
  - `--data-source synthetic` (forced), or
  - `--execution-mode synthetic` (offline), or
  - `--execution-mode auto` with live failure fallback (existing behaviour), and optionally
  - `--synthetic-dataset <dataset_id>` / `--synthetic-config config/synthetic_data_config.yml` (new; optional).
- [ ] Synthetic generation supports:
  - multi-asset correlation (factor / copula / DCC-like approximation),
  - regime switching,
  - volatility clustering (GARCH / stochastic volatility),
  - jump/crash events,
  - microstructure proxies (spread, slippage, liquidity, order flow) as optional side channels,
  - high-dimensional features (indicators + latent factors),
  - ML-based generator (TimeGAN-style) as an optional backend.
- [x] Synthetic datasets are persisted to `data/synthetic/` as partitioned parquet + manifest, optionally tracked with DVC; “latest” dataset pointer is maintained for automation.
- [ ] Cron task refreshes synthetic datasets and runs validation (stat checks + smoke pipeline), producing audit artifacts under `logs/automation/` and `logs/forecast_audits/`.

---

## 2) Proposed Repository Additions (Modular Layout)

- [ ] **New config**
  - `config/synthetic_data_config.yml` (single source of truth for generator)
  - Optional: `config/synthetic_data_profiles.yml` (named presets: efficient/inefficient/crisis/microstructure-heavy)
- [ ] **New ETL adapter**
  - `etl/synthetic_extractor.py` implementing `BaseExtractor`:
    - `extract_ohlcv()` returns canonical OHLCV (+ ticker column or MultiIndex per existing conventions)
    - `validate_data()` runs synthetic-specific schema + invariants
    - `get_metadata()` returns dataset_id, generator_version, config_hash, seed, regimes used
- [ ] **New generator package**
  - `etl/synthetic_data/` (or `synthetic/`) as a self-contained module:
    - `config.py` (load/validate YAML, env overrides)
    - `dataset_store.py` (persist parquet + manifest + “latest” pointer)
    - `models/` (GBM, OU, Jump, Heston, GARCH, regime switching, correlation)
    - `microstructure/` (spread/slippage/order flow simulators)
    - `features/` (indicators + factor features)
    - `calibration/` (fit params from real data when available)
    - `ml/` (GAN backends, optional deps)
    - `validation/` (stat tests + invariant checks)
- [ ] **New scripts**
  - `scripts/generate_synthetic_dataset.py` (CLI: generate/persist/validate; prints dataset_id)
  - `scripts/validate_synthetic_dataset.py` (CLI: validate a dataset_id and emit report JSON)
  - Optional: `scripts/serve_synthetic_api.py` (local FastAPI service; optional dependency)

---

## 3) Configuration Schema (No Hardcoding)

- [ ] Define YAML schema (and validate with a strict loader) for:
  - **dataset**
    - `dataset_id_strategy`: `hash` | `timestamped_hash`
    - `seed`, `start_date`, `end_date`, `frequency` (`B`, `1min`, etc)
    - `tickers` (explicit) OR `universe_profile` (derive from existing discovery/universe config)
  - **generation**
    - `market_condition`: `efficient` | `inefficient` | `mixed`
    - `regime_switching`: on/off + transition matrix / expected durations
    - `correlation`: `static` | `factor` | `rolling` (DCC-like) + target correlations
    - `price_model`: `gbm` | `ou` | `jump_diffusion` | `heston` | `hybrid`
    - `volatility_model`: `none` | `garch` | `stochastic_vol`
    - `event_library`: crash/flash-crash/geopolitical shock toggles + intensities
  - **microstructure (optional)**
    - spread model, depth/liquidity curve, slippage model, order flow (Hawkes optional)
  - **features**
    - indicator set (MA/RSI/MACD/etc) + horizons
    - factor features (market/sector/commodity factors)
  - **ml_generator (optional)**
    - backend `timegan`/`c-rnn-gan`/`diffusion_ts` (pluggable)
    - training config (epochs, batch size, window length, conditioning)
    - checkpoint path, device selection (cpu/cuda), determinism flags
  - **persistence**
    - output root `data/synthetic/`
    - format `parquet`, partitioning by ticker/frequency/date
    - retention/rotation (keep last N datasets)
  - **versioning (DVC)**
    - `enabled: true|false`
    - `remote_name`, `remote_url` (local path), `auto_add`, `auto_push` (default false)
  - **validation**
    - which statistical tests run, thresholds/tolerances, and “fail-fast” behaviour

---

## 4) Phased Implementation TODO (Config-Driven, Backward Compatible)

### Phase 0 — Scaffolding + Compatibility Layer

- [x] Add `synthetic` provider entry to `config/data_sources_config.yml` and adapter registry mapping to `etl.synthetic_extractor.SyntheticExtractor` (enabled at lowest priority so `--data-source synthetic` works without env toggles).
- [x] Implement `etl/synthetic_extractor.py` that:
  - [x] Supports deterministic generation with current behaviour as `generator_version=v0` (wrap existing `generate_synthetic_ohlcv` logic).
  - [x] Adds opt-in path for `generator_version=v1+` using the new modular generator package.
  - [x] Emits metadata via `DataFrame.attrs` (source, dataset_id, generator_version; manifest added by CLI).
- [x] Add `config/synthetic_data_config.yml` with a minimal “v0 compatible” profile (seed=123, GBM-like returns).
- [x] Add `scripts/generate_synthetic_dataset.py`:
  - [x] Generates a dataset per config, persists to `data/synthetic/<dataset_id>/...`, writes `manifest.json`.
  - [x] Produces a stable `dataset_id` and prints it for bash/cron wiring.
- [x] Add `scripts/validate_synthetic_dataset.py` for manifest-aware validation reports (per dataset_id).
- [x] Ensure `scripts/run_etl_pipeline.py` can:
  - [x] Load a persisted dataset by `dataset_id`/`dataset_path` (env) before regenerating.
  - [x] Fall back to in-process generation if no dataset is provided (backward compatible).
- [x] Add pipeline logging hooks with dataset_id/generator_version recorded in `PipelineLogger` during data_extraction + pipeline completion (checkpoint propagation optional for future phases).

### Phase 1 — Core Market Simulation Models (Institutional Baseline)

- [x] Implement core generators (each is a pluggable component selected by config):
  - [x] **GBM** (lognormal prices; drift/vol per asset)
  - [x] **Ornstein–Uhlenbeck** (mean reversion, commodity/rates proxy)
  - [x] **Jump Diffusion (Merton)** (Poisson jumps + jump size distribution)
  - [x] **Stochastic Volatility (Heston-like)** (vol process + leverage correlation)
  - [ ] Optional: **agent-based micro model** stub (for future; keep behind feature flag)
- [x] Implement **regime switching**:
  - [x] Markov chain over regimes (low/high vol; trend/mean-revert; liquidity regimes)
  - [x] Regime-conditioned parameters for drift/vol/jump intensity/spread
- [x] Implement **multi-asset dependence**:
  - [x] Static correlation via Cholesky on Gaussian shocks (baseline)
  - [x] Factor model (latent market/sector/commodity factors)
  - [ ] Optional: copula-based dependence for tail correlation (advanced)

### Phase 2 — Crises, Shocks, and Market Sessions

- [ ] Add event library with configurable intensity:
  - [x] flash crash, volatility spike, gap risk, bear-drift shift (baseline hooks implemented)
  - [ ] macro regime changes (rates up/down; commodity shock)
- [ ] Market hours/sessions model:
  - [ ] intraday seasonality for volume/spread/vol (if frequency < daily)
  - [ ] weekend/holiday calendar alignment when appropriate

### Phase 3 — Microstructure & Execution Proxies (Optional Channel)

- [ ] Order book proxy generator:
  - [ ] bid/ask spread series (state-dependent)
  - [ ] depth/liquidity proxy (affects slippage)
  - [ ] order-flow imbalance (optional Hawkes process)
- [ ] Execution simulator hooks:
  - [ ] slippage as function of volatility + liquidity + order size
  - [ ] transaction cost model outputs compatible with `scripts/estimate_transaction_costs.py`

### Phase 4 — Feature Engineering for High Dimensionality

- [ ] Generate technical indicators and engineered features:
  - [ ] MA/EMA, RSI, MACD, Bollinger bands, ATR, rolling vol, volume features
  - [ ] cross-asset spreads, cointegration proxies, factor exposures
- [ ] Persist features as separate parquet tables to avoid bloating OHLCV frames:
  - [ ] `data/synthetic/<dataset_id>/features/<ticker>.parquet`
  - [ ] Manifest references feature schema + generation config

### Phase 5 — Calibration to Real Data (Local-Only)

- [ ] Fit parameters from historical data already available in the project:
  - [ ] Use `data/portfolio_maximizer.db` and/or `data/raw/` parquet caches when present
  - [ ] Implement method-of-moments + MLE fits (drift/vol; OU theta; jump intensity/size)
  - [ ] Calibrate correlation/factor loadings from empirical returns
- [ ] Create calibration reports:
  - [ ] write JSON to `logs/automation/synthetic_calibration_report.json`
  - [ ] include fitted params, confidence intervals (bootstrap), and goodness-of-fit stats

### Phase 6 — ML Generator (GAN Family; Optional Dependency)

- [ ] Add an optional ML backend (do not force into base requirements):
  - [ ] Create `requirements-ml.txt` (or extras) for `torch` + any GAN utilities
  - [ ] TimeGAN-style sequence generator for returns/features with conditioning:
    - regime label
    - asset class
    - volatility bucket
- [ ] Training pipeline:
  - [ ] Local training script that consumes real + simulated data to learn residual structure
  - [ ] Checkpointing (model weights + optimizer state) to `models/synthetic/<run_id>/`
  - [ ] Determinism controls (seed, cudnn deterministic flags, reproducible dataloader)
- [ ] Inference pipeline:
  - [ ] Generate synthetic returns/features; recompose into OHLCV consistent with invariants
  - [ ] Export the same manifest metadata (model hash, commit SHA, config hash)

### Phase 7 — Automation (Cron Refresh + Evidence Artifacts)

- [ ] Add a new cron task to `bash/production_cron.sh`:
  - [ ] `synthetic_refresh` (generates a dataset, validates it, emits artifacts)
  - [ ] Env overrides: `CRON_SYNTHETIC_CONFIG`, `CRON_SYNTHETIC_PROFILE`, `CRON_SYNTHETIC_SEED`, `CRON_SYNTHETIC_OUTPUT_ROOT`
- [ ] Update `Documentation/CRON_AUTOMATION.md` with example crontab entries:
  - [ ] nightly dataset refresh (off-hours)
  - [ ] weekly “crisis scenario pack” refresh
- [ ] Ensure outputs land in:
  - [ ] `data/synthetic/<dataset_id>/...`
  - [ ] `logs/automation/synthetic_validation_<dataset_id>.json`
  - [ ] `logs/automation/synthetic_manifest_<dataset_id>.json`

### Phase 8 — DVC Data Versioning + Persistence (Local Remote)

- [ ] Add DVC without breaking existing workflows:
  - [ ] `dvc init` (repo-local)
  - [ ] `dvc remote add -d localstore <LOCAL_PATH>` (filesystem remote)
  - [ ] Track:
    - [ ] `data/synthetic/` datasets
    - [ ] `models/synthetic/` GAN checkpoints
    - [ ] optional large calibration reports
  - [ ] Commit `.dvc/`, `.dvcignore`, and `.dvc` files to git
- [ ] Add “operator docs”:
  - [ ] How to regenerate a dataset from a manifest
  - [ ] How to `dvc pull` / `dvc push` using a local remote

---

## 5) Testing Plan (Unit + Statistical + Integration)

### 5.1 Unit tests (deterministic invariants)

- [ ] Add tests under `tests/synthetic/` for:
  - [ ] Schema invariants:
    - OHLCV columns exist; datatypes sane; index monotonic; no duplicates
    - `High >= max(Open, Close)` and `Low <= min(Open, Close)` for every row
    - `Volume >= 0` (int-like), no NaNs in required fields
    - No negative prices; no infinite values
  - [ ] Backward compatibility:
    - `generator_version=v0` matches legacy `generate_synthetic_ohlcv` output within tolerance under fixed seed
  - [ ] Persistence invariants:
    - manifest contains config hash, dataset_id, generator version, git sha (when available)
    - load/save roundtrip yields identical data (or stable hash)

### 5.2 Statistical validation (scientific rigor)

All “distributional” tests must be written to be stable under finite samples:
- Prefer confidence intervals and tolerance bands over brittle point equality.
- For Monte Carlo validations, keep runtime bounded and seed fixed (or allow a CI-friendly reduced mode).

#### A) GBM (Geometric Brownian Motion)

Model:
`dS_t = μ S_t dt + σ S_t dW_t`  
Log-returns: `r_t = log(S_t/S_{t-1}) ~ Normal((μ - 0.5σ^2)Δt, σ^2 Δt)`

Tests:
- [ ] **Mean/variance consistency**: estimate `E[r]` and `Var[r]` and compare to theoretical values within CI.
- [ ] **Normality of log-returns**: KS test / Anderson–Darling (configured alpha), or moment checks (skew≈0, kurtosis≈3 within tolerance).

#### B) OU (Ornstein–Uhlenbeck)

Model:
`dX_t = θ(μ - X_t)dt + σ dW_t`

Tests:
- [ ] **Stationarity**: Augmented Dickey–Fuller test rejects unit root for mean-reverting configs (p < α).
- [ ] **Mean reversion**: autocorrelation at small lags consistent with negative/decaying structure; estimated half-life `ln(2)/θ` within tolerance.

#### C) Jump Diffusion (Merton)

Model:
`dS_t = μ S_t dt + σ S_t dW_t + J S_t dN_t`, `N_t ~ Poisson(λt)`

Tests:
- [ ] **Jump count**: observed jump occurrences over T follow Poisson(λT) (Chi-squared / overdispersion checks).
- [ ] **Jump size distribution**: validate configured jump size distribution (lognormal/normal) via KS test on jump magnitudes.
- [ ] **Tail behaviour**: excess kurtosis > GBM baseline when jumps enabled.

#### D) Stochastic Volatility (Heston-like)

Model:
`dS_t = μ S_t dt + sqrt(v_t) S_t dW_t`  
`dv_t = κ(θ - v_t)dt + ξ sqrt(v_t) dZ_t`

Tests:
- [ ] **Positivity**: `v_t >= 0` always; enforce discretization that preserves non-negativity.
- [ ] **Volatility clustering**: ACF of squared returns decays slowly; Ljung–Box on squared returns rejects white noise (p < α).
- [ ] Optional: validate approximate Feller condition `2κθ > ξ^2` when required by config.

#### E) ARCH/GARCH volatility clustering

Model (GARCH(1,1)):
`σ_t^2 = ω + α ε_{t-1}^2 + β σ_{t-1}^2`

Tests:
- [ ] **ARCH effect**: Engle ARCH test detects heteroskedasticity when enabled.
- [ ] **Unconditional variance**: empirical variance approximates `ω/(1-α-β)` for stationary configs (α+β < 1).

#### F) Regime switching (Markov)

Model:
Regime `R_t` with transition matrix `P`; parameters depend on `R_t`.

Tests:
- [ ] **Transition validity**: rows of `P` sum to 1; empirical transition frequencies approximate `P` within tolerance.
- [ ] **Regime separability**: realized vol/returns differ materially between regimes (effect size threshold).

#### G) Multi-asset dependence (correlations)

Tests:
- [ ] **Target correlation match**: sample correlation matrix close to target (Frobenius norm / max abs error tolerance).
- [ ] **PSD checks**: target correlation is positive semidefinite; repaired if needed (nearest-PSD).
- [ ] **Tail dependence** (if copula enabled): co-crash frequency above Gaussian baseline.

#### H) Microstructure proxies (spread/slippage/order flow)

Tests:
- [ ] **Spread distribution**: strictly positive; median/quantiles match config; heavier tails in “inefficient” regimes.
- [ ] **Slippage monotonicity**: slippage increases with order size and decreases with liquidity proxy.
- [ ] **Order-flow autocorrelation**: short-memory or Hawkes-like clustering if enabled.

### 5.3 Integration tests (pipeline + bash + cron)

- [ ] `scripts/run_etl_pipeline.py` end-to-end in synthetic mode using:
  - [ ] in-process generator (legacy, v0)
  - [x] persisted dataset (`--synthetic-dataset <id>`)
  - [x] `--data-source synthetic` via DataSourceManager adapter
- Note: `pipeline_20251217_220920` ran end-to-end on `syn_1dcce391f1ea` with the synthetic provider; viz deps (`kiwisolver`) are still absent.
- [ ] Brutal harness integration:
  - [ ] Add a brutal stage that validates generator invariants and runs a short pipeline smoke test (align with `Documentation/BRUTAL_TEST_README.md`).
- [ ] Cron smoke:
  - [ ] `bash/bash/production_cron.sh synthetic_refresh` runs locally and emits logs/artifacts.

---

## 6) Documentation Deliverables

- [ ] Add a user guide:
  - [ ] How to generate a dataset locally
  - [ ] How to select it in pipeline runs (`--execution-mode synthetic`, `--data-source synthetic`, `--synthetic-dataset`)
  - [ ] How to validate datasets and interpret validation reports
- [ ] Update existing docs to reference synthetic generator where appropriate:
  - [ ] `Documentation/CRON_AUTOMATION.md` (cron tasks)
  - [ ] `Documentation/implementation_checkpoint.md` (offline regression procedure)
  - [ ] `Documentation/arch_tree.md` (new module and staged rollout)
- [ ] Add runbooks for DVC local remotes and dataset reproduction.

---

## 7) Promotion / Defaulting Rules (When to Use Synthetic vs Live)

- [ ] Keep synthetic as the **default for regression/brutal/CI** until “profitability criteria” are exceeded on real (or paper) trading per:
  - `Documentation/QUANTIFIABLE_SUCCESS_CRITERIA.md`
  - `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md`
- [ ] Add a config-driven gate for automation:
  - [ ] If global quant health is RED (`scripts/check_quant_validation_health.py`), cron jobs should prefer synthetic-only runs for evidence rebuilding (no live execution).
  - [ ] If YELLOW, allow limited live extraction but keep synthetic scenario pack refresh running to stress strategies.
  - [ ] If GREEN, enable broader live runs, but keep synthetic refresh as a continuous regression guardrail.

---

## 8) Security / Governance Notes (Local)

- [ ] Do not store secrets in synthetic configs; keep parity with `Documentation/API_KEYS_SECURITY.md`.
- [ ] Ensure manifests/logs do not include API keys or private paths; only relative paths and hashes.
- [ ] Keep “synthetic performance” clearly labeled to avoid conflating with realised PnL (align with `Documentation/CRITICAL_REVIEW.md`).
