Feature Engineering Pipeline TODO (Institutional-Grade)
Evidence From Recent ETL Runs (logs/pipeline_run.log)

Frequent SARIMAX non-convergence warnings → Indicates potential feature scaling or instability issues in the feature set.

Multiple CV fold drift warnings (psi, vol_psi) → Regime shifts are not fully accounted for in the model, suggesting that we need to improve regime detection and modeling.

Missing data strategy overridden to forward fill in live mode → This indicates that the model is not handling missingness robustly. We need to include missingness features that capture gaps in the data accurately.

MSSA GPU fallback to CPU → Performance bottleneck observed for feature generation during iterative runs. We should optimize resource usage or switch to batch-based feature processing.

Visualization error (name 'datetime' is not defined) → Missing downstream diagnostics, indicating gaps in data consistency.

Sequenced TODO List (Feature Engineering)
1) Data Integrity + Feature Consistency Layer (Top Priority)

Goal: Eliminate leakage/misalignment and standardize feature windows across all tickers.

Tasks:

Enforce aligned timestamps and uniform horizons across models (no off-by-one errors).

Add missingness indicators (forward-filled segments, gap counters).

Standardize scaling metadata for each feature fold (mean/std for each feature).

Line count estimate: 120–180 LOC.

Data source validation method: Compare source row counts vs processed rows per ticker; ensure that the index is monotonic and no gaps exist beyond a defined threshold.

Performance test specification: Run pipeline with 3 tickers and assert feature count stability. Ensure no leakage warnings and measure RMSE changes vs baseline.

Failure mode handling: If alignment fails, drop to a safe baseline feature set and log an audit warning. Make necessary adjustments in the pipeline to maintain data integrity.

2) Regime/Drift Feature Block (Address CV Drift)

Goal: Make models aware of regime shifts flagged by PSI/vol_psi.

Tasks:

Add rolling volatility regime flags (low/med/high vol).

Introduce drift intensity features derived from PSI and realized volatility.

Persist drift diagnostics alongside features for auditing purposes.

Line count estimate: 80–140 LOC.

Data source validation method: Confirm that drift features exist for all folds and align with the split_drift_latest.json file for proper drift detection.

Performance test specification: Compare RMSE and directional accuracy with and without regime features on tickers like AAPL, MSFT, and NVDA.

Failure mode handling: If drift metrics are missing, default to neutral regime flags (0). This ensures that we do not introduce biases when drift data is unavailable.

3) Volatility & Tail-Risk Features (Fat-Tail Robustness)

Goal: Improve handling of fat tails and outliers without smoothing away critical signals.

Tasks:

Calculate realized volatility (rolling standard deviation of returns).

Compute downside volatility, drawdown depth, and a CVaR proxy.

Include kurtosis/skew rolling features to capture tail risk.

Line count estimate: 100–160 LOC.

Data source validation method: Verify that feature ranges are consistent and that there are no NaN or infinite values, especially for short windows.

Performance test specification: Evaluate tail-event weeks; compare hit-rate and max drawdown against the baseline model.

Failure mode handling: If feature windows are too short to capture meaningful tail events, fall back to simpler volatility-only features and log the adjustment.

4) Cross-Sectional Features (Panel Context)

Goal: Encode relative strength and liquidity within the ticker universe.

Tasks:

Compute cross-sectional rank returns over different time frames (1-day, 5-day, 20-day).

Add relative volume and turnover metrics compared to the median.

Use cross-sectional z-scores for returns and volatility to capture liquidity dynamics.

Line count estimate: 120–200 LOC.

Data source validation method: Ensure that all tickers are present for each timestamp. For missing tickers, perform median imputation and flag the missing data.

Performance test specification: Test on a universe of 10–50 tickers; measure the stability of ensemble weights and RMSE improvements.

Failure mode handling: If the cross-sectional cohort is too small, disable the features and log a fallback warning. This ensures that the model doesn't overfit when the data isn't sufficiently rich.

5) Market Microstructure (Optional for Intraday)

Goal: Improve intraday sensitivity without introducing noisy overfitting.

Tasks:

Track intraday range, ATR, and volatility-of-volatility.

Create time-of-day seasonality flags for intraday data.

Line count estimate: 80–140 LOC.

Data source validation method: Confirm that interval consistency is maintained. Reject any mixed granularity inputs that could lead to inconsistent results.

Performance test specification: Run an intraday window backtest with volatility-adjusted PnL to evaluate the impact on performance.

Failure mode handling: If interval mismatches are detected, disable microstructure features to avoid introducing noise into the model.

6) Feature Registry + Audit Hooks

Goal: Institutional auditability and reproducible feature lists.

Tasks:

Create a feature registry with versioning in the configuration file (feature sets per regime).

Persist the feature list in artifacts and database diagnostics for auditability.

Line count estimate: 100–180 LOC.

Data source validation method: Compare the persisted feature list against the runtime feature list to ensure that the features used in production match those defined in the registry.

Performance test specification: Ensure that the feature version tag appears in forecast artifacts for traceability.

Failure mode handling: If the registry is missing, fall back to the baseline feature set, and log the absence of the feature registry for further investigation.

Gating Criteria (Must Hold Before Promotion)

Backtest evidence: Ensure the model achieves >10% annual return and RMSE/DA improvements compared to the baseline model.

No leakage: Verify proper train/validation splits and ensure that feature windows are correctly aligned.

Ensemble health: Verify that the ensemble rows persist with weights, confidence, and metrics across runs. This is essential for ensuring that ensemble performance remains consistent over time.

Immediate Next Actions (Evidence-First)

Confirm ensemble persistence after a full run with tickers such as AAPL, MSFT, and NVDA.

Begin working on Feature Block #1 (Data Integrity + Feature Consistency Layer) and Feature Block #2 (Regime/Drift Feature Block). Validate the impact before introducing cross-sectional or tail-risk features.

Run `python scripts/audit_ohlcv_duplicates.py --tickers AAPL,MSFT,NVDA --export-deduped data/raw` to quantify duplicate DB rows and generate a clean snapshot for feature validation.

Fetch fresh parquet data and validate regime weights before feature work: `python scripts/fetch_fresh_data.py --tickers AAPL,MSFT,NVDA --start 2024-07-01 --end 2026-01-18 --output-dir data/raw`, then `python scripts/validate_regime_on_fresh_data.py --tickers AAPL,MSFT,NVDA --output-dir data/raw --regimes MODERATE_TRENDING,HIGH_VOL_TRENDING,CRISIS`.
