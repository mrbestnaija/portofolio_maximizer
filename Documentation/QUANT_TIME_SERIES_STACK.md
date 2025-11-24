Here’s a curated, institution-grade stack you can run fully locally on CPU/GPU, formatted in Markdown so you can drop it straight into your AI-companion’s knowledge base.

Quant & Time-Series Stack (Local CPU/GPU)

> **Nov 16, 2025 Update**  
> The Tier-1 stack now includes the hardened SQLite + statsmodels settings shipped alongside `etl/database_manager.py`, `forcester_ts/forecaster.py`, and `forcester_ts/sarimax.py`. Warning streams from these components are no longer suppressed—they are forwarded to `logs/warnings/warning_events.log` via `etl/warning_recorder.py`, and `forcester_ts/instrumentation.py` emits JSON audits (dataset shape, statistics, timing) for interpretable forecasting when `TS_FORECAST_AUDIT_DIR` is configured.

Focus: data science, ML, and statistical modeling for time series / investments / portfolios, widely used in academia and industry, all runnable locally.

> **Nov 24, 2025 Delta**  
> - Data-source-aware ticker resolver (`etl/data_universe.py`) added; auto-trader now resolves tickers via helper (explicit + frontier default, provider discovery when empty).  
> - LLM fallback defaults to enabled in trading runs; thresholds unchanged.  
> - Dashboard JSON emission hardened (ISO timestamps) to stop serialization warnings in live loops.

## Tiered Stack Overview (Project-Aligned)

> **Reward-to-Effort Integration:** For automation, monetization, and sequencing work, align with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`.

| Tier | Use Case | Components | Notes |
| --- | --- | --- | --- |
| Tier-1 | Mandatory for brutal suite + SARIMAX/SAMOSSA/MSSA pipelines | Python ≥3.10, NumPy, pandas, SciPy, statsmodels, arch, CuPy (optional), Matplotlib/Seaborn | Fits on a single 8 GB GPU or CPU-only dev box. |
| Tier-2 | Extended experimentation (Prophet, Nixtla neuralforecast, PyTorch/TensorFlow) | Adds torch, sktime, neuralforecast, RAPIDS/cuDF, mlflow | Only enable when business case proven; heavier GPU/RAM. |
| Tier-3 | Research/backtesting backlog | Adds FinRL, PyMC/Stan, mlfinlab stack, distributed tooling | Use offline so Phase limits stay intact. |

The ETL + forecasting codebase (`forcester_ts/*`, `etl/time_series_forecaster.py`, brutal harness) must treat Tier-1 as the default bill-of-materials. Higher tiers are opt-in per milestone gates documented in `AGENT_INSTRUCTION.md`.

### Tier-1 (Lean GPU Baseline)

- **Runtime**: Python 3.10–3.12 (aligns with `simpleTrader_env`) with `poetry`/`pip-tools` locking enabled.
- **Numerics**: NumPy + SciPy + pandas (CPU) and CuPy (optional) for accelerating MSSA/Hankel ops on laptops with a single NVIDIA GPU.
- **Classical TS Models**: statsmodels (SARIMAX), arch (GARCH), our in-house SAMOSSA/MSSA-RL modules.
- **Visualization / Diagnostics**: Matplotlib, seaborn, Plotly (static fallback) to keep `scripts/run_etl_pipeline.py` charts working offline.
- **Workflow**: JupyterLab / VS Code notebooks, pytest, brutal suite harness.

Deployments or tests targeting “lean GPU” build containers should pin to the Tier-1 list above; everything else is deferred until the AI assistant can justify cost/complexity per `QUANTIFIABLE_SUCCESS_CRITERIA.md`.

### SAMoSSA vs. SARIMAX — Spatial/Frequency Treatment

| Aspect | **SAMoSSA / mSSA** | **SARIMAX** |
| --- | --- | --- |
| Spatial-Structural Relationship | Uses stacked **Page** or **Hankel** matrices to retain cross-series structure before HSVT; the multivariate Page matrix `Zf` captures harmonics shared across tickers (per `SAMOSSA_IMPLEMENTATION_CHECKLIST.md`). | No explicit spatial stack; focuses on temporal dynamics of a single series via AR/MA blocks and optional exogenous regressors. |
| Frequency Handling | Frequency components extracted directly from the Page/Hankel decomposition; deterministic trends vs. AR noise are separated before forecasting. | Frequency inferred from the pandas index; mapped to seasonal periods (e.g., B → 5, M → 12) so seasonal ARIMA orders represent the cadence described in this stack. |
| Noise Modelling | Residual AutoReg fits operate on the stationary component after HSVT, respecting the same index frequency. | Seasonal differencing (D) + seasonal AR/MA (P/Q) capture periodic behaviour; warnings in statsmodels stay quiet only when frequency metadata is attached. |

When implementing either forecaster, ensure the chosen technique honours the frequency/seasonality expectations encoded here; refer to the respective checklists for validation steps.

**Contribution reminder**: Before proposing or merging any dependency changes, re-read this Tier-1 section and document exactly why the existing stack cannot satisfy the requirement. PRs that add time-series libraries without referencing this file will be rejected during review.

### AI-Companion Config Snippets

Provide the stack to any AI copilot/agent so prompts stay grounded in the approved tooling.

```yaml
ai_companion:
  knowledge_base:
    - Documentation/QUANT_TIME_SERIES_STACK.md
  recommended_stack:
    tier: Tier-1
    python: ">=3.10,<3.13"
    packages:
      core: [numpy, pandas, scipy]
      tsa: [statsmodels, arch]
      project_modules: [forcester_ts, etl.time_series_forecaster]
      optional_gpu: [cupy]
  guardrails:
    upgrade_path: "Escalate to Tier-2 only after profitability proof + GPU budget approval."
```

```json
{
  "ai_companion": {
    "knowledge_base": [
      "Documentation/QUANT_TIME_SERIES_STACK.md"
    ],
    "recommended_stack": {
      "tier": "Tier-1",
      "python": ">=3.10,<3.13",
      "packages": {
        "core": ["numpy", "pandas", "scipy"],
        "tsa": ["statsmodels", "arch"],
        "project_modules": ["forcester_ts", "etl.time_series_forecaster"],
        "optional_gpu": ["cupy"]
      }
    },
    "guardrails": {
      "upgrade_path": "Escalate to Tier-2 only after profitability proof + GPU budget approval."
    }
  }
}
```

Embed either snippet in `config/ai_companion.yml` or your automation harness so that every autonomous agent references the same stack without improvising dependencies.

1. Core Scientific Python Stack
1.1 Python (Language / Runtime)

Purpose: Primary language for quant research, DS/ML, scripting, and orchestration.

Homepage: https://www.python.org

Docs: https://docs.python.org/
 
Python.org

1.2 NumPy

Purpose: N-dimensional arrays, linear algebra, vectorized computation (base of most scientific Python).

CPU/GPU: CPU; pair with CuPy for GPU.

Homepage: https://numpy.org

Docs: https://numpy.org/doc/

1.3 SciPy

Purpose: Numerical optimization, linear algebra, statistics, signal processing; often used for maximum-likelihood fits in econometrics/time series.

CPU/GPU: CPU.

Homepage: https://scipy.org

Docs: https://docs.scipy.org/doc/scipy/

1.4 pandas

Purpose: Tabular time-indexed data, resampling, joins/merges; de-facto standard for financial time-series preprocessing. 
Pandas

CPU/GPU: CPU; GPU analogue via cuDF (RAPIDS).

Homepage: https://pandas.pydata.org

Docs: https://pandas.pydata.org/docs/
 
Pandas

2. Statistical Modeling & Time-Series Analysis
2.1 statsmodels

Purpose: Classical statistics & econometrics, including robust time-series models: AR, ARMA, ARIMA, VAR, state-space models, Kalman filters, etc. 
Statsmodels
+1

CPU/GPU: CPU (pure Python + NumPy/SciPy).

Homepage: https://www.statsmodels.org

Time-Series Docs: https://www.statsmodels.org/stable/tsa.html
 
Statsmodels

User Guide: https://www.statsmodels.org/stable/user-guide.html
 
Statsmodels

2.2 pmdarima

Purpose: Automated ARIMA/SARIMA/SARIMAX model selection (auto_arima) for univariate forecasting.

CPU/GPU: CPU.

Homepage / Docs: https://alkaline-ml.com/pmdarima/

2.3 Prophet (Facebook / Meta Prophet)

Purpose: Additive time-series model with trend/seasonality/holidays; widely used in business forecasting.

CPU/GPU: CPU (uses Stan under the hood); can leverage multi-core.

Homepage / Docs: https://facebook.github.io/prophet/

2.4 sktime

Purpose: Unified framework for time-series forecasting, classification, and regression; wrappers for statsmodels, arch, etc. 
Sktime

CPU/GPU: CPU; some estimators can use GPU via underlying libs.

Homepage / Docs: https://www.sktime.net

2.5 Nixtla statsforecast / neuralforecast

statsforecast

Purpose: Fast classical forecasting (ARIMA, ETS, Theta, etc.) with efficient implementations. 
Google Colab

Docs: https://nixtla.github.io/statsforecast

neuralforecast

Purpose: Deep learning forecasters (N-BEATS, TFT, etc.) for time-series.

CPU/GPU: CPU + GPU (PyTorch backend).

Docs: https://nixtla.github.io/neuralforecast

2.6 arch (Kevin Sheppard)

Purpose: ARCH/GARCH and related volatility models, unit-root and cointegration tests; heavily used in empirical finance. 
Arch Documentation
+2
PyPI
+2

CPU/GPU: CPU.

Homepage / Docs: https://arch.readthedocs.io

Intro to ARCH/GARCH: https://arch.readthedocs.io/en/stable/univariate/introduction.html
 
Arch Documentation

2.7 linearmodels

Purpose: Panel data models, instrumental variables, asset-pricing models (Fama-French, etc.).

CPU/GPU: CPU.

Docs: https://bashtage.github.io/linearmodels-doc/

3. ML Frameworks (General, but Essential for Quant)
3.1 scikit-learn

Purpose: Core ML algorithms (linear models, trees, ensembles, clustering, metrics); baseline models for financial prediction.

CPU/GPU: CPU; some drop-in GPU accelerations via RAPIDS cuML.

Homepage / Docs: https://scikit-learn.org/stable/

3.2 PyTorch

Purpose: Deep learning / tensor library, dynamic computation graphs; widely used in finance research for sequence models and RL.

CPU/GPU: CPU & GPU (CUDA, ROCm).

Homepage: https://pytorch.org

Docs: https://pytorch.org/docs/stable/

3.3 TensorFlow / Keras

Purpose: Deep learning framework, static + eager execution; Keras API used for time-series nets, LSTMs, transformers, etc.

CPU/GPU: CPU & GPU.

Homepage: https://www.tensorflow.org

Docs: https://www.tensorflow.org/api_docs

3.4 JAX

Purpose: Autodiff and XLA-compiled NumPy; excellent for custom probabilistic models and differentiable simulations.

CPU/GPU/TPU: Yes.

Homepage / Docs: https://jax.readthedocs.io

3.5 XGBoost / LightGBM

Purpose: Gradient-boosted trees; very strong tabular models for return prediction, default risk, etc.

CPU/GPU: Both support GPU acceleration.

XGBoost Docs: https://xgboost.readthedocs.io

LightGBM Docs: https://lightgbm.readthedocs.io

4. Portfolio Optimization & Risk Modeling
4.1 PyPortfolioOpt

Purpose: Markowitz mean-variance, Black-Litterman, risk parity, L2/L1 regularization, etc. Implements widely-used classical portfolio optimisation methods. 
PyPortfolioOpt
+2
Read the Docs
+2

CPU/GPU: CPU (NumPy/SciPy); can offload some linear algebra to GPU via lower layers if configured.

Docs: https://pyportfolioopt.readthedocs.io
 
PyPortfolioOpt
+1

PyPI: https://pypi.org/project/pyportfolioopt/
 
PyPI

4.2 cvxpy

Purpose: Convex optimization (quadratic, SOCP, etc.); used to formulate custom portfolio optimization (constraints, transaction costs, leverage).

CPU/GPU: CPU; GPU via some solvers/linear algebra backends.

Homepage / Docs: https://www.cvxpy.org

4.3 Riskfolio-Lib

Purpose: Portfolio optimization & risk management library (risk parity, CVaR, risk budgeting), built on top of pandas/NumPy.

CPU/GPU: CPU.

Docs: https://riskfolio-lib.readthedocs.io

4.4 QuantLib

Purpose: C++/Python library for fixed income, derivatives pricing, term structures, Monte Carlo; widely used in banks.

CPU/GPU: CPU (C++ core).

Homepage: https://www.quantlib.org

Docs: https://www.quantlib.org/documentation.shtml

4.5 PerformanceAnalytics (R)

Purpose: Return/risk metrics, drawdowns, factor analytics; de-facto standard in R for portfolio performance analysis.

Language: R.

Docs: https://cran.r-project.org/package=PerformanceAnalytics

5. Backtesting & Trading Simulation
5.1 backtrader

Purpose: Event-driven backtesting engine with broker simulation; widely used in retail and research.

CPU/GPU: CPU.

Docs: https://www.backtrader.com/docu/

5.2 zipline (Quantopian)

Purpose: Event-driven backtesting with pipeline API; historically used in institutional and academic settings.

CPU/GPU: CPU.

Docs: https://zipline.ml4trading.io

5.3 vectorbt / vectorbt-pro

Purpose: Vectorized backtesting & signal research on top of NumPy/pandas; very fast for portfolio-level simulations.

CPU/GPU: CPU; experimental GPU via NumPy/CuPy stack.

Docs: https://vectorbt.dev

5.4 Qlib (Microsoft)

Purpose: Research platform for AI in quantitative finance (data layer, models, backtesting, evaluation).

CPU/GPU: CPU & GPU (PyTorch/LightGBM backends).

Docs: https://microsoft.github.io/qlib/

6. Financial Machine Learning Toolboxes
6.1 mlfinlab

Purpose: Implements Marcos López de Prado-style financial ML: fractional differentiation, meta-labeling, triple-barrier, sampling schemes, feature engineering, backtest stats, etc. 
Hudson Thames
+2
GitHub
+2

CPU/GPU: Primarily CPU; GPU feasible when combined with GPU-enabled backends.

Homepage: https://hudsonthames.org/mlfinlab/
 
Hudson Thames

Docs: https://mlfinlab.com
 (documentation portal) 
Reddit

GitHub: https://github.com/hudson-and-thames/mlfinlab
 
GitHub

6.2 MLfin.py

Purpose: Advanced ML toolbox for financial applications, inspired by Lopez de Prado’s work. 
Mlfin.py

CPU/GPU: CPU; many models can use GPU via underlying frameworks.

Docs: https://mlfinpy.readthedocs.io
 
Mlfin.py

6.3 FinRL

Purpose: Deep reinforcement learning (DQN, PPO, SAC, etc.) applied to trading and portfolio management.

CPU/GPU: Strong GPU support (PyTorch / stable-baselines3).

Docs: https://finrl.readthedocs.io

7. Probabilistic Programming & Bayesian Time Series
7.1 PyMC

Purpose: Bayesian modeling with MCMC/VI; good for hierarchical time-series, stochastic volatility, state-space models.

CPU/GPU: CPU; some GPU offload via JAX/NumPyro in newer stacks.

Docs: https://www.pymc.io/projects/docs/en/stable/

7.2 Stan / CmdStanPy

Purpose: High-performance Bayesian inference engine (HMC/NUTS); used heavily in academia & some quant shops.

CPU/GPU: CPU (C++ core, multi-threaded).

Stan Docs: https://mc-stan.org/users/documentation

CmdStanPy: https://mc-stan.org/cmdstanpy/

7.3 Pyro

Purpose: Deep probabilistic programming on top of PyTorch; useful for custom stochastic volatility / latent factor models.

CPU/GPU: CPU & GPU.

Docs: https://pyro.ai/examples/

8. GPU-Accelerated Analytics (Local)
8.1 CuPy

Purpose: NumPy-compatible GPU array library; drop-in acceleration for many array ops.

GPU: CUDA (NVIDIA).

Docs: https://docs.cupy.dev/en/stable/

8.2 RAPIDS (cuDF, cuML, cuSignal)

Purpose: End-to-end GPU data science stack:

cuDF: GPU DataFrame API similar to pandas.

cuML: GPU ML algorithms (clustering, regression, etc.).

cuSignal: Signal processing on GPU (useful for high-frequency time series).

GPU: CUDA (NVIDIA).

Docs: https://docs.rapids.ai

9. R Time-Series & Portfolio Ecosystem (for completeness)

These are standard in academic and institutional quant courses:

xts / zoo – Time-indexed data structures

Docs:

https://cran.r-project.org/package=xts

https://cran.r-project.org/package=zoo

forecast / fable – ARIMA, ETS, advanced forecasting

https://cran.r-project.org/package=forecast

https://fable.tidyverts.org

rugarch – Univariate GARCH models (very widely used).

https://cran.r-project.org/package=rugarch

quantmod – Data acquisition, charting, technical indicators.

https://cran.r-project.org/package=quantmod

PortfolioAnalytics – Portfolio construction, optimization, risk.

https://cran.r-project.org/package=PortfolioAnalytics

TTR – Technical trading rules (indicators).

https://cran.r-project.org/package=TTR

10. Theory / Background Resources (For Your Agent’s Reasoning Layer)

Not libraries, but useful references your AI companion can cite when explaining methods:

Time Series Analysis in Python with statsmodels (paper / tutorial) 
Semantic Scholar

PDF: https://pdfs.semanticscholar.org/7c96/660127fefabe926214abaa80b298066af60d.pdf

PyPortfolioOpt: portfolio optimization in Python (paper) 
ResearchGate
+1

ARCH/ GARCH tutorials using arch 
Arch Documentation
+2
DataCamp Campus
+2

