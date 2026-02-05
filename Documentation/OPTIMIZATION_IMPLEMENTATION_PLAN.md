# 🚀 OPTIMIZATION IMPLEMENTATION PLAN

> **RUNTIME GUARDRAIL (WSL `simpleTrader_env` ONLY)**
> Supported runtime: WSL + Linux venv `simpleTrader_env/bin/python` (`source simpleTrader_env/bin/activate`).
> **Do not** use Windows interpreters/venvs (incl. `py`, `python.exe`, `.venv`, `simpleTrader_env\\Scripts\\python.exe`) — results are invalid.
> Before reporting runs, include the runtime fingerprint (command + output): `which python`, `python -V`, `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` (see `Documentation/RUNTIME_GUARDRAILS.md`).

> **Reward-to-Effort Integration:** For automation, monetization, and sequencing work, align with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`.

**Portfolio Maximizer v45 - Professional Standards Upgrade**

**Date**: October 14, 2025
**Status**: 🟢 UNBLOCKED – 2026-01-29 status aligns with `Documentation/PROJECT_STATUS.md`
**Priority**: **CRITICAL** - Required for institutional-grade production

---
**Status note (2025-12-28)**: The brutal suite completes end-to-end and the earlier 2025-11-15 blockers are resolved. Use `Documentation/arch_tree.md` and `Documentation/implementation_checkpoint.md` for the canonical verification trail.

> 📊 Telemetry note: `forcester_ts/instrumentation.py` records dataset diagnostics and benchmark metrics (RMSE, sMAPE, tracking error) per model with JSON audits under `logs/forecast_audits/`. Use these artifacts to quantify optimization impact.

**Project-wide sequencing (2026-01)**: See `Documentation/PROJECT_WIDE_OPTIMIZATION_ROADMAP.md` for the step-by-step plan to fix TS horizon alignment, execution cost realism, and run-local reporting before deeper optimization phases.

## Delta (2026-01-18)

- Live dashboard is now real-time (polls `visualizations/dashboard_data.json` every 5s) and renders trade-level visualization (price + entry/exit markers + realized PnL) from emitted run artifacts.
- Dashboard payload includes `positions`, `price_series`, `trade_events` and is rendered from the SQLite DB via `scripts/dashboard_db_bridge.py` (DB→JSON), with audit snapshots persisted by default to `data/dashboard_audit.db` (`--persist-snapshot` via bash orchestrators).
- Forecast audit dedupe fixed in `scripts/check_forecast_audits.py` to ensure dataset-window monitoring uses “latest evidence wins”.

## Delta (2026-01-29)

- Auto-trader now **resumes portfolio state by default** (`--resume`), with `--no-resume` / `bash/reset_portfolio.sh` for clean starts; existing DBs require `python scripts/migrate_add_portfolio_state.py`.
- Dashboard bridge filters trade events to the **latest run_id by default** and falls back to `trade_executions` when `portfolio_positions` is empty.
- Ops helpers added: `bash/run_daily_trader.sh` (daily + intraday passes) and `run_daily_trader.bat` (Windows Task Scheduler).

## 📊 EXECUTIVE SUMMARY

> **Ensemble status (canonical, current)**: `ENSEMBLE_MODEL_STATUS.md` is the single source of truth for whether the time-series ensemble is active, how to interpret `KEEP` vs `RESEARCH_ONLY`, and the latest audit gate decision. Do not cite ensemble status from older snapshots in this plan.

### Current State Assessment
- **Mathematical Foundation**: B+ (Solid but incomplete)
- **Statistical Rigor**: B (Needs enhancement)
- **Production Readiness**: B (Good but not institutional-grade)
- **Test Coverage**: A (Excellent - 529 tests; latest full suite green per `Documentation/PROJECT_STATUS.md`)

### Ensemble Gate Decision (Historical + Current SoT)
- Research-profile RMSE gate (`scripts/check_forecast_audits.py --config-path config/forecaster_monitoring.yml --max-files 500`) on 27 effective audits:
  - Violations: 1 (3.7% <= 25% cap), lift_fraction: 0% (<10% required) ⇒ **Decision: DISABLE ensemble as default** (insufficient lift vs BEST_SINGLE).
  - Historical note: at the time, `config/forecasting_config.yml` was set to `ensemble.enabled: false`.
- **Update (2026-02-04)**: The aggregate audit gate currently passes (Decision `KEEP`). See `ENSEMBLE_MODEL_STATUS.md` for the reproducible command + exact numbers, and the “per-forecast policy label vs aggregate audit gate” interpretation.
- Historical action (2026-01-11): keep ensemble research-only; revisit when lift_fraction ≥ 10% with RMSE ratio within tolerance over ≥20 effective audits. Current decision is governed by `ENSEMBLE_MODEL_STATUS.md`.

### Target State
- **Mathematical Foundation**: A+ (Institutional-grade)
- **Statistical Rigor**: A+ (Professional standards)
- **Production Readiness**: A+ (Institutional deployment ready)
- **Test Coverage**: A+ (Enhanced with statistical tests)

### Frontier Market Coverage Mandate (2025-11-15)
- `etl/frontier_markets.py` + the new `--include-frontier-tickers` flag ensure every optimization dry-run touches the guided frontier venues (Nigeria → Bulgaria list supplied in the liquidity brief). Multi-ticker bash workflows (`bash/run_pipeline_live.sh`, `bash/run_pipeline_dry_run.sh`, `bash/test_real_time_pipeline.sh`, brutal suite) now default to the flag, so model tuning inherently exercises these shapelier spread/latency scenarios.
- Optimization checkpoints must document whether live/API tickers are mapped (NGX/NSE/BSE suffixing) or whether the run used the synthetic fallback; reference `Documentation/arch_tree.md` for the canonical list when drafting future release notes.

---

## 🎯 PHASE 1: MATHEMATICAL FOUNDATION (Week 1)

### 1.1 Enhanced Portfolio Mathematics Engine
**File**: `etl/portfolio_math_enhanced.py` ✅ **COMPLETED**

**Implementation Status**: ✅ **READY**
- ✅ Sortino Ratio calculation
- ✅ CVaR/Expected Shortfall
- ✅ Information Ratio
- ✅ Calmar Ratio
- ✅ Correct Kelly Criterion
- ✅ Bootstrap confidence intervals
- ✅ Statistical significance testing
- ✅ Stress testing framework

> 🔄 **Barbell Migration Note (2025-11-24)**
> Long-volatility / tail-hedge and barbell “risk bucket” legs must be evaluated primarily with these asymmetric/tail-aware metrics (Sortino, Omega, CVaR, scenario analysis) at the **portfolio** level, not by Sharpe alone. See `BARBELL_INTEGRATION_TODO.md` for the dedicated barbell allocation and antifragility plan.

**Next Steps**:
```bash
# 1. Replace existing portfolio_math.py
mv etl/portfolio_math.py etl/portfolio_math_legacy.py
mv etl/portfolio_math_enhanced.py etl/portfolio_math.py

# 2. Update imports in dependent files
# 3. Run comprehensive tests
python -m pytest tests/etl/test_portfolio_math_enhanced.py -v
```

### 1.2 Enhanced Test Suite
**File**: `tests/etl/test_portfolio_math_enhanced.py` ✅ **COMPLETED**

**Implementation Status**: ✅ **READY**
- ✅ 500 lines of comprehensive tests
- ✅ Mathematical rigor validation
- ✅ Edge case testing
- ✅ Statistical significance testing
- ✅ Bootstrap validation

**Next Steps**:
```bash
# 1. Run enhanced tests
python -m pytest tests/etl/test_portfolio_math_enhanced.py -v

# 2. Integrate with existing test suite
python -m pytest tests/etl/ -v
```

### 1.3 Signal Validator Mathematical Fixes
**File**: `ai_llm/signal_validator.py` ⚠️ **NEEDS FIXES**

**Critical Issues Identified**:
1. **Kelly Criterion Implementation Flaw**
2. **Missing Statistical Significance Testing**
3. **Insufficient Regime Detection**

**Required Fixes**:
```python
# Fix 1: Correct Kelly Criterion
def calculate_kelly_fraction_correct(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Correct Kelly Criterion: f* = (bp - q) / b"""
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0

    b = avg_win / avg_loss  # Odds
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b
    return max(0, min(kelly, 0.25))  # Cap at 25%

# Fix 2: Add Statistical Significance Testing
def test_signal_significance(signals: List[Dict], actual_returns: np.ndarray) -> Dict:
    """Test if signals have statistically significant predictive power"""
    from scipy import stats

    predicted_directions = [1 if s['action'] == 'BUY' else -1 for s in signals]
    actual_directions = np.sign(actual_returns)

    # Binomial test for accuracy
    correct_predictions = sum(p == a for p, a in zip(predicted_directions, actual_directions))
    n = len(signals)

    # H0: p = 0.5 (random), H1: p > 0.5 (skill)
    p_value = stats.binom_test(correct_predictions, n, p=0.5, alternative='greater')

    return {
        'accuracy': correct_predictions / n,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'information_coefficient': np.corrcoef(predicted_directions, actual_directions)[0, 1]
    }

# Fix 3: Add Regime Detection
def detect_market_regime(returns: np.ndarray, window: int = 60) -> Dict:
    """Detect market regimes using statistical tests"""
    from scipy import stats

    rolling_vol = returns.rolling(window).std()
    regimes = []

    for i in range(window, len(returns)):
        recent_vol = rolling_vol.iloc[i-window:i]
        current_vol = rolling_vol.iloc[i]

        # Test if current volatility is significantly different
        t_stat, p_value = stats.ttest_1samp(recent_vol, current_vol)

        if p_value < 0.05:
            if current_vol > recent_vol.mean():
                regimes.append('high_vol')
            else:
                regimes.append('low_vol')
        else:
            regimes.append('normal')

    return {
        'regimes': regimes,
        'regime_probabilities': calculate_regime_probabilities(regimes)
    }
```

---

## 🎯 PHASE 2: STATISTICAL RIGOR (Week 2)

### 2.1 Advanced Statistical Testing Framework
**New File**: `etl/statistical_tests.py`

**Implementation Plan**:
```python
"""
Advanced Statistical Testing Framework
Professional-grade hypothesis testing and validation
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict, List, Tuple

class StatisticalTestSuite:
    """Comprehensive statistical testing for trading strategies."""

    def test_strategy_significance(self, strategy_returns: np.ndarray,
                                 benchmark_returns: np.ndarray) -> Dict:
        """Test if strategy returns are statistically significant."""
        # T-test for mean difference
        t_stat, p_value = stats.ttest_ind(strategy_returns, benchmark_returns)

        # Information ratio
        excess_returns = strategy_returns - benchmark_returns
        information_ratio = np.mean(excess_returns) / np.std(excess_returns)

        # F-test for variance equality
        f_stat, f_p_value = stats.f_oneway(strategy_returns, benchmark_returns)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'information_ratio': information_ratio,
            'f_statistic': f_stat,
            'f_p_value': f_p_value,
            'variance_equal': f_p_value > 0.05
        }

    def test_autocorrelation(self, returns: np.ndarray) -> Dict:
        """Test for serial correlation in returns (violates EMH)."""
        lb_stat, p_value = acorr_ljungbox(returns, lags=10, return_df=False)

        return {
            'ljung_box_statistic': lb_stat,
            'p_value': p_value,
            'serial_correlation': p_value < 0.05,
            'efficient_market': p_value > 0.05
        }

    def test_normality(self, returns: np.ndarray) -> Dict:
        """Test for normality of returns."""
        # Jarque-Bera test
        jb_stat, jb_p_value = stats.jarque_bera(returns)

        # Shapiro-Wilk test (for smaller samples)
        if len(returns) <= 5000:
            sw_stat, sw_p_value = stats.shapiro(returns)
        else:
            sw_stat, sw_p_value = None, None

        return {
            'jarque_bera_statistic': jb_stat,
            'jarque_bera_p_value': jb_p_value,
            'shapiro_wilk_statistic': sw_stat,
            'shapiro_wilk_p_value': sw_p_value,
            'is_normal': jb_p_value > 0.05
        }

    def test_stationarity(self, returns: np.ndarray) -> Dict:
        """Test for stationarity of returns."""
        from statsmodels.tsa.stattools import adfuller

        adf_stat, adf_p_value, used_lag, nobs, critical_values, icbest = adfuller(returns)

        return {
            'adf_statistic': adf_stat,
            'adf_p_value': adf_p_value,
            'critical_values': critical_values,
            'is_stationary': adf_p_value < 0.05
        }

    def bootstrap_validation(self, returns: np.ndarray, n_bootstrap: int = 1000) -> Dict:
        """Bootstrap validation for performance metrics."""
        metrics = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)

            # Calculate metrics
            sharpe = (np.mean(bootstrap_sample) - 0.02) / np.std(bootstrap_sample) * np.sqrt(252)
            max_dd = self.calculate_max_drawdown(bootstrap_sample)
            sortino = self.calculate_sortino_ratio(bootstrap_sample)

            metrics.append({
                'sharpe': sharpe,
                'max_drawdown': max_dd,
                'sortino': sortino
            })

        # Calculate confidence intervals
        sharpe_values = [m['sharpe'] for m in metrics]
        dd_values = [m['max_drawdown'] for m in metrics]
        sortino_values = [m['sortino'] for m in metrics]

        return {
            'sharpe_ci': (np.percentile(sharpe_values, 2.5), np.percentile(sharpe_values, 97.5)),
            'max_dd_ci': (np.percentile(dd_values, 2.5), np.percentile(dd_values, 97.5)),
            'sortino_ci': (np.percentile(sortino_values, 2.5), np.percentile(sortino_values, 97.5)),
            'sharpe_std': np.std(sharpe_values),
            'max_dd_std': np.std(dd_values),
            'sortino_std': np.std(sortino_values)
        }
```

### 2.2 Regime Detection and Adaptation
**New File**: `etl/regime_detector.py`

**Implementation Plan**:
```python
"""
Market Regime Detection and Adaptation
Statistical regime detection for dynamic strategy adjustment
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RegimeState:
    """Market regime state."""
    regime_type: str  # 'bull', 'bear', 'sideways', 'high_vol', 'low_vol'
    confidence: float  # 0-1
    duration: int  # Days in current regime
    transition_probability: float  # Probability of regime change

class RegimeDetector:
    """Statistical market regime detection."""

    def __init__(self, window_size: int = 60, significance_level: float = 0.05):
        self.window_size = window_size
        self.significance_level = significance_level
        self.current_regime = None
        self.regime_history = []

    def detect_volatility_regime(self, returns: np.ndarray) -> RegimeState:
        """Detect volatility regime using statistical tests."""
        if len(returns) < self.window_size * 2:
            return RegimeState('insufficient_data', 0.0, len(returns), 0.0)

        # Rolling volatility
        rolling_vol = pd.Series(returns).rolling(self.window_size).std()
        current_vol = rolling_vol.iloc[-1]
        historical_vol = rolling_vol.iloc[-self.window_size-1:-1]

        # Statistical test for regime change
        t_stat, p_value = stats.ttest_1samp(historical_vol.dropna(), current_vol)

        if p_value < self.significance_level:
            if current_vol > historical_vol.mean():
                regime_type = 'high_vol'
                confidence = 1 - p_value
            else:
                regime_type = 'low_vol'
                confidence = 1 - p_value
        else:
            regime_type = 'normal_vol'
            confidence = p_value

        return RegimeState(
            regime_type=regime_type,
            confidence=confidence,
            duration=self._calculate_regime_duration(regime_type),
            transition_probability=p_value
        )

    def detect_trend_regime(self, returns: np.ndarray) -> RegimeState:
        """Detect trend regime using statistical tests."""
        if len(returns) < self.window_size * 2:
            return RegimeState('insufficient_data', 0.0, len(returns), 0.0)

        # Rolling mean return
        rolling_mean = pd.Series(returns).rolling(self.window_size).mean()
        current_mean = rolling_mean.iloc[-1]
        historical_mean = rolling_mean.iloc[-self.window_size-1:-1]

        # Statistical test for trend change
        t_stat, p_value = stats.ttest_1samp(historical_mean.dropna(), current_mean)

        if p_value < self.significance_level:
            if current_mean > historical_mean.mean():
                regime_type = 'bull'
                confidence = 1 - p_value
            else:
                regime_type = 'bear'
                confidence = 1 - p_value
        else:
            regime_type = 'sideways'
            confidence = p_value

        return RegimeState(
            regime_type=regime_type,
            confidence=confidence,
            duration=self._calculate_regime_duration(regime_type),
            transition_probability=p_value
        )

    def adapt_strategy_parameters(self, signal: Dict, regime_state: RegimeState) -> Dict:
        """Adapt strategy parameters based on current regime."""
        regime_multipliers = {
            'bull': {'confidence_multiplier': 1.2, 'position_multiplier': 1.1},
            'bear': {'confidence_multiplier': 0.8, 'position_multiplier': 0.9},
            'sideways': {'confidence_multiplier': 1.0, 'position_multiplier': 1.0},
            'high_vol': {'confidence_multiplier': 0.7, 'position_multiplier': 0.8},
            'low_vol': {'confidence_multiplier': 1.3, 'position_multiplier': 1.2},
            'normal_vol': {'confidence_multiplier': 1.0, 'position_multiplier': 1.0}
        }

        multiplier = regime_multipliers.get(regime_state.regime_type,
                                          {'confidence_multiplier': 1.0, 'position_multiplier': 1.0})

        adapted_signal = signal.copy()
        adapted_signal['confidence'] *= multiplier['confidence_multiplier']
        adapted_signal['position_size'] *= multiplier['position_multiplier']
        adapted_signal['regime'] = regime_state.regime_type
        adapted_signal['regime_confidence'] = regime_state.confidence

        return adapted_signal

    def _calculate_regime_duration(self, regime_type: str) -> int:
        """Calculate duration of current regime."""
        if not self.regime_history:
            return 1

        duration = 1
        for i in range(len(self.regime_history) - 1, -1, -1):
            if self.regime_history[i].regime_type == regime_type:
                duration += 1
            else:
                break

        return duration
```

---

## 🎯 PHASE 3: ADVANCED FEATURES (Weeks 3-4)

### 3.1 Portfolio Optimization Engine
**New File**: `etl/portfolio_optimizer.py`

**Implementation Plan**:
```python
"""
Advanced Portfolio Optimization Engine
Mean-variance, risk parity, and factor model optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Portfolio optimization result."""
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    success: bool
    message: str
    iterations: int

class PortfolioOptimizer:
    """Advanced portfolio optimization with multiple methodologies."""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def mean_variance_optimization(self, returns: np.ndarray,
                                 target_return: Optional[float] = None,
                                 risk_aversion: float = 1.0) -> OptimizationResult:
        """Markowitz mean-variance optimization."""
        mu = np.mean(returns, axis=0)
        Sigma = np.cov(returns.T)
        n = len(mu)

        if target_return is not None:
            # Target return optimization
            def objective(w):
                return 0.5 * w.T @ Sigma @ w

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: w.T @ mu - target_return}
            ]
        else:
            # Risk aversion optimization
            def objective(w):
                return 0.5 * risk_aversion * w.T @ Sigma @ w - w.T @ mu

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        bounds = [(0, 1) for _ in range(n)]

        result = minimize(objective, x0=np.ones(n)/n, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            expected_return = weights.T @ mu
            expected_volatility = np.sqrt(weights.T @ Sigma @ weights)
            sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility
        else:
            weights = np.ones(n) / n
            expected_return = np.mean(mu)
            expected_volatility = np.sqrt(np.mean(np.diag(Sigma)))
            sharpe_ratio = 0.0

        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            success=result.success,
            message=result.message,
            iterations=result.nit
        )

    def risk_parity_optimization(self, returns: np.ndarray) -> OptimizationResult:
        """Risk parity portfolio optimization."""
        Sigma = np.cov(returns.T)
        n = len(Sigma)

        def objective(w):
            portfolio_vol = np.sqrt(w.T @ Sigma @ w)
            if portfolio_vol < 1e-8:
                return 1e6

            risk_contributions = (w * (Sigma @ w)) / portfolio_vol
            target_contribution = portfolio_vol / n

            return np.sum((risk_contributions - target_contribution)**2)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n)]

        result = minimize(objective, x0=np.ones(n)/n, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if result.success:
            weights = result.x
            mu = np.mean(returns, axis=0)
            expected_return = weights.T @ mu
            expected_volatility = np.sqrt(weights.T @ Sigma @ weights)
            sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility
        else:
            weights = np.ones(n) / n
            mu = np.mean(returns, axis=0)
            expected_return = np.mean(mu)
            expected_volatility = np.sqrt(np.mean(np.diag(Sigma)))
            sharpe_ratio = 0.0

        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            success=result.success,
            message=result.message,
            iterations=result.nit
        )

    def factor_model_optimization(self, returns: np.ndarray,
                                factors: np.ndarray) -> OptimizationResult:
        """Factor model portfolio optimization."""
        from sklearn.linear_model import LinearRegression

        # Fit factor model: R = α + β*F + ε
        model = LinearRegression()
        model.fit(factors, returns)

        # Factor model covariance: Σ = B*F*B' + D
        factor_cov = np.cov(factors.T)
        beta = model.coef_.T
        residual_cov = np.cov(returns - model.predict(factors), rowvar=False)

        Sigma = beta @ factor_cov @ beta.T + np.diag(np.diag(residual_cov))

        # Optimize using factor model covariance
        return self.mean_variance_optimization(returns, Sigma=Sigma)
```

### 3.2 Monte Carlo Risk Simulation
**New File**: `etl/monte_carlo_simulator.py`

**Implementation Plan**:
```python
"""
Monte Carlo Risk Simulation Engine
Comprehensive risk assessment through simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

@dataclass
class SimulationResult:
    """Monte Carlo simulation result."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    worst_case: float
    best_case: float
    probability_of_loss: float
    expected_loss_given_loss: float

class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio risk assessment."""

    def __init__(self, n_simulations: int = 10000, random_seed: Optional[int] = None):
        self.n_simulations = n_simulations
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_portfolio_risk(self, portfolio_returns: np.ndarray,
                              portfolio_weights: np.ndarray,
                              simulation_method: str = 'bootstrap') -> SimulationResult:
        """Simulate portfolio risk using Monte Carlo methods."""

        if simulation_method == 'bootstrap':
            simulated_returns = self._bootstrap_simulation(portfolio_returns, portfolio_weights)
        elif simulation_method == 'parametric':
            simulated_returns = self._parametric_simulation(portfolio_returns, portfolio_weights)
        elif simulation_method == 'historical':
            simulated_returns = self._historical_simulation(portfolio_returns, portfolio_weights)
        else:
            raise ValueError(f"Unknown simulation method: {simulation_method}")

        # Calculate risk metrics
        var_95 = np.percentile(simulated_returns, 5)
        var_99 = np.percentile(simulated_returns, 1)
        cvar_95 = simulated_returns[simulated_returns <= var_95].mean()
        cvar_99 = simulated_returns[simulated_returns <= var_99].mean()
        expected_shortfall = simulated_returns[simulated_returns < 0].mean()

        return SimulationResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_shortfall=expected_shortfall,
            worst_case=np.min(simulated_returns),
            best_case=np.max(simulated_returns),
            probability_of_loss=np.mean(simulated_returns < 0),
            expected_loss_given_loss=expected_shortfall
        )

    def stress_test_simulation(self, portfolio_returns: np.ndarray,
                             portfolio_weights: np.ndarray,
                             stress_scenarios: Dict[str, float]) -> Dict[str, SimulationResult]:
        """Stress test portfolio under various scenarios."""
        results = {}

        for scenario_name, shock_magnitude in stress_scenarios.items():
            # Apply shock to returns
            stressed_returns = portfolio_returns + shock_magnitude

            # Simulate stressed portfolio
            simulated_returns = self._bootstrap_simulation(stressed_returns, portfolio_weights)

            # Calculate stressed risk metrics
            var_95 = np.percentile(simulated_returns, 5)
            var_99 = np.percentile(simulated_returns, 1)
            cvar_95 = simulated_returns[simulated_returns <= var_95].mean()
            cvar_99 = simulated_returns[simulated_returns <= var_99].mean()
            expected_shortfall = simulated_returns[simulated_returns < 0].mean()

            results[scenario_name] = SimulationResult(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                expected_shortfall=expected_shortfall,
                worst_case=np.min(simulated_returns),
                best_case=np.max(simulated_returns),
                probability_of_loss=np.mean(simulated_returns < 0),
                expected_loss_given_loss=expected_shortfall
            )

        return results

    def _bootstrap_simulation(self, returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Bootstrap simulation of portfolio returns."""
        simulated_returns = []

        for _ in range(self.n_simulations):
            # Bootstrap sample
            bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
            portfolio_return = np.mean(bootstrap_sample)  # Equal weight for simplicity
            simulated_returns.append(portfolio_return)

        return np.array(simulated_returns)

    def _parametric_simulation(self, returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Parametric simulation assuming normal distribution."""
        mu = np.mean(returns)
        sigma = np.std(returns)

        return np.random.normal(mu, sigma, self.n_simulations)

    def _historical_simulation(self, returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Historical simulation using past scenarios."""
        # Use historical returns as scenarios
        return np.random.choice(returns, size=self.n_simulations, replace=True)
```

---

## 🎯 PHASE 4: INSTITUTIONAL FEATURES (Weeks 5-8)

### 4.1 Performance Attribution Analysis
**New File**: `etl/performance_attribution.py`

### 4.2 Regulatory Compliance Metrics
**New File**: `etl/regulatory_compliance.py`

### 4.3 Real-time Risk Monitoring
**Enhancement**: `risk/real_time_risk_manager.py`

### 4.4 Advanced Reporting Dashboard
**New File**: `reports/advanced_dashboard.py`

### 4.5 Intraday / LOB Execution Research (NEW)
**Scope**: Bring execution modeling in line with production-grade microstructure practices.

**Research items**
- **TWAP/VWAP scheduling**: add order slicing profiles with participation-rate caps and time-bucket scheduling.
- **Queue modeling**: simple queue position and fill-probability model using top-of-book sizes, spread, and short-term volatility.
- **Market-impact curves**: empirical impact curve (e.g., square-root or power-law) calibrated on synthetic/realized slippage; tie to notional and volatility.
- **Execution policy backtests**: compare MARKET vs limit/TWAP/VWAP on synthetic and real OHLCV with microstructure columns (Spread/Slippage/TxnCost/Impact).
- **Cost model integration**: feed execution cost estimates into `signal_routing.time_series.cost_model` and paper-trading slippage modeling.

**Execution benchmarks (short)**
- **Arrival price vs VWAP/TWAP**: measure average implementation shortfall vs arrival price and VWAP/TWAP; target <= 15 bps for liquid equities, <= 35 bps for frontier baskets, and <= 25 bps for major FX.
- **Slippage bps targets**: median slippage <= 10 bps (liquid equities), <= 25 bps (frontier), <= 15 bps (FX) under matched notional buckets.
- **Fill quality**: limit/TWAP/VWAP fill rates >= 85% with no more than 1.5x baseline impact vs market orders.

**Mapped experiment IDs**
- `EXP_EXEC_2025_001`: MARKET vs TWAP (arrival price + VWAP slippage) on AAPL/MSFT/GOOGL, 2020-2024.
- `EXP_EXEC_2025_002`: MARKET vs VWAP on frontier basket with synthetic microstructure (Spread/TxnCost/Impact columns).
- `EXP_EXEC_2025_003`: Queue model vs naive limit orders on top-of-book simulation; measure fill rate and adverse selection.
- `EXP_EXEC_2025_004`: Impact-curve calibration from paper-trading executions; validate against realized slippage buckets.

**Planned artifacts**
- `execution/execution_scheduler.py` (order slicing policies)
- `execution/market_impact_model.py` (impact curves + calibration)
- `tests/execution/test_execution_scheduler.py`
- `tests/execution/test_market_impact_model.py`

---

## ✅ Forecaster & Hyper‑Parameter Institutionalisation – Sequenced TODO

> **Objective**: Lift the TS forecaster + hyper‑parameter stack from “good research grade” to **institutional‑grade**, tightly coupled to brutal tests, numeric invariants, and quant validation automation.

1. **Higher‑Order TS Hyper‑Opt & Regime Backtesting**
   - [ ] Extend `run_strategy_optimization.py` / `run_post_eval.sh` to drive a higher‑order grid over model configs (SARIMAX caps/search modes + SARIMAX‑X exog policy, SAMOSSA caps/targets, MSSA‑RL settings) per ticker.
   - [ ] Reuse `TIME_SERIES_CV` infrastructure to run walk‑forward CV per ticker and per regime (using existing MSSA‑RL / realised‑vol regime tags).
   - [ ] Persist `(ticker, regime, model_config, CV_metrics)` into a small `ts_model_candidates` table for later analysis and dashboards.

2. **Regime‑ and Sleeve‑Aware Model Profiles**
   - [ ] Add a config section/file (e.g. `config/model_profiles.yml`) that defines model profiles keyed by `(sleeve, asset_class, regime)`.
   - [ ] For each profile, specify constrained SARIMAX caps/search modes (not fixed orders), SAMOSSA caps/targets, and MSSA‑RL toggles consistent with `QUANT_TIME_SERIES_STACK.md`.
   - [ ] Add a thin router in `TimeSeriesForecaster` that selects a profile based on regime diagnostics and sleeve (safe/core/spec/crypto from `config/barbell.yml`).

3. **Statistical Model‑Selection Tests**
   - [ ] Introduce or extend `etl/statistical_tests.py` to include Diebold–Mariano / variance‑ratio style tests comparing candidate models against a baseline.
   - [ ] Require the chosen hyper‑param configuration to be **stable across CV folds** (no single‑fold “winner by noise”); fail brutal tests when stability criteria are violated.
   - [ ] Wire these tests into numeric/scaling invariant suites so model selection is validated alongside existing invariants.

4. **Unified TS Automation Dashboard**
   - [x] Create `scripts/build_automation_dashboard.py` that consolidates:
       - TS threshold sweeps (`logs/automation/ts_threshold_sweep.json`),
       - Transaction cost estimates (`logs/automation/transaction_costs.json`),
       - Sleeve promotion/demotion plans (`logs/automation/sleeve_promotion_plan.json`),
       - TS model/hyper‑opt summaries (from `ts_model_candidates` or a JSON export).
   - [x] Emit a single `visualizations/dashboard_automation.json` snapshot referenced from `CRON_AUTOMATION.md` and brutal/monitoring scripts.

5. **Quant Validation Threshold Re‑Calibration**
   - [ ] Use distributions from the hyper‑opt / CV runs to recalibrate `config/forecaster_monitoring.yml`:
       - Separate **research** vs **production** thresholds for `min_profit_factor`, `min_win_rate`, `min_pass_rate`, and `max_negative_expected_profit_fraction`.
       - Ensure `scripts/check_quant_validation_health.py` and `scripts/summarize_quant_validation.py` consume these calibrated thresholds consistently.
   - [ ] Update `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` and `Documentation/CRITICAL_REVIEW.md` with the new institutional‑grade criteria and clearly document how CI/brutal gates enforce them.

6. **TS Model Candidate Inspection**
   - [x] Add `scripts/summarize_ts_candidates.py` to:
       - Read `ts_model_candidates` from the SQLite store,
       - Aggregate best candidates per `(ticker, regime)` by score,
       - Print a compact table (ticker, regime, candidate, score, stability, DM vs baseline) and optionally emit a JSON summary.
   - [ ] Wire this helper into research notebooks and the automation/dashboard layer so institutional-grade TS model choices are always backed by a transparent, inspectable trail.

---

## 📊 IMPLEMENTATION TIMELINE

### Week 1: Mathematical Foundation
- [ ] **Day 1-2**: Deploy enhanced portfolio mathematics engine
- [ ] **Day 3-4**: Fix signal validator mathematical issues
- [ ] **Day 5**: Run comprehensive tests and validation

### Week 2: Statistical Rigor
- [ ] **Day 1-2**: Implement statistical testing framework
- [ ] **Day 3-4**: Add regime detection and adaptation
- [ ] **Day 5**: Integrate with existing signal validation

### Week 3: Advanced Features
- [ ] **Day 1-2**: Portfolio optimization engine
- [ ] **Day 3-4**: Monte Carlo risk simulation
- [ ] **Day 5**: Integration and testing

### Week 4: Testing and Validation
- [ ] **Day 1-2**: Comprehensive test suite
- [ ] **Day 3-4**: Performance benchmarking
- [ ] **Day 5**: Documentation and deployment

### Weeks 5-8: Institutional Features
- [ ] Performance attribution analysis
- [ ] Regulatory compliance metrics
- [ ] Advanced reporting dashboard
- [ ] Real-time risk monitoring enhancements
- [ ] Intraday/LOB execution research (TWAP/VWAP, queue modeling, market-impact curves)

---

## 🎯 SUCCESS METRICS

### Mathematical Rigor Targets
- [ ] **100%** of financial formulas mathematically correct
- [ ] **95%** confidence intervals for all performance metrics
- [ ] **Statistical significance** testing for all strategies
- [ ] **Stress testing** under 5+ historical scenarios

### Performance Targets
- [ ] **Sortino Ratio** > 1.5
- [ ] **Information Ratio** > 0.5
- [ ] **CVaR (95%)** < 2% daily
- [ ] **Calmar Ratio** > 0.5

### Production Readiness Targets
- [ ] **Institutional-grade** risk management
- [ ] **Regulatory compliance** metrics
- [ ] **Real-time** risk monitoring
- [ ] **Automated** stress testing
- [ ] **Execution quality** controls (TWAP/VWAP scheduling + impact curves)

---

## 🚀 IMMEDIATE NEXT STEPS

### 1. Deploy Enhanced Portfolio Mathematics (Today)
```bash
# Backup existing implementation
cp etl/portfolio_math.py etl/portfolio_math_legacy.py

# Deploy enhanced version
cp etl/portfolio_math_enhanced.py etl/portfolio_math.py

# Run tests
python -m pytest tests/etl/test_portfolio_math_enhanced.py -v
```

### 2. Fix Signal Validator (This Week)
- Implement correct Kelly Criterion
- Add statistical significance testing
- Add regime detection

### 3. Implement Statistical Testing Framework (Next Week)
- Create `etl/statistical_tests.py`
- Add comprehensive hypothesis testing
- Integrate with existing validation

---

---

## Africa-First Sentiment Integration (Post-Optimization Milestone)

### 1. Objective and Constraints
- Integrate sentiment as a forecaster input once profitability gates are passed.
- Prioritise African exchanges (NGX, JSE, NSE Kenya, etc.) and free sources only.
- Run lexicon and transformer models locally (CPU or GPU) with no paid API calls.
- Treat sentiment as a weighted factor layered on top of price-based TS forecasts; LLM remains fallback only.
- Require statistical and economic validation before giving the factor production weight.

### 2. Africa-First Data Sources
**Exchange and market announcements (highest weight)**
- NGX media centre releases, weekly market reports, delayed price lists.
- JSE SENS announcements and ETF notices.
- NSE Kenya press releases.
- Free delayed snapshots such as afx.kwayisi.org to cross-check quotes.

**African business RSS / web feeds**
- Nairametrics (https://nairametrics.com/feed) and BusinessDay Nigeria feeds.
- Additional Nigerian and pan-African finance feeds (Premium Times business, Vanguard business, etc.).
- Respect robots.txt and ToS; scrape full text only when allowed.

**Global free sources (lower priority)**
- SEC EDGAR (for dual-listed African companies) and global macro feeds (Nasdaq, MarketWatch) as spillover signals.
- Weight global items lower for NGX/JSE/NSE assets unless macro shocks dominate.

### 3. Local Sentiment Models
**Lexicon / deterministic**
- VADER, FinVADER, Loughran-McDonald, FinSenticNet (all zero-cost and Python-ready).
- Store lexicon scores with document metadata for transparency.

**Transformer / learned**
- FinBERT variants (ProsusAI/finBERT, yiyanghkust/finbert-tone, peejm/finbert-financial-sentiment) and FinSoSent.
- Run locally via Hugging Face weights; fine-tune on African corpora when enough labelled text exists.

### 4. Document-Level Ensemble (Africa-aware)
1. Parse each headline/article/announcement d, compute lexicon scores s_d,m and transformer score s_d,finbert.
2. Tag metadata: region (africa/global), country (NG, ZA, KE, ...), source_type (exchange_announcement, africa_news, global_news), asset_id, timestamp.
3. Deterministic weights: set w_m proportional to each model's F1/accuracy on labelled African text, normalize so sum(w_m)=1, and keep separate weights per region if needed.
4. Optional stochastic weights: sample w ~ Dirichlet(alpha_m) with alpha_m = kappa * F1_m (africa) to quantify ensemble uncertainty; store E[s_d] and Var[s_d].

### 5. Asset-Time Aggregation with Africa-First Source Classes
- Define source classes: africa_exchange, africa_news, global_news.
- For asset i and bucket t compute S_i,t(class) = average document score for that class.
- Estimate regression r_{i,t+1} = alpha + beta_price * rhat_{i,t+1}(price) + sum_class beta_class * S_i,t(class).
- For African assets set weights w_class proportional to max(beta_class, 0) with africa_exchange > africa_news > global_news by default.
- Aggregate to S_i,t(agg) = sum_class w_class * S_i,t(class); store alongside diagnostics for QA.

### 6. Integration into the TS Forecaster
**Deterministic blend**
- Combine forecasts as rhat_{i,t+1}(combined) = alpha_i * rhat_price + beta_i * S_i,t(agg); calibrate alpha_i/beta_i with walk-forward CV on African histories.
- Add event-type refinements by tagging exchange announcements (earnings, listings, index weight changes) and fitting deltas per event type.

**Bayesian / regime-aware (optional)**
- Fit Bayesian regression with priors on beta_price and beta_sentiment to obtain posterior mean/variance for sentiment weight.
- Introduce regime features Z_t (NGX/JSE realized volatility, USD/NGN FX volatility, African news volume) and set sentiment weight w_s(t) = sigmoid(gamma0 + gamma * Z_t); down-weight sentiment during calm periods.

### 7. Evaluation Focused on African Markets
- Universe: NGX equities, JSE Top40, NSE blue chips, African ETFs/indices.
- Horizons: 1D, 3D, 5D bars; monitor return correlation, directional hit rate, forecast error reduction, and portfolio metrics (Sharpe, Calmar, turnover, drawdown).
- Run sub-period and sector breakdowns (banks, oil and gas, telcos, consumer) plus macro regimes (pre-COVID, FX devaluations, subsidy changes).
- If sentiment fails stability criteria the weight defaults to zero for that asset class.

### 8. Implementation Checklist
1. Configure high-priority sources (NGX/JSE/NSE + Nairametrics/BusinessDay) and lower-priority global feeds.
2. Build RSS + scraper jobs with metadata tagging and cache under data/raw/sentiment/.
3. Implement lexicon + transformer pipelines with local inference, caching intermediate logits for traceability.
4. Label several hundred African documents for validation, compute F1/accuracy, and derive ensemble weights.
5. Aggregate sentiment per asset/time bucket with Africa-first weighting; persist to data/processed/sentiment_factors.parquet.
6. Integrate deterministic blend into the TS forecaster, keep LLM path as minimum benchmark/fallback.
7. Add Bayesian/regime extensions when deterministic path proves profitable.
8. Enforce deployment gates: sentiment-enabled runs must beat no-sentiment baseline and pass statistical significance plus economic impact reviews.


## 📈 EXPECTED OUTCOMES

### Quantitative Improvements
- **30%** improvement in risk-adjusted returns
- **50%** reduction in tail risk exposure
- **20%** improvement in signal accuracy
- **100%** mathematical correctness

### Production Readiness
- **Institutional-grade** risk management
- **Professional** statistical validation
- **Regulatory** compliance ready
- **Real-time** monitoring capabilities

---

**Status**: ✅ **READY FOR IMPLEMENTATION**
**Priority**: **CRITICAL** - Required for institutional deployment
**Timeline**: 4-8 weeks for full optimization
**Investment**: High (mathematical expertise required)
**ROI**: High (institutional-grade system)

- 2025-11-23: Added stochastic StrategyOptimizer infrastructure (etl/strategy_optimizer.py, scripts/run_strategy_optimization.py, config/strategy_optimization_config.yml) with tests for sampling/constraints; no strategies hardcoded, all search spaces and objectives remain config-driven.
