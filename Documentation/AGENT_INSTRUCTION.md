# AI Developer Guardrails: Reality-Based Development Checklist

> **RUNTIME GUARDRAIL (WSL `simpleTrader_env` ONLY)**  
> Supported runtime: WSL + Linux venv `simpleTrader_env/bin/python` (`source simpleTrader_env/bin/activate`).  
> **Do not** use Windows interpreters/venvs (incl. `py`, `python.exe`, `.venv`, `simpleTrader_env\\Scripts\\python.exe`) — results are invalid.  
> Before reporting runs, include the runtime fingerprint (command + output): `which python`, `python -V`, `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` (see `Documentation/RUNTIME_GUARDRAILS.md`).

## Project Status (Updated: 2025-10-04)

### Current Phase: 4.5 - Time Series Cross-Validation ✅ COMPLETE
- **Implementation**: k-fold CV with expanding window
- **Test Coverage**: 85/85 tests passing (100%)
- **Configuration**: Modular, platform-agnostic architecture
- **Documentation**: Comprehensive (implementation_checkpoint.md, arch_tree.md, TIME_SERIES_CV.md)
- **Performance**: 5.5x temporal coverage improvement (15% → 83%)

### Completed Phases
1. ✅ **Phase 1**: ETL Foundation - Data extraction, validation, preprocessing, storage
2. ✅ **Phase 2**: Analysis Framework - Time series analysis with MIT standards
3. ✅ **Phase 3**: Visualization Framework - Publication-quality plots (7 types)
4. ✅ **Phase 4**: Caching Mechanism - 100% cache hit rate, 20x speedup
5. ✅ **Phase 4.5**: Time Series Cross-Validation - k-fold CV, backward compatible

### Next Phase: 5 - Portfolio Optimization
- **Focus**: Markowitz mean-variance optimization, risk parity
- **Prerequisites**: ✅ All data infrastructure complete
- **Status**: Ready to begin

### Integration Recovery Tracker
- Track and update pipeline remediation tasks in `Documentation/integration_fix_plan.md` before advancing Phase 5 work or refreshing `Documentation/INTEGRATION_TESTING_COMPLETE.md`.

## Concurrent Developer-Agent Coordination (Mandatory)

When multiple developer-agents or humans are active in the same repository:

- Start every implementation pass with `git status --porcelain`.
- Treat pre-existing modified/untracked files as active in-progress work by default.
- Inspect existing diffs before editing shared files (`git diff -- <file>`), and complement rather than overwrite.
- If ownership/intent is unclear for a shared file, stop and request direction before editing.
- Keep commits scoped to the task requested; do not bundle unrelated concurrent changes.
- Before finalizing, run compatibility checks (compile/smoke + fast regression lane where feasible).
- Final delivery must explicitly list:
  - files changed for the task,
  - files intentionally left untouched due to parallel ownership,
  - verification commands and outcomes.

## Auto-Portfolio Trader Remediation To-Do (Critical)

1. **Initial setup & scoping**: Review the auto-portfolio trader stack (`config/yfinance_config.yml`, `models/time_series_signal_generator.py`, `execution/paper_trading_engine.py`, related configs). Define the fix scope (align execution with forecast horizon, realistic cost model, live metrics) and stage iterative updates/backtests to avoid system disruption.
2. **Bar-aware trading loop**: Trigger trading only when a new daily bar arrives; if using intraday cadence, adjust lookback and add fallback logic. Test with a mock feed to confirm actions only fire on new bars or the intended intraday cadence.
3. **Forecast horizon alignment**: Match forecast horizon targets in `models/time_series_signal_generator.py` with entry/exit logic and `max_holding_days`; backtest to prove horizon, holding period, and targets stay in sync.
4. **Confidence calculation**: Refactor confidence to penalize weak/small edges and only inflate when signals are predictive and actionable. Compare confidence across models/regimes to ensure it discriminates rather than rewarding “safe” levels.
5. **Diagnostics scoring**: Rework `_evaluate_diagnostics` around forecast error and realized performance; compare old vs new diagnostics on known data so scores only rise with true quality improvements.
6. **Quant validation**: Gate on incremental edge (forecast error + performance uplift) and reduce bias toward always-long/short recent returns; validate in flat/down regimes so good models are not rejected unfairly.
7. **Cost model alignment**: Update routing logic and the `models/time_series_signal_generator.py` cost model for realistic roundtrip costs (slippage, fees) and adjust slippage handling in `execution/paper_trading_engine.py`. Test trades with/without the updated model to see net profit impacts.
8. **Execution simulation**: Add realistic LOB/depth-profile fallbacks for illiquid assets and refine slippage/execution cost calculations; keep paper-trading simulation consistent with the updated cost model and simulate illiquid names to measure slippage differences.
9. **Reporting fixes**: Ensure `win_rate` and `profit_factor` reflect only trades from the live run (no historical bleed). Track and store real-time forecast regression metrics to monitor forecaster health.
10. **Continuous testing & validation**: Add/refresh unit tests for confidence, horizon alignment, diagnostics scoring, and cost model; run iterative backtests per phase and maintain live monitoring for trades, confidence, execution costs, and portfolio performance.
11. **Final optimization**: After fixes validate in backtests, enable the barbell strategy (if applicable) in simulated/live-like mode and monitor for discrepancies during simulated capital runs.
12. **Documentation & reporting**: Document changes to horizons, execution modeling, diagnostics scoring, and cost handling; generate performance reports per phase vs baselines and update configs (`config/barbell.yml`, `execution/execution_config.yml`, etc.) to match new behaviour.
13. **Final iteration & ongoing monitoring**: Conduct a full review for stability/performance and continue iterative updates based on live monitoring, backtesting, and user feedback.

## Optimization & Hardening Backlog
- Use `Documentation/OPTIMIZATION_OPTIONS.md` as the canonical backlog for performance, scalability, caching, architecture, reliability, observability, security, and resource optimizations (vectorization gaps, cache invalidation, config consolidation, circuit breakers, structured logging, secrets rotation).
- Pull items only with profitability/stability justification and in alignment with the Core Directive and phase gates; prefer configuration-driven, incremental changes with targeted backtests over large refactors.

## Approved Time-Series Stack (Tier-1 default)
- Canonical reference: `Documentation/QUANT_TIME_SERIES_STACK.md` (pin in every AI companion's context). Consider Tier-1 the only sanctioned dependency set until profitability + GPU budget gates unlock higher tiers.
- Runtime: Python 3.10-3.12 inside `simpleTrader_env`, NumPy, pandas, SciPy.
 - Time-series libraries: statsmodels (SARIMAX), arch (GARCH), and in-repo SAMOSSA/MSSA-RL implementations.
 - TS governance: treat **SAMOSSA** as the primary Time Series baseline for regression metrics and ensemble comparisons; use SARIMAX only as a secondary candidate/fallback when SAMOSSA metrics are missing.
- Vectorization: default to **NumPy/pandas vectorized operations** for all numerical work on arrays/DataFrames (no row-by-row Python loops, `.iterrows()` or per-element `.apply()` in hot paths). Any non-vectorized implementation in ETL, analysis, forecasting, or portfolio math must be justified in comments and kept outside performance-critical code paths.
- Optional GPU assist: CuPy only when MSSA/SARIMAX workloads hit >70% CPU during brutal runs on <=8 GB GPUs. Escalate before introducing Tier-2 RAPIDS/torch stacks.
- Automation: Reuse the YAML/JSON snippets from `QUANT_TIME_SERIES_STACK.md` under `config/ai_companion.yml` so autonomous agents inherit the same guardrails automatically.
- NAV & Barbell: Treat `Documentation/NAV_RISK_BUDGET_ARCH.md` and `Documentation/NAV_BAR_BELL_TODO.md` as the canonical references for **TS-first, NAV-centric barbell wiring** (TS core signals as primary, LLM as capped fallback, options/derivatives strictly feature-flagged). Do not re-implement allocation logic ad hoc; route changes through these docs and `config/barbell.yml`/`risk/barbell_policy.py`.
- Quant validation & automation: When touching thresholds or quant gates, read `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` and `Documentation/QUANT_VALIDATION_AUTOMATION_TODO.md` first, and prefer updating configs + automation helpers (`scripts/sweep_ts_thresholds.py`, `scripts/estimate_transaction_costs.py`, `scripts/generate_config_proposals.py`) over baking constants into code.
- MTM & liquidation: Treat `Documentation/MTM_AND_LIQUIDATION_IMPLEMENTATION_PLAN.md` and the `scripts/liquidate_open_trades.py` contract as the canonical source for diagnostic mark-to-market behaviour; do not rely on this path for production PnL reporting.
- Synthetic data isolation: Synthetic datasets are strictly for pre-production/brutal validation. After that phase, disable `ENABLE_SYNTHETIC_PROVIDER`/`SYNTHETIC_ONLY`, drop `--data-source synthetic`/`--execution-mode synthetic` flags from production runners, point `PORTFOLIO_DB_PATH` back to the production DB, and ensure training/scoring/export paths reject synthetic-sourced rows. Live trading, dashboards, and model training must never consume synthetic data.
- Parallel defaults: `scripts/run_auto_trader.py` now defaults to parallel candidate prep + forecasts with GPU-first when available (`ENABLE_GPU_PARALLEL=1` + CUDA/torch present), otherwise CPU threads (`ENABLE_PARALLEL_TICKER_PROCESSING=1`, `ENABLE_PARALLEL_FORECASTS=1`). CPU-only environments are supported via `PARALLEL_TICKER_WORKERS` tuning. Stress evidence: `logs/automation/stress_parallel_20260107_202403/comparison.json`.
- Dependency baseline: `requirements.txt` now includes `torch==2.9.1` to support GPU-first defaults; `requirements-ml.txt` retains CUDA extras (CuPy/NVIDIA libs) for full GPU stacks.

### Time-Series Hyper-Parameters & Model Search

- Use `scripts/run_ts_model_search.py` + `ts_model_candidates` (see `etl/database_manager.py`) as the **canonical path** for exploring SARIMAX/SAMOSSA TS model candidates:
  - Do not introduce ad-hoc per-script SARIMAX grids; instead, extend the candidate grid in the model search script and the config-driven profiles in `config/model_profiles.yml`.
  - When comparing candidates, prefer the shared helpers in `etl/statistical_tests.py` (Diebold–Mariano-style tests, rank/stability metrics) instead of bespoke statistical code scattered across the repo.
- When building or modifying dashboards, prefer reading `visualizations/dashboard_automation.json` (emitted by `scripts/build_automation_dashboard.py`) so automation decisions (TS sweeps, costs, sleeve promotions, TS model candidates) remain **centralised and institutional-grade**.

### Heuristics vs Full Models vs ML Calibrators

- Real-time heuristics:
  - Use **lightweight, transparent heuristics** (e.g. quant validation pass/fail rates, simple MVS checks from `DatabaseManager.get_performance_summary()`, barbell gates) for:
    - Live monitoring dashboards,
    - Quick routing/guardrail decisions inside `scripts/run_auto_trader.py`,
    - Early warning signals in CI/brutal runs.
  - These should be cheap to compute, easy to explain, and always driven by config (no opaque state hidden in code).
- Full models:
  - Treat the **full TS ensemble + portfolio math** stack as the single source of truth for:
    - Official NAV calculation,
    - Risk reporting (drawdown, CVaR once implemented),
    - Production-grade backtesting and hyper-parameter evaluation.
  - Do not replace full-model outputs with heuristics in any place that feeds official reports or long-term research conclusions.
- ML calibrators:
  - Any future ML components (beyond TS baselines) should be used to **calibrate heuristics**, not to silently override full models:
    - Example: learning regime-aware bands for `min_expected_return` / `min_expected_profit` based on recent performance.
    - Example: predicting when a heuristic (e.g. simple pass-rate gate) is misaligned with full-model outcomes.
  - Respect the phase gates and capital thresholds in this file: do not introduce new ML calibrators for risk/threshold tuning until:
    - Spot-only TS stack is profitable over multiple regimes,
    - Quant health (GREEN/YELLOW) is stable in brutal reports,
    - A clear economic rationale for the calibrator has been written down.
  - All ML calibrators must remain **configuration-driven, replaceable, and observable** (their decisions should be logged and comparable to the underlying heuristics and full models they calibrate).

## Pre-Development AI Instructions

### Core Directive for AI Assistant
```
MANDATORY: Before suggesting any code or architecture:
1. Ask: "What evidence proves this approach is profitable?"
2. Require: Backtesting results with >10% annual returns
3. Demand: Working execution of previous phase before new features
4. Verify: All claims with actual data, not theoretical projections
5. Check: Configuration-driven design maintained
6. Ensure: Backward compatibility preserved
```

### Options / Derivatives Feature-Flag Discipline
- Treat **options/derivatives support as strictly opt-in** and configuration-driven.
- Do **not** touch core ETL/auto-trader paths for options unless all of the following are true:
  - `options_trading.enabled: true` in `config/options_config.yml`, and/or `ENABLE_OPTIONS=true` in the environment.
  - The existing spot-only brutal suite is green and `Documentation/INTEGRATION_TESTING_COMPLETE.md` reflects this.
  - `Documentation/BARBELL_OPTIONS_MIGRATION.md` remains consistent with any proposed changes.
- When adding options logic:
  - Route all long OTM options and synthetic convex structures into the **risk bucket** only.
  - Respect barbell guardrails (`max_options_weight`, `max_premium_pct_nav`) from `config/options_config.yml`.
  - Keep spot-only behaviour unchanged when options are disabled (feature-flag off = no behavioural regression).

### Barbell & Tail-Risk Evaluation Discipline
- Do **not** use Sharpe alone to judge tail-hedge / barbell components; they are insurance, not linear alpha.
- Prefer asymmetric / tail-aware metrics:
  - Sortino ratio (downside deviation only)
  - Omega ratio (gain vs loss mass around a MAR)
  - CVaR / Expected Shortfall and stress scenarios (1987/2008/2020-style shocks)
- Always assess barbell / long-vol strategies **at the portfolio level**:
  - Impact on skew/kurtosis and left-tail truncation
  - Cost of convexity vs reduction in crisis losses

## Phase-Gate Validation Checklist

### Before ANY Development Session
- [x] **Previous phase 100% complete?** Phase 4.5 complete (CV implementation)
- [ ] **Profitable strategy proven?** Awaiting Phase 5 (Portfolio Optimization)
- [x] **Working execution system?** Data pipeline + execution loop implemented (auto-trader + PaperTradingEngine/cTrader demo); live trading remains gated by quant health
- [x] **Budget constraints respected?** Free tier data sources only
- [ ] **Stack alignment locked?** Tier-1 stack from `QUANT_TIME_SERIES_STACK.md` confirmed as the only dependency delta; deviations documented + approved.

### Before Adding New Features
- [ ] **Business case proven?** How does this improve returns by >1%?
- [ ] **Complexity justified?** Can current system handle added complexity?
- [ ] **Dependencies validated?** All required data sources tested and working
- [ ] **Rollback plan ready?** How to revert if new feature fails

## Code Development Guards

### AI Instruction Template
```
You are developing a trading system with strict constraints:

REQUIREMENTS:
- Maximum 500 lines of code per phase
- Must work with free data sources only
- Prove profitability before adding complexity
- No ML models until Phase 7+ and $25K+ capital

FORBIDDEN RESPONSES:
- "This would be better with [complex framework]"
- "Consider using [ML model] for better performance"
- "You should architect this as microservices"
- Any suggestion requiring >$50/month operational cost

REQUIRED FOR EVERY CODE SUGGESTION:
- Exact line count estimate
- Data source validation method
- Performance test specification
- Failure mode handling
```

### Code Review Checklist (Every Session)
- Never delete files without explicit user permission (no `rm`/`unlink` without approval).
- [ ] **Line count under phase limit?** Count actual lines, not estimates
- [ ] **Dependencies verified working?** Test imports and data access
- [ ] **Error handling included?** Network failures, missing data, calculation errors
- [ ] **Performance metrics tracked?** Execution time, memory usage, accuracy
- [ ] **Documentation minimal but complete?** What it does, why it exists, how to test
- [ ] **Entry points compile?** Run targeted `python -m compileall` / smoke CLI to catch indentation or syntax regressions
- [ ] **Orchestrators decomposed?** Break pipelines >200 lines into helpers; keep CLI functions as thin coordinators

### Extractor Safety Requirements
- Never monkey-patch third-party clients (e.g., `requests.Session.request`); pass configuration via documented parameters
- Guard vectorised math (log returns, diff) against empty, single-row, or non-positive data before computing ratios
- Treat cache refreshes as optional: fall back gracefully when cache metadata is missing or stale

### Logging Discipline
- Library modules must avoid `logging.basicConfig`; configure logging in entry points or dedicated bootstrap modules only
- Use module-level `getLogger` consistently and honour the project-wide logging level

## Data Science Validation Protocol

### Before ANY ML Model Discussion
- [ ] **Baseline strategy profitable?** Simple moving average beating buy-and-hold
- [ ] **Data quality proven?** >95% data availability, <1% outliers
- [ ] **Feature engineering justified?** Clear hypothesis about why feature helps
- [ ] **Computational resources adequate?** Memory and GPU requirements calculated

### ML Model Introduction Gates

#### SARIMAX Introduction (Phase 7 Only)
**Prerequisites checklist:**
- [ ] 6+ months profitable paper trading documented
- [ ] Simple strategy Sharpe ratio >1.0 proven
- [ ] Clear economic hypothesis for features
- [ ] GPU memory requirements <8GB calculated

#### SAMOSSA Introduction (Phase 10 Only)
**Prerequisites checklist:**
- [ ] SARIMAX models beating baseline by >1% annually
- [ ] Mathematical understanding of SSA demonstrated
- [ ] GPU implementation tested on synthetic data
- [ ] Model interpretation framework ready

#### DQN Introduction (Phase 12+ Only)
**Prerequisites checklist:**
- [ ] $100,000+ live capital committed
- [ ] Other ML models adding proven value
- [ ] Reinforcement learning theory understood
- [ ] Multi-month training timeline accepted

## Architecture Discipline Checklist

### Before Any "Refactoring" or "Improvement"
- [ ] **Current system profitable?** Prove with 30+ days live results
- [ ] **Specific problem identified?** Performance bottleneck or profit limitation
- [ ] **Solution impact quantified?** Expected improvement in returns/speed
- [ ] **Implementation time bounded?** Maximum 1 week for any change

### Anti-Pattern Detection
**Red flag responses from AI:**
- [ ] Suggests complete rewrite of working code
- [ ] Proposes framework changes without profit justification
- [ ] Recommends "best practices" that add complexity
- [ ] Mentions "scalability" before achieving profitability
- [ ] Discusses "maintainability" for <1000 line codebase

## Performance Monitoring Framework

### Daily Development Metrics
```bash
# Run these commands every development session
echo "=== Reality Check ==="
echo "Lines of code: $(find . -name '*.py' | xargs wc -l | tail -1)"
echo "Git commits today: $(git log --since='1 day ago' --oneline | wc -l)"
echo "Working strategies: $(python test_strategies.py --count-profitable)"
echo "Monthly costs: $$(python calculate_costs.py)"
```

### Weekly Validation Protocol
- [ ] **Strategy performance test** - Run full backtest, verify >10% annual returns
- [ ] **System integration test** - Execute end-to-end trade simulation
- [ ] **Cost accounting** - Verify monthly expenses under budget
- [ ] **Complexity audit** - Count lines of code, assess if each is necessary

## AI Session Management

### Start-of-Session Protocol
```
Before discussing any development:
1. Show current profitability metrics
2. Confirm current phase completion status
3. State specific problem being solved
4. Set session time limit (maximum 2 hours)
```

### End-of-Session Validation
- [ ] **Code works?** All suggested code tested and functional
- [ ] **Performance measured?** Runtime, accuracy, resource usage documented
- [ ] **Next steps clear?** Specific tasks for next session defined
- [ ] **No scope creep?** Did not add features beyond current phase

## Emergency Stop Conditions

### Immediate Development Halt Required When:
- [ ] **Monthly costs exceed budget** by any amount
- [ ] **Strategy loses money** for >7 consecutive days
- [ ] **System cannot execute trades** for >24 hours
- [ ] **Code complexity** exceeds phase limits by >20%
- [ ] **AI suggests major architecture changes** without profit justification

### Recovery Protocol:
1. Stop all development immediately
2. Revert to last working version
3. Re-run profitability validation
4. Identify specific failure cause
5. Resume only after root cause addressed

## Documentation Requirements

### Mandatory Files (Updated Weekly)
- [ ] **PERFORMANCE.md** - Current strategy returns, Sharpe ratio, drawdowns
- [ ] **COSTS.md** - Monthly expense breakdown with receipts
- [ ] **ROADMAP.md** - Phase completion status, next priorities
- [ ] **FAILURES.md** - What didn't work and why, lessons learned

### Decision Log Template
```
Date: 2024-XX-XX
Decision: [What was decided]
Reason: [Why this approach chosen]
Evidence: [Data supporting decision]
Success Criteria: [How to measure if this worked]
Review Date: [When to reassess]
```

## Human Oversight Checkpoints

### Weekly Self-Assessment Questions
1. **Am I making money?** Show actual profitable trades, not backtests
2. **Am I staying focused?** Working on current phase, not future features
3. **Am I being realistic?** Costs under budget, timeline reasonable
4. **Am I avoiding complexity?** System simple enough to understand completely

### Monthly Reality Check
- [ ] **Third-party validation** - Show results to someone not involved in project
- [ ] **Competitor analysis** - How do results compare to index funds?
- [ ] **Stress test** - What happens if strategy fails for 30 days?
- [ ] **Exit planning** - At what point would you stop this project?

This checklist is designed to prevent the architectural over-engineering and analysis paralysis that has characterized your previous development cycles. The key is rigorous adherence to the phase-gate approach and absolute rejection of complexity until profitability is proven.


# Enhanced DevOps Guardrails: Testing, Version Control, and Deployment

## Testing Framework (Reality-Based)

### Unit Testing Requirements by Phase

#### Phase 1-3: Core Testing Only
**Maximum 200 lines of test code** - Don't spend more time testing than developing

**Standing instruction**: Always build a unit test for every critical implementation path before shipping changes.

```python
# test_core_functions.py - Essential tests only
import pytest
import pandas as pd
import numpy as np
from portfolio_system import PortfolioCalculator, DataFetcher

class TestCoreBusinessLogic:
    """Test only profit-critical functions"""
    
    def test_portfolio_return_calculation(self):
        # This is money - test thoroughly
        calc = PortfolioCalculator()
        prices = pd.Series([100, 110, 105])
        returns = calc.calculate_returns(prices)
        
        assert abs(returns.iloc[1] - 0.10) < 1e-6, "Return calculation wrong"
        assert abs(returns.iloc[2] - (-0.045)) < 1e-6, "Return calculation wrong"
    
    def test_position_sizing_math(self):
        # Cash management errors lose money
        calc = PortfolioCalculator()
        result = calc.calculate_position_size(
            cash=10000, target_weight=0.6, price=100
        )
        
        assert result == 60, f"Expected 60 shares, got {result}"
        assert calc.remaining_cash == 4000, "Cash tracking wrong"
    
    def test_data_fetcher_handles_missing_data(self):
        # Missing data causes strategy failures
        fetcher = DataFetcher()
        data = fetcher.fetch_prices(['INVALID_SYMBOL'])
        
        assert data.empty or data.isna().all().all(), "Should handle missing data gracefully"

# Mandatory assertions in production code
def calculate_portfolio_value(positions, prices):
    """Calculate total portfolio value with built-in validation"""
    assert len(positions) == len(prices), "Position-price mismatch"
    assert all(pos >= 0 for pos in positions), "Negative positions not allowed"
    assert all(price > 0 for price in prices), "Invalid prices detected"
    
    value = sum(pos * price for pos, price in zip(positions, prices))
    
    assert value >= 0, "Portfolio value cannot be negative"
    return value
```

#### Phase 4-6: Extended Testing
**Maximum 500 lines of test code** - Add integration tests only

```python
# test_integration.py - End-to-end validation
class TestTradingIntegration:
    
    def test_complete_trade_cycle(self):
        """Test buy -> hold -> sell -> cash reconciliation"""
        system = TradingSystem(initial_cash=10000)
        
        # Execute complete cycle
        system.buy('SPY', dollar_amount=5000)
        system.wait_days(30)  # Simulate holding period
        system.sell('SPY', shares='all')
        
        # Validate cash reconciliation
        assert abs(system.cash - system.initial_cash) < 100, "Cash not properly tracked"
        assert len(system.positions) == 0, "Positions not properly closed"
    
    def test_strategy_performance_validation(self):
        """Test strategy meets minimum performance requirements"""
        strategy = MomentumStrategy()
        historical_data = fetch_test_data('2020-01-01', '2023-12-31')
        
        performance = strategy.backtest(historical_data)
        
        assert performance.annual_return > 0.08, f"Strategy only returned {performance.annual_return:.2%}"
        assert performance.max_drawdown < 0.20, f"Drawdown too high: {performance.max_drawdown:.2%}"
        assert performance.sharpe_ratio > 0.8, f"Sharpe ratio too low: {performance.sharpe_ratio:.2f}"
```

### Test Automation Pipeline

#### Pre-Commit Testing (Git Hooks)
```bash
#!/bin/sh
# .git/hooks/pre-commit - Mandatory before any commit

echo "Running pre-commit validation..."

# 1. Run core business logic tests
python -m pytest test_core_functions.py -v
if [ $? -ne 0 ]; then
    echo "❌ Core tests failed - fix before committing"
    exit 1
fi

# 2. Validate strategy performance
python test_strategy_performance.py
if [ $? -ne 0 ]; then
    echo "❌ Strategy performance degraded - investigate"
    exit 1
fi

# 3. Check code complexity
lines=$(find . -name "*.py" | xargs wc -l | tail -1 | awk '{print $1}')
max_lines_for_phase=$(python get_phase_limits.py)
if [ $lines -gt $max_lines_for_phase ]; then
    echo "❌ Code too complex for current phase: $lines > $max_lines_for_phase"
    exit 1
fi

# 4. Verify costs are under budget
python calculate_monthly_costs.py --validate
if [ $? -ne 0 ]; then
    echo "❌ Monthly costs exceed budget"
    exit 1
fi

echo "✅ All validations passed"
```

#### Daily Automated Tests
```python
# automated_daily_tests.py - Run every day at market close
import sys
import logging
from datetime import datetime

class DailyValidation:
    def __init__(self):
        self.logger = logging.getLogger('daily_validation')
        
    def run_all_tests(self):
        """Run comprehensive daily validation"""
        results = {
            'data_quality': self.test_data_quality(),
            'strategy_performance': self.test_strategy_performance(),
            'system_health': self.test_system_health(),
            'cost_tracking': self.test_cost_tracking()
        }
        
        # Alert on any failures
        failures = [test for test, passed in results.items() if not passed]
        if failures:
            self.send_alert(f"Daily tests failed: {failures}")
            sys.exit(1)
            
        self.logger.info("All daily tests passed")
    
    def test_data_quality(self):
        """Validate market data is complete and accurate"""
        from data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        today_data = fetcher.get_latest_prices(['SPY', 'TLT', 'VTI'])
        
        # Data quality checks
        assert not today_data.empty, "No market data retrieved"
        assert not today_data.isna().any().any(), "Missing data in feed"
        assert (today_data > 0).all().all(), "Invalid negative prices"
        
        return True
    
    def test_strategy_performance(self):
        """Validate strategy is still profitable"""
        from strategy import CurrentStrategy
        
        strategy = CurrentStrategy()
        recent_performance = strategy.get_performance_last_30_days()
        
        # Performance validation
        assert recent_performance.total_return > -0.05, "Strategy losing too much money"
        assert recent_performance.max_drawdown < 0.10, "Drawdown too high recently"
        
        return True
    
    def test_system_health(self):
        """Check system resource usage and errors"""
        import psutil
        
        # Memory usage
        memory_percent = psutil.virtual_memory().percent
        assert memory_percent < 80, f"High memory usage: {memory_percent}%"
        
        # Disk space
        disk_usage = psutil.disk_usage('/').percent
        assert disk_usage < 90, f"Low disk space: {disk_usage}%"
        
        # Check error logs
        error_count = self.count_recent_errors()
        assert error_count < 10, f"Too many recent errors: {error_count}"
        
        return True

if __name__ == "__main__":
    validator = DailyValidation()
    validator.run_all_tests()
```

## Version Control Strategy

### Repository Structure
```
portfolio-system/
├── .git/
├── .gitignore                 # Exclude credentials, cache files
├── README.md                  # Current status, how to run
├── PERFORMANCE.md             # Latest strategy results
├── requirements.txt           # Exact dependency versions
├── 
├── src/                       # All source code
│   ├── core/                  # Business logic
│   ├── strategies/            # Trading strategies
│   ├── data/                  # Data management
│   └── execution/             # Trade execution
├── 
├── tests/                     # All test code
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── performance/           # Strategy validation
├── 
├── config/                    # Configuration files
├── scripts/                   # Deployment and utility scripts
└── docs/                      # Documentation
```

### Branching Strategy (Simplified)
```bash
# Main branches only - no complex git-flow
main           # Production-ready code only
development    # Daily development work
hotfix/*       # Emergency fixes to production
```

### Commit Standards
```bash
# Mandatory commit message format
git commit -m "TYPE: Brief description

VALIDATION:
- Tests passing: yes/no
- Performance impact: +X.X% / no change / -X.X%
- Cost impact: $X increase/decrease/no change
- Phase status: X% complete

EVIDENCE:
- Backtest results: [file/link]
- Performance metrics: [numbers]"

# Examples of good commits:
git commit -m "FIX: Portfolio calculation rounding error

VALIDATION:
- Tests passing: yes
- Performance impact: +0.1% (better accuracy)
- Cost impact: no change
- Phase status: 95% complete

EVIDENCE:
- Unit tests now pass 100%
- Backtest shows 0.1% improvement in Sharpe ratio"
```

### Release Tagging
```bash
# Tag every deployment with performance data
git tag -a v1.2.0 -m "Phase 2 Complete
Strategy Performance:
- Annual Return: 12.3%
- Sharpe Ratio: 1.45
- Max Drawdown: 8.2%
- Live trading: 30 days profitable
- Monthly cost: $23.50"
```

## Deployment Pipeline

### Pre-Deployment Checklist
```python
# deployment_validation.py - Must pass before any deployment
class DeploymentValidator:
    
    def validate_deployment_readiness(self):
        """Comprehensive pre-deployment validation"""
        
        checks = [
            self.check_strategy_profitability(),
            self.check_code_quality(),
            self.check_system_resources(),
            self.check_configuration(),
            self.check_backup_systems(),
            self.check_rollback_plan()
        ]
        
        assert all(checks), "Deployment validation failed"
        return True
    
    def check_strategy_profitability(self):
        """Strategy must be profitable before deployment"""
        from strategy_validator import StrategyValidator
        
        validator = StrategyValidator()
        performance = validator.validate_last_90_days()
        
        # Strict profitability requirements
        assert performance.total_return > 0.02, "Strategy not profitable enough"
        assert performance.win_rate > 0.45, "Win rate too low"
        assert performance.max_drawdown < 0.15, "Risk too high"
        
        self.log_validation("Strategy profitability: PASSED")
        return True
    
    def check_code_quality(self):
        """Code quality gates"""
        import subprocess
        
        # Run all tests
        test_result = subprocess.run(['python', '-m', 'pytest', '-x'], 
                                   capture_output=True)
        assert test_result.returncode == 0, "Tests failing"
        
        # Check code complexity
        complexity_result = subprocess.run(['python', 'check_complexity.py'], 
                                         capture_output=True)
        assert complexity_result.returncode == 0, "Code too complex"
        
        self.log_validation("Code quality: PASSED")
        return True
    
    def check_rollback_plan(self):
        """Verify rollback capability"""
        # Test that we can revert to previous version
        assert os.path.exists('rollback_script.sh'), "No rollback script"
        assert os.path.exists('config/previous_version.json'), "No previous config"
        
        # Test rollback script
        rollback_test = subprocess.run(['bash', 'rollback_script.sh', '--dry-run'], 
                                     capture_output=True)
        assert rollback_test.returncode == 0, "Rollback script broken"
        
        self.log_validation("Rollback plan: PASSED")
        return True
```

### Production Deployment Script
```bash
#!/bin/bash
# deploy.sh - Safe production deployment

set -e  # Exit on any error

echo "=== PORTFOLIO SYSTEM DEPLOYMENT ==="
echo "Timestamp: $(date)"
echo "Git commit: $(git rev-parse HEAD)"

# 1. Pre-deployment validation
echo "Running deployment validation..."
python deployment_validation.py
if [ $? -ne 0 ]; then
    echo "❌ Deployment validation failed"
    exit 1
fi

# 2. Create backup of current system
echo "Creating system backup..."
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/${timestamp}
cp -r config/ backups/${timestamp}/
cp -r data/ backups/${timestamp}/
echo "✅ Backup created: backups/${timestamp}"

# 3. Update system with minimal downtime
echo "Updating system..."
systemctl stop portfolio-system || true
sleep 5

# Deploy new code
cp -r src/* /opt/portfolio-system/
cp config/production.yaml /opt/portfolio-system/config/

# Update dependencies if needed
pip install -r requirements.txt

# 4. Start system and validate
systemctl start portfolio-system
sleep 10

# 5. Post-deployment health check
echo "Running post-deployment health check..."
python health_check.py --comprehensive
if [ $? -ne 0 ]; then
    echo "❌ Health check failed - rolling back"
    bash rollback_script.sh
    exit 1
fi

# 6. Create deployment record
echo "Recording deployment..."
cat > deployments/${timestamp}.json << EOF
{
    "timestamp": "$(date -Iseconds)",
    "git_commit": "$(git rev-parse HEAD)",
    "git_branch": "$(git branch --show-current)",
    "deployer": "$(whoami)",
    "validation_passed": true,
    "rollback_available": "backups/${timestamp}"
}
EOF

echo "✅ Deployment successful"
echo "Monitoring for 1 hour for any issues..."

# Monitor system for first hour
for i in {1..12}; do
    sleep 300  # 5 minutes
    python health_check.py --quick
    if [ $? -ne 0 ]; then
        echo "❌ Post-deployment issue detected"
        bash rollback_script.sh
        exit 1
    fi
    echo "✅ Health check $i/12 passed"
done

echo "🎉 Deployment completed successfully"
```

### Rollback Procedure
```bash
#!/bin/bash
# rollback_script.sh - Emergency rollback capability

set -e

echo "=== EMERGENCY ROLLBACK ==="
echo "Timestamp: $(date)"

# Stop current system
systemctl stop portfolio-system

# Get latest backup
latest_backup=$(ls -t backups/ | head -1)
echo "Rolling back to: ${latest_backup}"

# Restore configuration and data
cp -r backups/${latest_backup}/config/* config/
cp -r backups/${latest_backup}/data/* data/

# Restore previous code version
git checkout HEAD~1  # Go back one commit

# Restart system
systemctl start portfolio-system
sleep 10

# Validate rollback
python health_check.py --quick
if [ $? -eq 0 ]; then
    echo "✅ Rollback successful"
    
    # Send alert about rollback
    python send_alert.py "System rolled back to ${latest_backup} due to deployment issues"
else
    echo "❌ Rollback failed - manual intervention required"
    exit 1
fi
```

## Monitoring and Alerting

### Real-Time System Monitoring
```python
# system_monitor.py - Continuous system monitoring
import time
import logging
from datetime import datetime

class SystemMonitor:
    def __init__(self):
        self.alert_thresholds = {
            'memory_usage': 80,      # Percent
            'error_rate': 5,         # Errors per hour
            'strategy_drawdown': 10,  # Percent
            'data_staleness': 30     # Minutes
        }
    
    def continuous_monitoring(self):
        """Run continuous system monitoring"""
        while True:
            try:
                self.check_system_health()
                self.check_strategy_performance()
                self.check_data_quality()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.send_critical_alert(f"Monitoring system failed: {e}")
                time.sleep(60)  # Wait before retry
    
    def check_strategy_performance(self):
        """Monitor strategy for concerning patterns"""
        from strategy import get_current_strategy
        
        strategy = get_current_strategy()
        current_drawdown = strategy.get_current_drawdown()
        
        if current_drawdown > self.alert_thresholds['strategy_drawdown']:
            self.send_alert(f"High drawdown detected: {current_drawdown:.1f}%")
            
        # Check for unusual patterns
        recent_trades = strategy.get_recent_trades(hours=24)
        if len(recent_trades) > 50:  # Too many trades
            self.send_alert("Unusual trading activity detected")
```

This testing and deployment framework ensures your system remains profitable while adding necessary software engineering rigor. The key principle is that all testing and deployment complexity must be justified by the business value at stake - don't over-engineer for a simple trading system.
