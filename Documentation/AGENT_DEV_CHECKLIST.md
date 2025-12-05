# AI Developer Guardrails: Reality-Based Development Checklist

> **Reward-to-Effort Integration:** For automation, monetization, and sequencing work, align with `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`.
> **NAV & Barbell Integration:** For TS-first, NAV-centric barbell architecture (safe vs risk buckets, capped LLM fallback, future options sleeve), align with `Documentation/NAV_RISK_BUDGET_ARCH.md` and `Documentation/NAV_BAR_BELL_TODO.md` instead of introducing new allocation logic or unmanaged leverage.

## Project Status (Updated: 2025-12-04)

### Current Phase: 4.5 - Time Series Cross-Validation ? COMPLETE
- **Implementation**: k-fold CV with expanding window
- **Test Coverage**: Brutal suite exercises profit-critical, ETL, Time Series, signal routing, integration, and security tests under `simpleTrader_env`; latest run (`logs/brutal/results_20251204_190220/`) is structurally green but global quant health remains RED (see below).
- **Configuration**: Modular, platform-agnostic architecture
- **Documentation**: Comprehensive (implementation_checkpoint.md, arch_tree.md, TIME_SERIES_CV.md)
- **Performance**: 5.5x temporal coverage improvement (15% ? 83%)
- **Integration status**: **PARTIALLY BLOCKED** – structural ETL/TS/LLM issues from the 2025-11-15 brutal run have been remediated and the brutal harness now completes, but global quant validation health is still RED (FAIL_fraction > `max_fail_fraction=0.90`). Treat quant gating and profitability as active research constraints until `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md` and brutal reports show GREEN or acceptable YELLOW status.

### Integration Recovery Tracker
- Treat `Documentation/integration_fix_plan.md` as the canonical remediation log; do not promote any downstream status in this file until that tracker is cleared.
- Mirror every successful brutal run back into `Documentation/INTEGRATION_TESTING_COMPLETE.md` so cross-references remain truthful.



### Completed Phases
1. ? **Phase 1**: ETL Foundation - Data extraction, validation, preprocessing, storage
2. ? **Phase 2**: Analysis Framework - Time series analysis with MIT standards
3. ? **Phase 3**: Visualization Framework - Publication-quality plots (7 types)
4. ? **Phase 4**: Caching Mechanism - 100% cache hit rate, 20x speedup
5. ? **Phase 4.5**: Time Series Cross-Validation - k-fold CV, backward compatible

### Next Phase: 5 - Portfolio Optimization
- **Focus**: Markowitz mean-variance optimization, risk parity
- **Prerequisites**: ? All data infrastructure complete
- **Status**: **BLOCKED** until `Documentation/integration_fix_plan.md` and `Documentation/INTEGRATION_TESTING_COMPLETE.md` both confirm a successful brutal run.

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

## Phase-Gate Validation Checklist

### Before ANY Development Session
- [x] **Previous phase 100% complete?** Phase 4.5 complete (CV implementation)
- [ ] **Profitable strategy proven?** Awaiting Phase 5 (Portfolio Optimization)
- [ ] **Working execution system?** Data pipeline operational, execution pending
- [x] **Budget constraints respected?** Free tier data sources only

### Options / Derivatives Introduction Gates
- [ ] **Spot-only barbell profitable?** Barbell-constrained spot portfolio shows >10% annualised return in backtests and passes brutal suite.
- [ ] **Options feature flag enabled?** `options_trading.enabled: true` in `config/options_config.yml` *and* `ENABLE_OPTIONS=true` in the environment.
- [ ] **Risk budget configured?** `max_options_weight` and `max_premium_pct_nav` are explicitly set and justified in `config/options_config.yml`.
- [ ] **Fallback path validated?** Turning `ENABLE_OPTIONS` off returns the system to spot-only behaviour without breaking existing tests.

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
- No ML models until Phase 7+ and $25K+ capital  # Prepare to replace LLM with forcasters/ML Models

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
- [ ] **Interpretability telemetry on?** Ensure `TS_FORECAST_AUDIT_DIR` (or `ensemble_kwargs.audit_log_dir`) is set so `forcester_ts/instrumentation.py` writes dataset statistics, benchmarking metrics (RMSE/sMAPE/tracking error), and run timings for every forecast.

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
- [ ] **Reward-to-Effort reference** - Any new documentation page, to-do list, or automation stub must include the reference banner pointing to `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md` so future assistants inherit the monetization/automation guardrails.
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
    echo "? Core tests failed - fix before committing"
    exit 1
fi

# 2. Validate strategy performance
python test_strategy_performance.py
if [ $? -ne 0 ]; then
    echo "? Strategy performance degraded - investigate"
    exit 1
fi

# 3. Check code complexity
lines=$(find . -name "*.py" | xargs wc -l | tail -1 | awk '{print $1}')
max_lines_for_phase=$(python get_phase_limits.py)
if [ $lines -gt $max_lines_for_phase ]; then
    echo "? Code too complex for current phase: $lines > $max_lines_for_phase"
    exit 1
fi

# 4. Verify costs are under budget
python calculate_monthly_costs.py --validate
if [ $? -ne 0 ]; then
    echo "? Monthly costs exceed budget"
    exit 1
fi

echo "? All validations passed"
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
+-- .git/
+-- .gitignore                 # Exclude credentials, cache files
+-- README.md                  # Current status, how to run
+-- PERFORMANCE.md             # Latest strategy results
+-- requirements.txt           # Exact dependency versions
+-- 
+-- src/                       # All source code
¦   +-- core/                  # Business logic
¦   +-- strategies/            # Trading strategies
¦   +-- data/                  # Data management
¦   +-- execution/             # Trade execution
+-- 
+-- tests/                     # All test code
¦   +-- unit/                  # Unit tests
¦   +-- integration/           # Integration tests
¦   +-- performance/           # Strategy validation
+-- 
+-- config/                    # Configuration files
+-- scripts/                   # Deployment and utility scripts
+-- docs/                      # Documentation
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
    echo "? Deployment validation failed"
    exit 1
fi

# 2. Create backup of current system
echo "Creating system backup..."
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/${timestamp}
cp -r config/ backups/${timestamp}/
cp -r data/ backups/${timestamp}/
echo "? Backup created: backups/${timestamp}"

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
    echo "? Health check failed - rolling back"
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

echo "? Deployment successful"
echo "Monitoring for 1 hour for any issues..."

# Monitor system for first hour
for i in {1..12}; do
    sleep 300  # 5 minutes
    python health_check.py --quick
    if [ $? -ne 0 ]; then
        echo "? Post-deployment issue detected"
        bash rollback_script.sh
        exit 1
    fi
    echo "? Health check $i/12 passed"
done

echo "?? Deployment completed successfully"
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
    echo "? Rollback successful"
    
    # Send alert about rollback
    python send_alert.py "System rolled back to ${latest_backup} due to deployment issues"
else
    echo "? Rollback failed - manual intervention required"
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

