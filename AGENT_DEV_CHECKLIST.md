# AI Developer Guardrails: Reality-Based Development Checklist

## Pre-Development AI Instructions

### Core Directive for AI Assistant
```
MANDATORY: Before suggesting any code or architecture:
1. Ask: "What evidence proves this approach is profitable?"
2. Require: Backtesting results with >10% annual returns
3. Demand: Working execution of previous phase before new features
4. Verify: All claims with actual data, not theoretical projections
```

## Phase-Gate Validation Checklist

### Before ANY Development Session
- [ ] **Previous phase 100% complete?** No exceptions, no "mostly done"
- [ ] **Profitable strategy proven?** Show actual backtest results >10% annual
- [ ] **Working execution system?** Demonstrate end-to-end trade execution
- [ ] **Budget constraints respected?** Monthly costs documented and under limits

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
- [ ] **Line count under phase limit?** Count actual lines, not estimates
- [ ] **Dependencies verified working?** Test imports and data access
- [ ] **Error handling included?** Network failures, missing data, calculation errors
- [ ] **Performance metrics tracked?** Execution time, memory usage, accuracy
- [ ] **Documentation minimal but complete?** What it does, why it exists, how to test

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