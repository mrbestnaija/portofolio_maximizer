# Local Claude Agent Instructions: MIT-Rigor Portfolio Management System

## Agent Mission Statement
Guide systematic implementation of production-grade portfolio management system using MIT academic standards. Execute ONE task at a time with vectorized operations, requiring explicit approval before progression.

## Agent Operating Principles

### 1. Implementation Protocol
```python
TASK_EXECUTION_PATTERN = {
    "mathematical_foundation": "Establish theoretical basis first",
    "vectorized_implementation": "NumPy/Pandas operations only",
    "validation_criteria": "Objective pass/fail metrics",
    "approval_gate": "Explicit user confirmation required",
    "code_limit": "Maximum 50 lines per task"
}
```

### 2. Quality Standards
- **Mathematical Rigor**: Every implementation backed by quantitative theory
- **Vectorized Operations**: No explicit loops in production code
- **Minimal Code**: Shortest functional implementation
- **Production Ready**: Type hints, error handling, logging
- **Measurable Success**: Quantitative validation criteria

## Phase-by-Phase Agent Instructions

### PHASE 1: Foundation (Tasks 1-16)
**Objective**: Prove basic profitability with $1000 simulation

#### Task 1: Environment Setup
**Mathematical Foundation**: Statistical computing requirements for financial time series
**Implementation**: Core library validation with version pinning
**Success Criteria**: Import test of all required packages
**Approval Required**: Confirm package versions before Task 2

#### Task 2: Data Source Validation  
**Mathematical Foundation**: Market microstructure data requirements
**Implementation**: Vectorized data quality checks on 10 liquid ETFs
**Success Criteria**: 5+ years daily data with <1% missing values
**Approval Required**: Data quality report confirmation

#### Task 3: Portfolio Mathematics Engine
**Mathematical Foundation**: Portfolio theory fundamentals (Markowitz 1952)
**Implementation**: 
```python
def calculate_portfolio_metrics(returns: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    """Vectorized portfolio calculations"""
    portfolio_returns = np.dot(returns, weights)
    return {
        'total_return': np.prod(1 + portfolio_returns) - 1,
        'volatility': np.std(portfolio_returns) * np.sqrt(252),
        'sharpe_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
    }
```
**Success Criteria**: SPY 1-year return calculation matches Yahoo Finance
**Approval Required**: Mathematical validation before proceeding

### PHASE 2: Strategy Implementation (Tasks 4-8)
**Objective**: Single profitable strategy beating SPY

#### Task 4: Backtest Engine Core
**Mathematical Foundation**: Time series bootstrap methodology
**Implementation**: Vectorized rolling window backtesting
**Success Criteria**: Complete 60/40 SPY/TLT backtest (2015-2023)
**Approval Required**: Backtest methodology validation

#### Task 5: Performance Attribution
**Mathematical Foundation**: Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
**Implementation**: Vectorized performance calculations
**Success Criteria**: >8% annual returns, <15% max drawdown
**Approval Required**: Performance validation against benchmarks

### Agent Decision Tree for Task Progression

```python
def evaluate_task_completion(task_result: Dict) -> str:
    """Agent decision logic for task progression"""
    
    if task_result['success_criteria_met'] and task_result['user_approved']:
        return "PROCEED_NEXT_TASK"
    elif task_result['success_criteria_met'] and not task_result['user_approved']:
        return "AWAIT_USER_APPROVAL"
    else:
        return "ITERATE_CURRENT_TASK"

def should_advance_phase(current_phase: int, completed_tasks: List[str]) -> bool:
    """Phase advancement criteria"""
    phase_requirements = {
        1: ['profitable_strategy', 'working_execution', 'risk_controls'],
        2: ['risk_metrics', 'drawdown_protection', 'regime_detection'],
        3: ['technical_indicators', 'alternative_data', 'multi_factor']
    }
    
    return all(req in completed_tasks for req in phase_requirements[current_phase])
```

## Agent Response Format

### For Each Task Request:
```
## Task [N]: [Task Name]

### Mathematical Foundation
[Theoretical basis and relevant equations]

### Implementation Approach  
[Vectorized solution strategy]

### Code Implementation
```python
[Maximum 50 lines of production-ready code]
```

### Validation Criteria
[Specific, measurable success metrics]

### Required Approval
[What user must confirm before proceeding]
```

## Agent Constraints and Guardrails

### Mandatory Restrictions
1. **No progression** without explicit user approval
2. **Maximum 50 lines** per code implementation  
3. **Vectorized operations only** - no explicit loops
4. **Mathematical justification** required for all approaches
5. **Quantitative validation** before task completion

### Quality Gates
```python
QUALITY_CHECKLIST = {
    "mathematical_foundation": "Theory explained with equations",
    "vectorized_implementation": "NumPy/Pandas operations confirmed", 
    "type_hints": "All functions properly typed",
    "error_handling": "Exception handling implemented",
    "validation_metrics": "Quantitative success criteria defined",
    "user_approval": "Explicit confirmation received"
}
```

## Agent State Management

### Current Context Tracking
```python
agent_state = {
    "current_phase": 1,
    "current_task": 1,
    "completed_tasks": [],
    "pending_approval": None,
    "performance_metrics": {},
    "code_base_size": 0,  # Track total lines
    "last_validation_result": None
}
```

### Success Metrics Dashboard
```python
phase_success_criteria = {
    "phase_1": {
        "min_annual_return": 0.08,
        "max_drawdown": 0.15,
        "min_sharpe_ratio": 1.0,
        "beat_spy_by": 0.02,
        "max_code_lines": 1000
    },
    "phase_2": {
        "risk_metrics_implemented": True,
        "regime_detection_accuracy": 0.7,
        "correlation_tracking": True
    }
}
```

## Agent Initialization

**Ready to begin Phase 1, Task 1: Environment Setup**

Awaiting user confirmation to proceed with mathematical foundation establishment and vectorized implementation of core portfolio management infrastructure.

**Current Status**: Initialized and ready for first task execution
**Next Action**: Environment Setup with statistical computing validation
**Approval Required**: Confirm readiness to begin systematic implementation