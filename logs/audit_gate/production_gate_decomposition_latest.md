# Production Gate Decomposition

- Source artifact: `C:\Users\Bestman\personal_projects\portfolio_maximizer_v45\portfolio_maximizer_v45\logs\audit_gate\production_gate_latest.json`
- Source timestamp: `2026-03-16T18:39:15Z`
- Phase3 reason: `GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL`
- Phase3 ready: `0`

## Blockers
### PERFORMANCE_BLOCKER - FAIL
- lift_violation_rate: value=0.7727272727272727 threshold=<= 0.35 pass=0
- lift_fraction: value=0.09090909090909091 threshold=>= 0.25 pass=0
- proof_pass: value=False threshold=must_be_true pass=0
- profit_factor: value=0.5855819240449277 threshold=>= 1.3 pass=0
- win_rate: value=0.38095238095238093 threshold=>= 0.45 pass=0
- total_pnl: value=-1037.0793680975753 threshold=context metric pass=1
- closed_trades: value=42 threshold=>= 30 (runway) pass=1
- trading_days: value=12 threshold=>= 21 (runway) pass=0

### LINKAGE_BLOCKER - FAIL
- outcome_matched: value=0 threshold=>= 10 pass=0
- outcome_eligible: value=1 threshold=context metric pass=1
- matched_over_eligible: value=0.0 threshold=>= 0.80 pass=0
- linkage_waterfall:
  - raw_candidates:   45 ##############################
  - production_only:   45 ##############################
  - linked:    1 
  - hygiene_pass:    1 
  - matched:    0 

### HYGIENE_BLOCKER - FAIL
- non_trade_context_count: value=0 threshold=== 0 pass=1
- invalid_context_count: value=13 threshold=== 0 pass=0

## Reason Breakdown
- binding_match: `1`
- INVALID_CONTEXT top reasons:
  - MISSING_EXECUTION_METADATA: 13
- NON_TRADE_CONTEXT top reasons:
  - (none)
