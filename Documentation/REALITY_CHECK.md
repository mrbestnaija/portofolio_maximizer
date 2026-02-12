# PnL Integrity Framework: Reality Check
## What Was Actually Implemented vs What Was Specified

**Date**: 2026-02-11
**Auditor**: Second-pass verification
**Verdict**: PARTIAL IMPLEMENTATION - Application-level only, NO database constraints

---

## CRITICAL DISCREPANCY

**ENFORCEMENT_MAPPING.md says**:
> "The enforcer sits between the trading engine and the database as a **non-bypassable gateway**. The key design principle: constraints are enforced at the **SQLite CHECK constraint level**, not in application code."

**REALITY**:
- NO CHECK constraints implemented beyond pre-existing `CHECK(action IN ('BUY', 'SELL'))`
- NO non-bypassable gateway
- Enforcement is ONLY in one application code path (`paper_trading_engine.py`)
- Direct SQL CAN bypass all enforcement

---

## Item-by-Item Comparison

### ✗ ISSUE #1: PnL Double-Counting

**Specified**:
```sql
CHECK (CASE WHEN is_close = 0
       THEN realized_pnl IS NULL AND realized_pnl_pct IS NULL
       ELSE 1 END)
```

**Actually Implemented**:
- Application-level guard at `paper_trading_engine.py:1257-1264`
- Cleanup script that fixed 24 existing violations
- NO database CHECK constraint

**Gap**: Can be bypassed via direct SQL INSERT.

**Verified Result**: 0 opening legs currently have realized_pnl (cleanup worked), but NEW violations are possible.

---

### ✗ ISSUE #2: Exit Linkage Constraint

**Specified**:
```sql
CHECK (CASE WHEN is_close = 1
       THEN entry_trade_id IS NOT NULL
       ELSE 1 END)
```

**Actually Implemented**:
- Backfill script that linked 16 of 20 closes
- NO database CHECK constraint
- 4 closing legs still lack entry_trade_id

**Gap**: Can save closing trades without entry_trade_id.

**Verified Result**: 4 exits have NULL entry_trade_id (linkage incomplete).

---

### ✗ ISSUE #3: Diagnostic Mode Contamination

**Specified**:
```sql
CHECK (CASE WHEN is_diagnostic = 1
       THEN execution_mode != 'live'
       ELSE 1 END)
```

**Actually Implemented**:
- Automatic tagging: `is_diagnostic=1` if DIAGNOSTIC_MODE env var set
- VIEW filters: `production_closed_trades` excludes is_diagnostic=1
- NO database CHECK constraint

**Gap**: Could manually INSERT is_diagnostic=1 with execution_mode='live'.

**Verified Result**: Views correctly exclude diagnostic trades, but constraint missing.

---

### ✗ ISSUE #4: Synthetic Data Contamination

**Specified**:
```sql
CHECK (CASE WHEN is_synthetic = 1
       THEN execution_mode != 'live'
       ELSE 1 END)
```

**Actually Implemented**:
- Automatic tagging: `is_synthetic=1` if data_source contains "synthetic"
- VIEW filters: `production_closed_trades` excludes is_synthetic=1
- NO database CHECK constraint

**Gap**: Could manually INSERT is_synthetic=1 with execution_mode='live'.

**Verified Result**: Views correctly exclude synthetic trades, but constraint missing.

---

### ✗ ISSUE #5: Bar-Close Fill Price Audit

**Specified**:
- OHLC columns: `bar_open`, `bar_high`, `bar_low`, `bar_close`
- `_enforce_fill_price_audit()` detects exact-close fills

**Actually Implemented**:
- Columns added to schema: `bar_open`, `bar_high`, `bar_low`, `bar_close`
- NO code to populate these columns
- NO fill price audit function

**Gap**: Columns exist but are never populated. No audit trail.

**Verified Result**: Columns NULL for all trades.

---

### ✗ ISSUE #6: Confidence Calibration

**Specified**:
- `confidence_calibrated` column (0 or 1)
- `_check_confidence_calibration()` gate requiring ≥50 historical trades
- Kelly sizing conditional on calibration

**Actually Implemented**:
- Column added: `confidence_calibrated` (REAL)
- NO calibration logic
- NO Kelly gate

**Gap**: Column exists but never used.

**Verified Result**: Column NULL for all trades.

---

### ✗ ISSUE #7: Forecast Audit Progression

**Specified**:
- `validate_forecast_audit_progression()` checks monotonic increase
- Reset detection
- Health flag in integrity audit

**Actually Implemented**:
- NO forecast audit validation
- NO reset detection

**Gap**: Forecast audit issue not addressed.

---

### ✗ ISSUE #8: Portfolio State Reconstruction

**Specified**:
- `validate_portfolio_state_integrity()` computes positions from trade history
- Orphan detection vs external portfolio_state table

**Actually Implemented**:
- NO portfolio state validation
- Orphan detection exists in audit (identifies 8 orphans)
- NO reconstruction logic

**Gap**: Can detect orphans but cannot reconstruct positions.

---

### ✗ ISSUE #9: Artificial Leg Detection

**Specified**:
- `detect_artificial_legs()` queries for same-ticker, same-bar EXIT+ENTRY pairs
- Included in integrity audit

**Actually Implemented**:
- NO artificial leg detection
- NO same-bar pair queries

**Gap**: Reversal artifacts not detected.

---

## What WAS Successfully Implemented

### ✓ Database Schema Extensions
- 8 new columns added to trade_executions
- Migration script creates columns on existing databases
- Views created: `production_closed_trades`, `round_trips`

### ✓ Application-Level Enforcement
- PaperTradingEngine enforcement guard prevents opening legs from getting PnL
- Automatic tagging of is_diagnostic and is_synthetic
- Trades created via normal flow comply with invariants

### ✓ Audit & Repair Tooling
- `PnLIntegrityEnforcer` class with 6 integrity checks
- `get_canonical_metrics()` provides single source of truth
- `fix_opening_legs_pnl()` cleaned up 24 violations
- `backfill_entry_trade_ids()` linked 16 of 20 closes
- `print_report()` generates comprehensive reports

### ✓ CI Gate
- `scripts/ci_integrity_gate.py` exits non-zero on violations
- Identifies: 8 orphaned positions (HIGH), 4 unlinked closes (MEDIUM)

### ✓ Corrected Metrics
- Double-counting eliminated (0 opening legs with PnL)
- Canonical metrics verified: 20 round-trips, $909.18 PnL, 60% WR
- Views correctly filter diagnostic/synthetic trades

---

## Conservative Claims (What to Actually Say)

### ✓ ACCURATE Claims:
1. "Application-level PnL integrity enforcement in PaperTradingEngine"
2. "Audit tooling to detect and repair integrity violations"
3. "Canonical metrics via production_closed_trades view (20 round-trips, $909.18 PnL, 60% WR)"
4. "Double-counting eliminated in current data (0 opening legs with PnL)"
5. "CI gate detects violations before deployment"
6. "Views automatically exclude diagnostic and synthetic trades from production metrics"

### ✗ INACCURATE Claims to RETRACT:
1. ~~"Structural prevention at database level"~~ → Only application-level
2. ~~"Non-bypassable enforcement"~~ → Can be bypassed via direct SQL
3. ~~"CHECK constraints prevent violations"~~ → No CHECK constraints implemented
4. ~~"Opening legs CANNOT have realized_pnl"~~ → No constraint, just cleanup + guard
5. ~~"Closing legs MUST have entry_trade_id"~~ → 4 violations exist, no constraint
6. ~~"Bar-close fill price audit trail"~~ → Columns exist but not populated
7. ~~"Confidence calibration validation"~~ → Column exists but not used

---

## Gap Analysis

| Component | Specified | Implemented | Status |
|-----------|-----------|-------------|--------|
| CHECK constraints | 4 constraints | 0 constraints | **NOT IMPLEMENTED** |
| OHLC fill audit | Columns + logic | Columns only | **PARTIAL** |
| Confidence calibration | Gate + Kelly sizing | Column only | **PARTIAL** |
| Forecast audit tracking | Validation logic | Not implemented | **NOT IMPLEMENTED** |
| Portfolio reconstruction | Validation from ledger | Not implemented | **NOT IMPLEMENTED** |
| Artificial leg detection | Same-bar queries | Not implemented | **NOT IMPLEMENTED** |
| Entry linkage | Backfill + constraint | Backfill only | **PARTIAL** |
| Diagnostic exclusion | Tagging + view + constraint | Tagging + view | **PARTIAL** |
| Canonical metrics | Single source | Implemented | **COMPLETE** |
| CI gate | Blocking violations | Implemented | **COMPLETE** |

**Implementation Completeness**: ~40% of specified features

---

## Risk Assessment

### LOW RISK (Acceptable for Current Use):
- Application-level enforcement works for normal trading flow
- Views correctly compute canonical metrics
- CI gate catches violations before production
- Current data state is clean (0 double-counting violations)

### MEDIUM RISK (Requires Monitoring):
- Direct SQL access could bypass enforcement
- New code paths could violate invariants without CI catching it
- Orphaned positions (8) cannot be closed via normal flow
- Incomplete audit trail (4 unlinked closes, no OHLC data)

### HIGH RISK (Requires Fixes if Mission-Critical):
- No database-level guarantees of PnL correctness
- Confidence uncalibrated → Kelly sizing would be dangerous
- Forecast audit issue unresolved → validation gate perpetually unsatisfied
- Bar-close fill price advantage not audited

---

## Recommendations

### For Current Production Use (Accept):
1. Continue with application-level enforcement
2. Monitor CI gate for violations
3. Use `get_canonical_metrics()` as single source of truth
4. Document that enforcement is NOT database-level

### For Mission-Critical Deployment (Fix):
1. **Implement CHECK constraints** as specified in ENFORCEMENT_MAPPING.md
2. **Populate OHLC columns** in PaperTradingEngine
3. **Implement confidence calibration** logic + Kelly gate
4. **Add forecast audit progression** validation
5. **Build portfolio reconstruction** from trade ledger

### Immediate Actions:
1. ✓ **Update CLAUDE.md** to reflect application-level (not structural) enforcement
2. ✓ **Create this REALITY_CHECK.md** document
3. **Retract overstated claims** in commit messages and documentation
4. **Communicate limitations** to stakeholders
5. **Decide**: Accept current state OR implement full specification

---

## Conclusion

**What was delivered**: A **robust application-level enforcement framework** with **audit/repair tooling**, **canonical metric views**, and **CI gates**. Double-counting has been eliminated from current data. Metrics are verified accurate.

**What was NOT delivered**: **Database-level CHECK constraints** and **non-bypassable enforcement gateways** as specified in ENFORCEMENT_MAPPING.md.

**Verdict**: **Partial implementation suitable for continued research/development use**. For production deployment at scale, implement the missing CHECK constraints and validation logic.

**Metrics Accuracy**: **VERIFIED** - 20 round-trips, $909.18 PnL, 60% win rate are accurate and reproducible.

**Fail Conservatively**: Claims have been adjusted to match actual implementation. Under-promise, over-deliver going forward.
