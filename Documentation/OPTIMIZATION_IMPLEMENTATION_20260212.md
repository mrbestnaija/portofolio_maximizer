# Optimization Implementation Plan

**Date**: 2026-02-12 22:15 UTC
**Sprint Status**: Run 1/20 executing (background)
**Focus**: Critical optimizations while audit sprint accumulates

---

## Priority 3: Fix entry_trade_id Population [HIGH]

### Current Issue

New closing trades are missing `entry_trade_id` linkage, causing:
- MEDIUM severity warnings in integrity audit
- Incomplete round-trip attribution
- Manual repair required after each trading session

**Evidence**: IDs 53-55 (GOOG, JPM, GS) required manual repair after Feb 12 trading

### Root Cause Analysis

The `Portfolio` class (paper_trading_engine.py:106-116) tracks:
```python
@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, int]           # ticker -> shares
    entry_prices: Dict[str, float]      # ticker -> avg entry price
    entry_timestamps: Dict[str, datetime]  # ticker -> opened timestamp
    # ... other fields
```

**Missing**: `entry_trade_ids: Dict[str, int]` to track which trade ID opened each position

### Implementation Plan

#### Step 1: Extend Portfolio Class

**File**: `execution/paper_trading_engine.py:106-116`

**Change**:
```python
@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)
    entry_prices: Dict[str, float] = field(default_factory=dict)
    entry_timestamps: Dict[str, datetime] = field(default_factory=dict)
    entry_bar_timestamps: Dict[str, datetime] = field(default_factory=dict)
    entry_trade_ids: Dict[str, int] = field(default_factory=dict)  # NEW: ticker -> trade_id
    last_bar_timestamps: Dict[str, datetime] = field(default_factory=dict)
    holding_bars: Dict[str, int] = field(default_factory=dict)
    stop_losses: Dict[str, float] = field(default_factory=dict)
    target_prices: Dict[str, float] = field(default_factory=dict)
```

#### Step 2: Store Trade ID on Position Open

**File**: `execution/paper_trading_engine.py:_store_trade_execution` (around line 1272)

**Current flow**:
1. Trade executed and stored via `save_trade_execution()`
2. Returns without capturing trade ID
3. Portfolio updated with entry price/timestamp

**New flow**:
```python
# After line 1311 (after save_trade_execution call)
trade_id = self.db_manager.save_trade_execution(...)  # Returns trade ID

# NEW: Store entry_trade_id when opening position
if not is_close_ref and trade.action in ('BUY', 'SELL'):
    self.portfolio.entry_trade_ids[trade.ticker] = trade_id
    logger.debug("Stored entry_trade_id=%d for %s position", trade_id, trade.ticker)

# Return statement
return realized_pnl if realized_pct is not None else None, realized_pct
```

#### Step 3: Retrieve and Pass entry_trade_id on Close

**File**: `execution/paper_trading_engine.py:_store_trade_execution` (around line 1220-1265)

**Change**:
```python
# After line 1265 (before save_trade_execution call)
# NEW: Retrieve entry_trade_id when closing position
entry_trade_id_ref = None
if is_close_ref:
    entry_trade_id_ref = self.portfolio.entry_trade_ids.get(trade.ticker)
    if not entry_trade_id_ref:
        logger.warning(
            "Closing %s position but no entry_trade_id found - audit trail incomplete",
            trade.ticker
        )

# Add to save_trade_execution call (line 1272)
self.db_manager.save_trade_execution(
    # ... existing parameters ...
    is_close=is_close_ref,
    entry_trade_id=entry_trade_id_ref,  # NEW
    bar_timestamp=trade.bar_timestamp.isoformat() if isinstance(trade.bar_timestamp, datetime) else None,
    # ... remaining parameters ...
)
```

#### Step 4: Clean Up entry_trade_id on Position Exit

**File**: `execution/paper_trading_engine.py:_store_trade_execution` (after save_trade_execution)

**Change**:
```python
# NEW: Clean up entry_trade_id when position fully closed
if is_close_ref and position_after == 0:
    self.portfolio.entry_trade_ids.pop(trade.ticker, None)
    logger.debug("Removed entry_trade_id for %s (position fully closed)", trade.ticker)
```

### Testing Plan

1. **Unit Test**: Test Portfolio class with entry_trade_ids dict
2. **Integration Test**: Open position → verify entry_trade_id stored
3. **Integration Test**: Close position → verify entry_trade_id passed
4. **Integration Test**: Full cycle → verify round-trip linkage in DB
5. **Regression Test**: Run repair_unlinked_closes.py → expect 0 unlinked

### Verification

```bash
# After fix, run a trading cycle
python scripts/run_auto_trader.py --tickers AAPL --cycles 1

# Check for unlinked closes
python scripts/repair_unlinked_closes.py

# Expected output: "No unlinked closes found"
```

### Migration for Existing Positions

Existing open positions won't have `entry_trade_id` in Portfolio state. Handle gracefully:

```python
# In position close logic
entry_trade_id_ref = self.portfolio.entry_trade_ids.get(trade.ticker)
if not entry_trade_id_ref and is_close_ref:
    # Legacy position without entry_trade_id tracking
    # Try to backfill from recent BUY entries for this ticker
    recent_buy = self.db_manager.get_most_recent_buy(trade.ticker)
    if recent_buy:
        entry_trade_id_ref = recent_buy['id']
        logger.info("Backfilled entry_trade_id=%d for %s (legacy position)",
                   entry_trade_id_ref, trade.ticker)
```

---

## Priority 4: Isolate Adversarial Tests [MEDIUM]

### Current Issue

Adversarial test with `--disable-guardrails` contaminated production database with 4 attack artifacts (IDs 56-59), causing:
- 1 CRITICAL violation (opening leg with PnL)
- 4 HIGH violations (orphaned positions)
- Required manual cleanup

### Implementation Plan

#### Option A: Use Test Database Copy

**File**: `.github/workflows/ci.yml` or `bash/run_adversarial_tests.sh`

```bash
#!/bin/bash
# Adversarial integrity test with database isolation

TEST_DB="data/portfolio_maximizer_test.db"
PROD_DB="data/portfolio_maximizer.db"

echo "Creating test database copy..."
cp "$PROD_DB" "$TEST_DB"

echo "Running adversarial tests on test database..."
python scripts/adversarial_integrity_test.py \
    --db "$TEST_DB" \
    --disable-guardrails

EXIT_CODE=$?

echo "Cleaning up test database..."
rm -f "$TEST_DB"

exit $EXIT_CODE
```

#### Option B: Use In-Memory Database

**File**: `scripts/adversarial_integrity_test.py`

**Change**:
```python
DEFAULT_DB = ":memory:"  # SQLite in-memory database

# In main():
if args.db == ":memory:":
    # Copy production schema to in-memory DB
    prod_conn = sqlite3.connect(PROD_DB_PATH)
    mem_conn = sqlite3.connect(":memory:")
    prod_conn.backup(mem_conn)
    prod_conn.close()

    tester = AdversarialTest(":memory:", disable_guardrails=bool(args.disable_guardrails))
    tester.conn = mem_conn
else:
    tester = AdversarialTest(args.db, disable_guardrails=bool(args.disable_guardrails))
```

#### Option C: Transaction Rollback (Safest)

**File**: `scripts/adversarial_integrity_test.py`

**Change**:
```python
class AdversarialTest:
    def __init__(self, db_path: str, disable_guardrails: bool = False):
        self.db_path = db_path
        self.conn = guarded_sqlite_connect(db_path, enable_guardrails=not disable_guardrails)

        # NEW: Start transaction, never commit (always rollback)
        self.conn.execute("BEGIN TRANSACTION")
        self.is_test_mode = True  # All changes will be rolled back

    def __del__(self):
        if self.is_test_mode and self.conn:
            self.conn.rollback()  # Discard all test changes
            self.conn.close()
```

**Recommendation**: **Option C** (transaction rollback) - safest, no file operations, guaranteed cleanup

### Testing

```bash
# Before fix: Check production DB is clean
sqlite3 data/portfolio_maximizer.db "SELECT COUNT(*) FROM trade_executions WHERE ticker LIKE 'ATTACK%'"
# Expected: 0

# Run adversarial test
python scripts/adversarial_integrity_test.py --disable-guardrails

# After fix: Verify no contamination
sqlite3 data/portfolio_maximizer.db "SELECT COUNT(*) FROM trade_executions WHERE ticker LIKE 'ATTACK%'"
# Expected: 0 (attacks were rolled back)
```

---

## Priority 5: Monitor Ensemble Routing [ONGOING]

### Current Status

**Lift Gate**: 20% violation rate (5 audits, 1 violation)
- Threshold: 25% max
- Status: ✅ Within tolerance

**Audit Accumulation**: 5/20 effective audits
- Sprint will generate 20+ audits
- Expected completion: Tomorrow afternoon

### Monitoring Commands

```bash
# Check audit count
ls logs/forecast_audits/*.json | wc -l

# Check recent ensemble routing decisions
python -c "
import json
from pathlib import Path

audits = sorted(Path('logs/forecast_audits').glob('*.json'))[-10:]
for audit in audits:
    data = json.loads(audit.read_text())
    meta = data.get('ensemble_metadata', {})
    allowed = meta.get('allow_as_default', 'N/A')
    ratio = meta.get('rmse_ratio_vs_best', 'N/A')
    print(f'{audit.name}: allow_default={allowed}, ratio={ratio}')
"

# Check lift gate status
python scripts/check_forecast_regression.py \
    --audit-dir logs/forecast_audits \
    --config config/forecaster_monitoring.yml
```

### Expected Behavior

When violation rate > 25%:
- `allow_as_default=false` in ensemble_metadata
- `mean_forecast` should use single model (not ensemble)
- Signal confidence should match single model

### Alert Conditions

🚨 **Alert if**:
- Violation rate > 25% but `allow_as_default=true`
- Ensemble used despite preselection gate block
- Trading signals use ensemble confidence when blocked

---

## Sprint Monitoring

### Progress Tracking

```bash
# Check sprint progress
tail -100 C:\Users\Bestman\AppData\Local\Temp\claude\...\bc6f50f.output

# Check current run
ls -lt logs/audit_sprint/20260212_220853/

# Check audit accumulation
ls logs/forecast_audits/*.json | wc -l

# Check integrity status
python -c "
from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
with PnLIntegrityEnforcer('data/portfolio_maximizer.db') as e:
    violations = e.run_full_integrity_audit()
    print(f'Violations: {len(violations)}')
"
```

### Expected Milestones

| Time | Milestone | Verification |
|------|-----------|--------------|
| 22:08 | Sprint started | Run 1/20 executing |
| 23:08 | Run 2/20 | 2 audits accumulated |
| 00:08 | Run 3/20 | 3 audits accumulated |
| 06:08 | Run 10/20 | 10 audits accumulated |
| 12:08 | Run 18/20 | 18 audits accumulated |
| 18:08 | Run 20/20 complete | 20+ audits, gate ready |

### Success Criteria

✅ **Audit Sprint Success**:
- 20+ forecast audits generated
- Violation rate ≤ 25%
- 0 integrity violations
- Ensemble routing working correctly

✅ **Profitability Gate**:
- 30+ closed trades (currently 23)
- 20+ effective audits (sprint will deliver)
- Profit factor > 1.0 (currently 2.76 ✅)
- Win rate > 0% (currently 52% ✅)

---

## Implementation Timeline

### Today (Feb 12, 22:00-24:00 UTC)
- ✅ Sprint running (background)
- 🔄 Implement entry_trade_id fix (2-3 hours)
- 🔄 Implement adversarial test isolation (1 hour)

### Tomorrow (Feb 13, 00:00-18:00 UTC)
- 🔄 Sprint continues (runs 3-20)
- ✅ Entry_trade_id fix tested and deployed
- ✅ Adversarial test isolation tested
- ✅ Sprint completes (~18:00 UTC)

### Tomorrow Evening (Feb 13, 18:00+ UTC)
- ✅ Verify 20+ audits accumulated
- ✅ Run profitability gate check
- ✅ Expected: PASS (all criteria met)

---

## Risk Mitigation

### If Sprint Fails

**Backup Plan**: Natural accumulation
- Current rate: 2.5 audits/day
- Need: 15 more audits
- Timeline: 6 days (Feb 19)

### If entry_trade_id Fix Has Issues

**Fallback**: Manual repair script
- Run `repair_unlinked_closes.py` after each session
- Add to post-trading automation hook
- Permanent fix in next patch

### If Adversarial Test Isolation Fails

**Manual Cleanup**:
```sql
DELETE FROM trade_executions
WHERE ticker IN ('ATTACK1', 'ATTACK3', 'PRAGMA_ATTACK', 'UPDATE_ATTACK')
   OR run_id IS NULL;
```

---

## Next Actions

1. **Monitor sprint progress** (periodic checks every 2 hours)
2. **Implement entry_trade_id fix** (start now, complete tonight)
3. **Implement test isolation** (start tomorrow morning)
4. **Verify profitability gate** (tomorrow evening after sprint completes)

**Current Focus**: Implementing entry_trade_id population fix while sprint runs in background.
