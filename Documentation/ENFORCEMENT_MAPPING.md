# PnL Integrity Enforcement Framework
## Forensic Audit → Structural Prevention Mapping

**Date**: 2026-02-11
**Status**: Ready for integration
**Corrected Baseline**: 20 round-trips, $909.18 PnL, 60% win rate

---

## Architecture: Why This Works

The enforcer sits between the trading engine and the database as a **non-bypassable gateway**. The key design principle: constraints are enforced at the **SQLite CHECK constraint level**, not in application code. You cannot insert a row that violates the rules even if you bypass the Python layer and write SQL directly.

```
Trading Engine → PnLIntegrityEnforcer.validate_and_store() → SQLite (CHECK constraints) → DB
                 ↑ blocks invalid trades                      ↑ blocks invalid SQL
                 
Metrics Query  → PnLIntegrityEnforcer.get_canonical_metrics() → production_closed_trades VIEW
                 ↑ single source of truth                       ↑ auto-excludes diagnostic/synthetic
```

---

## Issue-by-Issue Enforcement

### CRITICAL #1: PnL Double-Counted on BUY Rows

**Finding**: 16 BUY rows carry the same `realized_pnl` as their matching SELL rows. Total PnL inflated by ~$436.

**Structural Prevention**:
- **DB constraint**: `CHECK (CASE WHEN is_close = 0 THEN realized_pnl IS NULL AND realized_pnl_pct IS NULL ELSE 1 END)`
- **Application layer**: `_enforce_pnl_rules()` raises `IntegrityViolation` if any entry row has non-NULL PnL
- **Canonical view**: `production_closed_trades` only includes `is_close = 1` rows

**Result**: Physically impossible to insert an entry row with realized_pnl. The database itself rejects it.

---

### CRITICAL #2: 44 "Closed Trades" = 20 Round-Trips + 24 Legs

**Finding**: Trade count inflated ~2x by counting both legs of round-trips.

**Structural Prevention**:
- **`production_closed_trades` view**: `WHERE is_close = 1` — only exit rows are counted
- **`round_trips` view**: Joins exits to entries via `entry_trade_id` for auditable pair tracking
- **`get_canonical_metrics()`**: Queries ONLY from `production_closed_trades`
- **Exit linkage constraint**: `CHECK (CASE WHEN is_close = 1 THEN entry_trade_id IS NOT NULL ELSE 1 END)`

**Result**: Every metric query counts exits only. Round-trip view provides audit trail of which entry each exit matches.

---

### CRITICAL #3: DIAGNOSTIC_MODE=1 Bypassed All Validation

**Finding**: Confidence floor 0.10, no return/risk gating. Every signal bypassed validation.

**Structural Prevention**:
- **Runtime detection**: `_block_diagnostic_mode()` checks `DIAGNOSTIC_MODE` and `PMX_DIAGNOSTIC_MODE` env vars
- **Automatic tagging**: All trades during diagnostic mode get `is_diagnostic = 1`
- **DB constraint**: `CHECK (CASE WHEN is_diagnostic = 1 THEN execution_mode != 'live' ELSE 1 END)` — diagnostic trades cannot masquerade as live
- **View exclusion**: `production_closed_trades` has `AND is_diagnostic = 0`

**Result**: Diagnostic mode still works for debugging, but its trades are permanently excluded from production metrics. You cannot accidentally report diagnostic results as real performance.

---

### CRITICAL #4: Confidence is Uncalibrated

**Finding**: Composite heuristic (model agreement + return magnitude) never validated against realized hit rate. 0.75 confidence ≠ 75% win probability.

**Structural Prevention**:
- **`confidence_calibrated` column**: Every trade is tagged 0 (uncalibrated) or 1 (calibrated)
- **Kelly sizing gate**: `_check_confidence_calibration()` requires ≥50 historical trades at similar confidence bands with empirical win rate matching within 10pp
- **Fallback**: Uncalibrated confidence → fixed 1% risk per trade (no Kelly)
- **Metrics flag**: `get_canonical_metrics()` returns `statistical_validity.sufficient_for_significance`

**Result**: Kelly criterion cannot be applied until confidence is empirically validated. Position sizing defaults to conservative fixed risk.

---

### CRITICAL #5: Bar-Close Used as Fill Price

**Finding**: `market_data["Close"].iloc[-1]` at paper_trading_engine.py:266 provides artificial timing advantage.

**Structural Prevention**:
- **OHLC audit trail**: `bar_open`, `bar_high`, `bar_low`, `bar_close` stored alongside `fill_price`
- **Slippage documentation**: `slippage_bps` required. If fill_price equals bar_close exactly, logged as `slippage_bps=0.0` with warning
- **`_enforce_fill_price_audit()`**: Detects and flags exact-close fills
- **Patch**: Fill price computation applies explicit slippage: `price * (1 + slippage_pct)` for buys

**Result**: Every trade has a full audit trail of bar context vs fill price. Exact-close fills are flagged for review.

---

### CRITICAL #6: Forecast Audits Never Accumulate

**Finding**: `effective_audits=1, required_audits=20` in every run. Fixed as-of-date doesn't generate new outcomes.

**Structural Prevention**:
- **`validate_forecast_audit_progression()`**: Queries forecast_audits table and checks monotonic increase
- **Reset detection**: Counts how many times effective_audits decreases between consecutive runs
- **Health flag**: Returns `is_healthy: false` if resets detected
- **Full audit report**: Included in `run_full_integrity_audit()` output

**Result**: Non-accumulating audits are detected and flagged in every integrity report. The holdout validation gate cannot be "satisfied" by resetting to 1 each run.

---

### HIGH #7: Portfolio State Last-Pass-Only

**Finding**: `portfolio_state` table overwrites on each save. 24 orphaned positions proved this.

**Structural Prevention**:
- **Append-only trade ledger**: `trade_executions_enforced` is INSERT-only. No updates, no deletes in production.
- **Position reconstruction**: `validate_portfolio_state_integrity()` computes implied positions from trade history: `SUM(CASE WHEN action='BUY' THEN shares ELSE -shares END)`
- **Orphan detection**: Compares implied positions to any external portfolio_state table

**Result**: Portfolio state is always reconstructable from the trade ledger. Overwrites cannot lose position data.

---

### HIGH #8: Flatten-Before-Reverse Creates Artificial Legs

**Finding**: One reversal = 2 trade rows with separate PnL.

**Structural Prevention**:
- **Atomic reversal**: `handle_reversal()` creates one EXIT + one ENTRY sharing the same `bar_timestamp`
- **Artificial leg detection**: `detect_artificial_legs()` queries for same-ticker, same-bar pairs where one is EXIT and one is ENTRY
- **Audit report**: Included in `run_full_integrity_audit()`

**Result**: Reversals are still two rows (EXIT + ENTRY) but linked by bar_timestamp. The EXIT carries PnL; the ENTRY does not. Detection query flags any existing problematic pairs.

---

### HIGH #9: Synthetic Data Injectable via Env Var

**Finding**: `SYNTHETIC_CONFIG_PATH` env var in synthetic_extractor.py with no gate preventing synthetic contamination in production.

**Structural Prevention**:
- **`is_synthetic` column**: Every trade tagged 0 or 1
- **DB constraint**: `CHECK (CASE WHEN is_synthetic = 1 THEN execution_mode != 'live' ELSE 1 END)` — synthetic trades cannot be tagged as live
- **View exclusion**: `production_closed_trades` has `AND is_synthetic = 0`
- **Application check**: `_enforce_provenance()` raises `IntegrityViolation` if `is_synthetic=1` with `execution_mode='live'`

**Result**: Synthetic data can exist in the database for testing purposes but is physically excluded from production metrics at the DB constraint level.

---

## Deployment Sequence

### Step 1: Create enforced schema (non-destructive)
```bash
python -c "
from integrity.pnl_integrity_enforcer import PnLIntegrityEnforcer
enforcer = PnLIntegrityEnforcer('data/portfolio_maximizer.db')
enforcer.initialize()
print('Schema created')
"
```

### Step 2: Migrate legacy trades with forensic corrections
```bash
python -m integrity.pnl_integrity_enforcer \
  --migrate-from data/portfolio_maximizer.db \
  --db data/portfolio_maximizer_enforced.db
```

### Step 3: Run full audit on migrated data
```bash
python -m integrity.pnl_integrity_enforcer \
  --db data/portfolio_maximizer_enforced.db \
  --output reports/integrity_audit.json
```

### Step 4: Wire into paper_trading_engine.py
Apply patches from `integration_patches.py` — 6 targeted changes.

### Step 5: Add CI gate
Add integrity audit to `.github/workflows/ci.yml` or `bash/run_end_to_end.sh`.

### Step 6: Update dashboard
Source all metrics from `enforcer.get_canonical_metrics()`.

---

## Post-Deployment Monitoring

After integration, every `run_auto_trader.py` session ends with:

```json
{
  "canonical_metrics": {
    "closed_trades": 20,
    "total_pnl": 909.18,
    "win_rate": 0.60,
    "contamination_audit": {
      "diagnostic_trades_excluded": 0,
      "synthetic_trades_excluded": 0,
      "production_trades_counted": 20
    },
    "statistical_validity": {
      "n_trades": 20,
      "sufficient_for_significance": false,
      "confidence_level": "INSUFFICIENT"
    }
  },
  "integrity_checks": {
    "double_counted_entries": {"count": 0, "status": "PASS"},
    "orphan_exits": {"count": 0, "status": "PASS"},
    "diagnostic_contamination": {"count": 0, "status": "PASS"},
    "artificial_legs": {"count": 0, "status": "PASS"}
  },
  "overall_status": "HEALTHY"
}
```

If ANY check returns `CRITICAL_FAIL`, the CI pipeline blocks deployment and the dashboard shows a red alert.

---

## 2026-02-12 Hardening Delta (Adversarial Pass 2)

### CRITICAL #10: Ensemble Selected as Default Despite RMSE Regression

**Finding**: Prior logic could mark ensemble `DISABLE_DEFAULT` after evaluation, but forecast-time selection still defaulted to ensemble output.

**Structural Prevention**:
- **Preselection gate in forecaster**: `strict_preselection_max_rmse_ratio = 1.0` enforced before default selection.
- **Forecast-time behavior**: when recent audit RMSE ratio exceeds `1.0`, ensemble remains available for diagnostics but is not used as `mean_forecast`.
- **Fallback routing**: default source switches to a single-model forecast payload (preferred model, then SAMOSSA/SARIMAX/GARCH/MSSA-RL fallback order).
- **Metadata audit trail**: `ensemble_metadata.preselection_gate` and `ensemble_metadata.allow_as_default` capture gate evidence and decision reason.

**Result**: Ensemble cannot remain the default forecast source when recent observed RMSE evidence is worse than the best single model.

### CI Blocking Benchmark: Adversarial Forecaster Suite

**New CI gate**:
- Script: `scripts/run_adversarial_forecaster_suite.py`
- Workflow integration: `.github/workflows/ci.yml` (blocking step)
- Threshold source: `config/forecaster_monitoring_ci.yml` under:
  - `regression_metrics.adversarial_suite.max_ensemble_under_best_rate`
  - `regression_metrics.adversarial_suite.max_avg_ensemble_ratio_vs_best`
  - `regression_metrics.adversarial_suite.max_ensemble_worse_than_rw_rate`

**Default matrix (same benchmark family used in hardening validation)**:
- 3 variants x 6 scenarios x 3 seeds = 54 runs
- Deterministic synthetic regimes to expose overfitting and model-selection brittleness

**Blocking rule**:
- Any threshold breach in any variant returns non-zero and fails CI.
