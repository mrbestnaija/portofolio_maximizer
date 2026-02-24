# Phase 7.13: Architectural Sanitization — ID Unification & Pipeline Wiring

**Status**: IN PROGRESS (2026-02-24)
**Sprint type**: 12-issue fix sprint (3 phases: A Critical, B High, C Medium)
**Audit file**: `logs/automation/arch_sanitization_audit.json`

---

## Background

A deep audit on 2026-02-24 found 12 critical architectural disconnects across
the signal generation, execution, and monitoring pipelines. The root problems:

1. **Signal IDs collide** — `TimeSeriesSignal.signal_id` was a per-instance
   counter resetting to 0 on every new generator. Nine overnight-refresh runs
   each created `signal_id=1` for a different ticker. `update_platt_outcomes.py`
   always matched 0 rows.

2. **ETL pipeline never executes trades** — `run_etl_pipeline.py` ends after
   signal routing. Overnight refresh generated JSONL but zero DB trade rows,
   so Platt scaling accumulated nothing indefinitely.

3. **`--execution-mode` CLI arg missing** — `run_auto_trader.py` read
   `EXECUTION_MODE` from env var only. Every doc/cron recommendation wrote
   `--execution-mode synthetic`, which Click silently rejected.

4. **`confidence_calibrated` missing from JSONL** — `_log_quant_validation()`
   wrote raw `confidence` but never `confidence_calibrated`, breaking the Platt
   feedback loop.

5. **Path inconsistency** — DB path and JSONL path hardcoded as CWD-relative
   strings in 4+ scripts with no central registry.

6. **Seven additional medium/low issues** documented in the audit.

---

## Fixes Applied

### Phase A: Critical (all applied 2026-02-24)

#### A1 — C3: `--execution-mode` CLI arg
- **File**: `scripts/run_auto_trader.py`
- **Change**: Added `@click.option("--execution-mode", type=click.Choice(["live","synthetic","auto"]))`
- **Wire**: CLI arg sets `os.environ["EXECUTION_MODE"]` before env read
- **Docs**: `CLAUDE.md` updated with correct synthetic sprint command
- **Commit**: `8a2aba2` "Phase 7.13-A1: Add --execution-mode CLI arg to run_auto_trader.py"

#### A2 — C1: `ts_signal_id` global uniqueness
- **Root cause**: `_signal_counter` reset to 0 per generator instance
- **Fix**: `_make_ts_signal_id()` → `ts_{ticker}_{run_suffix}_{counter:04d}`
  - `ticker_safe`: first 6 chars of uppercase ticker
  - `run_suffix`: timestamp portion of `_runtime_run_id` (14 chars)
  - `counter`: monotone within instance (4 digits, zero-padded)
- **Files changed**:
  - `models/time_series_signal_generator.py`: `signal_id` field → `Optional[str]`; `_make_ts_signal_id()`; `_current_ticker` tracking
  - `etl/database_manager.py`: `ts_signal_id TEXT` column + migration block + `save_trade_execution()` param
  - `execution/paper_trading_engine.py`: `Trade.ts_signal_id: Optional[str]`; wire from signal dict; pass to save
  - `models/signal_router.py`: `_signal_to_dict()` exposes `ts_signal_id`
  - `scripts/update_platt_outcomes.py`: query `ts_signal_id TEXT` (not `signal_id INTEGER`)
  - `scripts/migrate_add_ts_signal_id.py`: **NEW** — idempotent migration + legacy backfill
- **Migration applied**: 84 legacy rows backfilled with `legacy_{trade_date}_{id}`

#### A3 — C2: Overnight refresh executes signals
- **File**: `bash/overnight_refresh.sh`
- **Change**: Added Step 2.5 after pipeline loop:
  - Runs `run_auto_trader.py --cycles 1 --execution-mode synthetic --proof-mode --no-resume`
  - Runs `update_platt_outcomes.py` after auto_trader
  - Step 3 now also calls `production_audit_gate.py` to refresh forecast cache

### Phase B: High Priority (applied 2026-02-24)

#### B1 — H1: `confidence_calibrated` in JSONL
- **File**: `models/time_series_signal_generator.py` line ~2383
- **Change**: `'confidence_calibrated': signal.confidence_calibrated` added to JSONL entry dict
- **Impact**: Platt feedback loop now has access to calibrated probability in JSONL

#### B2 — H2: Shared signal generator factory
- **Status**: DEFERRED to Phase 7.14
- **Reason**: Requires refactoring both `run_etl_pipeline.py` and `run_auto_trader.py`;
  higher regression risk than benefit given current test coverage

#### B3 — H3: Orphan detection threshold tightened
- **File**: `bash/run_20_audit_sprint.sh`
- **Change**: Holdout `INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS` changed 60 → 14 days
- **Change**: Added `INTEGRITY_ORPHAN_WHITELIST_IDS=5,6,11,13,61,62,67` for known
  pre-persistence legacy opens

### Phase C: Medium Priority (applied 2026-02-24)

#### C1 — M1: Central path constants
- **New file**: `etl/paths.py`
  - `ROOT`, `DB_PATH`, `QUANT_VALIDATION_JSONL`, `FORECAST_AUDITS_DIR`, `FORECAST_AUDITS_CACHE`
  - `DB_PATH` respects `PORTFOLIO_DB_PATH` env var override
- **Consumers updated**: `update_platt_outcomes.py`, `check_quant_validation_health.py`,
  `ci_integrity_gate.py`, `production_audit_gate.py`
- All use `try: from etl.paths import ...` with inline fallback for robustness

#### C2 — M2: Forecast audit cache refresh wired
- **File**: `bash/overnight_refresh.sh` Step 3
- **Change**: Calls `production_audit_gate.py` (exit ignored) after health check
  so `forecast_audits_cache/latest_summary.json` is current at 7 AM cron

#### C3 — M3: Master gate orchestrator
- **New file**: `scripts/run_all_gates.py`
  - Runs: `ci_integrity_gate` → `check_quant_validation_health` → `production_audit_gate`
  - Flags: `--skip-forecast-gate`, `--skip-profitability-gate`, `--strict`, `--json`, `--db`
  - Returns JSON summary; exits non-zero if any blocking gate fails

#### C4 — M4: Signal router dead-end annotated
- **File**: `scripts/run_etl_pipeline.py` after signal routing stage
- **Change**: Added explicit `# NOTE (Phase 7.13-C4)` comment block explaining
  that signal routing is for JSONL logging only; no PaperTradingEngine call here

#### C5 — M5: Legacy ts_signal_id backfill
- **Script**: `scripts/migrate_add_ts_signal_id.py`
- **Applied**: 84 legacy rows backfilled with `legacy_{trade_date}_{id}`
- **Idempotent**: Second run shows 0 NULL rows

#### C6 — L1: DB path centralization
- **Handled by C1** (`etl/paths.py`). Scripts that previously used `"data/portfolio_maximizer.db"`
  now use `_DEFAULT_DB_PATH` from `etl.paths`.

---

## ID Unification Design

| ID Type | Old Format | New Format | Scope |
|---------|-----------|-----------|-------|
| `ts_signal_id` | `int` counter 1,2,3... (resets) | `ts_{ticker}_{run_suffix}_{counter:04d}` | Globally unique per TS signal |
| `run_id` | `pmx_ts_YYYYMMDDTHHMMSSZ_PID` | Unchanged | Per generator instance |
| `signal_id` (LLM) | `llm_signals.id` auto-increment | Unchanged | Per LLM signal (separate namespace) |

**Key design principle**: `trade_executions.signal_id INTEGER FK llm_signals` stays unchanged
for LLM-originated trades. New `trade_executions.ts_signal_id TEXT` column added for TS signals.
This avoids breaking the existing FK schema while enabling clean attribution.

---

## Verification Commands

```bash
# 1. Verify ts_signal_id uniqueness in JSONL
python -c "
import json, pathlib, collections
path = pathlib.Path('logs/signals/quant_validation.jsonl')
if path.exists():
    entries = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    ids = [e.get('signal_id') for e in entries if e.get('signal_id')]
    dupes = [k for k,v in collections.Counter(ids).items() if v > 1]
    print(f'Total entries: {len(entries)}, IDs: {len(ids)}, Duplicates: {len(dupes)}')
else:
    print('JSONL not found - run overnight_refresh.sh first')
"

# 2. Verify --execution-mode CLI arg
python scripts/run_auto_trader.py --help | grep execution-mode

# 3. Run all gates
python scripts/run_all_gates.py --json

# 4. Verify migration idempotency
python scripts/migrate_add_ts_signal_id.py  # run twice, expect 0 NULL rows

# 5. Verify Platt pairs accumulate after synthetic run
python scripts/run_auto_trader.py --tickers AAPL --cycles 1 --execution-mode synthetic --proof-mode --no-resume
python scripts/update_platt_outcomes.py
```

---

## Files Modified

| File | Change | Phase |
|------|--------|-------|
| `scripts/run_auto_trader.py` | `--execution-mode` CLI arg | A1 |
| `CLAUDE.md` | Updated synthetic sprint command | A1 |
| `models/time_series_signal_generator.py` | `_make_ts_signal_id()`, `signal_id→str`, `confidence_calibrated` JSONL | A2, B1 |
| `etl/database_manager.py` | `ts_signal_id TEXT` column + migration + `save_trade_execution()` param | A2 |
| `execution/paper_trading_engine.py` | `Trade.ts_signal_id`, signal wire, save wire | A2 |
| `models/signal_router.py` | `_signal_to_dict` exposes `ts_signal_id` | A2 |
| `scripts/update_platt_outcomes.py` | Query `ts_signal_id TEXT`; use `etl.paths` | A2, C1 |
| `scripts/migrate_add_ts_signal_id.py` | **NEW** migration + backfill | A2, C5 |
| `bash/overnight_refresh.sh` | Step 2.5 auto_trader; C2 gate refresh | A3, C2 |
| `bash/run_20_audit_sprint.sh` | Orphan threshold 60→14; whitelist IDs | B3 |
| `etl/paths.py` | **NEW** central path constants | C1 |
| `scripts/check_quant_validation_health.py` | Use `etl.paths` | C1 |
| `scripts/ci_integrity_gate.py` | Use `etl.paths` | C1 |
| `scripts/production_audit_gate.py` | Use `etl.paths` | C1 |
| `scripts/run_all_gates.py` | **NEW** gate orchestrator | C3 |
| `scripts/run_etl_pipeline.py` | Dead-end annotation comment | C4 |
| `logs/automation/arch_sanitization_audit.json` | **NEW** machine-readable audit | all |
| `Documentation/PHASE_7.13_ARCH_SANITIZATION.md` | **NEW** this file | all |

---

**Last Updated**: 2026-02-24
**Phase**: 7.13 (Architectural Sanitization)
