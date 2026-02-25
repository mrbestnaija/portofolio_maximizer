# Platt Starvation Diagnostic (2026-02-25)

## Scope

Independent adversarial investigation of why Platt calibration remains starved (`matched=0`) after overnight bootstrap.

## Observed Failure

- `python scripts/update_platt_outcomes.py --dry-run` reported:
  - `total=190 pending=190 matched=0 already_done=0 still_pending=190`
- `logs/signals/quant_validation.jsonl` had 190 entries, 0 with `outcome`.

## Hard Evidence

1. `signal_id` namespace mismatch at reconciliation time

- JSONL signal IDs are `ts_*` (165 distinct IDs in 190 rows).
- Closed trade IDs in DB are `legacy_*` only (39 distinct closed IDs).
- Distinct overlap between JSONL `signal_id` and closed DB `ts_signal_id`: `0`.

2. Bootstrap runs did not produce closes

- `trade_executions` showed 9 new `ts_*` rows from bootstrap.
- All 9 were opening legs only (`action=BUY`, `is_close=0`, `realized_pnl=NULL`).

3. Why `--cycles 8` did not help

- Execution log (`logs/automation/execution_log.jsonl`) for bootstrap run IDs
  (`20260225_204514`, `...204543`, `...204557`, `...204611`, `...204626`) showed:
  - cycle 1: few `EXECUTED` BUYs
  - cycles 2-8: `SKIPPED_SAME_BAR` for all tickers
- Per run: 72 events = 9 tickers x 8 cycles
  - typically: `2 EXECUTED BUY`, `7 REJECTED`, `63 SKIPPED_SAME_BAR`

## Root Cause

Bootstrap design used:

- fixed `--as-of-date`
- `--cycles 8`
- default `--bar-aware`
- `--no-resume`

This combination prevents position aging and close generation:

- Fixed as-of date means same last bar each cycle.
- Bar-aware gate skips cycles 2-8 as duplicate bars.
- `--no-resume` resets state each historical window, so open positions never carry into later dates.
- Result: no `is_close=1` rows in the `ts_*` namespace, so reconciliation remains `matched=0`.

## Fix Applied

Updated both launchers:

- `scripts/run_overnight_refresh.py`
- `bash/overnight_refresh.sh`

New bootstrap behavior:

- run **1 cycle per historical AS_OF date**
- first window uses `--no-resume`
- subsequent windows use `--resume`
- keep `--proof-mode`

This allows bar timestamps to progress across dates and enables lifecycle exits to produce closed `ts_*` trades that `update_platt_outcomes.py` can match.

## Verification Steps

1. Run bootstrap:
   - `python scripts/run_overnight_refresh.py --platt-bootstrap`
2. Confirm closes in `ts_*` namespace:
   - query `trade_executions` where `ts_signal_id LIKE 'ts_%' AND realized_pnl IS NOT NULL AND is_close=1`
3. Reconcile:
   - `python scripts/update_platt_outcomes.py --dry-run`
4. Confirm `matched > 0` and JSONL contains `outcome` fields.
