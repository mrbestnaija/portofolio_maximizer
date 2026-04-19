# THIN_LINKAGE Coverage Plan — 2026-04-19

## Canonical Constraint

Of the 58 open lots, 40 have `legacy_` ts_signal_ids and contribute **zero** THIN_LINKAGE
credit when they close. 18 have matching production audit files (the "covered" ceiling) and
each contributes +1 to `matched` when closed. 2 lots have canonical-format tsids but their
audit files were misrouted (one to `research/`, one to `quarantine/`) and also contribute 0
until the misrouting is corrected.

Code cannot force any close. The only path to `matched >= 10` before 2026-04-24 is
stop/target price action on the 18 covered lots, plus future round-trips opened after those
positions flatten (each new auto-trader position carries a canonical tsid and a production
audit file).

## Current State (2026-04-19)

| Metric | Value |
|--------|-------|
| `matched_current` | 1 |
| `matched_threshold` | 10 |
| `matched_needed` | 9 |
| `warmup_deadline` | 2026-04-24 |
| Open lots total | 58 |
| Covered lots (THIN_LINKAGE ceiling) | 18 |
| Legacy lots (zero credit) | 38 |
| Other (misrouted audit, zero credit) | 2 |

**Covered lots by ticker:**

| Ticker | Covered lots | Notes |
|--------|-------------|-------|
| AMZN | 6 | All 6 open lots have matching production audit |
| GOOG | 5 | 10 lots are legacy |
| NVDA | 6 | 8 lots are legacy; 1 lot has misrouted audit (quarantine) |
| AAPL | 1 | 6 lots are legacy; 1 lot has misrouted audit (research/) |

**Maximum matched from existing open lots:** 18. This exceeds the threshold of 10, but
only if enough covered lots close via stop/target before the deadline.

## Open-Lot Audit Coverage (Derived Hypothesis)

The 18 covered lots figure is a **derived hypothesis** based on scanning
`logs/forecast_audits/production/*.json` for matching `signal_context.ts_signal_id`.
The gate (`production_audit_gate.py`) performs its own join and reports `matched=1,
eligible=1`. These figures diverge because:

- Gate counts **closed** trades with confirmed linkage (currently 1).
- Coverage count measures **open** lots whose future closes *could* match (ceiling = 18).

The gate is the authoritative source for confirmed matched. The coverage count is a
planning estimate for maximum achievable matched.

## Pipeline Defect: Misrouted Audit Files

Two open lots have canonical ts_signal_ids but their audit files are not in the production
audit root:

| Trade ID | Ticker | ts_signal_id | Audit location | Root cause |
|----------|--------|-------------|---------------|------------|
| 253 | AAPL | `ts_AAPL_20260305T202651Z_6077_0003` | `research/` | Opened by ETL run (not auto-trader) |
| 316 | NVDA | `ts_NVDA_20260402_818b_0003` | `quarantine/` | Moved by evidence hygiene sweep |

**Fix:** Do NOT create stub audit files to patch this — that manufactures evidence.
The correct fix is:
1. For the quarantine lot (NVDA): verify the file was moved legitimately before potentially
   restoring it. Check `quarantine/` filenames for hygiene violation reasons.
2. For the research lot (AAPL): ETL-originated live trades should write audit files to
   `production/` not `research/`. Diagnose whether `run_etl_pipeline.py` uses the correct
   `audit_log_dir` when `execution_mode=live`.

A regression test (`TestThinLinkageSection::test_research_and_quarantine_subdirs_excluded_from_coverage`
in `tests/scripts/test_emit_canonical_snapshot.py`) documents that research/ and quarantine/
subdirs are intentionally excluded from the production THIN_LINKAGE scan.

## What Code Can and Cannot Do

**Can:**
- Ensure every lot opened by the 3×/day auto-trader has a production audit file written at
  open time (verify `_attach_signal_context_to_forecast_audit` in `scripts/run_auto_trader.py`)
- Ensure forced-exit close legs default `is_synthetic=0` (fixed in commit 15d9eeb)
- Provide a live countdown in `logs/canonical_snapshot_latest.json` via the `thin_linkage`
  block (added this session)

**Cannot:**
- Force legacy lots to close before the deadline
- Force price action to hit stops/targets on covered lots
- Guarantee matched reaches 10 by 2026-04-24

## Parallel Work (Does NOT Move matched)

All of the following are useful maintenance but do not change the THIN_LINKAGE count before
the deadline. They should be treated as "work to do while waiting for price action":

- NAV rotation (AAPL/GS demotion, NVDA/MSFT/GOOG promotion) — downstream of gate
- Exit geometry improvements (R:R gate, trailing stop) — affects future trades, not current lots
- Daily shadow NAV plan cron — deepens green history, not THIN_LINKAGE
- Ticker knockout / regime / family robustness analysis — research-only, n_trades < 100

## Monitoring

After every auto-trader cycle, `emit_canonical_snapshot.py` (wired in `bash/production_cron.sh`)
re-emits `logs/canonical_snapshot_latest.json` with an up-to-date `thin_linkage` section.
The human-readable CLI output shows:

```
THIN_LINKAGE       : 1/10 matched  need 9 more  deadline=2026-04-24
  covered lots     : 18 (AAPLx1  AMZNx6  GOOGx5  NVDAx6)  legacy=38  other=2
```

`matched_current` increments when the gate artifact is refreshed by a successful close. To
force a gate re-run:

```bash
python scripts/production_audit_gate.py
python scripts/emit_canonical_snapshot.py
```

## After the Deadline

If `matched < 10` on 2026-04-24, the THIN_LINKAGE warmup expires and the gate hard-fails
with `prior_gate_execution=FAIL`. The correct response is:

1. Do NOT lower `linkage_min_matched` or re-activate warmup.
2. Continue accumulating matched closes from existing covered lots and new round-trips.
3. The gate will pass once `matched >= 10` with `ratio >= 0.8`, regardless of warmup state.
4. Unattended readiness follows when the gate passes — it is a conjunction.
