# Production Security and Profitability Runbook

Last updated: 2026-02-12

## 1) Strict CVE Enforcement (Default)

The Windows production entrypoints now default to strict security gates:

- `run_daily_trader.bat`
- `schedule_backfill.bat`
- `setup_environment.bat`
- `scripts/docker_build_test.bat`

Default security posture:

- `ENABLE_SECURITY_CHECKS=1`
- `SECURITY_STRICT=1`
- `SECURITY_REQUIRE_PIP_AUDIT=1`
- `SECURITY_HARD_FAIL=1`
- `SECURITY_IGNORE_VULN_IDS=` (empty by default)

Security preflight executes:

1. `pip check`
2. `pip-audit` (required)

Runs hard-fail on unresolved vulnerabilities or dependency breakage.

## 2) Temporary Run-Through Override

For temporary continuity (for example, maintenance windows), allow workflows to continue even if security preflight fails:

```bat
set SECURITY_HARD_FAIL=0
run_daily_trader.bat
```

Use only for short-lived mitigation windows. Restore strict mode immediately after.

## 3) CVE Allowlist Policy

Allowlist support remains available for emergency exceptions via:

- `SECURITY_IGNORE_VULN_IDS` (comma-separated IDs)

Default is no allowlist. Any allowlist entry must include:

1. documented justification,
2. expiration/review date,
3. tracked removal task.

## 4) OpenBB Constraint Removal

To remove the prior `python-multipart` constraint and eliminate default CVE allowlisting:

- OpenBB packages were removed from secure default runtime profile.
- `requirements.txt` no longer installs `openbb`.
- `config/data_sources_config.yml` disables `openbb` provider by default.
- live provider priority is moved to `yfinance`/`alpha_vantage`/`finnhub`/`ctrader`.

Security-related dependency pins in `requirements.txt`:

- `pillow==12.1.1`
- `python-multipart==0.0.22`

## 5) Profitability Gate Clearance

Production profitability gate (`scripts/production_audit_gate.py`) requires both lift and proof validity.

Current proof thresholds (from live gate outputs):

- closed trades: minimum `30`
- trading days: minimum `21`
- forecast lift guardrails must pass configured violation thresholds

To clear profitability gate in production:

1. Accumulate additional closed round-trip trades until threshold is met.
2. Run enough sessions/days to meet the trading-day threshold.
3. Resolve DB integrity violations in execution flow (e.g. orphaned positions / close-entry linkage issues) before judging profitability.
4. Re-run:
   - `run_daily_trader.bat` (with production gate enabled), or
   - `schedule_backfill.bat production_gate`
5. Confirm artifact status:
   - `logs/audit_gate/production_gate_*.json`
   - `production_profitability_gate.status == "PASS"`

## 6) Evidence Artifacts

Every run produces auditable artifacts:

- Run audit JSONL: `logs/run_audit/*.jsonl`
- Security report: `logs/security/security_preflight_*.json`
- Dashboard status: `logs/audit_gate/dashboard_status_*.json`
- Production gate artifact: `logs/audit_gate/production_gate_*.json` (or scheduled-task equivalent)

### 6.1) Worktree Provenance (repo_state)

`scripts/production_audit_gate.py` records a `repo_state` block inside the production gate JSON artifact.
This captures the repo/worktree state at the time the gate ran (branch, ahead/behind, modified + untracked path list,
and per-path timestamps). This enables reproducible, time-bounded claims like:

- "At T0 we had X modified + Y untracked files; at T1 we had X' and Y'; these exact paths were added/removed/changed."

It does not attempt to attribute changes to a specific person/agent; it only snapshots state.

Compare two timestamped gate artifacts:

```powershell
.\simpleTrader_env\Scripts\python.exe scripts\compare_gate_artifacts.py `
  --a D:\pmx\logs\audit_gate\production_gate_latest_YYYYMMDD_HHMMSS.json `
  --b D:\pmx\logs\audit_gate\production_gate_latest_YYYYMMDD_HHMMSS.json
```

JSON output (for tooling/CI):

```powershell
.\simpleTrader_env\Scripts\python.exe scripts\compare_gate_artifacts.py --json --a <A.json> --b <B.json>
```

## 7) SQLite Runtime Guardrails (PRAGMA/DDL Hardening)

Operational Python DB connections now apply SQLite guardrails by default:

- defensive DB config (`SQLITE_DBCONFIG_DEFENSIVE=1`)
- trusted schema disabled (`SQLITE_DBCONFIG_TRUSTED_SCHEMA=0`)
- extension loading disabled
- authorizer sandbox blocks:
  - dangerous PRAGMAs (including `ignore_check_constraints`)
  - `ATTACH` / `DETACH`
  - runtime schema mutation (`CREATE` / `ALTER` / `DROP`) on hardened connections

Environment controls:

- `SECURITY_SQLITE_GUARDRAILS=1` (default; set `0` only for controlled maintenance)
- `SECURITY_SQLITE_GUARDRAILS_HARD_FAIL=1` (default; if guardrails fail to install, abort)

These guardrails are applied to operational paths, including:

- `etl/database_manager.py`
- `integrity/pnl_integrity_enforcer.py`
- `scripts/dashboard_db_bridge.py`
- `scripts/validate_profitability_proof.py`

## 8) Optional Gate-Lift Replay (Default Off)

`run_daily_trader.bat` now supports an optional historical replay stage to
accumulate holdout-style evidence with real market data (`--as-of-date`) while
keeping default live behavior unchanged.

Defaults:

- `ENABLE_GATE_LIFT_REPLAY=0`
- `GATE_LIFT_REPLAY_DAYS=0`
- `GATE_LIFT_REPLAY_START_OFFSET_DAYS=1`
- `GATE_LIFT_REPLAY_INTERVAL=1d`
- `GATE_LIFT_REPLAY_STRICT=0`

Enable example:

```bat
set ENABLE_GATE_LIFT_REPLAY=1
set GATE_LIFT_REPLAY_DAYS=5
set GATE_LIFT_REPLAY_START_OFFSET_DAYS=1
set GATE_LIFT_REPLAY_INTERVAL=1d
run_daily_trader.bat
```

Replay artifact:

- `logs/audit_gate/gate_lift_replay_<RUN_ID>.json`

Related helper:

- `scripts/run_gate_lift_replay.py`

## 9) Orphan Position Gate Policy Knobs

The orphan-position integrity gate now reconciles BUY/SELL inventory first and
supports explicit policy controls:

- `INTEGRITY_MAX_OPEN_POSITION_AGE_DAYS` (default `3`)
- `INTEGRITY_ORPHAN_WHITELIST_IDS` (comma-separated trade IDs)

Use these only with documented justification and audit trail.
