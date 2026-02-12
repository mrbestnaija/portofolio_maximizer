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

