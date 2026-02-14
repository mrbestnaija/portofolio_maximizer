---
name: portfolio-maximizer
description: Operate this Portfolio Maximizer repo safely (tests, audits, dashboards, offline runs) with TS-first guardrails. Includes optional OpenClaw notification hooks.
metadata: { "openclaw": { "requires": { "anyBins": ["python", "python3"] } } }
---

# Portfolio Maximizer

Use this skill when the user asks you to run/inspect Portfolio Maximizer locally, check audit gates, or summarize system state.

## Guardrails

- Default to offline/synthetic/diagnostic modes unless the user explicitly requests live runs.
- Do not use real broker credentials or execute live orders without explicit confirmation.
- Prefer read-only inspection of `data/portfolio_maximizer.db`. If you need to run experiments that mutate state, copy the DB first.
- Keep "Time Series primary, LLM fallback" intact. Do not enable LLM fallback by default.
- Secrets: never paste `.env` contents into chat. Use `scripts/validate_credentials.py` to confirm required keys are present (it never prints values).

## Common Commands

### Setup

- Linux/WSL/Git Bash: `bash setup_environment.sh`
- Windows: `setup_environment.bat`

### Tests

- Full suite: `python -m pytest tests/ -q`
- Targeted: `python -m pytest tests/scripts/test_auto_trader_lifecycle.py -q`

### Audit Gates

- Run the production gate: `python scripts/production_audit_gate.py`
- OpenClaw notify: auto-sends if `OPENCLAW_TO` is set (disable via `PMX_NOTIFY_OPENCLAW=0`); can also force with `python scripts/production_audit_gate.py --notify-openclaw`

### Health Checks

- Dashboard payload sanity: `python scripts/check_dashboard_health.py`
- Forecast audit gate summary: `python scripts/check_forecast_audits.py --audit-dir logs/forecast_audits --config-path config/forecaster_monitoring.yml`

### OpenClaw Notifications (Optional)

This repo includes:

- `scripts/openclaw_notify.py` (manual send helper)
- `scripts/error_monitor.py` (optional OpenClaw alert delivery if enabled)
- `scripts/production_audit_gate.py --notify-openclaw` (optional gate delivery)

Recommended env vars:

- `OPENCLAW_COMMAND` (default: `openclaw`, or `wsl openclaw` on Windows)
- `OPENCLAW_TO` (target string)
- `OPENCLAW_TIMEOUT_SECONDS` (default: `20`)

See `Documentation/OPENCLAW_INTEGRATION.md`.
