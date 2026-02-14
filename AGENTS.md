# Agent Guardrails (Repo Local)

This repo is designed to be operated by automation (humans + coding agents + OpenClaw) without leaking credentials.

## Non-Negotiables (Secrets)

- Never paste `.env` contents into chat/logs/issues/PRs.
- Treat any value from env vars containing `KEY`, `TOKEN`, `SECRET`, or `PASSWORD` as a secret.
- Use `etl/secret_loader.py` (`load_secret()` / `bootstrap_dotenv()`) for secret access.
- Prefer `*_FILE` secrets for anything you do not want stored in `.env`.
- Validate presence only (no values): `python scripts/validate_credentials.py`.

## OpenClaw + Gmail Defaults

- OpenClaw + SMTP email alert delivery is enabled by default in `config/error_monitoring_config.yml`, but is a **no-op until configured**.
- OpenClaw notifications:
  - Configure `OPENCLAW_TO` (and optionally `OPENCLAW_COMMAND`).
  - `scripts/production_audit_gate.py` auto-notifies if `OPENCLAW_TO` is set (disable with `PMX_NOTIFY_OPENCLAW=0`).
- Gmail/email alerts (SMTP):
  - Configure `PMX_EMAIL_USERNAME`, `PMX_EMAIL_PASSWORD` (app password recommended), and `PMX_EMAIL_TO`.

## Inbox Workflows (Gmail + Proton)

- Config: `config/inbox_workflows.yml`
- Scan script: `python scripts/inbox_workflow.py scan`
- Safe defaults:
  - Read-only scans (no mark-seen) by default.
  - Email sending is disabled by default. Enable explicitly with `PMX_INBOX_ALLOW_SEND=1` (or set `limits.allow_send: true` in the YAML).
- Proton Mail requires Proton Mail Bridge (local IMAP/SMTP endpoints).

## Safety Levels (Increase Over Time)

Start conservative and lift limits only after you are satisfied with behavior:

1. **Read-only**: inspect logs/configs, run tests, run audits.
2. **Safe automation**: run `scripts/production_audit_gate.py`, `scripts/error_monitor.py`, dashboards, and notifications.
3. **Code changes**: implement refactors/features with tests and compile checks.
4. **Trading operations**: only with explicit confirmation; never run live execution by default.

## References

- `Documentation/API_KEYS_SECURITY.md`
- `Documentation/OPENCLAW_INTEGRATION.md`
- `Documentation/AGENT_INSTRUCTION.md`
