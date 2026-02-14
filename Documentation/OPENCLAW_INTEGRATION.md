# OpenClaw Integration

This repository includes an optional integration with OpenClaw (https://openclaw.ai): a personal AI assistant you run on your own devices.

## Security Notes (Practical Defaults)

- `.env` remains local and git-ignored. Do not paste secret values into chats.
- Outbound alerts (OpenClaw + email) apply best-effort redaction for env var values whose names look like secrets (`*_KEY`, `*_TOKEN`, `*_PASSWORD`, etc.).
- Prefer `*_FILE` secrets for anything you do not want stored in `.env` (see `Documentation/API_KEYS_SECURITY.md`).

## Workspace Skill

OpenClaw loads workspace skills from `<workspace>/skills`. This repo ships a workspace skill at:

- `skills/portfolio-maximizer/SKILL.md`
- `skills/pmx-inbox/SKILL.md` (Gmail/Proton inbox workflows)

Once your OpenClaw workspace points at the repo root, the skill becomes available to guide safe commands (tests, audits, offline runs) without bypassing the project's TS-first and risk guardrails.

## OpenClaw Notifications

If you have the OpenClaw CLI installed/configured, Portfolio Maximizer can send notifications via:

- `openclaw message send --target ... --message ...`

### Environment Variables

Configure targets via environment variables (recommended so you do not hardcode personal targets in committed YAML):

- `OPENCLAW_COMMAND` (default: `openclaw`)  
  Examples: `openclaw`, `wsl openclaw`
- `OPENCLAW_TO` (required for notifications)  
  Examples: `+15551234567`, `discord:...`, `slack:...`
- `OPENCLAW_TIMEOUT_SECONDS` (default: `20`)

### Error Monitor Alerts (Default: Enabled, No-Op Until Configured)

1. Set at least `OPENCLAW_TO` (in `.env` or environment).
2. Run:
   - `python scripts/error_monitor.py`

When thresholds are exceeded, the error monitor will save a file alert under `logs/alerts/` and also deliver a truncated summary via OpenClaw (if configured).

### Production Audit Gate Alerts

`scripts/production_audit_gate.py` can send its PASS/FAIL summary via OpenClaw.

Defaults:

- If `OPENCLAW_TO` is configured, the script will auto-notify.
- Disable with `PMX_NOTIFY_OPENCLAW=0`.
- You can also force the behavior explicitly with `--notify-openclaw`.

## Manual Send Helper

For ad-hoc testing, you can send a message directly:

- `python scripts/openclaw_notify.py --to "<target>" --message "Hello from Portfolio Maximizer"`

## Gmail / Email Alerts (Optional)

The error monitor also supports sending alerts via SMTP (Gmail supported via STARTTLS).

Recommended env vars (set in `.env`):

- `PMX_EMAIL_SMTP_SERVER` (default: `smtp.gmail.com`)
- `PMX_EMAIL_SMTP_PORT` (default: `587`)
- `PMX_EMAIL_USE_TLS` (default: `1`)
- `PMX_EMAIL_USERNAME`
- `PMX_EMAIL_PASSWORD` (Gmail: app password recommended)
- `PMX_EMAIL_FROM` (defaults to username if omitted)
- `PMX_EMAIL_TO` (comma-separated recipients)

Control is in:

- `config/error_monitoring_config.yml` (`alerts.email.*`)

## Safe Limits (Dial Up Over Time)

All outbound notifications are intentionally short and rate-limited by default:

- Error alert cooldown: `config/error_monitoring_config.yml` -> `error_thresholds.alert_cooldown_minutes`
- OpenClaw max length: `config/error_monitoring_config.yml` -> `alerts.openclaw.max_message_chars`
- Email max length: `config/error_monitoring_config.yml` -> `alerts.email.max_body_chars`

Inbox workflows have their own safe limits:

- `config/inbox_workflows.yml` -> `limits.*`
- Sending is disabled by default; enable with `PMX_INBOX_ALLOW_SEND=1` (or `limits.allow_send: true`).

See: `Documentation/INBOX_WORKFLOWS.md`
