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
- `OPENCLAW_CHANNEL` (optional)  
  Examples: `whatsapp`, `telegram`, `discord`  
- `OPENCLAW_TARGETS` (optional, multi-target)  
  Comma-separated list. Items may be `channel:target`.  
  Example: `whatsapp:+15551234567,telegram:@mychat,discord:channel:123456789012345678`
- `OPENCLAW_TO` (recommended for notifications)  
  Examples: `+15551234567`, `discord:...`, `slack:...`
- `OPENCLAW_TIMEOUT_SECONDS` (default: `20`)
- `OPENCLAW_AGENT_TIMEOUT_SECONDS` (default: `600`) (prompt mode)
- `OPENCLAW_REPLY_CHANNEL` / `OPENCLAW_REPLY_TO` / `OPENCLAW_REPLY_ACCOUNT` (prompt mode optional)  
  Lets prompt mode deliver the reply to a different channel/target.

### Error Monitor Alerts (Default: Enabled, No-Op Until Configured)

1. Set at least `OPENCLAW_TARGETS` or `OPENCLAW_TO` (in `.env` or environment).
2. Run:
   - `python scripts/error_monitor.py`

When thresholds are exceeded, the error monitor will save a file alert under `logs/alerts/` and also deliver a truncated summary via OpenClaw (if configured).

### Production Audit Gate Alerts

`scripts/production_audit_gate.py` can send its PASS/FAIL summary via OpenClaw.

Defaults:

- If `OPENCLAW_TARGETS`/`OPENCLAW_TO` is configured, the script will auto-notify.
- Disable with `PMX_NOTIFY_OPENCLAW=0`.
- You can also force the behavior explicitly with `--notify-openclaw`.

## Manual Send Helper

For ad-hoc testing, you can send a message directly:

- `python scripts/openclaw_notify.py --to "<target>" --message "Hello from Portfolio Maximizer"`

To attach a file (image/audio/video/document):

- `python scripts/openclaw_notify.py --to "<target>" --media "path\\to\\file.png" --message "See attached"`

If `OPENCLAW_TARGETS`/`OPENCLAW_TO` is not set, the helper will try to infer a WhatsApp "message yourself" target from:

- `openclaw status --json`

This enables the simple workflow:

- `python scripts/openclaw_notify.py --message "Hello from Portfolio Maximizer"`

If you want to send to multiple channels, set:

- `OPENCLAW_TARGETS=whatsapp:+15551234567,telegram:@mychat`

Then:

- `python scripts/openclaw_notify.py --message "Hello from Portfolio Maximizer"`

### WhatsApp Prompting (Agent Turns)

If you want to *prompt* your OpenClaw agent and (optionally) deliver the reply back to WhatsApp:

- `python scripts/openclaw_notify.py --prompt --message "Summarize today's gate status"`

Notes:

- `openclaw agent` uses the OpenClaw Gateway. If prompting fails with connection refused, start/restart the gateway (e.g. `openclaw gateway restart`) and re-run.

To deliver the reply somewhere else (e.g. Telegram) while keeping the session keyed by your WhatsApp number:

- `python scripts/openclaw_notify.py --prompt --to +15551234567 --reply-channel telegram --reply-to "@mychat" --message "Summarize today's gate status" --deliver`

PowerShell note: targets like `@mychat` must be quoted (otherwise PowerShell treats them as splatting). Example:

- `python scripts/openclaw_notify.py --prompt --to +15551234567 --reply-channel telegram --reply-to "@mychat" --message "Summarize today's gate status" --deliver`

## Other Messaging Apps (Remote Notifications + Prompting)

OpenClaw supports multiple channels. On Windows, the most reliable "remote" options are usually Telegram or Discord bots.

1. Set the channel token(s) in `.env` (do not commit):

- `TELEGRAM_BOT_TOKEN` (Telegram)
- `DISCORD_BOT_TOKEN` (Discord)
- `SLACK_BOT_TOKEN` + `SLACK_APP_TOKEN` (Slack, if you use it)

2. Enable the plugin(s) and add the channel account (loads `.env` via the wrapper):

- `python scripts/openclaw_env.py plugins enable telegram`
- `python scripts/openclaw_env.py channels add --channel telegram --use-env`
- `python scripts/openclaw_env.py plugins enable discord`
- `python scripts/openclaw_env.py channels add --channel discord --use-env`

3. Restart the gateway:

- `openclaw gateway restart`

4. Send a test message:

- `python scripts/openclaw_notify.py --channel telegram --to "@mychat" --message "PMX test"`
- `python scripts/openclaw_notify.py --channel discord --to channel:123456789012345678 --message "PMX test"`

## Audio Notifications (Windows TTS)

If you're on Windows, this repo includes a small helper that generates a WAV via `System.Speech` and sends it as OpenClaw media:

- `python scripts/pmx_tts_notify.py --text "Run completed. Check the dashboard."`

## WhatsApp Connectivity Troubleshooting (408 Handshake Timeout)

If `openclaw channels status` shows WhatsApp `stopped/disconnected` with an "Opening handshake has timed out" error:

1. Confirm the gateway service is running:

- `python scripts/openclaw_env.py gateway status --json`
- If needed: `openclaw gateway restart`

2. Re-check channel status:

- `python scripts/openclaw_env.py channels status`

3. If it still fails, relink WhatsApp (interactive):

- `openclaw channels login --channel whatsapp --verbose`


## Running OpenClaw With Repo `.env` Loaded

PMX scripts already load the repo's `.env` (best-effort) via `etl/secret_loader.py`.

If you want the OpenClaw CLI itself to run with the same `.env` context, use:

- `python scripts/openclaw_env.py status --json`
- `python scripts/openclaw_env.py message send --channel whatsapp --target +15551234567 --message "Hi"`

## Model Provider Failover (Local + Remote, Avoid Lock-In)

You cannot bypass paid provider access, but you *can* avoid single-provider lock-in by configuring OpenClaw to fail over to **local open-source models** (Ollama) when a remote provider is down or keys are unavailable.

Key points:

- OpenClaw supports a **primary model** plus **fallbacks** (`agents.defaults.model.primary` + `agents.defaults.model.fallbacks`).
- On Windows, the OpenClaw Gateway runs as a **Scheduled Task**, so it may not inherit your repo `.env`. If you want OpenAI/Anthropic keys to work from gateway-run prompting, you must sync them into OpenClaw’s local auth store.
- OpenClaw currently expects an API key to be resolvable for the **Ollama** provider even for localhost. `scripts/openclaw_models.py` writes a harmless placeholder (`ollama:default=local`) into the OpenClaw auth store so local failover works.
- OpenClaw also maintains a **model allowlist** (`agents.defaults.models`). Fallbacks will be ignored unless they are allowlisted; `scripts/openclaw_models.py apply` updates the allowlist automatically.

### Quick Start (Recommended: Auto Strategy)

1. Ensure Ollama is running and at least one model is installed.
   - See: `Documentation/LLM_INTEGRATION.md`
   - Recommended (RTX 4060 Ti 16GB class):
     - `ollama pull qwen:14b-chat-q4_K_M`
     - `ollama pull deepseek-coder:6.7b-instruct-q4_K_M`
     - `ollama pull codellama:13b-instruct-q4_K_M`
2. Inspect current OpenClaw model config:
   - `python scripts/openclaw_models.py status --list-ollama-models`
3. Apply failover config (auto chooses local-first only when Ollama is reachable):
   - `python scripts/openclaw_models.py apply --strategy auto --restart-gateway`

### Local-First (Option A)

If you want local to be the default even when Ollama is temporarily down (OpenClaw will fall back to Qwen Portal):

- `python scripts/openclaw_models.py apply --strategy local-first --restart-gateway`

### Enable Paid Providers As Optional Fallbacks (OpenAI / Claude)

1. Put keys in repo `.env` (git-ignored): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (or `CLAUDE_API_KEY`).
2. Sync keys into OpenClaw auth store and configure providers + fallbacks:
   - `python scripts/openclaw_models.py apply --sync-auth --enable-openai-provider --enable-anthropic-provider --restart-gateway`

### Custom Model Chain (Explicit)

Set these in `.env` (see `.env.template`), then apply:

- `OPENCLAW_MODEL_STRATEGY=custom`
- `OPENCLAW_MODEL_PRIMARY=...`
- `OPENCLAW_MODEL_FALLBACKS=...`

Then:

- `python scripts/openclaw_models.py apply --restart-gateway`

## Log Location (Optional: Move To D:)

OpenClaw gateway logs on Windows commonly land under:

- `C:\tmp\openclaw\`

PMX script logs land under:

- `<repo>\logs\`

If you want both redirected to D: without changing code, run:

- `powershell -ExecutionPolicy Bypass -File scripts/redirect_logs_to_d.ps1 -DryRun`
- `powershell -ExecutionPolicy Bypass -File scripts/redirect_logs_to_d.ps1`

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
