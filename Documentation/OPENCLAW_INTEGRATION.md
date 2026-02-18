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
  Example: `whatsapp:+2347042437712,telegram:@mychat,discord:channel:123456789012345678`
- `OPENCLAW_TO` (recommended for notifications)  
  Examples: `+2347042437712`, `discord:...`, `slack:...`
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

## Cron Automation (Updated 2026-02-17)

OpenClaw runs 9 audit-aligned cron jobs via `agentTurn` mode. The local LLM (qwen3:8b)
receives the cron message, prefers structured orchestrator tools (not generic exec chaining), and either
reports anomalies or responds `NO_REPLY` (suppressing the notification).

For production gate/reconciliation requests, prompt templates now bias to:
- `run_production_audit_gate` tool (inside `scripts/llm_multi_model_orchestrator.py`)
- Prompt template source: `config/openclaw_prompt_templates.yml`

For human review workflows, a dedicated cron-safe forwarder is available:
- `scripts/forward_self_improvement_reviews.py`
- Source queue: `logs/llm_activity/proposals/`
- Delivery path: `scripts/openclaw_notify.py` with `OPENCLAW_TARGETS` fan-out (WhatsApp/Discord/Telegram)
- No-op when `OPENCLAW_TARGETS` / `OPENCLAW_TO` is unset.

### LLM Fine-Tune Command Default

`PMX_LLM_FINETUNE_COMMAND` is now internalized for scheduled training runs:
- Default is set in `config/training_priority.yml` (`training_priority.defaults.env`).
- `scripts/prepare_llm_finetune_dataset.py --run-trainer` resolves command order as:
  1. `--trainer-command` (CLI)
  2. `PMX_LLM_FINETUNE_COMMAND` (environment)
  3. internal default command template
- Template variables supported: `{dataset}`, `{dataset_path}`, `{project_root}`, `{python_bin}`.

### Progressive Task Reporting

OpenClaw orchestration now supports progressive status updates during long-running tasks
(tool priming, round execution, tool calls, fallback, completion), instead of only final replies.

Controls:
- `PMX_QWEN_PROGRESS_UPDATES` (default: `true`)
- `PMX_QWEN_PROGRESS_MIN_INTERVAL_SECONDS` (default profile-driven: `8` low-latency, `10` high-accuracy)
- `PMX_QWEN_PROGRESS_MAX_MESSAGE_CHARS` (default: `220`)

For explicit runtime setup prompts (for example: "install torch system-wide" or "pip install pandas"),
`openclaw_bridge` now uses a fast path that executes dependency tools directly:
- `install_torch_runtime`
- `install_python_package`

This avoids waiting for a full qwen tool-calling round when the task is clearly operational.

### OpenClaw Maintenance Guard

To keep WhatsApp/dashboard operations stable over long runs, use:
- `scripts/openclaw_maintenance.py`

What it does:
- Archives stale session lock files under `~/.openclaw/agents/*/sessions/*.jsonl.lock`.
- Optionally archives matching stale session `.jsonl` files.
- Checks gateway RPC health and can restart gateway on failure.
- Detects primary WhatsApp handshake timeout/disconnect states and restarts gateway for recovery.
- Re-checks channel status after restart so unresolved primary channel failures are never reported as `PASS`.
- Classifies DNS failures (`ENOTFOUND web.whatsapp.com`) separately from listener/session failures.
- Uses configurable restart attempts (`--primary-restart-attempts`, default 2) for transient primary-channel recovery.
- Marks unresolved primary channel failures as `FAIL` in maintenance reports to make outages explicit.
- Optionally toggles the primary account enabled=false/true and restarts gateway to recover a missing listener.
- Optionally disables broken non-primary channels (Telegram/Discord) when known auth/config errors persist.

Run once (apply mode):
- `python scripts/openclaw_maintenance.py --apply --disable-broken-channels --restart-gateway-on-rpc-failure --attempt-primary-reenable --recheck-delay-seconds 8 --primary-restart-attempts 2`

Cron path:
- `bash/production_cron.sh openclaw_maintenance`

Windows Task Scheduler helper:
- `powershell -ExecutionPolicy Bypass -File scripts/run_openclaw_maintenance.ps1`

### Job Definitions

Jobs are stored in `~/.openclaw/cron/jobs.json` (not in the repo). Each job uses
`sessionTarget: "isolated"` for clean execution context and `delivery.mode: "announce"`
(except Weekly Session Cleanup which uses `"none"`).

| Job | Priority | Schedule | Script |
|-----|----------|----------|--------|
| PnL Integrity Audit | P0 | `0 */4 * * *` (every 4h) | `python -m integrity.pnl_integrity_enforcer --db data/portfolio_maximizer.db` |
| Production Gate Check | P0 | `0 7 * * *` (daily 7 AM) | `run_production_audit_gate` tool (preferred) / `python scripts/production_audit_gate.py` fallback |
| Quant Validation Health | P0 | `30 7 * * *` (daily 7:30 AM) | Inline Python analyzing `logs/signals/quant_validation.jsonl` |
| Signal Linkage Monitor | P1 | `0 8 * * *` (daily 8 AM) | Inline Python querying DB for NULL signal_id, orphans |
| Ticker Health Monitor | P1 | `30 8 * * *` (daily 8:30 AM) | Inline Python checking per-ticker PnL and consecutive losses |
| GARCH Unit-Root Guard | P2 | `0 9 * * 1` (Mon 9 AM) | Inline Python scanning forecast audits for alpha+beta >= 0.98 |
| Overnight Hold Monitor | P2 | `0 9 * * 5` (Fri 9 AM) | Inline Python comparing intraday vs overnight PnL |
| Model Training Autopilot | P1 | `15 */6 * * *` (every 6h) | `python scripts/openclaw_training_autopilot.py --benchmark-factor 0.989` (detaches `run_training_priority_cycle.py` when below benchmarks) |
| System Health Check | -- | `0 */6 * * *` (every 6h) | `python scripts/llm_multi_model_orchestrator.py status` + `python scripts/error_monitor.py --check` |
| OpenClaw Maintenance | -- | `0 3 * * 0` (Sun 3 AM) | `bash/production_cron.sh openclaw_maintenance` (stale locks + gateway/channel guard) |
| Self-Improvement Review Forward | -- | Cron via `bash/production_cron.sh self_improvement_review_forward` | Sends sanitized pending proposal summaries to human reviewers |

#### Model Training Autopilot (Benchmark-Chasing, Detached)

The training autopilot is designed for OpenClaw cron: it runs fast, and when benchmarks are below target it starts a **detached** background training cycle (so OpenClaw cron sessions do not get stuck).

- Script: `scripts/openclaw_training_autopilot.py`
- Benchmarks (factor default: `0.989`):
  - Profitability proof vs `config/profitability_proof_requirements.yml`
  - Forecaster adversarial suite vs `config/forecaster_monitoring_ci.yml` (reads `logs/automation/training_priority/adversarial_forecaster_suite.json`)
- State: `logs/automation/training_autopilot/state.json` (restart-safe; prevents duplicate concurrent training)
- LLM activity logging:
  - proposal: `logs/llm_activity/proposals/training/`
  - feedback: `logs/llm_activity/feedback/`

### Announcement Rules

Each cron job message includes explicit rules for when to announce vs stay silent:
- **Core principle**: Only announce anomalies. Routine success = NO_REPLY.
- P0 jobs announce on CRITICAL/HIGH violations, gate FAIL, or FAIL rate >= 90%.
- P1 jobs announce on orphan detections, consecutive losses, or PnL thresholds.
- P2 jobs announce on unit-root or overnight drag thresholds.
- System Health Check announces any model offline or error monitor issues.

### Management Commands

```bash
# List all cron jobs with status
openclaw cron list

# Force-run a specific job (2-minute timeout)
openclaw cron run <job-id> --timeout 120000

# Enable/disable a job
openclaw cron enable <job-id>
openclaw cron disable <job-id>

# Check gateway health
openclaw gateway status
```

### Schema Notes

The cron job schema uses:
- `schedule.kind: "cron"` with `schedule.expr: "0 */4 * * *"` (NOT `cron` or `expression`)
- `delivery.mode: "announce"` or `"none"` (NOT `"silent"`)
- `payload.kind: "agentTurn"` for LLM-driven script execution
- `sessionTarget: "isolated"` for clean execution context
- Job IDs must be UUIDs

## 3-Model Local LLM Strategy (Updated 2026-02-16)

Portfolio Maximizer uses three specialized local models via Ollama, each with a
distinct role in the inference pipeline:

| Model | Role | Capabilities | VRAM | Speed |
|-------|------|-------------|------|-------|
| **deepseek-r1:8b** | Fast reasoning | Chain-of-thought, math, code-gen | 5.5GB | 25-35 tok/s |
| **deepseek-r1:32b** | Heavy reasoning | Deep multi-step analysis, long-context | 20GB | 10-15 tok/s |
| **qwen3:8b** | Tool orchestrator | Function-calling, structured output, thinking mode | 5.5GB | 30-40 tok/s |

### Task-to-Model Routing

| Task | Model |
|------|-------|
| Market analysis, signal generation, regime detection | deepseek-r1:8b |
| Portfolio optimization, adversarial audits | deepseek-r1:32b |
| Tool/function calling, API orchestration, social media | qwen3:8b |

### How It Works

**qwen3:8b** acts as the orchestrator: it receives requests via OpenClaw
social media channels (WhatsApp/Telegram/Discord) and can dispatch reasoning
tasks to the deepseek-r1 models as "tools" via Ollama's native tool-calling API.

```
User (WhatsApp/Telegram/Discord)
  -> OpenClaw Gateway
    -> qwen3:8b (orchestrator, tool-calling)
      -> deepseek-r1:8b (fast reasoning)
      -> deepseek-r1:32b (heavy reasoning)
    <- structured response
  <- notification to user
```

### Setup

```bash
# Pull all 3 models
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:32b
ollama pull qwen3:8b

# Apply to OpenClaw (local-first strategy with all 3 models)
python scripts/openclaw_models.py apply --strategy local-first --restart-gateway

# Verify
python scripts/openclaw_models.py status --list-ollama-models
```

### Multi-Model Orchestration CLI

```bash
# Check model availability
python scripts/llm_multi_model_orchestrator.py status

# Route a task to the best model
python scripts/llm_multi_model_orchestrator.py route --task "Analyze AAPL regime"

# Run orchestrated query (qwen3 dispatches to deepseek-r1 models)
python scripts/llm_multi_model_orchestrator.py orchestrate --prompt "Summarize gate status and suggest next actions"
```

### Override via Environment

Set `OPENCLAW_OLLAMA_MODEL_ORDER` in `.env` to override the default priority:

```
OPENCLAW_OLLAMA_MODEL_ORDER=deepseek-r1:8b,deepseek-r1:32b,qwen3:8b
```

## Interactions API Security Modes

The Interactions API (`scripts/pmx_interactions_api.py`) supports three auth
modes, controlled by `INTERACTIONS_AUTH_MODE`:

| Mode | Accepts API Key | Accepts JWT | Use Case |
|------|-----------------|-------------|----------|
| `any` (default) | Yes | Yes | Development, flexible auth |
| `jwt-only` | No | Yes | Production with Auth0 |
| `api-key-only` | Yes | No | Simple deployments |

### Configuration

```bash
# In .env:
INTERACTIONS_AUTH_MODE=jwt-only

# Requires Auth0 config:
AUTH0_DOMAIN=your-tenant.us.auth0.com
AUTH0_AUDIENCE=your-api-identifier
```

### Minimum Key Length

The minimum API key length defaults to 16 characters and can be raised
(never lowered) via `INTERACTIONS_MIN_KEY_LENGTH`:

```bash
INTERACTIONS_MIN_KEY_LENGTH=32
```

### CORS

Set `INTERACTIONS_CORS_ORIGINS` to a comma-separated list of allowed origins:

```bash
INTERACTIONS_CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

Omit the variable entirely to disable CORS headers (default).

### ngrok Integration

`scripts/start_ngrok_interactions.ps1` respects `INTERACTIONS_AUTH_MODE`:
- In `jwt-only` mode, it requires `AUTH0_DOMAIN` + `AUTH0_AUDIENCE` (API key alone
  is not sufficient).
- In `api-key-only` mode, it requires a strong `INTERACTIONS_API_KEY`.
- In `any` mode (default), either is accepted.
