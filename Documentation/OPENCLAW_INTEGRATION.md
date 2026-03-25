# OpenClaw Integration

This repository includes an optional integration with OpenClaw (https://openclaw.ai): a personal AI assistant you run on your own devices.

Implementation policy (repo-wide contract): `Documentation/OPENCLAW_IMPLEMENTATION_POLICY.md`

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

### Exec Environment Enforcement (Host, Sandbox, ACP)

To avoid host/sandbox/ACP drift between Windows and WSL maintenance paths, use
the enforcement script before running OpenClaw maintenance flows:

- `python scripts/enforce_openclaw_exec_environment.py`
- dry-run: `python scripts/enforce_openclaw_exec_environment.py --dry-run`

`scripts/run_openclaw_maintenance.ps1` now enforces this in both paths:

- Windows path: runs `scripts/enforce_openclaw_exec_environment.py` before maintenance.
- WSL path: runs the same enforcement command inside WSL before invoking cron maintenance.
- If `tools.exec.host=sandbox` is preferred but Docker is unavailable, the
  enforcement step automatically falls back to `node` on that host so cron and
  WhatsApp agent turns do not fail on sandbox-image inspection.

Runtime health snapshot:

- `python scripts/project_runtime_status.py --pretty`

The runtime status payload now includes explicit `openclaw_exec_env` signals:

- `invalid_exec_host` (`tools.exec.host` not in `sandbox|gateway|node`)
- `invalid_sandbox_mode` (when host is `sandbox` but defaults or exec-capable
  agent overrides are not `non-main|all`)
- `sandbox_runtime_unavailable` (when host is `sandbox` but the Docker-backed
  sandbox runtime is not reachable)
- `missing_acp_default_agent`
- `exec_env_valid`

### Autonomous-Run Security Guard (Prompt Injection + Sensitive Actions)

Prompt-mode runs (`python scripts/openclaw_notify.py --prompt ...`) now enforce a guard in
`utils/openclaw_cli.py` before any `openclaw agent` call:

- Blocks high-risk requests unless approval token is present:
  - secret exfiltration/entry requests
  - irreversible financial/account actions
  - CAPTCHA bypass requests
- Prefixes each agent request with `[PMX_AUTONOMY_POLICY]` so untrusted
  webpage/email instructions are treated as potential prompt injection.

Guard env vars:
- `OPENCLAW_AUTONOMY_GUARD_ENABLED` (default: `1`)
- `OPENCLAW_AUTONOMY_REQUIRE_APPROVAL_TOKEN` (default: `1`)
- `OPENCLAW_AUTONOMY_APPROVAL_TOKEN` (default: `PMX_APPROVE_HIGH_RISK`)
- `OPENCLAW_AUTONOMY_POLICY_PREFIX_ENABLED` (default: `1`)
- `OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS` (default: `0`, recommended `1` for unattended runs)

Recommended hardened profile for autonomous hosts:
- `OPENCLAW_AUTONOMY_GUARD_ENABLED=1`
- `OPENCLAW_AUTONOMY_REQUIRE_APPROVAL_TOKEN=1`
- `OPENCLAW_AUTONOMY_APPROVAL_TOKEN=<non-trivial token>`
- `OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS=1`

### Tavily API (Preferred Over Brave For Quota-Limited Search)

Use Tavily as the default web grounding path for PMX automations, with robust
fallback to free providers:

- Add `TAVILY_API_KEY` to `.env` (or `TAVILY_API_KEY_FILE`).
- Run direct Tavily search from PMX:
  - `python scripts/tavily_search.py --query "latest market regime for AAPL" --json`
  - Optional provider order override:
    - `python scripts/tavily_search.py --query "..." --providers "tavily,duckduckgo,wikipedia" --json`
Tavily MCP wiring is archived for this repo. PMX uses direct Tavily API search for web grounding.

Brave can remain configured as fallback only (`BRAVE_API_KEY`), but PMX defaults should prefer Tavily API.

Default search provider chain:
- `tavily -> duckduckgo -> wikipedia`
- Override with env: `PMX_WEB_SEARCH_PROVIDERS`
- Bridge fast-path now routes explicit online-search prompts (`web search`, `online search`, `search online`, `web lookup`, `internet lookup`, `look up online`, `find on the web`, `google`, `duckduckgo`, `wikipedia`, `tavily`) to `search_web_tavily` so WhatsApp/Discord requests can use robust fallback without Brave dependency.

If an agent replies with a Brave-key prompt (for example: "`web_search` requires `BRAVE_API_KEY`"), treat it as tool-routing drift and force PMX search path:

- `python scripts/tavily_search.py --query "your question" --json`
- Ensure `TAVILY_API_KEY` (or `TAVILY_API_KEY_FILE`) is present for Tavily depth.
- If no Tavily key is available, free-provider fallback still works.
- Re-run the same request with explicit instruction: "use PMX search tool, not web_search/Brave".

### WhatsApp Search Routing Prompt (Corrected To Current Project State)

Use this exact runbook for "web search ..." interactions over WhatsApp.

1. Restart gateway with the real command path:
- `openclaw gateway restart`

2. Verify provider fallback search works (no Brave dependency):
- `python scripts/tavily_search.py --query "OpenAI API docs" --providers "duckduckgo,wikipedia" --json`

3. Trigger PMX bridge on WhatsApp:
- `python scripts/llm_multi_model_orchestrator.py openclaw-bridge --channel whatsapp --reply-to +2347042437712 --message "web search latest OpenClaw docs"`

4. Validate event evidence in activity logs:
- `bridge_incoming`
- `bridge_web_fast_path_start`
- `bridge_web_fast_path_complete`

Compatibility note:
- Older runs may show `bridge_status_fast_path_start/complete` for status-only flow.
- Web-search fast-path validation should prefer `bridge_web_fast_path_*`.

5. One-command end-to-end validator (recommended):
- `python scripts/openclaw_search_routing_e2e.py --json --require-web-events --allow-legacy-status-events`

Important corrections:
- Do not use `python openclaw_agent.py restart` (not a project command).
- Do not use `python tavily_search.py ...` from repo root; use `python scripts/tavily_search.py ...`.
- WhatsApp integration here uses OpenClaw WhatsApp Web provider; Twilio webhook setup is not required for this repo path.
- "Discord via WhatsApp" is not a routing model in PMX. Use channel-specific routing (`whatsapp`, `discord`, `telegram`) explicitly.
- For quant-validation fail-rate checks, do not generate inline multiline `python -c` payloads on PowerShell. Use:
  - `python scripts/quant_validation_headroom.py --json`
  - `python scripts/check_quant_validation_health.py`

### GitHub Repository Check Path (WhatsApp-Friendly)

For WhatsApp requests like "check this project repo on GitHub", prefer the local deterministic command path:

- `python scripts/check_github_repo_status.py --pretty`

This avoids dependence on `web_search` provider credentials and returns:
- local branch/head/dirty state
- upstream tracking/ahead-behind
- parsed GitHub remote slug
- GitHub API repo/default-branch head checks (when reachable)

Bridge hardening (current behavior):
- `scripts/llm_multi_model_orchestrator.py` includes a dedicated repo fast-path (`check_github_repo_status`).
- Repo/GitHub requests are evaluated before generic health/status fast-path.
- A WhatsApp query like "check project repository on GitHub status" now returns repo branch/head/ahead-behind/issues instead of generic system health.

### Source-Code Attribution Tooling (Local LLM)

The orchestrator now exposes a safe source-inspection tool:
- Tool: `inspect_source_code`
- Purpose: read allowlisted PMX files and return attributable snippets with `path:line` citations.
- Index source: `logs/llm_activity/source_index.json` (auto-built via `python scripts/openclaw_self_improve.py index` when needed).

Practical prompt pattern:
- "Inspect source code for X, cite `path:line`, then propose an improvement."

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

## Discord Connectivity Troubleshooting (401 / Failed To Resolve Application ID)

If `openclaw status --deep` shows `Discord WARN failed (401) - getMe failed (401)` or the Discord channel logs show `Failed to resolve Discord application id`:

1. Confirm the Discord bot token is actually present in `.env`:

- `python scripts/validate_credentials.py`

2. If the interactions app is configured but `discord_bot_token` is still reported missing, add one of:

- `DISCORD_BOT_TOKEN`
- `DISCORD_TOKEN`

3. Re-apply the channel from `.env` and restart the gateway:

- `python scripts/openclaw_env.py plugins enable discord`
- `python scripts/openclaw_env.py channels add --channel discord --use-env`
- `openclaw gateway restart`

4. Re-check the live runtime:

- `openclaw channels status --json`
- `openclaw channels logs --channel discord --lines 80`
- `openclaw status --deep`

5. Do not count Discord as healthy until `openclaw status --deep` reports `Discord OK`.

For a sanitized repo-local audit and remediation record, see `Documentation/OPENCLAW_DISCORD_TOKEN_AUDIT_2026-03-24.md`.

## WhatsApp Connectivity Troubleshooting (408 Handshake Timeout)

If `openclaw channels status` shows WhatsApp `stopped/disconnected` with an "Opening handshake has timed out" error:

1. Confirm the gateway service is running:

- `python scripts/openclaw_env.py gateway status --json`
- If needed: `openclaw gateway restart`

2. Re-check channel status:

- `python scripts/openclaw_env.py channels status`

3. If it still fails, relink WhatsApp (interactive):

- `openclaw channels login --channel whatsapp --verbose`

### One-Command Fresh Relink (Recommended)

When WhatsApp repeatedly fails to link (for example with `status=405 Method Not Allowed`), use the PMX helper to rotate auth storage and force a fresh pairing token:

- `python scripts/openclaw_whatsapp_relink.py --fresh-auth`

This helper will:
- log out the selected WhatsApp account
- set a timestamped `authDir` (preserving old auth data)
- probe critical URLs (`http://127.0.0.1:18789/channels`, `https://openclaw.ai/`) before login
- auto-apply an idempotent OpenClaw WhatsApp runtime hotfix (with `.pmxbak` backups) for known Baileys WA-version `405` handshake failures
- restart the gateway
- run interactive `openclaw channels login --channel whatsapp --account default --verbose`
- verify post-login status with `--probe`

If you want to force the runtime hotfix even when no prior marker was detected:
- `python scripts/openclaw_whatsapp_relink.py --fresh-auth --force-wa-version-hotfix`

## Tool-Call Failure Hardening (PowerShell + Edit Schema)

When gateway logs show patterns like:
- `ScriptBlock should only be specified as a value of the Command parameter`
- `The term 'True' is not recognized ... CommandNotFoundException`
- `edit failed: Missing required parameter: newText`

apply this sequence (aligned to OpenClaw troubleshooting guidance):

1. Run diagnostics and auto-fix:
- `openclaw doctor`
- `openclaw doctor --fix`

2. Validate channel/runtime health:
- `python scripts/openclaw_env.py gateway status --json`
- `python scripts/openclaw_env.py channels status`

3. Apply PMX maintenance healer:
- `python scripts/openclaw_maintenance.py --apply --primary-channel whatsapp --restart-gateway-on-rpc-failure --attempt-primary-reenable`

4. If listener remains down, relink WhatsApp:
- `openclaw channels login --channel whatsapp --account default --verbose`

Operational guardrails (backward-compatible defaults):
- Use PowerShell-native syntax (`$true`/`$false`), never Python booleans in `exec`.
- Do not nest `powershell -Command` inside an existing PowerShell shell.
- Avoid unbounded `while` loops in tool execution paths.
- For edit tools, send complete payloads: `path` + `oldText` + `newText` (or `new_string`).


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

OpenClaw send-path storm suppression is also enabled by default in `utils/openclaw_cli.py`:

- `OPENCLAW_STORM_GUARD_ENABLED=1` (default) enables persistent cross-process suppression.
- `OPENCLAW_STORM_BASE_COOLDOWN_SECONDS` (default: `60`) sets first failure cooldown.
- `OPENCLAW_STORM_MAX_COOLDOWN_SECONDS` (default: `1800`) caps exponential backoff.
- `OPENCLAW_STORM_BACKOFF_MULTIPLIER` (default: `2.0`) controls cooldown growth.
- `OPENCLAW_STORM_RESET_WINDOW_SECONDS` (default: `900`) resets failure streak after quiet time.
- Storm state persists in `OPENCLAW_PERSISTENT_GUARD_STATE_PATH` so process restarts do not reset suppression.

Inbox workflows have their own safe limits:

- `config/inbox_workflows.yml` -> `limits.*`
- Sending is disabled by default; enable with `PMX_INBOX_ALLOW_SEND=1` (or `limits.allow_send: true`).

See: `Documentation/INBOX_WORKFLOWS.md`

## Cron Automation (Updated 2026-02-17)

OpenClaw runs an audit-aligned cron set via `agentTurn` mode. The local LLM (qwen3:8b)
receives the cron message, prefers structured orchestrator tools (not generic exec chaining), and either
reports anomalies or responds `NO_REPLY` (suppressing the notification).

### Read-Only Robustness Jobs (2026-03-03)

Two additional read-only automation jobs now belong to the supported cron set:

- **Quality pipeline**  
  Runs `.\simpleTrader_env\Scripts\python.exe scripts\run_quality_pipeline.py --json`
  on a schedule and announces only `WARN`/`ERROR` states. `PASS` must return
  `NO_REPLY`.
- **Nightly training curation**  
  Runs `.\simpleTrader_env\Scripts\python.exe scripts\build_training_dataset.py --json`
  and announces only when curation fails closed, writes zero rows unexpectedly,
  or errors.

Operational rules:

- These jobs are advisory/read-only. They do not edit routing config or relax any
  gate thresholds.
- Cron payloads must use checked-in scripts only; no inline `python -c`
  maintenance logic for these tasks.
- Notification posture remains anomaly-only:
  - routine success -> `NO_REPLY`
  - partial-data / fail-closed / hard error -> concise announcement
- Canonical operator reference:
  - `Documentation/ROBUSTNESS_AUTOMATION_RUNBOOK.md`

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

### Multi-Agent Coordination (Safe Parallel Work)

If multiple developer-agents are active at the same time (OpenClaw + IDE agents + humans):

- Begin each task with a workspace reality check: `git status --porcelain`.
- Treat existing modified/untracked files as potentially owned by another active agent.
- Inspect shared-file diffs before editing (`git diff -- <file>`), then complement rather than overwrite.
- Keep commit scopes narrow and task-specific; avoid bundling unrelated parallel edits.
- Before finalization, run non-breaking verification (compile/smoke and fast pytest lane when feasible).
- Report what changed, what was intentionally left untouched, and exact verification commands.

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
- Enforces a cross-process maintenance lock (`logs/automation/openclaw_maintenance.lock.json`) so only one healing loop runs at a time.
- Checks gateway RPC health and can restart gateway on failure.
- Detects primary WhatsApp handshake timeout/disconnect states and restarts gateway for recovery.
- Re-checks channel status after restart so unresolved primary channel failures are never reported as `PASS`.
- Classifies DNS failures (`ENOTFOUND web.whatsapp.com`) separately from listener/session failures.
- Detects detached-listener conflicts (`rpc.ok=true` while service state is `Ready/Stopped` and port remains busy) and avoids restart storms.
- Uses configurable restart attempts (`--primary-restart-attempts`, default 2) for transient primary-channel recovery.
- Applies restart cooldown/backoff controls to reduce restart thrashing across watchdog/task overlaps.
- Marks unresolved primary channel failures as `FAIL` in maintenance reports to make outages explicit.
- Optionally toggles the primary account enabled=false/true and restarts gateway to recover a missing listener.
- Optionally disables broken non-primary channels (Telegram/Discord) when known auth/config errors persist.
- Parses recent WhatsApp channel logs for delivery/session signals (`closed session`, deferred recovery budget, retry wait inflation) and surfaces targeted manual actions.

Run once (apply mode):
- `python scripts/openclaw_maintenance.py --apply --disable-broken-channels --restart-gateway-on-rpc-failure --attempt-primary-reenable --recheck-delay-seconds 8 --primary-restart-attempts 1`

### Gateway Restart Throttling (Prevent Terminal Interruptions)

The maintenance watchdog can restart the OpenClaw gateway very aggressively
when channels are flapping. Use these env vars (set in `.env`) to throttle
restart frequency:

| Env Var | Default | Recommended | Effect |
|---------|---------|-------------|--------|
| `OPENCLAW_GATEWAY_RESTART_COOLDOWN_SECONDS` | 120 | **600** | Min gap between maintenance-triggered gateway restarts |
| `OPENCLAW_PRIMARY_REENABLE_COOLDOWN_SECONDS` | 300 | **600** | Min gap between disable/enable toggle remediation |
| `OPENCLAW_PRIMARY_RESTART_ATTEMPTS` | 2 | **1** | Restart attempts per incident before giving up |
| `OPENCLAW_FAST_SUPERVISOR_INTERVAL_SECONDS` | 5 | **60** | Fast supervisor probe frequency in watch mode |
| `OPENCLAW_FAST_SUPERVISOR_FAILURE_THRESHOLD` | 2 | **3** | Consecutive failed probes required before restart |
| `OPENCLAW_FAST_SUPERVISOR_RESTART_COOLDOWN_SECONDS` | 20 | **300** | Min gap between fast-supervisor-triggered restarts |

These are set as active defaults in `.env.template`. To apply:
```bash
# One-time apply after updating .env:
python scripts/openclaw_models.py apply --strategy local-first
# Then restart gateway once to pick up new cooldown state:
openclaw gateway restart
```

**Local LLM as main agent** — enforce qwen3:8b as the only primary model:
```bash
# Ensure OPENCLAW_LOCAL_ONLY=1 and OPENCLAW_MODEL_STRATEGY=local-first are in .env, then:
python scripts/openclaw_models.py apply --strategy local-first
# Verify primary is ollama/qwen3:8b with no cloud fallbacks:
python scripts/openclaw_models.py status
```

Cron path:
- `bash/production_cron.sh openclaw_maintenance`

Windows Task Scheduler helper:
- `powershell -ExecutionPolicy Bypass -File scripts/run_openclaw_maintenance.ps1`

Persistent gateway + WhatsApp watchdog (recommended on Windows):
- `powershell -ExecutionPolicy Bypass -File scripts/install_whatsapp_watchdog.ps1`
- Optional tuning: `powershell -ExecutionPolicy Bypass -File scripts/install_whatsapp_watchdog.ps1 -WatchIntervalSeconds 120 -EnsureIntervalMinutes 5`
- If you intentionally need older one-shot schedulers kept, use: `powershell -ExecutionPolicy Bypass -File scripts/install_whatsapp_watchdog.ps1 -KeepLegacyMaintenanceTask`
- Remove later: `powershell -ExecutionPolicy Bypass -File scripts/install_whatsapp_watchdog.ps1 -Uninstall`

The installer creates two idempotent tasks:
- `PMX-OpenClaw-Guardian-Logon` (start guardian on logon)
- `PMX-OpenClaw-Guardian-KeepAlive` (re-run guardian launcher every few minutes)

Important:
- Do not run legacy `PMX-OpenClaw-Maintenance` and guardian watch mode in parallel unless you explicitly accept overlap risk.
- The watchdog installer now disables `PMX-OpenClaw-Maintenance` by default to prevent restart lock/port contention loops.

### Verified Recovery Runbook (Non-Dry Install + Live WhatsApp E2E)

Use this runbook when WhatsApp appears connected but interactions are unreliable, or when overlapping scheduler tasks may be causing restart contention.

1. Apply watchdog installer in non-dry mode:
- `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_whatsapp_watchdog.ps1`

Expected:
- `PMX-OpenClaw-Guardian-KeepAlive` updated.
- `PMX-OpenClaw-Maintenance` disabled (or already disabled).
- `PMX-OpenClaw-Guardian-Logon` may warn `Access is denied` in non-elevated shells; rerun elevated if task creation/update is required.

2. Verify scheduler state:
- `Get-ScheduledTask -TaskName 'PMX-OpenClaw-Guardian-KeepAlive','PMX-OpenClaw-Guardian-Logon','PMX-OpenClaw-Maintenance' | Select-Object TaskName,State,@{Name='Enabled';Expression={$_.Settings.Enabled}} | Format-Table -AutoSize`

Expected:
- `PMX-OpenClaw-Guardian-KeepAlive`: `Ready`, `Enabled=True`
- `PMX-OpenClaw-Guardian-Logon`: `Ready`, `Enabled=True`
- `PMX-OpenClaw-Maintenance`: `Disabled`, `Enabled=False`

3. Verify gateway and WhatsApp runtime health:
- `python scripts/openclaw_env.py gateway status --json`
- `openclaw channels status --json`

Expected:
- `gateway.rpc.ok=true`
- WhatsApp account `running=true`, `connected=true`, `lastError=null`

4. Execute live outbound send:
- `openclaw message send --channel whatsapp --target +2347042437712 --message "[PMX E2E <timestamp>] Reply with ACK" --json`

Capture:
- `messageId`
- `toJid`

5. Prove inbound path (end-to-end):
- Note: `openclaw message read --channel whatsapp` is not supported.
- Correlate logs directly:
- `openclaw channels logs --channel whatsapp --lines 120`
- `Select-String -Path C:\tmp\openclaw\openclaw-YYYY-MM-DD.log -Pattern 'Sent message <MESSAGE_ID>|\"body\":\"ACK\"|web-auto-reply'`

Expected proof points:
- Outbound line with `Sent message <MESSAGE_ID> -> ...`
- Inbound ACK line (`"body":"ACK"` or equivalent inbound text)
- `web-auto-reply` line showing reply context that references the same `<MESSAGE_ID>` (for strict linkage)

6. Confirm post-send receive activity via channel timestamps:
- `openclaw channels status --json`
- Check WhatsApp account timestamps:
- `lastInboundAt` should advance after your send window.
- `lastError` should remain null.

If checks fail:
- Gateway not healthy: `openclaw gateway restart`
- WhatsApp disconnected/handshake timeout: `openclaw channels login --channel whatsapp --account default --verbose`
- No inbound after repeated sends: inspect `C:\tmp\openclaw\openclaw-YYYY-MM-DD.log` for session-closure/decryption and deferred-recovery signals, then run:
- `python scripts/openclaw_maintenance.py --apply --primary-channel whatsapp --restart-gateway-on-rpc-failure --attempt-primary-reenable`

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

## 3-Model Local LLM Strategy (Updated 2026-02-18)

Portfolio Maximizer uses three specialized local models via Ollama, each with a
distinct role in the inference pipeline:

| Model | Role | Capabilities | VRAM | Speed |
|-------|------|-------------|------|-------|
| **qwen3:8b** | Primary orchestrator | Tool-calling, structured output, workflow control | 5.5GB | 30-40 tok/s |
| **deepseek-r1:8b** | Fast reasoning delegate | Chain-of-thought, math, code-gen | 5.5GB | 25-35 tok/s |
| **deepseek-r1:32b** | Heavy reasoning delegate | Deep multi-step analysis, long-context | 20GB | 10-15 tok/s |

### Task-to-Model Routing

| Task | Model |
|------|-------|
| Interactive OpenClaw tasks, tool/function calls, API orchestration | qwen3:8b |
| Fast delegated reasoning (analysis, signal checks, concise audits) | deepseek-r1:8b via `scripts/deepseek_reason.py` |
| Heavy delegated reasoning (deep audits, long-context decomposition) | deepseek-r1:32b via `scripts/deepseek_reason.py --model deepseek-r1:32b` |

### How It Works

**qwen3:8b** acts as the orchestrator: it receives requests via OpenClaw
channels (WhatsApp/Telegram/Discord) and delegates reasoning tasks to
deepseek-r1 models through `exec` calls to `scripts/deepseek_reason.py`.

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
OPENCLAW_OLLAMA_MODEL_ORDER=qwen3:8b,deepseek-r1:8b,deepseek-r1:32b
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

## Multi-Agent Architecture (Updated 2026-02-18)

Portfolio Maximizer uses 4 dedicated OpenClaw agents with isolated workspaces,
sessions, and tool sandboxes. This eliminates the "stuck session" pattern
caused by queue contention in a single `agent:main:main` lane.

Reference: https://docs.openclaw.ai/concepts/multi-agent

### Agent Definitions

| Agent | Role | Tools Profile | Key Restrictions |
|-------|------|---------------|------------------|
| **ops** (default) | System health, cron jobs, general queries | `full` | None (full access) |
| **trading** | PnL monitoring, signal quality, execution status | explicit allow/deny | No write/edit/messaging |
| **training** | Model training, backtesting, heavy analysis | explicit allow/deny | No messaging/browser |
| **notifier** | Alert delivery to WhatsApp/Telegram | `messaging` | No exec/write/edit/fs |

`trading` and `training` intentionally avoid the built-in OpenClaw
`coding` profile on current 2026.2.26 builds. PMX uses explicit allow/deny
tool policy there instead, which preserves the same restrictions without the
false `tools.profile (coding) ... unknown entries (apply_patch)` warning.

### Routing (Bindings)

Bindings use deterministic most-specific-first matching:

| Channel | Account | Agent |
|---------|---------|-------|
| WhatsApp | default | ops |
| Telegram | default | notifier |
| Discord | custom-1 | ops |
| (fallback) | * | ops (default) |

Trading and training agents have no channel bindings -- they are triggered
by cron jobs with explicit `agentId` overrides, not by inbound messages.

### Per-Agent Workspaces

Each agent has an isolated workspace with a tailored SOUL.md:

```
~/.openclaw/agents/ops/agent/          # ops state (agentDir)
<project-root>/                         # ops workspace (full project access)

~/.openclaw/agents/trading/agent/       # trading state
~/.openclaw/workspace-trading/          # trading workspace (SOUL.md only)

~/.openclaw/agents/training/agent/      # training state
~/.openclaw/workspace-training/         # training workspace (SOUL.md only)

~/.openclaw/agents/notifier/agent/      # notifier state
~/.openclaw/workspace-notifier/         # notifier workspace (SOUL.md only)
```

### Tool Sandboxing

Per-agent `tools.allow`/`tools.deny` lists enforce least-privilege:

- **ops**: Full tool access (`profile: full`). Runs cron jobs, health checks.
- **trading**: Read + exec only. Cannot modify code/config. Can query PnL DB.
- **training**: Full fs + runtime. Cannot send messages. Runs training pipelines.
- **notifier**: Messaging + web only. Sandboxed (`mode: all`). Cannot exec or write files.

### agentToAgent

Disabled by default. Agents do not communicate directly -- they share state
through the filesystem (data/portfolio_maximizer.db, logs/, etc.).

Enable only if explicit handoffs are needed:
```json
{ "tools": { "agentToAgent": { "enabled": true, "allow": ["ops", "trading"] } } }
```

### Loop Detection

Optional and version-dependent. Some OpenClaw builds reject this key.
Check support first with:
- `openclaw doctor`
- `openclaw doctor --fix` (if unknown keys are reported)

Enable only when your installed OpenClaw build supports it:
```json
{ "tools": { "loopDetection": { "enabled": true, "warningThreshold": 5, "criticalThreshold": 10 } } }
```

### Cron Job Agent Assignment

Assign cron jobs to specific agents to isolate workloads:

| Cron Job | Agent | Rationale |
|----------|-------|-----------|
| PnL Integrity Audit (4h) | trading | Isolated PnL context |
| Production Gate Check (daily) | trading | Trading-specific |
| Quant Validation Health (daily) | trading | Signal quality |
| Signal Linkage Monitor (daily) | trading | Trade linkage |
| Ticker Health Monitor (daily) | trading | Per-ticker PnL |
| Quality Pipeline (daily) | trading | Read-only eligibility/context/chart refresh |
| Nightly Training Curation (nightly) | training | Read-only curated dataset build |
| GARCH Unit-Root Guard (weekly) | training | Model diagnostics |
| Overnight Hold Monitor (weekly) | training | Strategy analysis |
| System Health Check (6h) | ops | General health |
| Weekly Session Cleanup (Sun 3AM) | ops | Maintenance |

### Verification

```bash
python scripts/verify_openclaw_config.py   # Validates multi-agent config
python scripts/openclaw_models.py status    # Model availability
```

---

## Skills for Portfolio Maximizer & Remote WhatsApp Development

OpenClaw 2026.2.19-2 ships a skill system (`openclaw skills`) that exposes
reusable workflows to agents. PMX uses two workspace skills (loaded from the
repo) and several bundled skills. The `clawhub` CLI can install additional
skills from the public registry.

### Currently Ready Skills (`openclaw skills`)

| Skill | Type | Status | Purpose |
|-------|------|--------|---------|
| `portfolio-maximizer` | workspace | ready | PMX-specific commands: gate checks, PnL audits, quant health |
| `pmx-inbox` | workspace | ready | Gmail/Proton inbox workflows for trade alerts |
| `coding-agent` | bundled | ready | Delegate coding tasks to Claude Code / Codex in background |
| `skill-creator` | bundled | ready | Create or update AgentSkills with scripts and references |
| `healthcheck` | bundled | ready | System health checks (model availability, gateway status) |
| `discord` | bundled | ready | Discord channel interactions |
| `weather` | bundled | ready | Weather lookup (useful for correlation context) |

Workspace skills are loaded automatically when your OpenClaw workspace points
at the repo root. Bundled skills are pre-installed with OpenClaw.

### Installing Additional PMX-Relevant Skills via ClawHub

Some useful skills are not pre-installed. Install them with `npx clawhub`:

```bash
# Search for available skills
npx clawhub search github
npx clawhub search tmux
npx clawhub search session-logs
npx clawhub search gh-issues

# Install individually
npx clawhub install github        # GitHub PR/issue management from agent
npx clawhub install tmux          # Terminal session management (remote dev)
npx clawhub install session-logs  # OpenClaw session log inspection
npx clawhub install gh-issues     # Create/triage GitHub issues via WhatsApp

# Sync all installed skills to latest versions
npx clawhub sync

# Check installed skill versions
npx clawhub update --check
```

After installing, reload skills in a running gateway:
```bash
openclaw skills reload
# or restart gateway
openclaw gateway restart
```

### Remote Development via WhatsApp + coding-agent

The `coding-agent` skill lets you send coding tasks from WhatsApp and have
them executed by Claude Code or Codex running as a background process on your
dev machine. This is the primary remote development path for PMX.

**Workflow:**

```
WhatsApp message
  -> OpenClaw Gateway (ops agent)
    -> coding-agent skill activated
      -> spawns Claude Code / Codex subprocess in project root
        -> edits files, runs tests, produces output
      -> result delivered back to WhatsApp
```

**Example WhatsApp prompts:**

```
"Use coding-agent to fix the 2 failing tests in tests/scripts/"
"coding-agent: add --exclude-mode flag to check_quant_validation_health.py"
"coding-agent: run pytest tests/ --tb=short -q and report failures"
"coding-agent: check git status and summarize staged changes"
```

**Important routing note**: `coding-agent` runs under the `ops` agent
(WhatsApp binding). It has full tool access and workspace access to the repo
root. Tasks involving model training should be prefixed with "use training
agent" to route to the `training` agent instead (via explicit agentId).

**Safety guardrails for remote coding tasks:**

- Prompts are prefixed with `[PMX_AUTONOMY_POLICY]` (autonomy guard, see above).
- High-risk actions (force push, drop table, delete branch) require
  `PMX_APPROVE_HIGH_RISK` token in the message.
- Git push requires explicit confirmation -- the agent will ask before pushing.
- Keep `OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS=1` for unattended sessions.

**Checking background coding task status:**

```bash
# From local terminal -- check what coding-agent spawned
openclaw sessions list
openclaw sessions tail <session-id>
```

Or from WhatsApp:
```
"What is the status of the last coding-agent task?"
"Show me the last 20 lines of the active coding session"
```

### skill-creator: Creating New PMX Skills

Use `skill-creator` to create or update workspace skills from WhatsApp:

```
"Use skill-creator to add a new skill for running the adversarial test suite"
"skill-creator: update portfolio-maximizer skill to include quant_validation_headroom command"
```

Skills are written to `skills/<name>/SKILL.md` in the repo. Run
`openclaw skills reload` after creation.

### Per-Agent Skill Routing

Skills are available to all agents by default, but PMX restricts by role:

| Skill | Best Agent | Reason |
|-------|-----------|--------|
| `portfolio-maximizer` | ops, trading | Gate checks, PnL reporting, quant health |
| `pmx-inbox` | ops | Inbox workflows need messaging + exec |
| `coding-agent` | ops | Full tool access required for code execution |
| `skill-creator` | ops | Write access to `skills/` directory |
| `github` (if installed) | ops | PR/issue management needs write access |
| `tmux` (if installed) | ops, training | Terminal session control for long jobs |
| `session-logs` (if installed) | ops | Read-only session inspection |
| `gh-issues` (if installed) | ops | Issue filing needs GitHub write |
| `healthcheck` | ops | System-wide health checks |
| `discord` | notifier | Discord delivery only |

To restrict a skill to specific agents, add `allowedAgents` to the skill's
`SKILL.md` front-matter (if OpenClaw version supports it):
```yaml
---
name: coding-agent
allowedAgents: [ops]
---
```

### Checking Skill Status & Troubleshooting

```bash
# List all skills and their status
openclaw skills

# Check which skills are ready vs missing
openclaw skills --filter ready
openclaw skills --filter missing

# Reload after workspace skill changes
openclaw skills reload

# View a specific skill definition
openclaw skills show portfolio-maximizer

# If a skill fails to activate (bundled skill shows "missing"):
npx clawhub sync                   # sync all installed skills
openclaw doctor --fix              # fix config schema issues
openclaw gateway restart           # reload skill registry
```

**From WhatsApp:**
```
"List available skills"
"Show me the portfolio-maximizer skill commands"
"Reload skills"
```
