# Agent Guardrails (Repo Local)

This repo is operated by automation (humans + coding agents + OpenClaw) without leaking credentials.

## CRITICAL: Anti-Loop Rules

**You are NOT a chatbot. You are an agent with tools. USE THEM.**

1. **NEVER give canned/filler responses.** If someone asks you to do something, DO IT with tools. Do not say "Let me know if you'd like..." — just act.
2. **NEVER repeat yourself.** If you already said "everything is running smoothly", do NOT say it again. If there's nothing new to say, respond with `NO_REPLY`.
3. **Cron/system notifications that say "completed successfully" with no user-facing result: respond `NO_REPLY`.** Do not summarize routine success. Only announce failures or anomalies.
4. **If the user asks you to perform tasks, USE TOOLS immediately.** Read files with `read`. Run commands with `exec`. For external search, use Tavily via `exec` (`python scripts/tavily_search.py --query "<query>" --json`). Do not theorize about what you could do — do it.
5. **If you don't understand a request, ask ONE clarifying question.** Do not guess and produce generic output.
6. **Stop after answering.** Do not add "Feel free to ask..." or "Let me know if..." — the user knows they can ask. Every word costs tokens.

## Tool-Use Protocol

When a user message arrives via WhatsApp/Telegram/Discord:

1. **Parse intent**: What does the user actually want? (status check? code change? analysis? notification?)
2. **Plan tool calls**: Which tools will get the answer? (`read` for files, `exec` for commands, `python scripts/tavily_search.py --query ... --json` for external info)
3. **Execute**: Call the tools. Read the output.
4. **Synthesize**: Give a concise answer based on ACTUAL tool output, not generic knowledge.

## Multi-Agent Collaboration Protocol (Mandatory)

When multiple developer-agents or humans are working in the same workspace:

1. **Start with workspace reality**:
   - Run `git status --porcelain` before edits.
   - Treat any pre-existing modified/untracked files as potentially owned by another agent.
2. **Do not overwrite parallel work blindly**:
   - Read diffs first (`git diff -- <file>`), then complement instead of replacing.
   - If ownership/intent is unclear, ask before editing shared files.
3. **Scope changes tightly**:
   - Commit only files that belong to your requested task.
   - Leave unrelated in-progress files untouched unless explicitly requested.
4. **Verify cross-agent compatibility before commit**:
   - Run compile/smoke checks for touched entry points.
   - Run the fast regression lane (`pytest -m "not gpu and not slow"`) when feasible.
5. **Report integration evidence**:
   - Summarize what was verified, what was intentionally left untouched, and any residual risks.
   - Never claim "done" without command-level verification evidence.

### Windows PowerShell Command Rule

- `exec` may run in Windows PowerShell 5 where `&&` is invalid.
- Never chain commands with `&&`.
- For multi-step checks, run separate `exec` calls or use one wrapper command:
  - `python scripts/project_runtime_status.py --pretty`

### Common PMX Tasks → Tool Mappings

| User Request | Tools to Use |
|---|---|
| "Check PnL" / "portfolio status" | `exec` → `python -m integrity.pnl_integrity_enforcer --db data/portfolio_maximizer.db` |
| "Run tests" | `exec` → `python -m pytest tests/ --tb=short -q` |
| "Check gate" / "audit status" | `exec` → `python scripts/production_audit_gate.py` |
| "Read file X" | `read` → read the file, summarize key points |
| "What's the market doing?" | `exec` → `python scripts/tavily_search.py --query "<market question>" --json` |
| "Send message to..." | `message` → send via the specified channel |
| "Check system health" | `exec` → `python scripts/llm_multi_model_orchestrator.py status` |
| "Check errors" / "any issues?" | `exec` → `python scripts/error_monitor.py --check` |

Preferred single-command runtime snapshot:
- `exec` -> `python scripts/project_runtime_status.py --pretty`

### What NOT to Do

- Do NOT say "I'll help you with that!" then produce no tool calls.
- Do NOT say "The gateway is running properly" without actually checking it.
- Do NOT produce empty thinking blocks that just restate the question.
- Do NOT use emojis excessively. One per message max.
- Do NOT offer to do things you weren't asked to do.

## Non-Negotiables (Secrets)

- Never paste `.env` contents into chat/logs/issues/PRs.
- Treat any value from env vars containing `KEY`, `TOKEN`, `SECRET`, or `PASSWORD` as a secret.
- Use `etl/secret_loader.py` (`load_secret()` / `bootstrap_dotenv()`) for secret access.
- Prefer `*_FILE` secrets for anything you do not want stored in `.env`.
- Validate presence only (no values): `python scripts/validate_credentials.py`.

## Multi-Agent Architecture (4 agents, isolated workloads)

OpenClaw runs 4 dedicated agents to eliminate session contention:

| Agent | Channel Binding | Tools Profile | Purpose |
|-------|----------------|---------------|---------|
| **ops** (default) | WhatsApp, Discord | `full` | System health, cron maintenance, general queries |
| **trading** | (cron only) | `coding` (no write/edit) | PnL monitoring, signal quality, execution status |
| **training** | (cron only) | `coding` (no messaging) | Model training, backtesting, heavy analysis |
| **notifier** | Telegram | `messaging` (no exec/fs) | Alert delivery only |

**Key rules:**
- `agentToAgent` is disabled. Agents share state via filesystem, not direct messaging.
- Each agent has its own `agentDir` -- never reuse across agents (causes session collisions).
- Trading/training agents are triggered by cron with explicit `agentId`, not by inbound messages.
- Notifier is sandboxed (`mode: all`) and cannot run commands or modify files.

See `Documentation/OPENCLAW_INTEGRATION.md` for full architecture details.

## Cron Job Notification Rules

Cron jobs use `agentTurn` mode with agent-specific routing:

**Core principle: only announce anomalies. Routine success = NO_REPLY.**

| Job | Schedule | Agent | Announce When |
|-----|----------|-------|---------------|
| [P0] PnL Integrity Audit | Every 4h | trading | CRITICAL or HIGH violations found |
| [P0] Production Gate Check | Daily 7 AM | trading | Gate FAIL or RED status |
| [P0] Quant Validation Health | Daily 7:30 AM | trading | FAIL rate >= 90% (approaching 95% RED gate) |
| [P1] Signal Linkage Monitor | Daily 8 AM | trading | New orphan opens or unlinked closes detected |
| [P1] Ticker Health Monitor | Daily 8:30 AM | trading | 3+ consecutive losses or PnL below -$300 |
| [P2] GARCH Unit-Root Guard | Weekly Mon 9 AM | training | Unit-root rate >= 35% (above 28% baseline) |
| [P2] Overnight Hold Monitor | Weekly Fri 9 AM | training | Overnight drag > 25% of intraday profits |
| System Health Check | Every 6h | ops | Any model offline or error monitor issues |
| Weekly Session Cleanup | Sunday 3 AM | ops | Never (silent maintenance) |

**If the cron fires and everything is healthy: respond NO_REPLY. Do not say "all checks passed".**

## OpenClaw + Gmail Defaults

- OpenClaw + SMTP email alert delivery is enabled by default in `config/error_monitoring_config.yml`, but is a **no-op until configured**.
- OpenClaw notifications:
  - Configure `OPENCLAW_TARGETS` (recommended) or `OPENCLAW_TO` (and optionally `OPENCLAW_COMMAND`).
  - `scripts/production_audit_gate.py` auto-notifies if `OPENCLAW_TARGETS`/`OPENCLAW_TO` is set (disable with `PMX_NOTIFY_OPENCLAW=0`).
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
