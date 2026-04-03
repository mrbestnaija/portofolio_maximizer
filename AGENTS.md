# Agent Guardrails (Repo Local)

This repo is operated by automation (humans + coding agents + OpenClaw) without leaking credentials.

## CRITICAL: GitHub Actions Version Rules (enforced 2026-03-28)

All `.github/workflows/*.yml` files MUST use these exact action versions.
**Using a higher (non-existent) version silently breaks every CI run.**

| Action | Required version |
|--------|-----------------|
| `actions/checkout` | `@v4` |
| `actions/setup-python` | `@v5` |
| `actions/cache` | `@v4` |
| `actions/upload-artifact` | `@v4` |
| `actions/download-artifact` | `@v4` |

Before committing any workflow file: `grep -r "uses: actions/" .github/workflows/`
and verify every version matches the table. If unsure of a version, leave it unchanged.

**Root cause of 2026-03-28 CI outage**: `checkout@v6`, `setup-python@v6`, `cache@v5`,
`upload-artifact@v6` were used — none exist. All CI runs failed at checkout with zero
Python code executed. Fixed in commit `ef3a123`.

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
   - For external web lookup: **do not use native OpenClaw `web_search`** in this repo workflow. Always use Tavily via `exec`.
   - If the user explicitly says "`web_search`", map it to Tavily command execution and continue.
3. **Execute**: Call the tools. Read the output.
4. **Synthesize**: Give a concise answer based on ACTUAL tool output, not generic knowledge.

## Integration Gate Rule (Non-Negotiable)

**No agent-produced code may enter master without explicit approval and integration by the
human + Claude Code pair.**

```
Agent proposes change (code, config, doc)
        ↓
Agent commits to a personal branch or patch file
        ↓
Human reviews the diff via Claude Code
        ↓
Claude Code integrates what passes review; rejects or defers the rest
        ↓
Merged to master only after targeted tests pass
```

Agents do NOT self-merge. Agents do NOT commit to master directly.
If a commit appears on master from an agent without the review step, it will be reverted.

See `Documentation/AGENT_COORDINATION_PROTOCOL_2026-03-08.md` for:
- Task delegation priority ladder (P0–P3) and delegation rule
- Agent A / B / C domain boundaries and out-of-scope lists
- Shared file ownership table
- Integration checklist (6-point, required before any merge)
- Current verified system state

**Delegation summary (enforced by the protocol)**:

| Agent | Primary focus | Receives delegations of |
|---|---|---|
| **A** | Model accuracy, signal quality, gate wiring, PnL-impacting logic | Nothing — A delegates out |
| **B** | Infrastructure, reporting, audit compatibility, pipeline orchestration | P2 tasks from A (pandas fixes, reporting fields, pipeline steps) |
| **C** | Experiment planning, readiness tracking, measurement reporting | P3 tasks from A (docs, status tables, acceptance test stubs) |

Agent A must not use its compute on P2/P3 work. When A identifies a lower-priority task,
it raises it to B or C with a one-sentence spec and moves on.

---

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
   - For every new implementation, run at least one targeted unit test for touched modules and run the fast regression lane (`pytest -m "not gpu and not slow"`).
   - Treat test execution as required delivery evidence (not optional); if a lane cannot run, report why and list exact blockers.
5. **Report integration evidence**:
   - Summarize what was verified, what was intentionally left untouched, and any residual risks.
   - Never claim "done" without command-level verification evidence.

### Windows PowerShell Command Rule

- `exec` runs in Windows PowerShell 5 — no `&&`, no nested `powershell -Command`, use checked-in scripts (`python scripts/...`) not inline `-c` multiline strings.
- Single-command runtime snapshot: `exec` -> `python scripts/project_runtime_status.py --pretty`

### What NOT to Do

- Do NOT say "I'll help you with that!" then produce no tool calls.
- Do NOT say "The gateway is running properly" without actually checking it.
- Do NOT use native `web_search` that requests `BRAVE_API_KEY`; use `python scripts/tavily_search.py --query ... --json`.

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
| [P1] Platt Contract Audit | Weekly Mon 9:30 AM | training | Any FAIL finding (wrong classifier, broken fallback chain, bootstrap produced 0 closes, no active calibration tier) |
| [P1] Model Improvement Check | Daily 10 AM | training | Any WARN or FAIL layer (SKIP = neutral, never announce) |
| [P2] GARCH Unit-Root Guard | Weekly Mon 9 AM | training | Unit-root rate >= 35% (above 28% baseline) |
| [P2] Overnight Hold Monitor | Weekly Fri 9 AM | training | Overnight drag > 25% of intraday profits |
| System Health Check | Every 6h | ops | Any model offline or error monitor issues |
| Weekly Session Cleanup | Sunday 3 AM | ops | Never (silent maintenance) |

Platt cron command: `exec` -> `python scripts/platt_contract_audit.py --json`
Model improvement cron command: `exec` -> `.\\simpleTrader_env\\Scripts\\python.exe scripts\\check_model_improvement.py --json`

**If the cron fires and everything is healthy: respond NO_REPLY. Do not say "all checks passed".**

---

## Measurement + Contracts (Condensed)

- Run health stack:
  - `python scripts/check_model_improvement.py --json`
  - `python scripts/run_all_gates.py --json`
  - `python scripts/platt_contract_audit.py --json`
- **SKIP != PASS**. Never treat missing-data layers as green.
- Calibration contract (`tests/scripts/test_platt_calibration_contract.py`) is strict:
  - If `_calibrate_confidence()` classifier changes, update docs + test + audit constant in the same patch.
  - Bootstrap must prove non-zero useful output; exit code alone is insufficient.
  - HOLD entries must be excluded from pending-outcome starvation metrics.
  - DB calibration tier is primary; JSONL starvation alone is not a failure.
- Adversarial diagnostics:
  - `python scripts/adversarial_diagnostic_runner.py --json --severity LOW`
  - Exit codes: `0=clean`, `1=confirmed CRITICAL/HIGH`, `2=runner error`.
- Full rationale and historical details: `Documentation/PHASE_7.39_PARANOID_REVIEW.md` and `Documentation/CORE_PROJECT_DOCUMENTATION.md`.

---

## Institutional Hardening Baseline (Mandatory)

Before claiming unattended-run readiness, run and report all of:

1. `python scripts/institutional_unattended_gate.py --json`
2. `python scripts/run_all_gates.py --json`
3. `python -m pytest tests/scripts/test_institutional_unattended_contract.py tests/scripts/test_institutional_unattended_gate.py tests/scripts/test_llm_runtime_install_policy.py tests/scripts/test_platt_calibration_contract.py tests/scripts/test_run_all_gates.py -q`
4. `python -m pytest -m "not gpu and not slow" --tb=short -q`

Do not use skip flags (`--skip-forecast-gate`, `--skip-profitability-gate`, `--skip-institutional-gate`) as final evidence for readiness.

Current institutional contracts that must remain true:
- Runtime pip install is default-deny (`PMX_ALLOW_RUNTIME_PIP_INSTALL=1` required to enable installs).
- Prompt-injection blocking is default-on in autonomous paths.
- Forecast gate max-files default is shared via `scripts/audit_gate_defaults.py`.
- `platt_contract_audit.py` must run standalone (no manual `PYTHONPATH` setup).
- No tracked shadow duplicates (`Dockerfile (1)`, `execution/lob_simulator (1).py`).

## Ensemble Lift Governance

- Keep `disable_ensemble_if_no_lift=false` until fresh post-fix evidence clears governance conditions in `config/forecaster_monitoring.yml`.
- Do not flip fail-closed based on stale or insufficient windows.

---

## OpenClaw + Gmail Defaults

- OpenClaw notifications: set `OPENCLAW_TARGETS` or `OPENCLAW_TO`. Auto-notifies from `production_audit_gate.py` (disable with `PMX_NOTIFY_OPENCLAW=0`).
- Gmail/SMTP alerts: set `PMX_EMAIL_USERNAME`, `PMX_EMAIL_PASSWORD`, `PMX_EMAIL_TO`.
- Inbox scan: `python scripts/inbox_workflow.py scan` (read-only by default; enable send with `PMX_INBOX_ALLOW_SEND=1`).

## References

- `Documentation/OPENCLAW_INTEGRATION.md`
- `Documentation/AGENT_INSTRUCTION.md`
