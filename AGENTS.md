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
   - For external web lookup: **do not use native OpenClaw `web_search`** in this repo workflow. Always use Tavily via `exec`.
   - If the user explicitly says "`web_search`", map it to Tavily command execution and continue.
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
| [P2] GARCH Unit-Root Guard | Weekly Mon 9 AM | training | Unit-root rate >= 35% (above 28% baseline) |
| [P2] Overnight Hold Monitor | Weekly Fri 9 AM | training | Overnight drag > 25% of intraday profits |
| System Health Check | Every 6h | ops | Any model offline or error monitor issues |
| Weekly Session Cleanup | Sunday 3 AM | ops | Never (silent maintenance) |

Platt cron command: `exec` -> `python scripts/platt_contract_audit.py --json`

**If the cron fires and everything is healthy: respond NO_REPLY. Do not say "all checks passed".**

---

## Implementation Contract Rules

**Rules enforced by `tests/scripts/test_platt_calibration_contract.py` and `scripts/platt_contract_audit.py`.**

These exist because two categories of silent failure were observed:

### Rule P-1: Never claim a calibration method without test coverage

If you change the classifier used in `_calibrate_confidence()`:
1. Update `PHASE_7.14_GATE_RECALIBRATION.md` to name the new method exactly
2. Update `test_platt_calibration_contract.py::TestClassifierIdentity` to match
3. Update `platt_contract_audit.py::EXPECTED_CLASSIFIER` constant
4. DO NOT leave docs claiming the old method -- mismatches are caught as CI failures

**Current contract**: `_calibrate_confidence()` uses `sklearn.linear_model.LogisticRegression`
(classic Platt scaling). NOT isotonic. NOT CalibratedClassifierCV. Any change must update all three targets above.

### Rule P-2: Bootstrap steps must verify output, not just exit code

If a bootstrap step exits 0 but produced zero useful output (e.g., 0 closed trades,
0 matched pairs), the pipeline MUST fail loudly. Silent success on zero output is a
first-class bug -- it conceals broken bootstrap design (wrong cycle count, missing
`--resume`, same-bar-repeat, etc.).

**Mechanism**: `run_overnight_refresh.py` counts `ts_* is_close=1` trades before and
after bootstrap. Delta == 0 increments `errors` and logs `[FAIL]`. The
`test_platt_calibration_contract.py::TestBootstrapOutcomeGuard` tests assert this guard
exists in code.

### Rule P-3: HOLD signals must not inflate reconciliation starvation metrics

`quant_validation.jsonl` logs ALL signal evaluations including HOLD actions. HOLD
signals structurally cannot produce `is_close=1` trades and therefore can never reconcile.
Counting them as "pending" makes starvation metrics look worse than they are.

**Mechanism**: `update_platt_outcomes.py` filters HOLD entries before building the
`pending` list. The `hold_skipped=N` field in the summary line makes the count explicit.

### Rule P-4: DB-based calibration is primary; JSONL is supplementary

`_calibrate_confidence()` has three tiers: JSONL -> DB-local -> DB-global. The JSONL
tier is frequently starved (logs HOLDs, unexecuted signals, legacy IDs). The DB tier
queries actual executed+closed trades directly and is more reliable.

**Do not treat JSONL starvation as a calibration failure.** Check `platt_contract_audit.py`
output for `calibration_active_tier` to see which tier is actually active.

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

## OpenClaw + Gmail Defaults

- OpenClaw notifications: set `OPENCLAW_TARGETS` or `OPENCLAW_TO`. Auto-notifies from `production_audit_gate.py` (disable with `PMX_NOTIFY_OPENCLAW=0`).
- Gmail/SMTP alerts: set `PMX_EMAIL_USERNAME`, `PMX_EMAIL_PASSWORD`, `PMX_EMAIL_TO`.
- Inbox scan: `python scripts/inbox_workflow.py scan` (read-only by default; enable send with `PMX_INBOX_ALLOW_SEND=1`).

## References

- `Documentation/OPENCLAW_INTEGRATION.md`
- `Documentation/AGENT_INSTRUCTION.md`
