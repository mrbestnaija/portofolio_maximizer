# Documentation Index — Portfolio Maximizer v45

**Purpose**: Canonical navigator for all project documentation.
**Last updated**: 2026-03-02 (Phase 7.25-7.31 complete)

> For the current phase status and next steps, start with [Planning and Status (Canonical)](#planning-and-status-canonical).

---

## Institutional Runbooks

Operational procedures for running the system in production/unattended modes.

| Document | Description |
|----------|-------------|
| [INSTITUTIONAL_UNATTENDED_RUN_GATE.md](INSTITUTIONAL_UNATTENDED_RUN_GATE.md) | Gate criteria and decision rules for unattended overnight execution |
| [INSTITUTIONAL_WORKFLOW_RUNBOOK.md](INSTITUTIONAL_WORKFLOW_RUNBOOK.md) | Step-by-step workflow for full institutional-grade run cycles |
| [PRODUCTION_SECURITY_AND_PROFITABILITY_RUNBOOK.md](PRODUCTION_SECURITY_AND_PROFITABILITY_RUNBOOK.md) | Security posture, adversarial checks, and profitability proof requirements |
| [CRON_AUTOMATION.md](CRON_AUTOMATION.md) | OpenClaw cron job definitions, schedules, and anomaly-announce rules |

---

## Core Standards

Reference specifications enforced across all phases and agents.

| Document | Description |
|----------|-------------|
| [AGENT_INSTRUCTION.md](AGENT_INSTRUCTION.md) | Agent behavioral guardrails and task execution standards |
| [AGENT_DEV_CHECKLIST.md](AGENT_DEV_CHECKLIST.md) | Pre-commit and pre-deploy checklist for all code changes |
| [QUANT_TIME_SERIES_STACK.md](QUANT_TIME_SERIES_STACK.md) | Canonical description of the TS forecasting stack (GARCH/SAMOSSA/MSSA-RL/ensemble) |
| [QUANT_VALIDATION_MONITORING_POLICY.md](QUANT_VALIDATION_MONITORING_POLICY.md) | Quant validation gate thresholds, FAIL/WARN/PASS rules, monitoring cadence |
| [REWARD_TO_EFFORT_INTEGRATION_PLAN.md](REWARD_TO_EFFORT_INTEGRATION_PLAN.md) | Sequencing heuristics — prioritize high-signal, low-effort improvements first |

---

## Planning and Status (Canonical)

Current state, next actions, and session records.  These three files are the primary
read target at the start of every session.

| Document | Description |
|----------|-------------|
| [PROJECT_STATUS.md](PROJECT_STATUS.md) | Current phase status, regression baseline, gate snapshot, pending tasks |
| [NEXT_TO_DO_SEQUENCED.md](NEXT_TO_DO_SEQUENCED.md) | Sequenced to-do list with reward-to-effort alignment |
| [SESSION_COMPLETE_SUMMARY.md](SESSION_COMPLETE_SUMMARY.md) | Rolling session record — most recent session at the top |

---

## Archived Historical Notes

Historical session summaries, status snapshots, and run logs.
New entries go into the appropriate subdirectory; do not edit archived files.

| Path | Contents |
|------|----------|
| [history/status/](history/status/) | Phase-end status snapshots (e.g., PROJECT_STATUS_2026-01-23.md) |
| [history/sessions/](history/sessions/) | Session summaries older than the current rolling window |
| [history/run_logs/](history/run_logs/) | Pipeline and audit run output archives |

---

## Compatibility Policy

When a document is moved or renamed, a stub is left at the previous path that redirects
to the new canonical location.  This prevents broken links in existing cron jobs, agent
references, and external tooling.

**Stub format** (insert at top of old file):

```markdown
> **[MOVED]** This document has been moved to [NEW_FILENAME.md](NEW_FILENAME.md).
> This stub will be removed after 2026-06-01.
```

Stubs older than 90 days are removed in the next hygiene cleanup pass.

---

## Full Document Listing

<details>
<summary>All documentation files (click to expand)</summary>

### Phase History

- [PHASE_7.3_COMPLETE.md](PHASE_7.3_COMPLETE.md) — GARCH ensemble integration
- [PHASE_7.3_COMPLETE_FIX.md](PHASE_7.3_COMPLETE_FIX.md)
- [PHASE_7.3_ENSEMBLE_FIX.md](PHASE_7.3_ENSEMBLE_FIX.md)
- [PHASE_7.3_FINAL_SUMMARY.md](PHASE_7.3_FINAL_SUMMARY.md)
- [PHASE_7.3_MULTI_TICKER_VALIDATION.md](PHASE_7.3_MULTI_TICKER_VALIDATION.md)
- [PHASE_7.4_COMPLETION_SUMMARY.md](PHASE_7.4_COMPLETION_SUMMARY.md) — Quantile calibration
- [PHASE_7.4_FINAL_SUMMARY.md](PHASE_7.4_FINAL_SUMMARY.md)
- [PHASE_7.5_VALIDATION.md](PHASE_7.5_VALIDATION.md) — Regime detection validation
- [PHASE_7.5_MULTI_TICKER_RESULTS.md](PHASE_7.5_MULTI_TICKER_RESULTS.md)
- [PHASE_7.7_FINAL_SUMMARY.md](PHASE_7.7_FINAL_SUMMARY.md)
- [PHASE_7.8_RESULTS.md](PHASE_7.8_RESULTS.md)

### Architecture and Design

- [CORE_PROJECT_DOCUMENTATION.md](CORE_PROJECT_DOCUMENTATION.md)
- [ENSEMBLE_MODEL_STATUS.md](ENSEMBLE_MODEL_STATUS.md) — Canonical ensemble policy labels
- [FORECASTING_IMPLEMENTATION_SUMMARY.md](FORECASTING_IMPLEMENTATION_SUMMARY.md)
- [MODEL_SIGNAL_REFACTOR_PLAN.md](MODEL_SIGNAL_REFACTOR_PLAN.md)
- [QUANT_TIME_SERIES_STACK.md](QUANT_TIME_SERIES_STACK.md)

### Integrity and Audit

- [ADVERSARIAL_AUDIT_20260216.md](ADVERSARIAL_AUDIT_20260216.md) — Original 10-finding audit
- [DEEP_AUDIT_SPRINT_INVESTIGATION.md](DEEP_AUDIT_SPRINT_INVESTIGATION.md)
- [EXIT_ELIGIBILITY_AND_PROOF_MODE.md](EXIT_ELIGIBILITY_AND_PROOF_MODE.md)
- [NUMERICAL_STABILITY_AUDIT_REPORT.md](NUMERICAL_STABILITY_AUDIT_REPORT.md)
- [NUMERIC_INVARIANTS_AND_SCALING_TESTS.md](NUMERIC_INVARIANTS_AND_SCALING_TESTS.md)
- [PNL_ROOT_CAUSE_AUDIT.md](PNL_ROOT_CAUSE_AUDIT.md)

### Operations

- [BOOTSTRAP.md](BOOTSTRAP.md) — Agent onboarding guide
- [HEARTBEAT.md](HEARTBEAT.md) — System status / active sessions
- [OPENCLAW_INTEGRATION.md](OPENCLAW_INTEGRATION.md) — OpenClaw cron + LLM orchestration
- [GIT_WORKFLOW.md](GIT_WORKFLOW.md) — Branching, commit, and PR workflow

### Metrics and Evaluation

- [METRICS_AND_EVALUATION.md](METRICS_AND_EVALUATION.md) — Canonical metric definitions
- [QUANTIFIABLE_SUCCESS_CRITERIA.md](QUANTIFIABLE_SUCCESS_CRITERIA.md)
- [PROFITABILITY_AUDIT_RESULTS.md](PROFITABILITY_AUDIT_RESULTS.md)

</details>
