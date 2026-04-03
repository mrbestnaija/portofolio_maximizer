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
| [OPENCLAW_DISCORD_TOKEN_AUDIT_2026-03-24.md](OPENCLAW_DISCORD_TOKEN_AUDIT_2026-03-24.md) | Sanitized Discord token audit, runtime evidence, and remediation steps for OpenClaw |
| [OBSERVABILITY_PROMETHEUS_GRAFANA.md](OBSERVABILITY_PROMETHEUS_GRAFANA.md) | Read-only Prometheus exporter, Alertmanager shadow routing, Loki/Alloy logs, Grafana ops dashboards, and Windows startup/provisioning |

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

Historical session summaries, status snapshots, and run logs.  All files here are
read-only records; do not edit archived files — append new entries in the canonical
planning files above instead.

### Session Notes

Completed session summaries archived after the rolling window in `SESSION_COMPLETE_SUMMARY.md`
moves on.

| File | Contents |
|------|----------|
| [history/sessions/SESSION_SUMMARY_2026_01_21.md](history/sessions/SESSION_SUMMARY_2026_01_21.md) | Phase 7.3 multi-session GARCH ensemble integration |

Add new entries to `history/sessions/` using the naming convention
`SESSION_SUMMARY_YYYY_MM_DD.md` before removing them from the rolling canonical record.

### Pipeline Run Logs

Reference records of notable pipeline runs (dry-run baselines and live execution evidence).

| File | Contents |
|------|----------|
| [history/run_logs/pipeline_live_run_log.md](history/run_logs/pipeline_live_run_log.md) | Live pipeline run evidence log |
| [history/run_logs/pipeline_dry_run_log.md](history/run_logs/pipeline_dry_run_log.md) | Dry-run / smoke-test baseline log |

Add new run logs to `history/run_logs/` using the naming convention
`pipeline_<mode>_run_log_YYYYMMDD.md`.

### Status Snapshots

| Path | Contents |
|------|----------|
| [history/status/](history/status/) | Phase-end status snapshots (e.g., PROJECT_STATUS_2026-01-23.md) |

---

## Compatibility Policy

Historical files that have been moved under `Documentation/history/` retain compatibility
stubs at their previous paths.  **Do not delete a stub until all inbound references
(cron jobs, agent instructions, external tooling) have been updated and validated.**

**Stub format** (insert at the top of the old file when moving it):

```markdown
> **[MOVED]** This document has been moved to
> [history/sessions/SESSION_SUMMARY_YYYY_MM_DD.md](history/sessions/SESSION_SUMMARY_YYYY_MM_DD.md).
> This stub will be removed once all inbound references are updated.
```

Stubs are reviewed quarterly; confirmed-dead stubs are removed in the next hygiene
cleanup pass after all referencing systems have been updated.

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
- [OPENCLAW_DISCORD_TOKEN_AUDIT_2026-03-24.md](OPENCLAW_DISCORD_TOKEN_AUDIT_2026-03-24.md) — Discord token troubleshooting and March 24 audit record
- [OBSERVABILITY_PROMETHEUS_GRAFANA.md](OBSERVABILITY_PROMETHEUS_GRAFANA.md) — Prometheus exporter, Alertmanager bridge, Loki/Alloy logs, Grafana ops dashboards, and Windows rollout contract
- [GIT_WORKFLOW.md](GIT_WORKFLOW.md) — Branching, commit, and PR workflow

### Metrics and Evaluation

- [METRICS_AND_EVALUATION.md](METRICS_AND_EVALUATION.md) — Canonical metric definitions
- [QUANTIFIABLE_SUCCESS_CRITERIA.md](QUANTIFIABLE_SUCCESS_CRITERIA.md)
- [PROFITABILITY_AUDIT_RESULTS.md](PROFITABILITY_AUDIT_RESULTS.md)

</details>
