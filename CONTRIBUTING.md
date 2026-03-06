# Contributing Guide

## Evidence Integrity Contract

This repo treats telemetry and monitoring semantics as a hard contract.

### Non-negotiable rules

- Telemetry labels MUST match reality. No metric name may imply outcome matching unless realized outcomes were actually joined and validated.
- Corrupted evidence MUST fail closed. Malformed or non-parseable execution JSONL must never be reported as loaded/healthy.
- Eligibility MUST be timezone-aware datetime logic. Date-only truncation (for example `.date()`) is not allowed for eligibility gates.
- Dashboard status MUST preserve maximum severity from backend status signals and must never downplay a worse backend state.
- Cache and sidecar write failures MUST be visible as warnings/errors; silent swallow is not allowed.

### Definition of telemetry change

A change is a telemetry change if it does any of the following:

- Adds, removes, renames, or repurposes telemetry keys/metrics.
- Changes denominator/meaning semantics of an existing metric.
- Changes status or integrity-state semantics.
- Changes sidecar wiring between producer, bridge, and dashboard UI.

### Mandatory telemetry-change checklist

Every telemetry change PR MUST include all items below:

1. Bump `telemetry_contract.schema_version` in `config/telemetry_contract.yml`.
2. Update adversarial diagnostics/tests for changed semantics.
3. Include verification evidence in PR notes:
   - Targeted pytest output.
   - `python scripts/adversarial_diagnostic_runner.py --json --severity LOW --fix-report` output.

### Hard fail conditions

CI/review is considered failed when any of these occur:

- Telemetry changed without schema version bump.
- Schema version bumped without adversarial coverage update.
- UI shows a less severe state than backend `overall_status`.

---

## Experiment Integrity Contract

All trading experiments must comply with the
[Research-Grade Experiment Protocol v3](Documentation/RESEARCH_EXPERIMENT_PROTOCOL.md).

### Anti-gaming safeguards (enforced)

The following practices are **explicitly rejected**:

- Exclude losing trades from denominators without a causal reason documented in the experiment log.
- Include `NON_TRADE_CONTEXT`, `INVALID_CONTEXT`, `NOT_DUE`, or `RESEARCH` windows in outcome denominators.
- Declare lift when the 95% bootstrap CI for the improvement crosses zero.
- Change gate thresholds in `config/forecaster_monitoring.yml` or `config/quant_success_config.yml` as part of an experiment.
- Use execution telemetry older than 24 hours as evidence of system health.
- Report `profit_factor` improvement when the post-experiment value is below 1.0.

### Strategy freeze rule

Experiments in Phases 1–3 (evidence stabilization, linkage coverage, evaluation completeness)
**must not modify** exit rules, risk caps, model weights, position sizing, or trade filtering
logic. Strategy changes are only permitted from Phase 5 onward and only after Phase-4
attribution analysis is complete.

### Experiment prerequisite

No Phase-5 experiment may run until:
1. `execution_log_freshness <= 24h`
2. `high_integrity_violation_count = 0`
3. `outcome_matched >= 10` with `matched/eligible >= 0.80`
4. `evaluation_metrics_coverage >= 80%`
5. Phase-4 attribution report generated
