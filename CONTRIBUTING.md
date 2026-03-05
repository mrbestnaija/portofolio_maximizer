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
