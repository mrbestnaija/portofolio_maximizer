# Clean Cohort Operations (2026-03-14)

This note defines how to start and operate a clean evidence cohort without mixing
legacy production evidence into new readiness decisions.

References:
- [AGENT_COORDINATION_PROTOCOL_2026-03-08.md](./AGENT_COORDINATION_PROTOCOL_2026-03-08.md)
- [MOMENTUM_SAFE_EVIDENCE_COORDINATION.md](./MOMENTUM_SAFE_EVIDENCE_COORDINATION.md)

## Goal

Preserve historical production evidence while creating a fresh cohort window whose
identity, routing, and proof artifacts are immutable and separately reviewable.

## Cohort Shape

Each cohort lives under:

`logs/forecast_audits/cohorts/<cohort_id>/`

With these key paths:

- `production/` — canonical production-trade evidence for the cohort
- `research/` — optional research-only evidence for the cohort
- `cohort_identity.json` — frozen identity tuple and env contract
- `activate_clean_cohort.ps1` — PowerShell env activation snippet
- `production_gate_latest.json` — cohort-scoped production gate output
- `proof_loop_latest.json` — cohort-scoped proof-loop summary

## Freeze Procedure

Freeze the cohort before collecting new live paper evidence:

```powershell
python scripts/clean_cohort_manager.py freeze --cohort-id 2026Q1_cleanroom --json
```

This writes:

- immutable `cohort_id`
- `build_fingerprint`
- `contract_version`
- `routing_mode`
- `strategy_config_fingerprint`
- derived `contract_fingerprint`

If the same `cohort_id` is frozen again with a different fingerprint, the helper fails
closed unless `--force` is explicitly used.

## Activation

Use the generated activation script to bind the live paper run to the cohort:

```powershell
. .\logs\forecast_audits\cohorts\2026Q1_cleanroom\activate_clean_cohort.ps1
```

This sets:

- `PMX_EVIDENCE_COHORT_ID`
- `PMX_BUILD_FINGERPRINT`
- `TS_FORECAST_AUDIT_DIR`

## Daily Proof Loop

Run the cohort-scoped proof loop:

```powershell
python scripts/clean_cohort_manager.py proof-loop --cohort-id 2026Q1_cleanroom --json
```

This runs:

- `scripts/replay_trade_evidence_chain.py` against a cohort replay directory
- `integrity.pnl_integrity_enforcer`
- `scripts/production_audit_gate.py --audit-dir <cohort production dir> --output <cohort gate artifact>`

Optional:

```powershell
python scripts/clean_cohort_manager.py proof-loop --cohort-id 2026Q1_cleanroom --include-global-gates --json
```

Use `--include-global-gates` only when you want a repo-global snapshot in the same run.
The authoritative clean-cohort readiness result remains the cohort-scoped production gate,
not the shared global gate artifact.

## Promotion Rule

Do not promote the clean cohort to canonical readiness unless all of the following hold
for 5 consecutive business-day runs under a stable `contract_fingerprint`:

- `quarantined_records == 0`
- `duplicate_conflicts == 0`
- `orphan_closes == 0`
- `linked_closes / eligible_closes >= 0.95`
- `matched_closes / eligible_closes >= 0.90`
- no mixed-context admissions
- no partial-write promotion incidents
- no contract-version drift
- no cohort-fingerprint drift

## Integration Note

This workflow is intentionally additive. It avoids edits to the protected global gate
orchestration while the evidence-core lane is still under review.
