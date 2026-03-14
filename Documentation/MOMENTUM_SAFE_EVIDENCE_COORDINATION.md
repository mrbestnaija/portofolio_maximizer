# Momentum-Safe Evidence Coordination

This document complements [AGENT_COORDINATION_PROTOCOL_2026-03-08.md](./AGENT_COORDINATION_PROTOCOL_2026-03-08.md)
for the evidence-recovery lane. Its purpose is to let multiple agents work side-by-side
without silently corrupting readiness evidence.

## Merge Order

Apply changes in this order, one concern per patch:

1. Shared evidence IO helpers
2. Audit routing and stamped-write fixes
3. Shadow `v2` lineage and semantic admission metadata
4. Audit summarizer / gate-readable counters
5. Replay and fault-injection coverage
6. Cohort promotion rules

Do not mix strategy changes with evidence-contract changes in the same patch.

## Cohort Identity

A clean cohort is identified by the immutable tuple:

- `cohort_id`
- `build_fingerprint`
- `contract_version`
- `routing_mode`
- `strategy_config_fingerprint`

The tuple is serialized into a `contract_fingerprint`. A cohort is promotion-eligible
only if the fingerprint is stable across the qualifying window.

## Semantic Admission Contract

A production artifact may exist in audit history without being readiness-eligible.

Buckets:

- `ELIGIBLE`: preserved and may count toward readiness
- `ACCEPTED_NONELIGIBLE`: preserved for audit history only
- `QUARANTINED`: preserved outside the canonical path and excluded from readiness

Write success does not imply gate eligibility. Gate consumers must treat storage and
readiness as separate decisions.

An artifact is admissible for readiness only if it is:

- schema-valid
- manifest-registered
- production-labeled
- `context_type == TRADE`
- causally complete for its stage
- not quarantined
- not superseded by a duplicate conflict

Minimum counters to publish for each cohort window:

- `accepted_records`
- `accepted_noneligible_records`
- `eligible_records`
- `eligible_opens`
- `eligible_closes`
- `linked_closes`
- `matched_closes`
- `orphan_closes`
- `quarantined_records`
- `duplicate_conflicts`

## Duplicate and Supersession Policy

- Identical duplicate payload for the same logical file: safe no-op
- Same logical ID with different payload: quarantine the conflicting artifact and raise an integrity signal
- Later correction event: append a correction artifact; do not rewrite historical canonical evidence

## Routing and Fail-Closed Behavior

- Production-intended writes must route deterministically to the production cohort.
- If routing cannot be determined unambiguously, the artifact may still be preserved for
  audit history, but it must be emitted as `ACCEPTED_NONELIGIBLE` at most.
- Mixed destination or mixed-context production writes are a hard readiness failure.
- No ambiguous or fallback root-path write may produce a gate-eligible production artifact.

## Retroactive Repair Rule

Historical repair artifacts may improve explainability, but they must not retroactively
improve readiness counts for a closed cohort unless the repair path was already part of
the approved evidence contract for that cohort.

## Promotion Thresholds

Promotion from shadow `v2` to canonical readiness requires all of the following across
5 consecutive business-day runs for the same cohort fingerprint:

- `quarantined_records == 0`
- `duplicate_conflicts == 0`
- `orphan_closes == 0`
- `linked_closes / eligible_closes >= 0.95`
- `matched_closes / eligible_closes >= 0.90`
- no mixed-context admissions
- no partial-write incident promoted to canonical artifacts
- no contract-version drift
- no cohort-fingerprint drift

## Coordination Notes

- Preserve historical evidence; never rewrite old production artifacts in place.
- Use stamped artifacts as the source of truth; `latest` files are conveniences only.
- Prefer targeted patches over broad refactors while multiple agents are active.
- If a touched file is owned by another lane, update this document and defer integration
  rather than forcing a cross-lane rewrite.
