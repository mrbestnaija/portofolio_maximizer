# Documentation Metadata Contract

Doc Type: doc_policy
Authority: canonical for metadata headers on temporary and generated documentation
Owner: Agent C
Last Verified: 2026-03-08
Verification Commands:
- `python scripts/doc_contract_audit.py --strict`
Artifacts:
- `Documentation/DOC_SOURCES_OF_TRUTH.md`
- `Documentation/GENERATED_RUNTIME_STATUS_SNAPSHOT.md`
Supersedes: none
Expires When: superseded by a newer metadata contract

Purpose:
- make documentation state machine-readable enough for validation
- distinguish canonical docs from temporary multi-agent docs
- reduce stale coordination notes and ambiguous status claims

## Required Header Fields

The following header fields are required on temporary or generated docs covered by the contract:

- `Doc Type:`
- `Authority:`
- `Owner:`
- `Last Verified:`
- `Verification Commands:`
- `Artifacts:`
- `Supersedes:`
- `Expires When:`

## Covered Documents

This contract currently applies to:

- `Documentation/AGENT_B_DASHBOARD_RUNTIME_TRUTH_HANDOFF_2026-03-08.md`
- `Documentation/AGENT_C_EXPERIMENT_BRIEF_EXP-R5-001_2026-03-08.md`
- `Documentation/AGENT_C_PHASED_IMPLEMENTATION_PLAN_2026-03-08.md`
- `Documentation/AGENT_C_PERSISTENCE_MANAGER_INTEGRATION_2026-03-08.md`
- `Documentation/AGENT_C_READINESS_BLOCKER_MATRIX_2026-03-08.md`
- `Documentation/AGENT_C_RESUME_PACK_2026-03-08_AM.md`
- `Documentation/GENERATED_RUNTIME_STATUS_SNAPSHOT.md`

## Field Meanings

- `Doc Type`
  - one of: `source_of_truth_map`, `doc_policy`, `status_snapshot`, `blocker_matrix`, `experiment_brief`, `handoff_note`, `integration_note`, `implementation_plan`, `resume_pack`
- `Authority`
  - explain whether the doc is canonical, generated, or temporary
- `Owner`
  - who maintains the file
- `Last Verified`
  - date of the latest evidence-backed update
- `Verification Commands`
  - commands used to support the current claims
- `Artifacts`
  - primary files or outputs referenced by the doc
- `Supersedes`
  - prior file replaced by this one, or `none`
- `Expires When`
  - clear deletion/supersession condition for temporary docs

## Generated Docs

Generated docs must also state the script that produced them and should not be hand-edited.

## Policy

- A temporary doc without an expiry condition is considered incomplete.
- A status doc without verification commands is considered incomplete.
- A generated doc edited by hand is out of contract.
