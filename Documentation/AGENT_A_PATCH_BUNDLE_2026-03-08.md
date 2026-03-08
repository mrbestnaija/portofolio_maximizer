# Agent A Patch Bundle (Clean Base)

Bundle ID: `agent_a_base_bundle_2026-03-08`  
Date: 2026-03-08  
Owner: Agent B (handoff to Agent A)

Primary artifacts (ordered split patches):
- `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.01-docs.patch`
- `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.02-eligibility-pipeline.patch`
- `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.03-exit-quality-stability.patch`
- `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.04-dashboard-robustness.patch`
- `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.05-openclaw-maintenance.patch`
- `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.06-runtime-status-semantics.patch`

Per-patch sidecars:
- each split patch has matching `.files.txt` and `.sha256.txt` in `Documentation/patch_bundles/`

Monolithic fallback artifact:
- `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.patch`
- `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.sha256.txt`

Machine-readable file list:
- `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.files.txt`

## Purpose

Provide a clean, conflict-minimized patch bundle containing only structural/additive wiring changes for Agent A integration.

## Included Files (Exact)

1. `Documentation/AGENT_A_PATCH_BUNDLE_2026-03-08.md`
2. `Documentation/AGENT_A_INTEGRATION_BASES_2026-03-08.md`
3. `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.files.txt`
4. `scripts/apply_ticker_eligibility_gates.py`
5. `tests/scripts/test_apply_ticker_eligibility_gates.py`
6. `scripts/run_quality_pipeline.py`
7. `tests/scripts/test_run_quality_pipeline.py`
8. `scripts/exit_quality_audit.py`
9. `tests/scripts/test_exit_quality_audit.py`
10. `scripts/dashboard_db_bridge.py`
11. `tests/scripts/test_dashboard_db_bridge.py`
12. `scripts/openclaw_maintenance.py`
13. `tests/scripts/test_openclaw_maintenance.py`
14. `scripts/project_runtime_status.py`
15. `tests/scripts/test_project_runtime_status.py`

## Explicitly Excluded (Overlap Risk with Agent A Core Track)

1. `scripts/check_model_improvement.py`
2. `tests/scripts/test_check_model_improvement.py`
3. `integrity/pnl_integrity_enforcer.py`
4. `tests/integrity/test_pnl_integrity_enforcer.py`
5. `logs/**`
6. `visualizations/**`

## Integration Commands (Agent A)

1. Apply ordered patches:
   - `git apply --3way Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.01-docs.patch`
   - `git apply --3way Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.02-eligibility-pipeline.patch`
   - `git apply --3way Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.03-exit-quality-stability.patch`
   - `git apply --3way Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.04-dashboard-robustness.patch`
   - `git apply --3way Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.05-openclaw-maintenance.patch`
   - `git apply --3way Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.06-runtime-status-semantics.patch`
2. Per-patch targeted tests:
   - After `.02`: `python -m pytest tests/scripts/test_apply_ticker_eligibility_gates.py tests/scripts/test_run_quality_pipeline.py -q`
   - After `.03`: `python -m pytest tests/scripts/test_exit_quality_audit.py -q`
   - After `.04`: `python -m pytest tests/scripts/test_dashboard_db_bridge.py -q`
   - After `.05`: `python -m pytest tests/scripts/test_openclaw_maintenance.py -q`
   - After `.06`: `python -m pytest tests/scripts/test_project_runtime_status.py tests/scripts/test_openclaw_implementation_contract.py -q`
3. Final regression lane:
   - `python -m pytest -m "not gpu and not slow" --tb=short -q`

## Patch-to-File Mapping

1. `01-docs`
   - `Documentation/AGENT_A_PATCH_BUNDLE_2026-03-08.md`
   - `Documentation/AGENT_A_INTEGRATION_BASES_2026-03-08.md`
   - `Documentation/patch_bundles/agent_a_base_bundle_2026-03-08.files.txt`
2. `02-eligibility-pipeline`
   - `scripts/apply_ticker_eligibility_gates.py`
   - `tests/scripts/test_apply_ticker_eligibility_gates.py`
   - `scripts/run_quality_pipeline.py`
   - `tests/scripts/test_run_quality_pipeline.py`
3. `03-exit-quality-stability`
   - `scripts/exit_quality_audit.py`
   - `tests/scripts/test_exit_quality_audit.py`
4. `04-dashboard-robustness`
   - `scripts/dashboard_db_bridge.py`
   - `tests/scripts/test_dashboard_db_bridge.py`
5. `05-openclaw-maintenance`
   - `scripts/openclaw_maintenance.py`
   - `tests/scripts/test_openclaw_maintenance.py`
6. `06-runtime-status-semantics`
   - `scripts/project_runtime_status.py`
   - `tests/scripts/test_project_runtime_status.py`

## Contract Notes

- No strategy mechanics or thresholds changed.
- Additive wiring only:
  - eligibility gate sidecar,
  - quality pipeline stage wiring,
  - exit-quality numeric stability,
  - dashboard optional-sidecar stale handling,
  - OpenClaw maintenance recovery safety.
