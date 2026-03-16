# OpenClaw Ops Patch Bundle (2026-03-15)

Bundle ID: `openclaw_ops_lane_2026-03-15.safe`
Date: 2026-03-15
Owner: Codex handoff for Claude Code + human integration

Primary artifacts:
- `Documentation/patch_bundles/openclaw_ops_lane_2026-03-15.safe.patch`
- `Documentation/patch_bundles/openclaw_ops_lane_2026-03-15.safe.files.txt`
- `Documentation/patch_bundles/openclaw_ops_lane_2026-03-15.safe.sha256.txt`

Source commit:
- `e7a25be` - `Add OpenClaw ops control plane and harden readiness CLI`

## Purpose

Provide a clean, reviewable bundle for the committed OpenClaw operations lane without
mixing in the current workspace's newer uncommitted parallel changes.

This bundle is intentionally commit-scoped. The active workspace now contains extra dirty
edits, including later modifications in the same OpenClaw files, so the human/Claude
integration target should review this bundle rather than a raw working-tree diff.

## Included Files

1. `bash/production_cron.sh`
2. `scripts/openclaw_ops_control_plane.py`
3. `scripts/openclaw_production_readiness.py`
4. `tests/scripts/test_openclaw_ops_control_plane.py`
5. `tests/scripts/test_openclaw_production_readiness.py`

## Explicitly Excluded

- all current uncommitted working-tree changes after `e7a25be`
- protected shared files outside this bundle, especially:
  - `scripts/run_all_gates.py`
  - `scripts/production_audit_gate.py`
  - `integrity/pnl_integrity_enforcer.py`
  - `scripts/llm_multi_model_orchestrator.py`
- logs, caches, and dashboard artifacts

## Verification

Verified in an isolated detached worktree at commit `e7a25be`.

Command:

`python -m pytest tests/scripts/test_openclaw_ops_control_plane.py tests/scripts/test_openclaw_production_readiness.py tests/scripts/test_openclaw_implementation_contract.py tests/scripts/test_windows_dashboard_manager.py tests/scripts/test_windows_persistence_manager.py tests/scripts/test_openclaw_regression_gate.py -q`

Result:

- `30 passed`

## Integration Guidance

Recommended apply path:

1. Start from the clean local integration worktree on `integration/readiness-20260315` at `C:\Users\Bestman\personal_projects\pmx_readiness_integration`.
2. `git apply --3way "C:\Users\Bestman\personal_projects\portfolio_maximizer_v45\portfolio_maximizer_v45\Documentation\patch_bundles\openclaw_ops_lane_2026-03-15.safe.patch"`
3. `python -m pytest tests/scripts/test_openclaw_ops_control_plane.py tests/scripts/test_openclaw_production_readiness.py tests/scripts/test_openclaw_implementation_contract.py tests/scripts/test_windows_dashboard_manager.py tests/scripts/test_windows_persistence_manager.py tests/scripts/test_openclaw_regression_gate.py -q`
4. `python scripts/openclaw_production_readiness.py --json`
5. run the repo fast lane after integration if the target branch policy requires it

See `Documentation/LOCAL_READINESS_INTEGRATION_WORKFLOW_2026-03-15.md` for the
two-terminal split and local-only integration rule.

## Coordination Note

`origin/master` does not share a merge base with this lane in the current local clone, so
the bundle is supplied as a file-scoped patch rather than a normal branch-range patch.
