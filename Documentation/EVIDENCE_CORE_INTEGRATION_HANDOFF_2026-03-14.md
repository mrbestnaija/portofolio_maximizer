# Evidence Core Integration Handoff (2026-03-14)

Doc Type: `integration_note`

This handoff scopes the evidence-core lane for Claude Code + human review. It exists
because the workspace contains parallel agent changes and the repo protocol forbids
bundling unrelated dirty files into a single integration.

References:
- [AGENT_COORDINATION_PROTOCOL_2026-03-08.md](./AGENT_COORDINATION_PROTOCOL_2026-03-08.md)
- [MOMENTUM_SAFE_EVIDENCE_COORDINATION.md](./MOMENTUM_SAFE_EVIDENCE_COORDINATION.md)
- [CLEAN_COHORT_OPERATIONS_2026-03-14.md](./CLEAN_COHORT_OPERATIONS_2026-03-14.md)

Patch bundle artifacts:

- `Documentation/patch_bundles/evidence_core_lane_2026-03-14.safe.patch`
- `Documentation/patch_bundles/evidence_core_lane_2026-03-14.safe.files.txt`
- `Documentation/patch_bundles/evidence_core_lane_2026-03-14.safe.sha256.txt`

## Review Scope

Safe additive files in this lane:

- `utils/evidence_io.py`
- `scripts/replay_trade_evidence_chain.py`
- `scripts/clean_cohort_manager.py`
- `Documentation/MOMENTUM_SAFE_EVIDENCE_COORDINATION.md`
- `Documentation/CLEAN_COHORT_OPERATIONS_2026-03-14.md`
- `tests/utils/test_evidence_io.py`
- `tests/scripts/test_replay_trade_evidence_chain.py`
- `tests/scripts/test_clean_cohort_manager.py`

Tracked files changed in this lane and verified locally:

- `execution/paper_trading_engine.py`
- `forcester_ts/instrumentation.py`
- `scripts/check_forecast_audits.py`
- `scripts/run_auto_trader.py`
- `tests/scripts/test_check_forecast_audits.py`

Shared files requiring Claude diff review because parallel modifications already existed:

- `forcester_ts/forecaster.py`
- `scripts/production_audit_gate.py`
- `tests/scripts/test_production_audit_gate.py`

## What To Defer

Do not bundle these unrelated workspace changes into the evidence-core review:

- `integrity/pnl_integrity_enforcer.py`
- `scripts/run_all_gates.py`
- `scripts/llm_multi_model_orchestrator.py`
- `visualizations/live_dashboard.html`
- `bash/production_cron.sh`
- untracked operational and visualization artifacts outside this lane

## Verified Evidence

Targeted suites:

- `python -m pytest tests/scripts/test_hygiene_wiring.py tests/scripts/test_production_audit_gate.py tests/scripts/test_check_forecast_audits.py tests/scripts/test_replay_trade_evidence_chain.py tests/utils/test_evidence_io.py -q`
- `python -m pytest tests/forcester_ts/test_forecaster_snapshot_integration.py tests/scripts/test_auto_trader_lifecycle.py tests/scripts/test_run_auto_trader_config_guard.py -q`
- `python -m pytest tests/scripts/test_production_audit_gate.py tests/scripts/test_replay_trade_evidence_chain.py tests/scripts/test_run_all_gates.py -q`
- `python -m pytest tests/scripts/test_clean_cohort_manager.py -q`
- `python -m pytest -m "not gpu and not slow" --tb=short -q`
- `python -m integrity.pnl_integrity_enforcer`
- `python scripts/replay_trade_evidence_chain.py --scenario happy_path --json`

Global gate snapshot:

- `python scripts/run_all_gates.py --json`

Current global result is still red for live-evidence reasons (`GATES_FAIL`, `THIN_LINKAGE`,
`EVIDENCE_HYGIENE_FAIL`). That is a production-readiness truth, not an implementation crash.

## Integration Order

1. Safe additive files
2. Tracked non-conflicting files
3. Shared reviewed hunks in `forecaster.py`
4. Shared reviewed hunks in `production_audit_gate.py`
5. Clean cohort freeze + proof loop
