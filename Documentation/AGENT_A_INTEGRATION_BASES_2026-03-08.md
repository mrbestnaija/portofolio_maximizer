# Agent A Integration Bases (From Agent B)

Date: 2026-03-08  
Scope: structural wiring, diagnostics, and reporting hardening only (no strategy/gate-threshold changes).

## 1) Agent A Current Track (Observed)

Primary anchor commit:
- `b925045` - Phase 7.40 hardening
  - R5 lift CI wiring fixed in `scripts/capital_readiness_check.py` (`n_used_windows` path, negative CI hard-fail).
  - `scripts/validate_profitability_proof.py` import-path collision hardening (lazy guarded sqlite import with fallback).
  - `forcester_ts/regime_detector.py` degenerate-series numerical stability.

Implication for integration:
- Keep Agent A semantics as source of truth for R5/proof/regime.
- Merge only complementary base wiring from this document.

## 2) Merge-Ready Base Deltas in Working Tree

## A. Layer-1 CI semantics parity in model-improvement checks

Files:
- `scripts/check_model_improvement.py`
- `tests/scripts/test_check_model_improvement.py`

Delta:
- CI spans zero (`ci_low <= 0 <= ci_high`) is advisory `WARN`.
- Definitively negative CI (`ci_high < 0`, sufficient windows) is `FAIL`.

Integration note:
- Overlaps conceptually with Agent A R5 semantics, but in layer-1 monitoring path.
- Keep a single consistent CI policy string/wording across readiness and layer-1 outputs.

## B. Ticker eligibility gate sidecar wiring (pipeline base)

Files:
- `scripts/apply_ticker_eligibility_gates.py` (new)
- `scripts/run_quality_pipeline.py`
- `tests/scripts/test_apply_ticker_eligibility_gates.py` (new)
- `tests/scripts/test_run_quality_pipeline.py`

Delta:
- Adds a dedicated `apply_ticker_eligibility_gates` pipeline step after eligibility computation.
- Emits additive artifact key: `artifacts.eligibility_gates`.
- Adds CLI arg: `--eligibility-gates-out`.

Semantics:
- Input: `logs/ticker_eligibility.json` with `tickers[*].status`.
- Output status: `PASS` / `WARN` (missing input) / `ERROR` (reserved for hard failures).

## C. Exit-quality numerical stability hardening

Files:
- `scripts/exit_quality_audit.py`
- `tests/scripts/test_exit_quality_audit.py`

Delta:
- Replaces chained dtype-fragile assignment with vectorized `np.where` ATR-proxy computation.
- Prevents NaN/dtype assignment failure on all-null ATR inputs.

Semantics:
- If `bar_high/bar_low` present, ATR proxy = `bar_high - bar_low`.
- Else fallback = `entry_price * 0.015` when entry price is valid.

## D. Dashboard robustness stale-sidecar truthfulness refinement

Files:
- `scripts/dashboard_db_bridge.py`
- `tests/scripts/test_dashboard_db_bridge.py`

Delta:
- Separates critical sidecars from optional sidecars for freshness severity.
- `forecast_summary` stale/unreadable now contributes `WARN`, not forced `STALE`.

Critical stale sidecars:
- `eligibility`, `context_quality`, `performance_metrics`.

Optional stale sidecar:
- `forecast_summary`.

## E. OpenClaw maintenance safety + detached-listener recovery

Files:
- `scripts/openclaw_maintenance.py`
- `tests/scripts/test_openclaw_maintenance.py`

Delta:
- PID-reuse-safe lock handling via process command-line identity checks.
- Detached gateway listener recovery with strict identity verification.
- Auto-terminate only when listener is verified as expected OpenClaw gateway process/port.

Safety semantics:
- No kill action for unverified listeners.
- Recovery path only under `apply=True` and conflict conditions.

## F. Runtime-status gate semantics alignment

Files:
- `scripts/project_runtime_status.py`
- `tests/scripts/test_project_runtime_status.py`

Delta:
- Runtime snapshot now invokes `production_audit_gate.py --unattended-profile`.
- Prevents status-tool drift where runtime snapshots could report a different gate interpretation than unattended gate orchestration.

Semantics:
- Runtime status uses unattended gate semantics for lift/proof interpretation.
- No threshold changes; only invocation-mode wiring consistency.

## 3) Suggested Integration Order for Agent A

1. Merge B (eligibility gate sidecar) first; low conflict, additive contract.
2. Merge C (exit-quality stability) second; unblocks known dtype/runtime fragility.
3. Merge D (dashboard stale severity split) third; UI truthfulness without gate math changes.
4. Merge E (OpenClaw maintenance safety) fourth; larger code surface but isolated.
5. Merge F (runtime-status semantics alignment) fifth.
6. Merge A only if needed for parity with Agent A branch state (avoid duplicate CI logic edits).

## 4) Contract Assumptions (Explicit)

For Agent C (gate/metric semantics assumed):
- Layer-1 CI:
  - spans-zero CI => advisory `WARN`.
  - both CI bounds negative => `FAIL`.
- Quality pipeline now has 5 steps (added eligibility-gate stage).
- Dashboard robustness:
  - stale critical sidecar => `STALE`.
  - stale optional forecast summary only => `WARN`.
- OpenClaw recovery:
  - restart/terminate only when listener identity is verified.
- Runtime status gate check:
  - invokes production gate in unattended profile.

For Agent A (backend fields relied on by these bases):
- Eligibility input:
  - `tickers.<symbol>.status` from ticker eligibility sidecar.
- Dashboard freshness:
  - `generated_utc` in sidecars (with file mtime fallback).
  - sidecar paths from `DEFAULT_*_PATH` constants.
- OpenClaw gateway payload:
  - `rpc.ok`
  - `service.runtime.status`, `service.runtime.state`
  - `gateway.port`
  - `port.status`
  - `port.listeners[].pid`, `port.listeners[].command`, `port.listeners[].commandLine`
- Trade quality data:
  - `entry_price`, `bar_high`, `bar_low`, `realized_pnl`.
- Runtime status output:
  - `failed_checks`, per-check `stdout/stderr`, and production gate semantics reflected from unattended profile.

## 5) Verification Evidence (Current Workspace)

Targeted integration suite:
- Command:
  - `python -m pytest tests/scripts/test_check_model_improvement.py tests/scripts/test_exit_quality_audit.py tests/scripts/test_run_quality_pipeline.py tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_openclaw_maintenance.py tests/scripts/test_apply_ticker_eligibility_gates.py -q`
- Result:
  - `85 passed`

Fast lane:
- Command:
  - `python -m pytest -m "not gpu and not slow" --tb=short -q`
- Result:
  - `1672 passed, 3 skipped, 28 deselected, 7 xfailed`

Runtime-status wiring regression:
- Command:
  - `python -m pytest tests/scripts/test_project_runtime_status.py tests/scripts/test_openclaw_implementation_contract.py -q`
- Result:
  - `12 passed`

## 6) Residual Risks / Conflict Hotspots

- `scripts/check_model_improvement.py` may conflict if Agent A has parallel edits in CI wording or status precedence.
- `scripts/dashboard_db_bridge.py` can conflict with any concurrent dashboard payload schema edits.
- `scripts/openclaw_maintenance.py` is high-churn; keep test-backed merge and avoid manual squash edits.

## 7) Agent B Patch Set (2026-03-08, follow-up)

Scope: fail-closed wiring + dashboard truthfulness + ETL numerical stability.

Files:
- `visualizations/live_dashboard.html`
- `scripts/run_all_gates.py`
- `scripts/institutional_unattended_gate.py`
- `etl/regime_detector.py`
- `tests/scripts/test_live_dashboard_template_contract.py` (new)
- `tests/scripts/test_run_all_gates.py`
- `tests/scripts/test_institutional_unattended_gate.py`
- `tests/scripts/test_phase_7_22_gate_unbypassability.py`
- `tests/etl/test_regime_detector_stability.py` (new)

Behavior changes:
- Dashboard JS parse corruption (`[ ]`) removed; script now compiles.
- Dashboard now renders unknown performance as `N/A` instead of synthetic `0`.
- `run_all_gates.py` writes a pre-institutional status artifact before invoking institutional gate.
- Institutional P4 prior-gate check is fail-closed on missing/stale artifact.
- `etl/regime_detector.py` now clamps non-finite t-test p-values to finite [0,1].

For Agent C (semantics assumed):
- Prior-gate verification (P4) is now blocking in unattended readiness if artifact is missing/stale/failed.
- `run_all_gates` publishes `status_stage` (`pre_institutional` then `final`) additively.
- Dashboard KPI semantics:
  - `performance_unknown=true` means KPI display must be non-numeric (`N/A`), not zero.
- ETL regime detector invariants:
  - `confidence` and `transition_probability` must always be finite and bounded [0,1], even on flat inputs.

For Agent A (backend fields relied on):
- `logs/gate_status_latest.json`:
  - `overall_passed`, `timestamp_utc`, `status_stage`, `gates`.
- Dashboard payload:
  - `performance_unknown`, `pnl.absolute`, `pnl.pct`, `win_rate`, `trade_count`.
- Institutional gate status dependency:
  - current-run pre-artifact must exist before institutional gate execution in `run_all_gates`.

Dependencies / constraints:
- Dashboard JS parse sanity uses local `node` in verification command.
- P4 fail-closed behavior assumes `run_all_gates.py` remains the canonical writer for `gate_status_latest.json`.
- No strategy mechanics, thresholds, or DB schema changed.
