# Agent C Phased Implementation Plan (2026-03-08)

Doc Type: implementation_plan
Authority: temporary Agent C execution plan within owned scope only
Owner: Agent C
Last Verified: 2026-03-08
Verification Commands:
- `git status --porcelain`
- `python scripts/project_runtime_status.py --pretty`
Artifacts:
- `Documentation/AGENT_C_READINESS_BLOCKER_MATRIX_2026-03-08.md`
- `Documentation/AGENT_C_RESUME_PACK_2026-03-08_AM.md`
Supersedes: none
Expires When: superseded by a newer Agent C implementation plan or retired when the lane closes

Owner: Agent C  
Scope: documentation, evidence normalization, handoff packaging, and isolated
support slices only

This plan is intentionally constrained to the Agent C lane. It does not claim
ownership of gate semantics, dashboard runtime fixes, or strategy mechanics.

## Objective

Create a clean handoff and monitoring path for the currently verified blockers:

1. runtime/dashboard truth drift
2. mixed-source provenance mislabeling
3. eligibility fail-open semantics
4. thin fresh linkage denominator
5. reboot-safe persistence slice packaging

## Phase 0 - Scope Lock

Goal: avoid cross-agent collisions.

Actions:
- do not edit Agent A/B-owned implementation files:
  - `scripts/check_model_improvement.py`
  - `scripts/dashboard_db_bridge.py`
  - `visualizations/live_dashboard.html`
  - `scripts/apply_ticker_eligibility_gates.py`
  - `forcester_ts/regime_detector.py`
- operate only in:
  - `Documentation/AGENT_C_*`
  - `scripts/windows_persistence_manager.py`
  - `scripts/run_persistence_manager.bat`
  - `tests/scripts/test_windows_persistence_manager.py`
- verify workspace state before any edit with `git status --porcelain`

Acceptance:
- no edits to shared gate/dashboard/math files

Dependencies:
- none beyond current workspace access

Limitations:
- technical: Agent C must not fix shared implementation lanes directly
- spatial: this plan assumes work remains local to the current workspace

## Phase 1 - Evidence Normalization

Goal: replace stale or optimistic Agent C notes with current command-backed state.

Actions:
- refresh the Agent C blocker matrix using:
  - `python scripts/project_runtime_status.py --pretty`
  - `python scripts/capital_readiness_check.py --json`
  - `python scripts/check_model_improvement.py --layer 1 --json`
  - `python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db --output logs\dashboard_data_review_tmp.json`
- explicitly separate:
  - code/test state
  - live runtime state
- record only verified blockers

Acceptance:
- blocker matrix no longer claims dashboard truth is resolved if the served
  payload still lags the checked-in schema
- blocker matrix uses the correct next trading date

Dependencies:
- local DB and logs must exist

Limitations:
- temporal: live evidence can change only when new trading-day artifacts arrive
- technical: one-shot bridge output may differ from the active daemon output

## Phase 2 - Agent A/B Handoff Package

Goal: make ownership unambiguous.

Actions:
- produce a handoff section for Agent A:
  - eligibility fail-open policy decision
  - any cross-runtime execution dependency decisions
- produce a handoff section for Agent B:
  - dashboard runtime drift
  - mixed-origin labeling
  - UI truth handling for stale/unknown data
- for each item include:
  - exact file references
  - verified command evidence
  - acceptance condition

Acceptance:
- each blocker is assigned to one owner only
- no blocker is described as “fixed” when it is only fixed in tests or in a
  one-shot render path

Dependencies:
- current verified evidence from Phase 1

Limitations:
- technical: Agent C cannot validate fixes in shared files until Agent A/B land them

## Phase 3 - Persistence Slice Packaging

Goal: keep the reboot-safe persistence work integrable without semantic bleed.

Actions:
- maintain only these files:
  - `scripts/windows_persistence_manager.py`
  - `scripts/run_persistence_manager.bat`
  - `tests/scripts/test_windows_persistence_manager.py`
  - `Documentation/AGENT_C_PERSISTENCE_MANAGER_INTEGRATION_2026-03-08.md`
- keep the status contract explicit:
  - `startup_registration.ok`
  - `startup_registration.method`
  - `startup_registration.returncode`
  - `startup_registration.details_tail`
- keep verification evidence current

Acceptance:
- Agent A can integrate this slice independently of any gate/dashboard changes

Dependencies:
- Windows host with either Task Scheduler access or HKCU Run-key access

Limitations:
- technical: Task Scheduler registration may fail on this workstation due to privileges
- spatial: persistence behavior is Windows-specific and workstation-local

## Phase 4 - Runtime Drift Monitoring

Goal: detect live mismatches that unit tests will not catch.

Actions:
- compare:
  - `visualizations/dashboard_data.json`
  - `logs/dashboard_data_review_tmp.json`
- track only:
  - presence of `payload_schema_version`
  - presence of `payload_digest`
  - presence of `performance_unknown`
  - presence of `positions_stale`
  - `data_origin` vs `trade_sources`
- keep denominator watcher monitoring limited to:
  - fresh `TRADE` exclusions
  - `fresh_linkage_included`
  - fresh production-valid `matched`

Acceptance:
- drift is explicitly marked as either:
  - resolved in live runtime, or
  - still live despite local code/tests

Dependencies:
- active dashboard bridge process
- latest watcher artifacts

Limitations:
- temporal: weekend/market-closed periods can only preserve state, not advance evidence
- technical: daemon restarts may be required before code fixes appear in served payload

## Phase 5 - Post-Merge Verification

Goal: confirm that Agent A/B fixes actually land in live runtime.

Commands:
- `python scripts/project_runtime_status.py --pretty`
- `python scripts/run_all_gates.py --json`
- `python -m scripts.dashboard_db_bridge --once --db-path data\portfolio_maximizer.db --output logs\dashboard_data_review_tmp.json`
- `python -m pytest tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_live_dashboard_wiring.py tests/scripts/test_check_model_improvement.py -q`

Acceptance:
- served dashboard payload matches current schema
- mixed provenance is labeled correctly
- eligibility behavior matches the chosen policy
- Agent C blocker matrix can be reduced without caveats

Dependencies:
- Agent A/B fixes merged into the active runtime branch

Limitations:
- technical: verification cannot force new production-valid matched rows to appear
- temporal: readiness evidence still depends on future trading-day accumulation

## Temporal / Spatial / Technical Constraints

### Temporal

- fresh production-valid linkage evidence cannot be accelerated under the current
  30-day-horizon production contract
- weekend periods do not produce new live denominator evidence
- the next trading-day evidence window is Monday, 2026-03-09

### Spatial

- persistence/startup findings are specific to this Windows workstation
- current startup persistence uses the per-user Run key, not a machine-wide service
- active runtime artifacts under `visualizations/` and `logs/` may lag checked-in code

### Technical

- Agent C is constrained away from shared implementation files to avoid
  multi-agent conflicts
- one-shot commands can prove local code behavior but cannot by themselves prove
  the long-running daemon has picked up the same code
- the dashboard bridge currently has two truths to compare:
  - checked-in local render path
  - active served artifact path

## Current External Dependencies

- SQLite DB: `data/portfolio_maximizer.db`
- watcher artifact: `logs/overnight_denominator/live_denominator_latest.json`
- audit summary artifact: `logs/forecast_audits_cache/latest_summary.json`
- served dashboard artifact: `visualizations/dashboard_data.json`
- fresh bridge render comparison artifact: `logs/dashboard_data_review_tmp.json`

## Exit Condition For Agent C

Agent C should stop active planning and switch to verification-only when:

1. the handoff package is current
2. the persistence slice is documented and isolated
3. live runtime drift has been clearly assigned to Agent B
4. policy decisions have been clearly assigned to Agent A

At that point, Agent C should monitor and report, not expand implementation scope.
