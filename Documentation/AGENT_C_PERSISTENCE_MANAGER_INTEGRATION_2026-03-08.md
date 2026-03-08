# Agent C Integration Note: Persistence Manager

Date: 2026-03-08  
Owner: Agent C  
Scope: reboot-safe persistence/data-integrity supervisor only  
Change class: additive wiring only, no gate-threshold or strategy semantics changes

## 1) What Agent C Added

Files:
- `scripts/windows_persistence_manager.py`
- `scripts/run_persistence_manager.bat`
- `tests/scripts/test_windows_persistence_manager.py`

No existing production gate, integrity, or watcher logic was edited in this slice.
The supervisor reuses existing entry points:
- `scripts/windows_dashboard_manager.py`
- `scripts/repair_unlinked_closes.py`
- `integrity/pnl_integrity_enforcer.py`
- `scripts/check_forecast_audits.py`
- `scripts/run_live_denominator_overnight.py`

## 2) Problem This Solves

Current repo behavior already had:
- dashboard startup helper
- live denominator watcher
- unlinked-close repair tool
- integrity audit
- forecast audit summarizer

But those were not tied together under one reboot-safe recovery path.

This slice adds a single Windows-native supervisor that, after reboot/login:
1. ensures dashboard bridge + HTTP server + live watcher are running
2. reconciles unlinked close legs
3. refreshes audit/linkage evidence
4. writes one authoritative status JSON

This is for persistence continuity and DB/evidence hygiene after system restart.

## 3) New Runtime Contract

Primary command:

```powershell
python scripts/windows_persistence_manager.py ensure --status-json logs\persistence_manager_status.json
```

Batch wrapper:

```powershell
scripts\run_persistence_manager.bat
```

Status artifact:
- `logs/persistence_manager_status.json`

The status JSON includes:
- dashboard process state
- reconciliation before/after counts
- integrity audit result
- forecast audit summary subset
- live watcher summary subset
- `startup_registration` details:
  - `ok`
  - `method` (`schtasks`, `registry_run_key`, or `none`)
  - `returncode`
  - `details_tail`

## 4) Persistence Behavior Across Shutdown/Login

Intended Windows persistence order:
1. try Task Scheduler registration
   - `schtasks /Create ... /SC ONSTART /RL HIGHEST`
2. if that fails, try lower-privilege fallback
   - `schtasks /Create ... /SC ONLOGON`
3. if Task Scheduler is blocked by Windows privileges, fall back to:
   - `HKCU\Software\Microsoft\Windows\CurrentVersion\Run`

Observed on this machine:
- Task Scheduler registration failed with `Access is denied.`
- Registry fallback succeeded
- Verified key exists:
  - `PortfolioMaximizer_PersistenceManager = "...\scripts\run_persistence_manager.bat"`

Implication for Agent A:
- this slice already survives normal reboot/login on this workstation via per-user startup
- do not treat missing Task Scheduler registration as a repo failure if the registry fallback is present

## 5) Verified Evidence

Targeted tests:

```powershell
python -m pytest tests/scripts/test_windows_persistence_manager.py tests/scripts/test_windows_dashboard_manager.py -q
```

Result:
- `7 passed`

Fast lane:

```powershell
python -m pytest -m "not gpu and not slow" --tb=short -q
```

Result:
- `1678 passed, 3 skipped, 28 deselected, 7 xfailed`

Runtime verification:

```powershell
scripts\run_persistence_manager.bat
```

Observed output:
- dashboard watcher `running=True`
- dashboard bridge `running=True`
- HTTP server `running=True`
- reconciliation `before=0 after=0 rc=0`
- audits `effective=16 matched=0 linkage_included=1`
- watcher `fresh_linkage_included=1 fresh_matched=0`

## 6) Current Runtime Status from This Slice

From `logs/persistence_manager_status.json` at verification time:
- startup registration:
  - `ok = true`
  - `method = registry_run_key`
- reconciliation:
  - `before.count = 0`
  - `after.count = 0`
- integrity:
  - `all_passed = true`
  - `orphan_detected = false`
- audits:
  - `effective_audits = 16`
  - `n_outcome_windows_matched = 0`
  - `n_linkage_denominator_included = 1`
- watcher:
  - `fresh_trade_rows = 1`
  - `fresh_linkage_included = 1`
  - `fresh_production_valid_matched = 0`

This means the persistence path is healthy, but Phase 3 is still blocked by denominator growth and lack of fresh production-valid matches.

## 7) Integration Guidance for Agent A

Recommended merge order:
1. add the three files from this slice only
2. run the targeted persistence tests
3. run the fast regression lane
4. run the wrapper once and inspect `logs/persistence_manager_status.json`

Suggested file list to stage:
- `scripts/windows_persistence_manager.py`
- `scripts/run_persistence_manager.bat`
- `tests/scripts/test_windows_persistence_manager.py`

Optional operational follow-up:
- keep the registry fallback as the default on this workstation
- only switch to Task Scheduler if Agent A explicitly wants admin-scoped startup instead of user-logon startup

## 8) Hard Boundaries

This slice does **not**:
- change `production_audit_gate.py`
- change gate thresholds
- change RMSE / lift semantics
- change watcher denominator semantics
- change integrity-check rules
- claim readiness improvement

It only ensures the existing recovery and evidence-refresh steps are run consistently after restart.

## 9) Known Residuals (Not Solved Here)

- `production_gate` still fails on:
  - `THIN_LINKAGE`
  - `EVIDENCE_HYGIENE_FAIL`
  - `matched=0/1`
- Fresh production-valid matched rows are still `0`
- Effective audits are still `16 < 20`

These are intentionally outside this persistence slice.

## 10) If Agent A Integrates This

Post-merge smoke commands:

```powershell
python -m pytest tests/scripts/test_windows_persistence_manager.py tests/scripts/test_windows_dashboard_manager.py -q
scripts\run_persistence_manager.bat
python scripts/project_runtime_status.py --pretty
```

Acceptance for this slice:
- startup supervisor runs without manual intervention
- status JSON is written
- startup registration method is explicit in the status artifact
- reconciliation result is explicit
- audit/linkage state is refreshed and visible after reboot/login
