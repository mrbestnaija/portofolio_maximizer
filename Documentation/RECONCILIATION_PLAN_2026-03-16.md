# Backup Reconciliation → Terminal B Integration Plan
**Created**: 2026-03-16
**Status**: IN PROGRESS — Phase 1 active
**Branch**: `integration/backup-reconciliation-20260316`
**Author**: Claude Code (session 2026-03-16)
**Survives shutdown**: YES — this document is the source of truth if session is interrupted

---

## Why This Document Exists

On 2026-03-16 a security fix (`fix/dependabot-security-deps-20260316`) was merged into
`origin/master`. Before merging, local master was discovered to have diverged from
`origin/master` with **91 uncommitted-to-origin commits** representing Phase 7.9 → 7.41 work.

Local master was reset to `origin/master` to allow the clean merge. Those 91 commits
were saved to `backup/local-master-20260316`.

**Risk**: The PnL/WR data in `data/portfolio_maximizer.db` (37 round-trips, 43.2% WR,
$673.22 PnL) was produced under Phase 7.35–7.41 code that is now only in the backup
branch. Any future trading runs on the current master will run under different code,
breaking causal attribution.

**Goal**: Promote the backup's code improvements (Phase 7.9–7.41) into canonical master
without regressing PnL integrity, WR metrics, or test count. Then apply Terminal B
patch bundles (lift semantics) on top.

---

## State Snapshot (at plan creation)

| Item | Value |
|------|-------|
| Current master HEAD | `fa8c829` (security fix merge) |
| Backup branch HEAD | `4f27fad` (Phase 7.41) |
| Integration branch base | `fa8c829` |
| Backup commits not in origin/master | 91 commits |
| Files with code delta (non-doc) | 47 modified, 12 new |
| Known conflicts with master | requirements.txt, requirements-ml.txt, .github/workflows, Dockerfile, .gitignore, platt_contract_audit.py |
| Local DB state | 37 round-trips, 43.2% WR, $673.22 PnL, 0 integrity violations |
| Platt calibration | 40 JSONL pairs, ECE=0.40 (threshold 0.15 — poor but active) |
| Lift gate | Layer 1 FAIL: lift_ci=[-0.3672,-0.1846] definitively negative |

---

## Conflict Resolution Rules (IMMUTABLE)

These rules apply at every merge step. Do not deviate.

| File / Group | Rule |
|---|---|
| `requirements.txt` | ALWAYS keep master version — PyJWT 2.12.0 (CVE-2026-32597) must not regress |
| `requirements-ml.txt` | ALWAYS keep master version — cupy-cuda12x 14.0.0, curand 10.3.10.19 |
| `.github/workflows/*.yml` | ALWAYS keep master version — actions pinned to v6 |
| `Dockerfile` | Keep master's `INSTALL_ML_EXTRAS` ARG; apply any other backup changes |
| `.gitignore` | Keep master's forecast_audit_manifest entries; add backup additions |
| `scripts/platt_contract_audit.py` | Keep master's WARN-on-no-DB logic; apply any other backup changes |
| `logs/forecast_audits/forecast_audit_manifest.jsonl` | NEVER re-add to git tracking |

---

## Phase Structure

### PHASE 1 — PnL-Critical Code (Gate: Integrity + WR)
**Branch**: `integration/backup-reconciliation-20260316`
**Status**: IN PROGRESS

Files to apply from backup:

| File | Risk | Notes |
|------|------|-------|
| `integrity/pnl_integrity_enforcer.py` | HIGH | Core PnL computation — run full integrity audit after |
| `scripts/capital_readiness_check.py` | HIGH | R1–R6 checks, R5 CI semantics |
| `scripts/check_model_improvement.py` | HIGH | Layer 1–4 lift computation |
| `scripts/production_audit_gate.py` | HIGH | Production gate logic |
| `scripts/exit_quality_audit.py` | MEDIUM | ATR proxy, exit classification |
| `scripts/validate_profitability_proof.py` | MEDIUM | Profitability proof logic |
| `scripts/run_all_gates.py` | MEDIUM | Gate orchestrator |
| `config/forecaster_monitoring.yml` | MEDIUM | Threshold config |

**Exit criteria for Phase 1**:
- [ ] `python -m integrity.pnl_integrity_enforcer` → 0 CRITICAL, 0 HIGH violations
- [ ] `python scripts/capital_readiness_check.py` → runs without crash (FAIL on data is OK)
- [ ] `python scripts/run_all_gates.py` → exits without crash
- [ ] Test count ≥ current master test count (no regression)
- [ ] `pytest tests/integrity/ tests/scripts/test_capital_readiness_check.py tests/scripts/test_check_model_improvement.py tests/scripts/test_production_audit_gate.py -q` → 0 failures

---

### PHASE 2 — Signal/Forecast Pipeline (Gate: Forecast Audit)
**Depends on**: Phase 1 complete

Files to apply from backup:

| File | Risk | Notes |
|------|------|-------|
| `forcester_ts/forecaster.py` | HIGH | Audit path, regime metadata |
| `forcester_ts/regime_detector.py` | MEDIUM | Hurst stability fix |
| `etl/regime_detector.py` | MEDIUM | Regime stability |
| `etl/data_universe.py` | LOW | Ticker universe unification |
| `scripts/run_etl_pipeline.py` | MEDIUM | Audit routing, ticker resolution |
| `scripts/apply_ticker_eligibility_gates.py` | LOW | New script — ticker gating |
| `scripts/build_training_dataset.py` | LOW | New script — dataset builder |
| `scripts/run_quality_pipeline.py` | MEDIUM | Quality pipeline steps |

**Exit criteria for Phase 2**:
- [ ] `python scripts/run_etl_pipeline.py --help` → no crash
- [ ] `pytest tests/forcester_ts/ tests/etl/ -q` → 0 new failures vs Phase 1 baseline
- [ ] `pytest tests/scripts/test_run_quality_pipeline.py -q` → 0 failures
- [ ] Forecast audit writes to `logs/forecast_audits/production/` (not root)

---

### PHASE 3 — Dashboard / Ops / OpenClaw (Gate: Dashboard truthfulness)
**Depends on**: Phase 2 complete

Files to apply from backup:

| File | Risk | Notes |
|------|------|-------|
| `scripts/dashboard_db_bridge.py` | MEDIUM | data_origin=mixed fix, stale positions |
| `scripts/project_runtime_status.py` | LOW | Runtime status semantics |
| `scripts/openclaw_maintenance.py` | LOW | Maintenance cron logic |
| `scripts/openclaw_notify.py` | LOW | Storm suppression |
| `scripts/institutional_unattended_gate.py` | MEDIUM | Gate counting logic |
| `scripts/run_live_denominator_overnight.py` | LOW | New script |
| `scripts/windows_persistence_manager.py` | LOW | New script |
| `scripts/start_openclaw_guardian.ps1` | LOW | Guardian hardening |
| `scripts/verify_openclaw_config.py` | LOW | Config validation |
| `scripts/windows_dashboard_manager.py` | LOW | Dashboard manager |
| `AGENTS.md` | LOW | Agent guardrails update |
| `CONTRIBUTING.md` | LOW | Contribution guide |
| `visualizations/live_dashboard.html` | LOW | Dashboard template |

**Exit criteria for Phase 3**:
- [ ] `python scripts/dashboard_db_bridge.py --dry-run` → no crash
- [ ] `pytest tests/scripts/test_dashboard_db_bridge.py tests/scripts/test_project_runtime_status.py -q` → 0 failures
- [ ] `python scripts/institutional_unattended_gate.py` → no crash (gate result irrelevant)

---

### PHASE 4 — New Tests + Test Updates (Gate: Full fast lane)
**Depends on**: Phase 3 complete

Apply all new test files from backup and updated test files.
Run full fast-lane test suite.

**Exit criteria for Phase 4**:
- [ ] `pytest tests/ -q --timeout=60` → 0 failures (xfail allowed)
- [ ] Test count ≥ backup branch test count (no regression vs backup)
- [ ] Zero new test files in `scripts/` (placement rule enforced)

---

### PHASE 5 — Commit and PR to Master
**Depends on**: Phase 4 complete

- Create PR from `integration/backup-reconciliation-20260316` → `master`
- PR title: "Reconcile Phase 7.9–7.41 backup commits to canonical master"
- CI must pass before merge

---

### PHASE 6 — Terminal B Patch Bundles (AFTER Phase 5 merged)
**Depends on**: Phase 5 merged to master

Apply the 3 Terminal A patch bundles in strict sequence on a new branch:

```
Step 1: lift_semantics_gate_truth_2026-03-15.reviewed.patch
Step 2: lift_semantics_baseline_parity_2026-03-15.reviewed.patch
Step 3: openclaw_ops_lane_2026-03-15.safe.patch
```

Verification at each step:
- SHA-256 checksum matches `.sha256.txt` file
- `pytest tests/ -q --timeout=60` → 0 failures
- `python scripts/run_all_gates.py` → gate results recorded

**Exit criteria for Phase 6**:
- [ ] Lift CI range shifts from `[-0.3672,-0.1846]` (pre-patch baseline)
- [ ] `python scripts/check_model_improvement.py` → no crash
- [ ] All 3 patches applied, verified, committed

---

### PHASE 7 — Directional Classifier Research Track (NEW)
**Depends on**: Phase 6 complete, lift gate re-evaluated

This phase begins the work identified by the external review and the 14pp win-rate gap
analysis. EXP-R5-001 (RMSE residual ensemble) is **suspended** pending lift gate
re-evaluation.

**Scope**:
1. Binary directional classifier (P(price_up_in_N_bars)) using existing feature set
2. Platt calibration against directional outcomes (not PnL magnitude)
3. Replace RMSE-derived confidence signal with classifier probability
4. Target: ECE < 0.15 (currently 0.40), WR > 45% (currently 43.2%)

**This is a research track, not a Phase 7 fix. Assign to Phase 8.**

---

## PnL / WR Protection Gates

These must be run and recorded at each phase boundary:

```bash
# Gate 1: Integrity
python -m integrity.pnl_integrity_enforcer --db data/portfolio_maximizer.db

# Gate 2: Capital readiness
python scripts/capital_readiness_check.py

# Gate 3: Production audit
python scripts/production_audit_gate.py

# Gate 4: All gates
python scripts/run_all_gates.py
```

**Baseline to protect** (as of 2026-03-16):
- Round-trips: 37
- Win rate: 43.2% (16W/21L)
- Total PnL: $673.22
- Profit factor: 1.85
- Integrity violations: 0 (with whitelist)
- Platt pairs: 40 JSONL

Any phase that drops WR below 40% or increases integrity violations above 0 (outside
whitelist) is a REGRESSION and must be rolled back before proceeding.

---

## Recovery Instructions (If Session Interrupted)

If this session is interrupted at any point, resume with:

```bash
# 1. Check current state
git branch --show-current
git log --oneline -5

# 2. If on integration branch, check which phase is complete
python -m integrity.pnl_integrity_enforcer
pytest tests/ -q --timeout=60 2>&1 | tail -5

# 3. Resume from last completed phase
# Reference this document for which files each phase covers
# Reference MEMORY.md for overall project context

# 4. If integration branch is lost, recreate from master + replay
git checkout -b integration/backup-reconciliation-20260316 master
# Then apply phases 1-N as documented above using backup branch diffs

# 5. Key reference commands
git diff fa8c829 backup/local-master-20260316 -- <file>  # get backup delta for any file
git log --oneline backup/local-master-20260316 ^origin/master  # full backup commit list
```

---

## Known Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Requirements version regression (PyJWT CVE) | Conflict resolution rule: always keep master requirements |
| Forecast audit manifest re-added to git | .gitignore rule + verify with `git status` before every commit |
| Reverted commits re-introduced | Skip commits `127652a` and `f849d3f` targets — `106b7aa` and `7416e17` were intentionally reverted |
| Test count regression | Require test count ≥ baseline at each phase exit |
| DB corruption during integration | `data/portfolio_maximizer.db` is in .gitignore — never touched by git ops |
| Phase 6 patch checksum mismatch | Always verify `.sha256.txt` before applying each patch bundle |

---

## Remaining Dependabot Item

1 HIGH severity vulnerability remains after the batch fix (GitHub alert on default branch).
Investigate and patch before or alongside Phase 5 PR.
Command: Check `https://github.com/mrbestnaija/portofolio_maximizer/security/dependabot`

---

## Document Maintenance

This document must be updated at each phase completion:
- Mark phase status: `IN PROGRESS` → `COMPLETE`
- Record actual test counts and gate results
- Note any deviations from plan with rationale

**Next action**: Begin Phase 1 — apply PnL-critical code files from backup branch.
