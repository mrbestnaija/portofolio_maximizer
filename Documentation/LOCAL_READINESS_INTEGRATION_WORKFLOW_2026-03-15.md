# Local Readiness Integration Workflow (2026-03-15)

Status: active local-only integration workflow
Authority: operational handoff for Claude Code + human review

## Policy

Use the most complete readiness-focused remote branch as the reference baseline, but stop
creating new remote branches for this readiness lane.

From this point forward:

- reference branch: `origin/codex/evidence-core-lane-20260314`
- canonical merge target: `origin/master`
- local integration branch: one fresh local-only branch created from `origin/master`
- integration method: reviewed commit selection and patch-based application only

Do not push new readiness-integration branches to `origin`.

## Interpreter Rule

For local Windows execution in this readiness lane, do not rely on whichever `python`
appears first on `PATH`.

Use the repo-pinned interpreter only:

- direct path: `.\simpleTrader_env\Scripts\python.exe`
- checked-in wrapper: `.\scripts\repo_python.ps1`

The wrapper is intentionally hard-coded to `simpleTrader_env` and fails if that interpreter
is missing.

## Reference Baseline Decision

The current reference baseline is fixed to:

- `origin/codex/evidence-core-lane-20260314`

Why this branch is the baseline:

- it is the most recent readiness-focused branch in the repo (`2026-03-15`, `4b97b4e`)
- it carries the latest evidence-contract, gate-truth, and admission-contract work
- it is materially ahead of `origin/master` (`128` unique commits vs `origin/master`)
- it is ahead of the nearest readiness ancestor branch `origin/agent-b-hardening-20260307`
  in freshness and coverage

This branch is the inspection baseline only. It is not the merge target.

## Why This Workflow

The readiness lane currently spans:

- evidence-contract hardening
- OpenClaw ops/readiness surfaces
- dashboard/runtime truth
- shared gate-adjacent files with parallel edits

The reference branch has the best overall coverage of these concerns, but it is not a safe
blind merge target for `master`. The local clone also does not have a clean merge-base path
for a simple branch merge from that line into `origin/master`.

Therefore the safe path is:

1. treat `origin/codex/evidence-core-lane-20260314` as the review baseline
2. review commits or patch bundles one concern at a time
3. apply only approved hunks onto one clean local integration branch from `origin/master`
4. verify there
5. integrate to `master` only after human + Claude review

## Branching Rule From This Point Forward

- No new readiness branches are pushed to `origin`.
- The only new readiness branch allowed is a single fresh local integration branch.
- Reviewed work enters that branch by patch application or explicitly reviewed commit
  selection only.
- Any new agent work for this lane should produce a patch bundle, not a new remote branch.

## Two-Terminal Split

Use two local terminals/worktrees.

### Terminal A - Lift Semantics / Measurement terminal

Purpose:

- inspect `origin/codex/evidence-core-lane-20260314`
- generate diffs and patch bundles
- review lift-semantics and measurement candidate commits
- answer the question: "if lift fails, is the failure mathematically real?"
- never use this terminal as the merge target

Primary concern set:

- readiness denominator semantics
- structured lift/RMSE summary truth
- `production_audit_gate.py` summary consumption
- baseline-definition parity (`BEST_SINGLE` vs other baselines)
- lift classification and measurement reporting

Recommended branch state:

- current evidence/reference workspace
- dirty edits allowed for inspection only

### Terminal B - Lift Inputs / Evidence Quality terminal

Purpose:

- clean application of approved patches
- targeted tests
- fast-lane regression
- answer the question: "can the gate trust the evidence population being measured?"
- final review-ready integration branch state

Primary concern set:

- producer-written admission contract
- evidence eligibility vs accepted-noneligible vs quarantined
- missing execution metadata
- duplicate conflict handling
- clean cohort intake and proof-loop isolation

Required branch state:

- fresh local branch from `origin/master`
- no unrelated edits
- patch application only

## Branch and Worktree Shape

Recommended local branch name:

- `integration/readiness-20260315`

Recommended local worktree path:

- `C:\Users\Bestman\personal_projects\pmx_readiness_integration`

This branch is local-only unless a human explicitly decides to publish it later.

## Current Local Setup

Prepared locally on 2026-03-15:

- Terminal A / lift semantics and measurement workspace:
  - branch: `codex/evidence-core-lane-20260314`
  - role: review baseline, measurement diff generation, patch preparation
- Terminal B / lift inputs and evidence-quality workspace:
  - branch: `integration/readiness-20260315`
  - upstream: `origin/master`
  - path: `C:\Users\Bestman\personal_projects\pmx_readiness_integration`
  - role: apply reviewed patches only, run tests, prepare master-ready integration state

## Exact Local Setup

Reference terminal commands:

```powershell
git fetch origin
git checkout codex/evidence-core-lane-20260314
git status --porcelain
```

Integration terminal setup:

```powershell
git fetch origin
git worktree add -b integration/readiness-20260315 C:\Users\Bestman\personal_projects\pmx_readiness_integration origin/master
cd C:\Users\Bestman\personal_projects\pmx_readiness_integration
git status --porcelain
```

Expected result:

- Terminal A remains on the reference branch for semantics and measurement review
- Terminal B remains on `integration/readiness-20260315`
- Terminal B stays clean except for reviewed patch application around evidence quality and approved lift fixes

## Allowed Integration Inputs

Allowed:

- reviewed patch bundles under `Documentation/patch_bundles/`
- reviewed file-scoped diffs from the reference branch
- selectively approved commits, applied by patch or cherry-pick after review

Not allowed:

- blind merge of `origin/codex/evidence-core-lane-20260314`
- blind merge of any other remote feature branch
- direct push from a dirty workspace to `origin/master`
- bundling unrelated operational logs or caches

## Patch Intake Methods

Preferred methods, in order:

1. `git apply --index --3way <patch>`
2. `git am <mbox patch>` for reviewed commit exports
3. manual hunk application only when Claude Code has reviewed the diff first

Patch generation examples from Terminal A:

```powershell
git format-patch -1 4b97b4e --stdout > Documentation\patch_bundles\evidence_core_4b97b4e.patch
git diff origin/master...origin/codex/evidence-core-lane-20260314 -- scripts/check_forecast_audits.py > Documentation\patch_bundles\check_forecast_audits_reviewed.patch
```

Patch application examples in Terminal B:

```powershell
git apply --index --3way C:\Users\Bestman\personal_projects\portfolio_maximizer_v45\portfolio_maximizer_v45\Documentation\patch_bundles\check_forecast_audits_reviewed.patch
git am C:\Users\Bestman\personal_projects\portfolio_maximizer_v45\portfolio_maximizer_v45\Documentation\patch_bundles\evidence_core_4b97b4e.patch
```

If a patch does not apply cleanly, stop and regenerate a narrower reviewed patch. Do not
solve it by merging the whole reference branch.

Verification examples in this lane should use the pinned interpreter:

```powershell
.\scripts\repo_python.ps1 -m pytest tests/scripts/test_check_forecast_audits.py tests/scripts/test_production_audit_gate.py -q
.\scripts\repo_python.ps1 scripts/production_audit_gate.py --unattended-profile
```

## Current Reference Priority

Use this order when harvesting reviewed work:

1. Lift semantics / measurement reviewed hunks
   - `scripts/check_forecast_audits.py`
   - `scripts/production_audit_gate.py`
   - `models/time_series_signal_generator.py`
   - `scripts/check_model_improvement.py`
2. Lift inputs / evidence-quality reviewed hunks
   - `scripts/run_auto_trader.py`
   - `scripts/replay_trade_evidence_chain.py`
   - evidence admission and hygiene tests
3. evidence-core lane additions and reviewed tracked-file hunks
   - `Documentation/EVIDENCE_CORE_INTEGRATION_HANDOFF_2026-03-14.md`
4. OpenClaw ops/readiness patch bundle after lift integrity is stable
   - `Documentation/OPENCLAW_OPS_PATCH_BUNDLE_2026-03-15.md`
5. ancestor/reporting branches only as supporting sources
   - `origin/agent-b-hardening-20260307`
   - `origin/codex/live-denominator-dashboard-docs-20260307`

## Integration Sequence

1. Start Terminal B from `origin/master`
2. Apply one reviewed patch bundle or one reviewed concern at a time
   - when the patch file exists only in Terminal A's workspace, apply it by absolute path
3. Start with lift semantics / measurement fixes
4. Run targeted tests for that concern
5. Move next to lift inputs / evidence-quality fixes
6. Keep notes on what was intentionally deferred
7. Run the integration checklist on the local integration branch

Minimum checks:

- `python scripts/run_all_gates.py --json`
- `python -m integrity.pnl_integrity_enforcer`
- targeted tests for each applied concern
- `python -m pytest -m "not gpu and not slow" --tb=short -q`

## Gate-Lift Decision Rule

Use the terminal split to answer these two questions separately before concluding that a
lift failure is real:

1. **Lift Semantics / Measurement**
   - Are the denominator rules correct?
   - Are structured lift and RMSE fields preserved on both pass and fail?
   - Is the same baseline definition used in signal evaluation and audit measurement?
   - Is the gate reading structured summary fields rather than inference from stdout?

2. **Lift Inputs / Evidence Quality**
   - Are production artifacts admitted with producer-written eligibility fields?
   - Are missing execution metadata and duplicate conflicts preserved unchanged to the gate?
   - Is the measured cohort clean, isolated, and free of mixed-context contamination?
   - Is the `accepted -> eligible -> matched` funnel trustworthy?

Do not call a lift failure "real" until both sides are green enough to trust the result.

## Integration Log Requirement

For each patch applied to the local integration branch, record:

- source branch or commit
- patch filename
- files touched
- tests run
- deferred files or hunks

This log should live alongside the integration branch notes and be updated before any
human + Claude review for `master`.

## Current Decision

For readiness work, the reference baseline is fixed to:

- `origin/codex/evidence-core-lane-20260314`

For actual integration, the working destination is fixed to:

- local branch `integration/readiness-20260315` from `origin/master`

Until this lane is merged, continue using patch-based local integration only.
