# Core Project Documentation (Institutional-Grade)

**Purpose**: Define the project’s canonical documentation set, evidence standards, and verification workflow so the repository remains publishable, auditable, and maintainable over long horizons (thesis/paper + production evolution).

This document is intentionally “policy-like”: it tells you what must be true for a result, report, or configuration change to be considered *research-grade* and *reproducible* in this codebase.

---

## 1. Canonical Documents (Single Source of Truth)

The project contains many documents; the following are the **core** ones that should remain consistent with the code and with each other:

- **Repo overview and onboarding**: `README.md`
- **Security policy (responsible disclosure)**: `SECURITY.md`
- **API key handling (local secrets)**: `Documentation/API_KEYS_SECURITY.md`
- **Version control policy (remote-first)**: `Documentation/GIT_WORKFLOW.md`
- **Architecture + navigation map**: `Documentation/arch_tree.md`
- **Current engineering health snapshot**: `Documentation/PROJECT_STATUS.md`
- **Chronological verification/evidence log**: `Documentation/implementation_checkpoint.md`
- **Research questions + experimental design**: `Documentation/RESEARCH_PROGRESS_AND_PUBLICATION_PLAN.md`
- **Automation/cron wiring (evidence generation)**: `Documentation/CRON_AUTOMATION.md`
- **Quant gating policy (GREEN/YELLOW/RED)**: `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md`
- **Numeric invariants (guard rails)**: `Documentation/NUMERIC_INVARIANTS_AND_SCALING_TESTS.md`
- **Metrics + evaluation definitions (math)**: `Documentation/METRICS_AND_EVALUATION.md`

If something is unclear or conflicting, treat this as the priority order for resolving ambiguity:

1. Code + tests (ground truth)
2. `Documentation/PROJECT_STATUS.md` (current intent)
3. `Documentation/METRICS_AND_EVALUATION.md` (definitions)
4. Other docs (supporting context)

---

## 2. Evidence Standard (What Counts as “Verified”)

A claim is “verified” only if it can be reproduced from the repo with a clear evidence trail:

### 2.1 Minimum provenance (required)

Every recorded experiment/run/report MUST include:

- **Commit SHA**: `git rev-parse HEAD`
- **Config snapshot identifiers**: exact file paths and content hashes for:
  - `config/quant_success_config.yml`
  - `config/signal_routing_config.yml` (or overrides)
  - `config/forecaster_monitoring.yml`
  - any run-specific overrides (hyperopt or per-run configs)
- **Environment**:
  - Python version (`python --version`)
  - OS (Windows/WSL/Linux)
  - `requirements.txt` version (and ideally a lock/snapshot)
- **Artifacts**:
  - paths under `logs/` / `reports/` / `visualizations/` that contain the outputs

### 2.2 “No recency bias” requirement

Institutional-grade reporting must not select windows only because they look good.

- Pre-declare evaluation windows (or window-selection rules).
- Report results across multiple regimes (bull/bear; low/high volatility), not only the latest window.
- Clearly separate:
  - **full-history** summary,
  - **recent-window** summary (e.g., last 60 days),
  - **hold-out** windows (walk-forward / blocked CV).

---

## 3. Verification Ladder (Fast → Strong)

Use this ladder to claim increasingly strong confidence.

### Level 0 — Static sanity (fast)

- Compile check: `python -m compileall -q ai_llm analysis backtesting etl execution forcester_ts models monitoring recovery risk scripts tools`

### Level 1 — Focused tests (targeted)

- Prefer a targeted test set relevant to the change (examples are documented in `Documentation/PROJECT_STATUS.md`).

### Level 2 — Integration verification (stronger)

- Run a broader integration slice (still smaller than “brutal”).

### Level 3 — Comprehensive brutal suite (evidence bundle)

- Run `bash/comprehensive_brutal_test.sh` (see evidence conventions below).
- Record the output directory and key artifacts (`test.log`, stage summary, DB snapshot).

### Level 4 — Automated evidence freshness (cron / scheduler)

- Cron/Task Scheduler runs keep evidence fresh:
  - `bash/production_cron.sh auto_trader_core`
  - Windows wrapper: `schedule_backfill.bat` (WSL-enabled default)

---

## 4. Artifact Conventions (Future-Proofing)

### CI verification (GitHub Actions)

CI is treated as part of the evidence trail for merges:

- Required merge blocker: `CI / test` must pass (dependency install + `pip check` + `pytest`).
- Automation workflows that depend on repository secrets (e.g., Projects sync) must skip cleanly when secrets are absent or PRs are from forks; they must not block merges.
- For dependency/security PRs: keep changes minimal, record which CVE/GHSA is addressed, and record what tests were run.

### 4.1 Recommended naming

- Brutal runs: `logs/brutal/results_<YYYYMMDD_HHMMSS>/`
- Reports: `reports/<topic>_<YYYYMMDD_HHMMSS>.md`
- Dashboards: `visualizations/<name>.json` and `<name>.png`

### 4.2 Minimum contents for a “bundle”

For any run you might cite later, keep at least:

- a human-readable summary (`final_report.md` or equivalent),
- raw logs,
- a DB snapshot (or the exact DB path used),
- the config files used (or hashes + commit SHA).

---

## 5. Math/Definitions Contract

All performance and validation claims must use the definitions in:

- `Documentation/METRICS_AND_EVALUATION.md` (canonical)
- and the code implementations in:
  - `etl/portfolio_math.py` (portfolio-level metrics)
  - `etl/statistical_tests.py` (DM-style tests, stability)
  - `etl/database_manager.py` (trade-level summaries such as profit factor)

If code and documentation disagree, update the documentation *or* fix the code; do not silently reinterpret metrics.

