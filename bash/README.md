# Bash Entry Points (Operations + Evidence)

This directory contains the repository’s human-friendly operational entrypoints: pipeline runs, automation helpers, and evidence/test harnesses.

## Canonical entrypoints (preferred)

### Pipeline

- `bash/run_pipeline.sh` – unified launcher (modes: `live`, `auto`, `synthetic`, `dry-run`)
- `bash/run_pipeline_live.sh` – stable wrapper for live runs (delegates to `bash/run_pipeline.sh --mode live`)
- `bash/run_pipeline_dry_run.sh` – stable wrapper for dry runs (delegates to `bash/run_pipeline.sh --mode dry-run`)

### Trading / orchestration

- `bash/run_auto_trader.sh` – paper trading loop (TS-first, LLM optional fallback)
- `bash/run_end_to_end.sh` – pipeline → auto-trader → dashboard refresh
- `bash/run_post_eval.sh` – higher-order post-eval / hyperopt orchestration

### Automation (cron / scheduler)

- `bash/production_cron.sh` – task multiplexer (see `Documentation/CRON_AUTOMATION.md`)
- `bash/run_core_auto_trader_once.sh` – WSL-friendly wrapper for `production_cron.sh auto_trader_core`

### Evidence and test gates

- `bash/comprehensive_brutal_test.sh` – canonical “brutal” evidence bundle
- `bash/full_test_run.sh` – full pytest suite runner
- `bash/test_profit_critical_functions.sh` – profit/PnL correctness subset (money-impacting invariants)

## Synthetic workflows

- `bash/run_synthetic_smoke.sh` – generate a synthetic dataset, then run the pipeline on it
- `bash/run_synthetic_latest.sh` – run the pipeline on the persisted dataset id `latest` (expects synthetic data already exists; use `bash/production_cron.sh synthetic_refresh` to regenerate)

## LLM checks (consolidated)

- `bash/run_llm_tests.sh`
  - `healthcheck` – calls `bash/ollama_healthcheck.sh`
  - `quick` – small pytest slice for `tests/ai_llm/`
  - `full` – full `tests/ai_llm/` suite

The legacy scripts `bash/test_llm_quick.sh`, `bash/test_llm_integration.sh`, and `bash/verify_fixes.sh` are retained as wrappers for backward compatibility.

## Notes / conventions

- All scripts assume the project virtual environment exists at `simpleTrader_env/` (see `requirements.txt`).
- Logs and evidence artifacts should land under `logs/`, `reports/`, and `visualizations/` (see `Documentation/CORE_PROJECT_DOCUMENTATION.md`).
- Shared helper functions live in `bash/lib/common.sh`.

