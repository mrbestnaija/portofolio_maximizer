# Pipeline Live/Auto Run Journal

Tracks production-style executions that exercise every pipeline stage against live network sources. Use this log to verify connectivity, failover behaviour, and synthetic fallbacks when APIs misbehave.

---

## 2025-10-19 — pipeline_20251019_122249

- **Execution command**: `simpleTrader_env/bin/python scripts/run_etl_pipeline.py --tickers DOESNOTEXIST --start 2024-01-02 --end 2024-01-19 --execution-mode auto`
- **Mode**: `auto` (live-first with synthetic fallback)
- **Intent**: Force a live data failure (invalid ticker) to confirm synthetic fallback wiring

### Stage Timing Snapshot

| Stage | Duration (s) |
| --- | ---: |
| data_extraction | 12.48 |
| data_validation | 0.00 |
| data_preprocessing | 0.04 |
| data_storage | 0.17 |

### Observations

- `yfinance` extraction failed with `YFTzMissingError`, triggering `auto_fallback_engaged` and synthetic data generation in-line.
- Synthetic dataset contained 14 business-day rows; persisted across DB, checkpoints, and Parquet splits.
- Auto mode preserved continuity—subsequent stages processed synthetic data with zero adjustments.

### Artifacts & Telemetry

- Database rows committed to `data/portfolio_maximizer.db` (source flagged as `synthetic(auto-fallback)`).
- Checkpoint: `data/checkpoints/pipeline_20251019_122249_data_extraction_20251019_122305.pkl`
- Parquet splits written to `data/training/training_20251019_122305_20251019.parquet` (9 rows), validation (2 rows), testing (3 rows).
- Event log entries: `logs/events/events.log` includes the fallback marker and per-stage timings.

### Follow-Ups

- Confirm API credentials for non-default providers (Alpha Vantage, Finnhub) before running in `--execution-mode live`.
- Add regression coverage to ensure new `auto` mode is exercised in CI using mocked extractor failures.

---

## Update Workflow

1. Prefer `./bash/run_pipeline_live.sh` for routine live/auto executions (env overrides: `TICKERS`, `EXECUTION_MODE`, `ENABLE_LLM`, etc.).
2. If a fallback occurs, capture the reason from `logs/events/events.log` and record a short blurb in this journal.
3. Cross-check database and Parquet outputs to verify the synthetic label sources are traceable.
4. For pure offline validations, continue to log runs separately in `Documentation/pipeline_dry_run_log.md`.
