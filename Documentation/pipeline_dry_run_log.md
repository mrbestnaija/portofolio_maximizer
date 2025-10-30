# Pipeline Dry-Run Journal

Traceable record of end-to-end dry-runs that validate portfolio pipeline wiring against the **UNIFIED_ROADMAP** expectations. Update this log after each invocation of `bash/run_pipeline_dry_run.sh` so stakeholders can audit stage coverage, timings, and LLM outputs over time.

---

## 2025-10-19 — pipeline_20251019_082659

- **Execution command**: `simpleTrader_env/bin/python scripts/run_etl_pipeline.py --tickers AAPL,MSFT --start 2024-01-02 --end 2024-01-19 --dry-run --enable-llm`
- **Invocation method**: manual CLI run (synthetic OHLCV data flow, checkpoints enabled, sequential execution)
- **LLM context**: qwen:14b-chat-q4_K_M served locally via Ollama

### Stage Timing Snapshot

| Stage | Duration (s) |
| --- | ---: |
| data_extraction | 0.07 |
| data_validation | 0.00 |
| data_preprocessing | 0.02 |
| llm_market_analysis | 34.44 |
| llm_signal_generation | 44.78 |
| llm_risk_assessment | 31.56 |
| data_storage | 0.03 |

### LLM Outputs (Database Snapshot)

- `llm_analyses`: AAPL (bearish, strength 5, 17.88s), MSFT (bearish, strength 5, 16.51s)
- `llm_signals`: AAPL (SELL, 80% confidence, latency 22.62s), MSFT (SELL, 80% confidence, latency 22.12s)
- `llm_risks`: AAPL high-risk assessment logged on 2025-10-17 (MSFT entry pending; investigate risk persistence wiring)

### Generated Artifacts

- Parquet splits: `data/training/training_20251019_082851_20251019.parquet` (19 rows), `data/validation/validation_20251019_082851_20251019.parquet` (4 rows), `data/testing/testing_20251019_082851_20251019.parquet` (5 rows)
- Database: `data/portfolio_maximizer.db` populated with OHLCV cache + latest LLM outputs
- Checkpoints: `data/checkpoints/pipeline_20251019_082659_*` recorded after each required stage
- Logs: `logs/events/events.log`, `logs/pipeline.log` (use `bash/run_pipeline_dry_run.sh` to capture future console output under `logs/dry_runs/`)

### Follow-Ups

- Verify `llm_risks` table captures fresh assessments for all tickers; only AAPL updated during this dry-run.
- Track latency trends for LLM stages; current aggregate LLM time ≈ 110s for two tickers.
- Consider archiving legacy parquet splits to keep storage directories tidy between iterations.

---

## Update Workflow

1. **Run the script**: `bash/run_pipeline_dry_run.sh` (override defaults via `TICKERS`, `START_DATE`, `END_DATE`, `ENABLE_LLM`, etc. as needed).
2. **Capture identifiers**: record the emitted pipeline ID, log path, and any notable console warnings.
3. **Summarize metrics**: add a new dated section above, mirroring the headings used here (Stage Timing Snapshot, LLM Outputs, Generated Artifacts, Follow-Ups).
4. **Highlight deviations**: flag missing database entries, abnormal latencies, or checkpoint issues to feed back into roadmap tasks.
5. **Iterate**: retain chronological order (newest first) so the file reads as an audit trail for Phase 5.4 readiness.
