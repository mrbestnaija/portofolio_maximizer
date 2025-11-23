## Portfolio Maximizer – Developer Briefing (Updated 2025-11-18)

Read this once. If you go off-script, you waste everyone’s time.

### 1. Quick Truths
- **Local-first + fragile**: assume nothing “just works”. Every service (SQLite, Ollama, scheduled scripts) lives on this workstation.
- **Phase status**: Per `Documentation/implementation_checkpoint.md` (v6.9, 2025-11-14), we are still in Phase 5.x. Portfolio optimisation, monetization, and live trading all stay blocked until we log ≥30 calendar days of backtests with annual_return ≥10 %, Sharpe ≥1.0, and max_drawdown ≤25 % (see `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md`’s monetization gate).
- **LLM runtime**: All generative code paths run through local Ollama. Expect 15–38 s latency (≈1.6–3.1 tok/s). If `ollama serve` or its models are missing, fix that before touching code.
- **Test cost**: Full pytest takes ≈50 s and hammers SQLite plus the TS/LLM stack. Run targeted suites while iterating, then the full run before handoff. “Didn’t run tests” is not acceptable.

### 2. Project Insight & Phase Discipline
- **Phase ledger (checkpoint 2025-11-14)**  
  1. ✅ ETL foundation (extraction, validation, preprocessing, storage)  
  2. ✅ Analysis framework (time-series analytics, statistical tooling)  
  3. ✅ Visualization framework (publication-grade plots)  
  4. ✅ Caching overhaul (≈20× faster, 100 % hit rate)  
  5. ✅ Phase 4.5–4.8: multi-source architecture, config-driven CV, checkpointing/logging  
  6. ✅ Phase 5.1–5.5: AlphaVantage/Finnhub adapters, local LLM integration, profit calc fix, Ollama health monitoring, error telemetry  
- **Current blockers (see brutal log 2025-11-15)**  
  - SQLite production DB corrupted (`database disk image is malformed`). `etl/database_manager.py` now traps that error, but validation requires a rebuilt DB and restored checkpoints.  
  - Stage 7 MSSA `change_points` bug, Matplotlib `autofmt_xdate(axis=…)` crash, and deprecated Period handling are fixed, yet `scripts/backfill_signal_validation.py` still needs timezone-aware timestamps before nightly jobs resume.  
  - Paper-trading engine + monetization gate remain TODO; no new trading logic without them.  
- **Progress hygiene**  
  - Update `Documentation/implementation_checkpoint.md` + `Documentation/NEXT_TO_DO_SEQUENCED.md` whenever a phase milestone, regression, or gate condition changes.  
  - Instrument results: `forcester_ts/instrumentation.py` must keep emitting RMSE / sMAPE / tracking error / duration metrics into `logs/forecast_audits/`. If telemetry is missing, fix that before adding features.  
  - Visualization/monitoring artifacts must mirror current metrics; stale PNGs or dashboards are worse than none.  
- **Reality filter**: If a proposal doesn’t advance Phase 5.x deliverables (validated signals, paper trading, monetization gate), park it as documentation—not code.

### 3. Architecture Reality Check
- **ETL** (`etl/`): monolithic, partially vectorized. Touching `data_storage`, `portfolio_math`, or any extractor has system-wide blast radius.  
- **Time-series stack** (`forcester_ts/`, `models/time_series_signal_generator.py`): SARIMAX + SAMOSSA + MSSA-RL ensemble is now the default signal source; LLM is fallback only. Every forecast run must log RMSE/sMAPE/tracking error and persist provenance for routing.  
- **LLM modules** (`ai_llm/`): client, signal generator, validators, risk assessment, monitoring. They plug into `scripts/run_etl_pipeline.py` behind feature flags in `config/llm_config.yml`.  
- **Database** (`etl/database_manager.py`): schema migrations auto-run. Risk assessments now accept `risk_level='extreme'`, and time-series/LLM signals share unified tables. Schema edits require migrations + regression tests.  
- **Automation**: `scripts/run_auto_trader.py` chains extraction → validation → forecasting → routing → execution with optional LLM fallback. Do not bloat the orchestrator; add helpers instead.

### 4. Guardrails You Cannot Ignore
1. **Feature flags**: any experimental, LLM, or monetization logic must be config-driven and off by default.  
2. **Quant proof**: no claims without logged evidence. Minimum bar: backtest window ≥30 days, annual_return ≥10 %, Sharpe ≥1.0, max_drawdown ≤25 %, and benchmark outperformance.  
3. **Paper trading only**: execution endpoints stop at simulated brokers until profitability + compliance gates are met.  
4. **Phase discipline**: confirm scope in `Documentation/implementation_checkpoint.md` before coding. Out-of-phase ideas belong in TODO lists.  
5. **Cost ceiling**: $0/month. No cloud LLMs, SaaS hooks, or paid data sources.  
6. **Reward-to-effort alignment**: new docs/code must reference `Documentation/REWARD_TO_EFFORT_INTEGRATION_PLAN.md` to ensure monetization, automation, and sequencing tasks respect its line budgets and monetization gate.  
7. **Environment**: always work inside `.simpleTrader_env`. No parasite environments.

### 5. Day-to-Day Workflow (No Surprises)
```powershell
python -m venv simpleTrader_env
simpleTrader_env\Scripts\Activate.ps1
pip install -r requirements.txt

# Primary pipeline
python scripts/run_etl_pipeline.py --execution-mode auto --enable-llm

# Synthetic fallback
python scripts/run_etl_pipeline.py --execution-mode synthetic

# Tests (targeted first, full suite before handoff)
pytest -q
pytest tests/ai_llm -q
```
If Ollama misbehaves: `ollama serve`, `curl http://localhost:11434/api/tags`, verify model exists.

### 6. Known Landmines
- **Ollama client** fails fast by design. Missing server/model = pipeline stop. Leave it that way.  
- **SignalQualityValidator** is mandatory. Every signal must log validation decisions + metrics (win rate, Kelly sizing, bootstrap diagnostics).  
- **Database migrations** run on startup; schema experiments without migrations/tests will brick the system. Use `tests/etl/test_database_manager_schema.py`.  
- **Long tests**: integration suites hammer SQLite, TS ensemble, and LLM. Use targeted pytest modules during dev, but always run full suite pre-merge.  
- **Docs lag**: before trusting any instruction, open the referenced file and confirm it matches code. Update stale docs immediately.

### 7. How to Be Useful
1. **Assess reality**: check `git status -sb`, open the touched files, skim `logs/pipeline_run.log`, `logs/llm_errors.log`, and `logs/forecast_audits/*.json`.  
2. **Quantify impact**: describe changes in terms of measurable metrics (latency, RMSE, Sharpe delta, drawdown). Describe how you will validate them.  
3. **Respect tests**: add/modify tests before refactoring. If tests are flaky, stabilize them rather than ignoring failures.  
4. **Leave breadcrumbs**: meaningful commit messages, PR summaries, and doc updates are mandatory. Every change touching monetization or automation must cite the reward-to-effort plan.  
5. **Monitor budgets**: monetization-related code must stay within the 700 LOC budget and honor the gate thresholds.

### 8. Off-Limits Until Proven Otherwise
- Reinforcement learning, cloud LLMs, copy-trading, or “smart” live execution.  
- Schema changes without migrations/tests.  
- `git reset --hard`, `rm -rf`, or destructive ops without explicit approval.  
- Claims about profitability, throughput, or latency without reproducible measurements.  
- Paid infra or APIs exceeding the $0/month constraint.

### 9. Mindset
- Be blunt about regressions. “Works on my machine” wastes time later.  
- Reliability beats cleverness. Optimize for deterministic, testable code.  
- Treat logs as first-class artifacts: RMSE/sMAPE/TE metrics, signal validation stats, and monetization gate decisions must be reproducible.  
- When in doubt: confirm the phase, read the source docs, write/extend a test, then fix the issue.

You are here to keep the system honest, statistically justified, and verifiable. Deliver that, and you’re useful. Anything else is noise.
