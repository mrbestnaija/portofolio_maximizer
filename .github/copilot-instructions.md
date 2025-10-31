## Portfolio Maximizer – Developer Briefing

Read this once. If you go off-script, you will waste everyone’s time.

### 1. Quick Truths
- This project is **local-first**, opinionated, and already fragile. Assume nothing “just works”.
- We are in **Phase 5.x** (see `Documentation/AGENT_INSTRUCTION.md`). Live trading is forbidden until a 30+ day backtest shows >10% annualized returns. You will not see that data here.
- LLM code runs through **Ollama** on the same box. It is slow, occasionally offline, and every expensive prompt will be on you to fix.
- Tests are heavy (≈50 s). Run them only when your change touches logic. No excuses for skipping `pytest`.

### 2. Project Insight & Phase Discipline
- **Phase ledger (most recent checkpoint `2025-10-22`)**
  1. ✅ Phase 1 – ETL foundation (extraction, validation, preprocessing, storage)
  2. ✅ Phase 2 – Analysis framework (time-series analytics, statistical tooling)
  3. ✅ Phase 3 – Visualization framework (publication-grade plots)
  4. ✅ Phase 4 – Caching overhaul (20× speedup, 100 % hit rate)
  5. ✅ Phase 4.5 – Time-series cross-validation (expanding k-fold, backward compatible)
  6. ✅ Phase 4.6–4.8 – Multi-source architecture, config-driven CV, checkpointing/logging
  7. ✅ Phase 5.1–5.5 – Alpha Vantage/Finnhub adapters, local LLM integration, profit calc fix, Ollama health, error & performance monitoring
- **Next major gate: Phase 5.x portfolio optimisation** (Markowitz, risk parity). Blocked on validated signals and paper trading engine. No new trading logic without meeting those prerequisites.
- **Progress tracking expectations**
  - Update `Documentation/implementation_checkpoint.md` when a phase milestone changes.
  - Reflect risk/LLM metrics in `logs/` and monitoring dashboards; no silent regressions.
  - Document setbacks in `Documentation/NEXT_TO_DO_SEQUENCED.md` or `FAILURES.md` so phase reviews stay honest.
  - Keep visualization deliverables (`visualizations/`, reporting scripts) aligned with current phase metrics—stale charts are worse than no charts.
- **Reality filter:** if your proposal doesn’t move the current phase forward, park it. “Future” ideas belong in docs, not the codebase.

### 3. Architecture Reality Check
- ETL layers live in `etl/`. They are long, interdependent, and partially vectorized. Breaking `data_storage` or `yfinance_extractor` will cascade across everything.
- LLM modules (`ai_llm/`) implement: client, market analysis, signal generation, risk assessment, validators, and performance monitoring. They are plugged into the pipeline via `scripts/run_etl_pipeline.py`; feature flags live in `config/llm_config.yml`.
- Database writes go through `etl/database_manager.py`. We now track signal validations and allow `risk_level='extreme'`. If you mess with the schema, add migrations and tests.

### 4. Guardrails You Cannot Ignore
1. **Feature flags**: any LLM or trading feature must be disabled by default. Respect config-driven toggles.
2. **Backtests**: do not claim improvements without quantitative proof (≥30 days, >10% annualized, beats buy‑and‑hold). The repo has no magical dataset—if you cannot prove it, do not promise it.
3. **No live trading**: every “execute” suggestion must stop at paper trading unless the docs explicitly say otherwise.
4. **Phase discipline**: confirm current phase in `Documentation/implementation_checkpoint.md`. If your idea isn’t in scope, park it in TODO form, not code.
5. **Cost ceiling**: keep solutions within $0/month budget. No cloud LLMs, no SaaS “shortcuts”.

### 5. Day-to-Day Workflow (No Surprises)
```powershell
python -m venv simpleTrader_env
simpleTrader_env\Scripts\Activate.ps1
pip install -r requirements.txt

# Pipeline
python scripts/run_etl_pipeline.py --execution-mode auto --enable-llm
# Synthetic fallback only
python scripts/run_etl_pipeline.py --execution-mode synthetic

# Tests
pytest -q
pytest tests/ai_llm -q
```
If Ollama is down, fix that first: `ollama serve`, `curl http://localhost:11434/api/tags`.

### 6. Known Landmines
- **Ollama client** fails fast. If it cannot reach the server or find the model, the pipeline stops. Keep it that way.
- **Signal validation** is not optional. `SignalQualityValidator` decides if an LLM signal survives. Always feed it fresh price data, log the result, and store it.
- **Database migrates on startup**. If you touch schema constraints, extend migrations and add regression tests (`tests/etl/test_database_manager_schema.py`).
- **Long-running tests**: integration suites hammer SQLite and the LLM pipeline. Consider `pytest <target>` while iterating, but run the full suite before you sign off.
- **Docs lag reality**. Before trusting an instruction, skim the referenced file to make sure it still matches the code.

### 7. How to Be Useful
1. **Assess current state**: read the touched files, check `git status`, and scan logs (`logs/llm_errors.log`).
2. **Propose in plain language**: explain why you’re touching something, what breaks if you don’t, and how you’ll prove it works.
3. **Respect the tests**: add or adjust tests before you refactor. No green tests → no merge.
4. **Leave breadcrumbs**: meaningful commit messages, crisp PR notes, and doc updates when behaviour changes.

### 8. Off-Limits Until Proven Otherwise
- Reinforcement learning, external broker integrations, cloud LLMs, or any “smart” automation of trades.
- Silent schema changes, background migrations, or destructive operations (`git reset --hard`, `rm -rf`) without explicit approval.
- Optimistic claims about profitability, throughput, or latency without measured evidence.

### 9. Mindset
- Be blunt about what is broken. “Works on my machine” burns time later.
- Optimize for reliability over cleverness.
- Flag anything suspicious. Surprises buried in the logs grow into outages later.
- When in doubt: confirm the phase, read the docs, write a test, then fix it.

You are here to keep the system honest, stable, and verifiable. Deliver that, and you’re useful. Anything else is noise.
