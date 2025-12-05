# Portfolio Maximizer Reward-to-Effort Integration Plan
**Updated:** 2025-11-18  
**Authoring context:** Implementation checkpoint v6.9, Optimization Implementation Plan, AGENT_INSTRUCTION guardrails, Monetization & Local Adaptation To-Do brief.

This playbook integrates monetization, automation, backtesting realism, and local adaptation requirements into a single execution roadmap. It keeps the focus on the highest reward-to-effort work, leverages existing infrastructure, and enforces the profitability gates defined in `AGENT_INSTRUCTION.md` and the Monetization brief.

---

## 1. Mission & Current State Snapshot
- **Goal:** Deliver monetizable signal outputs (alerts, reports, licensing) using proven foreign-market data while creating a fast handoff path for Nigerian manual execution.
- **Context:** Time-series, portfolio, and LLM stacks are in place (per `implementation_checkpoint.md`); the latest brutal suite run (`logs/brutal/results_20251204_190220/`) is structurally green (profit-critical, ETL, Time Series, routing, integration, security all passing), but earlier brutal runs exposed SQLite corruption, MSSA change-point edge cases, visualization regressions, and validator warnings that have since been remediated. Profitability proof and global quant validation health (currently RED per `Documentation/QUANT_VALIDATION_MONITORING_POLICY.md`) remain the top gating factors.
- **Key guardrails:** Free/low-cost infra only, phase-gate discipline, max line-count budgets, automation-first mindset, configuration-driven design, and backtest realism before new features.

---

## 2. Profitability & Complexity Guardrails
1. **Monetization Gate Enforcement**
   - Build `src/core/monetization_gate.py` (≤250 LOC) to classify `BLOCKED | EXPERIMENTAL | READY_FOR_PUBLIC` using recent performance metrics (`annual_return`, `sharpe`, `max_drawdown`, `trade_count`).
   - Provide CLI `python -m tools.check_monetization_gate --window 365` that prints metrics, explains gating failures, and exits non-zero when blocked.
   - Block monetization features whenever `annual_return < 0.10`, `sharpe < 1.0`, or `max_drawdown > 0.25`.
2. **Line-Count Budget**
   - Aggregate all new monetization code (alerts, licensing helpers, adapters) ≤700 LOC and guard with `tests/performance/test_line_budget_monetization.py`.
3. **Reward-first Prioritization**
   - Tackle features with the highest profit/reliability per unit effort before advanced ML or new data sources. Use foreign market coverage first, then adapt to Nigeria.
4. **Automation Mandate**
   - Prefer automated ETL, alert dispatch, and paper trading hooks to reduce manual cycles. Re-use Backtrader, MLfinlab, statsmodels assets before writing bespoke engines.

---

## 3. High-Impact Workstreams

### 3.1 Profitability & Complexity Gate
- **Why:** Prevent alert/monetization rollout when the core strategy is unproven.
- **Deliverables:**
  - `MonetizationStatus` enum and gate service (≤250 LOC).
  - CLI + JSON summary so scripts (`tools.push_daily_signals`, `tools.monetization_reality_check`) can fail fast.
  - Regression tests covering pass/block boundaries and metric fallbacks when DB rows are missing.
- **Reward-to-effort:** Small, centralized component unlocks safe automation and keeps future monetization work dormant until profitable.

### 3.2 Backtesting Realism & Risk Controls
- **Objective:** Ensure reward projections survive transaction costs, slippage, and liquidity limits—especially for NGX instruments.
- **Actions:**
  - Extend the Time Series backtesting harness (reuse Backtrader where feasible) with configurable cost/slippage models and liquidity gates per `config/markets/*.yml`.
  - Integrate a paper trading engine stub that simulates fills with slippage + fees using recorded spreads from recent brutal runs.
  - Update risk metrics to log per-strategy `max_drawdown`, `CVaR`, `win_rate`, and accuracy vs. actual fills (leveraging `forcester_ts/instrumentation.py` outputs).
  - Add regression tests covering Nigerian fee structures and thin-volume instruments to prevent unrealistic signals.

### 3.3 Standardize Signal Export Layer
- **Goal:** Make `TradeSignal` the single truth for downstream monetization, reports, and brokers.
- **Scope:**
  - `src/signals/domain.py`: Define `TradeSignal` dataclass with timestamp, symbol, side, entry/SL/TP, confidence, timeframe, strategy_id, notes.
  - `src/signals/exporter.py`: Provide dict/CSV/JSON exporters and `to_human_text(style="short|detailed")` for alerts.
  - Refactor existing pipelines/backtests to emit only `TradeSignal` objects before alert/report modules consume them.
  - Tests for serialization round-trips and to ensure no alert module re-implements formatting.
- **Reward-to-effort:** One conversion point slashes duplicated formatting logic and keeps CLI/alert modules lean.

### 3.4 Alert & Notification Stack (Indirect Monetization First)
- **Aim:** Offer Telegram/email ready signals using existing foreign-market strategies and keep WhatsApp manual.
- **Components:**
  - `src/alerts/telegram_notifier.py` (≤150 LOC) using Bot Token/Chat ID from `config/monetization.yml`.
  - `src/alerts/email_notifier.py` (≤150 LOC) using stdlib `smtplib`; no third-party paid services.
  - Optional `src/alerts/whatsapp_stub.py` (≤80 LOC) to output markdown text for manual pasting.
  - `src/alerts/alert_hub.py` orchestrator that reads enabled channels, triggers the monetization gate, and batches messages.
  - CLI `python -m tools.push_daily_signals --config config/monetization.yml --date YYYY-MM-DD`.
- **Safeguards:** Monetization gate rejection stops dispatch, logs reason, and returns non-zero exit status.

### 3.5 Strategy Packaging & Licensing Readiness
- **Objective:** Generate human-friendly performance summaries for prospective users, investors, or B2B deals.
- **Tasks:**
  - `src/reports/strategy_packager.py` (200–250 LOC) to produce JSON + Markdown factsheets including annualized return, Sharpe, max drawdown, win rate, avg R:R, worst trades, and disclaimers.
  - Ensure reports redact secrets (API keys, URLs) and include licensing cautions.
  - Add tests verifying report creation with sample historical data and raising clear errors when metrics missing.

### 3.6 Foreign Broker / Demo Bridges
- **Reason:** Provide one-click conversion from signal to foreign-market execution (demo-first) to raise realized reward.
- **Deliverables:**
  - `src/execution/ctrader_bridge.py` (≤200 LOC) converting `TradeSignal` to cTrader CSV + copy-paste text instructions.
  - `src/execution/broker_interface.py` (≤150 LOC) defining `IBrokerClient` (place/close/get positions). Keep implementations optional extras.
  - Guard any external SDKs behind extras in `pyproject.toml` and align with free tiers.

### 3.7 Nigerian Local Adaptation Path
- **Goal:** Use foreign-trained models while enabling manual NGX execution until stable APIs exist.
- **Artifacts:**
  - Market profiles (`config/markets/foreign_us.yml`, `nigeria_ngx.yml`, etc.) capturing lot sizes, trading hours, min tick sizes.
  - `src/local_adaptation/instrument_map.py` (~200 LOC) mapping foreign signals to NGX proxies with heuristics (equal weight, volatility scaling, cap per ticker).
  - `src/local_adaptation/nigeria_execution_sheet.py` (~200 LOC) generating phone-friendly CSV/Excel sheets with `Date | Ticker | Side | Qty | Entry | SL | TP | Notes`.
  - Unit tests verifying proxy basket limits, currency formatting, and CSV validity (UTF-8).

### 3.8 Automation, Monitoring & Usage Tracking
- **Purpose:** Reduce manual toil and ensure monetization features justify maintenance cost.
- **Items:**
  - Automated ETL/forecast jobs with retries + failover sources (Alpha Vantage, Finnhub, yfinance) per existing scripts; centralize scheduling notes in `Documentation/SYSTEM_STATUS_*.md`.
  - Alert pipeline automation that escalates errors and optionally triggers cTrader automation for demos (via cAlgo or REST once permitted).
  - `Documentation/MONETIZATION.md` summarizing each monetization feature with time-to-implement, complexity score, revenue hypothesis.
  - `src/monitoring/monetization_usage_tracker.py` (≤150 LOC) capturing alert/report usage counts in lightweight JSON/SQLite.
  - `python -m tools.monetization_reality_check` to print profitability metrics, counts of signals/reports, and keep/refactor/delete recommendations for each feature.

### 3.9 Monetization Strategy (Indirect ➜ Direct)
- **Phase 1 – Indirect:**
  - Trade signals personally on foreign brokers (demo/live), publish selective insights via blog/newsletter, and grow community credibility (Telegram, Substack, GitHub).
  - Use alert exports to seed a free channel that showcases track record without breaching guardrails.
- **Phase 2 – Direct:**
  - Launch subscription alert service once gate reports `READY_FOR_PUBLIC`.
  - Offer portfolio management or copy-trading on brokers that allow follower models (licensing requirements permitting).
  - Prepare white-label/B2B packages using the strategy packager outputs and Nigerian execution sheets.
  - Extend premium analytics (web dashboard or PDF packages) using existing optimization logic for added revenue streams.

---

## 4. Integrated To-Do Kanban (Reward-to-Effort Ordered)
| Priority | Track | Task & Deliverable | Success Criteria / Notes | Dependencies & Guardrails |
| --- | --- | --- | --- | --- |
| **Now** | Gatekeeping | Implement `monetization_gate.py` + CLI + tests | Returns status enum, enforces thresholds, fails fast in CI. | Requires clean performance DB; obey ≤250 LOC + gate thresholds. |
| **Now** | Backtesting Realism | Add transaction cost & slippage modeling to backtests/paper trading engine | Config-driven costs per market profile; updated reports proving difference vs. naive results. | Use existing instrumentation; prefer Backtrader or current engine extensions. |
| **Now** | Signals Core | Introduce `TradeSignal` dataclass + exporter + tests | All pipelines emit `TradeSignal`; alert/report modules rely solely on exporter formatting. | Maintain backward compatibility; minimal duplicated logic. |
| **Now** | Alerts | Build Telegram + email notifiers, WhatsApp stub, `alert_hub`, CLI entrypoint | `tools.push_daily_signals` sends/batches signals, blocks when gate says `BLOCKED`. | Respect line budget; config-driven credentials stored in `config/monetization.yml`. |
| **Next** | Reporting | Implement `strategy_packager` + tests | Generates JSON + Markdown reports, includes disclaimers and sample trades. | Needs reliable data retrieval; ensure no secrets leak. |
| **Next** | Overseas Execution | Create `ctrader_bridge` + `broker_interface` | Produces valid CSV/text order files; interface documented for future auto-clients. | Keep dependencies optional/free; align with demo usage first. |
| **Next** | Local Adaptation | Add market profiles, instrument proxy mapper, Nigeria execution sheet exporter | Proxy mapping adheres to risk caps; CSV importer confirmed with manual testing. | Data from `config/markets/*.yml`; tests cover mapping heuristics. |
| **Next** | Automation & Tracking | Add `MONETIZATION.md`, usage tracker, and reality-check CLI | Usage metrics stored, script prints 30-day stats + recommendations. | `Monetization gate` must be callable; instrumentation uses JSON/SQLite only. |
| **Later** | Monetization Scaling | Design indirect-to-direct funnel assets (blog templates, subscription infra, B2B deck) | Documented plan for promotional cadence, KPI targets, and evidence trails. | Requires `READY_FOR_PUBLIC` status + at least 90 days of audited performance. |
| **Later** | Advanced Enhancements | Integrate Backtrader/MLfinlab for optimizer loops, add partial automation to cTrader/copy trading | Documented ROI vs. custom code; prototypes passing guardrail review. | Only after profitability + line budget approvals; adhere to free-tier usage. |

> **Note:** Every to-do inherits the guardrails from `AGENT_INSTRUCTION.md` (line limits, free data, rollback plans, metrics logging). When delegating work to AI assistants, always inject this plan plus `AGENT_INSTRUCTION.md`, `AGENT_DEV_CHECKLIST.md`, and line budgets, and demand function signatures/tests/failure modes as per the Monetization brief.

---

## 5. Execution Guidance for Future Integrations
1. **Evidence First:** Before enabling monetization, capture 365-day rolling metrics via automated backtests/paper trading logs and store them where the gate can query without manual exports.
2. **Automation Hooks:** Re-use `scripts/run_auto_trader.py` and `config/signal_routing_config.yml` to bind new exporters/alerts rather than crafting new orchestrators.
3. **Foreign-to-Local Workflow:** Always test strategies on liquid US/EU/FX instruments first (higher data quality) and document tweaks made when mapping to NGX proxies.
4. **Alert Hygiene:** Format alerts with precise instructions (symbol, side, quantity heuristics, SL/TP). Provide plain-text fallback for manual brokers and ensure Telegram/email templates include disclaimers.
5. **Monetization Telemetry:** Instrument every feature (alerts sent, reports generated, execution sheets exported). Feed counts into the usage tracker + reality check CLI to decide whether to keep/refactor/delete.
6. **Documentation Discipline:** Update `Documentation/DOCUMENTATION_UPDATE_SUMMARY.md` and `Documentation/SYSTEM_STATUS_YYYY-MM-DD.md` whenever a monetization or automation component lands to keep future agents aligned.

---

## 6. References
- `Documentation/implementation_checkpoint.md` – current blockers and telemetry context.
- `Documentation/OPTIMIZATION_IMPLEMENTATION_PLAN.md` – dependency tree for optimization tasks.
- `Documentation/AGENT_INSTRUCTION.md` & `Documentation/AGENT_DEV_CHECKLIST.md` – enforce guardrails/phase gates.
- `Documentation/QUANT_TIME_SERIES_STACK.md` – approved stack + data source notes.
- Monetization brief excerpt (“Monetization & Local Adaptation – To-Do”) – authoritative instructions for modules listed above.

This integrated plan should be the go-to brief for any future automation or monetization sprint. Keep it updated as deliverables move through the to-do pipeline and as profitability evidence evolves.
