# Sentiment Signal Integration Plan (Profit-Gated)
**Updated:** 2025-11-30  
**Scope:** Future integration of sentiment-derived features/signals once profitability is **positive vs benchmark** and gating thresholds are met.

This playbook defers sentiment work until the strategy is already profitable, then adds sentiment as an incremental signal/filter to boost conviction and reduce false positives.

---

## 1) Gating & Readiness Criteria (must all be true)
- **Profitability:** Trailing 90d and 180d PnL both > 0 and above benchmark (e.g., local risk-free + ETF proxy).  
- **Risk/Quality:** Sharpe ≥ 1.1, Max Drawdown ≤ 0.22, win rate ≥ 52%, trade count ≥ minimum per strategy (avoid noise).  
- **Data Health:** No blocking issues in `QUANT_VALIDATION_MONITORING_POLICY`; forecast DB not degraded/corrupted.  
- **Ops Stability:** Brutal/TS/backtest pipelines green; latency budget headroom to add a new feature extractor.  
- **Approval:** Monetization/profit gate (`monetization_gate.py` when present) reports `READY_FOR_PUBLIC`.

If any gate fails, sentiment work remains dormant and only this plan is maintained.

---

## 2) Targets & Use-Cases
- **Overlay confidence:** Use sentiment to up/down-weight existing TS/ML signals (not to replace them).
- **Event filters:** Suppress trades near negative news bursts or regulator/FX shocks.
- **Position sizing nudges:** ±5–15% sizing adjustments when sentiment confirms/conflicts with model direction.
- **Reporting:** Add sentiment snapshots to reports/alerts for user context (optional after gating passes).

---

## 3) Data Sources & APIs (prefer free/low-friction)
- **News:** GDELT 2.0 (free, wide coverage), NewsAPI (free tier; daily caps), Google News RSS.  
- **Social/Community:** Reddit API (official), Stocktwits (public sentiment endpoint), optional X API (paid/limited; keep disabled by default).  
- **Finance-Specific Models:** HuggingFace Hub (no-cost inference for light usage), FinBERT (`ProsusAI/finbert`), `yiyanghkust/finbert-tone`, `cardiffnlp/twitter-roberta-base-sentiment`.  
- **Lexicons/Baselines:** VADER (NLTK), TextBlob/Pattern, spaCy pipelines for NER and source de-duplication.  
- **Storage:** SQLite/Parquet in `data/sentiment/` with minimal schemas to avoid DB bloat; reuse existing ETL patterns.

---

## 4) Architectural Sketch
1. **Ingest:** Pull headlines/posts via lightweight fetchers (`NewsFetcher`, `RedditFetcher`, `StocktwitsFetcher`) with strict rate limits and caching.  
2. **Normalize:** Deduplicate by URL/title/hash, tag tickers using existing symbol maps (`config/markets/*.yml`).  
3. **Score:** Run dual path – fast lexicon (VADER) + transformer (FinBERT/Roberta). Store `{doc_id, ticker, ts, score_lex, score_ml, source}`.  
4. **Aggregate:** Rolling features per ticker/timeframe: mean, EWMA, z-score, burst count, disagreement (|lex - ml|), source diversity.  
5. **Fuse with signals:**  
   - Confidence boost if sentiment aligns with directional signal and burst score < threshold.  
   - De-risk if negative burst conflicts with long signal or if disagreement high.  
6. **Safeguards:** Feature caps (clamp scores), missing-data fallback (no adjustment), and opt-out switch in config.

---

## 5) Phased Rollout (only after gates pass)
- **Phase 0 – Offline Benchmarks:** Backfill 6–12 months of news/social, compute IC vs future returns, measure hit-rate uplift, and record runtime/cost.  
- **Phase 1 – Shadow Mode:** Run sentiment extractors in CI/cron, log features side-by-side with trades; no trade impact.  
- **Phase 2 – Limited Impact:** Enable ±10% sizing nudge and conflict-based filter on low-liquidity names; monitor drawdown/Sharpe deltas weekly.  
- **Phase 3 – Full Optional:** Allow strategy-level opt-in with config flag; document effects in monetization docs (e.g., `REWARD_TO_EFFORT_INTEGRATION_PLAN.md`) once stable.

---

## 6) Implementation Stubs (when unblocked)
- `src/sentiment/fetchers/news_fetcher.py` – GDELT/NewsAPI/RSS ingest + caching; rate-limit guard.  
- `src/sentiment/fetchers/social_fetcher.py` – Reddit/Stocktwits; X disabled by default via config.  
- `src/sentiment/scorers/finbert.py` – HF transformers (batch, CPU-friendly), fallback to VADER for speed.  
- `src/sentiment/features.py` – Aggregations (EWMA, burst, disagreement); ticker-aware joins.  
- `src/sentiment/adjusters.py` – Combines base signals with sentiment features under clamps; returns adjusted confidence/size.  
- Config: `config/sentiment.yml` with toggles, rate limits, model names, and caps; disabled by default.  
- Tests: offline fixtures for headlines/social posts; regression tests for clamp logic and missing-data fallbacks.

---

## 7) Monitoring & Risk
- Track: extraction latency, API error rates, coverage per ticker, disagreement rate, IC vs returns, and PnL delta vs baseline.  
- Alert when: source coverage < threshold, sentiment burst conflicts with large position, or IC turns negative for 2+ weeks.  
- Privacy/Compliance: avoid PII storage, respect robots/noindex; keep tokens in env/config and never in logs.  
- Cost Control: prefer cached headlines and free tiers; batch HF inference; keep model sizes modest (distil/roberta base).

---

## 8) Success Criteria
- Neutral-to-positive PnL delta after rollout, **no drawdown degradation**, and Sharpe uplift ≥ 0.05 in A/B runs.  
- Runtime overhead < 15% on existing pipelines; no added flakiness in brutal/CI suites.  
- Config-driven enable/disable with safe defaults and clear documentation in alerts/reports.
