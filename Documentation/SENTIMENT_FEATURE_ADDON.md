# Sentiment Feature Add-on (Profit-Gated, Compute-Aware)

## Current State (As Of This Repo)

- Sentiment is **scaffolded but disabled** by default:
  - Config: `config/sentiment.yml` (`enabled: false`, `ops.dry_run: true`)
  - Test guard: `tests/sentiment/test_sentiment_config_scaffold.py`
- High-level plan exists and is intentionally conservative:
  - `Documentation/SENTIMENT_SIGNAL_INTEGRATION_PLAN.md`

This document turns that plan into a concrete **feature add-on spec** that can be plugged into the ETL pipeline once gates are met.

## What This Add-on Is Allowed To Do

Allowed (in order of promotion):

1. **Shadow mode (no trading impact)**:
   - fetch/score/aggregate sentiment
   - persist features and diagnostics
   - publish reports and OpenClaw notifications
2. **Signal overlay (bounded impact)**:
   - adjust confidence/position sizing within strict caps
   - block trades only when conflict criteria are met (and only after evidence)
3. **Exogenous model features (highest bar)**:
   - feed sentiment aggregates into SARIMAX-X / other models as exogenous variables

Not allowed (until separately justified):

- Replacing the primary signal generator (Time Series remains primary).
- Introducing heavy, high-latency network inference in live mode.

## Data Sources (Config-Driven)

The add-on should follow `config/sentiment.yml` and treat all sources as optional:

- News providers:
  - `gdelt` (preferred default for broad coverage)
  - `google_news_rss` (lightweight, low-friction)
  - `newsapi` (disabled by default; has caps and an API key)
- Social/community providers:
  - `reddit` (disabled by default; requires credentials)
  - `stocktwits` (disabled by default)
  - `x_api` (disabled by default)

Hard rule: in live mode, the pipeline should not block trading due to sentiment source outages.

## Document Normalization (Make Raw Text "Suitable")

Every fetched item becomes a normalized document record:

- `doc_id`: stable hash from canonical URL + title + published timestamp
- `source`: provider name
- `published_at`: timezone-aware timestamp (best effort; otherwise treat as "unknown")
- `title`: headline/title
- `body`: optional (do not require full text to start)
- `url`: optional
- `tickers`: list of tagged tickers (see below)

Normalization steps:

- Canonicalize URL (strip tracking query params where possible).
- Deduplicate by `doc_id` (idempotent ingests).
- Best-effort language filtering (optional): non-English can be dropped early to save compute.

## Ticker Tagging

Ticker tagging should be conservative to avoid false joins:

- Exact-match casings for known tickers.
- Optional alias maps (company names) should be opt-in and tested.
- If the add-on cannot tag tickers reliably, it should still store documents but only aggregate at a "global market sentiment" level.

## Scoring Strategy (Compute-Aware)

Two-tier scoring is recommended (matches `config/sentiment.yml`):

- Fast lexicon baseline: `vader`
- Transformer model: `ProsusAI/finbert` (fallback: `cardiffnlp/twitter-roberta-base-sentiment`)

Compute controls that must exist before enabling transformer scoring broadly:

- Batch inference with `models.max_batch_size` (default: 16).
- Device auto-selection (`models.device: auto`) with CPU fallback.
- Rate-limit all providers and cap docs per ticker/day (even in backfills).

Recommended optimization:

- Always compute VADER.
- Only compute transformer scores for:
  - the most relevant N documents per ticker per day, or
  - documents with large-magnitude VADER scores (high signal likelihood), or
  - a fixed sampling rate during shadow mode (for benchmarking).

Persist both scores when available:

- `score_lex`
- `score_ml`
- `score_ml_model` (model name/version)

## Aggregations (Turn Docs Into Features)

Aggregation must be time-indexed and leakage-safe.

For each `(ticker, bar_timestamp)` produce a compact feature block:

- Level:
  - `sent_mean_1d` (mean sentiment over last 1 day)
  - `sent_ewma_3d`, `sent_ewma_7d`, `sent_ewma_14d` (windows from `config/sentiment.yml`)
- Strength and reliability:
  - `sent_doc_count_1d`
  - `sent_burst_z` (z-score of doc counts vs trailing baseline)
  - `sent_source_diversity_1d` (unique sources)
- Disagreement and noise:
  - `sent_disagreement` (abs(score_lex - score_ml) aggregated)
- Guardrails:
  - clamp scores to `features.clamp_range` (default: [-3, 3])

Missing-data behavior must match config:

- `missing_data_behavior: passthrough` means "no adjustment when missing", not "fill with zeros that bias the model".

## Alignment To Price Bars (Leakage Prevention)

Core alignment rule:

- If you cannot prove the document timestamp is <= the bar timestamp, do not let it affect that bar.

Practical defaults:

- Daily bars:
  - map docs published during the day to that day's bar only if published before a conservative cutoff.
  - otherwise shift to next bar.
- Intraday bars:
  - assign to the next bar boundary after publication time.
  - if publication time is unknown, shift to the next day (conservative).

Always record the alignment rule used in metadata for auditability.

## How It Influences Trading (Only After Promotion)

When global and per-add-on gates pass, sentiment can influence trading only through bounded adjustments:

- Max position nudge: `adjustments.max_position_nudge_pct` (default cap: 0.15).
- Alignment boost: `adjustments.alignment_boost_pct` (default: 0.05).
- Conflict filter: `adjustments.conflict_filter_enabled` (default: true).

Interpretation examples (policy, not code):

- If base signal is BUY and `sent_ewma_7d` is strongly positive, increase confidence/size slightly (within cap).
- If base signal is BUY but sentiment shows a negative burst, either:
  - reduce size, or
  - demote BUY to HOLD if conflict severity exceeds threshold.

Sentiment should not create trades on its own.

## Gating (When Sentiment Is Allowed To Affect Anything)

Gating is already encoded in `config/sentiment.yml` and enforced by test:

- enabled must remain false by default
- risk and profit thresholds must remain strict

Additional promotion criteria recommended before enabling any live impact:

- Offline A/B evidence across multiple regimes:
  - Sharpe uplift >= 0.05 without drawdown degradation
  - no consistent negative IC (information coefficient) vs future returns
- Compute overhead:
  - < 15% walltime overhead vs baseline pipeline run, or a documented exception with measured lift.
- Stability:
  - does not increase pipeline flakiness (network failures do not fail trading).

## OpenClaw AI Defaults (Promotion Is A Human Decision)

When sentiment is evaluated (shadow or A/B), the pipeline should emit a short decision summary to OpenClaw AI by default (if configured):

- See `Documentation/OPENCLAW_INTEGRATION.md`
- Manual helper: `python scripts/openclaw_notify.py`

Minimum message contents:

- add-on name (`sentiment`)
- status (`shadow`, `rejected`, `promoted`)
- delta metrics (Sharpe, drawdown, runtime overhead)
- artifact pointer (report path under `reports/`)
