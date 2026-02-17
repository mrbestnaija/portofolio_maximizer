# Adversarial Audit -- 2026-02-16

**Date:** 2026-02-16
**Status:** CRITICAL -- Multiple structural weaknesses identified
**Phase:** 7.9 (Post PnL integrity enforcement)
**Database:** 83 rows, 39 round-trips, $670.94 total PnL, 41.0% WR, 1.84 PF

---

## Executive Summary

Adversarial stress-testing of the production pipeline reveals systemic issues
across forecasting, confidence calibration, execution, and signal linkage.
94.2% of quant validation builds FAIL, with only 0.8 percentage-point headroom
to the RED gate. The ensemble is worse than the best single model 92% of the
time. The system survives on magnitude asymmetry (avg win $91.59 vs avg loss
$34.54 = 2.65x ratio), not directional accuracy (41% win rate, below coin flip).

---

## Finding 1: Quant Validation Near-Gate-Breach

| Sprint Gate | Entries | PASS | FAIL | FAIL % | Ceiling | Headroom |
|-------------|---------|------|------|--------|---------|----------|
| Gate 1      | 80      | 5    | 75   | 93.8%  | 95%     | 1.2%     |
| Gate 2      | 100     | 6    | 94   | 94.0%  | 95%     | 1.0%     |
| Gate 3      | 120     | 7    | 113  | 94.2%  | 95%     | 0.8%     |

FAIL fraction is rising with each check. At this trajectory the next sprint
will breach 95% and RED-gate the system.

Per-ticker breakdown (120 JSONL entries):
- 8 of 10 tickers have **100% FAIL rate** (MSFT, NVDA, GOOG, AMZN, META, TSLA, JPM, V)
- Only AAPL passes at 50% (6/12) and GS at 8% (1/12)
- Top fail criteria: `expected_profit` (67), `sharpe_ratio` (54), `sortino_ratio` (54)

**Severity:** P0 -- system is 0.8% from automated shutdown

---

## Finding 2: Ensemble Systematically Worse Than Best Single Model

| Metric | Value |
|--------|-------|
| Audits where ensemble beats best single | 10 / 127 (7.9%) |
| Audits where ensemble is worse | 117 / 127 (92.1%) |
| Formal violations (ratio > 1.1) | 65 / 127 (51.2%) |
| Mean RMSE ratio (ens/best) | 1.160 |
| Worst ratio | 3.244 |
| Gate decisions: DISABLE or RESEARCH_ONLY | 87 / 127 (68.5%) |

The weighted-average blending dilutes the best model's forecast 92% of the
time. The preselection gate (`strict_preselection_max_rmse_ratio=1.0`) already
blocks ensemble at forecast time, so actual forecasts use best-single. But
audit metrics still evaluate ensemble, inflating violation counts.

**Severity:** P0 -- ensemble architecture is counter-productive

---

## Finding 3: MSSA-RL Under-Weighted

| Model | Avg Ensemble Weight | Best Single in Violations |
|-------|-------------------|--------------------------|
| SAMoSSA | 57.1% | 1.5% |
| GARCH | 37.9% | 38.5% |
| MSSA-RL | 8.7% | 60.0% |

MSSA-RL is the best single model in 60% of violation cases but receives only
8.7% ensemble weight. The quantile calibration is systematically
under-rewarding MSSA-RL.

**Severity:** P1 -- primary source of ensemble degradation

---

## Finding 4: Directional Accuracy Below Coin-Flip

| Model | Avg Dir. Accuracy | Below 50% Count |
|-------|-------------------|------------------|
| SAMoSSA | 46.4% | 79/127 |
| Ensemble | 45.1% | 90/127 |
| GARCH | 45.0% | 88/123 |
| MSSA-RL | 44.1% | 70/127 |

All production models predict direction worse than a coin flip. Profits come
purely from magnitude asymmetry, not accuracy.

**Severity:** P2 -- fundamental model limitation

---

## Finding 5: Confidence Calibration Broken

Quant validation entries:
- FAIL entries: confidence avg = 0.417, range 0.247-0.950
- PASS entries: confidence avg = 0.993, range 0.950-1.000

Trade executions:
- ALL 39 trades have confidence >= 0.8 (range: 0.900-1.000)
- Win rate at these "very high" confidence levels: 41%

A 0.9+ confidence should imply ~90% expected accuracy, not 41%.
Confidence provides zero filtering value.

**Severity:** P0 -- renders position sizing meaningless

---

## Finding 6: Ticker-Level Drag

| Ticker | Trades | Win Rate | Total PnL | Verdict |
|--------|--------|----------|-----------|---------|
| NVDA | 8 | 62.5% | +$706.97 | Carries portfolio |
| MSFT | 6 | 66.7% | +$188.21 | Solid |
| GOOG | 10 | 50.0% | +$180.29 | Break-even edge |
| GS | 5 | 0.0% | -$91.26 | Blacklist candidate |
| AAPL | 7 | 14.3% | -$325.08 | Blacklist candidate |

AAPL and GS together destroyed $416.34 (62% of gross profits). AAPL had 3
stop-losses on the same day (Feb 10) totaling -$318.27 with no intra-day
adaptive learning.

Single-trade dependency: Remove one NVDA trade ($497.83) and PnL drops 74%.

**Severity:** P1 -- concentration risk + persistent losers

---

## Finding 7: Multi-Day Trades Net Negative

| Bucket | Trades | Total PnL |
|--------|--------|-----------|
| Intraday | 27 | +$900.43 |
| 1-3 days | 10 | -$174.90 |
| 4-7 days | 2 | -$54.59 |

Overnight/multi-day holds collectively lost $229.49. Eliminating them would
raise total PnL to $900.43 (+34% improvement). The system has no overnight
edge.

**Severity:** P2 -- restrict to intraday or require higher confidence for holds

---

## Finding 8: GARCH Unit-Root Problem

| Metric | Value |
|--------|-------|
| Near unit root (alpha+beta >= 0.99) | 54 / 192 (28.1%) |
| Exact unit root (alpha=0, beta=1) | Multiple cases |

28% of GARCH fits are degenerate (IGARCH). These produce flat volatility
forecasts. Since GARCH gets 37.9% average ensemble weight, degenerate fits
pollute ~11% of all ensemble forecasts.

**Severity:** P2 -- add alpha+beta < 0.98 guard

---

## Finding 9: Signal-to-Trade Linkage Gap

- `signal_id` is NULL for all 39 trades
- `trading_signals` table is empty (0 rows)
- `model_type` / `barbell_bucket` are NULL across all executions
- 4 unlinked closes (no `entry_trade_id`)
- 9 orphan opens without matching closes

Without signal-to-trade linkage, it is impossible to attribute which model
generated profitable vs unprofitable signals.

**Severity:** P1 -- blocks model attribution and feedback loops

---

## Finding 10: Regime Detector Lacks Discrimination

| Regime | % of Detections | Avg Confidence |
|--------|----------------|----------------|
| MODERATE_TRENDING | 51.6% | 50.5% |
| MODERATE_MIXED | 38.7% | 35.8% |
| All others combined | 9.7% | varies |

90.3% of observations land in two low-confidence regimes. HIGH_VOL_TRENDING
(1.0%) and CRISIS (0.4%) are almost never triggered. The regime detector is
not providing meaningful model-selection signal.

**Severity:** P3 -- low impact given ensemble is already gated

---

## Exit Reason Analysis

| Exit Reason | Trades | Total PnL | Avg PnL |
|-------------|--------|-----------|---------|
| TAKE_PROFIT | 4 | +$1,012.10 | +$253.03 |
| TIME_EXIT | 24 | +$340.87 | +$14.20 |
| STOP_LOSS | 10 | -$681.70 | -$68.17 |

Only 4 trades (10.3%) hit take-profit, but they generated more PnL than the
total portfolio profit. The system's edge is entirely in rare large winners.

---

## Priority Recommendations

### P0 (Immediate -- blocks credibility)
1. **Disable ensemble as default**: Use preselection-gated best-single model
   for audit metrics (ensemble is already blocked at forecast time)
2. **Fix confidence calibration**: Add Platt scaling or isotonic regression on
   validation set so emitted confidence tracks realized win rate
3. **Separate proof-mode from production validation**: Use `execution_mode`
   tag to filter proof-mode entries from quant health checks

### P1 (Next sprint)
4. **Re-weight ensemble**: Raise MSSA-RL floor weight to 25-30% or recalibrate
   quantile scoring to match violation-window dominance
5. **Populate signal_id in trade_executions**: Enable model attribution and
   feedback loops
6. **Add per-ticker kill switch**: When consecutive losses exceed threshold
   (e.g., 3 stops), disable ticker for remainder of session

### P2 (Near-term)
7. **Restrict to intraday-only** or require higher confidence for overnight
   holds (+34% PnL improvement estimated)
8. **Add GARCH unit-root guard**: Skip or warn when alpha+beta >= 0.98, fall
   back to EWMA for degenerate fits
9. **Widen proof-mode max_holding** from 5 to 8-10 daily bars to allow
   winners to develop before TIME_EXIT

### P3 (Backlog)
10. **Retune regime detector**: Add momentum/correlation features or lower
    vol/trend thresholds to improve bucket discrimination
11. **Skip SARIMAX fit in audits**: Currently fitted in 72% of audits (~5s
    each) but never included in ensemble weights -- pure compute waste
12. **Investigate directional accuracy**: Current features may not carry
    directional signal; consider feature engineering review

---

## Data Sources

- Quant validation: `logs/signals/quant_validation.jsonl` (120 entries)
- Forecast audits: `logs/forecast_audits/` (531 files, 127 with RMSE)
- Trade database: `data/portfolio_maximizer.db` (83 rows, 39 round-trips)
- Sprint logs: `logs/audit_sprint/20260214_232321/` (latest successful sprint)
- Config: `config/forecaster_monitoring.yml`, `config/quant_success_config.yml`
