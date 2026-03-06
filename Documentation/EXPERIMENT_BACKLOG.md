# Experiment Backlog — Portfolio Maximizer v45

**Governing protocol**: [RESEARCH_EXPERIMENT_PROTOCOL.md](RESEARCH_EXPERIMENT_PROTOCOL.md)
**System baseline**: profit_factor ≈ 0.80, win_rate ≈ 40%, n_trades ≈ 40
**Last updated**: 2026-03-06
**Phase gate**: Phase 5 experiments are blocked until Phases 1–4 pass.

---

## Pre-Experiment Prerequisite Checklist

All five conditions must be confirmed before EXP-001 may begin:

- [ ] `execution_log_freshness <= 24h` (Phase 1)
- [ ] `outcome_matched >= 10 AND matched/eligible >= 0.80` (Phase 2)
- [ ] `evaluation_metrics_coverage >= 80%` (Phase 3)
- [ ] Attribution diagnostics complete — exit_reason breakdown, tail-loss list, PnL heatmap (Phase 4)
- [ ] `high_integrity_violation_count = 0` (ongoing)

---

## Current System Diagnosis (Basis for Backlog)

| Loss Driver | Evidence | Priority |
|-------------|----------|----------|
| Extreme tail losses | Stop-loss exits dominate PnL (-$233 in 5 stops vs +$22 in 8 time-exits, sprint data) | 1 — HIGHEST |
| Winners cut too early | `max_holding=8` time-exit clips profits before full trend | 2 |
| Unprofitable tickers | WEAK tickers in high-vol regimes generate negative expected return | 3 |
| Low-conviction entries | Signals near confidence floor trade like coin-flips | 4 |
| Oversized losing positions | LAB_ONLY tickers with thin evidence receive full sizing | 5 |

---

## EXP-001 — Tighter ATR Stop to Reduce Tail Losses

```
Experiment ID:       EXP-001
Status:              PENDING (blocked on Phase 1-4 gates)

Hypothesis:
  Reducing the ATR stop multiplier from 1.5x to 1.2x reduces the magnitude of
  extreme tail losses without materially reducing win rate, yielding a net
  improvement in profit factor.

Intervention:
  paper_trading_engine.py: atr_stop_multiplier 1.5 -> 1.2
  (config/signal_routing_config.yml: atr_multiplier: 1.5 -> 1.2)

Control condition:
  ATR multiplier 1.5x (current production baseline)

Experiment phase:    Phase 5a — extreme tail-loss reduction

Sample window:       >= 30 closed trades AND >= 21 calendar days after launch

Metrics evaluated:
  - profit_factor         (primary)
  - tail_loss_p95         (average of bottom 5% loss trades)
  - win_rate              (must not drop > 3pp)
  - average_drawdown
  - expected_return

Statistical method:  bootstrap mean difference (1000 resamples, seed=42, 95% CI)

Confidence threshold: 95%

Stopping rule:       >= 30 closed trades AND >= 21 days

Rollback condition:
  profit_factor < 0.80 (worse than baseline)
  OR win_rate drops > 3pp vs baseline
  OR high_integrity_violation_count > 0

Expected effect:     PF 0.80 -> ~0.92; tail_loss_p95 reduced ~20%

Rationale:
  Sprint data shows 5 stop-loss exits totalling -$233 vs 8 time-exits at +$22 combined.
  The 1.5x ATR multiplier on NVDA (57% annualized vol) places the stop ~8% below entry,
  allowing outsized losses to accumulate before triggering. Tightening to 1.2x cuts this
  to ~6.5%, reducing maximum loss per trade without moving win rate substantially since
  the stop is still wide relative to intraday noise.
```

---

## EXP-002 — Extend Max Holding to Reduce Premature Time-Exits

```
Experiment ID:       EXP-002
Status:              PENDING (requires EXP-001 result first)
Prerequisite:        EXP-001 ADOPT or REJECT decision documented

Hypothesis:
  Extending max_holding from 8 to 12 daily bars allows profitable trends to develop
  fully, increasing average win size and profit factor without degrading win rate.

Intervention:
  scripts/run_auto_trader.py proof-mode default_horizon 8 -> 12
  (or equivalent config parameter)

Control condition:
  max_holding = 8 (current, post-Phase-7.19 setting)

Experiment phase:    Phase 5b — correct-direction-but-loss repair

Sample window:       >= 30 closed trades AND >= 21 days

Metrics evaluated:
  - profit_factor         (primary)
  - avg_win_size          (average PnL of winning trades)
  - win_rate
  - time_exit_pct         (fraction of exits via TIME_EXIT)
  - holding_period_days

Statistical method:  bootstrap mean difference (1000 resamples, seed=42, 95% CI)

Confidence threshold: 95%

Stopping rule:       >= 30 closed trades AND >= 21 days

Rollback condition:
  profit_factor < 0.80
  OR drawdown_p95 increases > 20% vs baseline
  OR win_rate drops > 3pp

Expected effect:     avg_win_size +$15-25; PF 0.92 -> ~1.05

Rationale:
  Sprint audit: TIME_EXIT winners averaged only $2.71. Winners are being cut at 8 bars
  when the underlying trend continues. Extending the window gives MSSA-RL and SAMOSSA
  forecasts space to capture the full predicted move. The risk is that losers also run
  longer — mitigated by EXP-001's tighter stop already in place.
```

---

## EXP-003 — Regime Gate: Block Entries in HIGH_VOL for WEAK Tickers

```
Experiment ID:       EXP-003
Status:              PENDING (requires EXP-001 or EXP-002 decision first)

Hypothesis:
  Blocking new trade entries for tickers classified WEAK by
  compute_ticker_eligibility.py when the detected regime is HIGH_VOL or CRISIS
  reduces win-rate-draining trades without reducing total trade count significantly.

Intervention:
  models/signal_router.py: add regime_filter check — if ticker_status == WEAK
  AND regime in (HIGH_VOL, CRISIS): block signal (return None)

Control condition:
  No regime-based ticker filtering (current behavior)

Experiment phase:    Phase 5c — regime-based loss reduction

Sample window:       >= 30 closed trades AND >= 21 days

Metrics evaluated:
  - profit_factor         (primary)
  - win_rate
  - trades_blocked_pct    (% signals suppressed by new filter)
  - expected_return
  - per_ticker_pnl

Statistical method:  bootstrap mean difference (1000 resamples, seed=42, 95% CI)

Confidence threshold: 95%

Stopping rule:       >= 30 closed trades AND >= 21 days

Rollback condition:
  trades_blocked_pct > 40% (filter too aggressive — starves the system)
  OR profit_factor < 0.80

Expected effect:     Remove 20-30% of loss-generating trades; PF -> ~1.10-1.15

Rationale:
  Context quality attribution (from Phase 4) identifies which regimes generate negative
  expected return for each ticker. WEAK tickers in HIGH_VOL/CRISIS regimes have the lowest
  signal-to-noise ratio and are most likely to generate stop-loss exits. This filter is
  conservative (it only blocks entries, not force-closes existing positions) and targets
  the highest-loss subset first.
```

---

## EXP-004 — Confidence Gate: Raise Minimum Entry Confidence

```
Experiment ID:       EXP-004
Status:              PENDING

Hypothesis:
  Raising minimum entry confidence from 0.55 to 0.62 eliminates near-threshold
  signals that behave like coin-flips, improving win rate by 3-5pp and profit
  factor without unacceptable reduction in trade frequency.

Intervention:
  config/signal_routing_config.yml: confidence_threshold 0.55 -> 0.62
  (applies to all tickers without an explicit per-ticker override)

Control condition:
  confidence_threshold = 0.55 (Phase 7.14-A setting)

Experiment phase:    Phase 5d — low-conviction entry filtering

Sample window:       >= 30 closed trades AND >= 21 days

Metrics evaluated:
  - profit_factor         (primary)
  - win_rate
  - n_trades_per_day      (must not drop below 0.5/day)
  - expected_return
  - pnl_by_confidence_bucket (from compute_context_quality)

Statistical method:  bootstrap mean difference (1000 resamples, seed=42, 95% CI)

Confidence threshold: 95%

Stopping rule:       >= 30 closed trades AND >= 21 days

Rollback condition:
  n_trades_per_day < 0.3 (system becomes too inactive for evidence accumulation)
  OR profit_factor < 0.80

Expected effect:     win_rate 40% -> ~44%; PF -> ~1.15

Rationale:
  PnL by confidence bucket (Phase 4 attribution) will show whether the 0.55-0.62 bin
  is generating negative or near-zero expected return. If it is, raising the floor
  eliminates these coin-flip trades. Risk: fewer total trades slows evidence accumulation.
  Only run this if Phase 4 confirms the 0.55-0.62 bucket is loss-generating.
```

---

## EXP-005 — Position Sizing: Reduce Size for LAB_ONLY Tickers

```
Experiment ID:       EXP-005
Status:              PENDING

Hypothesis:
  Halving position size for tickers classified LAB_ONLY by
  compute_ticker_eligibility.py reduces the PnL impact of loss-generating
  experimental tickers without removing them from the system.

Intervention:
  execution/paper_trading_engine.py: when ticker_status == LAB_ONLY,
  multiply computed_position_size by 0.5 before order submission.

Control condition:
  Full position sizing regardless of ticker eligibility tier (current behavior)

Experiment phase:    Phase 5e — position-size calibration

Sample window:       >= 30 closed trades AND >= 21 days

Metrics evaluated:
  - profit_factor         (primary)
  - max_drawdown
  - pnl_by_ticker
  - expected_return
  - position_size_avg

Statistical method:  bootstrap mean difference (1000 resamples, seed=42, 95% CI)

Confidence threshold: 95%

Stopping rule:       >= 30 closed trades AND >= 21 days

Rollback condition:
  profit_factor < 0.80
  OR LAB_ONLY trades disappear entirely (size too small to execute)

Expected effect:     Reduced volatility of PnL; max_drawdown -15-20%; PF -> ~1.20+

Rationale:
  LAB_ONLY tickers have insufficient production evidence for full conviction sizing.
  A 0.5x multiplier preserves their signal-learning contribution (Platt pairs, forecast
  audits) while halving the capital at risk from their underdeveloped edge.
```

---

## Backlog Summary Table

| ID | Priority | Hypothesis | Prerequisite | Status |
|----|----------|------------|--------------|--------|
| EXP-001 | 1 | Tighter ATR stop reduces tail losses | Phase 1-4 gates pass | PENDING |
| EXP-002 | 2 | Extend max_holding allows winners to run | EXP-001 decision | PENDING |
| EXP-003 | 3 | Regime gate blocks WEAK-ticker HIGH_VOL entries | EXP-001 or EXP-002 decision | PENDING |
| EXP-004 | 4 | Raise confidence floor eliminates coin-flip trades | Phase 4 attribution confirms 0.55-0.62 bucket negative | PENDING |
| EXP-005 | 5 | Half-size LAB_ONLY reduces drawdown from experimental tickers | Any prior experiment decision | PENDING |

---

## Expected Path to profit_factor >= 1.30

| After | Expected PF | Key driver |
|-------|------------|------------|
| Baseline | 0.80 | Starting point |
| EXP-001 | ~0.92 | Tail-loss magnitude reduced ~20% |
| EXP-002 | ~1.05 | Average win size increases |
| EXP-003 | ~1.15 | WEAK-ticker HIGH_VOL losses eliminated |
| EXP-004 | ~1.22 | Coin-flip trades filtered |
| EXP-005 | ~1.30+ | LAB_ONLY drawdown reduced |

Each estimate assumes sequential adoption and no compounding regressions. Bootstrap CI
must confirm each step crosses zero before proceeding to the next.

---

## Experiment Log (Completed)

_None yet. First entry will appear after EXP-001 runs >= 30 trades._

| ID | Decision | PF before | PF after | CI | Date |
|----|----------|-----------|----------|----|------|
| — | — | — | — | — | — |
