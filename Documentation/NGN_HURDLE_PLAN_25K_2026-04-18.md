# NGN Hurdle Plan — $25k Capital Base (Rebased 2026-04-18)

Capital: $25,000 (confirmed `portfolio_cash_state.initial_capital`)
Baseline: `visualizations/performance/metrics_summary.json` (generated 2026-03-29)
Current state: 40 trades, 40% WR, PF 1.73, $620.01 closed PnL, 84 days

---

## Corrections to Prior Analyses

### C1 — Snapshot was stale (HIGH)

Prior analysis used 42 trades / 38% WR / $567.30. The live metrics file shows
40 trades / 40% WR / $620.01. All numbers below are rebased to the live file.

### C2 — Wrong ROI denominator (HIGH)

Prior calculations divided PnL by cumulative sum of notionals ($65k), not capital base ($25k).

| Metric | Prior (wrong denom) | Rebased (correct) |
|--------|--------------------|--------------------|
| Cumulative ROI | 0.87% | **2.48%** |
| Annualised ROI | 3.8% | **10.8%** |
| Gap multiplier | 7.4× | **2.60×** |
| Dollar gap | $3,603 | **$991** |

### C3 — Kelly proxy overstates deployment by 3.4× (MEDIUM)

"Avg notional / capital" treats each trade as if capital is tied up for the full
84-day window. The correct measure for a sequential trading system is time-weighted
capital: sum(notional × hold_days) / total_days.

| Proxy | Value | % of capital |
|-------|-------|-------------|
| Avg notional per trade | $1,558 | 6.2% |
| Time-weighted avg daily deployment | **$457** | **1.8%** |

The system has **98.2% idle cash** on average. Avg notional overstates daily deployment
by 3.4×. "42% of Kelly" (from prior analysis) was computed against the wrong basis;
the correct deployment-based Kelly fraction requires a risk-at-stop analysis, not just
notional / capital.

Kelly fraction on rebased edge (WR=40%, payoff=2.65×): **17.4% = $4,343/trade**
Avg notional as fraction of true Kelly: $1,558 / $4,343 = **35.9%**

Capital at risk per trade (actual) = avg stop-triggered loss = $34.54 (avg_loss from stops)
True Kelly ≈ $34.54 / $4,343 = 0.8% of capital at risk vs Kelly budget of $4,343 per trade.

### C4 — `rmsevalues` dismissal was too strong (MEDIUM)

Prior text said "not present in current code." Correct language: `_rmse_values` is
properly scoped in `ensemble.py:488` and no NameError is reproducible from the current
codebase. However, a deployment or artifact-version mismatch could produce this at runtime
without appearing in static analysis. **Required: verify against a verified failing run log
before treating as resolved or acting on it.**

### C5 — SNR adjustment is a hypothesis, not a settled fix (MEDIUM)

The `adj_snr_gate = snr_gate / ci_inflation_factor` formula assumes CI inflation and
SNR gate requirements scale linearly. This is an approximation that needs empirical
validation: run cycles with logging of `garch_convergence_ok`, `ci_inflation_factor`,
`raw_snr`, and `adj_snr_gate` before deploying. Do not hard-wire the adjustment without
at least 10 cycles of before/after comparison.

### C6 — Phase estimates were point estimates, not scenario ranges (MEDIUM)

The "+$717 and +$467 are not safely additive" critique is valid. Both levers interact:
exit geometry changes the WR/payoff distribution which changes Kelly, which changes
optimal sizing; AAPL/GS cull frees capital which affects what can be redeployed.
The correct representation is scenario ranges with an interaction discount.

---

## Rebased Gap Analysis

| Metric | Value |
|--------|-------|
| Capital | $25,000 |
| Closed PnL | $620.01 (40 trades, 84 days) |
| Ann ROI | 10.8% |
| NGN hurdle | 28% ann |
| Target PnL (same 84-day window) | $1,611 |
| **Dollar gap** | **$991** |
| Time-weighted daily deployment | $457 = 1.8% of capital |
| Kelly fraction | 17.4% ($4,343/trade) |
| Avg notional as % Kelly | 35.9% |

---

## Three Levers and Their Scenario Ranges

### Lever A — Exit geometry (Phase 1, code change)

Mechanism: Enforce R:R ≥ 2:1 at quant gate + trailing stop (ratchet to break-even
at +1× ATR) + extend max_hold to 15 bars for SNR ≥ 2.0. Converts TIME_EXIT events
into TAKE_PROFIT events. Delta per conversion = $253.03 − $14.20 = **$238.83**.

| Scenario | Conversions | PnL gain | Interaction discount | Net gain |
|----------|-------------|----------|----------------------|----------|
| Low | 1 TP | +$239 | none | **+$239** |
| Base | 3 TP | +$717 | 10% (Kelly fraction shifts) | **+$645** |
| High | 5 TP | +$1,195 | 15% | **+$1,016** |

Range: **+$239 to +$1,016**. Base case adds +$645 on a $991 gap.

Not safely additive with Lever B: exit geometry improvement increases WR and payoff,
which raises Kelly fraction, which ideally raises sizing on future trades — creating
a compounding effect that makes the point estimates conservative over time, not overstated.

### Lever B — AAPL/GS cull (Phase 2, governance)

Mechanism: Demote AAPL (8 trips, 1W/7L, -$376) and GS (5 trips, 0W/5L, -$91) to
LAB_ONLY. Prevent forward recurrence of historical drag pattern.

| Scenario | Assumption | Gain |
|----------|-----------|------|
| Low | 50% of pattern prevented (other losers fill some NAV) | +$234 |
| Base | Full drag prevented, freed capital idle | +$467 |
| High | Full drag prevented + freed capital redeployed at NVDA/GOOG/MSFT edge rate | +$611 |

Range: **+$234 to +$611**. The high case requires entry pipeline to work (Lever C).

Interaction with Lever A: if exits are fixed, the surviving losers cost less per stop.
AAPL loses less per trade when R:R is enforced (stops are the same; the bad entries
that would never have cleared R:R ≥ 2:1 never enter). So Levers A and B interact
positively — fixing exits *first* reduces the cost of continuing to trade AAPL during
the 2-cycle demotion window.

### Lever C — Capital utilization via entry pipeline (Phase 0, code/config)

**This is the lever both prior analyses missed.** Time-weighted capital deployment is
1.8% per day. All three main levers (A, B, and any sizing adjustment) are multiplied by
trade frequency. The system currently executes 0.48 trades/day. Entry rate is entirely
controlled by how many signals clear the SNR/confidence gate.

| Scenario | Trades/day | Proj PnL (40-trade edge) | Ann ROI |
|----------|------------|--------------------------|---------|
| Current (blocked) | 0.48 | $620 | 10.8% |
| Partial unblock | 0.95 | $1,237 | 21.5% |
| **Target** | **1.40** | **$1,823** | **31.7%** |

**Lever C alone, with no other change, clears the NGN hurdle at 1.4 trades/day.**

This requires unblocking 3 qualified entries per 2-day window. The blocker is the
GARCH EWMA CI inflation (1.5× CI → SNR halved at 1.5 gate). Mechanism to address:

1. **Verify hypothesis (10 cycles):** log `garch_convergence_ok`, `ci_inflation_factor`,
   `raw_snr` per signal. Confirm EWMA is the primary cause of gate failures before changing gates.
2. **Regime-aware SNR adjustment (hypothesis):** when `garch_fallback=EWMA`,
   apply `effective_snr_gate = snr_gate * inflation_factor` so the original SNR 1.5
   threshold remains meaningful in the inflated-CI regime. This is NOT lowering the gate —
   it is compensating for known input distortion.
3. **Empirical validation required:** run 10+ cycles before/after and measure new entry
   rate, WR, and PF. Only keep adjustment if WR does not degrade below 38%.

---

## Combined Scenario Matrix

Levers are partially independent (A and C multiply; A and B partially overlap;
B and C are independent). Combined estimates apply a 70% additivity factor to A+B
(conservative overlap discount), then multiply by the C scale factor.

| Scenario | A (exit) | B (cull) | A+B (70%) | C (utilization) | Combined PnL | Ann ROI |
|----------|----------|----------|-----------|-----------------|--------------|---------|
| Do nothing | — | — | $620 | ×1.0 | $620 | **10.8%** |
| A only (base) | +$645 | — | $1,265 | ×1.0 | $1,265 | **22.0%** |
| B only (base) | — | +$467 | $1,087 | ×1.0 | $1,087 | **18.9%** |
| C only (target) | — | — | $620 | ×2.92 | $1,810 | **31.5%** |
| A+B (base) | +$645 | +$327 | $1,592 | ×1.0 | $1,592 | **27.7%** |
| **A+B+C (base×target)** | +$645 | +$327 | $1,592 | ×2.92 | **$4,649** | **>100%** |
| A+B (low) | +$239 | +$234 | $1,093 | ×1.0 | $1,093 | **19.0%** |
| C only (partial) | — | — | $620 | ×1.98 | $1,228 | **21.4%** |

**Key reading:** A+B at base case just falls short of 28% (27.7%). Adding even partial
capital utilization improvement (1.5× scale, partial entry unblock) crosses the hurdle.
The levers are not binary — partial progress on each compounds.

---

## Phased Concrete Plan

### Phase 0 — Diagnose the entry pipeline (Days 1–7)

**Hard wall**: new BUYs fully blocked. Without entries, every other lever is academic.

| Action | File | What to do |
|--------|------|-----------|
| 0a | `scripts/run_auto_trader.py` | Run one cycle with LOG_LEVEL=DEBUG; capture `garch_convergence_ok`, `ci_inflation_factor`, raw SNR, adjusted SNR, routing reason per signal |
| 0b | `forcester_ts/garch.py` | Confirm EWMA fallback rate: log what fraction of tickers hit EWMA in this market regime |
| 0c | ensemble.py + logs | If `_rmse_values` NameError appears in logs, file as confirmed bug with stack trace; do not act on unverified claim |

**Evidence gate**: before any gate changes, have ≥ 10 cycles logged with SNR breakdown.
Only proceed to Phase 0 regime-aware adjustment if EWMA CI inflation explains ≥ 70% of blocked signals.

| Sub-action | Specifics |
|-----------|-----------|
| 0d — Regime-aware SNR (HYPOTHESIS ONLY until validated) | `models/time_series_signal_generator.py`: when `garch_fallback_type = "EWMA"`, compute `effective_snr_gate = min_snr * ci_inflation_factor`; log both; do not deploy until 10+ cycle comparison confirms WR does not drop |

**Promotion gate to Phase 1**: ≥ 2 qualified entries in next 10 cycles.

---

### Phase 1 — Fix exit geometry (Days 5–21, P0 code change)

The single highest-leverage code change. Converts TIME_EXIT to TAKE_PROFIT.
Addresses 78% of gross profit concentration in only 10% of trades.

| Action | File | Specifics |
|--------|------|-----------|
| 1a — R:R gate | `models/time_series_signal_generator.py:_calculate_targets()` | Block signal if `target_pct / stop_pct < 2.0`; emit `INSUFFICIENT_RR` reason; log R:R on every signal |
| 1b — Trailing stop | `execution/paper_trading_engine.py` | After position unrealised PnL ≥ +1× ATR: ratchet stop to break-even. After ≥ +2× ATR: ratchet to +0.5× ATR. Track `trailing_stop_active` in portfolio state |
| 1c — Max hold extension | `config/signal_routing_config.yml` | `max_holding_days: 15` when entry SNR ≥ 2.0; default 10 bars unchanged |
| 1d — Exit geometry audit | `scripts/run_auto_trader.py` | Add `rr_ratio` to routing diagnostic on every signal accepted/rejected |

**Expected PnL range**: +$239 (1 TP conversion) to +$1,016 (5 conversions).
**Base case (+$645) requires 3 conversions in next 40-trade window.**

**Promotion gate to Phase 2**: R:R gate confirmed active in logs on ≥ 5 signals;
trailing stop ratcheted at least once; WR does not drop below 35% over next 15 trades.

---

### Phase 2 — Cull weak tickers (Days 14–35, governance)

AAPL (8 trips, 12.5% WR, -$376) and GS (5 trips, 0% WR, -$91) fail the evidence
threshold. Demotion is evidence-based and rolling via `build_nav_rebalance_plan.py`.

| Action | File | Specifics |
|--------|------|-----------|
| 2a — Run 2 green weekly cycles | `bash/weekly_sleeve_maintenance.sh` | Confirm AAPL/GS classified WEAK in both cycles |
| 2b — Enable live apply | `scripts/apply_nav_reallocation.py` | Set `live_apply_allowed=True` after 2 consecutive green cycles; AAPL/GS → LAB_ONLY |
| 2c — Concentrate on NVDA/GOOG/MSFT | barbell config | These annualise 4.7–18.5% individually at current sizing |
| 2d — Maintain current position sizing | none | Do NOT reduce to "quarter-Kelly" — avg notional $1,558 = 35.9% of Kelly, already in reasonable band |

**Expected PnL range**: +$234 (partial) to +$611 (full drag prevention + redeployment).

**Note on re-admission**: demotion is rolling. If AAPL/GS rolling WR/PF recover to
portfolio level over a future 20-trade window, `build_nav_rebalance_plan.py` will
reclassify them automatically.

---

### Phase 3 — Accumulate evidence (Days 35–90, passive)

No code changes. Let the system accumulate the evidence needed for next-phase gates.

| Metric | Current | Target | Mechanism |
|--------|---------|--------|-----------|
| THIN_LINKAGE matched | 2 | ≥ 10 | Close 4 open positions + 6 new live round-trips |
| Platt pairs | 0 | ≥ 43 | Every live close adds a pair |
| Classifier labels/ticker | ~290 | ≥ 400 | `run_etl_pipeline.py --tickers AAPL,MSFT,NVDA,GOOG` backfill |
| OOS audit files (AAPL) | thin | ≥ 5 fresh | 5 `--as-of-date` ETL runs for AAPL |

**Promotion gate to Phase 4**: WR ≥ 42% AND PF ≥ 1.8 over 20+ new trades
(post Phase 1/2). Max drawdown < 15% of capital.

---

### Phase 4 — Scale if evidence supports (Day 90+)

Only enter if Phase 3 promotion gate passes.

| Action | Specifics |
|--------|-----------|
| Expand ticker universe to 6–8 names | Post THIN_LINKAGE ≥ 10; adds frequency, not just concentration |
| Increase trades/day target to 1.4 | More tickers + unblocked entries = Lever C activation |
| Ratchet sizing toward half-Kelly | 8% of capital per trade (~$2,000) on core tickers with ≥ 20-trade rolling evidence |
| Covariance heuristic | Max 30% capital in correlated mega-cap tech (NVDA+GOOG+MSFT) at once; no factor model needed yet |
| Platt calibration activates | At 43 pairs: confidence routing becomes calibrated; optional: size linearly with `effective_confidence` |

**Hard cap**: never exceed half-Kelly (8% of capital) until 3+ months of post-Phase-2
evidence at WR ≥ 45%, PF ≥ 1.8. Full Kelly requires a narrow WR CI — not achievable
on this trade count.

---

## Projected Outcomes by Phase

| Phase exit | Scenario | Incremental PnL | Cumulative | Ann ROI |
|------------|----------|-----------------|------------|---------|
| Baseline | — | — | $620 | 10.8% |
| Phase 0 (entries unblocked, 0.95/day) | partial util | +$617 | $1,237 | 21.5% |
| Phase 1 (exit fix, base) | +$645 net | +$645 | $1,882 | 32.7% ✓ |
| Phase 2 (cull AAPL/GS, base) | +$327 net | +$327 | $2,209 | 38.4% |
| Phase 4 (1.4/day freq) | ×1.5 from phase-2 base | ×1.5 | $3,314 | 57.7% |

**NGN hurdle (28%) is first crossed at Phase 1 exit in the base scenario.**

The Phase 0 partial unblock alone does not reliably clear the hurdle (21.5% ann).
Phase 1 (exit geometry) is the decisive code change. Phase 0 is a prerequisite to
generate the trades that Phase 1 improvements can act on.

---

## Anti-Patterns

| Anti-pattern | Why |
|---|---|
| Lower SNR gate below 1.5 without regime compensation | Fix the inflation input, not the gate |
| Apply `adj_snr_gate` without 10+ cycle validation | It's a hypothesis; test before deploying |
| Reduce position sizing to "apply quarter-Kelly" | Already at 35.9% of Kelly — reduction lowers returns |
| Treat $1,584 AAPL/GS PnL gap as closed once demoted | Rolling monitoring required; re-admit when evidence recovers |
| Act on `rmsevalues` bug without verified failing log | May be a prior version issue; confirm with a live log first |
| Build full covariance model before $50k+/trade | Concentration caps (30% mega-cap tech) are sufficient |
| Size up before Phase 1 exits show WR ≥ 38% maintained | Exit fix might temporarily reduce TP count during rollout |
| Treat "+$645 + +$327 = +$972 closes the $991 gap" as proven | These are base-case estimates with 70% additivity discount; the true gap closes in scenarios, not arithmetic |
